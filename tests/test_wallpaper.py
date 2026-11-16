from io import BytesIO
import subprocess
from pathlib import Path
import sys
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageChops, ImageOps

from app.main import create_app
from app import wallpaper


@pytest.fixture
def installed_model(tmp_path, monkeypatch):
    lab = tmp_path / "lab"
    weights = lab / "powerpaint-v2" / "weights"
    (lab / "powerpaint-v2" / "vendor").mkdir(parents=True, exist_ok=True)
    anime_seg = lab / "anime-seg" / "isnetis.onnx"
    anime_seg.parent.mkdir(parents=True, exist_ok=True)
    anime_seg.touch()
    for name in wallpaper.REQUIRED_FILES:
        path = weights / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for name in wallpaper.SDXL_FILES:
        path = lab / "sdxl" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(wallpaper, "LAB", lab)
    monkeypatch.setattr(wallpaper, "PYTHON", Path(sys.executable))
    return weights


def test_wallpaper_is_a_separate_fifo_export_and_survives_restart(
    tmp_path, monkeypatch, installed_model
):
    calls = []

    def run(argv, **kwargs):
        calls.append(list(argv))
        if "app.outpaint_runner" in argv:
            Image.new("RGB", (3840, 2160), "#336688").save(argv[-1])
        else:
            with Image.open(argv[argv.index("-i") + 1]) as image:
                image.resize((image.width * 4, image.height * 4)).save(
                    argv[argv.index("-o") + 1]
                )
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    def finished(client):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = client.get("/api/jobs").json()["jobs"][0]
            if job["status"] in ("done", "failed"):
                assert job["status"] == "done", job
                return job
            time.sleep(0.01)
        raise AssertionError("Worker did not finish")

    monkeypatch.setattr(subprocess, "run", run)
    content = BytesIO()
    Image.new("RGB", (16, 24), "#336688").save(content, format="PNG")
    settings = {"data_dir": tmp_path, "gpu_id": 0}
    with TestClient(create_app(settings)) as client:
        job = client.post(
            "/api/batches",
            files={"files": ("portrait.png", content.getvalue(), "image/png")},
        ).json()["jobs"][0]
        job = finished(client)
        original = client.get(job["source_url"]).content
        upscale = client.get(job["download_url"]).content
        response = client.post(
            f"/api/jobs/{job['id']}/wallpaper",
            data={"prompt": "blue sky, empty background"},
        )
        assert response.status_code == 202, response.text
        job = finished(client)
        assert job["wallpaper_available"] is True
        result = client.get(f"/api/jobs/{job['id']}/wallpaper")
        assert result.status_code == 200
        assert "portrait_3840x2160.png" in result.headers["content-disposition"]
        with Image.open(BytesIO(result.content)) as image:
            assert image.size == (3840, 2160)
        assert client.get(job["source_url"]).content == original
        assert client.get(job["download_url"]).content == upscale
        assert len(calls) == 2
        assert calls[1][calls[1].index("-m") + 1] == "app.outpaint_runner"
        assert (
            client.post(
                f"/api/jobs/{job['id']}/wallpaper",
                data={"prompt": "blue sky, empty background"},
            ).status_code
            == 202
        )
        assert len(calls) == 2
        assert (
            client.post(
                f"/api/jobs/{job['id']}/wallpaper", data={"prompt": "a" * 501}
            ).status_code
            == 422
        )
        assert client.post("/api/jobs/unknown/wallpaper").status_code == 404
        assert client.get("/api/jobs/unknown/wallpaper").status_code == 404
        second = client.post(
            "/api/batches",
            files={"files": ("landscape.png", content.getvalue(), "image/png")},
        ).json()["jobs"][0]
        second = finished(client)
        second_original = client.get(second["source_url"]).content
        second_upscale = client.get(second["download_url"]).content
        response = client.post(f"/api/jobs/{second['id']}/wallpaper")
        assert response.status_code == 202, response.text
        second = finished(client)
        assert second["wallpaper_available"] is True
        assert client.get(second["source_url"]).content == second_original
        assert client.get(second["download_url"]).content == second_upscale
        assert "--prompt=" in calls[-1]
    with TestClient(create_app(settings)) as client:
        assert client.get(f"/api/jobs/{job['id']}/wallpaper").content == result.content


@pytest.mark.parametrize("size", [(108, 192), (192, 108), (192, 54), (100, 100)])
def test_generated_area_cannot_change_the_fitted_foreground(size):
    source = Image.new("RGB", size, "#193d71")
    source.putpixel((size[0] // 2, size[1] // 2), (255, 50, 80))
    canvas, mask = wallpaper.prepare(source)
    assert canvas.size == mask.size == (768, 432)
    output = wallpaper.restore(source, Image.new("RGB", canvas.size, "#aabbcc"))
    main = ImageOps.contain(source, (3840, 2160), Image.Resampling.LANCZOS)
    x, y = (3840 - main.width) // 2, (2160 - main.height) // 2
    assert output.size == (3840, 2160)
    assert (
        ImageChops.difference(
            output.crop((x, y, x + main.width, y + main.height)), main
        ).getbbox()
        is None
    )
    if x or y:
        assert output.getpixel((0, 0)) == (170, 187, 204)
        assert mask.getpixel((0, 0)) == 255
    else:
        assert mask.getextrema() == (0, 0)


@pytest.mark.parametrize("failure", ["timeout", "failed", "wrong_size"])
def test_failed_generation_keeps_previous_exports(
    tmp_path, monkeypatch, installed_model, failure
):
    from app.engine import EngineError

    Image.new("RGB", (16, 24), "red").save(tmp_path / "native.png")
    Image.new("RGB", (3840, 2160), "blue").save(tmp_path / "wallpaper.png")
    before = {
        name: (tmp_path / name).read_bytes() for name in ("native.png", "wallpaper.png")
    }

    def run(argv, **kwargs):
        assert kwargs["timeout"] == 600
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert argv[argv.index("-m") + 1] == "app.outpaint_runner"
        esrgan_dir = tmp_path / "wallpaper.tmp-esrgan"
        esrgan_dir.mkdir(exist_ok=True)
        (esrgan_dir / "source.png").write_bytes(b"partial")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 600)
        Image.new("RGB", (32, 24), "green").save(argv[-1])
        return subprocess.CompletedProcess(
            argv, 1 if failure == "failed" else 0, b"", b""
        )

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(EngineError) as exc_info:
        wallpaper.create_wallpaper(tmp_path, "--not-a-shell-command", "anime")
    if failure == "wrong_size":
        assert "3840×2160" in str(exc_info.value)
    assert all(
        (tmp_path / name).read_bytes() == content for name, content in before.items()
    )
    assert not (tmp_path / "wallpaper.tmp.png").exists()
    assert not (tmp_path / "wallpaper.tmp-esrgan").exists()


def test_incomplete_model_and_transparent_image_are_rejected_before_gpu(
    tmp_path, monkeypatch, installed_model
):
    from app.engine import EngineError

    Image.new("RGBA", (8, 16), (255, 0, 0, 100)).save(tmp_path / "native.png")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("Must not start GPU")
    )
    with pytest.raises(EngineError, match="trong suốt"):
        wallpaper.create_wallpaper(tmp_path, "", "anime")
    (installed_model / wallpaper.REQUIRED_FILES[0]).unlink()
    assert wallpaper.is_available() is False
    with pytest.raises(EngineError, match="Thiếu"):
        wallpaper.create_wallpaper(tmp_path, "", "anime")


@pytest.mark.parametrize("missing_file", wallpaper.REQUIRED_FILES)
def test_is_available_false_when_powerpaint_file_is_missing(
    installed_model, missing_file
):
    assert wallpaper.is_available() is True
    (installed_model / missing_file).unlink()
    assert wallpaper.is_available() is False


def test_is_available_false_when_sdxl_repair_file_is_missing(installed_model):
    assert wallpaper.is_available()
    (wallpaper.LAB / "sdxl" / wallpaper.SDXL_FILES[0]).unlink()
    assert not wallpaper.is_available()


def test_is_available_false_when_anime_seg_onnx_is_missing(installed_model):
    assert wallpaper.is_available() is True
    (wallpaper.LAB / "anime-seg" / "isnetis.onnx").unlink()
    assert wallpaper.is_available() is False


def test_generation_exit_code_three_is_a_processing_error_and_keeps_files(
    tmp_path, monkeypatch, installed_model
):
    from app.engine import EngineError

    native_path = tmp_path / "native.png"
    wallpaper_path = tmp_path / "wallpaper.png"
    temp_path = tmp_path / "wallpaper.tmp.png"

    Image.new("RGB", (16, 24), "red").save(native_path)
    Image.new("RGB", (3840, 2160), "blue").save(wallpaper_path)

    native_before = native_path.read_bytes()
    wallpaper_before = wallpaper_path.read_bytes()

    def fake_run(argv, **kwargs):
        assert kwargs["timeout"] == 600
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert argv[argv.index("-m") + 1] == "app.outpaint_runner"
        return subprocess.CompletedProcess(
            argv,
            returncode=3,
            stdout=b"",
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(EngineError, match="lỗi xử lý") as exc_info:
        wallpaper.create_wallpaper(tmp_path, "", "anime")

    message = str(exc_info.value)
    assert "kiểm tra" not in message
    assert "mô tả" not in message
    assert native_path.read_bytes() == native_before
    assert wallpaper_path.read_bytes() == wallpaper_before
    assert not temp_path.exists()


def test_failed_generation_writes_stderr_tail_to_job_log(
    tmp_path, monkeypatch, installed_model
):
    from app.engine import EngineError

    Image.new("RGB", (16, 24), "red").save(tmp_path / "native.png")
    stderr = b"x" * 20_000 + "\nRuntimeError: CUDA out of memory ✗\n".encode()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, b"", stderr),
    )
    with pytest.raises(EngineError):
        wallpaper.create_wallpaper(tmp_path, "", "anime")
    log = (tmp_path / "wallpaper-error.log").read_text(encoding="utf-8")
    assert log.startswith("returncode=1\n")
    assert log.rstrip().endswith("RuntimeError: CUDA out of memory ✗")
    assert len(log) < 9000

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: (
            Image.new("RGB", (3840, 2160)).save(argv[-1]),
            subprocess.CompletedProcess(argv, 0, b"", b""),
        )[1],
    )
    wallpaper.create_wallpaper(tmp_path, "", "anime")
    assert not (tmp_path / "wallpaper-error.log").exists()


def test_intermediate_pixel_guard_allows_256mp_and_rejects_larger(
    tmp_path, monkeypatch, installed_model
):
    from app.engine import EngineError

    source_path = tmp_path / "native.png"
    Image.new("RGB", (8, 8), "red").save(source_path)
    real_open = Image.open

    class FakeSource:
        def __init__(self, width, height):
            self.width = width
            self.height = height

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getbands(self):
            return ("R", "G", "B")

    monkeypatch.setattr(
        wallpaper.Image,
        "open",
        lambda path, *a, **k: (
            FakeSource(16_000, 16_001)
            if Path(path) == source_path
            else real_open(path, *a, **k)
        ),
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("Must not start GPU")
    )
    with pytest.raises(EngineError, match="Ảnh trung gian quá lớn"):
        wallpaper.create_wallpaper(tmp_path, "", "anime")

    monkeypatch.setattr(
        wallpaper.Image,
        "open",
        lambda path, *a, **k: (
            FakeSource(16_000, 16_000)
            if Path(path) == source_path
            else real_open(path, *a, **k)
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: (
            Image.new("RGB", (3840, 2160)).save(argv[-1]),
            subprocess.CompletedProcess(argv, 0, b"", b""),
        )[1],
    )
    wallpaper.create_wallpaper(tmp_path, "", "anime")
    assert (tmp_path / "wallpaper.png").is_file()


def test_content_classifier_files_are_not_required(installed_model):
    for filename in (
        "safety_checker/config.json",
        "safety_checker/model.fp16.safetensors",
        "feature_extractor/preprocessor_config.json",
    ):
        (installed_model / filename).unlink(missing_ok=True)
    assert wallpaper.is_available() is True


def test_restore_feathered_inner_exact_ramped_edges_and_border_sides():
    source = Image.new("RGB", (108, 192), (20, 40, 80))
    source.putpixel((54, 96), (250, 60, 90))
    generated_native = Image.new("RGB", (3072, 1728), (200, 150, 100))
    width_px = 32

    out = wallpaper.restore_feathered(source, generated_native, width_px)
    main = ImageOps.contain(source, (3840, 2160), Image.Resampling.LANCZOS)
    x = (3840 - main.width) // 2
    y = (2160 - main.height) // 2
    assert y == 0 and main.height == 2160

    inner_out = out.crop((x + width_px, 0, x + main.width - width_px, 2160))
    inner_main = main.crop((width_px, 0, main.width - width_px, 2160))
    assert ImageChops.difference(inner_out, inner_main).getbbox() is None

    assert out.getpixel((x, 1080)) == (200, 150, 100)
    assert out.getpixel((x + main.width - 1, 1080)) == (200, 150, 100)
    mid_r, _, _ = out.getpixel((x + width_px // 2, 1080))
    assert 20 < mid_r < 200

    for px in (x + width_px, x + main.width // 2, x + main.width - width_px - 1):
        assert out.getpixel((px, 0)) == main.getpixel((px - x, 0))
        assert out.getpixel((px, 2159)) == main.getpixel((px - x, 2159))

    wide = Image.new("RGB", (192, 54), (30, 90, 40))
    out_wide = wallpaper.restore_feathered(wide, generated_native, width_px)
    main_wide = ImageOps.contain(wide, (3840, 2160), Image.Resampling.LANCZOS)
    wy = (2160 - main_wide.height) // 2
    assert out_wide.getpixel((1920, wy)) == (200, 150, 100)
    assert out_wide.getpixel((1920, wy + main_wide.height - 1)) == (200, 150, 100)
    assert out_wide.getpixel((0, 1080)) == main_wide.getpixel((0, 1080 - wy))
    assert out_wide.getpixel((3839, 1080)) == main_wide.getpixel((3839, 1080 - wy))


def test_restore_feathered_width_zero_matches_restore():
    source = Image.new("RGB", (108, 192), (25, 65, 115))
    source.putpixel((54, 96), (240, 80, 30))
    generated_native = Image.new("RGB", (3072, 1728), (180, 140, 90))

    hard = wallpaper.restore(source, generated_native)
    feathered_zero = wallpaper.restore_feathered(source, generated_native, 0)
    feathered_neg = wallpaper.restore_feathered(source, generated_native, -8)

    assert ImageChops.difference(feathered_zero, hard).getbbox() is None
    assert ImageChops.difference(feathered_neg, hard).getbbox() is None


def test_create_wallpaper_reads_sanitizes_and_cleans_scene_sidecar(
    tmp_path, monkeypatch, installed_model
):
    from app.engine import EngineError

    Image.new("RGB", (16, 24), "red").save(tmp_path / "native.png")
    sidecar = tmp_path / "wallpaper.tmp-scene.txt"

    def run_with_sidecar(text):
        def _run(argv, **kwargs):
            assert not sidecar.exists()
            Image.new("RGB", (3840, 2160), "#336688").save(argv[-1])
            if text is not None:
                sidecar.write_text(text, encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, b"", b"")

        return _run

    monkeypatch.setattr(subprocess, "run", run_with_sidecar("  outdoors, sky  \n"))
    assert wallpaper.create_wallpaper(tmp_path, "", "anime") == "outdoors, sky"
    assert not sidecar.exists()

    sidecar.write_text("stale scene", encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", run_with_sidecar(None))
    assert wallpaper.create_wallpaper(tmp_path, "user prompt", "anime") is None
    assert not sidecar.exists()

    noisy = " \x00outdoors,\x07\x1b sky, " + ("a" * 350) + "\n"
    monkeypatch.setattr(subprocess, "run", run_with_sidecar(noisy))
    sanitized = wallpaper.create_wallpaper(tmp_path, "", "anime")
    assert sanitized == ("outdoors, sky, " + ("a" * 350))[:300]
    assert len(sanitized) == 300
    assert sanitized.isprintable()
    assert not sidecar.exists()

    def failing_run(argv, **kwargs):
        sidecar.write_text("outdoors, sky", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 1, b"", b"boom")

    monkeypatch.setattr(subprocess, "run", failing_run)
    with pytest.raises(EngineError):
        wallpaper.create_wallpaper(tmp_path, "", "anime")
    assert not sidecar.exists()
