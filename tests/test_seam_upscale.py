from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from PIL import Image, ImageChops
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "lab/outpainting/seam_upscale.py"
OFF_CENTER_BOX = (0, 180, 540, 1080)


def _load_module():
    spec = importlib.util.spec_from_file_location("seam_upscale", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seam_upscale = _load_module()


def _make_gradient(size, base_rgb):
    img = Image.new("RGB", size, base_rgb)
    pixels = img.load()
    w, h = size
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            pixels[x, y] = (
                (base_rgb[0] + x) % 256,
                (base_rgb[1] + y) % 256,
                (base_rgb[2] + x + y) % 256,
            )
    return img


def _setup_fake_run(lab, run_name="run-sample", seed=42, prefix="powerpaint-v2"):
    run_dir = lab / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    x0, y0, x1, y1 = OFF_CENTER_BOX
    for idx, case_name in enumerate(("eval-01", "eval-02"), start=1):
        case_dir = run_dir / case_name
        case_dir.mkdir()
        generated = _make_gradient((768, 432), (100, 30 * idx, 60))
        lanczos = _make_gradient((1920, 1080), (90 + 10 * idx, 40, 70))
        foreground = _make_gradient((x1 - x0, y1 - y0), (200 - 20 * idx, 210, 30))
        lanczos.paste(foreground, (x0, y0))

        generated.save(case_dir / f"{prefix}-{seed}-generated.png", format="PNG")
        lanczos.save(case_dir / f"{prefix}-{seed}.png", format="PNG")

        records.append(
            {
                "case": case_name,
                "seed": seed,
                "output_box": list(OFF_CENTER_BOX),
                "seam_crop_boxes": {
                    "left": [0, 412, 256, 668],
                    "top": [142, 52, 398, 308],
                    "right": [412, 412, 668, 668],
                    "bottom": [142, 824, 398, 1080],
                },
            }
        )
    (run_dir / "records.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return run_dir


def _fake_adapter(tmp_path):
    return SimpleNamespace(directory=tmp_path / "fake-engine", gpu_id=0)


def test_run_two_cases_success_and_lock(tmp_path):
    _setup_fake_run(tmp_path, "run-01", seed=42, prefix="powerpaint-v2")
    adapter = _fake_adapter(tmp_path)

    lock_state = {"entered": 0, "active": False, "runner_saw_lock": []}
    calls = []
    engine_raw = _make_gradient((3072, 1728), (15, 120, 220))
    expected_engine_bg = engine_raw.convert("RGB").resize(
        (1920, 1080), Image.Resampling.LANCZOS
    )

    @contextmanager
    def fake_lock():
        lock_state["entered"] += 1
        lock_state["active"] = True
        try:
            yield
        finally:
            lock_state["active"] = False

    def fake_runner(argv, **kwargs):
        calls.append((argv, kwargs))
        lock_state["runner_saw_lock"].append(lock_state["active"])
        out_idx = argv.index("-o") + 1
        out_path = Path(argv[out_idx])
        engine_raw.save(out_path, format="PNG")
        return subprocess.CompletedProcess(argv, 0, b"ok", b"")

    records = seam_upscale.run(
        tmp_path,
        "run-01",
        "out-01",
        seed=42,
        prefix="powerpaint-v2",
        runner=fake_runner,
        adapter=adapter,
        lock=fake_lock,
    )

    assert lock_state["entered"] == 1
    assert lock_state["runner_saw_lock"] == [True, True]
    assert len(calls) == 2

    for argv, kwargs in calls:
        assert isinstance(argv, list)
        assert "-n" in argv
        assert argv[argv.index("-n") + 1] == "realesrgan-x4plus-anime"
        assert "-t" in argv
        assert argv[argv.index("-t") + 1] == "128"
        assert kwargs.get("timeout") == 120
        assert kwargs.get("capture_output") is True

    out_dir = tmp_path / "out-01"
    saved_records = json.loads((out_dir / "records.json").read_text(encoding="utf-8"))
    assert len(records) == 2
    assert len(saved_records) == 2

    for case_name in ("eval-01", "eval-02"):
        case_out = out_dir / case_name
        esrgan_path = case_out / "esrgan-42.png"
        lanczos_path = tmp_path / "run-01" / case_name / "powerpaint-v2-42.png"
        assert esrgan_path.is_file()
        assert not (tmp_path / "run-01" / case_name / "source.png").exists()
        assert not list(case_out.glob(".tmp-*"))

        with (
            Image.open(lanczos_path) as lanczos_img,
            Image.open(esrgan_path) as final_img,
        ):
            assert final_img.size == (1920, 1080)
            fg_diff = ImageChops.difference(
                final_img.crop(OFF_CENTER_BOX), lanczos_img.crop(OFF_CENTER_BOX)
            )
            assert fg_diff.getbbox() is None

            outside_right = (540, 0, 1920, 1080)
            outside_top = (0, 0, 540, 180)
            for outside_box in (outside_right, outside_top):
                bg_engine_diff = ImageChops.difference(
                    final_img.crop(outside_box),
                    expected_engine_bg.crop(outside_box),
                )
                assert bg_engine_diff.getbbox() is None
                bg_lanczos_diff = ImageChops.difference(
                    final_img.crop(outside_box),
                    lanczos_img.crop(outside_box),
                )
                assert bg_lanczos_diff.getbbox() is not None

        for side in ("left", "top", "right", "bottom"):
            cmp_path = case_out / f"seam-compare-{side}.png"
            assert cmp_path.is_file()
            with Image.open(cmp_path) as cmp_img:
                assert cmp_img.size == (512, 256)
                left_half = cmp_img.crop((0, 0, 256, 256))
                right_half = cmp_img.crop((256, 0, 512, 256))
                if side == "right":
                    # Box [412, 412, 668, 668]: x in [412, 540) -> local x in [0, 128) is inside output_box
                    inside_part = (0, 0, 128, 256)
                    outside_part = (128, 0, 256, 256)
                    assert (
                        ImageChops.difference(
                            left_half.crop(inside_part), right_half.crop(inside_part)
                        ).getbbox()
                        is None
                    )
                    assert (
                        ImageChops.difference(
                            left_half.crop(outside_part), right_half.crop(outside_part)
                        ).getbbox()
                        is not None
                    )
                elif side == "top":
                    # Box [142, 52, 398, 308]: y in [180, 308) -> local y in [128, 256) is inside output_box
                    inside_part = (0, 128, 256, 256)
                    outside_part = (0, 0, 256, 128)
                    assert (
                        ImageChops.difference(
                            left_half.crop(inside_part), right_half.crop(inside_part)
                        ).getbbox()
                        is None
                    )
                    assert (
                        ImageChops.difference(
                            left_half.crop(outside_part), right_half.crop(outside_part)
                        ).getbbox()
                        is not None
                    )
                else:
                    assert (
                        ImageChops.difference(left_half, right_half).getbbox() is None
                    )


def test_reject_existing_out_and_unsafe_names(tmp_path):
    _setup_fake_run(tmp_path, "run-01")
    existing_out = tmp_path / "out-existing"
    existing_out.mkdir()

    @contextmanager
    def dummy_lock():
        yield

    def dummy_runner(argv, **kwargs):
        raise AssertionError("Runner không được phép chạy khi tên thư mục không hợp lệ.")

    with pytest.raises(FileExistsError):
        seam_upscale.run(
            tmp_path,
            "run-01",
            "out-existing",
            runner=dummy_runner,
            adapter=_fake_adapter(tmp_path),
            lock=dummy_lock,
        )

    for bad_name in ("..", "../escape", "sub/dir", "sub\\dir", ""):
        with pytest.raises(ValueError):
            seam_upscale.run(
                tmp_path,
                bad_name,
                "out-ok",
                runner=dummy_runner,
                adapter=_fake_adapter(tmp_path),
                lock=dummy_lock,
            )
        with pytest.raises(ValueError):
            seam_upscale.run(
                tmp_path,
                "run-01",
                bad_name,
                runner=dummy_runner,
                adapter=_fake_adapter(tmp_path),
                lock=dummy_lock,
            )


def test_runner_nonzero_returncode_raises(tmp_path):
    _setup_fake_run(tmp_path, "run-rc")

    @contextmanager
    def dummy_lock():
        yield

    def failing_runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 2, b"", b"vulkan error")

    with pytest.raises(RuntimeError):
        seam_upscale.run(
            tmp_path,
            "run-rc",
            "out-rc",
            runner=failing_runner,
            adapter=_fake_adapter(tmp_path),
            lock=dummy_lock,
        )


def test_runner_wrong_output_dimensions_raises(tmp_path):
    _setup_fake_run(tmp_path, "run-bad-size")

    @contextmanager
    def dummy_lock():
        yield

    def wrong_size_runner(argv, **kwargs):
        out_path = Path(argv[argv.index("-o") + 1])
        Image.new("RGB", (1024, 1024), (0, 0, 0)).save(out_path, format="PNG")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    with pytest.raises(RuntimeError):
        seam_upscale.run(
            tmp_path,
            "run-bad-size",
            "out-bad-size",
            runner=wrong_size_runner,
            adapter=_fake_adapter(tmp_path),
            lock=dummy_lock,
        )
    assert not list((tmp_path / "out-bad-size").rglob(".tmp-*"))


def test_missing_required_file_raises(tmp_path):
    run_dir = _setup_fake_run(tmp_path, "run-missing")
    (run_dir / "eval-01" / "powerpaint-v2-42-generated.png").unlink()

    @contextmanager
    def dummy_lock():
        yield

    with pytest.raises(FileNotFoundError):
        seam_upscale.run(
            tmp_path,
            "run-missing",
            "out-missing",
            runner=lambda *a, **k: None,
            adapter=_fake_adapter(tmp_path),
            lock=dummy_lock,
        )


def test_output_box_out_of_bounds_or_missing_raises(tmp_path):
    lanczos = Image.new("RGB", (1920, 1080), (10, 20, 30))
    upscaled = Image.new("RGB", (3072, 1728), (40, 50, 60))

    invalid_boxes = [
        [-1, 180, 540, 1080],
        [0, -1, 540, 1080],
        [0, 180, 1921, 1080],
        [0, 180, 540, 1081],
        [540, 180, 540, 1080],
        [600, 180, 540, 1080],
        [0, 600, 540, 180],
    ]
    for bad_box in invalid_boxes:
        with pytest.raises((ValueError, RuntimeError)):
            seam_upscale.compose(lanczos, upscaled, bad_box)

    @contextmanager
    def dummy_lock():
        yield

    run_oob = _setup_fake_run(tmp_path, "run-oob")
    records = json.loads((run_oob / "records.json").read_text(encoding="utf-8"))
    records[0]["output_box"] = [0, 180, 1925, 1080]
    (run_oob / "records.json").write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises((ValueError, RuntimeError)):
        seam_upscale.run(
            tmp_path,
            "run-oob",
            "out-oob",
            runner=lambda *a, **k: None,
            adapter=_fake_adapter(tmp_path),
            lock=dummy_lock,
        )

    run_no_box = _setup_fake_run(tmp_path, "run-no-box")
    records = json.loads((run_no_box / "records.json").read_text(encoding="utf-8"))
    del records[0]["output_box"]
    (run_no_box / "records.json").write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises((ValueError, RuntimeError)):
        seam_upscale.run(
            tmp_path,
            "run-no-box",
            "out-no-box",
            runner=lambda *a, **k: None,
            adapter=_fake_adapter(tmp_path),
            lock=dummy_lock,
        )
