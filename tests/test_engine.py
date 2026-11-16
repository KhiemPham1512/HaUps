import subprocess

import pytest

from app.engine import EngineAdapter


def test_gpu_is_discovered_from_native_nvidia_vulkan_device(monkeypatch):
    monkeypatch.delenv("HAUPS_GPU_ID", raising=False)
    summary = b"GPU0:\n deviceName = Intel Graphics\n driverID = DRIVER_ID_INTEL_PROPRIETARY_WINDOWS\nGPU2:\n deviceName = NVIDIA GeForce RTX 4050 Laptop GPU\n driverID = DRIVER_ID_NVIDIA_PROPRIETARY\n"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, summary, b""),
    )
    assert EngineAdapter().gpu_id == 2


def test_out_of_memory_retries_once_with_smaller_tile_and_same_model(
    tmp_path, monkeypatch
):
    from PIL import Image

    folder = tmp_path
    Image.new("RGB", (8, 8), "red").save(folder / "source.png")
    calls = []

    def run(argv, **kwargs):
        calls.append(list(argv))
        if len(calls) == 1:
            return subprocess.CompletedProcess(
                argv, 1, b"", b"vkAllocateMemory failed: out of memory"
            )
        Image.new("RGB", (32, 32), "red").save(folder / "native.png")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", run)
    EngineAdapter(gpu_id=0).process(folder, 2, "photo")
    assert len(calls) == 2
    assert [argv[argv.index("-t") + 1] for argv in calls] == ["64", "32"]
    assert all(argv[argv.index("-n") + 1] == "realesrgan-x4plus" for argv in calls)
    assert all("-x" not in argv for argv in calls)
    with Image.open(folder / "result.png") as result:
        assert result.size == (16, 16)


@pytest.mark.parametrize("oom_failures", [0, 1, 2, 3])
def test_anime_uses_tile128_with_bounded_oom_fallback(
    tmp_path, monkeypatch, oom_failures
):
    from PIL import Image
    from app.engine import EngineError

    source = tmp_path / "source.png"
    Image.new("RGBA", (8, 6), (100, 80, 60, 120)).save(source)
    original = source.read_bytes()
    calls = []
    clock = iter((0, 1, 100, 599))
    monkeypatch.setattr("app.engine.monotonic", lambda: next(clock))

    def run(argv, **kwargs):
        calls.append((list(argv), kwargs["timeout"]))
        input_path = argv[argv.index("-i") + 1]
        with Image.open(input_path) as image:
            assert image.mode == "RGB"
            if len(calls) <= oom_failures:
                return subprocess.CompletedProcess(
                    argv, 1, b"", b"vkAllocateMemory failed"
                )
            image.resize((32, 24)).save(tmp_path / "native.png")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", run)
    engine = EngineAdapter(gpu_id=0)
    if oom_failures == 3:
        with pytest.raises(EngineError, match="chưa đủ bộ nhớ"):
            engine.process(tmp_path, 2, "anime")
        assert not (tmp_path / "native.png").exists()
        assert not (tmp_path / "result.png").exists()
    else:
        engine.process(tmp_path, 2, "anime")
        with Image.open(tmp_path / "result.png") as result:
            assert result.size == (16, 12)
            assert result.mode == "RGBA"
    assert not (tmp_path / "engine-input.png").exists()
    expected_calls = min(oom_failures + 1, 3)
    assert len(calls) == expected_calls
    assert [argv[argv.index("-t") + 1] for argv, _ in calls] == ["128", "64", "32"][
        :expected_calls
    ]
    assert [timeout for _, timeout in calls] == [599, 500, 1][:expected_calls]
    assert all(
        argv[argv.index("-n") + 1] == "haups-anime-x4" and "-x" not in argv
        for argv, _ in calls
    )
    assert source.read_bytes() == original


def test_anime_does_not_retry_non_memory_engine_failure(tmp_path, monkeypatch):
    from PIL import Image
    from app.engine import EngineError

    source = tmp_path / "source.png"
    Image.new("RGB", (8, 6), "red").save(source)
    original = source.read_bytes()
    calls = []

    def run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 1, b"", b"model parameter error")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(EngineError, match="xử lý thất bại"):
        EngineAdapter(gpu_id=0).process(tmp_path, 2, "anime")
    assert len(calls) == 1
    assert calls[0][calls[0].index("-t") + 1] == "128"
    assert source.read_bytes() == original


def test_timeout_reports_ten_minutes_and_keeps_source(tmp_path, monkeypatch):
    import pytest
    from PIL import Image
    from app.engine import EngineError

    Image.new("RGB", (8, 8), "red").save(tmp_path / "source.png")

    def timeout(argv, **kwargs):
        assert 590 < kwargs["timeout"] <= 600
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(EngineError, match="10 phút"):
        EngineAdapter(gpu_id=0).process(tmp_path, 8, "anime")
    assert (tmp_path / "source.png").is_file()
    assert not (tmp_path / "result.png").exists()


def test_variants_reuse_native_without_another_gpu_call(tmp_path, monkeypatch):
    from PIL import Image

    Image.new("RGBA", (16, 12), (100, 80, 60, 120)).save(tmp_path / "source.png")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        with Image.open(argv[argv.index("-i") + 1]) as engine_in:
            assert engine_in.mode == "RGB"
            engine_in.resize((64, 48)).save(tmp_path / "native.png")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", run)
    engine = EngineAdapter(gpu_id=0)
    engine.process(tmp_path, 2, "anime")
    assert not (tmp_path / "result-8.png").exists()
    engine.process(tmp_path, 8, "anime")
    assert len(calls) == 1
    for scale in (2, 4, 8):
        with Image.open(tmp_path / f"result-{scale}.png") as result:
            assert result.size == (16 * scale, 12 * scale)
            assert result.mode == "RGBA"


def test_rgba_source_passes_rgb_input_and_merges_lanczos_alpha(tmp_path, monkeypatch):
    from pathlib import Path
    from PIL import Image

    source_path = tmp_path / "source.png"
    source_img = Image.new("RGBA", (4, 3), (10, 20, 30, 40))
    source_img.putpixel((0, 0), (200, 150, 100, 250))
    source_img.putpixel((3, 2), (50, 60, 70, 0))
    source_img.save(source_path)
    original_bytes = source_path.read_bytes()
    expected_alpha = source_img.getchannel("A").resize(
        (16, 12), Image.Resampling.LANCZOS
    )
    seen_inputs = []

    def run(argv, **kwargs):
        input_file = Path(argv[argv.index("-i") + 1])
        seen_inputs.append(input_file)
        assert input_file == tmp_path / "engine-input.png"
        with Image.open(input_file) as engine_in:
            assert engine_in.mode == "RGB"
            assert "A" not in engine_in.getbands()
            Image.new("RGB", (engine_in.width * 4, engine_in.height * 4), (90, 80, 70)).save(
                argv[argv.index("-o") + 1]
            )
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", run)
    EngineAdapter(gpu_id=0).process(tmp_path, 4, "anime")

    assert len(seen_inputs) == 1
    assert not (tmp_path / "engine-input.png").exists()
    assert source_path.read_bytes() == original_bytes
    with Image.open(tmp_path / "native.png") as native_img:
        assert native_img.mode == "RGBA"
        assert native_img.getchannel("A").tobytes() == expected_alpha.tobytes()
    with Image.open(tmp_path / "result.png") as result_img:
        assert result_img.mode == "RGBA"
        assert result_img.size == (16, 12)
        assert result_img.getchannel("A").tobytes() == expected_alpha.tobytes()


def test_cached_outputs_share_storage_without_overwriting_native(tmp_path):
    from PIL import Image

    Image.new("RGB", (8, 6), "red").save(tmp_path / "source.png")
    Image.new("RGB", (32, 24), "blue").save(tmp_path / "native.png")
    original = (tmp_path / "native.png").read_bytes()
    engine = EngineAdapter(gpu_id=0)
    engine.process(tmp_path, 4, "photo")
    assert (tmp_path / "native.png").samefile(tmp_path / "result-4.png")
    assert (tmp_path / "result.png").samefile(tmp_path / "result-4.png")
    (tmp_path / "result.tmp").hardlink_to(tmp_path / "native.png")
    engine.process(tmp_path, 8, "photo")
    assert (tmp_path / "result.png").samefile(tmp_path / "result-8.png")
    assert (tmp_path / "native.png").read_bytes() == original
    for factor in (2, 4, 8):
        with Image.open(tmp_path / f"result-{factor}.png") as result:
            assert result.size == (8 * factor, 6 * factor)


def test_output_copy_fallback_keeps_cache_and_older_variants(tmp_path, monkeypatch):
    import os
    from PIL import Image

    Image.new("RGB", (8, 6), "red").save(tmp_path / "source.png")
    Image.new("RGB", (32, 24), "blue").save(tmp_path / "native.png")
    original = (tmp_path / "native.png").read_bytes()

    def unavailable(*args):
        raise OSError("Hard links unavailable on this volume")

    monkeypatch.setattr(os, "link", unavailable)
    engine = EngineAdapter(gpu_id=0)
    engine.process(tmp_path, 4, "photo")
    engine.process(tmp_path, 8, "photo")
    assert (tmp_path / "native.png").read_bytes() == original
    assert (tmp_path / "result-4.png").read_bytes() == original
    assert (tmp_path / "result.png").read_bytes() == (
        tmp_path / "result-8.png"
    ).read_bytes()
    assert not list(tmp_path.glob("*.tmp"))


def test_pillow_decompression_bomb_guard_supports_8x_outputs_from_16mp():
    from PIL import Image

    assert Image.MAX_IMAGE_PIXELS == 16_000_000 * 64 + 1
