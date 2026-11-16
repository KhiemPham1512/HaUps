from pathlib import Path
import subprocess

from PIL import Image
import pytest

from app.engine import EngineAdapter, EngineError


def test_alpha_cache_keeps_engine_alpha_and_original_native(tmp_path):
    Image.new("RGBA", (8, 6), (255, 255, 255, 128)).save(tmp_path / "source.png")
    Image.new("RGBA", (32, 24), (0, 0, 0, 80)).save(tmp_path / "native.png")
    native = (tmp_path / "native.png").read_bytes()
    source = (tmp_path / "source.png").read_bytes()
    engine = EngineAdapter(gpu_id=0)
    engine.process(tmp_path, 4, "anime")
    with Image.open(tmp_path / "result-4.png") as image:
        assert image.getpixel((8, 8)) == (0, 0, 0, 80)
    assert (tmp_path / "result-4.png").samefile(tmp_path / "native.png")
    engine.process(tmp_path, 8, "anime")
    for scale in (2, 4, 8):
        with Image.open(tmp_path / f"result-{scale}.png") as image:
            assert image.size == (8 * scale, 6 * scale)
            assert image.getchannel("A").getextrema() == (80, 80)
    assert (tmp_path / "native.png").read_bytes() == native
    assert (tmp_path / "source.png").read_bytes() == source


def test_export_failure_keeps_valid_native_cache(tmp_path, monkeypatch):
    fault_target = "result-8.tmp"
    source_path = tmp_path / "source.png"
    native_path = tmp_path / "native.png"
    Image.new("RGB", (8, 6), (200, 100, 60)).save(source_path)
    Image.new("RGB", (32, 24), (20, 20, 20)).save(native_path)

    EngineAdapter(gpu_id=0).process(tmp_path, 2, "anime")

    source_bytes = source_path.read_bytes()
    native_bytes = native_path.read_bytes()
    old_outputs = {
        p: p.read_bytes()
        for p in tmp_path.iterdir()
        if p.is_file() and p not in (source_path, native_path)
    }

    original_save = Image.Image.save

    def faulty_save(self, fp, *args, **kwargs):
        try:
            name = Path(fp).name
        except (TypeError, ValueError):
            name = None
        if name == fault_target:
            Path(fp).write_bytes(b"corrupted partial data")
            raise OSError(f"Simulated write failure on {fault_target}")
        return original_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", faulty_save)

    def fail_subprocess(*args, **kwargs):
        pytest.fail("subprocess.run should not be called because native already exists")

    monkeypatch.setattr(subprocess, "run", fail_subprocess)

    with pytest.raises(OSError):
        EngineAdapter(gpu_id=0).process(tmp_path, 8, "anime")

    assert source_path.read_bytes() == source_bytes
    assert native_path.read_bytes() == native_bytes
    for path, data in old_outputs.items():
        assert path.read_bytes() == data
    assert not list(tmp_path.glob("*.tmp"))

    monkeypatch.setattr(Image.Image, "save", original_save)
    EngineAdapter(gpu_id=0).process(tmp_path, 8, "anime")

    with Image.open(tmp_path / "result-8.png") as image:
        assert image.size == (64, 48)


@pytest.mark.parametrize("case", ["corrupt", "size", "alpha"])
def test_invalid_native_is_removed_without_touching_source(tmp_path, monkeypatch, case):
    source_path = tmp_path / "source.png"
    native_path = tmp_path / "native.png"
    Image.new("RGBA", (8, 6), (255, 255, 255, 128)).save(source_path)

    if case == "corrupt":
        native_path.write_bytes(b"corrupt bytes")
    elif case == "size":
        Image.new("RGBA", (31, 24), (0, 0, 0, 80)).save(native_path)
    elif case == "alpha":
        Image.new("RGB", (32, 24), (20, 20, 20)).save(native_path)

    source_bytes = source_path.read_bytes()

    def fail_subprocess(*args, **kwargs):
        pytest.fail("subprocess.run should not be called because native already exists")

    monkeypatch.setattr(subprocess, "run", fail_subprocess)

    with pytest.raises((EngineError, OSError)):
        EngineAdapter(gpu_id=0).process(tmp_path, 2, "anime")

    assert not native_path.exists()
    assert source_path.read_bytes() == source_bytes
    assert not (tmp_path / "result.png").exists()

