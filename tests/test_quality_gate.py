import importlib.util
import random
from pathlib import Path

from PIL import Image

spec = importlib.util.spec_from_file_location(
    "quality_gate", Path(__file__).resolve().parents[1] / "lab" / "upscale" / "quality_gate.py"
)
quality_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality_gate)


def _sample_image(size=(64, 48), seed=7):
    rng = random.Random(seed)
    image = Image.new("RGB", size)
    image.putdata(
        [
            (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for _ in range(size[0] * size[1])
        ]
    )
    return image


def test_make_variants_returns_three_rgb_images_and_is_deterministic():
    source = _sample_image((64, 48), seed=11)
    first = quality_gate.make_variants(source, seed=42)
    second = quality_gate.make_variants(source, seed=42)
    different = quality_gate.make_variants(source, seed=99)

    assert set(first.keys()) == {"clean", "jpeg", "noise"}
    for key in ("clean", "jpeg", "noise"):
        assert first[key].mode == "RGB"
        assert first[key].size == source.size
        assert first[key].tobytes() == second[key].tobytes()

    assert first["clean"].tobytes() == source.tobytes()
    assert first["jpeg"].tobytes() != first["clean"].tobytes()
    assert first["noise"].tobytes() != first["clean"].tobytes()
    assert first["noise"].tobytes() != different["noise"].tobytes()


def test_list_images_skips_remove_subdir_and_non_images(tmp_path):
    _sample_image((16, 16), 1).save(tmp_path / "a.png")
    _sample_image((16, 16), 2).save(tmp_path / "b.jpg")
    _sample_image((16, 16), 3).save(tmp_path / "c.jpeg")
    _sample_image((16, 16), 4).save(tmp_path / "d.webp")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    remove_dir = tmp_path / "remove"
    remove_dir.mkdir()
    _sample_image((16, 16), 5).save(remove_dir / "skipped.png")

    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    _sample_image((16, 16), 6).save(nested_dir / "nested.png")

    kept = quality_gate.list_images(tmp_path)
    assert [p.name for p in kept] == ["a.png", "b.jpg", "c.jpeg", "d.webp"]
    assert quality_gate.list_images(remove_dir) == []
