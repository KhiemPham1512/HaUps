import importlib.util
from pathlib import Path

import pytest
from PIL import Image, ImageChops

spec = importlib.util.spec_from_file_location(
    "placement_poc",
    Path(__file__).resolve().parents[1]
    / "lab/outpainting/placement_poc.py",
)
placement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(placement)


def test_portrait_leaves_space_above_and_below_and_preserves_foreground():
    source = Image.new("RGB", (81, 109), "#193d71")
    source.putpixel((40, 54), (255, 50, 80))
    before = source.tobytes()
    canvas, mask, box = placement.prepare(source, (0, 43, 0, 43))
    assert canvas.size == mask.size == (768, 432)
    assert box == (255, 43, 512, 389)
    _x0, y0, _x1, y1 = box
    assert y0 >= 43 and y1 <= 389
    assert mask.crop(box).getextrema() == (0, 0)
    assert mask.getpixel((384, 0)) == 255
    assert mask.getpixel((384, 431)) == 255
    output = placement.restore(source, Image.new("RGB", canvas.size, "red"), box)
    output_box = tuple(round(value * 2.5) for value in box)
    assert output_box == (638, 108, 1280, 972)
    left, top, right, bottom = output_box
    expected = source.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    assert output.size == (1920, 1080)
    assert ImageChops.difference(output.crop(output_box), expected).getbbox() is None
    assert source.tobytes() == before


@pytest.mark.parametrize("size", [(81, 109), (211, 83), (101, 101), (17, 219)])
@pytest.mark.parametrize(
    "margins",
    [(39, 0, 0, 0), (0, 43, 0, 0), (0, 0, 39, 0), (0, 0, 0, 43), (0, 0, 0, 0)],
)
def test_directional_placement_has_an_exact_protected_rectangle(size, margins):
    source = Image.new("RGB", size, "navy")
    source.putpixel((size[0] // 2, size[1] // 2), (255, 90, 20))
    before = source.tobytes()
    canvas, mask, box = placement.prepare(source, margins)
    left, top, right, bottom = box
    assert left >= margins[0] and top >= margins[1]
    assert right <= 768 - margins[2] and bottom <= 432 - margins[3]
    expected_mask = Image.new("L", canvas.size, 255)
    expected_mask.paste(0, box)
    assert ImageChops.difference(mask, expected_mask).getbbox() is None
    output = placement.restore(source, Image.new("RGB", canvas.size, "red"), box)
    output_box = tuple(round(v * 2.5) for v in box)
    x0, y0, x1, y1 = output_box
    fitted = source.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    assert ImageChops.difference(output.crop(output_box), fitted).getbbox() is None
    assert source.tobytes() == before


@pytest.mark.parametrize(
    "margins",
    [
        (-1, 0, 0, 0),
        (768, 0, 0, 0),
        (0, 216, 0, 216),
        (0, 0, 0),
        (True, 0, 0, 0),
        (float("nan"), 0, 0, 0),
    ],
)
def test_invalid_margins_are_rejected(margins):
    with pytest.raises(ValueError):
        placement.prepare(Image.new("RGB", (100, 100)), margins)


@pytest.mark.parametrize(
    "box", [(0, 0, 0, 5), (-1, 0, 8, 8), (0, 0, 769, 432), (0, 0, 8), (False, 0, 8, 8)]
)
def test_invalid_restore_box_is_rejected(box):
    with pytest.raises(ValueError):
        placement.restore(
            Image.new("RGB", (100, 100)), Image.new("RGB", (768, 432)), box
        )


def test_wrong_generated_dimensions_are_rejected():
    with pytest.raises(ValueError):
        placement.restore(
            Image.new("RGB", (100, 100)), Image.new("RGB", (1920, 1080)), (0, 0, 8, 8)
        )


def test_demo_refuses_traversal_and_existing_evidence(tmp_path):
    for name in ("../escape", "a/b", ".", "", "a\\b"):
        with pytest.raises(ValueError):
            placement.write_demo(tmp_path, name)
    (tmp_path / "existing").mkdir()
    with pytest.raises(FileExistsError):
        placement.write_demo(tmp_path, "existing")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["existing"]
