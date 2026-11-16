import importlib.util
from pathlib import Path

import pytest
from PIL import Image, ImageChops

spec = importlib.util.spec_from_file_location(
    "region_poc",
    Path(__file__).resolve().parents[1] / "lab/outpainting/region_poc.py",
)
regions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regions)


@pytest.mark.parametrize("direction", ["left", "top", "right", "bottom"])
def test_split_masks_partition_only_unknown_pixels(direction):
    box = (10, 8, 30, 24)
    mask = Image.new("L", (40, 32), 255)
    mask.paste(0, box)
    first, second = regions.split_masks(mask, box, direction)
    assert first.getbbox() is not None
    assert ImageChops.lighter(first, second).tobytes() == mask.tobytes()
    assert ImageChops.darker(first, second).getbbox() is None
    assert first.crop(box).getextrema() == second.crop(box).getextrema() == (0, 0)
    expected = {
        "left": (0, 8, 10, 24),
        "top": (10, 0, 30, 8),
        "right": (30, 8, 40, 24),
        "bottom": (10, 24, 30, 32),
    }
    assert first.getbbox() == expected[direction]


@pytest.mark.parametrize(
    "direction,box,mode",
    [
        ("unknown", (10, 8, 30, 24), "L"),
        ("top", (-1, 8, 30, 24), "L"),
        ("top", (10, 8, 30, 24), "RGB"),
    ],
)
def test_invalid_region_request_is_rejected(direction, box, mode):
    with pytest.raises(ValueError):
        regions.split_masks(Image.new(mode, (40, 32)), box, direction)
