"""Directional geometric masks, not semantic character segmentation."""

from PIL import Image, ImageChops


def split_masks(mask, box, direction):
    x0, y0, x1, y1 = box
    if mask.mode != "L" or not (
        0 <= x0 < x1 <= mask.width and 0 <= y0 < y1 <= mask.height
    ):
        raise ValueError("Invalid mask or placement")
    rectangles = {
        "left": (0, y0, x0, y1),
        "top": (x0, 0, x1, y0),
        "right": (x1, y0, mask.width, y1),
        "bottom": (x0, y1, x1, mask.height),
    }
    if direction not in rectangles:
        raise ValueError("Unknown direction")
    first = Image.new("L", mask.size, 0)
    first.paste(255, rectangles[direction])
    first = ImageChops.darker(first, mask)
    return first, ImageChops.subtract(mask, first)
