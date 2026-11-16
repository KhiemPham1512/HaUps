"""Geometry-only lab; no inference and no change to production placement.

Input must already be EXIF-oriented RGB. Final fitted dimensions are derived
from rounded inference-box endpoints, rather than a second contain operation;
integer rounding can slightly change aspect ratio. This is a lab recipe only.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageOps

INFERENCE = (768, 432)
OUTPUT = (1920, 1080)
RECIPE = "placement-geometry-v1"


def prepare(source, margins=(0, 0, 0, 0)):
    """Return canvas, mask, and a half-open box; margins are L/T/R/B pixels."""
    if source.mode != "RGB":
        raise ValueError("PoC requires an oriented RGB image")
    if len(margins) != 4 or any(type(v) is not int or v < 0 for v in margins):
        raise ValueError("Margins must be four nonnegative integers")
    left, top, right, bottom = margins
    width, height = INFERENCE[0] - left - right, INFERENCE[1] - top - bottom
    if width < 1 or height < 1:
        raise ValueError("Margins leave no image area")
    main = ImageOps.contain(source, (width, height), Image.Resampling.LANCZOS)
    x, y = left + (width - main.width) // 2, top + (height - main.height) // 2
    box = (x, y, x + main.width, y + main.height)
    canvas = Image.new("RGB", INFERENCE, "white")
    canvas.paste(main, (x, y))
    mask = Image.new("L", INFERENCE, 255)
    mask.paste(0, box)
    return canvas, mask, box


def restore(source, generated, box):
    """Map the same box to output using round (nearest, ties to even)."""
    if source.mode != "RGB" or generated.mode != "RGB" or generated.size != INFERENCE:
        raise ValueError("Expected RGB source and 768x432 RGB generated image")
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError("Invalid placement box")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= INFERENCE[0] and 0 <= y0 < y1 <= INFERENCE[1]):
        raise ValueError("Placement box is outside the inference canvas")
    output_box = tuple(round(value * 2.5) for value in box)
    left, top, right, bottom = output_box
    main = source.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    output = generated.resize(OUTPUT, Image.Resampling.LANCZOS)
    output.paste(main, (left, top))
    return output


def write_demo(parent, run_name):
    """Create fresh synthetic evidence under parent, never use private images."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_name):
        raise ValueError("Use a simple fresh run name, without paths")
    folder = Path(parent) / run_name
    folder.mkdir(exist_ok=False)
    cases = {
        "baseline": (0, 0, 0, 0),
        "left": (77, 0, 0, 0),
        "top": (0, 43, 0, 0),
        "right": (0, 0, 77, 0),
        "bottom": (0, 0, 0, 43),
        "vertical": (0, 43, 0, 43),
    }
    sizes = [(81, 109), (211, 83), (101, 101), (17, 219)]
    sheet = Image.new("RGB", (256 * len(cases), 168 * len(sizes)), "white")
    records = []
    for row, size in enumerate(sizes):
        source = Image.new("RGB", size, "#193d71")
        draw = ImageDraw.Draw(source)
        draw.line((0, 0, size[0] - 1, size[1] - 1), fill="orange", width=3)
        draw.rectangle(
            (size[0] // 4, size[1] // 4, size[0] * 3 // 4, size[1] * 3 // 4),
            outline="cyan",
            width=2,
        )
        before = hashlib.sha256(source.tobytes()).hexdigest()
        for column, (direction, margins) in enumerate(cases.items()):
            case = folder / f"{size[0]}x{size[1]}-{direction}"
            case.mkdir()
            canvas, mask, box = prepare(source, margins)
            generated = Image.new("RGB", INFERENCE, "#cc99bb")
            output = restore(source, generated, box)
            output_box = tuple(round(v * 2.5) for v in box)
            x0, y0, x1, y1 = output_box
            fitted = source.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
            assert (
                ImageChops.difference(output.crop(output_box), fitted).getbbox() is None
            )
            assert hashlib.sha256(source.tobytes()).hexdigest() == before
            source.save(case / "source.png")
            canvas.save(case / "canvas.png")
            mask.save(case / "mask.png")
            output.save(case / "restored.png")
            sheet.paste(output.resize((256, 144)), (column * 256, row * 168))
            ImageDraw.Draw(sheet).text(
                (column * 256 + 4, row * 168 + 146),
                f"{size[0]}x{size[1]} {direction}",
                fill="black",
            )
            records.append(
                {
                    "case": case.name,
                    "source_rgb_sha256": before,
                    "margins_ltrb": margins,
                    "inference_box": box,
                    "output_box": output_box,
                    "foreground_exact": True,
                }
            )
    sheet.save(folder / "contact-sheet.png")
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "recipe_version": RECIPE,
                "synthetic_only": True,
                "inference_run": False,
                "inference_size": INFERENCE,
                "output_size": OUTPUT,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name", help="Fresh evidence directory name")
    args = parser.parse_args()
    print(write_demo(Path(__file__).resolve().parents[2] / ".local-artifacts" / "outpainting", args.run_name))
