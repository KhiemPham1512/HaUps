"""Lab experiment for feathered seam blending on 3840x2160 AI wallpapers.

CLI:
  python lab/outpainting/seam_blend_lab.py RUN_NAME --images DIR [--limit N] [--widths 0 16 32 48 96]
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from time import perf_counter

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import outpaint_runner  # noqa: E402
from app.wallpaper import LAB, SIZE, restore_feathered  # noqa: E402

RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
CROP_SIZE = 384
GAP_PX = 4
DEFAULT_WIDTHS = (0, 16, 32, 48, 96)


def validate_run_name(run_name):
    if not isinstance(run_name, str) or not RUN_NAME_RE.fullmatch(run_name):
        raise ValueError("invalid run name")
    return run_name


def _has_remove_component(path):
    return any(part.lower() == "remove" for part in Path(path).parts)


def list_images(images_dir, limit=None):
    images_dir = Path(images_dir)
    if _has_remove_component(images_dir):
        return []
    paths = [
        p
        for p in sorted(images_dir.iterdir(), key=lambda item: item.name.lower())
        if p.is_file()
        and not _has_remove_component(p)
        and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if limit is not None:
        paths = paths[:limit]
    return paths


def contained_box_and_sides(source):
    main = ImageOps.contain(source, SIZE, Image.Resampling.LANCZOS)
    x = (SIZE[0] - main.width) // 2
    y = (SIZE[1] - main.height) // 2
    x1 = x + main.width
    y1 = y + main.height
    sides = []
    if x > 0:
        sides.append("left")
    if x1 < SIZE[0]:
        sides.append("right")
    if y > 0:
        sides.append("top")
    if y1 < SIZE[1]:
        sides.append("bottom")
    return [x, y, x1, y1], sides


def seam_crop_boxes(box, sides_ramped):
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    crops = []
    for side in sides_ramped:
        for frac in (1 / 3, 2 / 3):
            if side == "left":
                cx, cy = x0, round(y0 + h * frac)
            elif side == "right":
                cx, cy = x1, round(y0 + h * frac)
            elif side == "top":
                cx, cy = round(x0 + w * frac), y0
            elif side == "bottom":
                cx, cy = round(x0 + w * frac), y1
            else:
                raise ValueError(f"Unknown side: {side!r}")
            left = min(max(0, cx - CROP_SIZE // 2), SIZE[0] - CROP_SIZE)
            top = min(max(0, cy - CROP_SIZE // 2), SIZE[1] - CROP_SIZE)
            crops.append((left, top, left + CROP_SIZE, top + CROP_SIZE))
    return crops


def build_seam_sheet(images_by_width, widths, box, sides_ramped):
    crops = seam_crop_boxes(box, sides_ramped)
    rows = len(crops)
    cols = len(widths)
    if rows == 0 or cols == 0:
        raise ValueError("Seam sheet requires at least one ramped side and one width")
    sheet_w = cols * CROP_SIZE + (cols - 1) * GAP_PX
    sheet_h = rows * CROP_SIZE + (rows - 1) * GAP_PX
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    for r, crop_box in enumerate(crops):
        y = r * (CROP_SIZE + GAP_PX)
        for c, width in enumerate(widths):
            x = c * (CROP_SIZE + GAP_PX)
            patch = images_by_width[width].crop(crop_box)
            sheet.paste(patch, (x, y))
    return sheet


def _reset_cuda_peak():
    try:
        import torch

        if (
            hasattr(torch, "cuda")
            and callable(getattr(torch.cuda, "is_available", None))
            and torch.cuda.is_available()
            and callable(getattr(torch.cuda, "reset_peak_memory_stats", None))
        ):
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _peak_cuda_mb():
    try:
        import torch

        if hasattr(torch, "cuda") and callable(
            getattr(torch.cuda, "max_memory_allocated", None)
        ):
            is_avail = getattr(torch.cuda, "is_available", None)
            if not callable(is_avail) or is_avail():
                return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    except Exception:
        pass
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS)
    )
    args = parser.parse_args(argv)
    try:
        validate_run_name(args.run_name)
    except ValueError as err:
        parser.error(str(err))

    run_dir = LAB / args.run_name
    run_dir.mkdir()
    outpaint_runner.configure_offline_env()

    image_paths = list_images(args.images, limit=args.limit)
    records = []
    for idx, image_path in enumerate(image_paths, start=1):
        case_id = f"case-{idx:02d}"
        source = outpaint_runner.load_source(image_path)
        box, sides_ramped = contained_box_and_sides(source)
        _reset_cuda_peak()
        t_gen = perf_counter()
        generated, _ = outpaint_runner.generate(source, "anime", "")
        generate_seconds = round(perf_counter() - t_gen, 3)
        if generated is None:
            records.append(
                {
                    "case": case_id,
                    "skipped": True,
                    "source_size": list(source.size),
                    "box": box,
                    "sides_ramped": sides_ramped,
                    "generate_seconds": generate_seconds,
                    "upscale_seconds": 0.0,
                    "peak_cuda_mb": _peak_cuda_mb(),
                    "widths": list(args.widths),
                }
            )
            continue

        case_dir = run_dir / case_id
        case_dir.mkdir()
        generated.save(case_dir / "generated.png", format="PNG")

        work_dir = run_dir / f"{case_id}-esrgan"
        t_up = perf_counter()
        try:
            native = outpaint_runner.upscale_background(generated, work_dir, "anime")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        upscale_seconds = round(perf_counter() - t_up, 3)

        images_by_width = {}
        for width in args.widths:
            composited = restore_feathered(source, native, width)
            composited.save(case_dir / f"w{width}.png", format="PNG")
            images_by_width[width] = composited

        if sides_ramped and args.widths:
            sheet = build_seam_sheet(images_by_width, args.widths, box, sides_ramped)
            sheet.save(case_dir / "seams.jpg", format="JPEG", quality=90)

        records.append(
            {
                "case": case_id,
                "skipped": False,
                "source_size": list(source.size),
                "box": box,
                "sides_ramped": sides_ramped,
                "generate_seconds": generate_seconds,
                "upscale_seconds": upscale_seconds,
                "peak_cuda_mb": _peak_cuda_mb(),
                "widths": list(args.widths),
            }
        )

    summary = {
        "run_name": args.run_name,
        "widths": list(args.widths),
        "records": records,
    }
    (run_dir / "records.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    main()
