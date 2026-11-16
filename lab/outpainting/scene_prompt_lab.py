"""Lab experiment for automatic WD14 scene tags in the PowerPaint wallpaper prompt.

CLI:
  python lab/outpainting/scene_prompt_lab.py RUN_NAME --images DIR --baseline-run hires-refine-20261116 [--cases 1 3] [--tags-only]
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from time import perf_counter

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import app.wallpaper  # noqa: E402
from app import outpaint_runner  # noqa: E402
from app.wallpaper import LAB  # noqa: E402
import character_check  # noqa: E402
from scene_tags import (  # noqa: E402
    load_tagger,
    prompt_text,
    scene_prompt as _default_scene_prompt,
    scene_tags,
)
import seam_blend_lab  # noqa: E402

RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
COMPARE_CASE_SIZE = (1280, 720)
COMPARE_ALL_SIZE = (960, 540)
COMPARE_GAP_PX = 8

scene_prompt = _default_scene_prompt


def validate_run_name(run_name):
    if not isinstance(run_name, str) or not RUN_NAME_RE.fullmatch(run_name):
        raise ValueError("invalid run name")
    return run_name


def _check_character(image, mask768, session):
    thumb = image.resize((768, 432), Image.Resampling.LANCZOS)
    cm = character_check.compute_character_mask(thumb, session)
    extra = character_check.extra_character_score(cm, mask768)
    detached = character_check.detached_character_fraction(cm, mask768)
    flagged = (
        extra >= character_check.EXTRA_CHARACTER_THRESHOLD
        or detached >= character_check.DETACHED_CHARACTER_THRESHOLD
    )
    return {
        "extra_character_score": round(float(extra), 4),
        "detached_character_fraction": round(float(detached), 4),
        "flagged": bool(flagged),
    }


def _build_pair_image(left_img, right_img, cell_size, gap_px=COMPARE_GAP_PX):
    w, h = cell_size
    left_thumb = left_img.resize((w, h), Image.Resampling.LANCZOS)
    right_thumb = right_img.resize((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w * 2 + gap_px, h), (255, 255, 255))
    canvas.paste(left_thumb, (0, 0))
    canvas.paste(right_thumb, (w + gap_px, 0))
    return canvas


def _build_compare_all(pair_rows, cell_size=COMPARE_ALL_SIZE, gap_px=COMPARE_GAP_PX):
    w, h = cell_size
    row_w = w * 2 + gap_px
    total_h = len(pair_rows) * h + (len(pair_rows) - 1) * gap_px
    sheet = Image.new("RGB", (row_w, total_h), (255, 255, 255))
    for idx, row_img in enumerate(pair_rows):
        sheet.paste(row_img, (0, idx * (h + gap_px)))
    return sheet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--baseline-run", default="hires-refine-20261116")
    parser.add_argument("--cases", type=int, nargs="+", default=None)
    parser.add_argument("--tags-only", action="store_true")
    parser.add_argument("--suffix", default="", help="appended to a non-empty scene, e.g. 'no humans'")
    args = parser.parse_args(argv)
    try:
        validate_run_name(args.run_name)
        if args.baseline_run:
            validate_run_name(args.baseline_run)
    except ValueError as err:
        parser.error(str(err))

    run_dir = LAB / args.run_name
    run_dir.mkdir()

    tagger = load_tagger(LAB)
    session = None
    if not args.tags_only:
        outpaint_runner.configure_offline_env()
        session = character_check.load_session(LAB)

    all_paths = seam_blend_lab.list_images(args.images)
    selected_cases = set(args.cases) if args.cases is not None else None

    records = []
    compare_rows = []
    flagged_baseline_count = 0
    flagged_new_count = 0

    for idx, image_path in enumerate(all_paths, start=1):
        if selected_cases is not None and idx not in selected_cases:
            continue
        case_id = f"case-{idx:02d}"
        t_case = perf_counter()
        source = outpaint_runner.load_source(image_path)

        t_tag = perf_counter()
        tags = scene_tags(source, tagger)
        if scene_prompt is _default_scene_prompt:
            scene = ", ".join(prompt_text(name) for name, _ in tags)
        else:
            scene = scene_prompt(source, tagger)
        if scene and args.suffix:
            scene = f"{scene}, {args.suffix}"
        tagger_seconds = round(perf_counter() - t_tag, 3)
        tag_entries = [
            {"name": name, "score": round(float(score), 3)} for name, score in tags
        ]

        if args.tags_only:
            total_seconds = round(perf_counter() - t_case, 3)
            records.append(
                {
                    "case": case_id,
                    "tags": tag_entries,
                    "scene_prompt": scene,
                    "tagger_seconds": tagger_seconds,
                    "total_seconds": total_seconds,
                    "generated_size": None,
                    "baseline": None,
                    "new": None,
                }
            )
            continue

        generated, _ = outpaint_runner.generate(source, "anime", "", scene=scene)
        if generated is None:
            total_seconds = round(perf_counter() - t_case, 3)
            records.append(
                {
                    "case": case_id,
                    "skipped": True,
                    "tags": tag_entries,
                    "scene_prompt": scene,
                    "tagger_seconds": tagger_seconds,
                    "total_seconds": total_seconds,
                    "generated_size": None,
                    "baseline": None,
                    "new": None,
                }
            )
            continue

        case_dir = run_dir / case_id
        case_dir.mkdir()
        work_dir = run_dir / f"{case_id}-esrgan"
        try:
            native = outpaint_runner.upscale_background(generated, work_dir, "anime")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        final = app.wallpaper.restore_feathered(
            source, native, outpaint_runner.FEATHER_PX
        )
        final.save(case_dir / "final-tags.png", format="PNG")
        generated_size = list(generated.size)

        baseline_path = LAB / args.baseline_run / case_id / "final-s035.png"
        with Image.open(baseline_path) as base_img:
            baseline_final = base_img.convert("RGB")

        _, mask768 = app.wallpaper.prepare(source)
        baseline_check = _check_character(baseline_final, mask768, session)
        new_check = _check_character(final, mask768, session)
        if baseline_check["flagged"]:
            flagged_baseline_count += 1
        if new_check["flagged"]:
            flagged_new_count += 1

        compare_case = _build_pair_image(baseline_final, final, COMPARE_CASE_SIZE)
        compare_case.save(case_dir / "compare.jpg", format="JPEG", quality=90)
        compare_rows.append(_build_pair_image(baseline_final, final, COMPARE_ALL_SIZE))

        total_seconds = round(perf_counter() - t_case, 3)
        records.append(
            {
                "case": case_id,
                "tags": tag_entries,
                "scene_prompt": scene,
                "tagger_seconds": tagger_seconds,
                "total_seconds": total_seconds,
                "generated_size": generated_size,
                "baseline": baseline_check,
                "new": new_check,
                "baseline_flagged": baseline_check["flagged"],
                "new_flagged": new_check["flagged"],
            }
        )

    if compare_rows:
        compare_all = _build_compare_all(compare_rows)
        compare_all.save(run_dir / "compare-all.jpg", format="JPEG", quality=90)

    summary = {
        "run_name": args.run_name,
        "baseline_run": args.baseline_run,
        "tags_only": bool(args.tags_only),
        "flagged_baseline_count": flagged_baseline_count,
        "flagged_new_count": flagged_new_count,
        "records": records,
    }
    (run_dir / "records.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    main()
