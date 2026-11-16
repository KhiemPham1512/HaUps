"""Lab: simulate the app's 3-seed PowerPaint + conditional SDXL repair flow on the 32 eval cases.

Reuses frozen PowerPaint outputs in seed order (42, 123, 7), stops at the first candidate
that passes both extra-character checks, and runs SDXL whole-side repair only when every
seed is flagged.

CLI: python lab/outpainting/app_sim_lab.py RUN_NAME [--source ...] [--seed7-source ...]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import character_check as cc  # noqa: E402

LAB = Path(__file__).resolve().parents[2] / ".local-artifacts" / "outpainting"
SEEDS = (42, 123, 7)


def is_flagged(score, detached):
    return (
        score >= cc.EXTRA_CHARACTER_THRESHOLD
        or detached >= cc.DETACHED_CHARACTER_THRESHOLD
    )


def choose(candidates, evaluate):
    """Pick the best (seed, image) candidate using the app's extra-character policy.

    candidates: list of (seed, image).
    evaluate: callable(image) -> (score, detached).
    Returns (chosen_seed, chosen_image, flagged, tried).
    """
    best_key = None
    best_seed = None
    best_image = None
    tried = []
    for seed, image in candidates:
        score, detached = evaluate(image)
        flagged = is_flagged(score, detached)
        tried.append((seed, score, detached, flagged))
        key = (flagged, score)
        if best_key is None or key < best_key:
            best_key = key
            best_seed = seed
            best_image = image
        if not flagged:
            break
    return best_seed, best_image, best_key[0], tried


def evaluate_image(image, mask, session):
    cm = cc.compute_character_mask(image, session)
    return cc.extra_character_score(cm, mask), cc.detached_character_fraction(cm, mask)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    parser.add_argument("--source", default="v2-ppofficial-full-20261114")
    parser.add_argument("--seed7-source", default="v2-ppofficial-seed7-20261114")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_name):
        parser.error("invalid run name")
    out = LAB / args.run_name
    out.mkdir()  # never overwrite evidence
    session = cc.load_session(LAB)
    source_root = LAB / args.source
    seed7_root = LAB / args.seed7_source
    records = []
    for case in sorted(p for p in source_root.iterdir() if p.is_dir()):
        started = perf_counter()
        mask = Image.open(case / "mask.png").convert("L")
        candidates = [
            (42, Image.open(case / "powerpaint-v2-42-generated.png").convert("RGB")),
            (123, Image.open(case / "powerpaint-v2-123-generated.png").convert("RGB")),
            (7, Image.open(seed7_root / case.name / "powerpaint-v2-7-generated.png").convert("RGB")),
        ]
        chosen_seed, chosen_image, chosen_flagged, tried = choose(
            candidates, lambda img: evaluate_image(img, mask, session)
        )
        target = out / case.name
        target.mkdir()
        if chosen_flagged:
            import sdxl_repair_lab

            chosen_image.save(target / "before-repair.png")
            final_image = sdxl_repair_lab.repair_with_sdxl(chosen_image, mask, session)
            final_score, final_detached = evaluate_image(final_image, mask, session)
            final_flagged = is_flagged(final_score, final_detached)
            repaired = True
        else:
            final_image = chosen_image
            chosen_entry = next(t for t in tried if t[0] == chosen_seed)
            final_score, final_detached = chosen_entry[1], chosen_entry[2]
            final_flagged = False
            repaired = False
        final_image.save(target / "final.png")
        record = {
            "case": case.name,
            "tried_seeds": [t[0] for t in tried],
            "scores": [round(t[1], 4) for t in tried],
            "detached": [round(t[2], 4) for t in tried],
            "chosen_seed": chosen_seed,
            "chosen_flagged": chosen_flagged,
            "repaired": repaired,
            "final_score": round(final_score, 4),
            "final_detached": round(final_detached, 4),
            "final_flagged": final_flagged,
            "seconds": round(perf_counter() - started, 1),
        }
        records.append(record)
        print(record, flush=True)
    summary = {
        "cases": len(records),
        "repaired": sum(1 for r in records if r["repaired"]),
        "final_clean": sum(1 for r in records if not r["final_flagged"]),
        "seeds_used_mean": round(
            sum(len(r["tried_seeds"]) for r in records) / len(records), 4
        )
        if records
        else 0.0,
    }
    (out / "records.json").write_text(
        json.dumps({**summary, "summary": summary, "records": records}, indent=2),
        encoding="utf-8",
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
