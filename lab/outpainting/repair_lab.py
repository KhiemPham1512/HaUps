"""Lab: repaint only the extra-character blobs of flagged official outputs.

For every case/seed of SOURCE_RUN whose extra_character_score >= threshold, compose the
raw generation over the canvas, build extra_character_repair_mask, repaint that region
with the official recipe (up to PASSES passes, seed 42 then 123) and keep pixels outside
the repair mask unchanged. Writes before/after PNGs and records.json to a new run dir.

CLI: python lab/outpainting/repair_lab.py RUN_NAME [--source v2-ppofficial-full-20261114]
"""

import argparse
import json
import sys
import types
from pathlib import Path
from time import perf_counter

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import character_check as cc  # noqa: E402
import powerpaint_gpu as pp  # noqa: E402

PASSES = (42, 123)


def repaint(pipe, torch, image, mask, seed):
    recipe = pp.RECIPES_PP["official"]
    result = pipe(
        promptA=pp.PROMPT_A,
        promptB=pp.PROMPT_B,
        promptU=recipe["promptU"],
        tradoff=pp.TRADOFF,
        tradoff_nag=pp.TRADOFF_NAG,
        image=pp.prepare_brushnet_input(image, mask),
        mask=mask.convert("RGB"),
        num_inference_steps=recipe["num_inference_steps"],
        generator=torch.Generator(device="cpu").manual_seed(seed),
        brushnet_conditioning_scale=pp.BRUSHNET_CONDITIONING_SCALE,
        negative_promptA=pp.NEGATIVE_PROMPT_A,
        negative_promptB=pp.NEGATIVE_PROMPT_B,
        negative_promptU=recipe["negative_promptU"],
        guidance_scale=recipe["guidance_scale"],
        width=pp.INFERENCE_WIDTH,
        height=pp.INFERENCE_HEIGHT,
    )
    return Image.composite(result.images[0], image, mask)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    parser.add_argument("--source", default="v2-ppofficial-full-20261114")
    args = parser.parse_args(argv)
    if not pp.RUN_NAME_RE.fullmatch(args.run_name):
        parser.error("invalid run name")
    out = pp.LAB / args.run_name
    out.mkdir()  # never overwrite evidence
    session = cc.load_session(pp.LAB)
    cases = []
    for case in sorted(p for p in (pp.LAB / args.source).iterdir() if p.is_dir()):
        mask = Image.open(case / "mask.png").convert("L")
        canvas = Image.open(case / "canvas.png").convert("RGB")
        pipeline_mask = pp.expand_mask(mask, pp.RECIPES_PP["official"]["mask_overlap_px"])
        for seed in (42, 123):
            raw = Image.open(case / f"powerpaint-v2-{seed}-generated.png").convert("RGB")
            image = Image.composite(raw, canvas, pipeline_mask)
            score = cc.extra_character_score(cc.compute_character_mask(image, session), mask)
            if score >= cc.EXTRA_CHARACTER_THRESHOLD:
                cases.append((case.name, seed, mask, image, score))
    print(f"flagged={len(cases)}", flush=True)

    import torch

    gate = types.SimpleNamespace(LAB=pp.LAB, torch=torch)
    pipe = pp.load_powerpaint_pipeline(gate, pp.verify_powerpaint_preflight(pp.LAB))
    records = []
    try:
        for name, seed, mask, image, score in cases:
            target = out / f"{name}-{seed}"
            target.mkdir()
            image.save(target / "before.png")
            history = [round(score, 4)]
            started = perf_counter()
            for repair_seed in PASSES:
                repair = cc.extra_character_repair_mask(
                    cc.compute_character_mask(image, session), mask
                )
                if repair.getextrema() == (0, 0):
                    break
                image = repaint(pipe, torch, image, repair, repair_seed)
                score = cc.extra_character_score(cc.compute_character_mask(image, session), mask)
                history.append(round(score, 4))
                if score < cc.EXTRA_CHARACTER_THRESHOLD:
                    break
            image.save(target / "after.png")
            records.append({
                "case": name, "seed": seed, "scores": history,
                "fixed": history[-1] < cc.EXTRA_CHARACTER_THRESHOLD,
                "seconds": round(perf_counter() - started, 1),
            })
            print(records[-1], flush=True)
    finally:
        pp.release_pipeline(pipe, gate)
    (out / "records.json").write_text(json.dumps({"source": args.source, "passes": PASSES, "records": records}, indent=2), encoding="utf-8")
    print(f"fixed={sum(r['fixed'] for r in records)}/{len(records)}", flush=True)


if __name__ == "__main__":
    main()
