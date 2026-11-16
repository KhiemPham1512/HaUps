"""Lab: PowerPaint draws the background, SDXL-inpaint repaints only the extra characters.

For every case/seed of SOURCE_RUN whose extra_character_score >= threshold: compose the raw
PowerPaint generation over the canvas, build the repair mask (anime-seg blobs reaching the
far zone), upscale to 1024x576 and repaint that region with the plain SDXL inpainting UNet
(Animagine text encoders + neutral prompt: the config that drew no people 8/8). Up to PASSES
passes; pixels outside the (feathered) mask are kept. Writes before/after PNGs and records.json.

--mask shape: the dilated blobs; --mask box: one rectangle per side of the original frame
(removes the person-shaped hint that made PowerPaint redraw people); --mask side: the
whole generated strip of each affected side (no small pasted-looking rectangles).

CLI: python lab/outpainting/sdxl_repair_lab.py RUN_NAME [--mask shape|box]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import character_check as cc  # noqa: E402
import sdxl_inpaint_gpu as sx  # noqa: E402

LAB = sx.LAB
PASSES = (42, 123)
PROMPT = "empty scene, scenery, detailed background, no people"
NEGATIVE = (
    "person, people, man, woman, girl, boy, face, character, crowd, text, watermark, "
    "signature, blurry, lowres, worst quality"
)


def box_mask(repair, generated_mask, whole_side=False):
    """One rectangle per side of the original frame around the repair pixels on that side.

    whole_side: repaint the entire generated strip of every side that has repair pixels.
    """
    x0, y0, x1, y1 = ImageOps.invert(generated_mask).getbbox()
    w, h = repair.size
    out = Image.new("L", repair.size, 0)
    draw = ImageDraw.Draw(out)
    for side in ((0, 0, x0, h), (x1, 0, w, h), (x0, 0, x1, y0), (x0, y1, x1, h)):
        part = Image.new("L", repair.size, 0)
        part.paste(repair.crop(side), side[:2])
        bbox = part.getbbox()
        if bbox and whole_side:
            draw.rectangle((side[0], side[1], side[2] - 1, side[3] - 1), fill=255)
        elif bbox:
            draw.rectangle((bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1), fill=255)
    return ImageChops.multiply(out, generated_mask)


def repair_mask(image, mask, session, kind):
    repair = cc.extra_character_repair_mask(cc.compute_character_mask(image, session), mask)
    if kind in ("box", "side") and repair.getbbox():
        return box_mask(repair, mask, whole_side=kind == "side")
    return repair


def repaint(pipe, torch, image, region, seed):
    """Repaint region (768x432 L mask) of image at SDXL size; feathered composite back."""
    big = image.resize(sx.SIZE, Image.Resampling.LANCZOS)
    big_mask = region.resize(sx.SIZE, Image.Resampling.NEAREST)
    out = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=big,
        mask_image=big_mask,
        width=sx.SIZE[0],
        height=sx.SIZE[1],
        strength=1.0,
        num_inference_steps=sx.STEPS,
        guidance_scale=sx.GUIDANCE,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0].resize(image.size, Image.Resampling.LANCZOS)
    feather = region.filter(ImageFilter.GaussianBlur(3))
    return Image.composite(out, image, feather)


def repair_with_sdxl(image, mask, session):
    """App fallback: whole-side SDXL repaint of a 768x432 PowerPaint result with extra characters.

    mask: L, 255 = generated area. Loads and frees its own pipeline; returns the repaired image.
    """
    import gc

    import torch

    pipe = sx.load_pipeline(torch, unet_dir=sx.SDXL / "sdxl-inpaint-0.1" / "unet")
    try:
        for seed in PASSES:
            region = repair_mask(image, mask, session, "side")
            if not region.getbbox():
                break
            image = repaint(pipe, torch, image, region, seed)
            if not cc.is_extra_character(cc.compute_character_mask(image, session), mask):
                break
    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    return image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    parser.add_argument("--source", default="v2-ppofficial-full-20261114")
    parser.add_argument("--mask", choices=("shape", "box", "side"), default="shape")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_name):
        parser.error("invalid run name")
    out = LAB / args.run_name
    out.mkdir()  # never overwrite evidence
    session = cc.load_session(LAB)
    cases = []
    for case in sorted(p for p in (LAB / args.source).iterdir() if p.is_dir()):
        mask = Image.open(case / "mask.png").convert("L")
        canvas = Image.open(case / "canvas.png").convert("RGB")
        overlap = mask.filter(ImageFilter.MaxFilter(2 * 8 + 1))  # PowerPaint official overlap
        for seed in (42, 123):
            raw = Image.open(case / f"powerpaint-v2-{seed}-generated.png").convert("RGB")
            image = Image.composite(raw, canvas, overlap)
            score = cc.extra_character_score(cc.compute_character_mask(image, session), mask)
            if score >= cc.EXTRA_CHARACTER_THRESHOLD:
                cases.append((case.name, seed, mask, image, score))
    print(f"flagged={len(cases)}", flush=True)

    import torch

    pipe = sx.load_pipeline(torch, unet_dir=sx.SDXL / "sdxl-inpaint-0.1" / "unet")
    records = []
    for name, seed, mask, image, score in cases:
        target = out / f"{name}-{seed}"
        target.mkdir()
        image.save(target / "before.png")
        history = [round(score, 4)]
        started = perf_counter()
        for repair_seed in PASSES:
            region = repair_mask(image, mask, session, args.mask)
            if not region.getbbox():
                break
            region.save(target / f"region-{repair_seed}.png")
            image = repaint(pipe, torch, image, region, repair_seed)
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
    summary = {
        "source": args.source, "mask": args.mask, "passes": PASSES, "prompt": PROMPT,
        "negative": NEGATIVE, "unet": "sdxl-inpaint-0.1 + animagine text encoders + vae-fp16-fix",
        "fixed": sum(r["fixed"] for r in records), "records": records,
    }
    (out / "records.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"fixed={summary['fixed']}/{len(records)}", flush=True)


if __name__ == "__main__":
    main()
