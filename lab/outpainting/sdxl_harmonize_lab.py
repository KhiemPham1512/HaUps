"""Lab: harmonize repaired outpaints with a light Animagine XL 4.0 pass over the generated area.

Input: a sdxl_repair_lab run (case dirs "<case>-<seed>" with after.png at 768x432). The mask
of the generated area comes from the PowerPaint source run. The image is upscaled to
1024x576 and re-noised only inside the (feathered) generated area at low strength with the
plain Animagine UNet (4-channel; the inpaint pipeline blends latents outside the mask), so
layout stays and colour/line style is unified. Known pixels are pasted back exactly.

CLI: python lab/outpainting/sdxl_harmonize_lab.py RUN_NAME --repair-run RUN [--strengths 0.35 0.5]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import character_check as cc  # noqa: E402
import sdxl_inpaint_gpu as sx  # noqa: E402

LAB = sx.LAB
PROMPT = "no humans, scenery, detailed background, masterpiece, high score, great score, absurdres"
NEGATIVE = (
    "1girl, 1boy, multiple girls, people, person, face, lowres, bad anatomy, text, watermark, "
    "signature, worst quality, low quality, low score, blurry"
)


def load_animagine(torch):
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, StableDiffusionXLInpaintPipeline

    dtype = torch.float16
    vae = AutoencoderKL.from_pretrained(sx.SDXL / "sdxl-vae-fp16-fix", torch_dtype=dtype)
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(sx.SDXL / "animagine-xl-4.0", vae=vae, torch_dtype=dtype)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    parser.add_argument("--repair-run", required=True)
    parser.add_argument("--source", default="v2-ppofficial-full-20261114")
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.35, 0.5])
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_name):
        parser.error("invalid run name")
    out = LAB / args.run_name
    out.mkdir()  # never overwrite evidence

    import torch

    session = cc.load_session(LAB)
    pipe = load_animagine(torch)
    records = []
    for case_dir in sorted(p for p in (LAB / args.repair_run).iterdir() if p.is_dir()):
        case, seed = case_dir.name.rsplit("-", 1)
        mask = Image.open(LAB / args.source / case / "mask.png").convert("L")
        image = Image.open(case_dir / "after.png").convert("RGB")
        target = out / case_dir.name
        target.mkdir()
        big = image.resize(sx.SIZE, Image.Resampling.LANCZOS)
        big_mask = mask.resize(sx.SIZE, Image.Resampling.NEAREST)
        feather = big_mask.filter(ImageFilter.GaussianBlur(6))
        for strength in args.strengths:
            started = perf_counter()
            result = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE,
                image=big,
                mask_image=feather,
                width=sx.SIZE[0],
                height=sx.SIZE[1],
                strength=strength,
                num_inference_steps=40,  # ~14-20 effective steps at these strengths
                guidance_scale=5.0,
                generator=torch.Generator(device="cpu").manual_seed(int(seed)),
            ).images[0]
            # keep the original frame exact; blend softly at the boundary
            final = Image.composite(result, big, feather).resize(mask.size, Image.Resampling.LANCZOS)
            final = Image.composite(final, image, mask)
            final.save(target / f"harmonized-{strength:g}.png")
            score = cc.extra_character_score(cc.compute_character_mask(final, session), mask)
            records.append({"case": case_dir.name, "strength": strength, "score": round(score, 4),
                            "seconds": round(perf_counter() - started, 1)})
            print(records[-1], flush=True)
    summary = {"repair_run": args.repair_run, "prompt": PROMPT, "negative": NEGATIVE,
               "model": "animagine-xl-4.0 (plain) + vae-fp16-fix, EulerA 40 steps, CFG 5",
               "ok": {str(s): sum(r["score"] < cc.EXTRA_CHARACTER_THRESHOLD for r in records if r["strength"] == s) for s in args.strengths},
               "records": records}
    (out / "records.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary["ok"], flush=True)


if __name__ == "__main__":
    main()
