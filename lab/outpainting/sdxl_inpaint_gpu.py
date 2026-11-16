"""Lab: outpaint the frozen eval cases with the merged anime SDXL inpainting UNet.

Reuses canvas.png / mask.png of an existing PowerPaint run (768x432), upscales them to
1024x576 (SDXL needs ~1 MP), generates with StableDiffusionXLInpaintPipeline and composites
the result over the canvas so known pixels stay exact. Writes sdxl-<seed>.png, mask.png
per case and records.json to a NEW run dir.

CLI: python lab/outpainting/sdxl_inpaint_gpu.py RUN_NAME [--seeds 42 123] [--limit N] [--cases eval-5-left ...]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import character_check as cc  # noqa: E402

LAB = Path(__file__).resolve().parents[2] / ".local-artifacts" / "outpainting"
SDXL = LAB / "sdxl"
SIZE = (1024, 576)
OVERLAP_PX = 11  # same share of the frame as PowerPaint's 8 px at 768
PROMPT = "no humans, scenery, background, masterpiece, high score, great score, absurdres"
NEGATIVE = (
    "1girl, 1boy, 2girls, multiple girls, multiple boys, people, person, face, character, "
    "lowres, bad anatomy, text, watermark, signature, worst quality, low quality, low score"
)
STEPS = 28
GUIDANCE = 6.0


def prepare_case(case_dir, hires=False):
    """Canvas and mask of a 768x432 lab case at SDXL size; mask 255 = generate.

    hires: paste source.png straight into the scaled known box instead of upscaling the
    768-wide canvas (the lab canvas is exactly source resized into that box).
    """
    mask = Image.open(case_dir / "mask.png").convert("L")
    if hires:
        scale = SIZE[0] / mask.width
        x0, y0, x1, y1 = ImageOps.invert(mask).getbbox()
        box = tuple(round(v * scale) for v in (x0, y0, x1, y1))
        canvas = Image.new("RGB", SIZE, "white")
        with Image.open(case_dir / "source.png") as source:
            canvas.paste(source.convert("RGB").resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS), box[:2])
        mask = Image.new("L", SIZE, 255)
        mask.paste(0, box)
    else:
        canvas = Image.open(case_dir / "canvas.png").convert("RGB").resize(SIZE, Image.Resampling.LANCZOS)
        mask = mask.resize(SIZE, Image.Resampling.NEAREST)
    pipeline_mask = mask.filter(ImageFilter.MaxFilter(2 * OVERLAP_PX + 1))
    return canvas, mask, pipeline_mask


def prefill(canvas, mask):
    """Fooocus-style outpaint init: stretch the known box's edge pixels outward, blur them."""
    import numpy as np

    x0, y0, x1, y1 = ImageOps.invert(mask).getbbox()
    known = np.asarray(canvas.crop((x0, y0, x1, y1)))
    w, h = canvas.size
    filled = Image.fromarray(np.pad(known, ((y0, h - y1), (x0, w - x1), (0, 0)), mode="edge"))
    return Image.composite(filled.filter(ImageFilter.GaussianBlur(48)), canvas, mask)


def load_pipeline(torch, scheduler="euler_a", unet_dir=None, base_text_encoders=False):
    from diffusers import (
        AutoencoderKL,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        StableDiffusionXLInpaintPipeline,
        UNet2DConditionModel,
    )

    dtype = torch.float16
    unet_dir = unet_dir or SDXL / "animagine-inpaint-ad" / "unet"
    kwargs = {"variant": "fp16"} if (unet_dir / "diffusion_pytorch_model.fp16.safetensors").is_file() else {}
    unet = UNet2DConditionModel.from_pretrained(unet_dir, torch_dtype=dtype, **kwargs)
    vae = AutoencoderKL.from_pretrained(SDXL / "sdxl-vae-fp16-fix", torch_dtype=dtype)
    extra = {}
    if base_text_encoders:  # SDXL base 1.0 encoders, matching the original inpainting UNet
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        base = SDXL / "sdxl-base-1.0"
        extra = {
            "text_encoder": CLIPTextModel.from_pretrained(base / "text_encoder", variant="fp16", torch_dtype=dtype),
            "text_encoder_2": CLIPTextModelWithProjection.from_pretrained(base / "text_encoder_2", variant="fp16", torch_dtype=dtype),
            "tokenizer": CLIPTokenizer.from_pretrained(base / "tokenizer"),
            "tokenizer_2": CLIPTokenizer.from_pretrained(base / "tokenizer_2"),
        }
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        SDXL / "animagine-xl-4.0", unet=unet, vae=vae, torch_dtype=dtype, **extra
    )
    if scheduler == "dpmpp_2m_karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    else:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()  # one sub-model on the 6 GB GPU at a time
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    parser.add_argument("--source", default="v2-ppofficial-full-20261114")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scheduler", choices=("euler_a", "dpmpp_2m_karras"), default="euler_a")
    parser.add_argument("--guidance", type=float, default=GUIDANCE)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative", default=NEGATIVE)
    parser.add_argument("--unet", choices=("merged", "sdxl-inpaint"), default="merged",
                        help="sdxl-inpaint = original SDXL inpainting UNet without the anime merge (diagnosis)")
    parser.add_argument("--base-text-encoders", action="store_true", help="use SDXL base 1.0 text encoders/tokenizers")
    parser.add_argument("--prefill", action="store_true", help="pre-fill the unknown area with blurred stretched edges")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--hires", action="store_true", help="build the 1024 canvas from source.png")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_name):
        parser.error("invalid run name")
    cases = sorted(p for p in (LAB / args.source).iterdir() if p.is_dir())
    if args.cases:
        cases = [p for p in cases if p.name in args.cases]
    cases = cases[: args.limit]
    out = LAB / args.run_name
    out.mkdir()  # never overwrite evidence

    import torch

    session = cc.load_session(LAB)
    pipe = load_pipeline(torch, args.scheduler, SDXL / "sdxl-inpaint-0.1" / "unet" if args.unet == "sdxl-inpaint" else None, args.base_text_encoders)
    merge = json.loads((SDXL / "animagine-inpaint-ad" / "merge.json").read_text(encoding="utf-8"))
    records = []
    for case in cases:
        canvas, mask, pipeline_mask = prepare_case(case, args.hires)
        if args.prefill:
            canvas = prefill(canvas, mask)
        target = out / case.name
        target.mkdir()
        mask.save(target / "mask.png")
        for seed in args.seeds:
            torch.cuda.reset_peak_memory_stats()
            started = perf_counter()
            generated = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative,
                image=canvas,
                mask_image=pipeline_mask,
                width=SIZE[0],
                height=SIZE[1],
                strength=args.strength,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).images[0]
            seconds = perf_counter() - started
            if generated.size != SIZE:
                raise SystemExit(f"unexpected output size {generated.size}")
            result = Image.composite(generated, canvas, pipeline_mask)
            result.save(target / f"sdxl-{seed}.png")
            score = cc.extra_character_score(cc.compute_character_mask(result, session), mask)
            records.append({
                "case": case.name, "seed": seed, "seconds": round(seconds, 1),
                "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                "extra_character_score": round(score, 4),
            })
            print(len(records), records[-1], flush=True)
    summary = {
        "model": "animagine-xl-4.0 + (sdxl-inpaint-0.1 - sdxl-base-1.0)" if args.unet == "merged" else "sdxl-inpaint-0.1 unet + animagine text encoders/vae-fp16-fix",
        "merged_unet_sha256": merge["merged_sha256"],
        "size": SIZE, "steps": args.steps, "guidance": args.guidance, "strength": args.strength, "prefill": args.prefill,
        "prompt": args.prompt, "negative": args.negative, "overlap_px": OVERLAP_PX,
        "scheduler": args.scheduler, "hires": args.hires, "base_text_encoders": args.base_text_encoders, "source_run": args.source,
        "ok": sum(r["extra_character_score"] < cc.EXTRA_CHARACTER_THRESHOLD for r in records),
        "records": records,
    }
    (out / "records.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"ok={summary['ok']}/{len(records)}", flush=True)


if __name__ == "__main__":
    main()
