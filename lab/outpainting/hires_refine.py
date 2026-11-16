"""Second-pass 1536x864 img2img hires refinement for AI wallpaper outpainting.

CLI:
  python lab/outpainting/hires_refine.py RUN_NAME --from-run seam-blend-20261116 --images DIR [--strengths 0.35 0.5] [--cases 1 3] [--probe]
"""

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import sys
from time import perf_counter
import types

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from app import outpaint_runner  # noqa: E402
from app.wallpaper import prepare, restore_feathered  # noqa: E402
import seam_blend_lab  # noqa: E402

LAB = ROOT / ".local-artifacts" / "outpainting"
RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")

SIZE = (1536, 864)
OVERLAP_PX = 16
STEPS = 20
STRENGTH = 0.5
OVERVIEW_THUMB_SIZE = (960, 540)


def validate_run_name(run_name):
    if not isinstance(run_name, str) or not RUN_NAME_RE.fullmatch(run_name):
        raise ValueError(f"Invalid run name: {run_name!r}")
    return run_name


def refine_timesteps(strength, steps):
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or math.isnan(strength)
        or not (0 < strength <= 1)
    ):
        raise ValueError("strength must be in (0, 1]")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be an int >= 1")
    start = min(999, max(1, round(999 * strength)))
    if steps == 1:
        return [start]
    raw = [round(start + (1 - start) * i / (steps - 1)) for i in range(steps)]
    out = []
    for t in raw:
        val = int(t)
        if not out or val < out[-1]:
            out.append(val)
    return out


def build_init(source, generated):
    if not isinstance(source, Image.Image):
        raise ValueError("source must be a PIL Image")
    if not isinstance(generated, Image.Image) or generated.mode != "RGB":
        raise ValueError("generated must be an RGB PIL Image")
    canvas, mask = prepare(source, size=SIZE)
    background = generated.resize(SIZE, Image.Resampling.LANCZOS)
    init_rgb = Image.composite(background, canvas, mask)
    return init_rgb, canvas, mask


def refine(
    pipe,
    torch,
    source,
    generated,
    prompt_u,
    negative_prompt_u,
    seed,
    strength=STRENGTH,
    steps=STEPS,
    guidance_scale=10.0,
):
    import diffusers
    import numpy as np
    import powerpaint_gpu as pp

    init_rgb, canvas, mask = build_init(source, generated)
    pipeline_mask = pp.expand_mask(mask, OVERLAP_PX)
    brush_image = pp.prepare_brushnet_input(canvas, pipeline_mask)

    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    else:
        pipe.vae.enable_tiling()

    arr = np.asarray(init_rgb, dtype=np.float32) / 127.5 - 1.0
    x = (
        torch.from_numpy(arr)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device="cuda", dtype=pipe.vae.dtype)
    )
    with torch.no_grad():
        init_latents = (
            pipe.vae.encode(x).latent_dist.mode() * pipe.vae.config.scaling_factor
        )

    DPMSolverMultistepScheduler = diffusers.DPMSolverMultistepScheduler
    original = pipe.scheduler
    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(original.config)
        ts = refine_timesteps(strength, steps)
        noise = torch.randn(
            init_latents.shape,
            generator=torch.Generator("cpu").manual_seed(seed),
            dtype=torch.float32,
        ).to(init_latents.device, init_latents.dtype)
        # noise the init to timestep ts[0] with the DDPM forward formula (DPM++ add_noise needs a set schedule)
        alpha_t = pipe.scheduler.alphas_cumprod[ts[0]].item()
        latents = (alpha_t**0.5) * init_latents + ((1 - alpha_t) ** 0.5) * noise
        result = pipe(
            promptA=pp.PROMPT_A,
            promptB=pp.PROMPT_B,
            promptU=prompt_u,
            tradoff=pp.TRADOFF,
            tradoff_nag=pp.TRADOFF_NAG,
            image=brush_image,
            mask=pipeline_mask.convert("RGB"),
            timesteps=ts,
            num_inference_steps=len(ts),
            latents=latents,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            brushnet_conditioning_scale=pp.BRUSHNET_CONDITIONING_SCALE,
            negative_promptA=pp.NEGATIVE_PROMPT_A,
            negative_promptB=pp.NEGATIVE_PROMPT_B,
            negative_promptU=negative_prompt_u,
            guidance_scale=guidance_scale,
            width=SIZE[0],
            height=SIZE[1],
        )
    finally:
        pipe.scheduler = original

    output = result.images[0]
    if output.size != SIZE:
        raise RuntimeError(f"Unexpected refined output size: {output.size}")
    return output


def build_overview(images_by_col, col_keys):
    if not col_keys:
        raise ValueError("Overview requires at least one column")
    tw, th = OVERVIEW_THUMB_SIZE
    gap = seam_blend_lab.GAP_PX
    cols = len(col_keys)
    sheet = Image.new("RGB", (cols * tw + (cols - 1) * gap, th), (255, 255, 255))
    for idx, key in enumerate(col_keys):
        thumb = images_by_col[key].resize((tw, th), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (idx * (tw + gap), 0))
    return sheet


def _strength_tag(strength):
    return f"s{int(round(strength * 100)):03d}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    parser.add_argument("--from-run", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--strengths", type=float, nargs="+", default=[0.35, 0.5]
    )
    parser.add_argument("--cases", type=int, nargs="+", default=None)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate_run_name(args.run_name)
        validate_run_name(args.from_run)
        for s in args.strengths:
            refine_timesteps(s, STEPS)
    except ValueError as err:
        parser.error(str(err))
    return args


def main(argv=None):
    args = parse_args(argv)
    from_run_dir = LAB / args.from_run
    out = LAB / args.run_name
    out.mkdir()

    outpaint_runner.configure_offline_env()
    image_paths = seam_blend_lab.list_images(args.images)

    if args.cases is not None:
        candidate_indices = [
            idx
            for idx in args.cases
            if 1 <= idx <= len(image_paths)
            and (from_run_dir / f"case-{idx:02d}" / "generated.png").is_file()
        ]
    else:
        candidate_indices = [
            idx
            for idx in range(1, len(image_paths) + 1)
            if (from_run_dir / f"case-{idx:02d}" / "generated.png").is_file()
        ]

    strengths = list(args.strengths)
    if args.probe:
        candidate_indices = candidate_indices[:1]
        strengths = strengths[:1]

    ordered_strengths = sorted(strengths)

    import character_check
    import powerpaint_gpu as pp
    import torch

    model_info = pp.verify_powerpaint_preflight(LAB)
    gate = types.SimpleNamespace(LAB=LAB, torch=torch)
    pipe = pp.load_powerpaint_pipeline(gate, model_info)
    recipe = pp.RECIPES_PP["official"]
    prompt_u = recipe["promptU"]
    negative = recipe["negative_promptU"]
    session = character_check.load_session(LAB)

    case_records = []
    try:
        for idx in candidate_indices:
            case_id = f"case-{idx:02d}"
            case_out = out / case_id
            case_out.mkdir()

            source = outpaint_runner.load_source(image_paths[idx - 1])
            with Image.open(from_run_dir / case_id / "generated.png") as gen_img:
                generated = gen_img.convert("RGB")

            _, _, mask_hires = build_init(source, generated)
            box, sides_ramped = seam_blend_lab.contained_box_and_sides(source)

            refined_images = {}
            refine_stats = []
            for s in strengths:
                tag = _strength_tag(s)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                t0 = perf_counter()
                refined = refine(
                    pipe,
                    torch,
                    source,
                    generated,
                    prompt_u,
                    negative,
                    seed=42,
                    strength=s,
                )
                seconds = round(perf_counter() - t0, 3)
                alloc_mb = (
                    round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
                    if torch.cuda.is_available()
                    else 0.0
                )
                reserved_mb = (
                    round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2)
                    if torch.cuda.is_available()
                    else 0.0
                )
                refined.save(case_out / f"refined-{tag}.png", format="PNG")
                cm = character_check.compute_character_mask(refined, session)
                score = round(
                    character_check.extra_character_score(cm, mask_hires), 4
                )
                detached = round(
                    character_check.detached_character_fraction(cm, mask_hires), 4
                )
                flagged = bool(
                    score >= character_check.EXTRA_CHARACTER_THRESHOLD
                    or detached >= character_check.DETACHED_CHARACTER_THRESHOLD
                )
                refined_images[tag] = refined
                refine_stats.append(
                    {
                        "strength": s,
                        "tag": tag,
                        "seconds": seconds,
                        "peak_allocated_mb": alloc_mb,
                        "peak_reserved_mb": reserved_mb,
                        "extra_character_score": score,
                        "detached_character_fraction": detached,
                        "flagged": flagged,
                    }
                )
                print(
                    f"case={idx} strength={s:.2f} seconds={seconds:.2f}s peak_mb={alloc_mb:.1f} flagged={flagged}",
                    flush=True,
                )

            finals = {}
            work_base = out / f"{case_id}-esrgan-base"
            try:
                native_base = outpaint_runner.upscale_background(
                    generated, work_base, "anime"
                )
            finally:
                shutil.rmtree(work_base, ignore_errors=True)
            final_base = restore_feathered(
                source, native_base, outpaint_runner.FEATHER_PX
            )
            final_base.save(case_out / "final-base.png", format="PNG")
            finals["base"] = final_base

            for s in ordered_strengths:
                tag = _strength_tag(s)
                work_s = out / f"{case_id}-esrgan-{tag}"
                try:
                    native_s = outpaint_runner.upscale_background(
                        refined_images[tag], work_s, "anime"
                    )
                finally:
                    shutil.rmtree(work_s, ignore_errors=True)
                final_s = restore_feathered(
                    source, native_s, outpaint_runner.FEATHER_PX
                )
                final_s.save(case_out / f"final-{tag}.png", format="PNG")
                finals[tag] = final_s

            col_keys = ["base"] + [_strength_tag(s) for s in ordered_strengths]
            if sides_ramped and col_keys:
                seams = seam_blend_lab.build_seam_sheet(
                    finals, col_keys, box, sides_ramped
                )
                seams.save(case_out / "seams.jpg", format="JPEG", quality=90)
            overview = build_overview(finals, col_keys)
            overview.save(case_out / "overview.jpg", format="JPEG", quality=90)

            case_records.append(
                {
                    "case_index": idx,
                    "case": case_id,
                    "source_size": list(source.size),
                    "box": box,
                    "sides_ramped": sides_ramped,
                    "refines": refine_stats,
                }
            )
    finally:
        pp.release_pipeline(pipe, gate)
        pipe = None

    columns_desc = ["base"] + [_strength_tag(s) for s in ordered_strengths]
    summary = {
        "args": {
            "run_name": args.run_name,
            "from_run": args.from_run,
            "strengths": list(strengths),
            "cases": list(candidate_indices),
            "probe": bool(args.probe),
        },
        "columns": columns_desc,
        "records": case_records,
    }
    (out / "records.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    main()
