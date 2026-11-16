"""Subprocess entrypoint for 3840x2160 wallpaper outpainting with the optional local CUDA environment."""

import argparse
import gc
import math
import os
from pathlib import Path
import shutil
import sys
import types

from PIL import Image, ImageOps

from app.engine import ROOT, EngineAdapter
from app.wallpaper import (
    LAB,
    SIZE,
    prepare,
    restore,
    restore_feathered,
)

# Blend the seam over the band PowerPaint redraws (8 px overlap at 768 wide, refine overlap 16 px at 1536 = 40 px at 3840);
# seam-blend-20261116: 32-48 px joins strands/colors, 96 px ghosts the blurry copy of the source.
FEATHER_PX = 40

# hires refine at 1536x864 joins strokes across the seam; 0.35 keeps the first-pass layout (hires-refine-20261116).
REFINE_STRENGTH = 0.35

SEEDS = (42, 123, 7)


def _is_cuda_oom(exc, torch):
    oom_cls = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if (
        isinstance(oom_cls, type)
        and issubclass(oom_cls, BaseException)
        and isinstance(exc, oom_cls)
    ):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def configure_offline_env():
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HOME=str(LAB / "cache"),
    )


def load_source(path):
    with Image.open(path) as image:
        # Shrink huge 4x native intermediates before full-resolution transpose/RGB allocation to cap peak RAM.
        w, h = image.size
        exif = image.getexif()
        if exif.get(0x0112) in (5, 6, 7, 8):
            w, h = h, w
        s = min(SIZE[0] / w, SIZE[1] / h)
        k = max(1, math.floor(1 / s)) if s < 1 else 1
        if k > 1:
            if exif and "exif" not in image.info:
                image.info["exif"] = exif.tobytes()
            image = image.reduce(k)
        return ImageOps.exif_transpose(image).convert("RGB")


def auto_scene(source, preset, prompt):
    if preset != "anime" or prompt.strip():
        return ""
    lab_code = str(ROOT / "lab" / "outpainting")
    if lab_code not in sys.path:
        sys.path.insert(0, lab_code)
    try:
        import scene_tags

        tagger = scene_tags.load_tagger(LAB)
    except (RuntimeError, ImportError):
        return ""
    scene = scene_tags.scene_prompt(source, tagger)
    del tagger
    gc.collect()
    return scene


def generate(source, preset, prompt="", scene=""):
    canvas, mask = prepare(source)
    if mask.getextrema() == (0, 0):
        return None, mask
    lab_code = str(ROOT / "lab" / "outpainting")
    if lab_code not in sys.path:
        sys.path.insert(0, lab_code)
    import character_check
    import powerpaint_gpu as pp
    import torch

    model_info = pp.verify_powerpaint_preflight(LAB)
    gate = types.SimpleNamespace(LAB=LAB, torch=torch)
    pipe = pp.load_powerpaint_pipeline(gate, model_info)
    recipe = pp.RECIPES_PP["official"]
    assert recipe["promptU"].endswith(", empty scene")
    base = recipe["promptU"][: -len(", empty scene")]
    prompt = prompt.strip()
    scene = scene.strip()
    if prompt:
        prompt_u = f"{base}, {prompt}, empty scene"
    elif scene:
        prompt_u = f"{base}, {scene}, empty scene"
    else:
        prompt_u = recipe["promptU"]
    if preset != "anime":
        prompt_u = prompt_u.replace("anime illustration", "photograph", 1)
    pipeline_mask = pp.expand_mask(mask, recipe["mask_overlap_px"])
    image = pp.prepare_brushnet_input(canvas, pipeline_mask)
    try:
        session = character_check.load_session(LAB)
        best_generated = None
        best_key = None
        best_seed = None
        for seed in SEEDS:
            result = pipe(
                promptA=pp.PROMPT_A,
                promptB=pp.PROMPT_B,
                promptU=prompt_u,
                tradoff=pp.TRADOFF,
                tradoff_nag=pp.TRADOFF_NAG,
                image=image,
                mask=pipeline_mask.convert("RGB"),
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
            generated = result.images[0]
            del result
            cm = character_check.compute_character_mask(generated, session)
            score = character_check.extra_character_score(cm, mask)
            flagged = (
                score >= character_check.EXTRA_CHARACTER_THRESHOLD
                or character_check.detached_character_fraction(cm, mask)
                >= character_check.DETACHED_CHARACTER_THRESHOLD
            )
            key = (flagged, score)
            if best_key is None or key < best_key:
                best_key = key
                best_generated = generated
                best_seed = seed
            if not flagged:
                break
        generated = best_generated
        if not best_key[0]:
            import hires_refine

            try:
                refined = hires_refine.refine(
                    pipe,
                    torch,
                    source,
                    best_generated,
                    prompt_u,
                    recipe["negative_promptU"],
                    seed=best_seed,
                    strength=REFINE_STRENGTH,
                )
            except Exception as exc:
                if not _is_cuda_oom(exc, torch):
                    raise
                empty_cache = getattr(getattr(torch, "cuda", None), "empty_cache", None)
                if callable(empty_cache):
                    empty_cache()
            else:
                _, _, hires_mask = hires_refine.build_init(source, best_generated)
                cm = character_check.compute_character_mask(refined, session)
                flagged = (
                    character_check.extra_character_score(cm, hires_mask)
                    >= character_check.EXTRA_CHARACTER_THRESHOLD
                    or character_check.detached_character_fraction(cm, hires_mask)
                    >= character_check.DETACHED_CHARACTER_THRESHOLD
                )
                if not flagged:
                    generated = refined
    finally:
        pp.release_pipeline(pipe, gate)
        pipe = None
    if best_key[0]:
        # every seed drew extra characters: SDXL repaints the affected sides (PowerPaint is off the GPU)
        import sdxl_repair_lab

        generated = sdxl_repair_lab.repair_with_sdxl(generated, mask, session)
    if generated.size not in ((768, 432), (1536, 864)):
        raise SystemExit(1)
    return generated, mask


def upscale_background(generated, work_dir, preset):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    generated.save(work_dir / "source.png", format="PNG")
    EngineAdapter().process(work_dir, 4, preset)
    with Image.open(work_dir / "native.png") as native:
        native.load()
        if native.size != (generated.width * 4, generated.height * 4):
            raise SystemExit(1)
        return native.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="")
    parser.add_argument("--preset", choices=("anime", "photo"), required=True)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if len(args.prompt) > 500:
        parser.error("Background description is too long")
    configure_offline_env()
    source = load_source(args.source)
    scene = auto_scene(source, args.preset, args.prompt)
    if args.preset == "anime" and args.prompt.strip() == "":
        args.output.with_name(args.output.stem + "-scene.txt").write_text(
            scene, encoding="utf-8"
        )
    generated, _ = generate(source, args.preset, args.prompt, scene=scene)
    if generated is None:
        restore(source, Image.new("RGB", (768, 432), "white")).save(
            args.output, format="PNG"
        )
        return
    work = args.output.parent / f"{args.output.stem}-esrgan"
    work.mkdir()
    try:
        native = upscale_background(generated, work, args.preset)
        final = restore_feathered(source, native, FEATHER_PX)
        final.save(args.output, format="PNG")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
