"""Independent quality gate script: compares CFG 7.0 vs CFG 4.0 outpainting across 5 reference groups."""
from pathlib import Path
import hashlib
import json
import os
import sys
from time import perf_counter

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HOME"] = str(Path(__file__).resolve().parents[2] / ".local-artifacts/outpainting/cache")

import torch
import diffusers
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageChops, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.lock import single_instance
from app.wallpaper import SIZE, prepare, restore

LAB = ROOT / ".local-artifacts/outpainting"
CFGS = (7.0, 4.0)
NEGATIVE_PROMPT = "person, people, face, body, hands, duplicate character, nudity, text, watermark, frame, border, blurry, artifacts"

PROMPTS = {
    "anime": "anime illustration, continuation of the existing background, matching colors, perspective and lighting, empty background, no new subjects",
    "photo": "photograph, continuation of the existing background, matching colors, perspective and lighting, empty background, no new subjects",
}

GROUPS = {
    "portrait": {
        "path": ROOT / ".local-artifacts/anime-focus/gate-face/portrait/source.png",
        "style": "photo",
    },
    "landscape": {
        "path": ROOT / ".local-artifacts/anime-focus/gate-v2/landscape/source.png",
        "style": "photo",
    },
    "text": {
        "path": ROOT / ".local-artifacts/anime-focus/gate-v2/text/source.png",
        "style": "anime",
    },
    "anime": {
        "path": ROOT / ".local-artifacts/anime-focus/gate-v2/anime/source.png",
        "style": "anime",
    },
    "jpeg30": {
        "path": ROOT / ".local-artifacts/anime-focus/gate-v2/jpeg30/source.png",
        "style": "photo",
    },
}


def get_model_revision():
    return json.loads((LAB / "manifest.json").read_text(encoding="utf-8"))["revision"]


def parse_output_dir():
    if len(sys.argv) != 2:
        raise ValueError("Exactly one output directory argument is required")
    out_dir = Path(sys.argv[1]).resolve()
    if not out_dir.is_relative_to(LAB) or out_dir == LAB:
        raise ValueError(f"Output path must be under {LAB}: {out_dir}")
    if out_dir.exists():
        raise FileExistsError(f"Output path already exists (refusing to overwrite evidence): {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def main():
    with single_instance():
        out_dir = parse_output_dir()
        model_revision = get_model_revision()
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            LAB / "model",
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()

        all_records = []

        for group_name, group_info in GROUPS.items():
            print(f"Evaluating group: {group_name} ...", flush=True)
            source_path = group_info["path"]
            style = group_info["style"]
            prompt = PROMPTS[style]

            original_bytes = source_path.read_bytes()
            original_hash = hashlib.sha256(original_bytes).hexdigest()

            with Image.open(source_path) as img:
                source = ImageOps.exif_transpose(img).convert("RGB")

            case_dir = out_dir / group_name
            case_dir.mkdir(parents=True, exist_ok=False)
            (case_dir / "source.png").write_bytes(original_bytes)

            canvas, mask = prepare(source, (768, 432))
            wallpapers = {}
            group_records = []

            for cfg in CFGS:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                started = perf_counter()
                result = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=canvas,
                    mask_image=mask,
                    width=768,
                    height=432,
                    num_inference_steps=30,
                    guidance_scale=cfg,
                    generator=torch.Generator(device="cpu").manual_seed(42),
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_seconds = perf_counter() - started

                assert result.nsfw_content_detected is None, "Safety checker flag detected"

                generated = result.images[0]
                assert generated.size == (768, 432), f"Expected generated size (768, 432), got {generated.size}"

                gen_path = case_dir / f"generated-{cfg:g}.png"
                generated.save(gen_path)

                wallpaper = restore(source, generated)
                wall_path = case_dir / f"wallpaper-{cfg:g}.png"
                wallpaper.save(wall_path)

                wallpapers[cfg] = wallpaper

                with Image.open(wall_path) as verified:
                    verified.load()
                    assert verified.format == "PNG", f"Expected PNG format, got {verified.format}"
                    assert verified.size == SIZE, f"Expected size {SIZE}, got {verified.size}"

                main_img = ImageOps.contain(source, SIZE, Image.Resampling.LANCZOS)
                mx = (SIZE[0] - main_img.width) // 2
                my = (SIZE[1] - main_img.height) // 2
                diff = ImageChops.difference(
                    wallpaper.crop((mx, my, mx + main_img.width, my + main_img.height)), main_img
                ).getbbox()
                assert diff is None, "Foreground content does not exactly match source"

                peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
                peak_res = torch.cuda.max_memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0.0

                record = {
                    "group": group_name,
                    "hash": original_hash,
                    "source_size": list(source.size),
                    "model_revision": model_revision,
                    "inference_size": [768, 432],
                    "output_size": [1920, 1080],
                    "cfg": cfg,
                    "steps": 30,
                    "seed": 42,
                    "style": style,
                    "prompt": prompt,
                    "torch_version": torch.__version__,
                    "diffusers_version": diffusers.__version__,
                    "gpu_name": gpu_name,
                    "inference_seconds": inference_seconds,
                    "torch_allocated_peak_mib": peak_alloc,
                    "torch_reserved_peak_mib": peak_res,
                    "foreground_exact": True,
                }
                group_records.append(record)
                all_records.append(record)

            # Overview comparison: two columns FIT 640x360
            col1 = ImageOps.fit(wallpapers[7.0], (640, 360), Image.Resampling.LANCZOS)
            col2 = ImageOps.fit(wallpapers[4.0], (640, 360), Image.Resampling.LANCZOS)
            overview = Image.new("RGB", (1280, 24 + 360))
            overview.paste(col1, (0, 24))
            overview.paste(col2, (640, 24))
            draw = ImageDraw.Draw(overview)
            draw.text((8, 5), "CFG 7 / FIT", fill=(255, 255, 255))
            draw.text((648, 5), "CFG 4 / FIT", fill=(255, 255, 255))
            overview.save(case_dir / "comparison-overview.png")

            # Seam comparisons: 100% crop 384px per column, centered vertically, clamped
            main_img = ImageOps.contain(source, SIZE, Image.Resampling.LANCZOS)
            mx = (SIZE[0] - main_img.width) // 2
            crop_w, crop_h = 384, 384
            sy1 = max(0, min(SIZE[1] - crop_h, (SIZE[1] - crop_h) // 2))
            sy2 = sy1 + crop_h

            left_x1 = max(0, min(SIZE[0] - crop_w, mx - crop_w // 2))
            left_x2 = left_x1 + crop_w
            left_box = [left_x1, sy1, left_x2, sy2]

            seam_left = Image.new("RGB", (crop_w * 2, 24 + crop_h))
            seam_left.paste(wallpapers[7.0].crop((left_x1, sy1, left_x2, sy2)), (0, 24))
            seam_left.paste(wallpapers[4.0].crop((left_x1, sy1, left_x2, sy2)), (crop_w, 24))
            draw = ImageDraw.Draw(seam_left)
            draw.text((8, 5), "CFG 7 / 100%", fill=(255, 255, 255))
            draw.text((crop_w + 8, 5), "CFG 4 / 100%", fill=(255, 255, 255))
            seam_left.save(case_dir / "comparison-seam-left.png")

            right_x1 = max(0, min(SIZE[0] - crop_w, (mx + main_img.width) - crop_w // 2))
            right_x2 = right_x1 + crop_w
            right_box = [right_x1, sy1, right_x2, sy2]

            seam_right = Image.new("RGB", (crop_w * 2, 24 + crop_h))
            seam_right.paste(wallpapers[7.0].crop((right_x1, sy1, right_x2, sy2)), (0, 24))
            seam_right.paste(wallpapers[4.0].crop((right_x1, sy1, right_x2, sy2)), (crop_w, 24))
            draw = ImageDraw.Draw(seam_right)
            draw.text((8, 5), "CFG 7 / 100%", fill=(255, 255, 255))
            draw.text((crop_w + 8, 5), "CFG 4 / 100%", fill=(255, 255, 255))
            seam_right.save(case_dir / "comparison-seam-right.png")

            crop_boxes = {
                "left_seam": left_box,
                "right_seam": right_box,
            }
            for rec in group_records:
                rec["crop_boxes"] = crop_boxes

            (case_dir / "records.json").write_text(json.dumps(group_records, indent=2), encoding="utf-8")
            assert hashlib.sha256(source_path.read_bytes()).hexdigest() == original_hash

            timing = ", ".join(f"cfg {r['cfg']:g}: {r['inference_seconds']:.2f}s" for r in group_records)
            print(f"[{group_name}] {timing}", flush=True)

        (out_dir / "records.json").write_text(json.dumps(all_records, indent=2), encoding="utf-8")
        print("Quality gate evaluation completed successfully.", flush=True)


if __name__ == "__main__":
    main()
