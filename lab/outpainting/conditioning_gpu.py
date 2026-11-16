"""Anime-only reference conditioning lab: true baseline vs IP-Adapter Plus SD1.5 (scale=0.5).

CLI:
  python .local-artifacts/outpainting/conditioning_gpu.py RUN_NAME [--model {sd15,dreamshaper8,anylora}] [--sampler {ddim,deis}] [--dataset {development,evaluation,extended-evaluation-v2}] [--recipes {all,baseline}] [--full]

Default smoke runs the last group bottom cut seed 42 across the selected recipes
(300s outer wrapper timeout expected). Passing --full runs all dataset groups x 4
cut directions x 2 seeds x selected recipes (900s outer wrapper timeout expected).
Never used in production or for Photo/upscale/classifier paths.
"""

import argparse
import gc
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageChops, ImageDraw, ImageOps

LAB = Path(__file__).resolve().parents[2] / ".local-artifacts" / "outpainting"
PADDING_RGB = (127, 127, 127)
RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")

BASELINE_RECIPE = "baseline"
ADAPTER_RECIPE = "ip-adapter-0.5"
RECIPES = (BASELINE_RECIPE, ADAPTER_RECIPE)
ADAPTER_SCALE = 0.5

DEFAULT_MODEL = "sd15"
DEFAULT_SAMPLER = "ddim"
SAMPLERS = {
    "ddim": "DDIMScheduler",
    "deis": "DEISMultistepScheduler",
}
DEFAULT_DATASET = "development"
DATASETS = ("development", "evaluation", "extended-evaluation-v2")
EVALUATION_DIR_NAME = "evaluation-anime-20261112-v1"
EVALUATION_DATASET_ID = "anime-evaluation-v1"
EVALUATION_GROUPS = ("eval-a", "eval-b")
EVALUATION_DATASETS = {
    "evaluation": (
        EVALUATION_DIR_NAME,
        EVALUATION_DATASET_ID,
        EVALUATION_GROUPS,
    ),
    "extended-evaluation-v2": (
        "evaluation-anime-extended-20261114-v2",
        "anime-extended-evaluation-v2",
        (
            "eval-1",
            "eval-2",
            "eval-3",
            "eval-4",
            "eval-5",
            "eval-6",
            "eval-7",
            "eval-8",
        ),
    ),
}
CORE_FP16_SAFETENSORS = (
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "text_encoder/model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)
BASE_MODELS = {
    "sd15": {
        "model_dir": "model",
        "manifest": "manifest.json",
        "repository": "stable-diffusion-v1-5/stable-diffusion-inpainting",
        "revision": "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb",
        "in_channels": 9,
        "inpaint_mode": "native9",
        "require_complete": False,
    },
    "dreamshaper8": {
        "model_dir": "candidate-models/dreamshaper8",
        "manifest": "candidate-models/dreamshaper8/manifest.json",
        "repository": "Lykon/dreamshaper-8-inpainting",
        "revision": "a09e9d0bb9849aaf46c8fc59dfa604796da15a39",
        "in_channels": 9,
        "inpaint_mode": "native9",
        "require_complete": True,
    },
    "anylora": {
        "model_dir": "candidate-models/anylora",
        "manifest": "candidate-models/anylora/manifest.json",
        "repository": "Lykon/AnyLoRA",
        "revision": "5dac4354b7f5d32f140e1f0fbc849e92b744f347",
        "in_channels": 4,
        "inpaint_mode": "latent-mask-blend4",
        "require_complete": True,
    },
}

ADAPTER_REVISION = "018e402774aeeddd60609b4ecdb7e298259dc729"
ADAPTER_SUBFOLDER = "models"
ADAPTER_WEIGHT_NAME = "ip-adapter-plus_sd15.safetensors"
IMAGE_ENCODER_FOLDER = "models/image_encoder"
CONFIG_GIT_BLOB_SHA1 = "9c841dd0314e27141ed01a24df1734e407a95c4a"

ADAPTER_FILES = {
    "models/ip-adapter-plus_sd15.safetensors": (
        98183288,
        "a1c250be40455cc61a43da1201ec3f1edaea71214865fb47f57927e06cbe4996",
    ),
    "models/image_encoder/model.safetensors": (
        2528373448,
        "6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030",
    ),
    "models/image_encoder/config.json": (560, None),
}


def reference_image(source):
    """Pad an oriented RGB cropped source to a neutral-gray square without resizing."""
    if source.mode != "RGB":
        raise ValueError("Reference conditioning requires an oriented RGB image")
    width, height = source.size
    side = max(width, height)
    square = Image.new("RGB", (side, side), PADDING_RGB)
    x = (side - width) // 2
    y = (side - height) // 2
    square.paste(source, (x, y))
    return square


def new_output(parent, run_name):
    """Create a fresh run directory under parent, refusing traversal or existing paths."""
    if not isinstance(run_name, str) or not RUN_NAME_RE.fullmatch(run_name):
        raise ValueError("Use a simple fresh run name, without paths")
    base = Path(parent).resolve()
    folder = Path(parent) / run_name
    resolved = folder.resolve()
    if not resolved.is_relative_to(base) or resolved == base:
        raise ValueError("Use a simple fresh run name, without paths")
    folder.mkdir(exist_ok=False)
    return folder


def sha256_file(path):
    """Stream SHA-256 in 1 MiB chunks without loading large weights into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1048576):
            digest.update(chunk)
    return digest.hexdigest()


def verify_adapter_preflight(adapter_root):
    """Verify adapter manifest and all 3 runtime files before loading any GPU model."""
    manifest_path = Path(adapter_root) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing IP-Adapter manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assert manifest.get("complete") is True, "IP-Adapter manifest is not marked complete"
    assert (
        manifest.get("revision") == ADAPTER_REVISION
    ), f"Unexpected IP-Adapter revision: {manifest.get('revision')}"

    recorded = {
        Path(entry["file"]).as_posix(): entry for entry in manifest.get("files", [])
    }
    verified_hashes = {}
    for rel_path, (expected_bytes, expected_lfs_sha256) in ADAPTER_FILES.items():
        assert rel_path in recorded, f"Missing {rel_path} in IP-Adapter manifest"
        entry = recorded[rel_path]
        file_path = Path(adapter_root) / rel_path
        assert file_path.is_file(), f"Missing IP-Adapter runtime file: {file_path}"
        actual_bytes = file_path.stat().st_size
        assert (
            actual_bytes == expected_bytes == int(entry["bytes"])
        ), f"Size mismatch for {rel_path}: {actual_bytes}"
        actual_sha256 = sha256_file(file_path)
        recorded_sha256 = str(entry["sha256"]).lower()
        assert (
            actual_sha256 == recorded_sha256
        ), f"SHA-256 mismatch vs manifest for {rel_path}"
        if expected_lfs_sha256 is not None:
            assert (
                actual_sha256 == expected_lfs_sha256
            ), f"SHA-256 mismatch vs pinned LFS hash for {rel_path}"
        if rel_path == "models/image_encoder/config.json":
            raw = file_path.read_bytes()
            git_blob_sha1 = hashlib.sha1(
                f"blob {len(raw)}\0".encode("ascii") + raw
            ).hexdigest()
            assert (
                git_blob_sha1 == CONFIG_GIT_BLOB_SHA1
            ), f"Unexpected config.json Git blob SHA-1: {git_blob_sha1}"
        verified_hashes[rel_path] = actual_sha256

    return {
        "revision": ADAPTER_REVISION,
        "hashes": verified_hashes,
        "config_git_blob_sha1": CONFIG_GIT_BLOB_SHA1,
    }


def verify_base_model_preflight(
    lab_root, model_name=DEFAULT_MODEL, sampler=DEFAULT_SAMPLER
):
    """Verify selected base checkpoint manifest, UNet channels, and sampler scheduler config."""
    if model_name not in BASE_MODELS:
        raise ValueError(f"Unsupported base model: {model_name}")
    if sampler not in SAMPLERS:
        raise ValueError(f"Unsupported sampler: {sampler}")
    spec = BASE_MODELS[model_name]
    lab_path = Path(lab_root)
    model_dir = lab_path / spec["model_dir"]
    resolved_model_dir = model_dir.resolve()
    manifest_path = lab_path / spec["manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing base model manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assert (
        manifest.get("revision") == spec["revision"]
    ), f"Unexpected revision for {model_name}: {manifest.get('revision')}"

    manifest_repo = (
        manifest.get("repo")
        or manifest.get("repository")
        or manifest.get("repo_id")
        or manifest.get("model_id")
    )
    if spec["require_complete"]:
        assert manifest.get("complete") is True, (
            f"{model_name} manifest is not marked complete"
        )
        assert (
            manifest_repo == spec["repository"]
        ), f"Unexpected repository for {model_name}: {manifest_repo}"
    elif manifest_repo is not None:
        assert (
            manifest_repo == spec["repository"]
        ), f"Unexpected repository for {model_name}: {manifest_repo}"

    unet_config_path = model_dir / "unet" / "config.json"
    if not unet_config_path.is_file():
        raise FileNotFoundError(f"Missing UNet config: {unet_config_path}")
    unet_config = json.loads(unet_config_path.read_text(encoding="utf-8-sig"))
    actual_in_channels = unet_config.get("in_channels")
    assert (
        actual_in_channels == spec["in_channels"]
    ), f"Unexpected UNet in_channels for {model_name}: {actual_in_channels}"

    if sampler == "ddim":
        scheduler_config_path = (
            lab_path / "model" / "scheduler" / "scheduler_config.json"
        )
    else:
        scheduler_config_path = model_dir / "scheduler" / "scheduler_config.json"
    if not scheduler_config_path.is_file():
        raise FileNotFoundError(
            f"Missing scheduler config for sampler={sampler}: {scheduler_config_path}"
        )
    scheduler_config_sha256 = sha256_file(scheduler_config_path)
    scheduler_config = json.loads(
        scheduler_config_path.read_text(encoding="utf-8-sig")
    )
    if not isinstance(scheduler_config, dict):
        raise ValueError(f"Invalid scheduler config JSON: {scheduler_config_path}")
    source_scheduler_class = scheduler_config.get("_class_name", "DDIMScheduler")
    if sampler == "ddim":
        assert (
            source_scheduler_class == "DDIMScheduler"
        ), f"Unexpected baseline scheduler class: {source_scheduler_class}"
    scheduler_class = SAMPLERS[sampler]

    verified_hashes = {}
    if spec["require_complete"]:
        entries = manifest.get("files")
        assert isinstance(entries, list) and entries, (
            f"No files listed in {model_name} manifest"
        )
        for entry in entries:
            rel_raw = (
                entry.get("file", entry.get("path"))
                if isinstance(entry, dict)
                else None
            )
            if not isinstance(rel_raw, str) or not rel_raw.strip():
                raise ValueError(f"Invalid file entry in {model_name} manifest")
            rel_obj = Path(rel_raw)
            if rel_obj.is_absolute() or ".." in rel_obj.parts:
                raise ValueError(f"Manifest path escapes model directory: {rel_raw}")
            file_path = model_dir / rel_obj
            resolved_file = file_path.resolve()
            if (
                not resolved_file.is_relative_to(resolved_model_dir)
                or resolved_file == resolved_model_dir
            ):
                raise ValueError(f"Manifest path escapes model directory: {rel_raw}")
            if file_path.is_symlink() or not resolved_file.is_file():
                raise FileNotFoundError(
                    f"Missing or non-regular model file: {file_path}"
                )
            expected_bytes = int(
                entry["bytes"] if "bytes" in entry else entry["size"]
            )
            actual_bytes = resolved_file.stat().st_size
            assert (
                actual_bytes == expected_bytes
            ), f"Size mismatch for {rel_raw}: {actual_bytes}"
            expected_sha256 = str(entry["sha256"]).lower()
            actual_sha256 = sha256_file(resolved_file)
            assert (
                actual_sha256 == expected_sha256
            ), f"SHA-256 mismatch vs manifest for {rel_raw}"
            verified_hashes[rel_obj.as_posix()] = actual_sha256

        for core_file in CORE_FP16_SAFETENSORS:
            assert core_file in verified_hashes, (
                f"Missing required core fp16 safetensors in {model_name} manifest: {core_file}"
            )
        if sampler == "deis" and "scheduler/scheduler_config.json" in verified_hashes:
            assert (
                scheduler_config_sha256
                == verified_hashes["scheduler/scheduler_config.json"]
            ), f"SHA-256 mismatch for {model_name} scheduler_config.json"
    else:
        for entry in manifest.get("files", []):
            if isinstance(entry, dict):
                rel_raw = entry.get("file", entry.get("path"))
                sha_val = entry.get("sha256")
                if rel_raw and sha_val:
                    verified_hashes[Path(rel_raw).as_posix()] = str(sha_val).lower()

    return {
        "model_name": model_name,
        "model_dir": model_dir,
        "repository": spec["repository"],
        "revision": spec["revision"],
        "in_channels": spec["in_channels"],
        "inpaint_mode": spec["inpaint_mode"],
        "hashes": verified_hashes,
        "selected_sampler": sampler,
        "scheduler_class": scheduler_class,
        "scheduler_source_class": source_scheduler_class,
        "scheduler_config_path": scheduler_config_path.relative_to(lab_path).as_posix(),
        "scheduler_config": scheduler_config,
        "scheduler_config_sha256": scheduler_config_sha256,
    }


def evaluation_sources(lab_root, dataset="evaluation"):
    """Validate frozen anime evaluation dataset manifest and return (group->Path, manifest_sha256)."""
    if not isinstance(dataset, str) or dataset not in EVALUATION_DATASETS:
        raise ValueError(f"Unsupported evaluation dataset: {dataset}")
    eval_dir_name, expected_dataset_id, expected_groups = EVALUATION_DATASETS[
        dataset
    ]
    lab_path = Path(lab_root)
    eval_dir = lab_path / eval_dir_name
    if eval_dir.is_symlink() or not eval_dir.is_dir():
        raise FileNotFoundError(f"Missing evaluation dataset directory: {eval_dir}")
    resolved_eval_dir = eval_dir.resolve()
    manifest_path = eval_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing evaluation dataset manifest: {manifest_path}")
    resolved_manifest = manifest_path.resolve()
    if (
        not resolved_manifest.is_relative_to(resolved_eval_dir)
        or resolved_manifest.parent != resolved_eval_dir
    ):
        raise ValueError("Evaluation manifest escapes dataset directory")

    manifest_sha256 = sha256_file(resolved_manifest)
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict):
        raise ValueError("Evaluation manifest must be a JSON object")
    if manifest.get("dataset") != expected_dataset_id:
        raise ValueError(f"Unexpected evaluation dataset id: {manifest.get('dataset')}")
    if manifest.get("frozen") is not True:
        raise ValueError("Evaluation dataset manifest must be marked frozen=true")
    if manifest.get("purpose") != "evaluation_only_not_training":
        raise ValueError(
            f"Unexpected evaluation dataset purpose: {manifest.get('purpose')}"
        )
    if manifest.get("blind_holdout") is not False:
        raise ValueError("Evaluation dataset blind_holdout must be false")

    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != len(expected_groups):
        raise ValueError(
            f"Evaluation manifest must list exactly {len(expected_groups)} files"
        )

    verified = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Evaluation manifest file entry must be an object")
        group = entry.get("group")
        file_name = entry.get("file")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError("Evaluation manifest entry missing file name")
        file_rel = Path(file_name)
        if (
            file_rel.is_absolute()
            or ".." in file_rel.parts
            or len(file_rel.parts) != 1
        ):
            raise ValueError(f"Evaluation file escapes dataset directory: {file_name}")
        if group not in expected_groups:
            raise ValueError(f"Unsupported evaluation group: {group}")
        if group in verified:
            raise ValueError(f"Duplicate evaluation group: {group}")
        expected_file_name = f"{group}.png"
        if file_name != expected_file_name:
            raise ValueError(
                f"Evaluation file for {group} must be {expected_file_name}, got {file_name}"
            )
        candidate_path = eval_dir / expected_file_name
        if candidate_path.is_symlink():
            raise ValueError(
                f"Evaluation file must not be a symlink: {candidate_path}"
            )
        resolved_file = candidate_path.resolve()
        if (
            not resolved_file.is_relative_to(resolved_eval_dir)
            or resolved_file.parent != resolved_eval_dir
        ):
            raise ValueError(f"Evaluation file escapes dataset directory: {file_name}")
        if not resolved_file.is_file():
            raise FileNotFoundError(f"Missing evaluation image file: {candidate_path}")

        expected_sha256 = str(entry.get("sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError(f"Invalid sha256 in evaluation manifest for {group}")
        actual_sha256 = sha256_file(resolved_file)
        assert (
            actual_sha256 == expected_sha256
        ), f"SHA-256 mismatch vs evaluation manifest for {expected_file_name}"
        verified[group] = resolved_file

    ordered_sources = {group: verified[group] for group in expected_groups}
    return ordered_sources, manifest_sha256


def direction_crop_and_margins(size, direction):
    """Return 20% simulated cut crop_box and single-direction placement margins."""
    w, h = size
    if direction == "left":
        return (w // 5, 0, w, h), (77, 0, 0, 0)
    if direction == "right":
        return (0, 0, w * 4 // 5, h), (0, 0, 77, 0)
    if direction == "top":
        return (0, h // 5, w, h), (0, 43, 0, 0)
    if direction == "bottom":
        return (0, 0, w, h * 4 // 5), (0, 0, 0, 43)
    raise ValueError(f"Unsupported direction: {direction}")


def seam_crop_boxes(output_box):
    """Return four 256x256 100% seam crop boxes centered on each output_box edge."""
    x0, y0, x1, y1 = output_box
    mid_x = (x0 + x1) // 2
    mid_y = (y0 + y1) // 2
    anchors = {
        "left": (x0, mid_y),
        "top": (mid_x, y0),
        "right": (x1, mid_y),
        "bottom": (mid_x, y1),
    }
    boxes = {}
    for edge, (ax, ay) in anchors.items():
        sx = max(0, min(1920 - 256, ax - 128))
        sy = max(0, min(1080 - 256, ay - 128))
        boxes[edge] = [sx, sy, sx + 256, sy + 256]
    return boxes


def render_pair_row(baseline_image, adapter_image, seed):
    """Render a 1280x384 side-by-side overview row without cropping either 16:9 image."""
    row = Image.new("RGB", (1280, 384), "black")
    left_fit = ImageOps.contain(baseline_image, (640, 360), Image.Resampling.LANCZOS)
    right_fit = ImageOps.contain(adapter_image, (640, 360), Image.Resampling.LANCZOS)
    assert left_fit.size == (640, 360) and right_fit.size == (640, 360)
    row.paste(left_fit, (0, 24))
    row.paste(right_fit, (640, 24))
    draw = ImageDraw.Draw(row)
    draw.text((8, 4), f"{BASELINE_RECIPE} seed={seed}", fill="white")
    draw.text((648, 4), f"{ADAPTER_RECIPE} seed={seed}", fill="white")
    return row


def update_case_overviews(case, seed):
    """Save per-seed and per-case baseline vs adapter overviews when counterpart exists."""
    pair_outputs = case["outputs"].get(seed, {})
    if BASELINE_RECIPE not in pair_outputs or ADAPTER_RECIPE not in pair_outputs:
        return
    row = render_pair_row(
        pair_outputs[BASELINE_RECIPE], pair_outputs[ADAPTER_RECIPE], seed
    )
    row.save(case["dir"] / f"overview-seed-{seed}.png")
    case["pair_rows"][seed] = row

    completed_seeds = [s for s in case["seeds"] if s in case["pair_rows"]]
    sheet = Image.new("RGB", (1280, 384 * len(completed_seeds)), "black")
    for idx, done_seed in enumerate(completed_seeds):
        sheet.paste(case["pair_rows"][done_seed], (0, idx * 384))
    sheet.save(case["dir"] / "overview.png")
    sheet.save(case["dir"] / "comparison.png")


def load_base_pipeline(
    gate, model_name=DEFAULT_MODEL, base_info=None, sampler=DEFAULT_SAMPLER
):
    """Load selected local SD1.5 inpainting pipeline with allowlisted scheduler and no safety_checker."""
    if base_info is None:
        base_info = verify_base_model_preflight(
            gate.LAB, model_name, sampler=sampler
        )
    elif base_info.get("selected_sampler", DEFAULT_SAMPLER) != sampler:
        raise ValueError(
            f"base_info sampler {base_info.get('selected_sampler')} does not match requested sampler {sampler}"
        )
    pipe = gate.StableDiffusionInpaintPipeline.from_pretrained(
        base_info["model_dir"],
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        torch_dtype=gate.torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    )
    scheduler_class = base_info["scheduler_class"]
    if scheduler_class == "DDIMScheduler":
        scheduler_factory = gate.diffusers.DDIMScheduler
    elif scheduler_class == "DEISMultistepScheduler":
        scheduler_factory = gate.diffusers.DEISMultistepScheduler
    else:
        raise ValueError(f"Unsupported scheduler class: {scheduler_class}")
    pipe.scheduler = scheduler_factory.from_config(
        base_info["scheduler_config"]
    )
    assert pipe.safety_checker is None
    assert pipe.unet.config.in_channels == base_info["in_channels"]
    return pipe


def release_pipeline(pipe, gate):
    """Free accelerate offload hooks, pipeline references, Python GC, and CUDA cache."""
    if pipe is not None:
        if hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
        if hasattr(pipe, "remove_all_hooks"):
            pipe.remove_all_hooks()
    del pipe
    gc.collect()
    if gate.torch.cuda.is_available():
        gate.torch.cuda.empty_cache()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_name", help="Fresh output directory name under .local-artifacts/outpainting"
    )
    parser.add_argument(
        "--model",
        choices=tuple(BASE_MODELS),
        default=DEFAULT_MODEL,
        help="Registered local base model checkpoint (default: sd15)",
    )
    parser.add_argument(
        "--sampler",
        choices=tuple(SAMPLERS),
        default=DEFAULT_SAMPLER,
        help="Registered scheduler sampler (default: ddim)",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default=DEFAULT_DATASET,
        help="Anime fixture dataset to run (default: development)",
    )
    parser.add_argument(
        "--recipes",
        choices=("all", "baseline"),
        default="all",
        help="Recipes to run: all or baseline only (default: all)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all anime outputs across dataset groups, 4 cut directions, 2 seeds, and selected recipes",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HOME"] = str(LAB / "cache")

    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))

    import placement_poc as placement
    import quality_gate as gate
    from region_gpu import NEGATIVE, SOFT

    with gate.single_instance():
        base_info = verify_base_model_preflight(
            gate.LAB, args.model, sampler=args.sampler
        )
        selected_recipes = (
            (BASELINE_RECIPE,) if args.recipes == "baseline" else RECIPES
        )
        adapter_root = gate.LAB / "ip-adapter-sd15"
        adapter_info = (
            verify_adapter_preflight(adapter_root)
            if ADAPTER_RECIPE in selected_recipes
            else {
                "revision": None,
                "hashes": None,
                "config_git_blob_sha1": None,
            }
        )
        if args.dataset in EVALUATION_DATASETS:
            all_sources, evaluation_manifest_sha256 = evaluation_sources(
                gate.LAB, dataset=args.dataset
            )
        else:
            all_sources = {
                "closeup": gate.GROUPS["anime"]["path"],
                "winter": gate.ROOT / ".local-artifacts/anime-focus/winter/source.png",
            }
            evaluation_manifest_sha256 = None
        folder = new_output(gate.LAB, args.run_name)
        gpu_name = (
            gate.torch.cuda.get_device_name(0)
            if gate.torch.cuda.is_available()
            else "CPU"
        )

        group_order = tuple(all_sources)
        selected_groups = group_order if args.full else (group_order[-1],)
        selected_directions = (
            ("left", "top", "right", "bottom") if args.full else ("bottom",)
        )
        selected_seeds = (42, 123) if args.full else (42,)
        total_expected = (
            len(selected_groups)
            * len(selected_directions)
            * len(selected_seeds)
            * len(selected_recipes)
        )

        cases = []
        for group in selected_groups:
            source_path = all_sources[group]
            full_sha256 = sha256_file(source_path)
            with Image.open(source_path) as raw_image:
                full = ImageOps.exif_transpose(raw_image).convert("RGB")
            for direction in selected_directions:
                crop_box, margins = direction_crop_and_margins(full.size, direction)
                source = full.crop(crop_box)
                assert source.mode == "RGB"
                source_bytes = source.tobytes()
                source_rgb_sha256 = hashlib.sha256(source_bytes).hexdigest()

                canvas, mask, box = placement.prepare(source, margins)
                ref_square = reference_image(source)
                output_box = tuple(round(v * 2.5) for v in box)
                seams = seam_crop_boxes(output_box)

                case_dir = folder / f"{group}-{direction}"
                case_dir.mkdir(parents=True, exist_ok=False)
                source.save(case_dir / "source.png")
                canvas.save(case_dir / "canvas.png")
                mask.save(case_dir / "mask.png")
                ref_square.save(case_dir / "reference-square.png")
                full.save(case_dir / "full-reference-evaluation-only.png")

                cases.append(
                    {
                        "group": group,
                        "direction": direction,
                        "name": f"{group}-{direction}",
                        "dir": case_dir,
                        "path": source_path,
                        "full_sha256": full_sha256,
                        "full_size": full.size,
                        "source": source,
                        "source_bytes": source_bytes,
                        "source_rgb_sha256": source_rgb_sha256,
                        "crop_box": crop_box,
                        "margins": margins,
                        "canvas": canvas,
                        "canvas_bytes": canvas.tobytes(),
                        "mask": mask,
                        "mask_bytes": mask.tobytes(),
                        "box": box,
                        "output_box": output_box,
                        "seam_boxes": seams,
                        "ref_square": ref_square,
                        "ref_bytes": ref_square.tobytes(),
                        "ref_sha256": hashlib.sha256(ref_square.tobytes()).hexdigest(),
                        "seeds": selected_seeds,
                        "outputs": {},
                        "pair_rows": {},
                        "records": [],
                    }
                )

        records = []
        for recipe in selected_recipes:
            pipe = None
            try:
                pipe = load_base_pipeline(
                    gate,
                    args.model,
                    base_info=base_info,
                    sampler=args.sampler,
                )
                if recipe == ADAPTER_RECIPE:
                    pipe.load_ip_adapter(
                        str(adapter_root),
                        subfolder=ADAPTER_SUBFOLDER,
                        weight_name=ADAPTER_WEIGHT_NAME,
                        image_encoder_folder=IMAGE_ENCODER_FOLDER,
                        local_files_only=True,
                    )
                    pipe.set_ip_adapter_scale(ADAPTER_SCALE)
                assert pipe.safety_checker is None
                pipe.enable_model_cpu_offload()
                pipe.enable_vae_slicing()

                for case in cases:
                    for seed in selected_seeds:
                        assert case["source"].tobytes() == case["source_bytes"]
                        assert case["canvas"].tobytes() == case["canvas_bytes"]
                        assert case["mask"].tobytes() == case["mask_bytes"]
                        assert case["ref_square"].tobytes() == case["ref_bytes"]

                        call_kwargs = {
                            "prompt": SOFT,
                            "negative_prompt": NEGATIVE,
                            "image": case["canvas"],
                            "mask_image": case["mask"],
                            "width": 768,
                            "height": 432,
                            "num_inference_steps": 30,
                            "guidance_scale": 7.0,
                            "generator": gate.torch.Generator(device="cpu").manual_seed(
                                seed
                            ),
                        }
                        if recipe == ADAPTER_RECIPE:
                            cond_ref = reference_image(case["source"])
                            assert cond_ref.tobytes() == case["ref_bytes"]
                            call_kwargs["ip_adapter_image"] = cond_ref

                        if gate.torch.cuda.is_available():
                            gate.torch.cuda.reset_peak_memory_stats()
                            gate.torch.cuda.synchronize()
                        started = perf_counter()
                        result = pipe(**call_kwargs)
                        if gate.torch.cuda.is_available():
                            gate.torch.cuda.synchronize()
                        seconds = perf_counter() - started

                        assert pipe.safety_checker is None
                        assert result.nsfw_content_detected is None
                        generated = result.images[0]
                        assert generated.size == (768, 432)

                        composited = Image.composite(
                            generated, case["canvas"], case["mask"]
                        )
                        assert (
                            ImageChops.difference(
                                composited.crop(case["box"]),
                                case["canvas"].crop(case["box"]),
                            ).getbbox()
                            is None
                        )
                        output = placement.restore(
                            case["source"], composited, case["box"]
                        )

                        stem = f"{recipe}-{seed}"
                        generated.save(case["dir"] / f"{stem}-generated.png")
                        output_path = case["dir"] / f"{stem}.png"
                        output.save(output_path)

                        with Image.open(output_path) as decoded:
                            decoded.load()
                            assert decoded.format == "PNG"
                            assert decoded.size == (1920, 1080)

                        obox = case["output_box"]
                        fitted = case["source"].resize(
                            (obox[2] - obox[0], obox[3] - obox[1]),
                            Image.Resampling.LANCZOS,
                        )
                        assert (
                            ImageChops.difference(
                                output.crop(obox), fitted
                            ).getbbox()
                            is None
                        )
                        assert sha256_file(case["path"]) == case["full_sha256"]
                        assert (
                            hashlib.sha256(case["source"].tobytes()).hexdigest()
                            == case["source_rgb_sha256"]
                        )

                        for edge, sbox in case["seam_boxes"].items():
                            output.crop(tuple(sbox)).save(
                                case["dir"] / f"{stem}-seam100-{edge}.png"
                            )

                        case["outputs"].setdefault(seed, {})[recipe] = output
                        update_case_overviews(case, seed)

                        peak_alloc_mib = (
                            gate.torch.cuda.max_memory_allocated() / 1048576
                            if gate.torch.cuda.is_available()
                            else 0.0
                        )
                        peak_reserved_mib = (
                            gate.torch.cuda.max_memory_reserved() / 1048576
                            if gate.torch.cuda.is_available()
                            else 0.0
                        )

                        record = {
                            "dataset": args.dataset,
                            "recipes_selected": args.recipes,
                            "evaluation_manifest_sha256": evaluation_manifest_sha256,
                            "group": case["group"],
                            "direction": case["direction"],
                            "case": case["name"],
                            "seed": seed,
                            "recipe": recipe,
                            "ip_adapter_scale": (
                                ADAPTER_SCALE if recipe == ADAPTER_RECIPE else None
                            ),
                            "source_full_sha256": case["full_sha256"],
                            "source_sha256": case["full_sha256"],
                            "cropped_rgb_sha256": case["source_rgb_sha256"],
                            "source_full_size": list(case["full_size"]),
                            "cropped_source_size": list(case["source"].size),
                            "crop_box": list(case["crop_box"]),
                            "margins": list(case["margins"]),
                            "box": list(case["box"]),
                            "output_box": list(case["output_box"]),
                            "seam_crop_boxes": case["seam_boxes"],
                            "prompt": SOFT,
                            "negative_prompt": NEGATIVE,
                            "cfg": 7.0,
                            "steps": 30,
                            "reference_padding_rgb": list(PADDING_RGB),
                            "reference_square_size": list(case["ref_square"].size),
                            "reference_square_sha256": case["ref_sha256"],
                            "evaluation_only_full_reference": "full-reference-evaluation-only.png",
                            "base_model_name": base_info["model_name"],
                            "base_model_repository": base_info["repository"],
                            "base_model_revision": base_info["revision"],
                            "model_revision": base_info["revision"],
                            "base_unet_in_channels": base_info["in_channels"],
                            "base_inpaint_mode": base_info["inpaint_mode"],
                            "base_weights_hashes": base_info["hashes"],
                            "selected_sampler": base_info["selected_sampler"],
                            "scheduler_class": base_info["scheduler_class"],
                            "scheduler_source_class": base_info[
                                "scheduler_source_class"
                            ],
                            "scheduler_config_path": base_info[
                                "scheduler_config_path"
                            ],
                            "scheduler_config_sha256": base_info[
                                "scheduler_config_sha256"
                            ],
                            "adapter_revision": adapter_info["revision"],
                            "adapter_hashes": adapter_info["hashes"],
                            "adapter_config_git_blob_sha1": adapter_info[
                                "config_git_blob_sha1"
                            ],
                            "torch_version": gate.torch.__version__,
                            "diffusers_version": gate.diffusers.__version__,
                            "gpu_name": gpu_name,
                            "seconds": seconds,
                            "torch_allocated_peak_mib": peak_alloc_mib,
                            "torch_reserved_peak_mib": peak_reserved_mib,
                            "foreground_exact": True,
                            "safety_checker_none": True,
                            "nsfw_content_detected": None,
                        }
                        case["records"].append(record)
                        records.append(record)
                        (case["dir"] / "records.json").write_text(
                            json.dumps(case["records"], indent=2), encoding="utf-8"
                        )
                        (folder / "records.json").write_text(
                            json.dumps(records, indent=2), encoding="utf-8"
                        )
                        print(
                            f"{len(records)}/{total_expected} [{args.model}/{args.sampler}/{args.dataset}] {case['name']} {stem} ({seconds:.2f}s)",
                            flush=True,
                        )
            finally:
                release_pipeline(pipe, gate)
                pipe = None
                gc.collect()
                if gate.torch.cuda.is_available():
                    gate.torch.cuda.empty_cache()

        print(
            f"Completed {len(records)} records for model={args.model} sampler={args.sampler} dataset={args.dataset} in {folder}",
            flush=True,
        )


if __name__ == "__main__":
    main()
