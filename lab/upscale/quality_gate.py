"""Quality gate comparing the stock anime NCNN model with a fine-tuned NCNN model.

Runs the exact realesrgan-ncnn-vulkan binary and flags used by app/engine.py across
clean, JPEG-compressed, and noisy+JPEG variants of held-out images, recording wall time,
peak GPU memory delta, tile fallback checks, and 4-column comparison crop sheets:
  Lanczos | old | old+preserve 50% | new

CLI: python lab/upscale/quality_gate.py RUN_NAME --images DIR [--images DIR ...]
       [--new-models DIR] [--new-name haups-anime-x4]
Output: .local-artifacts/upscale-train/gate/RUN_NAME/{records.json, sheet-*.jpg}
"""

import argparse
import hashlib
import io
import json
import random
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from time import perf_counter

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENGINE = ROOT / ".local-artifacts" / "engine-poc" / "engine" / "realesrgan-ncnn-vulkan.exe"
DEFAULT_OLD_MODELS = DEFAULT_ENGINE.parent / "models"
DEFAULT_OLD_NAME = "realesrgan-x4plus-anime"
DEFAULT_NEW_MODELS = (
    ROOT / ".local-artifacts" / "upscale-train" / "runs" / "sr-fid-v1-interp50-20261116" / "models"
)
DEFAULT_NEW_NAME = "haups-anime-x4"
SCALE = 4
CROP = 96
TILE = 128
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def list_images(directory):
    """Non-recursive listing of image files, skipping any path with a 'remove' component."""
    if isinstance(directory, (list, tuple)):
        return [p for d in directory for p in list_images(d)]
    folder = Path(directory)
    if "remove" in (part.lower() for part in folder.parts):
        return []
    paths = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if "remove" in (part.lower() for part in path.parts):
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
    return paths


def _jpeg_roundtrip(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def make_variants(image, seed):
    """Return {'clean', 'jpeg', 'noise'} RGB variants deterministically for a given seed."""
    clean = image.convert("RGB")
    jpeg = _jpeg_roundtrip(clean, quality=55)
    try:
        import numpy as np

        rng = np.random.default_rng(seed)
        array = np.asarray(clean, dtype=np.float32)
        noisy = np.clip(array + rng.normal(0.0, 6.0, size=array.shape), 0.0, 255.0).round().astype(np.uint8)
        noisy_image = Image.fromarray(noisy, "RGB")
    except ImportError:
        rng = random.Random(seed)
        noisy_image = Image.new("RGB", clean.size)
        noisy_image.putdata(
            [
                (
                    min(255, max(0, round(r + rng.gauss(0.0, 6.0)))),
                    min(255, max(0, round(g + rng.gauss(0.0, 6.0)))),
                    min(255, max(0, round(b + rng.gauss(0.0, 6.0)))),
                )
                for r, g, b in clean.getdata()
            ]
        )
    noise = _jpeg_roundtrip(noisy_image, quality=80)
    return {"clean": clean, "jpeg": jpeg, "noise": noise}


def detail_boxes(image, crop=CROP, count=2):
    """Choose the highest-detail far-apart LR crop boxes."""
    upscale_dir = str(Path(__file__).resolve().parent)
    if upscale_dir not in sys.path:
        sys.path.insert(0, upscale_dir)
    import eval_sr  # noqa: E402

    if crop == eval_sr.CROP:
        return eval_sr.detail_crops(image, count=count)
    cells = []
    for top in range(0, max(1, image.height - crop + 1), crop // 2):
        for left in range(0, max(1, image.width - crop + 1), crop // 2):
            box = (left, top, left + crop, top + crop)
            cells.append((eval_sr.ts.detail_score(image.crop(box)), box))
    cells.sort(reverse=True)
    chosen = []
    for _, box in cells:
        if all(abs(box[0] - c[0]) + abs(box[1] - c[1]) >= crop * 2 for c in chosen):
            chosen.append(box)
        if len(chosen) == count:
            break
    return chosen


def _query_gpu_memory_mb():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                return float(line)
            except ValueError:
                return None
    return None


class _VramMonitor:
    """Poll nvidia-smi memory.used every 0.2 s in a background thread and report peak minus baseline."""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.baseline = _query_gpu_memory_mb()
        self.samples = [self.baseline] if self.baseline is not None else []
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if self.baseline is not None:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def _poll(self):
        while not self._stop.is_set():
            reading = _query_gpu_memory_mb()
            if reading is not None:
                self.samples.append(reading)
            self._stop.wait(self.interval)

    def __exit__(self, exc_type, exc, tb):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2)
            reading = _query_gpu_memory_mb()
            if reading is not None:
                self.samples.append(reading)

    @property
    def peak_delta_mb(self):
        if self.baseline is None or not self.samples:
            return None
        return round(max(0.0, max(self.samples) - self.baseline), 2)


_GPU = []


def gpu_id():
    """Same NVIDIA Vulkan device the app picks (vulkaninfo / HAUPS_GPU_ID); index 0 can be the iGPU."""
    if not _GPU:
        sys.path.insert(0, str(ROOT))
        from app.engine import EngineAdapter

        _GPU.append(EngineAdapter._gpu())
    return _GPU[0]


def run_engine(engine, src_path, out_path, model_dir, model_name, expected_size, index, variant, tile=TILE):
    """Run realesrgan-ncnn-vulkan with app flags and validate output dimensions."""
    out_path.unlink(missing_ok=True)
    command = [
        str(engine),
        "-i",
        str(src_path),
        "-o",
        str(out_path),
        "-m",
        str(model_dir),
        "-n",
        str(model_name),
        "-s",
        str(SCALE),
        "-g",
        str(gpu_id()),
        "-t",
        str(tile),
        "-j",
        "1:1:1",
        "-f",
        "png",
    ]
    with _VramMonitor() as monitor:
        start = perf_counter()
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        elapsed = perf_counter() - start
    stderr_tail = (result.stderr or "")[-400:]
    if result.returncode != 0 or not out_path.is_file():
        raise SystemExit(
            f"engine failed on index={index} variant={variant} model={model_name} tile={tile}: "
            f"rc={result.returncode} {stderr_tail}"
        )
    with Image.open(out_path) as generated:
        output = generated.convert("RGB")
    if output.size != expected_size:
        raise SystemExit(
            f"size mismatch on index={index} variant={variant} model={model_name} tile={tile}: "
            f"got {output.size}, expected {expected_size} {stderr_tail}"
        )
    return output, round(elapsed, 3), monitor.peak_delta_mb


def _rel_to_root(path):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def _mean_metrics(entries, key):
    times = [e[key]["time_s"] for e in entries]
    vrams = [e[key]["peak_vram_mb"] for e in entries if e[key]["peak_vram_mb"] is not None]
    return {
        "time_s": round(sum(times) / len(times), 3) if times else None,
        "peak_vram_mb": round(sum(vrams) / len(vrams), 2) if vrams else None,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    parser.add_argument("--images", action="append", required=True)
    parser.add_argument("--new-models", type=Path, default=DEFAULT_NEW_MODELS)
    parser.add_argument("--new-name", default=DEFAULT_NEW_NAME)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = ROOT / ".local-artifacts" / "upscale-train" / "gate" / args.run_name
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir()  # fail if exists, never overwrite evidence

    tmp_dir = ROOT / ".local-artifacts" / "tmp" / f"gate-{args.run_name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    paths = [p for d in args.images for p in list_images(d)]
    if not paths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise SystemExit("no images found in --images directories")

    records = []
    tile_check = {}
    side = CROP * SCALE
    try:
        for index, path in enumerate(paths, 1):
            with Image.open(path) as src_img:
                base_image = src_img.convert("RGB")
            variants = make_variants(base_image, seed=index)
            for variant_name, variant_img in variants.items():
                src_png = tmp_dir / "source.png"
                old_png = tmp_dir / "old.png"
                new_png = tmp_dir / "new.png"
                variant_img.save(src_png, format="PNG")
                expected_size = (variant_img.width * SCALE, variant_img.height * SCALE)

                old_out, old_time, old_vram = run_engine(
                    DEFAULT_ENGINE,
                    src_png,
                    old_png,
                    DEFAULT_OLD_MODELS,
                    DEFAULT_OLD_NAME,
                    expected_size,
                    index,
                    variant_name,
                    tile=TILE,
                )
                new_out, new_time, new_vram = run_engine(
                    DEFAULT_ENGINE,
                    src_png,
                    new_png,
                    args.new_models,
                    args.new_name,
                    expected_size,
                    index,
                    variant_name,
                    tile=TILE,
                )

                if index == 1 and variant_name == "clean":
                    for fallback_tile in (64, 32):
                        tile_png = tmp_dir / f"tile-{fallback_tile}.png"
                        _, tile_time, tile_vram = run_engine(
                            DEFAULT_ENGINE,
                            src_png,
                            tile_png,
                            args.new_models,
                            args.new_name,
                            expected_size,
                            index,
                            variant_name,
                            tile=fallback_tile,
                        )
                        tile_check[str(fallback_tile)] = {
                            "ok": True,
                            "time_s": tile_time,
                            "peak_vram_mb": tile_vram,
                        }
                        tile_png.unlink(missing_ok=True)

                lanczos_x4 = variant_img.resize(expected_size, Image.Resampling.LANCZOS)
                old_preserve = Image.blend(old_out, lanczos_x4, 0.5)
                boxes = detail_boxes(variant_img, crop=CROP, count=2)
                sheet = Image.new(
                    "RGB",
                    (side * 4 + 12, side * len(boxes) + 4 * (len(boxes) - 1)),
                    "white",
                )
                for row, box in enumerate(boxes):
                    big = tuple(v * SCALE for v in box)
                    for column, picture in enumerate((lanczos_x4, old_out, old_preserve, new_out)):
                        sheet.paste(picture.crop(big), (column * (side + 4), row * (side + 4)))
                sheet.save(out_dir / f"sheet-{index:02d}-{variant_name}.jpg", quality=92)

                entry = {
                    "index": index,
                    "variant": variant_name,
                    "size": [variant_img.width, variant_img.height],
                    "old": {"time_s": old_time, "peak_vram_mb": old_vram},
                    "new": {"time_s": new_time, "peak_vram_mb": new_vram},
                }
                records.append(entry)
                print(f"index={index:02d} variant={variant_name} old={old_time}s new={new_time}s", flush=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    means = {}
    for variant_name in ("clean", "jpeg", "noise"):
        subset = [r for r in records if r["variant"] == variant_name]
        if subset:
            means[variant_name] = {
                "old": _mean_metrics(subset, "old"),
                "new": _mean_metrics(subset, "new"),
            }

    sha256 = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(args.new_models).glob(f"{args.new_name}.*"))
        if p.suffix in (".param", ".bin")
    }
    summary = {
        "engine": _rel_to_root(DEFAULT_ENGINE),
        "old_models": _rel_to_root(DEFAULT_OLD_MODELS),
        "old_name": DEFAULT_OLD_NAME,
        "new_models": _rel_to_root(args.new_models),
        "new_name": args.new_name,
        "sha256": sha256,
        "columns": "Lanczos | old | old+preserve 50% | new",
        "tile_check": tile_check,
        "mean": means,
        "records": records,
    }
    (out_dir / "records.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
