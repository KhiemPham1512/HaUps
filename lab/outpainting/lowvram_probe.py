"""Measure GPU VRAM, wall time, and process RAM for upscale tiles and low-VRAM wallpaper outpainting.

CLI:
  python lab/outpainting/lowvram_probe.py upscale RUN_NAME --images DIR --cases 1 2 3 --tiles 128 64 32 16
  python lab/outpainting/lowvram_probe.py outpaint RUN_NAME --images DIR --case N --mode normal|sequential
  python lab/outpainting/lowvram_probe.py compare RUN_NAME --case N
"""

import argparse
import ctypes
import json
import math
from pathlib import Path
import re
import sys
from time import perf_counter

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
_UPSCALE_DIR = ROOT / "lab" / "upscale"
if str(_UPSCALE_DIR) not in sys.path:
    sys.path.insert(0, str(_UPSCALE_DIR))

from app import engine as app_engine  # noqa: E402
from app import outpaint_runner  # noqa: E402
import quality_gate  # noqa: E402
import seam_blend_lab  # noqa: E402

LAB = ROOT / ".local-artifacts" / "outpainting"
RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
BG_SIZE = (768, 432)


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def validate_run_name(run_name):
    if not isinstance(run_name, str) or not RUN_NAME_RE.fullmatch(run_name):
        raise ValueError(f"Invalid run name: {run_name!r}")
    return run_name


def peak_ram_mb():
    """Return peak process working set in MB via Windows GetProcessMemoryInfo, or None."""
    try:
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return None
        kernel32 = getattr(windll, "kernel32", None)
        if kernel32 is None:
            return None
        get_current_process = getattr(kernel32, "GetCurrentProcess", None)
        if get_current_process is None:
            return None
        get_current_process.argtypes = []
        get_current_process.restype = ctypes.c_void_p

        func = getattr(kernel32, "K32GetProcessMemoryInfo", None)
        if func is None:
            psapi = getattr(windll, "psapi", None)
            func = getattr(psapi, "GetProcessMemoryInfo", None) if psapi is not None else None
        if func is None:
            return None

        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        func.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
            ctypes.c_ulong,
        ]
        func.restype = ctypes.c_int
        ok = func(get_current_process(), ctypes.byref(counters), counters.cb)
        if not ok:
            return None
        return round(counters.PeakWorkingSetSize / (1024 * 1024), 2)
    except Exception:
        return None


def _swap_on_call(module, device):
    module.to("cpu")
    orig = module.forward

    def _forward(*a, **kw):
        module.to(device)
        try:
            return orig(*a, **kw)
        finally:
            module.to("cpu")
            try:
                import torch
            except ImportError:
                torch = None
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    module.forward = _forward
    return module


def wrap_powerpaint_sequential(original):
    def wrapper(gate, model_info=None, **kw):
        kw["offload"] = False
        pipe = original(gate, model_info, **kw)
        pipe._exclude_from_cpu_offload = list(
            getattr(pipe, "_exclude_from_cpu_offload", [])
        ) + ["text_encoder", "text_encoder_brushnet"]
        pipe.enable_sequential_cpu_offload()
        for name in ("text_encoder", "text_encoder_brushnet"):
            module = getattr(pipe, name, None)
            if module is not None:
                _swap_on_call(module, "cuda")
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        return pipe

    return wrapper


def wrap_sdxl_sequential(original):
    def wrapper(*a, **kw):
        pipe = original(*a, **kw)
        if hasattr(pipe, "remove_all_hooks"):
            pipe.remove_all_hooks()
        pipe.enable_sequential_cpu_offload()
        return pipe

    return wrapper


def run_upscale(args):
    out = LAB / args.run_name
    out.mkdir()

    image_paths = seam_blend_lab.list_images(args.images)
    engine = app_engine.DEFAULT_ENGINE / "realesrgan-ncnn-vulkan.exe"
    model_dir = app_engine.DEFAULT_ENGINE / "models"
    model_name = app_engine.MODELS["anime"]

    records = []
    for idx in args.cases:
        if idx < 1 or idx > len(image_paths):
            raise SystemExit(f"case index {idx} out of range (1..{len(image_paths)})")
        case_id = f"case-{idx:02d}"
        src_path = out / f"{case_id}-src.png"
        bg_path = out / f"{case_id}-bg768.png"
        try:
            with Image.open(image_paths[idx - 1]) as raw:
                src_img = ImageOps.exif_transpose(raw).convert("RGB")
            src_img.save(src_path, format="PNG")
            bg_img = src_img.resize(BG_SIZE, Image.Resampling.LANCZOS)
            bg_img.save(bg_path, format="PNG")
            inputs = [
                ("source", src_path, src_img.size),
                ("bg768", bg_path, bg_img.size),
            ]
            del src_img, bg_img

            for input_kind, in_path, (w, h) in inputs:
                expected_size = (w * 4, h * 4)
                for tile in args.tiles:
                    out_path = out / f"{case_id}-{input_kind}-t{tile}.png"
                    seconds_list = []
                    vram_list = []
                    error_msg = None
                    for _ in range(2):
                        try:
                            _, secs, peak_mb = quality_gate.run_engine(
                                engine,
                                in_path,
                                out_path,
                                model_dir,
                                model_name,
                                expected_size,
                                idx,
                                input_kind,
                                tile=tile,
                            )
                            seconds_list.append(secs)
                            vram_list.append(peak_mb)
                        except SystemExit as exc:
                            error_msg = str(exc)
                            break
                        finally:
                            out_path.unlink(missing_ok=True)
                    entry = {
                        "case": case_id,
                        "input": input_kind,
                        "size": [w, h],
                        "tile": tile,
                        "seconds": seconds_list,
                        "peak_vram_mb": vram_list,
                    }
                    if error_msg is not None:
                        entry["error"] = error_msg
                    records.append(entry)
        finally:
            src_path.unlink(missing_ok=True)
            bg_path.unlink(missing_ok=True)

    (out / "records.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return records


def run_outpaint(args):
    run_dir = LAB / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    case_id = f"case-{args.case:02d}"
    case_dir = run_dir / f"{args.mode}-{case_id}"
    case_dir.mkdir()

    outpaint_runner.configure_offline_env()
    image_paths = seam_blend_lab.list_images(args.images)
    if args.case < 1 or args.case > len(image_paths):
        raise SystemExit(f"case index {args.case} out of range (1..{len(image_paths)})")
    source = outpaint_runner.load_source(image_paths[args.case - 1])
    scene = outpaint_runner.auto_scene(source, "anime", "")

    import powerpaint_gpu as pp
    import sdxl_inpaint_gpu as sx
    import sdxl_repair_lab
    import torch

    orig_pp_load = pp.load_powerpaint_pipeline
    orig_sx_load = sx.load_pipeline
    orig_sdxl_repair = sdxl_repair_lab.repair_with_sdxl
    sdxl_ran = False

    def _tracked_repair(*a, **kw):
        nonlocal sdxl_ran
        sdxl_ran = True
        return orig_sdxl_repair(*a, **kw)

    sdxl_repair_lab.repair_with_sdxl = _tracked_repair
    if args.mode == "sequential":
        pp.load_powerpaint_pipeline = wrap_powerpaint_sequential(orig_pp_load)
        sx.load_pipeline = wrap_sdxl_sequential(orig_sx_load)

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with quality_gate._VramMonitor() as monitor:
            t0 = perf_counter()
            generated, _ = outpaint_runner.generate(source, "anime", "", scene=scene)
            seconds = round(perf_counter() - t0, 3)
    finally:
        pp.load_powerpaint_pipeline = orig_pp_load
        sx.load_pipeline = orig_sx_load
        sdxl_repair_lab.repair_with_sdxl = orig_sdxl_repair

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
    ram_mb = peak_ram_mb()

    if generated is None:
        record = {
            "case": case_id,
            "mode": args.mode,
            "skipped": True,
            "seconds": seconds,
            "gpu_peak_delta_mb": monitor.peak_delta_mb,
            "gpu_baseline_mb": monitor.baseline,
            "torch_max_allocated_mb": alloc_mb,
            "torch_max_reserved_mb": reserved_mb,
            "ram_peak_mb": ram_mb,
            "sdxl_ran": sdxl_ran,
            "generated_size": None,
        }
    else:
        generated.save(case_dir / "generated.png", format="PNG")
        record = {
            "case": case_id,
            "mode": args.mode,
            "seconds": seconds,
            "gpu_peak_delta_mb": monitor.peak_delta_mb,
            "gpu_baseline_mb": monitor.baseline,
            "torch_max_allocated_mb": alloc_mb,
            "torch_max_reserved_mb": reserved_mb,
            "ram_peak_mb": ram_mb,
            "sdxl_ran": sdxl_ran,
            "generated_size": list(generated.size),
        }

    (case_dir / "record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def compare_images(img_a, img_b):
    size_equal = img_a.size == img_b.size
    if not size_equal:
        return {
            "size_equal": False,
            "mean_abs_diff": None,
            "max_abs_diff": None,
            "psnr": None,
            "share_pixels_diff_gt_8": None,
        }
    try:
        import numpy as np

        arr_a = np.asarray(img_a.convert("RGB"), dtype=np.float64)
        arr_b = np.asarray(img_b.convert("RGB"), dtype=np.float64)
        abs_diff = np.abs(arr_a - arr_b)
        mean_abs_diff = float(np.mean(abs_diff))
        max_abs_diff = float(np.max(abs_diff))
        mse = float(np.mean((arr_a - arr_b) ** 2))
        share_gt_8 = float(np.mean(np.any(abs_diff > 8.0, axis=-1)))
    except ImportError:
        bytes_a = img_a.convert("RGB").tobytes()
        bytes_b = img_b.convert("RGB").tobytes()
        n_vals = len(bytes_a)
        n_pixels = n_vals // 3
        total_abs = 0
        total_sq = 0
        max_abs = 0
        gt_8_pixels = 0
        for i in range(0, n_vals, 3):
            d0 = abs(bytes_a[i] - bytes_b[i])
            d1 = abs(bytes_a[i + 1] - bytes_b[i + 1])
            d2 = abs(bytes_a[i + 2] - bytes_b[i + 2])
            total_abs += d0 + d1 + d2
            total_sq += d0 * d0 + d1 * d1 + d2 * d2
            pix_max = max(d0, d1, d2)
            if pix_max > max_abs:
                max_abs = pix_max
            if pix_max > 8:
                gt_8_pixels += 1
        mean_abs_diff = float(total_abs / n_vals) if n_vals else 0.0
        max_abs_diff = float(max_abs)
        mse = float(total_sq / n_vals) if n_vals else 0.0
        share_gt_8 = float(gt_8_pixels / n_pixels) if n_pixels else 0.0
    psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10((255.0**2) / mse))
    return {
        "size_equal": True,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
        "psnr": psnr,
        "share_pixels_diff_gt_8": share_gt_8,
    }


def run_compare(args):
    run_dir = LAB / args.run_name
    case_id = f"case-{args.case:02d}"
    normal_path = run_dir / f"normal-{case_id}" / "generated.png"
    seq_path = run_dir / f"sequential-{case_id}" / "generated.png"
    with Image.open(normal_path) as normal_raw:
        normal_img = normal_raw.convert("RGB")
    with Image.open(seq_path) as seq_raw:
        seq_img = seq_raw.convert("RGB")

    metrics = compare_images(normal_img, seq_img)
    result = {"case": case_id, **metrics}
    out_path = run_dir / f"compare-{case_id}.json"
    text = json.dumps(result, indent=2)
    out_path.write_text(text, encoding="utf-8")
    print(text, flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    up = subparsers.add_parser("upscale")
    up.add_argument("run_name")
    up.add_argument("--images", type=Path, required=True)
    up.add_argument("--cases", type=int, nargs="+", default=[1, 2, 3])
    up.add_argument("--tiles", type=int, nargs="+", default=[128, 64, 32, 16])

    op = subparsers.add_parser("outpaint")
    op.add_argument("run_name")
    op.add_argument("--images", type=Path, required=True)
    op.add_argument("--case", type=int, required=True)
    op.add_argument("--mode", choices=("normal", "sequential"), required=True)

    cmp_p = subparsers.add_parser("compare")
    cmp_p.add_argument("run_name")
    cmp_p.add_argument("--case", type=int, required=True)

    args = parser.parse_args(argv)
    try:
        validate_run_name(args.run_name)
    except ValueError as err:
        parser.error(str(err))
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "upscale":
        return run_upscale(args)
    if args.command == "outpaint":
        return run_outpaint(args)
    if args.command == "compare":
        return run_compare(args)
    raise ValueError(f"Unknown command: {args.command!r}")


if __name__ == "__main__":
    main()
