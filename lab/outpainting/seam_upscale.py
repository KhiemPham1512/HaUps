import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic

from PIL import Image, ImageChops

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.engine import EngineAdapter
from app.lock import single_instance

LAB = Path(__file__).resolve().parents[2] / ".local-artifacts" / "outpainting"
GENERATED_SIZE = (768, 432)
UPSCALED_SIZE = (3072, 1728)
FINAL_SIZE = (1920, 1080)
COMPARE_TILE_SIZE = (256, 256)
COMPARE_SIZE = (512, 256)
SIDES = ("left", "top", "right", "bottom")
METHOD_NAME = "esrgan-x4plus-anime-tile128+lanczos"


def _validate_dir_name(lab, name):
    if (
        not isinstance(name, str)
        or not name
        or name in (".", "..")
        or ".." in name
        or "/" in name
        or "\\" in name
    ):
        raise ValueError(f"Tên thư mục không hợp lệ: {name!r}")
    candidate = lab / name
    if candidate.is_symlink():
        raise ValueError(f"Không chấp nhận symlink: {name!r}")
    if candidate.resolve().parent != lab.resolve():
        raise ValueError(f"Đường dẫn thoát khỏi thư mục lab: {name!r}")
    return candidate


class InvalidOutputBoxError(ValueError, RuntimeError):
    pass


def _validate_output_box(output_box):
    if (
        not isinstance(output_box, (list, tuple))
        or len(output_box) != 4
        or any(isinstance(v, bool) or not isinstance(v, int) for v in output_box)
    ):
        raise InvalidOutputBoxError(
            f"output_box phải gồm 4 số nguyên [x0, y0, x1, y1]: {output_box!r}"
        )
    x0, y0, x1, y1 = (int(v) for v in output_box)
    if not (0 <= x0 < x1 <= FINAL_SIZE[0] and 0 <= y0 < y1 <= FINAL_SIZE[1]):
        raise InvalidOutputBoxError(
            f"output_box nằm ngoài khung {FINAL_SIZE[0]}x{FINAL_SIZE[1]} hoặc kích thước không hợp lệ: {output_box!r}"
        )
    return (x0, y0, x1, y1)


def compose(lanczos_final, upscaled_bg, output_box):
    box = _validate_output_box(output_box)
    x0, y0, _, _ = box
    lanczos_rgb = lanczos_final.convert("RGB")
    if lanczos_rgb.size != FINAL_SIZE:
        raise InvalidOutputBoxError(
            f"Ảnh lanczos_final phải có kích thước {FINAL_SIZE[0]}x{FINAL_SIZE[1]}."
        )
    bg = upscaled_bg.convert("RGB").resize(FINAL_SIZE, Image.Resampling.LANCZOS)
    region = lanczos_rgb.crop(box)
    bg.paste(region, (x0, y0))
    if ImageChops.difference(bg.crop(box), region).getbbox() is not None:
        raise RuntimeError("Vùng nhân vật không khớp pixel tuyệt đối sau khi ghép.")
    return bg


def _build_seam_compare(lanczos_img, esrgan_img, box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise RuntimeError("seam_crop_boxes phải gồm [x0, y0, x1, y1].")
    crop_box = tuple(int(v) for v in box)
    left_crop = lanczos_img.crop(crop_box)
    right_crop = esrgan_img.crop(crop_box)
    if left_crop.size != COMPARE_TILE_SIZE:
        left_crop = left_crop.resize(COMPARE_TILE_SIZE, Image.Resampling.LANCZOS)
    if right_crop.size != COMPARE_TILE_SIZE:
        right_crop = right_crop.resize(COMPARE_TILE_SIZE, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", COMPARE_SIZE)
    canvas.paste(left_crop.convert("RGB"), (0, 0))
    canvas.paste(right_crop.convert("RGB"), (COMPARE_TILE_SIZE[0], 0))
    return canvas


def _load_case_inputs(run_dir, seed, prefix):
    records_path = run_dir / "records.json"
    if not records_path.is_file():
        raise FileNotFoundError("Thiếu file records.json trong RUN.")
    try:
        raw_records = json.loads(records_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError("Không đọc được records.json của RUN.") from error
    if not isinstance(raw_records, list):
        raise RuntimeError("records.json phải là danh sách record.")

    record_map = {}
    for item in raw_records:
        if isinstance(item, dict) and "case" in item and "seed" in item:
            record_map[(str(item["case"]), int(item["seed"]))] = item

    case_dirs = sorted(
        path
        for path in run_dir.iterdir()
        if path.is_dir() and not path.is_symlink() and path.name.startswith("eval-")
    )
    if not case_dirs:
        raise RuntimeError("Không tìm thấy thư mục ca eval-* trong RUN.")

    validated = []
    for case_dir in case_dirs:
        case_name = case_dir.name
        generated_path = case_dir / f"{prefix}-{seed}-generated.png"
        lanczos_path = case_dir / f"{prefix}-{seed}.png"
        for required in (generated_path, lanczos_path):
            if not required.is_file():
                raise FileNotFoundError(
                    f"Thiếu file bắt buộc {required.name} trong {case_name}."
                )
        record = record_map.get((case_name, int(seed)))
        if record is None:
            raise RuntimeError(
                f"Không tìm thấy record trong records.json cho {case_name} (seed={seed})."
            )
        if "output_box" not in record or record["output_box"] is None:
            raise RuntimeError(
                f"Thiếu output_box trong records.json cho {case_name} (seed={seed})."
            )
        output_box = _validate_output_box(record["output_box"])
        boxes = record.get("seam_crop_boxes")
        if not isinstance(boxes, dict) or any(side not in boxes for side in SIDES):
            raise RuntimeError(
                f"Thiếu seam_crop_boxes hợp lệ cho {case_name} (seed={seed})."
            )
        validated.append((case_name, generated_path, lanczos_path, output_box, boxes))
    return validated


def run(
    lab,
    run_name,
    out_name,
    seed=42,
    prefix="powerpaint-v2",
    runner=subprocess.run,
    adapter=None,
    lock=single_instance,
):
    lab_path = Path(lab).resolve()
    if not lab_path.is_dir() or Path(lab).is_symlink():
        raise ValueError("Thư mục lab không hợp lệ.")
    run_dir = _validate_dir_name(lab_path, run_name)
    out_dir = _validate_dir_name(lab_path, out_name)

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Thư mục RUN không tồn tại: {run_name}")
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"Thư mục OUT đã tồn tại: {out_name}")

    cases = _load_case_inputs(run_dir, seed, prefix)
    if adapter is None:
        adapter = EngineAdapter()

    out_dir.mkdir(parents=False, exist_ok=False)
    output_records = []

    with lock():
        for case_name, generated_path, lanczos_path, output_box, boxes in cases:
            with Image.open(generated_path) as gen_check:
                gen_check.load()
                if gen_check.size != GENERATED_SIZE:
                    raise RuntimeError(
                        f"Ảnh generated của {case_name} không đúng kích thước 768x432."
                    )
            with Image.open(lanczos_path) as lanczos_check:
                lanczos_check.load()
                if lanczos_check.size != FINAL_SIZE:
                    raise RuntimeError(
                        f"Ảnh Lanczos của {case_name} không đúng kích thước 1920x1080."
                    )

            case_out_dir = out_dir / case_name
            case_out_dir.mkdir(parents=False, exist_ok=False)
            tmp_png = case_out_dir / f".tmp-esrgan-{seed}.png"

            command = [
                str(Path(adapter.directory) / "realesrgan-ncnn-vulkan.exe"),
                "-i",
                str(generated_path),
                "-o",
                str(tmp_png),
                "-m",
                str(Path(adapter.directory) / "models"),
                "-n",
                "realesrgan-x4plus-anime",
                "-s",
                "4",
                "-g",
                str(adapter.gpu_id),
                "-t",
                "128",
                "-j",
                "1:1:1",
                "-f",
                "png",
            ]

            try:
                started = monotonic()
                result = runner(command, capture_output=True, timeout=120)
                seconds_engine = monotonic() - started
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Real-ESRGAN thất bại ở {case_name} (rc={result.returncode})."
                    )
                if not tmp_png.is_file():
                    raise RuntimeError(f"Real-ESRGAN không tạo file kết quả ở {case_name}.")
                try:
                    with Image.open(tmp_png) as upscaled_img:
                        upscaled_img.load()
                        if upscaled_img.size != UPSCALED_SIZE:
                            raise RuntimeError(
                                f"Ảnh kết quả engine ở {case_name} không đúng 3072x1728."
                            )
                        upscaled_copy = upscaled_img.copy()
                except Exception as error:
                    if isinstance(error, RuntimeError):
                        raise
                    raise RuntimeError(
                        f"Không mở được ảnh kết quả engine ở {case_name}."
                    ) from error
            finally:
                tmp_png.unlink(missing_ok=True)

            with Image.open(lanczos_path) as lanczos_img:
                lanczos_img.load()
                composed = compose(lanczos_img, upscaled_copy, output_box)
                composed.save(case_out_dir / f"esrgan-{seed}.png", format="PNG")
                for side in SIDES:
                    compare = _build_seam_compare(lanczos_img, composed, boxes[side])
                    compare.save(
                        case_out_dir / f"seam-compare-{side}.png", format="PNG"
                    )

            output_records.append(
                {
                    "case": case_name,
                    "seed": int(seed),
                    "seconds_engine": round(seconds_engine, 4),
                    "foreground_exact": True,
                    "method": METHOD_NAME,
                }
            )
            print(f"{case_name}: {seconds_engine:.2f}s")

    (out_dir / "records.json").write_text(
        json.dumps(output_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_records


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Upscale nền outpaint 768x432 bằng Real-ESRGAN x4 rồi giảm về 1920x1080."
    )
    parser.add_argument("run_name", metavar="RUN_NAME")
    parser.add_argument("out_name", metavar="OUT_NAME")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="powerpaint-v2")
    args = parser.parse_args(argv)
    run(LAB, args.run_name, args.out_name, seed=args.seed, prefix=args.prefix)


if __name__ == "__main__":
    main()
