"""Offline outpainting boundary and geometry; the app environment needs only Pillow."""

import os
import shutil
import subprocess

from PIL import Image, ImageChops, ImageDraw, ImageOps

from app.engine import ROOT, EngineError

LAB = ROOT / ".local-artifacts/outpainting"
PYTHON = ROOT / ".local-artifacts/quality-upgrade/.venv/Scripts/python.exe"
SIZE = (3840, 2160)
REQUIRED_FILES = (
    "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors",
    "PowerPaint_Brushnet/pytorch_model.bin",
    "realisticVisionV60B1_v51VAE/unet/diffusion_pytorch_model-002.safetensors",
    "realisticVisionV60B1_v51VAE/text_encoder/model.safetensors",
    "realisticVisionV60B1_v51VAE/vae/diffusion_pytorch_model.safetensors",
)

# SDXL fallback that repaints sides PowerPaint filled with extra characters (paths under LAB/sdxl)
SDXL_FILES = (
    "sdxl-inpaint-0.1/unet/diffusion_pytorch_model.fp16.safetensors",
    "sdxl-inpaint-0.1/unet/config.json",
    "animagine-xl-4.0/model_index.json",
    "animagine-xl-4.0/text_encoder/model.safetensors",
    "animagine-xl-4.0/text_encoder_2/model.safetensors",
    "sdxl-vae-fp16-fix/diffusion_pytorch_model.safetensors",
)


def is_available():
    powerpaint = LAB / "powerpaint-v2"
    weights = powerpaint / "weights"
    return (
        PYTHON.is_file()
        and (powerpaint / "vendor").is_dir()
        and (LAB / "anime-seg" / "isnetis.onnx").is_file()
        and all((weights / name).is_file() for name in REQUIRED_FILES)
        and all((LAB / "sdxl" / name).is_file() for name in SDXL_FILES)
    )


def prepare(source, size=(768, 432)):
    main = ImageOps.contain(source, size, Image.Resampling.LANCZOS)
    x, y = (size[0] - main.width) // 2, (size[1] - main.height) // 2
    canvas = Image.new("RGB", size, "white")
    canvas.paste(main, (x, y))
    mask = Image.new("L", size, 255)
    ImageDraw.Draw(mask).rectangle(
        (x, y, x + main.width - 1, y + main.height - 1), fill=0
    )
    return canvas, mask


def _box_edge_ramp(main_size, box_xy, width_px):
    mw, mh = main_size
    x, y = box_xy
    width_px = int(width_px)
    left_ramped = x > 0
    right_ramped = x + mw < SIZE[0]
    top_ramped = y > 0
    bottom_ramped = y + mh < SIZE[1]

    ax = [255] * mw
    if width_px > 0:
        if left_ramped:
            for i in range(min(width_px, mw)):
                ax[i] = min(ax[i], round(255 * i / width_px))
        if right_ramped:
            for d in range(min(width_px, mw)):
                i = mw - 1 - d
                ax[i] = min(ax[i], round(255 * d / width_px))

    ay = [255] * mh
    if width_px > 0:
        if top_ramped:
            for j in range(min(width_px, mh)):
                ay[j] = min(ay[j], round(255 * j / width_px))
        if bottom_ramped:
            for d in range(min(width_px, mh)):
                j = mh - 1 - d
                ay[j] = min(ay[j], round(255 * d / width_px))

    h_ramp = Image.frombytes("L", (mw, 1), bytes(ax)).resize(
        (mw, mh), Image.Resampling.NEAREST
    )
    v_ramp = Image.frombytes("L", (1, mh), bytes(ay)).resize(
        (mw, mh), Image.Resampling.NEAREST
    )
    ramp = Image.new("L", SIZE, 0)
    ramp.paste(ImageChops.darker(h_ramp, v_ramp), (x, y))
    return ramp


def restore(source, generated):
    main = ImageOps.contain(source, SIZE, Image.Resampling.LANCZOS)
    x, y = (SIZE[0] - main.width) // 2, (SIZE[1] - main.height) // 2
    output = generated.resize(SIZE, Image.Resampling.LANCZOS)
    output.paste(main, (x, y))
    if ImageChops.difference(
        output.crop((x, y, x + main.width, y + main.height)), main
    ).getbbox():
        raise EngineError("Không giữ được nội dung ảnh chính khi ghép nền.")
    return output


def restore_feathered(source, generated_native, width_px):
    width_px = int(width_px)
    if width_px <= 0:
        return restore(source, generated_native)
    main = ImageOps.contain(source, SIZE, Image.Resampling.LANCZOS)
    x, y = (SIZE[0] - main.width) // 2, (SIZE[1] - main.height) // 2
    background = generated_native.resize(SIZE, Image.Resampling.LANCZOS)
    mw, mh = main.size
    left_ramped = x > 0
    right_ramped = x + mw < SIZE[0]
    top_ramped = y > 0
    bottom_ramped = y + mh < SIZE[1]

    alpha = _box_edge_ramp(main.size, (x, y), width_px)
    foreground = Image.new("RGB", SIZE)
    foreground.paste(main, (x, y))
    output = Image.composite(foreground, background, alpha)

    ix0 = width_px if left_ramped else 0
    ix1 = mw - (width_px if right_ramped else 0)
    iy0 = width_px if top_ramped else 0
    iy1 = mh - (width_px if bottom_ramped else 0)
    if (
        ix0 >= ix1
        or iy0 >= iy1
        or ImageChops.difference(
            output.crop((x + ix0, y + iy0, x + ix1, y + iy1)),
            main.crop((ix0, iy0, ix1, iy1)),
        ).getbbox()
    ):
        raise EngineError("Không giữ được nội dung ảnh chính khi ghép nền.")
    return output


def create_wallpaper(folder, prompt, preset):
    if not is_available():
        raise EngineError("Thiếu môi trường hoặc model outpainting local. Xem README.")
    source = folder / "detail-native.png"
    if not source.is_file():
        source = folder / "native.png"
    if not source.is_file():
        source = folder / "result.png"
    with Image.open(source) as image:
        # 4x native intermediate from a 16 MP input (queue.MAX_PIXELS * 16; kept literal to avoid import cycle).
        if image.width * image.height > 16_000_000 * 16:
            raise EngineError(
                "Ảnh trung gian quá lớn để tạo nền AI; hãy tạo lại bản upscale 2×."
            )
        if "A" in image.getbands() and image.getchannel("A").getextrema() != (255, 255):
            raise EngineError(
                "Nền AI hiện chưa hỗ trợ ảnh trong suốt. Bản upscale vẫn được giữ."
            )
    temporary = folder / "wallpaper.tmp.png"
    sidecar = folder / "wallpaper.tmp-scene.txt"
    error_log = folder / "wallpaper-error.log"
    error_log.unlink(missing_ok=True)
    sidecar.unlink(missing_ok=True)
    try:
        result = subprocess.run(
            [
                str(PYTHON),
                "-m",
                "app.outpaint_runner",
                f"--prompt={prompt}",
                "--preset",
                preset,
                str(source),
                str(temporary),
            ],
            cwd=ROOT,
            capture_output=True,
            timeout=600,
            env={
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HOME": str(LAB / "cache"),
            },
        )
        if result.returncode != 0:
            # Diagnostics only: the runner prints tracebacks/paths, never pixel data.
            tail = (result.stderr or b"")[-8000:].decode("utf-8", "replace")
            error_log.write_text(
                f"returncode={result.returncode}\n{tail}", encoding="utf-8"
            )
            raise EngineError(
                "Tạo nền AI gặp lỗi xử lý. Bản upscale vẫn được giữ."
            )
        with Image.open(temporary) as image:
            image.load()
            if image.size != SIZE or image.format != "PNG":
                raise EngineError(
                    "Ảnh nền AI không đúng định dạng hoặc kích thước 3840×2160."
                )
        temporary.replace(folder / "wallpaper.png")
        if sidecar.is_file():
            raw = sidecar.read_text(encoding="utf-8", errors="replace")
            return "".join(ch for ch in raw if ch.isprintable()).strip()[:300]
        return None
    except subprocess.TimeoutExpired as error:
        raise EngineError(
            "Tạo nền AI quá 10 phút và đã dừng. Bản upscale vẫn được giữ."
        ) from error
    except FileNotFoundError as error:
        raise EngineError("Thiếu môi trường hoặc kết quả outpainting local.") from error
    finally:
        temporary.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        shutil.rmtree(folder / "wallpaper.tmp-esrgan", ignore_errors=True)
