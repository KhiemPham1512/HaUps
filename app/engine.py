from pathlib import Path
import subprocess
import os
import shutil
import re
from time import monotonic

from PIL import Image


# Allow 8x outputs from 16 MP inputs (up to 1024 MP); untrusted uploads are gated by queue.MAX_PIXELS before decode.
Image.MAX_IMAGE_PIXELS = 16_000_000 * 64 + 1

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = ROOT / ".local-artifacts/engine-poc/engine"
# Anime: x4plus-anime fine-tuned on anime characters/backgrounds, 50% weight-interpolated with stock (lab/upscale, gate 2026-11-16).
MODELS = {"photo": "realesrgan-x4plus", "anime": "haups-anime-x4"}


class EngineError(RuntimeError):
    pass


def _share_output(source, target):
    temporary = target.with_suffix(".tmp")
    try:
        temporary.unlink(missing_ok=True)
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        # Replace the directory entry; writing through a link would corrupt the cache.
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _save_png(image, target):
    temporary = target.with_suffix(".tmp")
    try:
        image.save(temporary, format="PNG")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


class EngineAdapter:
    def __init__(self, engine_dir=DEFAULT_ENGINE, gpu_id=None):
        self.directory = Path(os.environ.get("HAUPS_ENGINE_DIR", engine_dir))
        self.gpu_id = gpu_id if gpu_id is not None else self._gpu()

    @staticmethod
    def _gpu():
        override = os.environ.get("HAUPS_GPU_ID")
        if override is not None:
            if not override.isdigit():
                raise RuntimeError("HAUPS_GPU_ID phải là số GPU Vulkan không âm.")
            return int(override)
        try:
            result = subprocess.run(
                ["vulkaninfo", "--summary"], capture_output=True, timeout=20
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(
                "Không kiểm tra được Vulkan. Cần vulkaninfo hoặc cấu hình HAUPS_GPU_ID đúng thiết bị NVIDIA."
            ) from error
        blocks = re.split(r"GPU(\d+):", result.stdout.decode("utf-8", errors="replace"))
        for index in range(1, len(blocks), 2):
            if re.search(
                r"driverID\s*=\s*DRIVER_ID_NVIDIA_PROPRIETARY", blocks[index + 1]
            ):
                return int(blocks[index])
        raise RuntimeError(
            "Không thấy GPU NVIDIA Vulkan native. Hãy kiểm tra driver trước khi chạy."
        )

    def process(self, folder, scale, preset):
        model = MODELS[preset]
        tiles = ("128", "64", "32") if preset == "anime" else ("64", "32")
        source_path = folder / "source.png"
        engine_input = folder / "engine-input.png"
        native = folder / "native.png"
        command = [
            str(self.directory / "realesrgan-ncnn-vulkan.exe"),
            "-i",
            str(source_path),
            "-o",
            str(native),
            "-m",
            str(self.directory / "models"),
            "-n",
            model,
            "-s",
            "4",
            "-g",
            str(self.gpu_id),
            "-t",
            tiles[0],
            "-j",
            "1:1:1",
            "-f",
            "png",
        ]
        deadline = monotonic() + 600
        native_valid = False
        alpha_channel = None
        try:
            if not native.is_file():
                with Image.open(source_path) as pre_source:
                    if (
                        "A" in pre_source.getbands()
                        or "transparency" in pre_source.info
                        or pre_source.mode in ("LA", "PA")
                    ):
                        with pre_source.convert("RGBA") as rgba_source:
                            alpha_channel = rgba_source.getchannel("A")
                            with rgba_source.convert("RGB") as rgb_source:
                                _save_png(rgb_source, engine_input)
                        command[command.index("-i") + 1] = str(engine_input)
                for tile in tiles:
                    command[command.index("-t") + 1] = tile
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        timeout=max(0.1, deadline - monotonic()),
                    )
                    if result.returncode == 0:
                        break
                    stderr = result.stderr.lower()
                    if not any(
                        marker in stderr
                        for marker in (b"out of memory", b"vkallocatememory failed")
                    ):
                        raise EngineError(
                            "Engine xử lý thất bại. Kiểm tra model và GPU rồi thêm lại ảnh."
                        )
                else:
                    raise EngineError(
                        "GPU chưa đủ bộ nhớ dù đã giảm tile. Đóng bớt ứng dụng GPU hoặc chọn ảnh nhỏ hơn."
                    )
            if not native.is_file():
                raise EngineError("Engine không tạo được ảnh kết quả.")
            with Image.open(source_path) as source:
                source_size = source.size
                source_has_alpha = (
                    "A" in source.getbands()
                    or "transparency" in source.info
                    or source.mode in ("LA", "PA")
                )
            merged_native = None
            with Image.open(native) as generated:
                generated.load()
                if generated.size != (source_size[0] * 4, source_size[1] * 4):
                    raise EngineError("Kích thước kết quả engine không hợp lệ.")
                if alpha_channel is not None:
                    merged_native = generated.convert("RGBA")
                    with alpha_channel.resize(
                        generated.size, Image.Resampling.LANCZOS
                    ) as resized_alpha:
                        merged_native.putalpha(resized_alpha)
            if merged_native is not None:
                try:
                    _save_png(merged_native, native)
                finally:
                    merged_native.close()
            with Image.open(native) as rendered:
                rendered.load()
                if source_has_alpha and "A" not in rendered.getbands():
                    raise EngineError("Engine không giữ được kênh trong suốt của ảnh.")
                native_valid = True
                for factor in (2, 4, 8):
                    target = folder / f"result-{factor}.png"
                    if factor > scale or target.is_file():
                        continue
                    if factor == 4:
                        _share_output(native, target)
                    else:
                        with rendered.resize(
                            (source_size[0] * factor, source_size[1] * factor),
                            Image.Resampling.LANCZOS,
                        ) as output:
                            _save_png(output, target)
            _share_output(folder / f"result-{scale}.png", folder / "result.png")
        except subprocess.TimeoutExpired as error:
            native.unlink(missing_ok=True)
            raise EngineError(
                "Ảnh xử lý quá 10 phút. Tác vụ đã dừng; hãy chọn ảnh nhỏ hơn."
            ) from error
        except FileNotFoundError as error:
            raise EngineError(
                "Thiếu engine hoặc model. Kiểm tra cài đặt trước khi thử lại."
            ) from error
        except Exception:
            if not native_valid:
                native.unlink(missing_ok=True)
            raise
        finally:
            if alpha_channel is not None:
                alpha_channel.close()
            engine_input.unlink(missing_ok=True)
