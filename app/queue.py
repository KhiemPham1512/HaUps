from collections import deque
from pathlib import Path
from io import BytesIO
import re
import json
import shutil
from threading import Condition, Thread
from time import perf_counter, time
from uuid import uuid4

from PIL import Image, ImageOps, ImageCms
from app.engine import EngineError
from app.wallpaper import create_wallpaper, is_available


MAX_BYTES = 50 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_PIXELS_8X = 10_000_000  # 8x output <= 640 MP (user 2026-11-16: overnight runs); needs several GB RAM and a large PNG
STORAGE_LIMIT = 16 * 1024**3
MAX_PENDING = 100


class StorageError(RuntimeError):
    pass


class JobQueue:
    def __init__(self, directory, engine):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.engine = engine
        self.jobs = {}
        self.pending = deque()
        self.state_file = self.directory / "queue.json"
        if self.state_file.exists():
            for job in json.loads(self.state_file.read_text(encoding="utf-8")):
                if not re.fullmatch(r"[0-9a-f]{32}", job["id"]):
                    raise RuntimeError(
                        "Trạng thái hàng đợi không hợp lệ; không tự ghi đè."
                    )
                if job["status"] == "running":
                    job["status"], job["error"] = (
                        "failed",
                        "Tác vụ bị gián đoạn khi ứng dụng dừng. Hãy thêm lại ảnh.",
                    )
                    job["finished_at"] = time()
                    if job.get("task") != "wallpaper":
                        job["download_url"] = None
                self.jobs[job["id"]] = job
                if job["status"] == "queued":
                    self.pending.append(job["id"])
        self.condition = Condition()
        self.stopping = False
        self.storage_error = False
        self._cleanup()
        self.worker = Thread(target=self._work, name="haups-gpu-worker", daemon=True)
        self.worker.start()

    def add(self, files, scale, preset):
        created, rejected = [], []
        with self.condition:
            if not 1 <= len(files) <= 20:
                raise ValueError("Mỗi lần chọn từ 1 đến 20 ảnh.")
            if (
                sum(
                    job["status"] in ("queued", "running") for job in self.jobs.values()
                )
                + len(files)
                > MAX_PENDING
            ):
                raise ValueError(
                    "Hàng đợi có tối đa 100 ảnh chưa hoàn tất. Hãy chờ bớt ảnh."
                )
            try:
                self._cleanup()
            except OSError as error:
                raise StorageError(
                    "Không dọn được dữ liệu tạm. Kiểm tra quyền ghi và dung lượng đĩa."
                ) from error
            if self.storage_error:
                raise StorageError(
                    "Không lưu được trạng thái hàng đợi. Kiểm tra dung lượng đĩa rồi khởi động lại ứng dụng."
                )
            used = sum(
                path.stat().st_size
                for path in self.directory.rglob("*")
                if path.is_file()
            )
            reserved = sum(
                job["width"] * job["height"] * (500 if job["scale"] == 8 else 140)
                for job in self.jobs.values()
                if job["status"] in ("queued", "running")
            )
            for filename, stream in files:
                filename = (
                    re.sub(
                        r"[\x00-\x1f\x7f]",
                        "",
                        (filename or "anh").replace("\\", "/").rsplit("/", 1)[-1],
                    )[:120]
                    or "anh"
                )
                try:
                    content = stream.read(MAX_BYTES + 1)
                    if len(content) > MAX_BYTES:
                        raise ValueError("Ảnh vượt giới hạn 50 MB.")
                    with Image.open(BytesIO(content)) as opened:
                        if (
                            opened.format not in ("PNG", "JPEG", "WEBP")
                            or getattr(opened, "n_frames", 1) != 1
                        ):
                            raise ValueError("Chỉ hỗ trợ ảnh tĩnh PNG, JPG hoặc WebP.")
                        if opened.width * opened.height > MAX_PIXELS:
                            raise ValueError("Ảnh vượt giới hạn 16 triệu pixel.")
                        if scale == 8 and opened.width * opened.height > MAX_PIXELS_8X:
                            rejected.append(
                                {
                                    "filename": filename,
                                    "error": "Ảnh lớn hơn 10 triệu pixel chỉ hỗ trợ 2× hoặc 4×.",
                                }
                            )
                            continue
                        if (
                            opened.format == "PNG" and content[24] > 8
                        ) or opened.mode not in ("RGB", "RGBA", "L", "LA", "P"):
                            raise ValueError(
                                "Chỉ hỗ trợ ảnh 8-bit RGB hoặc trong suốt."
                            )
                        opened.load()
                        image = ImageOps.exif_transpose(opened)
                        image = image.convert(
                            "RGBA"
                            if "A" in image.getbands() or "transparency" in image.info
                            else "RGB"
                        )
                        if opened.info.get("icc_profile"):
                            image = ImageCms.profileToProfile(
                                image,
                                ImageCms.ImageCmsProfile(
                                    BytesIO(opened.info["icc_profile"])
                                ),
                                ImageCms.createProfile("sRGB"),
                                outputMode=image.mode,
                            )
                        image.info.clear()
                        width, height = image.size
                        estimate = width * height * (500 if scale == 8 else 140)
                        if (
                            used + reserved + estimate > STORAGE_LIMIT
                            or shutil.disk_usage(self.directory).free
                            < reserved + estimate + 100 * 1024 * 1024
                        ):
                            rejected.append(
                                {
                                    "filename": filename,
                                    "error": "Chưa đủ dung lượng cho ảnh này. Chờ dọn kết quả cũ hoặc chọn ảnh nhỏ hơn.",
                                }
                            )
                            continue
                        reserved += estimate
                        job_id = uuid4().hex
                        folder = self.directory / job_id
                        folder.mkdir()
                        image.save(folder / "source.png")
                except (
                    OSError,
                    ValueError,
                    Image.DecompressionBombError,
                    ImageCms.PyCMSError,
                ):
                    rejected.append(
                        {
                            "filename": filename,
                            "error": "Ảnh không hợp lệ: cần PNG/JPG/WebP tĩnh 8-bit, tối đa 50 MB và 16 triệu pixel.",
                        }
                    )
                    continue
                job = {
                    "id": job_id,
                    "filename": filename,
                    "scale": scale,
                    "preset": preset,
                    "width": width,
                    "height": height,
                    "status": "queued",
                    "created_at": time(),
                    "finished_at": None,
                    "error": None,
                    "seconds": None,
                    "source_url": f"/api/jobs/{job_id}/source",
                    "download_url": None,
                }
                self.jobs[job_id] = job
                self.pending.append(job_id)
                created.append(dict(job))
            self._save()
            self.condition.notify()
        return {"jobs": created, "rejected": rejected}

    def snapshot(self):
        with self.condition:
            return [dict(job) for job in self.jobs.values() if not job.get("removed")]

    def remove(self, job_id):
        with self.condition:
            job = self.jobs.get(job_id)
            if not job or job.get("removed"):
                raise KeyError(job_id)
            if job["status"] == "running":
                raise ValueError(
                    "Ảnh đang được GPU xử lý. Hãy chờ hoàn tất để bỏ khỏi hàng đợi."
                )
            if self.storage_error:
                raise StorageError(
                    "Không lưu được thay đổi hàng đợi. Kiểm tra ổ đĩa rồi thử lại."
                )
            previous = dict(job)
            job["removed"] = True
            if job["status"] == "queued":
                job["status"] = "cancelled"
                job["finished_at"] = time()
            self._save()
            if self.storage_error:
                self.jobs[job_id] = previous
                raise StorageError(
                    "Không lưu được thay đổi hàng đợi. Kiểm tra ổ đĩa rồi thử lại."
                )
            if job_id in self.pending:
                self.pending.remove(job_id)

    def request_scale(self, job_id, scale):
        with self.condition:
            job = self.jobs.get(job_id)
            if not job or job.get("removed"):
                raise KeyError(job_id)
            if scale in job.get(
                "available_scales", [job["scale"]] if job["status"] == "done" else []
            ):
                return
            if job["status"] in ("queued", "running"):
                raise ValueError("Ảnh đang chờ hoặc đang xử lý. Hãy chờ hoàn tất.")
            if scale == 8 and job["width"] * job["height"] > MAX_PIXELS_8X:
                raise ValueError("Ảnh lớn hơn 10 triệu pixel chỉ hỗ trợ 2× hoặc 4×.")
            if self.storage_error:
                raise StorageError("Không lưu được trạng thái hàng đợi.")
            reserved = sum(
                j["width"] * j["height"] * (500 if j["scale"] == 8 else 140)
                for j in self.jobs.values()
                if j["status"] in ("queued", "running")
            )
            needed = job["width"] * job["height"] * (500 if scale == 8 else 140)
            used = sum(
                p.stat().st_size for p in self.directory.rglob("*") if p.is_file()
            )
            if (
                used + reserved + needed > STORAGE_LIMIT
                or shutil.disk_usage(self.directory).free
                < reserved + needed + 100 * 1024 * 1024
            ):
                raise ValueError("Chưa đủ dung lượng để tạo kích thước này.")
            previous = dict(job)
            job.update(
                task="upscale",
                scale=scale,
                status="queued",
                error=None,
                finished_at=None,
                download_url=None,
            )
            self._save()
            if self.storage_error:
                self.jobs[job_id] = previous
                raise StorageError("Không lưu được trạng thái hàng đợi.")
            self.pending.append(job_id)
            self.condition.notify()

    def request_wallpaper(self, job_id, prompt):
        with self.condition:
            job = self.jobs.get(job_id)
            if not job or job.get("removed"):
                raise KeyError(job_id)
            if job["status"] in ("queued", "running") or not job.get(
                "available_scales"
            ):
                raise ValueError("Hãy chờ ảnh upscale hoàn tất trước khi tạo nền AI.")
            if (
                job.get("wallpaper_available")
                and job.get("wallpaper_completed_prompt") == prompt
                and (self.directory / job_id / "wallpaper.png").is_file()
            ):
                return
            if not is_available():
                raise ValueError("Chưa cài môi trường hoặc model outpainting local.")
            if self.storage_error:
                raise StorageError("Không lưu được trạng thái hàng đợi.")
            reserved = sum(
                j["width"] * j["height"] * (500 if j["scale"] == 8 else 140)
                for j in self.jobs.values()
                if j["status"] in ("queued", "running")
            )
            used = sum(
                p.stat().st_size for p in self.directory.rglob("*") if p.is_file()
            )
            if (
                used + reserved + 32 * 1024**2 > STORAGE_LIMIT
                or shutil.disk_usage(self.directory).free < reserved + 132 * 1024**2
                or sum(j["status"] in ("queued", "running") for j in self.jobs.values())
                >= MAX_PENDING
            ):
                raise ValueError(
                    "Chưa đủ dung lượng hoặc chỗ trong hàng đợi để tạo nền AI."
                )
            previous = dict(job)
            job.update(
                task="wallpaper",
                wallpaper_prompt=prompt,
                status="queued",
                error=None,
                finished_at=None,
            )
            self._save()
            if self.storage_error:
                self.jobs[job_id] = previous
                raise StorageError("Không lưu được trạng thái hàng đợi.")
            self.pending.append(job_id)
            self.condition.notify()

    def _work(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopping or self.pending)
                if self.stopping:
                    return
                job = self.jobs[self.pending.popleft()]
                job["status"] = "running"
                self._save()
            start = perf_counter()
            scene = None
            try:
                if self.storage_error:
                    raise EngineError(
                        "Không lưu được trạng thái. Tác vụ đã dừng để bảo vệ dữ liệu; kiểm tra ổ đĩa rồi khởi động lại."
                    )
                if job.get("task") == "wallpaper":
                    scene = create_wallpaper(
                        self.directory / job["id"],
                        job["wallpaper_prompt"],
                        job["preset"],
                    )
                else:
                    self.engine.process(
                        self.directory / job["id"],
                        job["scale"],
                        job["preset"],
                    )
                status, error = "done", None
            except EngineError as failure:
                status, error = "failed", str(failure)
            except Exception:
                status, error = (
                    "failed",
                    "Không xử lý được ảnh. Kiểm tra engine, GPU hoặc dung lượng đĩa rồi thử lại.",
                )
            with self.condition:
                job["seconds"] = round(perf_counter() - start, 2)
                job["finished_at"] = time()
                job["status"], job["error"] = status, error
                if status == "done":
                    if job.get("task") == "wallpaper":
                        job["wallpaper_available"] = True
                        job["wallpaper_completed_prompt"] = job["wallpaper_prompt"]
                        job["wallpaper_scene"] = scene
                    job["available_scales"] = [
                        factor
                        for factor in (2, 4, 8)
                        if (
                            self.directory / job["id"] / f"result-{factor}.png"
                        ).is_file()
                    ]
                    job["download_url"] = f"/api/jobs/{job['id']}/download"
                self._save()

    def _cleanup(self):
        for job_id, job in list(self.jobs.items()):
            finished = job.get("finished_at")
            if (
                job["status"] not in ("done", "failed", "cancelled")
                or not finished
                or time() - finished < 24 * 3600
            ):
                continue
            folder = self.directory / job_id
            if (
                folder.is_symlink()
                or getattr(folder, "is_junction", lambda: False)()
                or folder.resolve().parent != self.directory.resolve()
            ):
                continue
            if folder.exists():
                known = {
                    "source.png",
                    "native.png",
                    "detail-native.png",
                    "detail-native.tmp",
                    "result.png",
                    "result-2.png",
                    "result-4.png",
                    "result-8.png",
                    "wallpaper.png",
                    "wallpaper.tmp.png",
                }
                if any(
                    path.name not in known or not path.is_file() or path.is_symlink()
                    for path in folder.iterdir()
                ):
                    continue
                for name in known:
                    (folder / name).unlink(missing_ok=True)
                folder.rmdir()
            del self.jobs[job_id]
        self._save()

    def _save(self):
        temporary = self.state_file.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(list(self.jobs.values()), ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(self.state_file)
        except OSError:
            self.storage_error = True

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify()
        self.worker.join()
