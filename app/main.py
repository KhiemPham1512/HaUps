from contextlib import asynccontextmanager
from pathlib import Path
import os
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.engine import MODELS, EngineAdapter, ROOT
from app.queue import JobQueue, StorageError
from app.lock import single_instance
from app.security import LocalOnly
from app.wallpaper import is_available


def create_app(settings=None):
    settings = settings or {}

    @asynccontextmanager
    async def lifespan(app):
        with single_instance():
            engine = EngineAdapter(
                **{
                    key: settings[key]
                    for key in ("engine_dir", "gpu_id")
                    if key in settings
                }
            )
            app.state.queue = JobQueue(
                settings.get(
                    "data_dir", os.environ.get("HAUPS_DATA_DIR", ROOT / ".haups-data")
                ),
                engine,
            )
            try:
                yield
            finally:
                app.state.queue.close()

    app = FastAPI(title="HaUps", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(LocalOnly, allow_test_host="data_dir" in settings)

    @app.get("/")
    def home():
        return FileResponse(ROOT / "web/index.html")

    app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")

    @app.get("/api/capabilities")
    def capabilities():
        directory = app.state.queue.engine.directory
        available = [
            preset
            for preset, model in MODELS.items()
            if (directory / "realesrgan-ncnn-vulkan.exe").is_file()
            and all(
                (directory / "models" / (model + extension)).is_file()
                for extension in (".bin", ".param")
            )
        ]
        return {
            "presets": available,
            "gpu_id": app.state.queue.engine.gpu_id,
            "outpainting": is_available(),
        }

    @app.post("/api/batches", status_code=202)
    def submit(
        files: Annotated[list[UploadFile], File()],
        scale: Annotated[Literal["2", "4", "8"], Form()] = "2",
        preset: Annotated[Literal["photo", "anime"], Form()] = "photo",
    ):
        try:
            result = app.state.queue.add(
                [(f.filename, f.file) for f in files],
                int(scale),
                preset,
            )
        except StorageError as error:
            raise HTTPException(503, str(error)) from error
        except ValueError as error:
            raise HTTPException(429, str(error)) from error
        if not result["jobs"]:
            raise HTTPException(422, result["rejected"])
        return result

    def _with_file_sizes(job):
        folder = app.state.queue.directory / job["id"]
        result_bytes = {}
        for factor in (2, 4, 8):
            path = folder / f"result-{factor}.png"
            if not path.is_file() and factor == job.get("scale"):
                path = folder / "result.png"
            try:
                if path.is_file():
                    result_bytes[str(factor)] = path.stat().st_size
            except OSError:
                pass
        wallpaper_bytes = None
        wallpaper_path = folder / "wallpaper.png"
        try:
            if wallpaper_path.is_file():
                wallpaper_bytes = wallpaper_path.stat().st_size
        except OSError:
            pass
        return {
            **job,
            "result_bytes": result_bytes,
            "wallpaper_bytes": wallpaper_bytes,
        }

    def _jobs_snapshot():
        return [_with_file_sizes(job) for job in app.state.queue.snapshot()]

    @app.get("/api/jobs")
    def jobs():
        return {"jobs": _jobs_snapshot()}

    @app.get("/api/jobs/{job_id}/{kind}")
    def image(
        job_id: str,
        kind: Literal["source", "download", "wallpaper"],
        inline: bool = False,
        scale: Literal["2", "4", "8"] | None = None,
    ):
        jobs = {job["id"]: job for job in app.state.queue.snapshot()}
        if job_id not in jobs:
            raise HTTPException(404, "Chưa có ảnh kết quả.")
        job = jobs[job_id]
        if kind == "wallpaper":
            return wallpaper(job_id, inline)
        factor = int(scale) if scale else job["scale"]
        available = job.get(
            "available_scales", [job["scale"]] if job["status"] == "done" else []
        )
        if kind == "download" and (
            factor not in available
            if scale or job.get("task") == "wallpaper"
            else job["status"] != "done"
        ):
            raise HTTPException(404, "Chưa có ảnh kết quả.")
        filename = "source.png" if kind == "source" else "result.png"
        if (
            kind == "download"
            and scale
            and (app.state.queue.directory / job_id / f"result-{factor}.png").is_file()
        ):
            filename = f"result-{factor}.png"
        return FileResponse(
            app.state.queue.directory / job_id / filename,
            media_type="image/png",
            filename=f"{Path(job['filename']).stem}_x{factor}.png"
            if kind == "download" and not inline
            else None,
        )

    @app.post("/api/jobs/{job_id}/remove")
    def remove_job(job_id: str):
        try:
            app.state.queue.remove(job_id)
        except KeyError as error:
            raise HTTPException(404, "Ảnh không còn trong hàng đợi.") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        except StorageError as error:
            raise HTTPException(503, str(error)) from error
        return {"removed": job_id}

    @app.post("/api/jobs/{job_id}/wallpaper", status_code=202)
    def request_wallpaper(
        job_id: str, prompt: Annotated[str, Form(max_length=500)] = ""
    ):
        try:
            app.state.queue.request_wallpaper(job_id, prompt.strip())
        except KeyError as error:
            raise HTTPException(404, "Không tìm thấy ảnh.") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        except StorageError as error:
            raise HTTPException(503, str(error)) from error
        return {"jobs": _jobs_snapshot()}

    def wallpaper(job_id: str, inline: bool = False):
        job = next((j for j in app.state.queue.snapshot() if j["id"] == job_id), None)
        if not job or not job.get("wallpaper_available"):
            raise HTTPException(404, "Chưa có hình nền AI.")
        path = app.state.queue.directory / job_id / "wallpaper.png"
        if not path.is_file():
            raise HTTPException(404, "Hình nền AI không còn trong dữ liệu tạm.")
        return FileResponse(
            path,
            media_type="image/png",
            filename=None if inline else f"{Path(job['filename']).stem}_3840x2160.png",
        )

    @app.post("/api/jobs/{job_id}/scale")
    def change_scale(job_id: str, scale: Annotated[Literal["2", "4", "8"], Form()]):
        try:
            app.state.queue.request_scale(job_id, int(scale))
        except KeyError as error:
            raise HTTPException(404, "Không tìm thấy ảnh.") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        except StorageError as error:
            raise HTTPException(503, str(error)) from error
        return {"jobs": _jobs_snapshot()}

    return app


app = create_app()
