from io import BytesIO
from pathlib import Path
from threading import Event
import subprocess
import time

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app


def png():
    image = Image.new("RGB", (16, 12), "#336688")
    image.putpixel((0, 0), (255, 255, 255))
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def wait_finished(client, expected):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        jobs = client.get("/api/jobs").json()["jobs"]
        if len(jobs) == expected and all(
            j["status"] in ("done", "failed") for j in jobs
        ):
            return jobs
        time.sleep(0.02)
    raise AssertionError("Queue did not finish")


def test_multiple_images_are_queued_and_processed_one_at_a_time(tmp_path, monkeypatch):
    entered, release = Event(), Event()
    running = 0
    peak = 0
    order = []

    def engine_run(argv, **kwargs):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        order.append(argv[argv.index("-i") + 1])
        entered.set()
        assert release.wait(3)
        with Image.open(argv[argv.index("-i") + 1]) as image:
            image.resize((image.width * 4, image.height * 4)).save(
                argv[argv.index("-o") + 1]
            )
        running -= 1
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", engine_run)
    with TestClient(create_app({"data_dir": tmp_path, "gpu_id": 0})) as client:
        response = client.post(
            "/api/batches",
            files=[
                ("files", ("same.png", png(), "image/png")),
                ("files", ("same.png", png(), "image/png")),
            ],
            data={"scale": "2", "preset": "photo"},
        )
        assert response.status_code == 202, response.text
        assert entered.wait(2)
        jobs = client.get("/api/jobs").json()["jobs"]
        assert [j["status"] for j in jobs] == ["running", "queued"]
        assert len({j["id"] for j in jobs}) == 2
        assert jobs[0]["filename"] == jobs[1]["filename"] == "same.png"
        assert running == peak == 1
        appended = client.post(
            "/api/batches", files={"files": ("later.png", png(), "image/png")}
        )
        assert appended.status_code == 202
        assert [job["status"] for job in client.get("/api/jobs").json()["jobs"]] == [
            "running",
            "queued",
            "queued",
        ]
        assert peak == 1
        release.set()
        jobs = wait_finished(client, 3)
        assert [j["status"] for j in jobs] == ["done", "done", "done"]
        assert peak == 1 and len(order) == 3
        assert order == [str(tmp_path / job["id"] / "source.png") for job in jobs]
        for job in jobs:
            download = client.get(job["download_url"])
            assert download.status_code == 200
            with Image.open(BytesIO(download.content)) as result:
                assert result.size == (32, 24)
            assert client.get(job["source_url"]).content


def test_remove_queue_job_skips_pending_and_persists_without_deleting_source(
    tmp_path, monkeypatch
):
    entered, release = Event(), Event()
    calls = []

    def engine_run(argv, **kwargs):
        calls.append(argv)
        entered.set()
        assert release.wait(5)
        with Image.open(argv[argv.index("-i") + 1]) as image:
            image.resize((image.width * 4, image.height * 4)).save(
                argv[argv.index("-o") + 1]
            )
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", engine_run)
    settings = {"data_dir": tmp_path, "gpu_id": 0}
    with TestClient(create_app(settings)) as client:
        try:
            jobs = client.post(
                "/api/batches",
                files=[
                    ("files", ("first.png", png(), "image/png")),
                    ("files", ("second.png", png(), "image/png")),
                ],
            ).json()["jobs"]
            assert entered.wait(2)
            first, second = [job["id"] for job in jobs]
            assert client.post(f"/api/jobs/{first}/remove").status_code == 409
            assert (
                client.post(
                    f"/api/jobs/{second}/remove",
                    headers={"Origin": "https://outside.invalid"},
                ).status_code
                == 403
            )
            assert client.post(f"/api/jobs/{second}/remove").status_code == 200
            assert client.post("/api/jobs/unknown/remove").status_code == 404
            assert [j["id"] for j in client.get("/api/jobs").json()["jobs"]] == [first]
            assert (tmp_path / second / "source.png").is_file()
        finally:
            release.set()
        assert wait_finished(client, 1)[0]["status"] == "done"
        assert len(calls) == 1
        replace = Path.replace

        def disk_failure(path, target):
            if path.name == "queue.tmp":
                raise OSError("Disk full")
            return replace(path, target)

        with monkeypatch.context() as failure:
            failure.setattr(Path, "replace", disk_failure)
            assert client.post(f"/api/jobs/{first}/remove").status_code == 503
            assert client.get("/api/jobs").json()["jobs"][0]["id"] == first
    with TestClient(create_app(settings)) as client:
        assert client.post(f"/api/jobs/{first}/remove").status_code == 200
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/jobs").json()["jobs"] == []
        assert client.get(f"/api/jobs/{second}/source").status_code == 404
    assert len(calls) == 1


def test_engine_failure_does_not_stop_remaining_images(tmp_path, monkeypatch):
    calls = []

    def engine_run(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            return subprocess.CompletedProcess(argv, 1, b"", b"GPU failure")
        with Image.open(argv[argv.index("-i") + 1]) as image:
            image.resize((image.width * 4, image.height * 4)).save(
                argv[argv.index("-o") + 1]
            )
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", engine_run)
    with TestClient(create_app({"data_dir": tmp_path, "gpu_id": 0})) as client:
        response = client.post(
            "/api/batches",
            files=[
                ("files", ("bad.png", png(), "image/png")),
                ("files", ("next.png", png(), "image/png")),
            ],
        )
        assert response.status_code == 202
        jobs = wait_finished(client, 2)
        assert [j["status"] for j in jobs] == ["failed", "done"]
        assert jobs[0]["error"] and jobs[0]["download_url"] is None
        assert client.get(f"/api/jobs/{jobs[0]['id']}/download").status_code == 404


def test_invalid_file_is_rejected_without_blocking_valid_file(tmp_path, monkeypatch):
    def engine_run(argv, **kwargs):
        with Image.open(argv[argv.index("-i") + 1]) as image:
            image.resize((image.width * 4, image.height * 4)).save(
                argv[argv.index("-o") + 1]
            )
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", engine_run)
    with TestClient(create_app({"data_dir": tmp_path, "gpu_id": 0})) as client:
        response = client.post(
            "/api/batches",
            files=[
                ("files", ("broken.png", b"not an image", "image/png")),
                ("files", ("../../same.png", png(), "image/png")),
            ],
        )
        assert response.status_code == 202
        assert response.json()["rejected"][0]["filename"] == "broken.png"
        jobs = wait_finished(client, 1)
        assert jobs[0]["status"] == "done"
        assert jobs[0]["filename"] == "same.png"
        assert not (tmp_path.parent / "same.png").exists()


def test_completed_queue_and_download_survive_server_restart(tmp_path, monkeypatch):
    def engine_run(argv, **kwargs):
        with Image.open(argv[argv.index("-i") + 1]) as image:
            image.resize((image.width * 4, image.height * 4)).save(
                argv[argv.index("-o") + 1]
            )
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", engine_run)
    settings = {"data_dir": tmp_path, "gpu_id": 0}
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/batches", files={"files": ("one.png", png(), "image/png")}
        )
        assert response.status_code == 202
        before = wait_finished(client, 1)[0]
    with TestClient(create_app(settings)) as client:
        after = client.get("/api/jobs").json()["jobs"]
        assert len(after) == 1 and after[0]["id"] == before["id"]
        assert after[0]["status"] == "done"
        assert client.get(after[0]["download_url"]).status_code == 200


def test_second_server_cannot_start_a_second_gpu_worker(tmp_path):
    import pytest

    with TestClient(create_app({"data_dir": tmp_path / "first", "gpu_id": 0})):
        with pytest.raises(RuntimeError, match="đang chạy"):
            with TestClient(create_app({"data_dir": tmp_path / "second", "gpu_id": 0})):
                pass


def test_storage_budget_rejects_more_files_without_running_gpu(tmp_path, monkeypatch):
    import app.queue as queue_module

    calls = []
    monkeypatch.setattr(queue_module, "STORAGE_LIMIT", 1, raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with TestClient(create_app({"data_dir": tmp_path, "gpu_id": 0})) as client:
        response = client.post(
            "/api/batches", files={"files": ("one.png", png(), "image/png")}
        )
        assert response.status_code == 422
        assert client.get("/api/jobs").json()["jobs"] == []
        assert not calls


def test_expired_results_are_cleaned_without_touching_other_files(
    tmp_path, monkeypatch
):
    import json

    def engine_run(argv, **kwargs):
        with Image.open(argv[argv.index("-i") + 1]) as image:
            image.resize((image.width * 4, image.height * 4)).save(
                argv[argv.index("-o") + 1]
            )
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", engine_run)
    settings = {"data_dir": tmp_path, "gpu_id": 0}
    untouched = tmp_path / "keep-user-file.txt"
    untouched.write_text("keep")
    with TestClient(create_app(settings)) as client:
        client.post("/api/batches", files={"files": ("one.png", png(), "image/png")})
        job = wait_finished(client, 1)[0]
    state = json.loads((tmp_path / "queue.json").read_text(encoding="utf-8"))
    state[0]["finished_at"] = time.time() - 25 * 3600
    (tmp_path / "queue.json").write_text(json.dumps(state), encoding="utf-8")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/jobs").json()["jobs"] == []
    assert untouched.read_text() == "keep"
    assert not (tmp_path / job["id"]).exists()


def test_foreign_sites_cannot_enqueue_images_and_large_bodies_are_refused(tmp_path):
    with TestClient(create_app({"data_dir": tmp_path, "gpu_id": 0})) as client:
        response = client.post(
            "/api/batches",
            files={"files": ("one.png", png(), "image/png")},
            headers={"Origin": "https://example.org"},
        )
        assert response.status_code == 403
        assert client.get("/api/jobs").json()["jobs"] == []
        response = client.post(
            "/api/batches",
            content=b"x",
            headers={"Content-Length": str(1024 * 1024 * 1024 + 1)},
        )
        assert response.status_code == 413
        assert (
            response.json()["detail"] == "Tổng dung lượng một lần tải tối đa 1 GB."
        )


def test_state_storage_failure_marks_jobs_failed_instead_of_killing_worker(
    tmp_path, monkeypatch
):
    calls = []
    with TestClient(create_app({"data_dir": tmp_path, "gpu_id": 0})) as client:
        monkeypatch.setattr(
            subprocess, "run", lambda *args, **kwargs: calls.append(args)
        )
        original = Path.replace

        def disk_full(path, target):
            if path.name == "queue.tmp":
                raise OSError("disk full")
            return original(path, target)

        monkeypatch.setattr(Path, "replace", disk_full)
        response = client.post(
            "/api/batches", files={"files": ("one.png", png(), "image/png")}
        )
        assert response.status_code in (202, 503)
        if response.status_code == 202:
            jobs = wait_finished(client, 1)
            assert jobs[0]["status"] == "failed"
        assert not calls


def test_uploaded_copy_applies_orientation_and_removes_private_metadata(
    tmp_path, monkeypatch
):
    from PIL import ImageCms

    entered, release = Event(), Event()

    def run(argv, **kwargs):
        entered.set()
        assert release.wait(3)
        return subprocess.CompletedProcess(argv, 1, b"", b"failed")

    monkeypatch.setattr(subprocess, "run", run)
    stream = BytesIO()
    image = Image.new("RGB", (16, 12), "#336688")
    exif = Image.Exif()
    exif[274], exif[270] = 6, "PRIVATE_TEST"
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    image.save(stream, format="JPEG", exif=exif, icc_profile=profile)
    try:
        with TestClient(create_app({"data_dir": tmp_path, "gpu_id": 0})) as client:
            response = client.post(
                "/api/batches",
                files={"files": ("portrait.jpg", stream.getvalue(), "image/jpeg")},
            )
            assert response.status_code == 202
            job = client.get("/api/jobs").json()["jobs"][0]
            with Image.open(
                BytesIO(client.get(job["source_url"]).content)
            ) as normalized:
                assert normalized.size == (12, 16)
                assert not normalized.getexif()
                assert "icc_profile" not in normalized.info
            release.set()
    finally:
        release.set()


def test_anime_8x_skips_tta_with_ten_minute_engine_budget(tmp_path, monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs["timeout"]))
        with Image.open(argv[argv.index("-i") + 1]) as image:
            image.resize((image.width * 4, image.height * 4)).save(
                argv[argv.index("-o") + 1]
            )
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", run)
    with TestClient(create_app({"data_dir": tmp_path, "gpu_id": 0})) as client:
        response = client.post(
            "/api/batches",
            files={"files": ("art.png", png(), "image/png")},
            data={"scale": "8", "preset": "anime"},
        )
        assert response.status_code == 202
        job = wait_finished(client, 1)[0]
        with Image.open(BytesIO(client.get(job["download_url"]).content)) as result:
            assert result.size == (128, 96)
        assert "-x" not in calls[0][0]
        assert calls[0][0][calls[0][0].index("-t") + 1] == "128"
        assert 590 < calls[0][1] <= 600
        assert job["available_scales"] == [2, 4, 8]
        for factor in (2, 4, 8):
            response = client.get(job["download_url"], params={"scale": factor})
            with Image.open(BytesIO(response.content)) as result:
                assert result.size == (16 * factor, 12 * factor)
        assert len(calls) == 1


def test_scale_upgrade_reuses_native_and_survives_restart(tmp_path, monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        with Image.open(argv[argv.index("-i") + 1]) as source:
            source.resize((source.width * 4, source.height * 4)).save(
                argv[argv.index("-o") + 1]
            )
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", run)
    settings = {"data_dir": tmp_path, "gpu_id": 0}
    with TestClient(create_app(settings)) as client:
        job = client.post(
            "/api/batches", files={"files": ("one.png", png(), "image/png")}
        ).json()["jobs"][0]
        first = wait_finished(client, 1)[0]
        assert first["available_scales"] == [2]
        assert client.get(first["download_url"], params={"scale": 8}).status_code == 404
        assert (
            client.post(f"/api/jobs/{job['id']}/scale", data={"scale": 4}).status_code
            == 200
        )
        assert wait_finished(client, 1)[0]["available_scales"] == [2, 4]
    with TestClient(create_app(settings)) as client:
        assert (
            client.post(f"/api/jobs/{job['id']}/scale", data={"scale": 8}).status_code
            == 200
        )
        assert wait_finished(client, 1)[0]["available_scales"] == [2, 4, 8]
        assert (
            client.post(f"/api/jobs/{job['id']}/scale", data={"scale": 2}).status_code
            == 200
        )
        assert (
            client.post(f"/api/jobs/{job['id']}/scale", data={"scale": 3}).status_code
            == 422
        )
    assert len(calls) == 1


def test_upload_accepts_seven_megapixels_and_rejects_larger(tmp_path, monkeypatch):
    import app.queue as queue_module

    assert queue_module.MAX_BYTES == 50 * 1024 * 1024
    assert queue_module.MAX_PIXELS == 16_000_000
    assert queue_module.MAX_PIXELS_8X == 10_000_000
    assert queue_module.STORAGE_LIMIT == 16 * 1024**3
    assert queue_module.MAX_PENDING == 100

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, b"", b"test"),
    )
    with TestClient(create_app({"data_dir": tmp_path, "gpu_id": 0})) as client:
        for size, status in [
            ((4000, 4000), 202),
            ((16_000_001, 1), 422),
            ((4001, 4000), 422),
        ]:
            stream = BytesIO()
            Image.new("RGB", size, "red").save(stream, "PNG", compress_level=1)
            response = client.post(
                "/api/batches",
                files={"files": ("large.png", stream.getvalue(), "image/png")},
            )
            assert response.status_code == status
            if status == 422:
                assert (
                    response.json()["detail"][0]["error"]
                    == "Ảnh không hợp lệ: cần PNG/JPG/WebP tĩnh 8-bit, tối đa 50 MB và 16 triệu pixel."
                )


def test_upload_rejects_over_fifty_megabytes_and_enforces_hundred_pending_cap(
    tmp_path, monkeypatch
):
    import app.queue as queue_module

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, b"", b"test"),
    )
    app = create_app({"data_dir": tmp_path, "gpu_id": 0})
    with TestClient(app) as client:
        with monkeypatch.context() as small_cap:
            small_cap.setattr(queue_module, "MAX_BYTES", 64)
            too_big = client.post(
                "/api/batches",
                files={"files": ("huge.png", b"x" * 65, "image/png")},
            )
        assert too_big.status_code == 422
        assert (
            too_big.json()["detail"][0]["error"]
            == "Ảnh không hợp lệ: cần PNG/JPG/WebP tĩnh 8-bit, tối đa 50 MB và 16 triệu pixel."
        )

        queue = app.state.queue
        with queue.condition:
            for idx in range(100):
                job_id = f"{idx:032x}"
                queue.jobs[job_id] = {
                    "id": job_id,
                    "filename": f"{idx}.png",
                    "scale": 2,
                    "preset": "photo",
                    "width": 16,
                    "height": 12,
                    "status": "queued",
                    "created_at": 0,
                    "finished_at": None,
                    "error": None,
                    "seconds": None,
                    "source_url": f"/api/jobs/{job_id}/source",
                    "download_url": None,
                }
        full_resp = client.post(
            "/api/batches",
            files={"files": ("extra.png", png(), "image/png")},
        )
        assert full_resp.status_code == 429
        assert (
            full_resp.json()["detail"]
            == "Hàng đợi có tối đa 100 ảnh chưa hoàn tất. Hãy chờ bớt ảnh."
        )


def test_scale_8x_rejects_over_ten_megapixels_and_accepts_nine_mp(
    tmp_path, monkeypatch
):
    def engine_run(argv, **kwargs):
        with Image.open(argv[argv.index("-i") + 1]) as image:
            image.resize((32, 32)).save(argv[argv.index("-o") + 1])
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", engine_run)

    def make_png(size):
        stream = BytesIO()
        Image.new("RGB", size, "red").save(stream, "PNG", compress_level=1)
        return stream.getvalue()

    large_10_5mp = make_png((3500, 3000))
    nine_mp = make_png((3000, 3000))

    with TestClient(create_app({"data_dir": tmp_path, "gpu_id": 0})) as client:
        batch = client.post(
            "/api/batches",
            files=[
                ("files", ("too_large_8x.png", large_10_5mp, "image/png")),
                ("files", ("small.png", png(), "image/png")),
            ],
            data={"scale": "8"},
        )
        assert batch.status_code == 202
        body = batch.json()
        assert body["rejected"] == [
            {
                "filename": "too_large_8x.png",
                "error": "Ảnh lớn hơn 10 triệu pixel chỉ hỗ trợ 2× hoặc 4×.",
            }
        ]
        assert [j["filename"] for j in body["jobs"]] == ["small.png"]
        wait_finished(client, 1)

        accepted = client.post(
            "/api/batches",
            files={"files": ("nine_mp.png", nine_mp, "image/png")},
            data={"scale": "8"},
        )
        assert accepted.status_code == 202
        assert accepted.json()["rejected"] == []
        wait_finished(client, 2)

        large_2x = client.post(
            "/api/batches",
            files={"files": ("large_2x.png", large_10_5mp, "image/png")},
            data={"scale": "2"},
        )
        assert large_2x.status_code == 202
        large_job_id = large_2x.json()["jobs"][0]["id"]
        wait_finished(client, 3)

        upgrade = client.post(
            f"/api/jobs/{large_job_id}/scale",
            data={"scale": "8"},
        )
        assert upgrade.status_code == 409
        assert (
            upgrade.json()["detail"]
            == "Ảnh lớn hơn 10 triệu pixel chỉ hỗ trợ 2× hoặc 4×."
        )


def test_legacy_state_and_form_fields_are_ignored_cleanly(tmp_path, monkeypatch):
    import json

    def engine_run(argv, **kwargs):
        with Image.open(argv[argv.index("-i") + 1]) as image:
            image.resize((image.width * 4, image.height * 4)).save(
                argv[argv.index("-o") + 1]
            )
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", engine_run)
    legacy_key = "preserve" + "_details"
    job_id = "a" * 32
    folder = tmp_path / job_id
    folder.mkdir()
    Image.new("RGB", (16, 12), "red").save(folder / "source.png")
    (tmp_path / "queue.json").write_text(
        json.dumps(
            [
                {
                    "id": job_id,
                    "filename": "legacy.png",
                    "scale": 2,
                    "preset": "anime",
                    legacy_key: True,
                    "width": 16,
                    "height": 12,
                    "status": "queued",
                    "created_at": time.time(),
                    "finished_at": None,
                    "error": None,
                    "seconds": None,
                    "source_url": f"/api/jobs/{job_id}/source",
                    "download_url": None,
                }
            ]
        ),
        encoding="utf-8",
    )
    with TestClient(create_app({"data_dir": tmp_path, "gpu_id": 0})) as client:
        done = wait_finished(client, 1)[0]
        assert done["status"] == "done"
        response = client.post(
            "/api/batches",
            files={"files": ("new.png", png(), "image/png")},
            data={"preset": "photo", "scale": "2", legacy_key: "true"},
        )
        assert response.status_code == 202
        created = response.json()["jobs"][0]
        assert legacy_key not in created
        wait_finished(client, 2)


def test_wallpaper_job_records_scene_returned_by_create_wallpaper(
    tmp_path, monkeypatch
):
    import app.queue as queue_module

    def engine_run(argv, **kwargs):
        with Image.open(argv[argv.index("-i") + 1]) as image:
            image.resize((image.width * 4, image.height * 4)).save(
                argv[argv.index("-o") + 1]
            )
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    def fake_create_wallpaper(folder, prompt, preset):
        Image.new("RGB", (3840, 2160), "#336688").save(folder / "wallpaper.png")
        return None if prompt else "outdoors, sky"

    monkeypatch.setattr(subprocess, "run", engine_run)
    monkeypatch.setattr(queue_module, "is_available", lambda: True)
    monkeypatch.setattr(queue_module, "create_wallpaper", fake_create_wallpaper)

    with TestClient(create_app({"data_dir": tmp_path, "gpu_id": 0})) as client:
        job = client.post(
            "/api/batches",
            files={"files": ("scene.png", png(), "image/png")},
        ).json()["jobs"][0]
        wait_finished(client, 1)

        response = client.post(
            f"/api/jobs/{job['id']}/wallpaper",
            data={"prompt": ""},
        )
        assert response.status_code == 202
        done_auto = wait_finished(client, 1)[0]
        assert done_auto["wallpaper_available"] is True
        assert done_auto["wallpaper_completed_prompt"] == ""
        assert done_auto["wallpaper_scene"] == "outdoors, sky"

        response = client.post(
            f"/api/jobs/{job['id']}/wallpaper",
            data={"prompt": "custom sunset"},
        )
        assert response.status_code == 202
        done_custom = wait_finished(client, 1)[0]
        assert done_custom["wallpaper_completed_prompt"] == "custom sunset"
        assert done_custom["wallpaper_scene"] is None



