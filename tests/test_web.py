from functools import lru_cache, partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import os
from pathlib import Path
import re
from threading import Thread
from urllib.parse import urlsplit

from PIL import Image
from playwright.sync_api import expect, sync_playwright
import pytest

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=32)
def image_bytes(width, height, color):
    stream = BytesIO()
    Image.new("RGB", (width, height), color).save(stream, "PNG")
    return stream.getvalue()


def _chrome_path():
    # Ưu tiên HAUPS_CHROME_PATH; không có thì lấy Chromium mới nhất trong cache Playwright local.
    override = os.environ.get("HAUPS_CHROME_PATH")
    if override:
        return override
    cache = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    found = sorted(cache.glob("chromium-*/chrome-win64/chrome.exe"))
    return str(found[-1]) if found else None


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        instance = p.chromium.launch(
            executable_path=_chrome_path(),
            headless=True,
        )
        yield instance
        instance.close()


@pytest.fixture
def ui(browser):
    class Handler(SimpleHTTPRequestHandler):
        def translate_path(self, path):
            return super().translate_path(path.replace("/assets/", "/", 1))

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Handler, directory=str(ROOT / "web"))
    )
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    context = browser.new_context(
        viewport={"width": 1440, "height": 900}, has_touch=True
    )
    page = context.new_page()
    page.set_default_timeout(5000)
    state = {"jobs": [], "posts": [], "errors": [], "wallpaper_posts": []}
    page.on("pageerror", lambda error: state["errors"].append(str(error)))

    def api(route):
        path = urlsplit(route.request.url).path
        if path == "/api/capabilities":
            route.fulfill(
                json={
                    "presets": ["photo", "anime"],
                    "gpu_id": 2,
                    "outpainting": state.get("outpainting", False),
                }
            )
        elif path == "/api/jobs":
            route.fulfill(json={"jobs": state["jobs"]})
        elif path == "/api/batches":
            state["posts"].append(route.request.post_data_buffer)
            route.fulfill(status=202, json={"jobs": state["jobs"], "rejected": []})
        elif path.endswith("/wallpaper") and route.request.method == "POST":
            from urllib.parse import parse_qs, unquote_plus

            raw_body = route.request.post_data or ""
            state["wallpaper_posts"].append(unquote_plus(raw_body))
            prompt = parse_qs(raw_body, keep_blank_values=True).get("prompt", [""])[0]
            job_id = path.split("/")[-2]
            for job in state["jobs"]:
                if job["id"] == job_id:
                    job["task"] = "wallpaper"
                    job["status"] = "queued"
                    job["wallpaper_prompt"] = prompt
            route.fulfill(status=202, json={"jobs": state["jobs"]})
        elif path.endswith("/remove"):
            if state.get("remove_error"):
                route.fulfill(status=409, json={"detail": "Ảnh đang được GPU xử lý."})
            else:
                job_id = path.split("/")[-2]
                state["jobs"] = [j for j in state["jobs"] if j["id"] != job_id]
                route.fulfill(json={"removed": job_id})
        elif path.endswith("/scale"):
            from urllib.parse import parse_qs

            factor = int(parse_qs(route.request.post_data)["scale"][0])
            state.setdefault("scales", []).append(factor)
            job = next(j for j in state["jobs"] if j["id"] == path.split("/")[-2])
            job["scale"] = factor
            job["available_scales"] = [s for s in (2, 4, 8) if s <= factor]
            route.fulfill(json={"jobs": state["jobs"]})
        elif state.get("image_error"):
            route.fulfill(status=404, body="Image unavailable")
        else:
            job_id = path.split("/")[-2]
            job = next((j for j in state["jobs"] if j["id"] == job_id), None)
            if path.endswith("/wallpaper"):
                w, h, color = 3840, 2160, "#118855"
            else:
                w = job["width"] if job else 666
                h = job["height"] if job else 1000
                color = "#dd3344" if path.endswith("/source") else "#2266cc"
            route.fulfill(
                content_type="image/png",
                body=image_bytes(w, h, color),
            )

    page.route("**/api/**", api)
    page.goto(f"http://127.0.0.1:{server.server_port}")
    yield page, state
    try:
        assert not state["errors"], state["errors"]
    finally:
        context.close()
        server.shutdown()
        server.server_close()
        worker.join()


def test_scale_options_show_original_and_output_dimensions_before_upload(ui):
    page, state = ui
    page.locator("#files").set_input_files(
        {
            "name": "portrait.png",
            "mimeType": "image/png",
            "buffer": image_bytes(666, 1000, "red"),
        }
    )
    expect(page.locator("#original-size")).to_have_text("666 × 1000 px")
    expect(page.locator('[data-scale-size="2"]')).to_have_text("1332 × 2000 px")
    expect(page.locator('[data-scale-size="4"]')).to_have_text("2664 × 4000 px")
    page.get_by_role("radio", name="4×", exact=False).check()
    expect(page.locator('[data-scale-size="8"]')).to_have_text("5328 × 8000 px")
    page.get_by_role("radio", name="8×", exact=False).check()
    page.get_by_role("button", name="Thêm vào hàng đợi", exact=True).click()
    expect(page.locator("#selected-count")).to_have_text("0 ảnh đã chọn")
    assert b'name="scale"\r\n\r\n8' in state["posts"][0]


def test_batch_size_preview_follows_the_file_selected_for_viewing(ui):
    page, _ = ui
    page.locator("#files").set_input_files(
        [
            {
                "name": "portrait.png",
                "mimeType": "image/png",
                "buffer": image_bytes(666, 1000, "red"),
            },
            {
                "name": "landscape.png",
                "mimeType": "image/png",
                "buffer": image_bytes(320, 180, "blue"),
            },
        ]
    )
    expect(page.locator("#original-size")).to_have_text("666 × 1000 px")
    page.get_by_role("button", name="landscape.png", exact=True).click()
    expect(page.locator("#original-size")).to_have_text("320 × 180 px")
    expect(page.locator('[data-scale-size="2"]')).to_have_text("640 × 360 px")
    expect(page.locator('[data-scale-size="4"]')).to_have_text("1280 × 720 px")
    expect(
        page.get_by_role("button", name="landscape.png", exact=True)
    ).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#selected-count")).to_have_text("2 ảnh đã chọn")


def test_remove_selected_files_updates_preview_and_upload(ui):
    page, state = ui
    files = [
        {"name": name, "mimeType": "image/png", "buffer": image_bytes(w, 100, "red")}
        for name, w in [("first.png", 200), ("second.png", 300), ("third.png", 400)]
    ]
    page.locator("#files").set_input_files(files)
    expect(page.locator("#original-size")).to_have_text("200 × 100 px")
    page.get_by_role("button", name="Bỏ chọn second.png", exact=True).click()
    expect(page.locator("#selected-count")).to_have_text("2 ảnh đã chọn")
    expect(page.locator("#original-size")).to_have_text("200 × 100 px")
    page.get_by_role("button", name="Bỏ chọn first.png", exact=True).click()
    expect(page.locator("#original-size")).to_have_text("400 × 100 px")
    page.locator("#submit").click()
    assert b'filename="third.png"' in state["posts"][0]
    assert b'filename="first.png"' not in state["posts"][0]
    assert b'filename="second.png"' not in state["posts"][0]
    page.locator("#files").set_input_files(files[0])
    page.get_by_role("button", name="Bỏ chọn first.png", exact=True).click()
    expect(page.locator("#selection")).to_be_hidden()
    expect(page.locator("#submit")).to_be_disabled()
    expect(page.locator("#original-size")).to_have_text("Chọn ảnh để xem")
    page.locator("#files").set_input_files(files)
    page.get_by_role("button", name="Bỏ chọn tất cả", exact=True).click()
    expect(page.locator("#selection")).to_be_hidden()
    expect(page.locator("#submit")).to_be_disabled()


def completed_job(
    job_id="result-one", filename="portrait.png", width=666, height=1000
):
    return {
        "id": job_id,
        "filename": filename,
        "width": width,
        "height": height,
        "scale": 2,
        "preset": "photo",
        "status": "done",
        "seconds": 6.7,
        "error": None,
        "source_url": f"/api/jobs/{job_id}/source",
        "download_url": f"/api/jobs/{job_id}/download",
    }


def test_remove_queue_job_clears_preview_and_keeps_running_job(ui):
    page, state = ui
    running = {**completed_job("running"), "status": "running", "download_url": None}
    pending = {
        **completed_job("pending", "pending.png"),
        "status": "queued",
        "download_url": None,
    }
    state["jobs"] = [running, pending, completed_job()]
    page.locator(".thumbnail-button").last.click()
    expect(page.locator("#preview-download")).to_be_visible()
    expect(page.locator(".remove-job").first).to_be_disabled()
    state["remove_error"] = True
    page.get_by_role("button", name="Bỏ khỏi hàng đợi pending.png", exact=True).click()
    expect(page.locator("#notice")).to_contain_text("GPU")
    expect(page.locator(".job")).to_have_count(3)
    state["remove_error"] = False
    page.get_by_role("button", name="Bỏ khỏi hàng đợi pending.png", exact=True).click()
    expect(page.locator(".job")).to_have_count(2)
    page.locator(".remove-job").last.click()
    expect(page.locator(".job")).to_have_count(1)
    expect(page.locator("#preview-download")).to_be_hidden()
    expect(page.locator("#preview-placeholder")).to_be_visible()
    page.reload()
    expect(page.locator(".job")).to_have_count(1)
    expect(page.locator(".remove-job")).to_be_disabled()


def test_inline_comparison_prioritizes_images_and_supports_drag_and_keyboard(ui):
    page, state = ui
    state["jobs"] = [completed_job()]
    page.locator(".thumbnail-button").first.click()
    expect(page.locator("#preview-panel")).to_be_visible()
    expect(page.locator("dialog")).to_have_count(0)
    expect(page.locator("#compare-range")).to_be_enabled()
    expect(page.locator("#split-divider")).to_be_visible()
    expect(page.locator(".image-label")).to_have_count(0)
    expect(page.locator("#comparison-help")).to_be_empty()
    page.wait_for_function(
        'document.querySelector("#before").complete && document.querySelector("#after").naturalWidth > 0'
    )
    panel = page.locator("#preview-panel").bounding_box()
    settings = page.locator(".settings").bounding_box()
    assert panel["width"] > settings["width"] * 1.5
    frame = page.locator(".comparison").bounding_box()
    slider = page.locator("#compare-range").bounding_box()
    assert (
        frame["x"] <= slider["x"]
        and frame["y"] <= slider["y"] < frame["y"] + frame["height"]
    )
    page.mouse.move(
        frame["x"] + frame["width"] * 0.5, frame["y"] + frame["height"] * 0.5
    )
    page.mouse.down()
    page.mouse.move(
        frame["x"] + frame["width"] * 0.8, frame["y"] + frame["height"] * 0.5, steps=8
    )
    page.mouse.up()
    assert 78 <= int(page.locator("#compare-range").input_value()) <= 82
    page.locator("#compare-range").focus()
    page.keyboard.press("Home")
    expect(page.locator("#compare-range")).to_have_value("0")
    with Image.open(BytesIO(page.locator(".comparison").screenshot())) as screenshot:
        assert screenshot.getpixel((screenshot.width // 2, screenshot.height // 2))[
            :3
        ] == (34, 102, 204)
    page.keyboard.press("End")
    expect(page.locator("#compare-range")).to_have_value("100")
    with Image.open(BytesIO(page.locator(".comparison").screenshot())) as screenshot:
        assert screenshot.getpixel((screenshot.width // 2, screenshot.height // 2))[
            :3
        ] == (221, 51, 68)


def test_local_preview_never_reuses_a_previous_result_and_polling_keeps_focus(ui):
    page, state = ui
    state["jobs"] = [completed_job()]
    page.wait_for_function('document.querySelector("#after").naturalWidth > 0')
    page.locator("#files").set_input_files(
        {
            "name": "new-landscape.png",
            "mimeType": "image/png",
            "buffer": image_bytes(320, 180, "green"),
        }
    )
    expect(page.locator("#preview-name")).to_have_text("new-landscape.png")
    expect(page.locator("#original-size")).to_have_text("320 × 180 px")
    expect(page.locator("#after")).to_be_hidden()
    expect(page.locator("#compare-range")).to_be_disabled()
    assert page.locator("#before").get_attribute("src").startswith("blob:")
    state["jobs"].append(completed_job("another", "other-result.png"))
    expect(page.locator(".job")).to_have_count(2, timeout=5000)
    expect(page.locator("#preview-name")).to_have_text("new-landscape.png")
    assert page.locator("#before").get_attribute("src").startswith("blob:")


def test_clearing_selection_clears_the_local_image_and_dimensions(ui):
    page, _ = ui
    page.locator("#files").set_input_files(
        {
            "name": "temporary.png",
            "mimeType": "image/png",
            "buffer": image_bytes(120, 80, "green"),
        }
    )
    expect(page.locator("#original-size")).to_have_text("120 × 80 px")
    expect(page.locator(".comparison")).to_be_visible()
    page.get_by_role("button", name="Bỏ chọn tất cả", exact=True).click()
    expect(page.locator("#original-size")).to_have_text("Chọn ảnh để xem")
    expect(page.locator(".comparison")).to_be_hidden()
    expect(page.locator("#preview-placeholder")).to_be_visible()
    expect(page.locator("#submit")).to_be_disabled()


def test_touch_position_and_poll_updates_preserve_the_comparison(ui):
    page, state = ui
    state["jobs"] = [completed_job()]
    page.locator(".thumbnail-button").first.click()
    page.set_viewport_size({"width": 375, "height": 850})
    expect(page.locator("#compare-range")).to_be_enabled()
    page.locator(".comparison").scroll_into_view_if_needed()
    frame = page.locator(".comparison").bounding_box()
    page.touchscreen.tap(
        frame["x"] + frame["width"] * 0.23,
        frame["y"] + frame["height"] * 0.5,
    )
    value = page.locator("#compare-range").input_value()
    assert 21 <= int(value) <= 25
    state["jobs"].append(completed_job("second", "next-result.png"))
    expect(page.locator(".job")).to_have_count(2)
    expect(page.locator("#compare-range")).to_have_value(value)
    expect(page.locator("#preview-name")).to_have_text("portrait.png")
    for width in (375, 768, 1024, 1440):
        page.set_viewport_size({"width": width, "height": 900})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert (
            page.locator("#before").evaluate("(el) => getComputedStyle(el).objectFit")
            == "contain"
        )
        if width <= 800:
            preview = page.locator("#preview-panel").bounding_box()
            settings = page.locator(".settings").bounding_box()
            assert preview["y"] < settings["y"]


def test_pending_job_only_shows_source_then_reveals_its_own_result(ui):
    page, state = ui
    job = completed_job()
    job.update(status="queued", download_url=None, seconds=None)
    state["jobs"] = [job]
    page.locator("#files").set_input_files(
        {
            "name": "portrait.png",
            "mimeType": "image/png",
            "buffer": image_bytes(666, 1000, "red"),
        }
    )
    expect(page.locator("#original-size")).to_have_text("666 × 1000 px")
    page.locator("#submit").click()
    expect(page.locator("#preview-name")).to_have_text("portrait.png")
    expect(page.locator("#after")).to_be_hidden()
    expect(page.locator("#compare-range")).to_be_disabled()
    job.update(status="done", download_url="/api/jobs/result-one/download", seconds=6.7)
    expect(page.locator("#compare-range")).to_be_enabled(timeout=5000)
    expect(page.locator("#after")).to_be_visible()
    expect(page.locator("#preview-meta")).to_have_text("Kết quả 2× · 1332 × 2000 px")
    expect(page.locator("#preview-download")).to_have_attribute(
        "href", "/api/jobs/result-one/download?scale=2"
    )


def test_photo_scale_hint_explains_useful_size_and_eightfold_cost(ui):
    page, _ = ui
    page.locator("#preset").select_option("photo")
    page.locator("#scale-2").check()
    expect(page.locator("#scale-hint")).to_contain_text("Nên bắt đầu ở 2×")
    page.locator("#scale-8").check()
    expect(page.locator("#scale-hint")).to_contain_text("gấp 16 lần 2×")
    expect(page.locator("#scale-hint")).to_contain_text("không thêm bước phục hồi")


def test_type_help_and_exif_oriented_dimensions(ui):
    page, _ = ui
    page.locator("#preset").select_option("photo")
    expect(page.locator("#preset-help")).to_contain_text("máy ảnh")
    page.locator("#preset").select_option("anime")
    expect(page.locator("#preset-help")).to_contain_text("ảnh vẽ, truyện tranh")
    image = Image.new("RGB", (16, 12), "green")
    exif = Image.Exif()
    exif[274] = 6
    stream = BytesIO()
    image.save(stream, "JPEG", exif=exif)
    page.locator("#files").set_input_files(
        {"name": "rotated.jpg", "mimeType": "image/jpeg", "buffer": stream.getvalue()}
    )
    expect(page.locator("#original-size")).to_have_text("12 × 16 px")
    expect(page.locator('[data-scale-size="2"]')).to_have_text("24 × 32 px")


def test_compare_can_retry_a_preview_that_failed_to_load(ui):
    page, state = ui
    state["image_error"] = True
    state["jobs"] = [completed_job()]
    expect(page.locator("#preview-placeholder")).to_contain_text(
        "Không tải được ảnh", timeout=5000
    )
    state["image_error"] = False
    page.locator(".thumbnail-button").first.click()
    expect(page.locator(".comparison")).to_be_visible()
    expect(page.locator("#compare-range")).to_be_enabled()


def test_frontend_module_files_have_a_browser_supported_javascript_mime_type():
    from fastapi.testclient import TestClient
    from app.main import create_app

    client = TestClient(create_app(), base_url="http://127.0.0.1")
    try:
        for name in ("app.js", "preview.js"):
            response = client.get("/assets/" + name)
            assert response.status_code == 200
            assert response.headers["content-type"].split(";")[0] in (
                "text/javascript",
                "application/javascript",
                "application/x-javascript",
            )
    finally:
        client.close()


def test_index_and_logo_serve_haups_branding():
    from fastapi.testclient import TestClient
    from app.main import create_app

    client = TestClient(create_app(), base_url="http://127.0.0.1")
    try:
        index = client.get("/")
        assert index.status_code == 200
        assert "<title>HaUps" in index.text
        assert 'href="/assets/logo.svg"' in index.text

        logo = client.get("/assets/logo.svg")
        assert logo.status_code == 200
        assert "svg" in logo.headers["content-type"].lower()
    finally:
        client.close()


def test_zoom_fits_image_and_centers_each_level(ui):
    page, state = ui
    state["jobs"] = [completed_job()]
    page.locator(".thumbnail-button").first.click()
    expect(page.locator("#zoom-value")).to_have_text("100%")
    viewport = page.locator("#image-viewport")
    stage = page.locator(".comparison").bounding_box()
    vp_box = viewport.bounding_box()
    assert stage["height"] <= vp_box["height"] + 1
    assert stage["width"] <= vp_box["width"] + 1
    viewport.focus()
    for _ in range(2):
        page.keyboard.press("+")
    expect(page.locator("#zoom-value")).to_have_text("150%")
    enlarged = page.locator(".comparison").bounding_box()
    assert abs(enlarged["width"] / stage["width"] - 1.5) < 0.01
    assert viewport.evaluate(
        "el => Math.abs(el.scrollTop - (el.scrollHeight - el.clientHeight) / 2) < 2"
    )
    for _ in range(6):
        page.keyboard.press("+")
    expect(page.locator("#zoom-value")).to_have_text("300%")
    assert viewport.evaluate(
        "el => Math.abs(el.scrollLeft - (el.scrollWidth - el.clientWidth) / 2) < 2"
    )
    for w, h in [(800, 200), (200, 800), (400, 400)]:
        page.locator("#files").set_input_files(
            {
                "name": "shape.png",
                "mimeType": "image/png",
                "buffer": image_bytes(w, h, "red"),
            }
        )
        expect(page.locator("#zoom-value")).to_have_text("100%")
        expect(page.locator("#preview-name")).to_have_text("shape.png")
        page.wait_for_function(
            "document.querySelector('#before').naturalWidth === " + str(w)
        )
        frame = page.locator(".comparison").bounding_box()
        assert abs(frame["width"] / frame["height"] - w / h) < 0.02


def test_result_scale_switches_cached_variants_and_requests_missing_scale(ui):
    page, state = ui
    state["jobs"] = [{**completed_job(), "available_scales": [2]}]
    page.locator(".filename.open-job").click()
    expect(page.locator("#compare-range")).to_be_enabled()
    page.locator("#scale-4").check()
    expect(page.locator("#preview-meta")).to_contain_text("4×")
    expect(page.locator("#preview-download")).to_have_attribute(
        "href", "/api/jobs/result-one/download?scale=4"
    )
    page.locator("#scale-8").check()
    expect(page.locator("#preview-meta")).to_contain_text("8×")
    page.locator("#scale-2").check()
    expect(page.locator("#preview-meta")).to_contain_text("2×")
    assert state["scales"] == [4, 8]
    expect(page.get_by_role("button", name="So sánh", exact=True)).to_have_count(0)


def test_left_drag_pans_right_click_positions_and_handle_drags(ui):
    page, state = ui
    state["jobs"] = [completed_job()]
    page.locator(".thumbnail-button").first.click()
    expect(page.locator("#compare-range")).to_be_enabled()
    viewport = page.locator("#image-viewport")
    bounds = viewport.bounding_box()
    viewport.focus()
    for _ in range(4):
        page.keyboard.press("+")
    expect(page.locator("#zoom-value")).to_have_text("200%")
    page.locator("#compare-range").fill("80")
    scroll_before = viewport.evaluate("el => el.scrollTop")
    x, y = bounds["x"] + 80, bounds["y"] + 120
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x, y - 80, steps=5)
    page.mouse.up()
    assert viewport.evaluate("el => el.scrollTop") - scroll_before >= 75
    expect(page.locator("#compare-range")).to_have_value("80")
    frame = page.locator(".comparison").bounding_box()
    right_x = frame["x"] + 100
    page.mouse.click(right_x, y, button="right")
    assert (
        abs(
            int(page.locator("#compare-range").input_value())
            - (right_x - frame["x"]) * 100 / frame["width"]
        )
        < 2
    )
    page.get_by_role("button", name="Vừa khung", exact=True).click()
    expect(page.locator("#zoom-value")).to_have_text("100%")
    handle = page.locator("#split-divider span").bounding_box()
    page.mouse.move(
        handle["x"] + handle["width"] / 2, handle["y"] + handle["height"] / 2
    )
    page.mouse.down()
    page.mouse.move(
        handle["x"] + handle["width"] / 2 + 80,
        handle["y"] + handle["height"] / 2,
        steps=5,
    )
    page.mouse.up()
    assert int(page.locator("#compare-range").input_value()) > 20


def test_anime_batch_has_no_detail_checkbox_and_submits_cleanly(ui):
    page, state = ui
    removed_id = "preserve" + "-details"
    assert removed_id not in page.content()
    expect(page.locator("#detail-option")).to_have_count(0)
    page.locator("#preset").select_option("anime")
    page.locator("#files").set_input_files(
        {
            "name": "first.png",
            "mimeType": "image/png",
            "buffer": image_bytes(10, 10, "red"),
        }
    )
    with page.expect_response("**/api/batches"):
        page.locator("#submit").click()
    expect(page.locator("#selection")).to_be_hidden()
    assert b'name="preset"\r\n\r\nanime' in state["posts"][0]
    assert ("preserve" + "_details").encode() not in state["posts"][0]

    state["jobs"] = [{**completed_job(), "preset": "anime"}]
    page.reload()
    expect(page.locator(".job-meta")).to_have_text(
        "666 × 1000 → 1332 × 2000 · Anime"
    )


def test_wallpaper_uses_current_image_without_description(ui):
    page, state = ui
    job = {**completed_job("result-anime", "illustration.png"), "preset": "anime"}
    state["jobs"] = [job]
    state["outpainting"] = True

    page.reload()
    page.locator(".thumbnail-button").first.click()

    expect(page.locator("#wallpaper-form")).to_be_visible()
    expect(page.locator('label[for="wallpaper-prompt"]')).to_be_visible()
    expect(page.locator("#wallpaper-prompt")).to_be_visible()
    expect(page.locator("#wallpaper-prompt")).to_have_attribute("maxlength", "500")
    expect(page.locator("#wallpaper-prompt-hint")).to_have_text(
        "Để trống: AI tự đoán nền từ ảnh."
    )
    expect(page.locator("#wallpaper-create")).to_be_visible()
    expect(page.locator("#wallpaper-create")).to_have_text("Tạo hình nền AI")
    expect(page.locator("#wallpaper-result")).to_be_hidden()

    with page.expect_response(
        lambda r: r.request.method == "POST"
        and f"/api/jobs/{job['id']}/wallpaper" in r.url
    ):
        page.locator("#wallpaper-create").click()

    assert len(state["wallpaper_posts"]) == 1
    assert state["wallpaper_posts"][0] == "prompt="

    expect(page.locator("#wallpaper-create")).to_be_disabled()
    expect(page.locator("#wallpaper-prompt")).to_be_disabled()
    expect(page.locator("#wallpaper-status")).to_have_text("Đang chờ tạo nền AI…")
    expect(page.locator("#wallpaper-status")).to_have_attribute(
        "title", "Đang chờ tạo nền AI…"
    )

    job["status"] = "done"
    job["wallpaper_available"] = True
    job["wallpaper_completed_prompt"] = ""

    page.reload()
    page.locator(".thumbnail-button").first.click()

    expect(page.locator("#wallpaper-create")).to_be_visible()
    expect(page.locator("#wallpaper-create")).to_have_text("Tạo lại")
    expect(page.locator("#wallpaper-create")).to_be_disabled()
    expect(page.locator("#wallpaper-result")).to_be_visible()
    expect(page.locator("#wallpaper-download")).to_have_text("Tải hình nền")
    expect(page.locator("#wallpaper-download")).to_have_attribute(
        "href", f"/api/jobs/{job['id']}/wallpaper"
    )
    expect(page.locator("#wallpaper-open")).to_have_text("Mở")
    expect(page.locator("#wallpaper-open")).to_have_attribute(
        "href", re.compile(rf"^/api/jobs/{job['id']}/wallpaper\?inline=1")
    )

    expect(page.locator("#preview-download")).to_be_visible()
    expect(page.locator("#preview-download")).to_have_attribute(
        "href", f"/api/jobs/{job['id']}/download?scale={job['scale']}"
    )
    expect(page.locator(".job .download").first).to_be_visible()
    expect(page.locator(".job .download").first).to_have_attribute(
        "href", job["download_url"]
    )
    expect(page.locator("#before")).to_have_attribute("src", job["source_url"])


def test_wallpaper_prompt_submission_and_regenerate_button_state(ui):
    page, state = ui
    job1 = {
        **completed_job("job-wp-1", "first.png"),
        "preset": "anime",
        "available_scales": [2],
    }
    job2 = {
        **completed_job("job-wp-2", "second.png"),
        "preset": "photo",
        "available_scales": [2],
        "wallpaper_prompt": "forest trail",
    }
    state["jobs"] = [job1, job2]
    state["outpainting"] = True

    page.reload()
    page.locator(".thumbnail-button").first.click()

    prompt_input = page.locator("#wallpaper-prompt")
    create_btn = page.locator("#wallpaper-create")
    expect(prompt_input).to_have_value("")
    expect(create_btn).to_have_text("Tạo hình nền AI")
    expect(create_btn).to_be_enabled()

    prompt_input.fill("  beach at sunset  ")
    # Polling for the same job must not overwrite while typing
    state["jobs"][0]["seconds"] = 7.1
    page.wait_for_timeout(1400)
    expect(prompt_input).to_have_value("  beach at sunset  ")

    with page.expect_response(
        lambda r: r.request.method == "POST"
        and f"/api/jobs/{job1['id']}/wallpaper" in r.url
    ):
        prompt_input.press("Enter")

    assert len(state["wallpaper_posts"]) == 1
    assert "prompt=beach at sunset" in state["wallpaper_posts"][0]
    expect(create_btn).to_be_disabled()
    expect(prompt_input).to_be_disabled()

    # Complete the wallpaper with prompt 'beach at sunset'
    job1["status"] = "done"
    job1["wallpaper_available"] = True
    job1["wallpaper_completed_prompt"] = "beach at sunset"

    expect(create_btn).to_have_text("Tạo lại", timeout=5000)
    expect(create_btn).to_be_disabled()
    expect(prompt_input).to_be_enabled()

    # Changing prompt enables 'Tạo lại'; matching completed prompt disables it again
    prompt_input.fill("misty mountains")
    expect(create_btn).to_be_enabled()
    prompt_input.fill("  beach at sunset ")
    expect(create_btn).to_be_disabled()
    prompt_input.fill("")
    expect(create_btn).to_be_enabled()

    # Switching focused job updates input to that job's wallpaper_prompt
    page.locator(".thumbnail-button").last.click()
    expect(prompt_input).to_have_value("forest trail")


def test_header_zoom_controls_anchor_wheel_dblclick_and_keyboard(ui):
    page, _ = ui
    page.locator("#files").set_input_files(
        {
            "name": "wide.png",
            "mimeType": "image/png",
            "buffer": image_bytes(2000, 1000, "teal"),
        }
    )
    expect(page.locator(".comparison")).to_be_visible()
    zoom_value = page.locator("#zoom-value")
    expect(zoom_value).to_have_text("100%")

    viewport = page.locator("#image-viewport")
    vp_box = viewport.bounding_box()

    page.get_by_role("button", name="Kích thước thật", exact=True).click()
    assert abs(page.locator(".comparison").bounding_box()["width"] - 2000) < 2

    page.get_by_role("button", name="Vừa khung", exact=True).click()
    expect(zoom_value).to_have_text("100%")

    # Double-click toggles 100 <-> 200 anchored at cursor
    cursor_x = vp_box["x"] + vp_box["width"] * 0.25
    cursor_y = vp_box["y"] + vp_box["height"] * 0.5
    page.mouse.dblclick(cursor_x, cursor_y)
    expect(zoom_value).to_have_text("200%")
    rel_x = viewport.evaluate(
        f"el => (el.scrollLeft + {vp_box['width'] * 0.25}) / parseFloat(document.querySelector('.comparison').style.width)"
    )
    assert abs(rel_x - 0.25) < 0.03
    page.mouse.dblclick(cursor_x, cursor_y)
    expect(zoom_value).to_have_text("100%")

    # Mouse wheel zooms by 25 per notch anchored at cursor on desktop width
    page.mouse.move(cursor_x, cursor_y)
    page.mouse.wheel(0, -100)
    expect(zoom_value).to_have_text("125%")
    page.mouse.wheel(0, 100)
    expect(zoom_value).to_have_text("100%")

    # Keyboard shortcuts on focused viewport anchor at viewport center
    viewport.focus()
    for _ in range(4):
        page.keyboard.press("+")
    expect(zoom_value).to_have_text("200%")
    viewport.evaluate("el => { el.scrollLeft = 60; el.scrollTop = 40; }")
    before_center_x = viewport.evaluate(
        "el => (el.scrollLeft + el.clientWidth / 2) / parseFloat(document.querySelector('.comparison').style.width)"
    )
    page.keyboard.press("+")
    expect(zoom_value).to_have_text("225%")
    after_center_x = viewport.evaluate(
        "el => (el.scrollLeft + el.clientWidth / 2) / parseFloat(document.querySelector('.comparison').style.width)"
    )
    assert abs(before_center_x - after_center_x) < 0.01
    page.keyboard.press("-")
    expect(zoom_value).to_have_text("200%")
    left_before = viewport.evaluate("el => el.scrollLeft")
    client_w = viewport.evaluate("el => el.clientWidth")
    page.keyboard.press("ArrowRight")
    left_after = viewport.evaluate("el => el.scrollLeft")
    assert abs((left_after - left_before) - round(client_w * 0.1)) <= 2
    page.keyboard.press("0")
    expect(zoom_value).to_have_text("100%")


def test_bulk_download_and_updated_limits(ui):
    page, state = ui
    expect(page.locator(".limits")).to_have_text(
        "PNG/JPG/WebP · ≤ 50 MB · ≤ 16 MP · ≤ 20 ảnh/lần"
    )
    expect(page.locator("#wallpaper-form h3")).to_have_text(
        "Hình nền AI · 3840 × 2160"
    )
    expect(page.locator("#wallpaper-open")).to_have_attribute(
        "title", "Xem hình nền AI 3840 × 2160 ở tab mới"
    )

    job1 = {
        **completed_job("job-1", "first.png"),
        "wallpaper_available": True,
    }
    job2 = completed_job("job-2", "second.png")
    queued = {
        **completed_job("job-3", "queued.png"),
        "status": "queued",
        "download_url": None,
    }
    state["jobs"] = [job1, job2, queued]
    page.reload()

    expect(page.locator(".job")).to_have_count(3)
    expect(page.locator(".job-select")).to_have_count(2)
    expect(page.get_by_role("checkbox", name="Chọn first.png")).to_be_visible()
    expect(page.get_by_role("checkbox", name="Chọn second.png")).to_be_visible()
    expect(page.get_by_role("checkbox", name="Kèm hình nền AI")).to_be_checked()

    download_selected = page.get_by_role("button", name="Tải đã chọn", exact=True)
    download_all = page.get_by_role("button", name="Tải tất cả", exact=True)
    select_all = page.get_by_role("checkbox", name="Chọn tất cả", exact=True)
    expect(download_selected).to_be_disabled()
    expect(download_all).to_be_enabled()

    downloads = []
    page.on("download", lambda d: downloads.append(d.url))
    download_all.click()
    for _ in range(50):
        if len(downloads) >= 3:
            break
        page.wait_for_timeout(50)
    assert len(downloads) == 3
    assert any(u.endswith("/api/jobs/job-1/download") for u in downloads)
    assert any(u.endswith("/api/jobs/job-1/wallpaper") for u in downloads)
    assert any(u.endswith("/api/jobs/job-2/download") for u in downloads)

    downloads.clear()
    page.get_by_role("checkbox", name="Chọn first.png").check()
    assert select_all.evaluate("el => el.indeterminate") is True
    expect(download_selected).to_be_enabled()
    page.get_by_role("checkbox", name="Kèm hình nền AI").uncheck()
    with page.expect_download() as dl_info:
        download_selected.click()
    page.wait_for_timeout(400)
    assert len(downloads) == 1
    assert dl_info.value.url.endswith("/api/jobs/job-1/download")


def test_viewer_header_two_line_title_and_footer_action_groups(ui):
    page, state = ui
    state["outpainting"] = True

    # Local preview of an unprocessed file shows no footer actions
    page.locator("#files").set_input_files(
        {
            "name": "local-raw.png",
            "mimeType": "image/png",
            "buffer": image_bytes(640, 480, "orange"),
        }
    )
    expect(page.locator("#preview-name")).to_have_text("local-raw.png")
    expect(page.locator("#preview-meta")).to_have_text(
        "640 × 480 px · Ảnh gốc, chưa xử lý"
    )
    expect(page.locator("#preview-meta")).to_have_attribute(
        "title", "640 × 480 px · Ảnh gốc, chưa xử lý"
    )
    expect(page.locator("#preview-footer")).to_be_hidden()
    expect(page.locator("#result-group")).to_be_hidden()
    expect(page.locator("#wallpaper-form")).to_be_hidden()

    # Header has ONLY two-line title block on left and 3 zoom controls on right
    assert page.locator(".preview-heading").evaluate(
        """el => {
            const titleIds = [...el.querySelectorAll('.preview-title > *')].map(n => n.id);
            const controlIds = [...el.querySelectorAll('.viewer-controls > *')].map(n => n.id);
            return JSON.stringify({titleIds, controlIds});
        }"""
    ) == '{"titleIds":["preview-name","preview-meta"],"controlIds":["zoom-value","zoom-fit","zoom-actual"]}'
    name_box = page.locator("#preview-name").bounding_box()
    meta_box = page.locator("#preview-meta").bounding_box()
    assert meta_box["y"] >= name_box["y"] + name_box["height"] - 2

    # Completed job reveals footer below image viewport with both groups
    page.get_by_role("button", name="Bỏ chọn tất cả", exact=True).click()
    job = completed_job("job-footer", "completed-artwork.png", width=875, height=1280)
    state["jobs"] = [job]
    page.reload()
    page.locator(".thumbnail-button").first.click()

    expect(page.locator("#preview-footer")).to_be_visible()
    expect(page.locator("#result-group")).to_be_visible()
    expect(page.locator("#result-group .group-label")).to_have_text("Kết quả")
    expect(page.locator("#preview-download")).to_have_text("Tải PNG")
    expect(page.locator("#source-link")).to_have_text("Mở ảnh gốc")
    expect(page.locator("#result-link")).to_have_text("Kích thước thật")
    expect(page.locator("#result-link")).to_have_attribute(
        "title", "Mở kết quả ở kích thước thật"
    )
    expect(page.locator("#result-link")).to_have_attribute(
        "aria-label", "Mở kết quả ở kích thước thật"
    )

    expect(page.locator("#wallpaper-form")).to_be_visible()
    expect(page.locator("#wallpaper-form .group-label")).to_have_text(
        "Hình nền AI · 3840 × 2160"
    )
    expect(page.locator("#wallpaper-create")).to_be_visible()
    expect(page.locator("#wallpaper-result")).to_be_hidden()

    vp_box = page.locator("#image-viewport").bounding_box()
    footer_box = page.locator("#preview-footer").bounding_box()
    assert footer_box["y"] >= vp_box["y"] + vp_box["height"] - 1

    # At 1280x720, no page scroll and no button label truncation
    page.set_viewport_size({"width": 1280, "height": 720})
    assert page.evaluate(
        "document.scrollingElement.scrollHeight <= window.innerHeight + 1"
    )
    truncated = page.evaluate(
        """() => [...document.querySelectorAll('button:not(.filename):not(.file-choice), .button, .footer-link')]
            .filter(el => el.offsetParent !== null)
            .filter(el => el.scrollWidth > el.clientWidth + 1 || el.getBoundingClientRect().height < 24)
            .map(el => el.id || el.textContent.trim())"""
    )
    assert truncated == []


def test_app_shell_no_page_scroll_tall_portrait_fit_and_fixed_viewer_box(ui):
    page, state = ui
    portrait = completed_job("tall-job", "tall.png", width=700, height=2000)
    landscape = completed_job("wide-job", "wide.png", width=2000, height=700)
    state["jobs"] = [portrait, landscape]
    page.set_viewport_size({"width": 1440, "height": 900})
    page.reload()

    page.locator(".thumbnail-button").first.click()
    expect(page.locator("#preview-name")).to_have_text("tall.png")
    page.wait_for_function(
        "document.querySelector('#after').naturalHeight === 2000"
    )
    assert page.evaluate(
        "document.scrollingElement.scrollHeight <= window.innerHeight + 1"
    )
    vp_portrait = page.locator("#image-viewport").bounding_box()
    stage_portrait = page.locator(".comparison").bounding_box()
    assert stage_portrait["width"] <= vp_portrait["width"] + 1
    assert stage_portrait["height"] <= vp_portrait["height"] + 1
    assert stage_portrait["x"] >= vp_portrait["x"] - 1
    assert stage_portrait["y"] >= vp_portrait["y"] - 1
    assert (
        stage_portrait["x"] + stage_portrait["width"]
        <= vp_portrait["x"] + vp_portrait["width"] + 1
    )
    assert (
        stage_portrait["y"] + stage_portrait["height"]
        <= vp_portrait["y"] + vp_portrait["height"] + 1
    )

    # Viewer box size does not change between portrait and landscape results
    page.locator(".thumbnail-button").last.click()
    expect(page.locator("#preview-name")).to_have_text("wide.png")
    page.wait_for_function(
        "document.querySelector('#after').naturalWidth === 2000"
    )
    vp_landscape = page.locator("#image-viewport").bounding_box()
    assert abs(vp_portrait["width"] - vp_landscape["width"]) < 1
    assert abs(vp_portrait["height"] - vp_landscape["height"]) < 1

    # Also fits one screen at 1280x720
    page.set_viewport_size({"width": 1280, "height": 720})
    assert page.evaluate(
        "document.scrollingElement.scrollHeight <= window.innerHeight + 1"
    )


def test_no_zoom_slider_and_no_button_overlaps_placeholder(ui):
    page, _ = ui
    expect(page.locator("#zoom")).to_have_count(0)
    expect(page.locator(".zoom-toolbar")).to_have_count(0)
    expect(page.locator("#preview-placeholder")).to_be_visible()
    ph = page.locator("#preview-placeholder").bounding_box()
    buttons = page.locator("button:visible").all()
    assert len(buttons) > 0
    for btn in buttons:
        box = btn.bounding_box()
        if not box:
            continue
        intersects = not (
            box["x"] + box["width"] <= ph["x"]
            or ph["x"] + ph["width"] <= box["x"]
            or box["y"] + box["height"] <= ph["y"]
            or ph["y"] + ph["height"] <= box["y"]
        )
        assert not intersects, f"Button {btn.inner_text()} overlaps placeholder"


def test_wheel_zooms_on_desktop_and_ctrl_wheel_with_hint_on_stacked_layout(ui):
    page, _ = ui
    page.set_viewport_size({"width": 1440, "height": 900})
    page.locator("#files").set_input_files(
        {
            "name": "sample.png",
            "mimeType": "image/png",
            "buffer": image_bytes(800, 600, "purple"),
        }
    )
    expect(page.locator(".comparison")).to_be_visible()
    vp = page.locator("#image-viewport").bounding_box()
    page.mouse.move(vp["x"] + vp["width"] / 2, vp["y"] + vp["height"] / 2)
    page.mouse.wheel(0, -100)
    expect(page.locator("#zoom-value")).to_have_text("125%")
    page.get_by_role("button", name="Vừa khung", exact=True).click()
    expect(page.locator("#zoom-value")).to_have_text("100%")

    # At 700px width, plain wheel does not zoom and shows transient hint; Ctrl+wheel zooms
    page.set_viewport_size({"width": 700, "height": 900})
    vp_mobile = page.locator("#image-viewport").bounding_box()
    page.mouse.move(
        vp_mobile["x"] + vp_mobile["width"] / 2,
        vp_mobile["y"] + vp_mobile["height"] / 2,
    )
    page.mouse.wheel(0, -100)
    expect(page.locator("#zoom-value")).to_have_text("100%")
    expect(page.locator("#wheel-hint")).to_be_visible()
    expect(page.locator("#wheel-hint")).to_have_text(
        "Ctrl + lăn chuột để phóng to"
    )

    page.keyboard.down("Control")
    page.mouse.wheel(0, -100)
    page.keyboard.up("Control")
    expect(page.locator("#zoom-value")).to_have_text("125%")


def test_queue_list_scrolls_internally_without_page_scroll(ui):
    page, state = ui
    state["jobs"] = [
        completed_job(f"job-{i}", f"image-{i}.png") for i in range(15)
    ]
    page.set_viewport_size({"width": 1440, "height": 900})
    page.reload()
    expect(page.locator(".job")).to_have_count(15)
    assert page.evaluate(
        "document.scrollingElement.scrollHeight <= window.innerHeight + 1"
    )
    assert page.locator("#jobs").evaluate(
        "el => el.scrollHeight > el.clientHeight && el.clientHeight > 0"
    )


def test_wallpaper_viewer_mode_toggle_zoom_and_auto_switch_once(ui):
    page, state = ui
    state["outpainting"] = True
    job = {
        **completed_job("wp-job", "scenery.png", width=800, height=600),
        "available_scales": [2],
    }
    state["jobs"] = [job]
    page.set_viewport_size({"width": 1440, "height": 900})
    page.reload()
    page.locator(".thumbnail-button").first.click()

    upscale_btn = page.get_by_role("button", name="Upscale", exact=True)
    wallpaper_btn = page.get_by_role("button", name="Hình nền AI", exact=True)

    # Toggle is hidden when job has no wallpaper_available
    expect(page.locator("#preview-mode")).to_be_hidden()
    expect(upscale_btn).to_be_hidden()
    expect(wallpaper_btn).to_be_hidden()

    # Simulate wallpaper job running then completing for the currently previewed job
    job["task"] = "wallpaper"
    job["status"] = "running"
    expect(page.locator("#wallpaper-status")).to_have_text(
        "GPU đang tạo nền AI…", timeout=5000
    )
    expect(page.locator("#preview-mode")).to_be_hidden()

    job["status"] = "done"
    job["wallpaper_available"] = True

    # Auto-switches once to 'Hình nền AI' upon completion
    expect(page.locator("#preview-mode")).to_be_visible(timeout=5000)
    expect(wallpaper_btn).to_have_attribute("aria-pressed", "true")
    expect(upscale_btn).to_have_attribute("aria-pressed", "false")
    expect(page.locator("#preview-meta")).to_have_text(
        "Hình nền AI · 3840 × 2160"
    )
    expect(page.locator("#before")).to_have_attribute(
        "src", f"/api/jobs/{job['id']}/wallpaper?inline=1"
    )
    expect(page.locator("#after")).to_be_hidden()
    expect(page.locator("#split-divider")).to_be_hidden()
    expect(page.locator("#compare-range")).to_be_hidden()
    expect(page.locator("#compare-range")).to_be_disabled()

    # User switches back to 'Upscale'; subsequent polls must not force it back
    upscale_btn.click()
    expect(upscale_btn).to_have_attribute("aria-pressed", "true")
    expect(wallpaper_btn).to_have_attribute("aria-pressed", "false")
    expect(page.locator("#preview-meta")).to_have_text(
        "Kết quả 2× · 1600 × 1200 px"
    )
    expect(page.locator("#before")).to_have_attribute("src", job["source_url"])
    expect(page.locator("#after")).to_be_visible()
    expect(page.locator("#split-divider")).to_be_visible()
    expect(page.locator("#compare-range")).to_be_enabled()

    state["jobs"].append(completed_job("second-job", "second.png"))
    expect(page.locator(".job")).to_have_count(2, timeout=5000)
    page.wait_for_timeout(1400)
    expect(upscale_btn).to_have_attribute("aria-pressed", "true")
    expect(wallpaper_btn).to_have_attribute("aria-pressed", "false")
    expect(page.locator("#preview-meta")).to_have_text(
        "Kết quả 2× · 1600 × 1200 px"
    )
    expect(page.locator("#split-divider")).to_be_visible()

    # Clicking 'Hình nền AI' loads wallpaper in viewer with full zoom/pan behaviour
    wallpaper_btn.click()
    expect(wallpaper_btn).to_have_attribute("aria-pressed", "true")
    expect(upscale_btn).to_have_attribute("aria-pressed", "false")
    expect(page.locator("#preview-meta")).to_have_text(
        "Hình nền AI · 3840 × 2160"
    )
    expect(page.locator("#before")).to_have_attribute(
        "src", f"/api/jobs/{job['id']}/wallpaper?inline=1"
    )
    expect(page.locator("#split-divider")).to_be_hidden()
    expect(page.locator("#compare-range")).to_be_hidden()
    expect(page.locator("#compare-range")).to_be_disabled()
    page.wait_for_function(
        "document.querySelector('#before').naturalWidth === 3840 && document.querySelector('#before').naturalHeight === 2160"
    )
    expect(page.locator("#zoom-value")).to_have_text("100%")

    vp = page.locator("#image-viewport").bounding_box()
    stage = page.locator(".comparison").bounding_box()
    assert stage["width"] <= vp["width"] + 1
    assert stage["height"] <= vp["height"] + 1

    page.mouse.move(vp["x"] + vp["width"] / 2, vp["y"] + vp["height"] / 2)
    page.mouse.wheel(0, -100)
    expect(page.locator("#zoom-value")).to_have_text("125%")

    page.get_by_role("button", name="Kích thước thật", exact=True).click()
    stage_actual = page.locator(".comparison").bounding_box()
    assert abs(stage_actual["width"] - 3840) < 2
    assert abs(stage_actual["height"] - 2160) < 2
    fit_ratio = min(vp["width"] / 3840, vp["height"] / 2160)
    expect(page.locator("#zoom-value")).to_have_text(
        f"{round(100 / fit_ratio)}%"
    )

    page.get_by_role("button", name="Vừa khung", exact=True).click()
    expect(page.locator("#zoom-value")).to_have_text("100%")

    assert page.evaluate(
        "document.scrollingElement.scrollHeight <= window.innerHeight + 1"
    )


def test_jobs_api_reports_result_and_wallpaper_file_sizes(tmp_path):
    from time import sleep, time
    from fastapi.testclient import TestClient
    from app.main import create_app

    result_data = image_bytes(40, 40, "blue")
    wallpaper_data = image_bytes(80, 45, "green")

    with TestClient(
        create_app({"data_dir": tmp_path, "gpu_id": 0}),
        base_url="http://127.0.0.1",
    ) as client:
        def fake_process(folder, scale, preset):
            (folder / f"result-{scale}.png").write_bytes(result_data)
            (folder / "result.png").write_bytes(result_data)

        client.app.state.queue.engine.process = fake_process
        response = client.post(
            "/api/batches",
            data={"scale": "2", "preset": "photo"},
            files=[("files", ("sample.png", image_bytes(20, 20, "red"), "image/png"))],
        )
        assert response.status_code == 202
        job_id = response.json()["jobs"][0]["id"]

        deadline = time() + 5
        job = None
        while time() < deadline:
            job = client.get("/api/jobs").json()["jobs"][0]
            if job["status"] == "done":
                break
            sleep(0.02)
        assert job["status"] == "done"
        expected_result_size = (tmp_path / job_id / "result-2.png").stat().st_size
        assert job["result_bytes"] == {"2": expected_result_size}
        assert job["wallpaper_bytes"] is None

        (tmp_path / job_id / "result-2.png").unlink()
        fallback_job = client.get("/api/jobs").json()["jobs"][0]
        assert fallback_job["result_bytes"] == {"2": expected_result_size}

        wp_path = tmp_path / job_id / "wallpaper.png"
        wp_path.write_bytes(wallpaper_data)
        with_wp = client.get("/api/jobs").json()["jobs"][0]
        assert with_wp["result_bytes"] == {"2": expected_result_size}
        assert with_wp["wallpaper_bytes"] == wp_path.stat().st_size


def test_download_buttons_and_bulk_bar_show_formatted_sizes(ui):
    page, state = ui
    state["outpainting"] = True
    job1 = {
        **completed_job("job-size-1", "first.png"),
        "available_scales": [2, 4],
        "result_bytes": {"2": 23456789, "4": 512 * 1024},
        "wallpaper_available": True,
        "wallpaper_bytes": 12687770,
    }
    job2 = {
        **completed_job("job-size-2", "second.png"),
        "available_scales": [2],
        "result_bytes": {"2": 10 * 1024 * 1024},
    }
    state["jobs"] = [job1, job2]
    page.reload()

    page.locator(".thumbnail-button").first.click()
    expect(page.locator("#preview-download")).to_have_text("Tải PNG · 22,4 MB")
    expect(page.locator("#wallpaper-download")).to_have_text(
        "Tải hình nền · 12,1 MB"
    )

    page.locator("#scale-4").check()
    expect(page.locator("#preview-download")).to_have_text("Tải PNG · 512 KB")

    page.get_by_role("checkbox", name="Chọn tất cả", exact=True).check()
    expect(page.locator("#bulk-selected-count")).to_have_text(
        "Đã chọn 2 ảnh · 44,5 MB"
    )

    page.get_by_role("checkbox", name="Kèm hình nền AI").uncheck()
    expect(page.locator("#bulk-selected-count")).to_have_text(
        "Đã chọn 2 ảnh · 32,4 MB"
    )


def test_wallpaper_scene_hint_shows_detected_tags_empty_fallback_and_hides_for_user_prompt(
    ui,
):
    page, state = ui
    state["outpainting"] = True
    job = {
        **completed_job("job-scene", "anime.png"),
        "preset": "anime",
        "available_scales": [2],
        "wallpaper_available": True,
        "wallpaper_completed_prompt": "",
        "wallpaper_scene": "outdoors, sky",
    }
    state["jobs"] = [job]
    page.reload()
    page.locator(".thumbnail-button").first.click()

    scene_el = page.locator("#wallpaper-scene")
    expect(scene_el).to_be_visible()
    expect(scene_el).to_have_text("AI nhận ra: outdoors, sky")
    expect(scene_el).to_have_attribute("title", "AI nhận ra: outdoors, sky")

    job["wallpaper_scene"] = ""
    page.reload()
    page.locator(".thumbnail-button").first.click()
    empty_msg = "AI không nhận ra bối cảnh — nên nhập mô tả rồi bấm Tạo lại."
    expect(scene_el).to_be_visible()
    expect(scene_el).to_have_text(empty_msg)
    expect(scene_el).to_have_attribute("title", empty_msg)

    job["wallpaper_prompt"] = "cherry blossoms"
    job["wallpaper_completed_prompt"] = "cherry blossoms"
    job["wallpaper_scene"] = "outdoors, sky"
    page.reload()
    page.locator(".thumbnail-button").first.click()
    expect(scene_el).to_be_hidden()

