import json
from PIL import Image, ImageChops
import pytest

from app import outpaint_runner, wallpaper
from lab.outpainting import seam_blend_lab


@pytest.mark.parametrize("name", ["run1", "A", "seam_blend-2026", "a" * 64])
def test_validate_run_name_accepts_valid_names(name):
    assert seam_blend_lab.validate_run_name(name) == name


@pytest.mark.parametrize(
    "bad_name",
    ["", "-leading", "_leading", "../escape", "sub/dir", "bad space", "a" * 65],
)
def test_validate_run_name_rejects_invalid_names(tmp_path, bad_name):
    with pytest.raises(ValueError):
        seam_blend_lab.validate_run_name(bad_name)
    with pytest.raises(SystemExit):
        seam_blend_lab.main([bad_name, "--images", str(tmp_path)])


def test_list_images_non_recursive_skips_remove_and_non_images(tmp_path):
    img_dir = tmp_path / "inputs"
    img_dir.mkdir()
    for name in ("b.JPG", "a.png", "c.jpeg", "d.webp"):
        Image.new("RGB", (8, 8), "red").save(img_dir / name)
    (img_dir / "notes.txt").write_text("ignore", encoding="utf-8")
    (img_dir / "meta.json").write_text("{}", encoding="utf-8")

    nested = img_dir / "nested"
    nested.mkdir()
    Image.new("RGB", (8, 8), "blue").save(nested / "inside.png")

    remove_sub = img_dir / "remove"
    remove_sub.mkdir()
    Image.new("RGB", (8, 8), "green").save(remove_sub / "skip.png")

    listed = seam_blend_lab.list_images(img_dir)
    assert [p.name for p in listed] == ["a.png", "b.JPG", "c.jpeg", "d.webp"]
    assert [p.name for p in seam_blend_lab.list_images(img_dir, limit=2)] == [
        "a.png",
        "b.JPG",
    ]
    assert seam_blend_lab.list_images(remove_sub) == []


def test_build_seam_sheet_layout_size():
    widths = [0, 16, 32]
    images_by_width = {
        w: Image.new("RGB", (3840, 2160), (w + 10, 50, 100)) for w in widths
    }
    box = [1312, 0, 2527, 2160]
    sides_ramped = ["left", "right"]

    sheet = seam_blend_lab.build_seam_sheet(
        images_by_width, widths, box, sides_ramped
    )
    expected_w = 3 * 384 + 2 * 4
    expected_h = 4 * 384 + 3 * 4
    assert sheet.size == (expected_w, expected_h)
    assert sheet.getpixel((384, 10)) == (255, 255, 255)
    assert sheet.getpixel((10, 384)) == (255, 255, 255)


def test_main_end_to_end_with_synthetic_image(tmp_path, monkeypatch):
    images_dir = tmp_path / "source_images"
    images_dir.mkdir()
    secret_filename = "secret-character-name.png"
    source_path = images_dir / secret_filename
    source_img = Image.new("RGB", (18, 32), (30, 60, 120))
    source_img.save(source_path, format="PNG")

    out_root = tmp_path / "lab_out"
    out_root.mkdir()
    monkeypatch.setattr(seam_blend_lab, "LAB", out_root)

    def fake_generate(source, preset, prompt=""):
        assert preset == "anime"
        assert prompt == ""
        _, mask = wallpaper.prepare(source)
        return Image.new("RGB", (768, 432), (180, 90, 40)), mask

    def fake_upscale_background(generated, work_dir, preset):
        assert generated.size == (768, 432)
        assert preset == "anime"
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "temp.bin").write_bytes(b"tmp")
        return Image.new("RGB", (3072, 1728), (210, 140, 70))

    monkeypatch.setattr(outpaint_runner, "generate", fake_generate)
    monkeypatch.setattr(
        outpaint_runner, "upscale_background", fake_upscale_background
    )

    summary = seam_blend_lab.main(
        [
            "seam-smoke-01",
            "--images",
            str(images_dir),
            "--limit",
            "1",
            "--widths",
            "0",
            "16",
            "32",
        ]
    )

    run_dir = out_root / "seam-smoke-01"
    case_dir = run_dir / "case-01"
    assert case_dir.is_dir()
    assert not (run_dir / "case-01-esrgan").exists()

    with Image.open(case_dir / "generated.png") as gen_img:
        assert gen_img.size == (768, 432)

    for w in (0, 16, 32):
        with Image.open(case_dir / f"w{w}.png") as w_img:
            assert w_img.size == (3840, 2160)

    with Image.open(case_dir / "w0.png") as w0_img:
        expected_w0 = wallpaper.restore(
            source_img, Image.new("RGB", (3072, 1728), (210, 140, 70))
        )
        assert ImageChops.difference(w0_img, expected_w0).getbbox() is None

    with Image.open(case_dir / "seams.jpg") as seams_img:
        assert seams_img.size == (3 * 384 + 2 * 4, 4 * 384 + 3 * 4)

    records_text = (run_dir / "records.json").read_text(encoding="utf-8")
    assert secret_filename not in records_text
    records_data = json.loads(records_text)
    assert records_data == summary
    assert len(records_data["records"]) == 1
    rec = records_data["records"][0]
    assert rec["case"] == "case-01"
    assert rec["skipped"] is False
    assert rec["source_size"] == [18, 32]
    assert rec["sides_ramped"] == ["left", "right"]
    assert rec["widths"] == [0, 16, 32]
