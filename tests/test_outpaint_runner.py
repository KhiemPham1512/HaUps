import sys
import types

from PIL import Image, ImageChops, ImageFilter, ImageOps
import pytest

from app.engine import EngineError
import app.outpaint_runner
from app.wallpaper import prepare

OFFICIAL_PROMPT_U = (
    "anime illustration, continue the same existing character beyond the cropped edge, "
    "coherent anatomy, matching hair, clothing, line art, colors and lighting, "
    "continuation of the existing scene, empty scene"
)
NEGATIVE_PROMPT_U = (
    "duplicate character, extra face, extra eyes, extra limbs, malformed anatomy, "
    "text, watermark, frame, border, blurry, artifacts"
)


REPAIRED_COLOR = (1, 2, 3)


def _setup_fake_runner_deps(
    monkeypatch,
    engine_raises=False,
    scores=None,
    detached=None,
    refine_error=None,
):
    state = {
        "events": [],
        "call_kwargs": {},
        "engine_calls": [],
        "released": False,
        "pipe": None,
        "gate": None,
        "model_info": None,
        "seeds": [],
        "candidate_colors": [],
        "esrgan_source_colors": [],
        "esrgan_source_sizes": [],
        "session_loads": [],
        "checked_masks": [],
        "prepare_source_sizes": [],
        "restore_source_sizes": [],
        "refine_calls": [],
        "empty_cache_calls": [],
    }
    real_prepare = app.outpaint_runner.prepare
    real_restore = app.outpaint_runner.restore

    def tracking_prepare(source, *args, **kwargs):
        state["prepare_source_sizes"].append(source.size)
        return real_prepare(source, *args, **kwargs)

    def tracking_restore(source, generated):
        state["restore_source_sizes"].append(source.size)
        return real_restore(source, generated)

    real_feathered = app.outpaint_runner.restore_feathered

    def tracking_feathered(source, generated, width_px):
        state["restore_source_sizes"].append(source.size)
        state.setdefault("feather_widths", []).append(width_px)
        return real_feathered(source, generated, width_px)

    monkeypatch.setattr(app.outpaint_runner, "prepare", tracking_prepare)
    monkeypatch.setattr(app.outpaint_runner, "restore", tracking_restore)
    monkeypatch.setattr(app.outpaint_runner, "restore_feathered", tracking_feathered)
    score_iter = iter(scores if scores is not None else [0.1])
    detached_iter = iter(detached if detached is not None else [])
    sentinel_session = object()

    class FakeGenerator:
        def __init__(self, device=None):
            self.device = device

        def manual_seed(self, seed):
            self.seed = seed
            return self

    class FakePipe:
        safety_checker = None
        feature_extractor = None
        requires_safety_checker = False

        def __call__(self, **kwargs):
            state["events"].append("pipe_call")
            state["call_kwargs"].update(kwargs)
            seed = kwargs["generator"].seed
            state["seeds"].append(seed)
            color = (len(state["seeds"]) * 30, 100, 200)
            state["candidate_colors"].append(color)
            generated = Image.new("RGB", (768, 432), color)
            return types.SimpleNamespace(
                images=[generated],
                nsfw_content_detected=None,
            )

    fake_cc = types.ModuleType("character_check")
    fake_cc.EXTRA_CHARACTER_THRESHOLD = 0.40
    fake_cc.DETACHED_CHARACTER_THRESHOLD = 0.05

    def load_session(lab):
        state["session_loads"].append(lab)
        return sentinel_session

    def compute_character_mask(image, session):
        assert session is sentinel_session
        return Image.new("L", image.size, 0)

    def extra_character_score(character_mask, generated_mask):
        assert isinstance(character_mask, Image.Image)
        state["checked_masks"].append(generated_mask.copy())
        return next(score_iter, 0.1)

    def detached_character_fraction(character_mask, generated_mask):
        assert isinstance(character_mask, Image.Image)
        assert isinstance(generated_mask, Image.Image)
        return next(detached_iter, 0.0)

    fake_cc.load_session = load_session
    fake_cc.compute_character_mask = compute_character_mask
    fake_cc.extra_character_score = extra_character_score
    fake_cc.detached_character_fraction = detached_character_fraction

    fake_pp = types.ModuleType("powerpaint_gpu")
    fake_pp.PROMPT_A = " P_ctxt"
    fake_pp.PROMPT_B = " P_ctxt"
    fake_pp.NEGATIVE_PROMPT_A = " P_obj"
    fake_pp.NEGATIVE_PROMPT_B = " P_obj"
    fake_pp.TRADOFF = 1.0
    fake_pp.TRADOFF_NAG = 1.0
    fake_pp.BRUSHNET_CONDITIONING_SCALE = 1.0
    fake_pp.INFERENCE_WIDTH = 768
    fake_pp.INFERENCE_HEIGHT = 432
    fake_pp.RECIPES_PP = {
        "official": {
            "recipe_name": "official",
            "promptU": OFFICIAL_PROMPT_U,
            "negative_promptU": NEGATIVE_PROMPT_U,
            "guidance_scale": 10.0,
            "num_inference_steps": 45,
            "mask_overlap_px": 8,
        }
    }

    def expand_mask(mask, px):
        if px == 0:
            return mask.copy()
        return mask.filter(ImageFilter.MaxFilter(2 * px + 1))

    def prepare_brushnet_input(canvas, mask):
        black = Image.new("RGB", canvas.size, (0, 0, 0))
        return Image.composite(black, canvas, mask)

    def verify_powerpaint_preflight(lab):
        state["events"].append("verify_preflight")
        return {"lab": lab}

    def load_powerpaint_pipeline(gate, model_info=None):
        state["events"].append("load_pipeline")
        state["gate"] = gate
        state["model_info"] = model_info
        pipe = FakePipe()
        state["pipe"] = pipe
        return pipe

    def release_pipeline(pipe, gate):
        assert pipe is state["pipe"]
        assert gate is state["gate"]
        state["events"].append("release_pipeline")
        state["released"] = True

    fake_pp.expand_mask = expand_mask
    fake_pp.prepare_brushnet_input = prepare_brushnet_input
    fake_pp.verify_powerpaint_preflight = verify_powerpaint_preflight
    fake_pp.load_powerpaint_pipeline = load_powerpaint_pipeline
    fake_pp.release_pipeline = release_pipeline

    class FakeEngineAdapter:
        def process(self, work, scale, preset):
            state["events"].append("engine_process")
            assert state["released"] is True
            assert work.is_dir()
            with Image.open(work / "source.png") as src:
                assert src.size in ((768, 432), (1536, 864))
                state["esrgan_source_sizes"].append(src.size)
                state["esrgan_source_colors"].append(src.getpixel((0, 0)))
                out_size = (src.width * scale, src.height * scale)
            state["engine_calls"].append((work, scale, preset))
            if engine_raises:
                raise EngineError("Engine xử lý thất bại.")
            Image.new("RGB", out_size, (10, 120, 220)).save(
                work / "native.png", format="PNG"
            )

    fake_sr = types.ModuleType("sdxl_repair_lab")

    def repair_with_sdxl(image, mask, session):
        assert state["released"] is True  # PowerPaint must leave the GPU first
        assert session is sentinel_session
        state["events"].append("sdxl_repair")
        state["repair_inputs"].append((image.getpixel((0, 0)), mask.copy()))
        return Image.new("RGB", image.size, REPAIRED_COLOR)

    fake_sr.repair_with_sdxl = repair_with_sdxl
    state["repair_inputs"] = []

    fake_hr = types.ModuleType("hires_refine")
    fake_hr.SIZE = (1536, 864)

    def build_init(source, generated):
        canvas, mask = real_prepare(source, size=fake_hr.SIZE)
        background = generated.resize(fake_hr.SIZE, Image.Resampling.LANCZOS)
        init_rgb = Image.composite(background, canvas, mask)
        return init_rgb, canvas, mask

    def refine(
        pipe,
        torch_mod,
        source,
        generated,
        prompt_u,
        negative_prompt_u,
        seed,
        strength=0.5,
    ):
        assert state["released"] is False
        assert pipe is state["pipe"]
        state["refine_calls"].append(
            {
                "pipe": pipe,
                "torch": torch_mod,
                "source_size": source.size,
                "generated_size": generated.size,
                "generated_color": generated.getpixel((0, 0)),
                "prompt_u": prompt_u,
                "negative_prompt_u": negative_prompt_u,
                "seed": seed,
                "strength": strength,
            }
        )
        if refine_error is not None:
            raise refine_error
        return Image.new("RGB", fake_hr.SIZE, generated.getpixel((0, 0)))

    fake_hr.build_init = build_init
    fake_hr.refine = refine

    class FakeOutOfMemoryError(RuntimeError):
        pass

    fake_cuda = types.SimpleNamespace(
        OutOfMemoryError=FakeOutOfMemoryError,
        empty_cache=lambda: state["empty_cache_calls"].append(True),
    )
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.Generator = FakeGenerator
    fake_torch.cuda = fake_cuda

    sentinel_tagger = object()
    fake_st = types.ModuleType("scene_tags")

    def load_tagger(lab_dir):
        state.setdefault("tagger_loads", []).append(lab_dir)
        return sentinel_tagger

    def scene_prompt(image, tagger):
        assert tagger is sentinel_tagger
        state.setdefault("scene_prompt_images", []).append(image.size)
        return ""

    fake_st.load_tagger = load_tagger
    fake_st.scene_prompt = scene_prompt

    monkeypatch.setitem(sys.modules, "character_check", fake_cc)
    monkeypatch.setitem(sys.modules, "powerpaint_gpu", fake_pp)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "sdxl_repair_lab", fake_sr)
    monkeypatch.setitem(sys.modules, "hires_refine", fake_hr)
    monkeypatch.setitem(sys.modules, "scene_tags", fake_st)
    monkeypatch.setattr(app.outpaint_runner, "EngineAdapter", FakeEngineAdapter)
    return state, fake_pp


def test_automatic_background_runs_without_content_classifier(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    work_dir = tmp_path / "output-esrgan"
    source_image = Image.new("RGB", (16, 24), "#336688")
    source_image.putpixel((8, 12), (255, 100, 50))
    source_image.save(source_path, format="PNG")
    source_bytes_before = source_path.read_bytes()

    state, fake_pp = _setup_fake_runner_deps(monkeypatch)

    for env_name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_HOME",
    ):
        monkeypatch.setenv(env_name, "")

    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "anime", str(source_path), str(output_path)],
    )

    app.outpaint_runner.main()

    assert state["pipe"].safety_checker is None
    assert state["pipe"].feature_extractor is None
    assert state["pipe"].requires_safety_checker is False
    assert state["events"] == [
        "verify_preflight",
        "load_pipeline",
        "pipe_call",
        "release_pipeline",
        "engine_process",
    ]
    assert state["engine_calls"] == [(work_dir, 4, "anime")]
    assert not work_dir.exists()

    call_kwargs = state["call_kwargs"]
    assert call_kwargs["guidance_scale"] == 10.0
    assert call_kwargs["num_inference_steps"] == 45
    assert "empty scene" in call_kwargs["promptU"]
    assert call_kwargs["promptU"] == OFFICIAL_PROMPT_U
    assert call_kwargs["negative_promptU"] == NEGATIVE_PROMPT_U
    assert call_kwargs["promptA"] == " P_ctxt"
    assert call_kwargs["promptB"] == " P_ctxt"
    assert call_kwargs["negative_promptA"] == " P_obj"
    assert call_kwargs["negative_promptB"] == " P_obj"
    assert call_kwargs["tradoff"] == 1.0
    assert call_kwargs["tradoff_nag"] == 1.0
    assert call_kwargs["brushnet_conditioning_scale"] == 1.0
    assert call_kwargs["width"] == 768
    assert call_kwargs["height"] == 432
    assert call_kwargs["generator"].device == "cpu"
    assert call_kwargs["generator"].seed == 42

    raw_canvas, raw_mask = prepare(source_image)
    expected_mask = fake_pp.expand_mask(raw_mask, 8)
    expected_input = fake_pp.prepare_brushnet_input(raw_canvas, expected_mask)
    pipe_image = call_kwargs["image"]
    pipe_mask = call_kwargs["mask"]
    assert pipe_image.size == (768, 432)
    assert pipe_mask.size == (768, 432)
    assert pipe_mask.mode == "RGB"
    assert ImageChops.difference(pipe_mask, expected_mask.convert("RGB")).getbbox() is None
    assert ImageChops.difference(pipe_mask, raw_mask.convert("RGB")).getbbox() is not None
    assert ImageChops.difference(pipe_image, expected_input).getbbox() is None

    assert output_path.is_file()
    with Image.open(output_path) as out_img:
        assert out_img.format == "PNG"
        assert out_img.size == (3840, 2160)
        expected_main = ImageOps.contain(
            source_image, (3840, 2160), Image.Resampling.LANCZOS
        )
        main_x = (3840 - expected_main.width) // 2
        main_y = (2160 - expected_main.height) // 2
        # portrait source fills the height: left/right edges are feathered, the inner area stays exact
        band = app.outpaint_runner.FEATHER_PX
        inner = (band, 0, expected_main.width - band, expected_main.height)
        cropped = out_img.crop(
            (main_x + inner[0], main_y, main_x + inner[2], main_y + inner[3])
        )
        assert ImageChops.difference(cropped, expected_main.crop(inner)).getbbox() is None
        edge = out_img.crop((main_x, main_y, main_x + 1, main_y + expected_main.height))
        assert ImageChops.difference(edge, expected_main.crop((0, 0, 1, expected_main.height))).getbbox() is not None

    assert source_path.read_bytes() == source_bytes_before


def test_runner_removes_work_dir_when_engine_process_raises(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    work_dir = tmp_path / "output-esrgan"
    Image.new("RGB", (16, 24), "#336688").save(source_path, format="PNG")

    state, _ = _setup_fake_runner_deps(monkeypatch, engine_raises=True)
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "anime", str(source_path), str(output_path)],
    )

    with pytest.raises(EngineError):
        app.outpaint_runner.main()

    assert state["released"] is True
    assert state["engine_calls"] == [(work_dir, 4, "anime")]
    assert not work_dir.exists()
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("preset", "prompt", "expected_prompt_u"),
    [
        ("anime", "", OFFICIAL_PROMPT_U),
        (
            "photo",
            "",
            OFFICIAL_PROMPT_U.replace("anime illustration", "photograph", 1),
        ),
        (
            "anime",
            "  cherry blossoms at dusk  ",
            f"{OFFICIAL_PROMPT_U[: -len(', empty scene')]}, cherry blossoms at dusk, empty scene",
        ),
        (
            "photo",
            "  misty pine forest  ",
            f"{OFFICIAL_PROMPT_U[: -len(', empty scene')]}, misty pine forest, empty scene".replace(
                "anime illustration", "photograph", 1
            ),
        ),
    ],
)
def test_runner_prompt_u_for_preset_and_user_prompt(
    tmp_path, monkeypatch, preset, prompt, expected_prompt_u
):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (16, 24), "#336688").save(source_path, format="PNG")

    state, _ = _setup_fake_runner_deps(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "app.outpaint_runner",
            f"--prompt={prompt}",
            "--preset",
            preset,
            str(source_path),
            str(output_path),
        ],
    )

    app.outpaint_runner.main()

    assert state["call_kwargs"]["promptU"] == expected_prompt_u
    assert "empty scene" in state["call_kwargs"]["promptU"]
    assert state["engine_calls"] == [(tmp_path / "output-esrgan", 4, preset)]
    assert not (tmp_path / "output-esrgan").exists()


def test_runner_stops_on_first_seed_when_score_below_threshold(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    source_image = Image.new("RGB", (16, 24), "#336688")
    source_image.save(source_path, format="PNG")

    state, _ = _setup_fake_runner_deps(monkeypatch, scores=[0.1])
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "anime", str(source_path), str(output_path)],
    )

    app.outpaint_runner.main()

    assert state["seeds"] == [42]
    assert len(state["session_loads"]) == 1
    assert state["events"].count("release_pipeline") == 1
    assert state["esrgan_source_colors"] == [state["candidate_colors"][0]]
    _, raw_mask = prepare(source_image)
    _, hires_mask = prepare(source_image, size=(1536, 864))
    assert len(state["checked_masks"]) == 2
    assert ImageChops.difference(state["checked_masks"][0], raw_mask).getbbox() is None
    assert ImageChops.difference(state["checked_masks"][1], hires_mask).getbbox() is None


def test_runner_repairs_with_sdxl_when_every_seed_has_extra_characters(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (16, 24), "#336688").save(source_path, format="PNG")

    state, _ = _setup_fake_runner_deps(monkeypatch, scores=[0.9, 0.5, 0.6])
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "anime", str(source_path), str(output_path)],
    )

    app.outpaint_runner.main()

    assert state["seeds"] == [42, 123, 7]
    assert len(state["session_loads"]) == 1
    assert state["events"] == [
        "verify_preflight",
        "load_pipeline",
        "pipe_call",
        "pipe_call",
        "pipe_call",
        "release_pipeline",
        "sdxl_repair",
        "engine_process",
    ]
    # the lowest-score candidate is repaired with the raw (unexpanded) mask, then upscaled
    (color, mask), = state["repair_inputs"]
    assert color == state["candidate_colors"][1]
    _, raw_mask = prepare(Image.new("RGB", (16, 24)))
    assert ImageChops.difference(mask, raw_mask).getbbox() is None
    assert state["esrgan_source_colors"] == [REPAIRED_COLOR]


def test_runner_stops_after_second_seed_when_below_threshold(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (16, 24), "#336688").save(source_path, format="PNG")

    state, _ = _setup_fake_runner_deps(monkeypatch, scores=[0.9, 0.2])
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "anime", str(source_path), str(output_path)],
    )

    app.outpaint_runner.main()

    assert state["seeds"] == [42, 123]
    assert len(state["session_loads"]) == 1
    assert state["events"] == [
        "verify_preflight",
        "load_pipeline",
        "pipe_call",
        "pipe_call",
        "release_pipeline",
        "engine_process",
    ]
    assert state["esrgan_source_colors"] == [state["candidate_colors"][1]]


def test_runner_repairs_with_sdxl_when_only_detached_blobs_flag_every_seed(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (16, 24), "#336688").save(source_path, format="PNG")

    state, _ = _setup_fake_runner_deps(
        monkeypatch, scores=[0.1, 0.1, 0.1], detached=[0.3, 0.3, 0.3]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "anime", str(source_path), str(output_path)],
    )

    app.outpaint_runner.main()

    assert state["seeds"] == [42, 123, 7]
    assert state["events"] == [
        "verify_preflight",
        "load_pipeline",
        "pipe_call",
        "pipe_call",
        "pipe_call",
        "release_pipeline",
        "sdxl_repair",
        "engine_process",
    ]
    assert state["esrgan_source_colors"] == [REPAIRED_COLOR]


def test_runner_keeps_unflagged_seed_over_lower_score_flagged_seed(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (16, 24), "#336688").save(source_path, format="PNG")

    state, _ = _setup_fake_runner_deps(
        monkeypatch, scores=[0.05, 0.2], detached=[0.3, 0.0]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "anime", str(source_path), str(output_path)],
    )

    app.outpaint_runner.main()

    assert state["seeds"] == [42, 123]
    assert "sdxl_repair" not in state["events"]
    assert state["esrgan_source_colors"] == [state["candidate_colors"][1]]


def test_runner_reduces_large_portrait_source_and_preserves_small_source(tmp_path, monkeypatch):
    large_source = tmp_path / "large.png"
    large_output = tmp_path / "large_out.png"
    Image.new("L", (8000, 12000), 128).save(large_source, format="PNG", compress_level=1)

    state, _ = _setup_fake_runner_deps(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "anime", str(large_source), str(large_output)],
    )

    app.outpaint_runner.main()

    assert len(state["prepare_source_sizes"]) == 1
    assert state["prepare_source_sizes"] == state["restore_source_sizes"]
    # seam is blended over the band PowerPaint redraws (8 px overlap at 768 wide = 40 px at 3840)
    assert state["feather_widths"] == [40]
    reduced_w, reduced_h = state["prepare_source_sizes"][0]
    assert reduced_w == 1600
    assert 2160 <= reduced_h < 2 * 2160
    with Image.open(large_output) as out_img:
        assert out_img.size == (3840, 2160)

    small_source = tmp_path / "small.png"
    small_output = tmp_path / "small_out.png"
    Image.new("RGB", (16, 24), "#336688").save(small_source, format="PNG")

    state_small, _ = _setup_fake_runner_deps(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "anime", str(small_source), str(small_output)],
    )

    app.outpaint_runner.main()

    assert state_small["prepare_source_sizes"] == [(16, 24)]
    assert state_small["restore_source_sizes"] == [(16, 24)]


def test_runner_reduces_large_exif_rotated_source_using_transposed_axes(tmp_path, monkeypatch):
    source_path = tmp_path / "rotated.png"
    output_path = tmp_path / "rotated_out.png"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("L", (12000, 8000), 128).save(
        source_path, format="PNG", exif=exif.tobytes(), compress_level=1
    )

    state, _ = _setup_fake_runner_deps(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "anime", str(source_path), str(output_path)],
    )

    app.outpaint_runner.main()

    assert state["prepare_source_sizes"] == [(1600, 2400)]
    assert state["restore_source_sizes"] == [(1600, 2400)]
    with Image.open(output_path) as out_img:
        assert out_img.size == (3840, 2160)


def test_refactored_helpers_load_generate_and_upscale(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (16, 24), "#336688").save(source_path, format="PNG")
    state, _ = _setup_fake_runner_deps(monkeypatch)

    source = app.outpaint_runner.load_source(source_path)
    assert source.mode == "RGB"
    assert source.size == (16, 24)

    generated, mask = app.outpaint_runner.generate(source, "anime", "")
    assert generated.size == (1536, 864)
    assert mask.size == (768, 432)
    assert state["released"] is True

    work_dir = tmp_path / "work"
    native = app.outpaint_runner.upscale_background(generated, work_dir, "anime")
    assert native.size == (6144, 3456)
    assert state["engine_calls"] == [(work_dir, 4, "anime")]

    full_16_9 = Image.new("RGB", (192, 108), "#112233")
    gen_none, full_mask = app.outpaint_runner.generate(full_16_9, "anime", "")
    assert gen_none is None
    assert full_mask.getextrema() == (0, 0)


def test_upscale_background_supports_1536x864_input(tmp_path, monkeypatch):
    calls = []

    class FakeHiresEngineAdapter:
        def process(self, work, scale, preset):
            assert work.is_dir()
            with Image.open(work / "source.png") as src:
                assert src.size == (1536, 864)
                out_size = (src.width * scale, src.height * scale)
            calls.append((work, scale, preset))
            Image.new("RGB", out_size, (20, 40, 60)).save(
                work / "native.png", format="PNG"
            )

    monkeypatch.setattr(app.outpaint_runner, "EngineAdapter", FakeHiresEngineAdapter)
    work_dir = tmp_path / "work-hires"
    hires = Image.new("RGB", (1536, 864), "#224466")
    native = app.outpaint_runner.upscale_background(hires, work_dir, "anime")
    assert native.size == (6144, 3456)
    assert calls == [(work_dir, 4, "anime")]


def test_runner_refines_unflagged_best_candidate_at_1536x864_before_upscale(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (16, 24), "#336688").save(source_path, format="PNG")

    state, _ = _setup_fake_runner_deps(monkeypatch, scores=[0.9, 0.2, 0.1])
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "anime", str(source_path), str(output_path)],
    )

    app.outpaint_runner.main()

    assert app.outpaint_runner.REFINE_STRENGTH == 0.35
    assert state["seeds"] == [42, 123]
    assert len(state["refine_calls"]) == 1
    call = state["refine_calls"][0]
    assert call["source_size"] == (16, 24)
    assert call["generated_size"] == (768, 432)
    assert call["generated_color"] == state["candidate_colors"][1]
    assert call["prompt_u"] == OFFICIAL_PROMPT_U
    assert call["negative_prompt_u"] == NEGATIVE_PROMPT_U
    assert call["seed"] == 123
    assert call["strength"] == 0.35
    assert state["esrgan_source_sizes"] == [(1536, 864)]
    assert state["esrgan_source_colors"] == [state["candidate_colors"][1]]
    with Image.open(output_path) as out_img:
        assert out_img.size == (3840, 2160)


def test_runner_falls_back_to_768x432_candidate_when_refined_image_is_flagged(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (16, 24), "#336688").save(source_path, format="PNG")

    state, _ = _setup_fake_runner_deps(monkeypatch, scores=[0.1, 0.75])
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "anime", str(source_path), str(output_path)],
    )

    app.outpaint_runner.main()

    assert len(state["refine_calls"]) == 1
    assert "sdxl_repair" not in state["events"]
    assert state["esrgan_source_sizes"] == [(768, 432)]
    assert state["esrgan_source_colors"] == [state["candidate_colors"][0]]


def test_runner_falls_back_to_768x432_candidate_when_refine_raises_oom(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (16, 24), "#336688").save(source_path, format="PNG")

    state, _ = _setup_fake_runner_deps(
        monkeypatch,
        scores=[0.1],
        refine_error=RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "anime", str(source_path), str(output_path)],
    )

    app.outpaint_runner.main()

    assert len(state["refine_calls"]) == 1
    assert len(state["empty_cache_calls"]) == 1
    assert state["released"] is True
    assert state["esrgan_source_sizes"] == [(768, 432)]
    assert state["esrgan_source_colors"] == [state["candidate_colors"][0]]
    assert output_path.is_file()


def test_runner_skips_refine_when_every_seed_is_flagged(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (16, 24), "#336688").save(source_path, format="PNG")

    state, _ = _setup_fake_runner_deps(monkeypatch, scores=[0.8, 0.6, 0.7])
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "anime", str(source_path), str(output_path)],
    )

    app.outpaint_runner.main()

    assert state["refine_calls"] == []
    assert "sdxl_repair" in state["events"]
    assert state["esrgan_source_sizes"] == [(768, 432)]
    assert state["esrgan_source_colors"] == [REPAIRED_COLOR]


def test_runner_propagates_non_oom_error_from_refine(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (16, 24), "#336688").save(source_path, format="PNG")

    state, _ = _setup_fake_runner_deps(
        monkeypatch,
        scores=[0.1],
        refine_error=ValueError("strength must be in (0, 1]"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "anime", str(source_path), str(output_path)],
    )

    with pytest.raises(ValueError, match="strength must be in"):
        app.outpaint_runner.main()

    assert len(state["refine_calls"]) == 1
    assert state["released"] is True
    assert state["engine_calls"] == []


def test_generate_scene_parameter_updates_default_prompt_and_respects_user_prompt(
    monkeypatch,
):
    soft = OFFICIAL_PROMPT_U[: -len(", empty scene")]
    source = Image.new("RGB", (16, 24), "#336688")

    state, _ = _setup_fake_runner_deps(monkeypatch)
    app.outpaint_runner.generate(source, "anime", "", scene="outdoors, sky")
    expected_anime = f"{soft}, outdoors, sky, empty scene"
    assert state["call_kwargs"]["promptU"] == expected_anime
    assert state["refine_calls"][0]["prompt_u"] == expected_anime

    state_photo, _ = _setup_fake_runner_deps(monkeypatch)
    app.outpaint_runner.generate(source, "photo", "", scene="outdoors, sky")
    assert state_photo["call_kwargs"]["promptU"] == expected_anime.replace(
        "anime illustration", "photograph", 1
    )

    state_user, _ = _setup_fake_runner_deps(monkeypatch)
    app.outpaint_runner.generate(
        source, "anime", "cherry blossoms at dusk", scene="outdoors, sky"
    )
    assert (
        state_user["call_kwargs"]["promptU"]
        == f"{soft}, cherry blossoms at dusk, empty scene"
    )

    state_user_photo, _ = _setup_fake_runner_deps(monkeypatch)
    app.outpaint_runner.generate(
        source, "photo", "misty pine forest", scene="outdoors, sky"
    )
    assert state_user_photo["call_kwargs"]["promptU"] == (
        f"{soft}, misty pine forest, empty scene".replace(
            "anime illustration", "photograph", 1
        )
    )

    state_empty, _ = _setup_fake_runner_deps(monkeypatch)
    app.outpaint_runner.generate(source, "anime", "", scene="")
    assert state_empty["call_kwargs"]["promptU"] == OFFICIAL_PROMPT_U


def test_runner_wires_auto_scene_for_anime_empty_prompt(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (16, 24), "#336688").save(source_path, format="PNG")

    state, _ = _setup_fake_runner_deps(monkeypatch)
    fake_st = sys.modules["scene_tags"]
    monkeypatch.setattr(fake_st, "scene_prompt", lambda image, tagger: "outdoors, sky")
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "anime", str(source_path), str(output_path)],
    )

    app.outpaint_runner.main()

    assert len(state["tagger_loads"]) == 1
    assert ", outdoors, sky, empty scene" in state["call_kwargs"]["promptU"]
    assert state["call_kwargs"]["promptU"].endswith(", outdoors, sky, empty scene")


def test_runner_skips_tagger_when_anime_has_user_prompt(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (16, 24), "#336688").save(source_path, format="PNG")

    state, _ = _setup_fake_runner_deps(monkeypatch)
    fake_st = sys.modules["scene_tags"]

    def forbid_load_tagger(lab_dir):
        raise AssertionError("load_tagger should not be called when user prompt is provided")

    monkeypatch.setattr(fake_st, "load_tagger", forbid_load_tagger)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "app.outpaint_runner",
            "--prompt=sunset beach",
            "--preset",
            "anime",
            str(source_path),
            str(output_path),
        ],
    )

    app.outpaint_runner.main()

    soft = OFFICIAL_PROMPT_U[: -len(", empty scene")]
    assert (
        state["call_kwargs"]["promptU"]
        == f"{soft}, sunset beach, empty scene"
    )


def test_runner_skips_tagger_for_photo_preset(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (16, 24), "#336688").save(source_path, format="PNG")

    state, _ = _setup_fake_runner_deps(monkeypatch)
    fake_st = sys.modules["scene_tags"]

    def forbid_load_tagger(lab_dir):
        raise AssertionError("load_tagger should not be called for photo preset")

    monkeypatch.setattr(fake_st, "load_tagger", forbid_load_tagger)
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "photo", str(source_path), str(output_path)],
    )

    app.outpaint_runner.main()

    assert state["call_kwargs"]["promptU"] == OFFICIAL_PROMPT_U.replace(
        "anime illustration", "photograph", 1
    )


def test_runner_falls_back_to_default_prompt_when_load_tagger_raises_runtime_error(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (16, 24), "#336688").save(source_path, format="PNG")

    state, _ = _setup_fake_runner_deps(monkeypatch)
    fake_st = sys.modules["scene_tags"]

    def failing_load_tagger(lab_dir):
        raise RuntimeError("Missing WD tagger manifest")

    monkeypatch.setattr(fake_st, "load_tagger", failing_load_tagger)
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.outpaint_runner", "--preset", "anime", str(source_path), str(output_path)],
    )

    app.outpaint_runner.main()

    assert state["call_kwargs"]["promptU"] == OFFICIAL_PROMPT_U
    assert (tmp_path / "output-scene.txt").read_text(encoding="utf-8") == ""


def test_runner_writes_scene_sidecar_only_in_auto_mode(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (16, 24), "#336688").save(source_path, format="PNG")

    auto_out = tmp_path / "wallpaper.tmp.png"
    auto_sidecar = tmp_path / "wallpaper.tmp-scene.txt"
    state, _ = _setup_fake_runner_deps(monkeypatch)
    fake_st = sys.modules["scene_tags"]
    monkeypatch.setattr(fake_st, "scene_prompt", lambda image, tagger: "outdoors, sky")
    sidecar_seen_before_generate = []
    real_generate = app.outpaint_runner.generate

    def check_sidecar_then_generate(*args, **kwargs):
        sidecar_seen_before_generate.append(
            auto_sidecar.read_text(encoding="utf-8")
        )
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(app.outpaint_runner, "generate", check_sidecar_then_generate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "app.outpaint_runner",
            "--prompt=   ",
            "--preset",
            "anime",
            str(source_path),
            str(auto_out),
        ],
    )
    app.outpaint_runner.main()
    assert sidecar_seen_before_generate == ["outdoors, sky"]
    assert auto_sidecar.read_text(encoding="utf-8") == "outdoors, sky"
    assert len(state["tagger_loads"]) == 1

    monkeypatch.setattr(app.outpaint_runner, "generate", real_generate)

    prompt_out = tmp_path / "prompted.png"
    _setup_fake_runner_deps(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "app.outpaint_runner",
            "--prompt=sunset beach",
            "--preset",
            "anime",
            str(source_path),
            str(prompt_out),
        ],
    )
    app.outpaint_runner.main()
    assert not (tmp_path / "prompted-scene.txt").exists()

    photo_out = tmp_path / "photo.png"
    _setup_fake_runner_deps(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "app.outpaint_runner",
            "--preset",
            "photo",
            str(source_path),
            str(photo_out),
        ],
    )
    app.outpaint_runner.main()
    assert not (tmp_path / "photo-scene.txt").exists()


def test_main_anime_with_prompt_uses_box_mode(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    source = Image.new("RGB", (108, 192), "#336688")
    source.save(source_path, format="PNG")

    state, fake_pp = _setup_fake_runner_deps(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "app.outpaint_runner",
            "--prompt=night city, neon lights",
            "--preset",
            "anime",
            str(source_path),
            str(output_path),
        ],
    )

    app.outpaint_runner.main()

    _, box_mask = prepare(source)
    pipe_mask = state["call_kwargs"]["mask"]
    expected_expanded_box = fake_pp.expand_mask(box_mask, 8).convert("RGB")
    assert ImageChops.difference(pipe_mask, expected_expanded_box).getbbox() is None
    assert state["feather_widths"] == [app.outpaint_runner.FEATHER_PX]
    assert output_path.is_file()

