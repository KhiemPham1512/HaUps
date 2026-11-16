import importlib.util
from pathlib import Path

from PIL import Image
import pytest

pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "character_check", REPO_ROOT / "lab" / "outpainting" / "character_check.py"
)
character_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(character_check)


def test_extra_character_score_ignores_adjacent_continuation_and_flags_far_character():
    gen = Image.new("L", (768, 432), 0)
    gen.paste(255, (0, 0, 384, 432))

    adjacent_char = Image.new("L", (768, 432), 0)
    adjacent_char.paste(255, (344, 0, 384, 432))
    assert character_check.extra_character_score(adjacent_char, gen) == 0.0

    far_left_char = Image.new("L", (768, 432), 0)
    far_left_char.paste(255, (0, 0, 320, 432))
    assert character_check.extra_character_score(far_left_char, gen) >= 0.9

    margin_gen = Image.new("L", (768, 432), 0)
    margin_gen.paste(255, (0, 0, 260, 432))
    char_x200 = Image.new("L", (768, 432), 0)
    char_x200.paste(255, (0, 0, 200, 432))
    assert character_check.extra_character_score(char_x200, margin_gen) >= 0.9
    assert (
        character_check.extra_character_score(char_x200, gen)
        > character_check.EXTRA_CHARACTER_THRESHOLD
    )

    empty_gen = Image.new("L", (768, 432), 0)
    assert character_check.extra_character_score(far_left_char, empty_gen) == 0.0


def test_detached_character_fraction_and_is_extra_character():
    gen = Image.new("L", (768, 432), 0)
    gen.paste(255, (0, 0, 384, 432))

    adjacent_char = Image.new("L", (768, 432), 0)
    adjacent_char.paste(255, (344, 0, 384, 432))
    assert character_check.detached_character_fraction(adjacent_char, gen) == 0.0
    assert character_check.is_extra_character(adjacent_char, gen) is False

    detached_char = Image.new("L", (768, 432), 0)
    detached_char.paste(255, (40, 100, 120, 432))
    assert (
        character_check.extra_character_score(detached_char, gen)
        < character_check.EXTRA_CHARACTER_THRESHOLD
    )
    assert (
        character_check.detached_character_fraction(detached_char, gen)
        > character_check.DETACHED_CHARACTER_THRESHOLD
    )
    assert character_check.is_extra_character(detached_char, gen) is True

    empty_char = Image.new("L", (768, 432), 0)
    assert character_check.detached_character_fraction(empty_char, gen) == 0.0
    assert character_check.is_extra_character(empty_char, gen) is False
    assert (
        character_check.detached_character_fraction(detached_char, Image.new("L", (768, 432), 0))
        == 0.0
    )


def test_repair_mask_covers_far_blobs_whole_and_keeps_adjacent_continuation():
    gen = Image.new("L", (768, 432), 0)
    gen.paste(255, (0, 0, 384, 432))  # left half generated, right half is the original
    char = Image.new("L", (768, 432), 0)
    char.paste(255, (344, 0, 384, 200))  # hair continuing from the original: keep
    char.paste(255, (40, 100, 120, 432))  # separate far figure: repair
    char.paste(255, (150, 300, 383, 340))  # far figure touching the original edge: repair whole
    mask = character_check.extra_character_repair_mask(char, gen)
    assert mask.mode == "L" and mask.size == (768, 432)
    assert mask.getpixel((80, 300)) == 255
    assert mask.getpixel((40 - 8, 300)) == 255  # dilated past the blob edge
    assert mask.getpixel((300, 320)) == 255 and mask.getpixel((380, 320)) == 255
    assert mask.getpixel((364, 100)) == 0
    assert mask.crop((384, 0, 768, 432)).getextrema() == (0, 0)  # never touches the original
    assert character_check.extra_character_repair_mask(Image.new("L", (768, 432)), gen).getextrema() == (0, 0)
