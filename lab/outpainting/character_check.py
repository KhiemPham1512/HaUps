"""Anime character segmentation and extra-character detection on CPU."""

from pathlib import Path

from PIL import Image, ImageFilter

SEG_SIZE = 1024
GRID = (192, 108)
FAR_CELLS = 16
EXTRA_CHARACTER_THRESHOLD = 0.40
DETACHED_CHARACTER_THRESHOLD = 0.05
SEG_MODEL_RELATIVE = "anime-seg/isnetis.onnx"


def load_session(lab_root):
    """Load the local anime-seg ONNX session on CPU."""
    import onnxruntime

    return onnxruntime.InferenceSession(
        str(Path(lab_root) / SEG_MODEL_RELATIVE),
        providers=["CPUExecutionProvider"],
    )


def compute_character_mask(image, session):
    """Character alpha (L, image size, 255 = character) via skytnt/anime-seg ISNet ONNX."""
    import numpy as np


    w0, h0 = image.size
    if h0 > w0:
        h, w = SEG_SIZE, max(1, round(SEG_SIZE * w0 / h0))
    else:
        h, w = max(1, round(SEG_SIZE * h0 / w0)), SEG_SIZE
    ph, pw = SEG_SIZE - h, SEG_SIZE - w
    canvas = np.zeros((SEG_SIZE, SEG_SIZE, 3), dtype=np.float32)
    resized = image.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
    canvas[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = np.asarray(resized, dtype=np.float32) / 255.0
    out = session.run(None, {"img": canvas.transpose(2, 0, 1)[None]})[0][0, 0]
    out = out[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    alpha = Image.fromarray((np.clip(out, 0.0, 1.0) * 255).astype(np.uint8), "L")
    return alpha.resize((w0, h0), Image.Resampling.BILINEAR)


def extra_character_score(character_mask, generated_mask):
    """Share of generated pixels farther than ~64 px (at 768 wide) from the original image that are character pixels; >= threshold means a probable extra character (calibrated 2026-11-14 on 128 extended-evaluation-v2 official outputs)."""
    char, gen, far = _grid_masks(character_mask, generated_mask)
    if far.sum() == 0:
        return 0.0
    return float((char & far).sum() / far.sum())


def detached_character_fraction(character_mask, generated_mask):
    """Share of the generated area covered by character blobs not connected to the original frame (calibrated 2026-11-16 on 128 official outputs: extra-character cases >= 0.138, clean cases <= 0.012)."""
    char, gen, _ = _grid_masks(character_mask, generated_mask)
    if gen.sum() == 0:
        return 0.0
    blob = char & gen
    attached = blob & _grow(~gen)
    while True:
        grown = _grow(attached) & blob
        if (grown == attached).all():
            break
        attached = grown
    return float((blob & ~attached).sum() / gen.sum())


def is_extra_character(character_mask, generated_mask):
    """Return True if far-zone character score or detached character fraction reaches threshold."""
    return (
        extra_character_score(character_mask, generated_mask) >= EXTRA_CHARACTER_THRESHOLD
        or detached_character_fraction(character_mask, generated_mask)
        >= DETACHED_CHARACTER_THRESHOLD
    )


REPAIR_DILATE_PX = 12


def _grow(cells):
    import numpy as np

    padded = np.pad(cells, 1, constant_values=False)
    return cells | padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]


def _grid_masks(character_mask, generated_mask):
    import numpy as np

    char = np.asarray(character_mask.resize(GRID, Image.Resampling.BILINEAR)) > 127
    gen = np.asarray(generated_mask.resize(GRID, Image.Resampling.NEAREST)) > 127
    near = ~gen
    for _ in range(FAR_CELLS):
        near = _grow(near)
    return char, gen, gen & ~near


def extra_character_repair_mask(character_mask, generated_mask):
    """L mask (255 = repaint) of whole generated character blobs that reach the far zone.

    Blobs that stay near the original (hair/limbs continuing it) are kept; the mask is
    dilated a little and never covers the original (known) region.
    """
    import numpy as np

    char, gen, far = _grid_masks(character_mask, generated_mask)
    blob = char & gen
    repair = blob & far
    while True:  # flood each far seed through its 4-connected character blob
        grown = _grow(repair) & blob
        if (grown == repair).all():
            break
        repair = grown
    mask = Image.fromarray(repair.astype(np.uint8) * 255, "L").resize(
        generated_mask.size, Image.Resampling.NEAREST
    )
    mask = mask.filter(ImageFilter.MaxFilter(2 * REPAIR_DILATE_PX + 1))
    return Image.composite(mask, Image.new("L", mask.size, 0), generated_mask.point(lambda v: 255 if v > 127 else 0))
