"""Compare the stock anime upscaler with a fine-tuned run on held-out images.

Two checks per image (never trained on):
  synthetic: bicubic x4-downscale the image, upscale it back, PSNR against the original
             (measures how much real detail is recovered);
  real:      upscale the image itself x4 (the app's use case) and save 100% crop sheets:
             Lanczos | stock | fine-tuned | stock+Lanczos 50% blend (the app's "Giữ chi tiết nguồn").

CLI: python lab/upscale/eval_sr.py RUN_NAME --images DIR [--images DIR ...] [--checkpoint step-1000]
Output: .local-artifacts/upscale-train/runs/RUN_NAME/eval[-checkpoint]/{records.json, sheet-*.jpg}
"""

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sr as ts  # noqa: E402

TILE = 192  # LR tile
OVERLAP = 16
CROP = 96  # LR crop shown at 100% of the x4 output (384 px)


def upscale(torch, model, image):
    """Tiled x4 inference (fp16), overlapping tiles blended by simple cropping."""
    width, height = image.size
    source = ts.to_tensor(torch, [image.convert("RGB")]).cuda()
    output = torch.zeros(1, 3, height * ts.SCALE, width * ts.SCALE, device="cuda")
    step = TILE - 2 * OVERLAP
    for top in range(0, height, step):
        for left in range(0, width, step):
            y0, x0 = max(0, top - OVERLAP), max(0, left - OVERLAP)
            y1, x1 = min(height, top + step + OVERLAP), min(width, left + step + OVERLAP)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                tile = model(source[:, :, y0:y1, x0:x1]).float()
            oy, ox = (top - y0) * ts.SCALE, (left - x0) * ts.SCALE
            h = (min(height, top + step) - top) * ts.SCALE
            w = (min(width, left + step) - left) * ts.SCALE
            output[:, :, top * ts.SCALE:top * ts.SCALE + h, left * ts.SCALE:left * ts.SCALE + w] = tile[:, :, oy:oy + h, ox:ox + w]
    array = (output[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, "RGB")


def psnr(a, b):
    import numpy as np

    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    mse = ((x - y) ** 2).mean()
    return 99.0 if mse == 0 else 10 * math.log10(255.0**2 / mse)


def detail_crops(image, count=2):
    """LR boxes of the most detailed grid cells (far apart), for the 100% sheets."""
    cells = []
    for top in range(0, max(1, image.height - CROP + 1), CROP // 2):
        for left in range(0, max(1, image.width - CROP + 1), CROP // 2):
            box = (left, top, left + CROP, top + CROP)
            cells.append((ts.detail_score(image.crop(box)), box))
    cells.sort(reverse=True)
    chosen = []
    for _, box in cells:
        if all(abs(box[0] - c[0]) + abs(box[1] - c[1]) >= CROP * 2 for c in chosen):
            chosen.append(box)
        if len(chosen) == count:
            break
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    parser.add_argument("--images", action="append", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args(argv)
    run_dir = ts.WORK / "runs" / args.run_name
    weights = (run_dir / args.checkpoint if args.checkpoint else run_dir) / "model.safetensors"
    out = run_dir / ("eval-" + args.checkpoint if args.checkpoint else "eval")
    out.mkdir()  # never overwrite evidence

    import torch
    from safetensors.torch import load_file

    stock = ts.load_model(torch).cuda().eval()
    tuned = ts.load_model(torch).cuda().eval()
    tuned.load_state_dict(load_file(str(weights)))
    paths = [p for d in args.images for p in ts.list_images(d)]
    records = []
    for index, path in enumerate(paths, 1):
        image = Image.open(path).convert("RGB")
        image = image.crop((0, 0, image.width // 4 * 4, image.height // 4 * 4))
        low = image.resize((image.width // 4, image.height // 4), Image.Resampling.BICUBIC)
        record = {
            "index": index, "size": list(image.size),
            "psnr_lanczos": round(psnr(low.resize(image.size, Image.Resampling.LANCZOS), image), 3),
            "psnr_stock": round(psnr(upscale(torch, stock, low), image), 3),
            "psnr_tuned": round(psnr(upscale(torch, tuned, low), image), 3),
        }
        records.append(record)
        print(record, flush=True)
        real_stock, real_tuned = upscale(torch, stock, image), upscale(torch, tuned, image)
        lanczos = image.resize(real_stock.size, Image.Resampling.LANCZOS)
        blend = Image.blend(real_stock, lanczos, 0.5)
        crops = detail_crops(image)
        side = CROP * ts.SCALE
        sheet = Image.new("RGB", (side * 4 + 12, side * len(crops) + 4 * (len(crops) - 1)), "white")
        for row, box in enumerate(crops):
            big = tuple(v * ts.SCALE for v in box)
            for column, picture in enumerate((lanczos, real_stock, real_tuned, blend)):
                sheet.paste(picture.crop(big), (column * (side + 4), row * (side + 4)))
        sheet.save(out / f"sheet-{index:02d}.jpg", quality=92)
    mean = {k: round(sum(r[k] for r in records) / len(records), 3) for k in ("psnr_lanczos", "psnr_stock", "psnr_tuned")}
    (out / "records.json").write_text(json.dumps({"weights": str(weights.relative_to(ts.WORK)), "columns": "lanczos | stock | tuned | stock+lanczos 50%", "mean": mean, "records": records}, indent=2), encoding="utf-8")
    print("mean", mean, flush=True)


if __name__ == "__main__":
    main()
