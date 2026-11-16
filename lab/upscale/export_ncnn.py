"""Export a fine-tuned run to NCNN (pnnx) and check the NCNN output matches PyTorch.

The app runs realesrgan-ncnn-vulkan, which expects blobs named "data" (input) and "output".
pnnx names them in0/out0, so the .param is renamed after export.

CLI: python lab/upscale/export_ncnn.py RUN_NAME --images DIR [--name haups-anime-x4]
Output: .local-artifacts/upscale-train/runs/RUN_NAME/models/{NAME.param, NAME.bin, check.json}
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_sr  # noqa: E402
import train_sr as ts  # noqa: E402

ENGINE = ts.ROOT / ".local-artifacts" / "engine-poc" / "engine" / "realesrgan-ncnn-vulkan.exe"


def rename_blobs(param_text):
    """Rename pnnx's in0/out0 blobs to the data/output names realesrgan-ncnn-vulkan extracts."""
    lines = param_text.splitlines()
    out = lines[:2]
    for line in lines[2:]:
        fields = line.split()
        if not fields:
            continue
        count_in, count_out = int(fields[2]), int(fields[3])
        names = fields[4:4 + count_in + count_out]
        names = ["data" if n == "in0" else "output" if n == "out0" else n for n in names]
        out.append(" ".join(fields[:4] + names + fields[4 + count_in + count_out:]))
    text = "\n".join(out) + "\n"
    if " data" not in text or " output" not in text:
        raise SystemExit("in0/out0 blobs not found in pnnx output")
    return text


def export(torch, model, work, name):
    import pnnx

    work.mkdir(parents=True)
    pt = work / "model.pt"
    example, example2 = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 96, 80)  # two shapes -> dynamic H/W
    pnnx.export(model, pt.as_posix(), example, example2)  # pnnx writes the path into generated .py; backslashes break it
    param = work / "model.ncnn.param"
    (work / f"{name}.param").write_text(rename_blobs(param.read_text()), encoding="utf-8")
    (work / "model.ncnn.bin").replace(work / f"{name}.bin")
    for leftover in work.iterdir():  # keep only the NCNN pair
        if leftover.suffix not in (".param", ".bin") or leftover.stem != name:
            leftover.unlink() if leftover.is_file() else shutil.rmtree(leftover)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    parser.add_argument("--images", required=True)
    parser.add_argument("--name", default="haups-anime-x4")
    args = parser.parse_args(argv)
    run_dir = ts.WORK / "runs" / args.run_name
    work = run_dir / "models"  # realesrgan-ncnn-vulkan rejects -m dirs without "models" in the path

    import torch
    from safetensors.torch import load_file

    model = ts.load_model(torch).eval()
    model.load_state_dict(load_file(str(run_dir / "model.safetensors")))
    with torch.no_grad():
        export(torch, model, work, args.name)

    model = model.cuda()
    tmp = ts.ROOT / ".local-artifacts" / "tmp" / f"ncnn-check-{args.run_name}"
    tmp.mkdir(parents=True)
    records = []
    try:
        for index, path in enumerate(ts.list_images(args.images), 1):
            image = Image.open(path).convert("RGB")
            image.save(tmp / "source.png")
            command = [str(ENGINE), "-i", str(tmp / "source.png"), "-o", str(tmp / "native.png"),
                       "-m", str(work), "-n", args.name, "-s", "4", "-t", "128", "-j", "1:1:1", "-f", "png"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=600)
            if result.returncode != 0 or not (tmp / "native.png").is_file():
                raise SystemExit(f"engine failed on image {index}: {result.returncode} {result.stderr[-400:]}")
            ncnn = Image.open(tmp / "native.png").convert("RGB")
            reference = eval_sr.upscale(torch, model, image)
            if ncnn.size != reference.size:
                raise SystemExit(f"size mismatch on image {index}: {ncnn.size} vs {reference.size}")
            records.append({"index": index, "size": list(image.size), "psnr_ncnn_vs_torch": round(eval_sr.psnr(ncnn, reference), 3)})
            print(records[-1], flush=True)
            (tmp / "native.png").unlink()
    finally:
        shutil.rmtree(tmp)
    digest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(work.glob(f"{args.name}.*"))}
    summary = {"min_psnr": min(r["psnr_ncnn_vs_torch"] for r in records), "sha256": digest, "records": records}
    (work / "check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("min_psnr", summary["min_psnr"], flush=True)


if __name__ == "__main__":
    main()
