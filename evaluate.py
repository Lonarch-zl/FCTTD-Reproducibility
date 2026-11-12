import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import dataset
from model import build_model, load_checkpoint, model_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    device = torch.device(args.device)
    model = build_model(load_checkpoint(args.checkpoint)).to(device).eval()
    loader = DataLoader(
        dataset(args.data_root, False, download=not args.no_download),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            correct += (logits.argmax(1).cpu() == labels).sum().item()
            total += labels.numel()
    if total != 10000:
        raise RuntimeError(f"Expected 10000 CIFAR-10 test images, got {total}")
    result = {
        "checkpoint": str(args.checkpoint),
        "correct": correct,
        "total": total,
        "top1_accuracy": correct / total,
        "batch_size": args.batch_size,
        "precision": "float32",
        "device": str(device),
        "torch": str(torch.__version__),
        "model": model_metadata(model),
    }
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
