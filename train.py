import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from data import dataset
from fcttd.layers import TuckerKruskalLinear
from model import build_model, load_checkpoint, model_metadata


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/end_to_end"))
    parser.add_argument("--split", type=Path, default=Path("configs/split_seed43.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--classifier-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--minimum-lr", type=float, default=1e-7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--decompose-fc1", action="store_true")
    parser.add_argument("--cp-rank", type=int, default=400)
    parser.add_argument("--tucker-iters", type=int, default=10)
    parser.add_argument("--cp-iters", type=int, default=200)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    source = load_checkpoint(args.checkpoint)
    model = build_model(source).to(device)
    if args.decompose_fc1:
        if not isinstance(model.classifier[0], nn.Linear):
            raise ValueError("FC1 is already compressed")
        model.classifier[0] = TuckerKruskalLinear.from_linear(
            model.classifier[0], [16, 32, 49], [16, 16, 16], [8] * 6,
            args.cp_rank, decompose_device=device,
            n_iter_tucker=args.tucker_iters, n_iter_cp=args.cp_iters,
        )
    model.requires_grad_(True)
    split = torch.load(args.split, map_location="cpu", weights_only=True)
    train_indices = torch.as_tensor(split["train_indices"], dtype=torch.long)
    val_indices = torch.as_tensor(split["validation_indices"], dtype=torch.long)
    combined = torch.cat([train_indices, val_indices])
    if len(train_indices) != 45000 or len(val_indices) != 5000 or not torch.equal(combined.sort().values, torch.arange(50000)):
        raise ValueError("Split must partition 50000 images into disjoint 45000/5000 subsets")
    training = Subset(dataset(args.data_root, True, True, not args.no_download), train_indices.tolist())
    validation = Subset(dataset(args.data_root, True, False, not args.no_download), val_indices.tolist())
    loaders = [DataLoader(
        data, batch_size=args.batch_size, shuffle=training_mode,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed) if training_mode else None,
    ) for data, training_mode in ((training, True), (validation, False))]
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": args.backbone_lr},
        {"params": model.classifier.parameters(), "lr": args.classifier_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.minimum_lr)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata = model_metadata(model)
    metadata.update({
        "protocol": "end-to-end cross-entropy fine-tuning",
        "configuration": configuration,
        "initial_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "initial_checkpoint_metadata": source.get("metadata", {}),
        "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
        "training_samples": 45000,
        "validation_samples": 5000,
    })
    torch.save({"train_indices": train_indices, "validation_indices": val_indices}, args.output_dir / "split.pt")
    (args.output_dir / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    def validate():
        model.eval()
        correct = total = 0
        loss_sum = 0.0
        with torch.inference_mode():
            for images, labels in loaders[1]:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(images)
                loss_sum += F.cross_entropy(logits.float(), labels, reduction="sum").item()
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.numel()
        return correct / total, loss_sum / total
    def save(path, epoch, accuracy, loss):
        record = dict(metadata, epoch=epoch, validation_top1=accuracy, validation_loss=loss)
        torch.save({"model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "metadata": record}, path)
    best_accuracy, best_loss = validate()
    save(args.output_dir / "best.pt", 0, best_accuracy, best_loss)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = total = 0
        for images, labels in loaders[0]:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * labels.numel()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
        accuracy, validation_loss = validate()
        scheduler.step()
        row = {"epoch": epoch, "training_loss": loss_sum / total, "training_accuracy": correct / total, "validation_accuracy": accuracy, "validation_loss": validation_loss}
        history.append(row)
        if accuracy > best_accuracy or (accuracy == best_accuracy and validation_loss < best_loss):
            best_accuracy, best_loss = accuracy, validation_loss
            save(args.output_dir / "best.pt", epoch, accuracy, validation_loss)
        save(args.output_dir / "last.pt", epoch, accuracy, validation_loss)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
