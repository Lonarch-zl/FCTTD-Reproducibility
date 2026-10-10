import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2430, 0.2610)
HF_CIFAR10_STD = (0.2023, 0.1994, 0.2010)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 10
DEFAULT_FC1_OUT_SHAPE = (16, 16, 16)
DEFAULT_FC1_IN_SHAPE = (32, 28, 28)


def prod(values: Iterable[int]) -> int:
    return math.prod(int(v) for v in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VGG-16 on CIFAR-10 with optional fc1 Tucker-Kruskal decomposition.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./runs/vgg16_cifar10"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--lr-step-size", type=int, default=20)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--train-size", type=int, default=45000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--norm", choices=("cifar10", "hf-cifar10", "imagenet"), default=None)
    parser.add_argument(
        "--model-source",
        choices=("torchvision", "torchvision-bn-hf-features", "hf-cifar10"),
        default="torchvision",
    )
    parser.add_argument("--hf-model-id", default="edadaltocg/vgg16_bn_cifar10")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument(
        "--freeze-loaded-hf-weights",
        action="store_true",
        help="Freeze same-name/same-shape tensors loaded from the HF checkpoint.",
    )
    parser.add_argument("--init-weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--load-checkpoint", type=Path, default=None, help="Load a full VGG-16 checkpoint before optional decomposition.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume a checkpoint matching the current model structure.")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--profile-macs", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-interval", type=int, default=0, help="Print training batch metrics every N batches.")

    parser.add_argument("--decompose-fc1", action="store_true", help="Replace VGG-16 classifier[0] only.")
    parser.add_argument(
        "--fc1-compression",
        choices=("tucker", "tucker-kruskal", "kruskal-from-tucker"),
        default="tucker-kruskal",
    )
    parser.add_argument(
        "--train-only-compressed-fc1",
        action="store_true",
        help="After fc1 decomposition, freeze every parameter except classifier[0].",
    )
    parser.add_argument("--fc1-out-shape", type=int, nargs=3, default=DEFAULT_FC1_OUT_SHAPE)
    parser.add_argument("--fc1-in-shape", type=int, nargs=3, default=DEFAULT_FC1_IN_SHAPE)
    parser.add_argument("--fc1-tucker-ranks", type=int, nargs=6, default=(8, 8, 8, 8, 8, 8))
    parser.add_argument("--fc1-cp-rank", type=int, default=128)
    parser.add_argument("--tucker-iters", type=int, default=10)
    parser.add_argument("--cp-iters", type=int, default=200)
    parser.add_argument("--cp-tol", type=float, default=1e-6)
    parser.add_argument("--decompose-device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def resolve_image_size(args: argparse.Namespace) -> int:
    if args.image_size is not None:
        return args.image_size
    return 32 if args.model_source in {"hf-cifar10", "torchvision-bn-hf-features"} else 224


def resolve_norm(args: argparse.Namespace) -> str:
    if args.norm is not None:
        return args.norm
    return "hf-cifar10" if args.model_source in {"hf-cifar10", "torchvision-bn-hf-features"} else "cifar10"


def build_vgg16(init_weights: str) -> nn.Module:
    if init_weights == "imagenet":
        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1
            model = models.vgg16(weights=weights)
        except AttributeError:
            model = models.vgg16(pretrained=True)
    else:
        try:
            model = models.vgg16(weights=None)
        except TypeError:
            model = models.vgg16(pretrained=False)

    model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
    return model


def build_vgg16_bn(init_weights: str) -> nn.Module:
    if init_weights == "imagenet":
        try:
            weights = models.VGG16_BN_Weights.IMAGENET1K_V1
            model = models.vgg16_bn(weights=weights)
        except AttributeError:
            model = models.vgg16_bn(pretrained=True)
    else:
        try:
            model = models.vgg16_bn(weights=None)
        except TypeError:
            model = models.vgg16_bn(pretrained=False)

    model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
    return model


def replace_fc1_with_empty_tucker(model: nn.Module, args: argparse.Namespace) -> None:
    from tensor_layers import TuckerLinear

    base_model = unwrap_model(model)
    fc1 = base_model.classifier[0]
    if not isinstance(fc1, nn.Linear):
        raise TypeError(f"classifier[0] must be nn.Linear before empty Tucker replacement, got {type(fc1)}.")
    if prod(args.fc1_in_shape) != fc1.in_features:
        raise ValueError(f"prod(--fc1-in-shape) must equal {fc1.in_features}, got {args.fc1_in_shape}.")
    if prod(args.fc1_out_shape) != fc1.out_features:
        raise ValueError(f"prod(--fc1-out-shape) must equal {fc1.out_features}, got {args.fc1_out_shape}.")
    base_model.classifier[0] = TuckerLinear(
        in_shape=args.fc1_in_shape,
        out_shape=args.fc1_out_shape,
        tucker_ranks=args.fc1_tucker_ranks,
        bias=fc1.bias is not None,
        dtype=fc1.weight.dtype,
        device=fc1.weight.device,
    )


def load_hf_matching_weights_into_torchvision(model: nn.Module, args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download

    weight_path = hf_hub_download(
        repo_id=args.hf_model_id,
        filename="pytorch_model.bin",
        cache_dir=args.hf_cache_dir,
        local_files_only=args.hf_local_files_only,
    )
    hf_state = torch.load(weight_path, map_location="cpu")
    model_state = model.state_dict()
    matched_state = {
        key: value
        for key, value in hf_state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    missing = sorted(set(hf_state) - set(matched_state))

    model_state.update(matched_state)
    model.load_state_dict(model_state, strict=True)

    frozen_param_count = 0
    if args.freeze_loaded_hf_weights:
        matched_param_names = set(matched_state)
        for name, parameter in model.named_parameters():
            if name in matched_param_names:
                parameter.requires_grad = False
                frozen_param_count += parameter.numel()
        model._frozen_module_names = ["features"]

    print(f"Loaded matching HF weights into torchvision model from {weight_path}")
    print(f"  matched tensors: {len(matched_state)}")
    print(f"  skipped tensors: {len(missing)}")
    if args.freeze_loaded_hf_weights:
        print(f"  frozen matched parameters: {frozen_param_count:,}")
        print("  frozen modules kept in eval mode during training: features")
    if missing:
        print(f"  skipped examples: {missing[:5]}")


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.model_source == "hf-cifar10":
        if args.decompose_fc1:
            raise ValueError(
                "--model-source hf-cifar10 loads a CIFAR-specific VGG16-BN with head.fc=Linear(512, 10). "
                "It does not contain torchvision VGG's classifier[0]=Linear(25088, 4096), so --decompose-fc1 "
                "is only supported with --model-source torchvision."
            )
        from hf_cifar_vgg import load_hf_cifar_vgg16_bn

        model, metadata = load_hf_cifar_vgg16_bn(
            model_id=args.hf_model_id,
            cache_dir=args.hf_cache_dir,
            map_location="cpu",
            local_files_only=args.hf_local_files_only,
        )
        print(f"Loaded HuggingFace CIFAR VGG16-BN: {metadata['model_id']}")
        print(f"  config:  {metadata['config_path']}")
        print(f"  weights: {metadata['weight_path']}")
        return model.to(device)

    if args.model_source == "torchvision-bn-hf-features":
        if args.decompose_fc1:
            print(
                "Using standard torchvision vgg16_bn with HF feature weights. "
                "The decomposed fc1 will still be initialized from the standard classifier[0], "
                "not from the HF checkpoint, because the HF checkpoint has no matching fc1."
            )
        model = build_vgg16_bn("none")
        load_hf_matching_weights_into_torchvision(model, args)
        if args.decompose_fc1 and args.fc1_compression == "kruskal-from-tucker":
            if args.load_checkpoint is None:
                raise ValueError("--fc1-compression kruskal-from-tucker requires --load-checkpoint with a Tucker fc1 checkpoint.")
            replace_fc1_with_empty_tucker(model, args)
            load_model_weights(model, args.load_checkpoint, strict=True)
            model = model.to(device)
            decompose_vgg16_fc1(model, args, train_device=device)
            return model
        if args.load_checkpoint is not None:
            load_model_weights(model, args.load_checkpoint, strict=True)
        model = model.to(device)
        if args.decompose_fc1:
            decompose_vgg16_fc1(model, args, train_device=device)
        return model

    model = build_vgg16(args.init_weights)
    if args.load_checkpoint is not None:
        load_model_weights(model, args.load_checkpoint, strict=True)
    model = model.to(device)
    if args.decompose_fc1:
        decompose_vgg16_fc1(model, args, train_device=device)
    return model


def extract_model_state(checkpoint: Dict) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state", "model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_trusted_checkpoint(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model_weights(model: nn.Module, checkpoint_path: Path, strict: bool = True) -> None:
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = strip_module_prefix(extract_model_state(checkpoint))
    model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded model weights from {checkpoint_path}")


def resolve_decompose_device(value: str, train_device: torch.device) -> torch.device:
    if value == "auto":
        return train_device
    return torch.device(value)


def compression_parameter_breakdown(
    in_shape: Iterable[int],
    out_shape: Iterable[int],
    tucker_ranks: Iterable[int],
    cp_rank: Optional[int],
    bias: bool,
) -> Dict:
    in_shape = tuple(int(value) for value in in_shape)
    out_shape = tuple(int(value) for value in out_shape)
    tucker_ranks = tuple(int(value) for value in tucker_ranks)
    dimensions = out_shape + in_shape

    outer_by_mode = [dimension * rank for dimension, rank in zip(dimensions, tucker_ranks)]
    outer_params = sum(outer_by_mode)
    dense_core_params = prod(tucker_ranks)
    bias_params = prod(out_shape) if bias else 0

    core_factor_by_mode = None
    core_factor_params = None
    if cp_rank is not None:
        core_factor_by_mode = [rank * int(cp_rank) for rank in tucker_ranks]
        core_factor_params = sum(core_factor_by_mode)

    stored_core_params = dense_core_params if core_factor_params is None else core_factor_params
    stored_weight_params = outer_params + stored_core_params
    stored_fc1_params = stored_weight_params + bias_params
    dense_tucker_fc1_params = outer_params + dense_core_params + bias_params

    return {
        "outer_factor_params_by_mode": outer_by_mode,
        "core_factor_params_by_mode": core_factor_by_mode,
        "outer_factor_params": outer_params,
        "core_factor_params": core_factor_params,
        "bias_params": bias_params,
        "dense_core_params": dense_core_params,
        "stored_core_params": stored_core_params,
        "stored_weight_params": stored_weight_params,
        "stored_fc1_params": stored_fc1_params,
        "dense_tucker_fc1_params": dense_tucker_fc1_params,
        "outer_factor_share_of_compressed_fc1": outer_params / stored_fc1_params,
        "core_factor_share_of_compressed_fc1": (
            core_factor_params / stored_fc1_params if core_factor_params is not None else None
        ),
        "stored_core_share_of_compressed_fc1": stored_core_params / stored_fc1_params,
        "bias_share_of_compressed_fc1": bias_params / stored_fc1_params,
        "dense_core_share_of_dense_tucker_fc1": dense_core_params / dense_tucker_fc1_params,
        "core_factor_compression_ratio_vs_dense_core": (
            dense_core_params / core_factor_params if core_factor_params else None
        ),
    }


def decompose_vgg16_fc1(model: nn.Module, args: argparse.Namespace, train_device: torch.device) -> None:
    try:
        from tensor_layers import (
            TuckerLinear,
            TuckerKruskalLinear,
            full_linear_parameter_count,
            tucker_parameter_count,
            tucker_kruskal_parameter_count,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The --decompose-fc1 option requires tensorly. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    base_model = unwrap_model(model)
    fc1 = base_model.classifier[0]
    if args.fc1_compression == "kruskal-from-tucker":
        if not isinstance(fc1, TuckerLinear):
            raise TypeError(f"classifier[0] must be TuckerLinear before kruskal-from-tucker, got {type(fc1)}.")
    elif not isinstance(fc1, nn.Linear):
        raise TypeError(f"classifier[0] must be nn.Linear before decomposition, got {type(fc1)}.")

    if prod(args.fc1_in_shape) != fc1.in_features:
        raise ValueError(
            f"prod(--fc1-in-shape) must equal {fc1.in_features}, got {args.fc1_in_shape}."
        )
    if prod(args.fc1_out_shape) != fc1.out_features:
        raise ValueError(
            f"prod(--fc1-out-shape) must equal {fc1.out_features}, got {args.fc1_out_shape}."
        )

    full_params = full_linear_parameter_count(args.fc1_in_shape, args.fc1_out_shape, bias=fc1.bias is not None)
    if args.fc1_compression == "tucker":
        compressed_params = tucker_parameter_count(
            args.fc1_in_shape,
            args.fc1_out_shape,
            args.fc1_tucker_ranks,
            bias=fc1.bias is not None,
        )
    else:
        compressed_params = tucker_kruskal_parameter_count(
            args.fc1_in_shape,
            args.fc1_out_shape,
            args.fc1_tucker_ranks,
            args.fc1_cp_rank,
            bias=fc1.bias is not None,
        )
    cp_rank = args.fc1_cp_rank if args.fc1_compression in {"tucker-kruskal", "kruskal-from-tucker"} else None
    parameter_breakdown = compression_parameter_breakdown(
        args.fc1_in_shape,
        args.fc1_out_shape,
        args.fc1_tucker_ranks,
        cp_rank=cp_rank,
        bias=fc1.bias is not None,
    )
    if parameter_breakdown["stored_fc1_params"] != compressed_params:
        raise RuntimeError(
            "Compressed fc1 parameter accounting mismatch: "
            f"helper={parameter_breakdown['stored_fc1_params']}, expected={compressed_params}."
        )

    print("Decomposing VGG-16 classifier[0]")
    print(f"  compression method:    {args.fc1_compression}")
    print(f"  full fc1 params:       {full_params:,}")
    print(f"  decomposed fc1 params: {compressed_params:,}")
    print(f"  compression ratio:     {full_params / compressed_params:.2f}x")
    print(f"  in_shape:              {tuple(args.fc1_in_shape)}")
    print(f"  out_shape:             {tuple(args.fc1_out_shape)}")
    print(f"  Tucker ranks:          {tuple(args.fc1_tucker_ranks)}")
    if args.fc1_compression in {"tucker-kruskal", "kruskal-from-tucker"}:
        print(f"  CP/Kruskal rank:       {args.fc1_cp_rank}")
    print(f"  outer factor params:   {parameter_breakdown['outer_factor_params']:,}")
    if parameter_breakdown["core_factor_params"] is not None:
        print(f"  core factor params:    {parameter_breakdown['core_factor_params']:,}")
    print(f"  bias params:           {parameter_breakdown['bias_params']:,}")
    print(f"  dense Tucker core:     {parameter_breakdown['dense_core_params']:,}")

    decompose_device = resolve_decompose_device(args.decompose_device, train_device)
    if args.fc1_compression == "tucker":
        decomposed_fc1, stats = TuckerLinear.from_linear(
            fc1,
            in_shape=args.fc1_in_shape,
            out_shape=args.fc1_out_shape,
            tucker_ranks=args.fc1_tucker_ranks,
            decompose_device=decompose_device,
            n_iter_tucker=args.tucker_iters,
        )
        print(f"  Weight error absolute: {stats['absolute_weight_error']:.6f}")
        print(f"  Weight error relative: {stats['relative_weight_error']:.6f}")
    elif args.fc1_compression == "kruskal-from-tucker":
        decomposed_fc1, stats = TuckerKruskalLinear.from_tucker_linear(
            fc1,
            cp_rank=args.fc1_cp_rank,
            decompose_device=decompose_device,
            n_iter_cp=args.cp_iters,
            cp_tol=args.cp_tol,
        )
        print(f"  Core CP error absolute:{stats['absolute_core_error']:.6f}")
        print(f"  Core CP error relative:{stats['relative_core_error']:.6f}")
        print(f"  Weight error absolute: {stats['absolute_weight_error']:.6f}")
        print(f"  Weight error relative: {stats['relative_weight_error']:.6f}")
    else:
        decomposed_fc1, stats = TuckerKruskalLinear.from_linear(
            fc1,
            in_shape=args.fc1_in_shape,
            out_shape=args.fc1_out_shape,
            tucker_ranks=args.fc1_tucker_ranks,
            cp_rank=args.fc1_cp_rank,
            decompose_device=decompose_device,
            n_iter_tucker=args.tucker_iters,
            n_iter_cp=args.cp_iters,
            cp_tol=args.cp_tol,
            return_stats=True,
        )
        print(f"  Core CP error absolute:{stats['absolute_core_error']:.6f}")
        print(f"  Core CP error relative:{stats['relative_core_error']:.6f}")
        print(f"  Weight error absolute: {stats['absolute_weight_error']:.6f}")
        print(f"  Weight error relative: {stats['relative_weight_error']:.6f}")
    base_model.classifier[0] = decomposed_fc1.to(train_device)
    base_model._compression_stats = {
        "method": args.fc1_compression,
        "in_shape": list(args.fc1_in_shape),
        "out_shape": list(args.fc1_out_shape),
        "tucker_ranks": list(args.fc1_tucker_ranks),
        "cp_rank": args.fc1_cp_rank if args.fc1_compression in {"tucker-kruskal", "kruskal-from-tucker"} else None,
        "full_fc1_params": full_params,
        "compressed_fc1_params": compressed_params,
        "compression_ratio": full_params / compressed_params,
        **parameter_breakdown,
        **stats,
    }
    if args.train_only_compressed_fc1:
        freeze_all_except_classifier0(model)
        print("Training only decomposed classifier[0]; all other parameters are frozen.")
        print(f"  trainable parameters after freeze: {count_trainable_parameters(model):,}")


def freeze_all_except_classifier0(model: nn.Module) -> None:
    base_model = unwrap_model(model)
    for parameter in base_model.parameters():
        parameter.requires_grad = False
    for parameter in base_model.classifier[0].parameters():
        parameter.requires_grad = True

    frozen_modules = set(getattr(base_model, "_frozen_module_names", []))
    frozen_modules.add("features")
    base_model._frozen_module_names = sorted(frozen_modules)


def build_transforms(image_size: int, norm: str) -> Tuple[transforms.Compose, transforms.Compose]:
    if norm == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif norm == "hf-cifar10":
        mean, std = CIFAR10_MEAN, HF_CIFAR10_STD
    else:
        mean, std = CIFAR10_MEAN, CIFAR10_STD

    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, eval_transform


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    image_size = resolve_image_size(args)
    norm = resolve_norm(args)
    train_transform, eval_transform = build_transforms(image_size, norm)
    train_source = datasets.CIFAR10(root=args.data_dir, train=True, transform=train_transform, download=True)
    val_source = datasets.CIFAR10(root=args.data_dir, train=True, transform=eval_transform, download=True)
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=eval_transform, download=True)

    if args.train_size <= 0 or args.train_size >= len(train_source):
        raise ValueError(f"--train-size must be between 1 and {len(train_source) - 1}.")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train_source), generator=generator).tolist()
    train_indices = indices[: args.train_size]
    val_indices = indices[args.train_size :]

    train_dataset = Subset(train_source, train_indices)
    val_dataset = Subset(val_source, val_indices)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader, test_loader


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found.")
    if args.optimizer == "adamw":
        return torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(
        trainable_params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == targets).sum().item())


def iterate(loader: Iterable, no_progress: bool, desc: str):
    if tqdm is None or no_progress:
        return loader
    return tqdm(loader, desc=desc, leave=False)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    no_progress: bool,
    epoch: int,
    log_interval: int,
) -> Tuple[float, float]:
    model.train()
    set_frozen_modules_eval(model)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    num_batches = len(loader)
    for step, (images, targets) in enumerate(iterate(loader, no_progress, f"train {epoch}"), start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += accuracy_from_logits(logits.detach(), targets)
        total_samples += batch_size

        if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == num_batches):
            running_loss = total_loss / total_samples
            running_acc = total_correct / total_samples
            print(
                f"Epoch {epoch:03d} batch {step:04d}/{num_batches:04d} "
                f"loss={float(loss.detach().item()):.4f} "
                f"running_loss={running_loss:.4f} running_acc={running_acc:.4f}",
                flush=True,
            )

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    no_progress: bool,
    desc: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in iterate(loader, no_progress, desc):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += accuracy_from_logits(logits, targets)
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    epoch: int,
    best_val_acc: float,
    args: argparse.Namespace,
) -> None:
    state = {
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "args": json_safe(vars(args)),
    }
    torch.save(state, path)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(json_safe(value), file, indent=2)
    temporary_path.replace(path)


def load_existing_history(args: argparse.Namespace) -> list:
    if args.resume is None and not args.test_only:
        return []

    candidates = [args.output_dir / "history.json"]
    if args.resume is not None:
        candidates.append(args.resume.parent / "history.json")

    seen = set()
    for path in candidates:
        resolved_path = path.resolve()
        if resolved_path in seen or not path.exists():
            continue
        seen.add(resolved_path)
        with open(path, "r", encoding="utf-8") as file:
            history = json.load(file)
        if not isinstance(history, list):
            raise ValueError(f"Expected a list in history file {path}, got {type(history)}.")
        return history
    return []


def resume_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[int, float]:
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(strip_module_prefix(extract_model_state(checkpoint)), strict=True)
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_acc = float(checkpoint.get("best_val_acc", 0.0))
    print(f"Resumed training from {checkpoint_path}, start_epoch={start_epoch}, best_val_acc={best_val_acc:.4f}")
    return start_epoch, best_val_acc


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def set_frozen_modules_eval(model: nn.Module) -> None:
    base_model = unwrap_model(model)
    for module_name in getattr(base_model, "_frozen_module_names", []):
        try:
            base_model.get_submodule(module_name).eval()
        except AttributeError:
            continue


def maybe_profile_macs(model: nn.Module, image_size: int, device: torch.device) -> None:
    try:
        from torchprofile import profile_macs
    except ImportError:
        print("torchprofile is not installed; skip MAC profiling.")
        return

    model.eval()
    example = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        macs = profile_macs(model, example)
    print(f"MACs: {macs:,}")


def main() -> None:
    run_start_time = time.perf_counter()
    args = parse_args()
    image_size = resolve_image_size(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    print(f"Device: {device}")
    print(f"AMP: {use_amp}")

    model = build_model(args, device)
    set_seed(args.seed)
    print(f"Reset RNG state after model initialization/decomposition with seed={args.seed}.")

    compression_stats = getattr(unwrap_model(model), "_compression_stats", None)
    if compression_stats is not None:
        write_json(args.output_dir / "compression_stats.json", compression_stats)

    if args.data_parallel:
        if torch.cuda.device_count() <= 1:
            print("--data-parallel was set, but fewer than 2 CUDA devices are available; using a single device.")
        else:
            model = nn.DataParallel(model)

    print(unwrap_model(model))
    total_model_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable_params:,}")

    train_loader, val_loader, test_loader = build_loaders(args)
    optimizer = build_optimizer(model, args)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    scaler = GradScaler(enabled=use_amp)

    history = load_existing_history(args)
    start_epoch = 1
    best_val_acc = 0.0
    if args.resume is not None:
        start_epoch, best_val_acc = resume_training(model, optimizer, scheduler, args.resume, device)
        history = [record for record in history if int(record.get("epoch", 0)) < start_epoch]

    best_epoch = None
    if history:
        best_record = max(history, key=lambda record: float(record.get("val_acc", float("-inf"))))
        historical_best_val_acc = float(best_record["val_acc"])
        if historical_best_val_acc >= best_val_acc:
            best_val_acc = historical_best_val_acc
            best_epoch = int(best_record["epoch"])

    if not args.test_only:
        for epoch in range(start_epoch, args.epochs + 1):
            start_time = time.time()
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                use_amp,
                args.no_progress,
                epoch,
                args.log_interval,
            )
            val_loss, val_acc = evaluate(model, val_loader, device, use_amp, args.no_progress, f"val {epoch}")
            scheduler.step()

            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": lr,
                "seconds": elapsed,
            }
            history.append(record)

            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"lr={lr:.6g} time={elapsed:.1f}s"
            )

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                best_epoch = epoch

            save_checkpoint(args.output_dir / "last.pt", model, optimizer, scheduler, epoch, best_val_acc, args)
            if is_best:
                save_checkpoint(args.output_dir / "best.pt", model, optimizer, scheduler, epoch, best_val_acc, args)
                print(f"Saved new best checkpoint: val_acc={best_val_acc:.4f}")

            write_json(args.output_dir / "history.json", history)

    write_json(args.output_dir / "history.json", history)

    best_path = args.output_dir / "best.pt"
    selected_checkpoint = None
    if best_path.exists():
        checkpoint = load_trusted_checkpoint(best_path, map_location=device)
        model.load_state_dict(strip_module_prefix(extract_model_state(checkpoint)), strict=True)
        best_epoch = int(checkpoint.get("epoch", 0)) or best_epoch
        best_val_acc = float(checkpoint.get("best_val_acc", best_val_acc))
        selected_checkpoint = best_path
        print(f"Loaded best checkpoint for test: {best_path}")
    elif args.test_only and args.resume is None:
        print("No best checkpoint found; evaluating current model.")
    elif args.resume is not None:
        selected_checkpoint = args.resume

    test_loss, test_acc = evaluate(model, test_loader, device, use_amp, args.no_progress, "test")
    print(f"Test loss={test_loss:.4f} test_acc={test_acc:.4f}")

    if args.profile_macs:
        maybe_profile_macs(model, image_size, device)

    total_runtime_seconds = time.perf_counter() - run_start_time
    result = {
        "args": vars(args),
        "compression_stats": compression_stats,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "history": history,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "total_runtime_seconds": total_runtime_seconds,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "amp": use_amp,
        "total_model_params": total_model_params,
        "trainable_params": trainable_params,
        "selected_checkpoint": selected_checkpoint,
    }
    result_path = args.output_dir / "result.json"
    write_json(result_path, result)
    print(f"Saved final experiment result: {result_path}")


if __name__ == "__main__":
    main()
