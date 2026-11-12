from pathlib import Path

import torch
from torch import nn
from torchvision.models import vgg16

from fcttd.layers import TuckerKruskalLinear


def load_checkpoint(path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def checkpoint_state(checkpoint):
    for key in ("model_state", "model_state_dict", "state_dict", "model"):
        if isinstance(checkpoint.get(key), dict):
            state = checkpoint[key]
            break
    else:
        state = checkpoint
    return {key.removeprefix("module."): value for key, value in state.items()}


def build_model(checkpoint):
    state = checkpoint_state(checkpoint)
    metadata = checkpoint.get("metadata", {})
    model = vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, 10)
    if "classifier.0.core_factors.0" in state:
        model.classifier[0] = TuckerKruskalLinear(
            metadata.get("in_shape", [16, 32, 49]),
            metadata.get("out_shape", [16, 16, 16]),
            metadata.get("tucker_ranks", [8] * 6),
            metadata.get("cp_rank", state["classifier.0.core_factors.0"].shape[1]),
        )
    model.load_state_dict(state, strict=True)
    return model


def model_metadata(model):
    layer = model.classifier[0]
    result = {"model": "torchvision.vgg16", "batch_normalization": False, "num_classes": 10}
    if isinstance(layer, TuckerKruskalLinear):
        result.update({
            "method": "FCTTD",
            "in_shape": list(layer.in_shape),
            "out_shape": list(layer.out_shape),
            "tucker_ranks": list(layer.tucker_ranks),
            "cp_rank": layer.cp_rank,
            "fc1_weight_parameters": sum(p.numel() for name, p in layer.named_parameters() if name != "bias"),
            "fc1_bias_parameters": 0 if layer.bias is None else layer.bias.numel(),
        })
    else:
        result["method"] = "Dense"
    return result
