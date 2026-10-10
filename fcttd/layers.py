import math
from typing import Iterable, Optional, Sequence, Tuple, Union

import tensorly as tl
import torch
import torch.nn as nn
from tensorly.decomposition import parafac, tucker


tl.set_backend("pytorch")


def prod(values: Iterable[int]) -> int:
    return math.prod(int(v) for v in values)


def _as_tuple(values: Sequence[int], length: int, name: str) -> Tuple[int, ...]:
    values = tuple(int(v) for v in values)
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} integers, got {values}.")
    if any(v <= 0 for v in values):
        raise ValueError(f"{name} must contain positive integers, got {values}.")
    return values


def _tucker(weight_tensor: torch.Tensor, ranks: Sequence[int], n_iter_max: int):
    try:
        return tucker(weight_tensor, rank=list(ranks), init="svd", n_iter_max=n_iter_max)
    except TypeError:
        return tucker(weight_tensor, ranks=list(ranks), init="svd", n_iter_max=n_iter_max)


def _relative_error(absolute_error: float, reference_norm: float) -> float:
    if reference_norm > 0.0:
        return absolute_error / reference_norm
    return 0.0 if absolute_error == 0.0 else math.inf


def _factor_orthogonality_diagnostics(factors: Sequence[torch.Tensor]) -> dict:
    frobenius_errors = []
    relative_frobenius_errors = []
    max_absolute_errors = []
    for factor in factors:
        rank = factor.shape[1]
        identity = torch.eye(rank, device=factor.device, dtype=factor.dtype)
        error = factor.transpose(0, 1) @ factor - identity
        frobenius_error = torch.linalg.matrix_norm(error).item()
        frobenius_errors.append(frobenius_error)
        relative_frobenius_errors.append(frobenius_error / math.sqrt(rank))
        max_absolute_errors.append(error.abs().max().item())

    return {
        "per_factor_frobenius_error": frobenius_errors,
        "per_factor_relative_frobenius_error": relative_frobenius_errors,
        "per_factor_max_absolute_error": max_absolute_errors,
        "max_frobenius_error": max(frobenius_errors, default=0.0),
        "max_relative_frobenius_error": max(relative_frobenius_errors, default=0.0),
        "max_absolute_error": max(max_absolute_errors, default=0.0),
    }


def _multilinear_image_norm(tensor: torch.Tensor, factors: Sequence[torch.Tensor]) -> float:
    """Return the norm after applying one factor matrix along each tensor mode."""
    transformed = tensor
    for mode, factor in enumerate(factors):
        gram = factor.transpose(0, 1) @ factor
        transformed = torch.tensordot(gram, transformed, dims=([1], [mode])).movedim(0, mode)
    squared_norm = torch.sum(tensor * transformed).clamp_min(0.0)
    return torch.sqrt(squared_norm).item()


class TuckerKruskalLinear(nn.Module):
    """Linear layer whose weight is Tucker-decomposed and whose core is CP/Kruskal-decomposed.

    The represented weight has shape:
        out_shape[0] x out_shape[1] x out_shape[2] x in_shape[0] x in_shape[1] x in_shape[2]

    The Tucker factors map the original input/output dimensions to the Tucker ranks.
    The Tucker core is not stored directly; it is represented by CP factors.
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        tucker_ranks: Sequence[int],
        cp_rank: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        super().__init__()
        self.in_shape = _as_tuple(in_shape, 3, "in_shape")
        self.out_shape = _as_tuple(out_shape, 3, "out_shape")
        self.tucker_ranks = _as_tuple(tucker_ranks, 6, "tucker_ranks")
        self.cp_rank = int(cp_rank)
        if self.cp_rank <= 0:
            raise ValueError(f"cp_rank must be positive, got {self.cp_rank}.")

        self.in_features = prod(self.in_shape)
        self.out_features = prod(self.out_shape)

        factory_kwargs = {"dtype": dtype, "device": device}

        self.output_factors = nn.ParameterList(
            nn.Parameter(torch.empty(dim, rank, **factory_kwargs))
            for dim, rank in zip(self.out_shape, self.tucker_ranks[:3])
        )
        self.input_factors = nn.ParameterList(
            nn.Parameter(torch.empty(dim, rank, **factory_kwargs))
            for dim, rank in zip(self.in_shape, self.tucker_ranks[3:])
        )
        self.core_factors = nn.ParameterList(
            nn.Parameter(torch.empty(rank, self.cp_rank, **factory_kwargs))
            for rank in self.tucker_ranks
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for factor in list(self.output_factors) + list(self.input_factors) + list(self.core_factors):
            nn.init.normal_(factor, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _effective_mode_factors(self) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Combine Tucker and CP factors without materializing the dense Tucker core.

        If ``U_k`` is an outer Tucker factor and ``C_k`` is the corresponding
        CP factor of the Tucker core, then ``U_k @ C_k`` is the CP factor in the
        original tensor mode.  Keeping this multiplication in the autograd graph
        preserves the exact parameterization and gradients.
        """
        outer_factors = list(self.output_factors) + list(self.input_factors)
        effective = [
            outer @ core for outer, core in zip(outer_factors, self.core_factors)
        ]
        return effective[:3], effective[3:]

    @staticmethod
    def _khatri_rao_three(factors: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(factors) != 3:
            raise ValueError(f"Expected three factors, got {len(factors)}.")
        first, second, third = factors
        if not (first.shape[1] == second.shape[1] == third.shape[1]):
            raise ValueError(
                "All Khatri-Rao factors must have the same column count, got "
                f"{[tuple(factor.shape) for factor in factors]}."
            )
        return torch.einsum("ir,jr,kr->ijkr", first, second, third).reshape(
            -1, first.shape[1]
        )

    @classmethod
    def from_tucker_linear(
        cls,
        tucker_layer: "TuckerLinear",
        cp_rank: int,
        decompose_device: Union[torch.device, str],
        n_iter_cp: int = 200,
        cp_tol: float = 1e-6,
    ) -> Tuple["TuckerKruskalLinear", dict]:
        target_device = tucker_layer.core.device
        target_dtype = tucker_layer.core.dtype
        layer = cls(
            in_shape=tucker_layer.in_shape,
            out_shape=tucker_layer.out_shape,
            tucker_ranks=tucker_layer.tucker_ranks,
            cp_rank=cp_rank,
            bias=tucker_layer.bias is not None,
            dtype=target_dtype,
            device=target_device,
        )

        core = tucker_layer.core.detach().to(device=decompose_device, dtype=torch.float32).contiguous()
        with torch.no_grad():
            cp_weights, cp_factors = parafac(
                core,
                rank=cp_rank,
                init="svd",
                n_iter_max=n_iter_cp,
                tol=cp_tol,
                normalize_factors=False,
            )

            cp_factors = list(cp_factors)
            if cp_weights is not None:
                cp_factors[0] = cp_factors[0] * cp_weights.reshape(1, -1)

            for dst, src in zip(layer.core_factors, cp_factors):
                dst.copy_(src.to(device=target_device, dtype=target_dtype))
            for dst, src in zip(layer.output_factors, tucker_layer.output_factors):
                dst.copy_(src.detach().to(device=target_device, dtype=target_dtype))
            for dst, src in zip(layer.input_factors, tucker_layer.input_factors):
                dst.copy_(src.detach().to(device=target_device, dtype=target_dtype))
            if layer.bias is not None and tucker_layer.bias is not None:
                layer.bias.copy_(tucker_layer.bias.detach().to(device=target_device, dtype=target_dtype))

            reconstructed_core = layer.reconstruct_core(device=decompose_device, dtype=torch.float32)
            absolute_error = torch.linalg.vector_norm(reconstructed_core - core).item()
            relative_error = absolute_error / torch.linalg.vector_norm(core).item()
            reconstructed_weight = layer.reconstruct_weight(device=decompose_device, dtype=torch.float32)
            source_weight = tucker_layer.reconstruct_weight(device=decompose_device, dtype=torch.float32)
            absolute_weight_error = torch.linalg.vector_norm(reconstructed_weight - source_weight).item()
            relative_weight_error = absolute_weight_error / torch.linalg.vector_norm(source_weight).item()

        stats = {
            "absolute_core_error": absolute_error,
            "relative_core_error": relative_error,
            "absolute_weight_error": absolute_weight_error,
            "relative_weight_error": relative_weight_error,
        }
        return layer, stats

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        tucker_ranks: Sequence[int],
        cp_rank: int,
        decompose_device: torch.device | str,
        n_iter_tucker: int = 10,
        n_iter_cp: int = 200,
        cp_tol: float = 1e-6,
        return_stats: bool = False,
    ) -> Union["TuckerKruskalLinear", Tuple["TuckerKruskalLinear", dict]]:
        in_shape = _as_tuple(in_shape, 3, "in_shape")
        out_shape = _as_tuple(out_shape, 3, "out_shape")
        tucker_ranks = _as_tuple(tucker_ranks, 6, "tucker_ranks")

        if linear.in_features != prod(in_shape):
            raise ValueError(
                f"Linear in_features={linear.in_features}, but prod(in_shape)={prod(in_shape)}."
            )
        if linear.out_features != prod(out_shape):
            raise ValueError(
                f"Linear out_features={linear.out_features}, but prod(out_shape)={prod(out_shape)}."
            )

        target_device = linear.weight.device
        target_dtype = linear.weight.dtype
        layer = cls(
            in_shape=in_shape,
            out_shape=out_shape,
            tucker_ranks=tucker_ranks,
            cp_rank=cp_rank,
            bias=linear.bias is not None,
            dtype=target_dtype,
            device=target_device,
        )

        weight_tensor = (
            linear.weight.detach()
            .to(device=decompose_device, dtype=torch.float32)
            .reshape(*out_shape, *in_shape)
            .contiguous()
        )

        with torch.no_grad():
            core, tucker_factors = _tucker(weight_tensor, tucker_ranks, n_iter_tucker)
            tucker_factors = list(tucker_factors)
            cp_weights, cp_factors = parafac(
                core,
                rank=cp_rank,
                init="svd",
                n_iter_max=n_iter_cp,
                tol=cp_tol,
                normalize_factors=False,
            )

            cp_factors = list(cp_factors)
            if cp_weights is not None:
                cp_factors[0] = cp_factors[0] * cp_weights.reshape(1, -1)

            for dst, src in zip(layer.output_factors, tucker_factors[:3]):
                dst.copy_(src.to(device=target_device, dtype=target_dtype))
            for dst, src in zip(layer.input_factors, tucker_factors[3:]):
                dst.copy_(src.to(device=target_device, dtype=target_dtype))
            for dst, src in zip(layer.core_factors, cp_factors):
                dst.copy_(src.to(device=target_device, dtype=target_dtype))
            if layer.bias is not None and linear.bias is not None:
                layer.bias.copy_(linear.bias.detach().to(device=target_device, dtype=target_dtype))

            if return_stats:
                reconstructed_core = torch.einsum(
                    "ar,br,cr,dr,er,fr->abcdef",
                    cp_factors[0],
                    cp_factors[1],
                    cp_factors[2],
                    cp_factors[3],
                    cp_factors[4],
                    cp_factors[5],
                )
                original_weight_norm = torch.linalg.vector_norm(weight_tensor).item()
                tucker_core_norm = torch.linalg.vector_norm(core).item()
                kruskal_core_norm = torch.linalg.vector_norm(reconstructed_core).item()

                # Tucker factors returned by HOOI have orthonormal columns, making
                # the Tucker reconstruction an orthogonal projection of the weight.
                tucker_error_squared = max(original_weight_norm**2 - tucker_core_norm**2, 0.0)
                tucker_absolute_weight_error = math.sqrt(tucker_error_squared)
                tucker_relative_weight_error = _relative_error(
                    tucker_absolute_weight_error,
                    original_weight_norm,
                )

                core_delta = reconstructed_core - core
                absolute_core_error = torch.linalg.vector_norm(core_delta).item()
                relative_core_error = _relative_error(absolute_core_error, tucker_core_norm)
                kruskal_increment_absolute_weight_error = _multilinear_image_norm(
                    core_delta,
                    tucker_factors,
                )
                kruskal_increment_relative_weight_error = _relative_error(
                    kruskal_increment_absolute_weight_error,
                    original_weight_norm,
                )

                # The Tucker projection residual is orthogonal to every tensor in
                # its multilinear subspace, including the propagated CP residual.
                absolute_weight_error = math.hypot(
                    tucker_absolute_weight_error,
                    kruskal_increment_absolute_weight_error,
                )
                relative_weight_error = _relative_error(absolute_weight_error, original_weight_norm)
                stats = {
                    "original_weight_norm": original_weight_norm,
                    "tucker_core_norm": tucker_core_norm,
                    "kruskal_core_norm": kruskal_core_norm,
                    "tucker_absolute_weight_error": tucker_absolute_weight_error,
                    "tucker_relative_weight_error": tucker_relative_weight_error,
                    "absolute_core_error": absolute_core_error,
                    "relative_core_error": relative_core_error,
                    "kruskal_increment_absolute_weight_error": kruskal_increment_absolute_weight_error,
                    "kruskal_increment_relative_weight_error": kruskal_increment_relative_weight_error,
                    "absolute_weight_error": absolute_weight_error,
                    "relative_weight_error": relative_weight_error,
                    "factor_orthogonality": _factor_orthogonality_diagnostics(tucker_factors),
                }
                return layer, stats

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        x = x.reshape(-1, self.in_features)
        effective_output, effective_input = self._effective_mode_factors()
        output_matrix = self._khatri_rao_three(effective_output)
        input_matrix = self._khatri_rao_three(effective_input)
        latent = x @ input_matrix
        output = (latent @ output_matrix.transpose(0, 1)).reshape(
            *leading_shape, self.out_features
        )

        if self.bias is not None:
            output = output + self.bias
        return output

    def reconstruct_weight(
        self,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        output_factors, input_factors = self._effective_mode_factors()
        if device is not None or dtype is not None:
            output_factors = [factor.to(device=device, dtype=dtype) for factor in output_factors]
            input_factors = [factor.to(device=device, dtype=dtype) for factor in input_factors]
        output_matrix = self._khatri_rao_three(output_factors)
        input_matrix = self._khatri_rao_three(input_factors)
        return output_matrix @ input_matrix.transpose(0, 1)

    def reconstruct_core(
        self,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        core_factors = list(self.core_factors)
        if device is not None or dtype is not None:
            core_factors = [factor.to(device=device, dtype=dtype) for factor in core_factors]
        return torch.einsum(
            "ar,br,cr,dr,er,fr->abcdef",
            core_factors[0],
            core_factors[1],
            core_factors[2],
            core_factors[3],
            core_factors[4],
            core_factors[5],
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"in_shape={self.in_shape}, out_shape={self.out_shape}, "
            f"tucker_ranks={self.tucker_ranks}, cp_rank={self.cp_rank}, "
            f"bias={self.bias is not None}"
        )


class TuckerLinear(nn.Module):
    """Linear layer with a Tucker-decomposed 6-D weight tensor."""

    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        tucker_ranks: Sequence[int],
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        super().__init__()
        self.in_shape = _as_tuple(in_shape, 3, "in_shape")
        self.out_shape = _as_tuple(out_shape, 3, "out_shape")
        self.tucker_ranks = _as_tuple(tucker_ranks, 6, "tucker_ranks")
        self.in_features = prod(self.in_shape)
        self.out_features = prod(self.out_shape)

        factory_kwargs = {"dtype": dtype, "device": device}
        self.core = nn.Parameter(torch.empty(*self.tucker_ranks, **factory_kwargs))
        self.output_factors = nn.ParameterList(
            nn.Parameter(torch.empty(dim, rank, **factory_kwargs))
            for dim, rank in zip(self.out_shape, self.tucker_ranks[:3])
        )
        self.input_factors = nn.ParameterList(
            nn.Parameter(torch.empty(dim, rank, **factory_kwargs))
            for dim, rank in zip(self.in_shape, self.tucker_ranks[3:])
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.core, mean=0.0, std=0.02)
        for factor in list(self.output_factors) + list(self.input_factors):
            nn.init.normal_(factor, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        tucker_ranks: Sequence[int],
        decompose_device: Union[torch.device, str],
        n_iter_tucker: int = 10,
    ) -> Tuple["TuckerLinear", dict]:
        in_shape = _as_tuple(in_shape, 3, "in_shape")
        out_shape = _as_tuple(out_shape, 3, "out_shape")
        tucker_ranks = _as_tuple(tucker_ranks, 6, "tucker_ranks")

        if linear.in_features != prod(in_shape):
            raise ValueError(
                f"Linear in_features={linear.in_features}, but prod(in_shape)={prod(in_shape)}."
            )
        if linear.out_features != prod(out_shape):
            raise ValueError(
                f"Linear out_features={linear.out_features}, but prod(out_shape)={prod(out_shape)}."
            )

        target_device = linear.weight.device
        target_dtype = linear.weight.dtype
        layer = cls(
            in_shape=in_shape,
            out_shape=out_shape,
            tucker_ranks=tucker_ranks,
            bias=linear.bias is not None,
            dtype=target_dtype,
            device=target_device,
        )

        weight_tensor = (
            linear.weight.detach()
            .to(device=decompose_device, dtype=torch.float32)
            .reshape(*out_shape, *in_shape)
            .contiguous()
        )

        with torch.no_grad():
            core, tucker_factors = _tucker(weight_tensor, tucker_ranks, n_iter_tucker)

            layer.core.copy_(core.to(device=target_device, dtype=target_dtype))
            for dst, src in zip(layer.output_factors, tucker_factors[:3]):
                dst.copy_(src.to(device=target_device, dtype=target_dtype))
            for dst, src in zip(layer.input_factors, tucker_factors[3:]):
                dst.copy_(src.to(device=target_device, dtype=target_dtype))
            if layer.bias is not None and linear.bias is not None:
                layer.bias.copy_(linear.bias.detach().to(device=target_device, dtype=target_dtype))

            reconstructed = layer.reconstruct_weight(device=decompose_device, dtype=torch.float32)
            original = linear.weight.detach().to(device=decompose_device, dtype=torch.float32)
            absolute_error = torch.linalg.vector_norm(reconstructed - original).item()
            relative_error = absolute_error / torch.linalg.vector_norm(original).item()

        stats = {
            "absolute_weight_error": absolute_error,
            "relative_weight_error": relative_error,
        }
        return layer, stats

    def reconstruct_weight(
        self,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        core = self.core
        output_factors = list(self.output_factors)
        input_factors = list(self.input_factors)
        if device is not None or dtype is not None:
            core = core.to(device=device, dtype=dtype)
            output_factors = [factor.to(device=device, dtype=dtype) for factor in output_factors]
            input_factors = [factor.to(device=device, dtype=dtype) for factor in input_factors]

        weight = torch.einsum(
            "abcdef,ia,jb,kc,ld,me,nf->ijklmn",
            core,
            output_factors[0],
            output_factors[1],
            output_factors[2],
            input_factors[0],
            input_factors[1],
            input_factors[2],
        )
        return weight.reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        x = x.reshape(-1, self.in_features).reshape(-1, *self.in_shape)

        compressed_input = torch.einsum(
            "nabc,ad,be,cf->ndef",
            x,
            self.input_factors[0],
            self.input_factors[1],
            self.input_factors[2],
        )
        hidden = torch.einsum("abcdef,ndef->nabc", self.core, compressed_input)
        output = torch.einsum(
            "nabc,ia,jb,kc->nijk",
            hidden,
            self.output_factors[0],
            self.output_factors[1],
            self.output_factors[2],
        ).reshape(*leading_shape, self.out_features)

        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"in_shape={self.in_shape}, out_shape={self.out_shape}, "
            f"tucker_ranks={self.tucker_ranks}, bias={self.bias is not None}"
        )


def tucker_kruskal_parameter_count(
    in_shape: Sequence[int],
    out_shape: Sequence[int],
    tucker_ranks: Sequence[int],
    cp_rank: int,
    bias: bool = True,
) -> int:
    return tucker_kruskal_parameter_breakdown(
        in_shape,
        out_shape,
        tucker_ranks,
        cp_rank,
        bias=bias,
    )["total_params"]


def tucker_kruskal_parameter_breakdown(
    in_shape: Sequence[int],
    out_shape: Sequence[int],
    tucker_ranks: Sequence[int],
    cp_rank: int,
    bias: bool = True,
) -> dict:
    in_shape = _as_tuple(in_shape, 3, "in_shape")
    out_shape = _as_tuple(out_shape, 3, "out_shape")
    tucker_ranks = _as_tuple(tucker_ranks, 6, "tucker_ranks")
    cp_rank = int(cp_rank)
    if cp_rank <= 0:
        raise ValueError(f"cp_rank must be positive, got {cp_rank}.")

    tucker_factor_params = sum(dim * rank for dim, rank in zip(out_shape + in_shape, tucker_ranks))
    core_cp_params = sum(rank * cp_rank for rank in tucker_ranks)
    bias_params = prod(out_shape) if bias else 0
    return {
        "outer_tucker_factor_params": tucker_factor_params,
        "kruskal_core_factor_params": core_cp_params,
        "bias_params": bias_params,
        "total_params": tucker_factor_params + core_cp_params + bias_params,
        "dense_tucker_core_elements": prod(tucker_ranks),
    }


def tucker_parameter_count(
    in_shape: Sequence[int],
    out_shape: Sequence[int],
    tucker_ranks: Sequence[int],
    bias: bool = True,
) -> int:
    in_shape = _as_tuple(in_shape, 3, "in_shape")
    out_shape = _as_tuple(out_shape, 3, "out_shape")
    tucker_ranks = _as_tuple(tucker_ranks, 6, "tucker_ranks")

    core_params = prod(tucker_ranks)
    factor_params = sum(dim * rank for dim, rank in zip(out_shape + in_shape, tucker_ranks))
    bias_params = prod(out_shape) if bias else 0
    return core_params + factor_params + bias_params


def full_linear_parameter_count(in_shape: Sequence[int], out_shape: Sequence[int], bias: bool = True) -> int:
    weight_params = prod(in_shape) * prod(out_shape)
    return weight_params + (prod(out_shape) if bias else 0)
