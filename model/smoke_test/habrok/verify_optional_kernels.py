#!/usr/bin/env python3
"""Verify the pinned optional Qwen3.5 kernel packages and their CUDA path."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from unittest.mock import patch


EXPECTED = {
    "torch": "2.14.1+cu130",
    "transformers": "5.20.2",
    "causal-conv1d": "1.7.0",
    "fla-core": "0.5.2",
    "flash-linear-attention": "0.5.2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="also execute a causal-conv1d CUDA forward/backward check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actual = {name: version(name) for name in EXPECTED}
    print("optional-kernel versions:", actual)
    if actual != EXPECTED:
        raise SystemExit(f"unexpected package versions: {actual}")

    import torch
    from causal_conv1d import causal_conv1d_fn
    from transformers.integrations.hub_kernels import (
        use_kernel_func_from_hub_with_fallback,
    )

    def torch_fallback(*args: object, **kwargs: object) -> str:
        del args, kwargs
        return "torch"

    with patch(
        "transformers.integrations.hub_kernels.use_kernel_forward_from_hub",
        return_value=lambda function: function,
    ):
        wrapped = use_kernel_func_from_hub_with_fallback(
            "chunk_gated_delta_rule", "fla"
        )(torch_fallback)

    implementation = next(
        cell.cell_contents
        for cell in wrapped.__closure__ or ()
        if callable(cell.cell_contents)
    )
    implementation_module = getattr(implementation, "__module__", "")
    print("resolved chunk_gated_delta_rule:", implementation_module)
    if implementation is torch_fallback or not implementation_module.startswith("fla."):
        raise SystemExit("Transformers selected the PyTorch gated-delta fallback")

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    print("causal_conv1d_fn:", causal_conv1d_fn.__module__)
    print("direct chunk_gated_delta_rule:", chunk_gated_delta_rule.__module__)

    if not args.require_cuda:
        print("imports passed; CUDA execution was not requested")
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA was required but is not available")

    device = torch.device("cuda")
    x = torch.randn(2, 8, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(8, 4, device=device, dtype=torch.bfloat16, requires_grad=True)
    output = causal_conv1d_fn(x, weight, activation="silu")
    output.float().sum().backward()

    if x.grad is None or weight.grad is None:
        raise SystemExit("causal-conv1d backward did not produce gradients")
    if not torch.isfinite(x.grad).all() or not torch.isfinite(weight.grad).all():
        raise SystemExit("causal-conv1d produced non-finite gradients")

    print("CUDA device:", torch.cuda.get_device_name(device))
    print("causal-conv1d CUDA forward/backward passed with finite gradients")


if __name__ == "__main__":
    main()
