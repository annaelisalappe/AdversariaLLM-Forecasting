"""
Batch processing utilities with automatic memory management.

This module provides utilities for processing data in batches with automatic
batch size adjustment when out-of-memory errors occur.
"""

import logging
import sys
from typing import Callable

import torch
from tqdm import tqdm

from ..io_utils import free_vram


# Triton (used by flash-linear-attention's kernels for Qwen3.5/3.8) reports a CUDA OOM as a bare
# RuntimeError "Triton Error [CUDA]: out of memory", not as torch.cuda.OutOfMemoryError.
TRITON_OOM_MESSAGE = "Triton Error [CUDA]: out of memory"


def _batch_size_limit_reason(exc: RuntimeError) -> str | None:
    """Why a RuntimeError means "batch too large" (retry with half), or None to re-raise it."""
    text = str(exc)
    if TRITON_OOM_MESSAGE in text:
        return "Triton CUDA OOM"
    if "canUse32BitIndexMath" in text:
        # A kernel refusing a tensor beyond 32-bit indexing, e.g. the causal conv1d of Qwen3.5/3.8's
        # linear-attention layers once batch x 8192 channels x sequence length passes 2**31 elements.
        return "32-bit index limit"
    return None


def with_max_batchsize(function: Callable, *inputs, initial_batch_size: int | None = None, verbose: bool = False, success_threshold: int = 64):
    """
    Dynamically adjust batch size if an OOM error occurs.
    Also tries increasing batch size when many batches succeed consecutively.

    Args:
        function: Callable
            A function that takes one or more arguments and returns a tensor or list of tensors.
            All input arguments should have the same length in their first dimension (batch dimension).
        *inputs:
            The inputs to pass to the function. All inputs must be tensors or lists and have the same length in their first dimension.
        initial_batch_size: int | None, default None
            Starting batch size for execution. If None, the batch size is set to the closest power of two of the input length.
        verbose: bool, default False
            Whether to print progress.
        success_threshold: int, default 64
            Number of consecutive successful batches before attempting to double batch size.
    Returns:
        The output of the function.
    """
    if not inputs:
        raise ValueError("At least one input must be provided")

    # Verify all inputs have the same length
    input_length = len(inputs[0])
    for i, inp in enumerate(inputs[1:], 1):
        if len(inp) != input_length:
            raise ValueError(f"All inputs must have the same length. Input 0 has length {input_length}, but input {i} has length {len(inp)}")

    # Handle empty input case
    if input_length == 0:
        return function(*inputs)

    outputs = []
    batch_start = 0
    successful_batches = 0

    def next_power_of_two(n):
        return 1 << (n - 1).bit_length()

    if initial_batch_size is None:
        initial_batch_size = next_power_of_two(input_length)

    batch_size = min(initial_batch_size, next_power_of_two(input_length))
    pbar = tqdm(total=input_length, desc=f"Running function b={batch_size}", file=sys.stdout) if verbose else None

    while batch_start < input_length:
        try:
            free_vram()
            # Create chunks for all inputs
            chunks = [inp[batch_start:batch_start + batch_size] for inp in inputs]
            output = function(*chunks)
            outputs.append(output)
            batch_start += batch_size  # Move to the next batch
            successful_batches += 1

            # Try to double batch size if we've had many successful batches
            if (successful_batches >= success_threshold and
                batch_size * 2 <= input_length):
                batch_size = batch_size * 2
                successful_batches = 0  # Reset counter
                if pbar is not None:
                    pbar.set_description(f"Increasing batch size to b={batch_size}")

            if pbar is not None:
                pbar.update(batch_size)
        except torch.cuda.OutOfMemoryError:
            # If we hit OOM, reduce batch size and retry the same chunk
            batch_size = batch_size // 2
            successful_batches = 0  # Reset counter after OOM
            if pbar is not None:
                pbar.set_description(f"Running function b={batch_size}")
            if batch_size < 1:
                raise RuntimeError(
                    "OOM even with batch_size=1; cannot generate further."
                )
        except RuntimeError as exc:
            reason = _batch_size_limit_reason(exc)
            if reason is None:
                raise
            prev_batch_size = batch_size
            batch_size = batch_size // 2
            successful_batches = 0
            logging.warning(
                "with_max_batchsize caught %s at batch_size=%d; retrying with batch_size=%d",
                reason,
                prev_batch_size,
                batch_size,
            )
            if pbar is not None:
                pbar.set_description(f"Running function b={batch_size}")
            if batch_size < 1:
                raise RuntimeError(f"{reason} even with batch_size=1; cannot generate further.") from exc
    if pbar is not None:
        pbar.close()

    if all(isinstance(x, torch.Tensor) for x in outputs):
        outputs = torch.cat(outputs, dim=0)
        assert len(outputs) == input_length
    elif all(isinstance(x, tuple) for x in outputs):
        # Transpose and concatenate tuple outputs
        # Handle both tensors and lists within tuples
        outputs_processed = []
        for i in range(len(outputs[0])):
            elements = [x[i] for x in outputs]
            if all(isinstance(e, torch.Tensor) for e in elements):
                outputs_processed.append(torch.cat(elements, dim=0))
            elif all(isinstance(e, list) for e in elements):
                outputs_processed.append([item for sublist in elements for item in sublist])
            else:
                types = ", ".join(f"{type(e).__name__}" for e in elements)
                raise TypeError(f"Wrapped functions may only return Tensors or lists, not {types}")
        outputs = tuple(outputs_processed)
        assert all(len(o) == input_length for o in outputs)
    elif all(isinstance(x, dict) for x in outputs):
        outputs = {k: [item for o in outputs for item in o[k]] for k in outputs[0].keys()}
        assert all(len(v) == input_length for v in outputs.values())
    else:
        outputs = [item for sublist in outputs for item in sublist]
        assert len(outputs) == input_length
    return outputs
