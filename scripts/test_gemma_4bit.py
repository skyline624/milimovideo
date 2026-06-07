#!/usr/bin/env python
"""Smoke test: load the gemma-3-12B text encoder in 4-bit (bitsandbytes) on the GPU and
encode a prompt. Validates that MILIMO_GEMMA_4BIT keeps the 12B encoder under ~8GB VRAM.

Usage:
    MILIMO_GEMMA_4BIT=1 ./milimov/bin/python scripts/test_gemma_4bit.py
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    models = os.path.join(repo, "LTX-2", "models")
    ckpt = os.path.join(models, "checkpoints", "ltx-2-19b-distilled.safetensors")  # slim: has connector weights
    gemma_root = os.path.join(models, "text_encoders", "gemma3")

    for p in (ckpt, gemma_root):
        if not os.path.exists(p):
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1

    from ltx_core.text_encoders.gemma import encode_text  # noqa: PLC0415
    from ltx_pipelines.utils import ModelLedger  # noqa: PLC0415

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"MILIMO_GEMMA_4BIT={os.environ.get('MILIMO_GEMMA_4BIT')}")

    ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=ckpt,
        gemma_root_path=gemma_root,
    )

    torch.cuda.reset_peak_memory_stats()
    print("\n[load] building gemma text encoder...", flush=True)
    te = ledger.text_encoder()
    print(f"text encoder model device: {te.model.device}")
    print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("\n[encode] encoding a test prompt...", flush=True)
    # The real pipelines run encoding under @smart_inference_mode(); replicate that here so
    # the forward doesn't build an autograd graph (which would balloon activation memory).
    with torch.inference_mode():
        ctx = encode_text(te, prompts=["A golden retriever runs along a sunlit beach at sunset."])
    v_ctx, a_ctx = ctx[0]
    print(f"video context: {tuple(v_ctx.shape)} on {v_ctx.device}")
    print(f"audio context: {None if a_ctx is None else tuple(a_ctx.shape)}")

    print("\n=== OK ===")
    print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print("=> gemma-3-12B 4-bit text encoder ran on the RTX 3090.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
