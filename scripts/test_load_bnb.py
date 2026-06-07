#!/usr/bin/env python
"""Smoke test: load the LTX-2 19B transformer in 4-bit via bitsandbytes (nf4) on the GPU,
building the bf16 weights from the (USB) checkpoint. Reports load time + VRAM.

Usage:
    MILIMO_TRANSFORMER_BNB=1 ./milimov/bin/python scripts/test_load_bnb.py [--ckpt PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/usb/ltx-2-19b-distilled.safetensors",
                    help="bf16 checkpoint with the transformer weights (e.g. the USB bf16).")
    args = ap.parse_args()
    os.environ.setdefault("MILIMO_TRANSFORMER_BNB", "1")

    if not os.path.exists(args.ckpt):
        print(f"ERROR: checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 1

    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1

    from ltx_pipelines.utils import ModelLedger  # noqa: PLC0415

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"checkpoint (bf16): {args.ckpt}")

    ledger = ModelLedger(dtype=torch.bfloat16, device=device, checkpoint_path=args.ckpt)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    print("\n[load] building transformer (bf16 mmap) + bitsandbytes 4-bit...", flush=True)
    transformer = ledger.transformer()
    dt = time.time() - t0

    print("\n=== OK ===")
    print(f"loaded: {type(transformer).__name__} in {dt:.1f}s")
    print(f"VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"VRAM peak:      {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print("=> LTX-2 19B transformer loaded in 4-bit (bitsandbytes) on the RTX 3090.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
