#!/usr/bin/env python
"""Smoke test: load the int8-quantized LTX-2 transformer onto the GPU via the
fast (prequantized) path, and report VRAM. Validates Phase 1+2 end-to-end on the
real inference machine — no gemma / upscalers / backend needed.

Usage:
    ./milimov/bin/python scripts/test_load_int8.py --mode int8-quanto
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="int8-quanto")
    ap.add_argument("--ckpt", default=None,
                    help="Checkpoint path (default: LTX-2/models/checkpoints/ltx-2-19b-distilled.safetensors)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ckpt = args.ckpt or os.path.join(repo, "LTX-2", "models", "checkpoints", "ltx-2-19b-distilled.safetensors")
    if not os.path.exists(ckpt):
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available — this test must run on the GPU machine.", file=sys.stderr)
        return 1

    from ltx_pipelines.utils.model_ledger import ModelLedger  # noqa: PLC0415

    from ltx_core.quantization import prequantized_paths  # noqa: PLC0415

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)")
    print(f"checkpoint: {ckpt}")

    w, q = prequantized_paths(ckpt, args.mode, ())
    print(f"expected prequantized (base): {os.path.basename(w)}  exists={os.path.exists(w)}")
    if not (os.path.exists(w) and os.path.exists(q)):
        print("ERROR: prequantized files for the base variant not found next to the checkpoint.", file=sys.stderr)
        return 1

    ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=ckpt,
        quant_mode=args.mode,
    )

    torch.cuda.reset_peak_memory_stats()
    print(f"\n[load] building transformer via fast path ({args.mode})...", flush=True)
    transformer = ledger.transformer()

    n_params = sum(p.numel() for p in transformer.parameters())
    print("\n=== OK ===")
    print(f"loaded: {type(transformer).__name__}  ({n_params/1e9:.1f}B params)")
    print(f"VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"VRAM peak:      {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print("=> int8 transformer loaded on the RTX 3090. Phase 1+2 validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
