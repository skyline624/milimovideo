#!/usr/bin/env python
"""Build + bitsandbytes-4bit-quantize + SAVE both transformer variants (base and
distilled-lora) from the bf16 checkpoint, so runtime loads them from local disk (~10GB each)
instead of re-reading the 43GB bf16 over USB and re-quantizing.

Run once (reads the bf16 from USB, ~8 min per variant):
    ./milimov/bin/python scripts/save_transformer_bnb.py \
        --ckpt /mnt/usb/ltx-2-19b-distilled.safetensors \
        --lora LTX-2/models/checkpoints/ltx-2-19b-distilled-lora-384.safetensors \
        --out-dir LTX-2/models/checkpoints
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/usb/ltx-2-19b-distilled.safetensors",
                    help="bf16 checkpoint holding the transformer weights (e.g. the USB bf16).")
    ap.add_argument("--lora", default=None, help="distilled LoRA path (for the distilled variant).")
    ap.add_argument("--out-dir", default=None, help="local dir to write the bnb 4-bit files into.")
    ap.add_argument("--no-lora-variant", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    out_dir = args.out_dir or os.path.join(repo, "LTX-2", "models", "checkpoints")
    lora = args.lora or os.path.join(out_dir, "ltx-2-19b-distilled-lora-384.safetensors")

    os.environ["MILIMO_TRANSFORMER_BNB"] = "1"
    os.environ["MILIMO_TRANSFORMER_BNB_SAVE"] = "1"
    os.environ["MILIMO_BNB_DIR"] = out_dir

    if not os.path.exists(args.ckpt):
        print(f"ERROR: missing bf16 checkpoint {args.ckpt}", file=sys.stderr)
        return 1

    import torch  # noqa: PLC0415

    from ltx_core.loader import LoraPathStrengthAndSDOps  # noqa: PLC0415
    from ltx_core.quantization import bnb_transformer_path  # noqa: PLC0415
    from ltx_pipelines.utils import ModelLedger  # noqa: PLC0415

    device = torch.device("cuda")
    variants: list[tuple[str, list]] = [("base", [])]
    if not args.no_lora_variant and os.path.exists(lora):
        variants.append(("distilled-lora", [LoraPathStrengthAndSDOps(lora, 1.0, None)]))

    for name, loras in variants:
        out = bnb_transformer_path(out_dir, tuple(loras))
        if os.path.exists(out):
            print(f"[skip] {name}: already saved -> {os.path.basename(out)}")
            continue
        print(f"\n[{name}] building bf16 + bnb 4-bit quantize + save (reads bf16 from {args.ckpt})...", flush=True)
        t0 = time.time()
        ledger = ModelLedger(dtype=torch.bfloat16, device=device, checkpoint_path=args.ckpt, loras=loras)
        tr = ledger.transformer()  # builds, quantizes, saves (env SAVE set)
        print(f"[{name}] done in {time.time()-t0:.0f}s -> {out} ({os.path.getsize(out)/1e9:.1f} GB)")
        del tr, ledger
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== Saved bnb 4-bit transformers. Runtime can now load them from local disk ===")
    print("   (point the checkpoint at the local slim; the bf16/USB is no longer needed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
