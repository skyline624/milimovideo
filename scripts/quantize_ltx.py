#!/usr/bin/env python
r"""Quantize the LTX-2 19B transformer to int4/int8 (optimum-quanto) ONCE on a
high-RAM machine, and save the prequantized weights for fast low-RAM loading.

Why: quantizing from the bf16 checkpoint needs the whole transformer in RAM
(~38 GB for 19B). Run this on a box with plenty of RAM (>= 64 GB), then copy the
generated ``*.<mode>.<hash>.safetensors`` + ``*.qmap.json`` files into
``LTX-2/models/checkpoints/`` on your 24 GB-GPU machine. There, set
``MILIMO_QUANT=int4-quanto`` and the backend loads the int weights via an empty
``meta`` skeleton (never materializing the 38 GB model).

It produces BOTH variants the pipelines use:
  - "base"          : transformer with no extra LoRA (loras=())
  - "distilled-lora": transformer with the distilled LoRA-384 fused in

--------------------------------------------------------------------------------
WINDOWS USAGE (inside the milimov venv)
--------------------------------------------------------------------------------
PowerShell:
    .\milimov\Scripts\python.exe scripts\quantize_ltx.py --download --mode int4-quanto

cmd.exe:
    milimov\Scripts\python.exe scripts\quantize_ltx.py --download --mode int4-quanto

Linux/macOS:
    ./milimov/bin/python scripts/quantize_ltx.py --download --mode int4-quanto

Options:
    --download            Download the bf16 checkpoint + distilled LoRA from HuggingFace first.
    --mode  int4-quanto   Quantization mode: int4-quanto (default) | int8-quanto | int2-quanto.
    --models-dir PATH     Override the LTX-2/models directory (default: <repo>/LTX-2/models).
    --device cuda|cpu     Device used for quantization (default: cuda if available, else cpu).
    --no-lora-variant     Only quantize the base variant (skip the distilled-LoRA one).
    --ckpt-name NAME      Override the checkpoint filename.
    --lora-name NAME      Override the distilled LoRA filename.
    --repo  REPO_ID       Override the HuggingFace repo id (default: Lightricks/LTX-2).
    --threads N           CPU threads for quantization (sets OMP/MKL + torch.set_num_threads).
                          Default = PyTorch's default (physical cores). Hyperthreads rarely help.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys


# Defaults from the project README (https://github.com/mainza-ai/milimovideo)
DEFAULT_REPO = "Lightricks/LTX-2"
DEFAULT_CKPT = "ltx-2-19b-distilled.safetensors"
DEFAULT_LORA = "ltx-2-19b-distilled-lora-384.safetensors"


def _add_local_packages_to_path() -> None:
    """Allow running without `pip install -e` by adding the vendored package srcs."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    for pkg in ("ltx-core", "ltx-pipelines"):
        src = os.path.join(repo, "LTX-2", "packages", pkg, "src")
        if os.path.isdir(src) and src not in sys.path:
            sys.path.insert(0, src)


def _resolve_models_dir(arg: str | None) -> str:
    if arg:
        return arg
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    return os.path.join(repo, "LTX-2", "models")


def download_models(repo: str, models_dir: str, ckpt_name: str, lora_name: str, want_lora: bool) -> None:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    checkpoints_dir = os.path.join(models_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    targets = [ckpt_name] + ([lora_name] if want_lora else [])
    for name in targets:
        dest = os.path.join(checkpoints_dir, name)
        if os.path.exists(dest):
            print(f"[skip] already present: {dest}")
            continue
        print(f"[download] {repo}/{name} -> {checkpoints_dir}")
        hf_hub_download(repo_id=repo, filename=name, local_dir=checkpoints_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description="Quantize LTX-2 19B transformer (optimum-quanto).")
    ap.add_argument("--mode", default="int4-quanto",
                    choices=["int2-quanto", "int4-quanto", "int8-quanto", "fp8-quanto"])
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--device", default=None, choices=["cuda", "cpu", "mps"])
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--no-lora-variant", action="store_true")
    ap.add_argument("--ckpt-name", default=DEFAULT_CKPT)
    ap.add_argument("--lora-name", default=DEFAULT_LORA)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads for quantization (default: PyTorch's default = physical cores). "
                         "Going beyond physical cores (into hyperthreads) rarely helps.")
    args = ap.parse_args()

    # Must be set BEFORE torch is imported to influence the OpenMP/MKL thread pools.
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    _add_local_packages_to_path()

    models_dir = _resolve_models_dir(args.models_dir)
    ckpt_path = os.path.join(models_dir, "checkpoints", args.ckpt_name)
    lora_path = os.path.join(models_dir, "checkpoints", args.lora_name)
    want_lora = not args.no_lora_variant

    if args.download:
        download_models(args.repo, models_dir, args.ckpt_name, args.lora_name, want_lora)

    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}\n"
              f"Run with --download, or place the file there manually.", file=sys.stderr)
        return 1
    if want_lora and not os.path.exists(lora_path):
        print(f"WARNING: distilled LoRA not found ({lora_path}); skipping the LoRA variant.")
        want_lora = False

    import torch  # noqa: PLC0415

    if args.threads:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(args.threads)
        except RuntimeError:
            pass  # interop threads can only be set once / before any parallel work
        print(f"[threads] requested {args.threads} | torch.get_num_threads()={torch.get_num_threads()}")

    from ltx_core.loader.primitives import LoraPathStrengthAndSDOps  # noqa: PLC0415
    from ltx_core.loader.registry import DummyRegistry  # noqa: PLC0415
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder  # noqa: PLC0415
    from ltx_core.model.transformer import (  # noqa: PLC0415
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXModelConfigurator,
    )
    from ltx_core.quantization import prequantized_paths, quantize_model, save_quantized  # noqa: PLC0415

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("WARNING: no CUDA detected — quantizing on CPU (slower).")

    # Variants must match how ModelLedger builds each stage:
    #   base stage          -> loras = ()
    #   distilled-lora stage -> loras = (distilled-lora-384,)
    variants: list[tuple[str, tuple]] = [("base", ())]
    if want_lora:
        variants.append(
            ("distilled-lora", (LoraPathStrengthAndSDOps(lora_path, 1.0, None),))
        )

    print(f"\n=== Quantizing LTX-2 transformer: mode={args.mode}, device={device} ===")
    print(f"checkpoint: {ckpt_path}\n")

    for name, loras in variants:
        weights_path, qmap_path = prequantized_paths(ckpt_path, args.mode, loras)
        if os.path.exists(weights_path) and os.path.exists(qmap_path):
            print(f"[skip] {name}: already quantized -> {os.path.basename(weights_path)}")
            continue

        print(f"[build] {name}: loading bf16 transformer on CPU (this needs ~38 GB RAM)...")
        builder = Builder(
            model_class_configurator=LTXModelConfigurator,
            model_path=ckpt_path,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
            loras=loras,
            registry=DummyRegistry(),
        )
        inner = builder.build(device="cpu", dtype=torch.bfloat16)

        print(f"[quantize] {name}: {args.mode} block-by-block on {device}...")
        quantize_model(inner, args.mode, device=device)

        save_quantized(inner, weights_path, qmap_path)
        print(f"[done] {name}:")
        print(f"        {weights_path}")
        print(f"        {qmap_path}\n")

        del inner, builder
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("=== Finished. Copy the *.{mode}.*.safetensors + *.qmap.json files into\n"
          "    LTX-2/models/checkpoints/ on your 24GB machine, then set MILIMO_QUANT="
          f"{args.mode}. ===")
    print("Note: the bf16 checkpoint is still needed at runtime for the VAE / audio /\n"
          "text-encoder components (only the big transformer is int-quantized).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
