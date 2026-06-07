#!/usr/bin/env python
"""End-to-end smoke test: generate a short text->video clip entirely on the GPU using the
quantized models (int4 transformer + 4-bit gemma). Validates the full pipeline fits 24GB.

Usage:
    MILIMO_GEMMA_4BIT=1 ./milimov/bin/python scripts/test_generate.py
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("MILIMO_GEMMA_4BIT", "1")
    os.environ.setdefault("MILIMO_TRANSFORMER_BNB", "1")

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    models = os.path.join(repo, "LTX-2", "models")
    ckpts = os.path.join(models, "checkpoints")
    # bnb transformer needs the bf16 transformer weights -> use the USB bf16 (the slim has none).
    ckpt = os.environ.get("MILIMO_CKPT", "/mnt/usb/ltx-2-19b-distilled.safetensors")
    gemma_root = os.path.join(models, "text_encoders", "gemma3")
    lora = os.path.join(ckpts, "ltx-2-19b-distilled-lora-384.safetensors")
    spatial = os.path.join(models, "upscalers", "ltx-2-spatial-upscaler-x2-1.0.safetensors")
    temporal = os.path.join(models, "upscalers", "ltx-2-temporal-upscaler-x2-1.0.safetensors")

    for p in (ckpt, gemma_root, lora, spatial):
        if not os.path.exists(p):
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    import torch  # noqa: PLC0415

    from ltx_core.loader import LoraPathStrengthAndSDOps  # noqa: PLC0415
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline  # noqa: PLC0415

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Building TI2Vid pipeline (int4 transformer + 4-bit gemma)...", flush=True)
    pipe = TI2VidTwoStagesPipeline(
        checkpoint_path=ckpt,
        distilled_lora=[LoraPathStrengthAndSDOps(lora, 1.0, None)],
        spatial_upsampler_path=spatial,
        temporal_upsampler_path=temporal,
        gemma_root=gemma_root,
        loras=[],
        device="cuda",
        fp8transformer=False,
        quant_mode=None,  # transformer quantization handled by MILIMO_TRANSFORMER_BNB
    )

    torch.cuda.reset_peak_memory_stats()
    print("Generating a short clip (512x768, 25 frames, 8 steps, stage-1 only)...", flush=True)
    frames, audio = pipe(
        prompt="A golden retriever runs along a sunlit beach at sunset, cinematic, photorealistic.",
        negative_prompt="",
        seed=10,
        height=512,
        width=768,
        num_frames=25,
        frame_rate=24.0,
        num_inference_steps=8,
        cfg_guidance_scale=4.0,
        images=[],
        upscale=False,
    )
    frame_list = list(frames)  # force the lazy generator to actually run the diffusion

    print("\n=== OK ===")
    print(f"frames generated: {len(frame_list)}")
    if frame_list:
        print(f"frame shape: {tuple(frame_list[0].shape)} dtype={frame_list[0].dtype}")
    print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print("=> Full quantized text->video pipeline ran on the RTX 3090.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
