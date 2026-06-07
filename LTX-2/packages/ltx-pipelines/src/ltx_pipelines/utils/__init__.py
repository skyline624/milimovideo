import torch

from ltx_pipelines.utils.model_ledger import ModelLedger


def context_to_device(context, device):
    """Move a ``(video_context, audio_context)`` tuple to ``device``.

    The audio context may be ``None``. This is a no-op when the tensors are already on
    ``device`` — used so the text encoder can run on CPU (MILIMO_TEXT_ENCODER_CPU) while the
    tiny prompt embeddings are moved to the compute device before the transformer runs.
    """
    video, audio = context
    return (
        video.to(device) if video is not None else None,
        audio.to(device) if audio is not None else None,
    )


__all__ = [
    "ModelLedger",
    "context_to_device",
]
