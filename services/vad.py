import numpy as np
import torch


def load_vad() -> torch.nn.Module:
    """Load Silero VAD model."""
    model, *_ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    return model


def is_speech(chunk: np.ndarray, model: torch.nn.Module, sample_rate: int = 16000) -> bool:
    """Return True if the audio chunk contains speech."""
    if model is None:
        return True
    prob = model(torch.from_numpy(chunk).float(), sample_rate).item()
    return prob > 0.5
