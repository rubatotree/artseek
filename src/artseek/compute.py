from __future__ import annotations

import gc
import shutil

import torch

CUDA_WHEEL_INDEX = "https://download.pytorch.org/whl/cu130"


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float32":
        return torch.float32
    return torch.float16 if device.type == "cuda" else torch.float32


def configure_inference(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def release_accelerator() -> None:
    """Drop cached GPU blocks after a model has been deleted."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return
    mps = getattr(torch, "mps", None)
    empty_cache = getattr(mps, "empty_cache", None) if mps is not None else None
    backends = getattr(torch, "backends", None)
    mps_backend = getattr(backends, "mps", None) if backends is not None else None
    if callable(empty_cache) and mps_backend is not None and mps_backend.is_available():
        empty_cache()


def nvidia_gpu_detected() -> bool:
    if torch.cuda.is_available():
        return True
    return shutil.which("nvidia-smi") is not None


def describe_runtime(requested: str = "auto") -> str:
    device = resolve_device(requested)
    dtype = resolve_dtype("auto", device)
    if device.type == "cuda" and torch.cuda.is_available():
        return f"{device} · {torch.cuda.get_device_name(0)} · {dtype} · {torch.__version__}"
    return f"{device} · {dtype} · {torch.__version__}"


def cuda_mismatch_hint() -> str | None:
    if torch.cuda.is_available():
        return None
    if not nvidia_gpu_detected():
        return None
    return (
        "检测到 NVIDIA GPU，但当前 PyTorch 是 CPU 构建，索引会走 CPU。"
        "请在项目根目录执行：scripts/install_cuda_torch.ps1"
        f"（或 uv pip uninstall torch torchvision 后，"
        f"uv pip install torch torchvision --index-url {CUDA_WHEEL_INDEX}）"
    )
