#!/usr/bin/env python3
# main.py — Entry point for Flint (terminal UI)
#
# For the web UI, run:
#   uvicorn web.server:app --host 0.0.0.0 --port 8000
# or:
#   python web/server.py

import sys
import os
from pathlib import Path

# Ensure the project root is on sys.path regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Guard early — several modules use Python 3.10+ syntax (union types, list[int], etc.)
if sys.version_info < (3, 10):
    print(
        f"Error: Python 3.10 or newer is required.\n"
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}\n"
        f"Install 3.10+ from https://python.org"
    )
    sys.exit(1)


def check_requirements():
    """Verify dependencies before loading the model."""
    errors   = []
    warnings = []

    import config

    # Model weights are optional when using the dummy fallback.  If the
    # file is missing we record a warning rather than an error.
    if not os.path.exists(config.MODEL_PATH):
        warnings.append(
            f"Model file not found: {config.MODEL_PATH}. Using dummy model." 
            f"  If you wish to use a real RWKV model, download it from "
            f"https://huggingface.co/BlinkDL/rwkv-7-world and place the .pth "
            f"file in {os.path.dirname(config.MODEL_PATH)}/"
        )

    suffix = Path(config.MODEL_PATH).suffix.lower()

    try:
        import rwkv  # noqa: F401
    except ImportError:
        if suffix == ".pth":
            warnings.append(
                "rwkv not installed. Using dummy model. To enable RWKV support, run: pip install rwkv"
            )

    if suffix == ".onnx":
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            warnings.append(
                "onnxruntime not installed. ONNX models will fall back to dummy mode. Run: pip install onnxruntime"
            )

    try:
        import torch
        if not torch.cuda.is_available():
            warnings.append(
                "CUDA not available — running on CPU.\n"
                "  Set MODEL_STRATEGY = 'cpu fp32' in config_local.py"
            )
    except ImportError:
        errors.append("PyTorch not installed.  Run: pip install torch")

    peft_path = PROJECT_ROOT / "RWKV-PEFT"
    if not peft_path.exists():
        warnings.append(
            "RWKV-PEFT not found — LoRA training unavailable.\n"
            f"  Run from {PROJECT_ROOT}:\n"
            "  git clone https://github.com/JL-er/RWKV-PEFT && "
            "pip install -e RWKV-PEFT/"
        )

    return errors, warnings


def main():
    print("RWKV Companion — starting...\n")

    errors, warnings = check_requirements()

    for w in warnings:
        print(f"⚠  {w}\n")

    if errors:
        print("Startup errors that must be resolved:\n")
        for e in errors:
            print(f"✗  {e}\n")
        sys.exit(1)

    from interface.terminal import run_chat
    run_chat()


if __name__ == "__main__":
    main()
