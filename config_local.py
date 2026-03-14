"""
Local configuration overrides for Flint.  This file is imported at the
end of config.py if present.  You can override settings here without
modifying the default config.  Only attributes you set here will
override the defaults.
"""

# Use a dummy model path during development/test.  When a real model
# path is configured here, Flint will load it on startup.  The dummy
# path triggers the fallback behaviour in core.model.CompanionModel,
# allowing the backend to run without a large RWKV file.
MODEL_PATH = "/home/daogeshi/flint/models/rwkv7-g1e-2.9b-20260312-ctx8192.pth"

# Use CPU for inference by default in test environment
MODEL_STRATEGY = "cuda fp16"
