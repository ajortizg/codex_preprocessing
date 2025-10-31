import logging
import os
import sys
import warnings

import absl.logging

# ─── General Python warning filters ──────────────────────────────────────────
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")

# Spatialdata: old Dask compatibility (#139)
warnings.filterwarnings("ignore", message="ignoring keyword argument 'read_only'")
warnings.filterwarnings("ignore", message=".*legacy Dask DataFrame implementation is deprecated.*")

# Cellpose: missing parameter set in model call (#141)
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

# Generic plugin / framework future warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="The plugin infrastructure in")

# TensorFlow / Protobuf / PyTorch-Wavelets
warnings.filterwarnings(
    "ignore",
    message="Protobuf gencode version.*is exactly one major version older.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

# ─── TensorFlow / Abseil / C++ backend noise ─────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF backend INFO/WARNING/ERROR logs
logging.getLogger("tensorflow").setLevel(logging.FATAL)
absl.logging.set_verbosity(absl.logging.ERROR)

# Optional: silence Python DeprecationWarnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning)
