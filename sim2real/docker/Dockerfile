# syntax=docker/dockerfile:1.7
FROM maniskill/base@sha256:ebe375d0242b5e980ac7be6c48c71782c597e88572bbf93c88c53b5c42aa656f

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
 && apt-get install -y --no-install-recommends \
    dbus-x11 \
    ffmpeg \
    fontconfig \
    gdk-pixbuf2.0-bin \
    mesa-utils \
    x11-apps \
 && rm -rf /var/lib/apt/lists/*

# Small config/runtime helpers first: fast to install, stable to cache.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install \
    ConfigArgParse==1.7.5 \
    addict==2.4.0 \
    antlr4-python3-runtime==4.9.3 \
    attrs==25.4.0 \
    click==8.1.8 \
    fastjsonschema==2.21.2 \
    iniconfig==2.1.0 \
    jsonschema-specifications==2025.9.1 \
    jsonschema==4.25.1 \
    loguru==0.7.3 \
    omegaconf==2.3.0 \
    pluggy==1.6.0 \
    pyquaternion==0.9.9 \
    pytest==8.4.2 \
    referencing==0.36.2 \
    retrying==1.4.2 \
    rpds-py==0.27.1 \
    tomli==2.4.1

# Data/scientific layer stays separate so heavy runtime packages don't invalidate it.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install \
    joblib==1.5.3 \
    pandas==2.3.3 \
    pytz==2026.1.post1 \
    scikit-learn==1.6.1 \
    threadpoolctl==3.6.0 \
    tzdata==2025.3

# Heavy runtime and visualization packages last: most likely to fail or change.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked     python -m pip install     Flask==3.1.3     Werkzeug==3.1.6     blinker==1.9.0     comm==0.2.3     dash==4.0.0     ipywidgets==8.1.8     itsdangerous==2.2.0     jupyter_core==5.8.1     jupyterlab_widgets==3.0.16     narwhals==2.18.0     nbformat==5.10.4     nest-asyncio==1.6.0     open3d==0.19.0     plotly==6.6.0     rtree==1.4.1     widgetsnbextension==4.0.15

# Upgrade ManiSkill to the validated runtime version from the working container.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked     python -m pip install mani_skill==3.0.0b21

# These two wheels worked in the validated container with numpy 1.26.x.
# Install them without re-resolving dependencies, otherwise current upstream
# metadata tries to force numpy>=2 and breaks ManiSkill.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked     python -m pip install --no-deps     coacd==1.0.7     fast_simplification==0.1.9

# Upgrade PyTorch to 2.7.1+cu128 for RTX 5090 (Blackwell / sm_120) support.
# The base image's stock torch (and the cu126 wheels) do NOT ship sm_120 kernels
# (their arch_list stops at sm_90) and fail with "CUDA error: no kernel image is
# available for execution on the device". Only the cu128 build has sm_100/sm_120/
# compute_120. Verify with:
#   python -c "import torch; print(torch.cuda.get_arch_list())"
# --no-deps preserves ManiSkill's numpy/scipy pinning.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --no-deps \
    torch==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Because torch is installed with --no-deps, its CUDA runtime libraries are not
# pulled in automatically and the base image only carries an older set. Torch
# 2.7.1+cu128 needs the exact versions below (from its Requires-Dist metadata);
# older ones break the import, e.g.:
#   ImportError: libcufile.so.0: cannot open shared object file
#   ImportError: libtorch_cpu.so: undefined symbol: cuptiActivityEnableDriverApi
# --no-deps again avoids re-resolving the full dependency tree.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --no-deps \
    nvidia-cublas-cu12==12.8.3.14 \
    nvidia-cuda-cupti-cu12==12.8.57 \
    nvidia-cuda-nvrtc-cu12==12.8.61 \
    nvidia-cuda-runtime-cu12==12.8.57 \
    nvidia-cudnn-cu12==9.7.1.26 \
    nvidia-cufft-cu12==11.3.3.41 \
    nvidia-cufile-cu12==1.13.0.11 \
    nvidia-curand-cu12==10.3.9.55 \
    nvidia-cusolver-cu12==11.7.2.55 \
    nvidia-cusparse-cu12==12.5.7.53 \
    nvidia-cusparselt-cu12==0.6.3 \
    nvidia-nccl-cu12==2.26.2 \
    nvidia-nvjitlink-cu12==12.8.61 \
    nvidia-nvtx-cu12==12.8.55
