ARG BASE_IMAGE=cobiz:jetson
FROM ${BASE_IMAGE}

ARG INSTALL_FLASH_ATTN=0
ARG TORCH_WHEEL_URL=https://download-r2.pytorch.org/whl/cu126/torch-2.6.0%2Bcu126-cp310-cp310-linux_aarch64.whl
ARG TORCHVISION_WHEEL_URL=https://download-r2.pytorch.org/whl/cu126/torchvision-0.21.0-cp310-cp310-linux_aarch64.whl

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libopenblas-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/model_server.txt requirements/model_server.txt

RUN python3 -m pip install --upgrade pip "setuptools<80" wheel

RUN python3 -m pip install --no-cache-dir --force-reinstall \
    "${TORCH_WHEEL_URL}"

# Keep torchvision paired with the Jetson/aarch64 PyTorch wheel. --no-deps
# prevents pip from replacing the CUDA-enabled torch package.
RUN python3 -m pip install --no-cache-dir --no-deps \
    "${TORCHVISION_WHEEL_URL}"

RUN python3 -m pip install --no-cache-dir --no-deps \
    "diffusion_policy @ git+https://github.com/real-stanford/diffusion_policy.git@5ba07ac6661db573af695b419a7947ecb704690f"

# cobiz:jetson ships blinker as a distutils-managed Ubuntu package. Install a
# pip-managed copy first so the requirements step does not try to uninstall it.
RUN python3 -m pip install --no-cache-dir --ignore-installed "blinker>=1.9,<2.0"

RUN python3 -m pip install --no-cache-dir -r requirements/model_server.txt

# FlashAttention is optional. Jetson/aarch64 deployments generally use the
# Transformers SDPA backend instead; set INSTALL_FLASH_ATTN=1 only when a
# compatible wheel or toolchain is available.
RUN if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then \
        python3 -m pip install --no-cache-dir flash_attn==2.7.4.post1; \
    fi

COPY . .
RUN python3 -m pip install --no-deps --editable . \
    && python3 -c "import torch; assert torch.__version__.startswith('2.6.0'), torch.__version__; import torchvision; assert torchvision.__version__.startswith('0.21.0'), torchvision.__version__; import transformers; import diffusers; import internnav.agent.internvla_n1_agent_realworld; print('InternNav model-server imports OK:', torch.__version__, torchvision.__version__, torch.version.cuda)"

EXPOSE 5801

CMD ["python3", "scripts/realworld/http_internvla_server.py", "--device", "cuda:0", "--model_path", "/app/checkpoints/InternVLA-N1-DualVLN", "--attn-implementation", "sdpa", "--host", "0.0.0.0", "--port", "5801"]
