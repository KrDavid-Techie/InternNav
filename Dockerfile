ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.02-py3
FROM ${BASE_IMAGE}

ARG INSTALL_FLASH_ATTN=0

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
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/model_server.txt requirements/model_server.txt

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir --no-deps \
        "diffusion_policy @ git+https://github.com/real-stanford/diffusion_policy.git@5ba07ac6661db573af695b419a7947ecb704690f" \
    && python3 -m pip install --no-cache-dir -r requirements/model_server.txt

# FlashAttention is optional. Jetson/aarch64 deployments generally use the
# Transformers SDPA backend instead; set INSTALL_FLASH_ATTN=1 only when a
# compatible wheel or toolchain is available.
RUN if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then \
        python3 -m pip install --no-cache-dir flash_attn==2.7.4.post1; \
    fi

COPY . .
RUN python3 -m pip install --no-deps --editable . \
    && python3 -c "import torch; import transformers; import diffusers; import internnav.agent.internvla_n1_agent_realworld; print('InternNav model-server imports OK')"

EXPOSE 5801

CMD ["python3", "scripts/realworld/http_internvla_server.py", "--device", "cuda:0", "--model_path", "/app/checkpoints/InternVLA-N1", "--attn-implementation", "sdpa", "--host", "0.0.0.0", "--port", "5801"]
