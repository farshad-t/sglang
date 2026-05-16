FROM lmsysorg/sgl-model-gateway:latest
SHELL ["/bin/bash", "-c"]

#ARG SGLANG_REPO=https://github.com/sgl-project/sglang.git
ARG SGLANG_REPO=https://github.com/attafosu/sglang.git
ARG VER_SGLANG=feat/attafosu/amx-cpu-awq-support

RUN apt-get update && \
    apt-get full-upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    git \
    curl \
    wget \
    vim \
    gcc \
    g++ \
    make \
    cmake \
    libsqlite3-dev \
    google-perftools \
    libtbb-dev \
    libnuma-dev \
    numactl

WORKDIR /opt

RUN echo -e '[[index]]\nname = "torch"\nurl = "https://download.pytorch.org/whl/cpu"\n\n[[index]]\nname = "torchvision"\nurl = "https://download.pytorch.org/whl/cpu"\n\n[[index]]\nname = "torchaudio"\nurl = "https://download.pytorch.org/whl/cpu"\n\n[[index]]\nname = "triton"\nurl = "https://download.pytorch.org/whl/cpu"' > /opt/venv/uv.toml

ENV UV_CONFIG_FILE=/opt/venv/uv.toml

WORKDIR /sgl-workspace
RUN source /opt/venv/bin/activate && \
    git clone ${SGLANG_REPO} sglang && \
    cd sglang && \
    git checkout ${VER_SGLANG} && \
    cd python && \
    cp pyproject_cpu.toml pyproject.toml && \
    /root/.local/bin/uv pip install . && \
    cd ../sgl-kernel && \
    cp pyproject_cpu.toml pyproject.toml && \
    make build && \
    /root/.local/bin/uv pip install dist/sglang_kernel_cpu-*.whl

RUN /opt/venv/bin/python -c "import sglang; import sglang_router"

RUN /opt/venv/bin/python - <<'PY'
import torch
if not torch._C._cpu._is_amx_tile_supported():
    raise RuntimeError("AMX tile support not detected by torch")
import sgl_kernel
if not hasattr(torch.ops.sgl_kernel, "convert_weight_packed"):
    raise RuntimeError("sgl_kernel common_ops missing: convert_weight_packed not found")
PY

ENV SGLANG_USE_CPU_ENGINE=1
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4:/usr/lib/x86_64-linux-gnu/libtbbmalloc.so:/opt/venv/lib/libiomp5.so
RUN echo 'source /opt/venv/bin/activate' >> /root/.bashrc

WORKDIR /sgl-workspace/sglang
