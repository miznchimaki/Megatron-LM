FROM nvcr.io/nvidia/pytorch:25.06-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
            software-properties-common \
            build-essential \
            python3-pip \
            python3-dev \
            python-is-python3 \
            bash \
            git \
            vim \
            tmux \
            tree \
            default-jre && \
    rm -rf /var/lib/apt/list/*

COPY bazel-dist /opt/bazel-dist/
COPY CLIP /opt/CLIP/
# ENV TENSORSTORE_BAZEL_STARTUP_OPTIONS="--distdir=/opt/bazel-dist"

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && \
    pip install --upgrade pip && \
    pip install --no-cache-dir \
        einops einops-exts sentencepiece braceexpand webdataset packaging \
        "transformers==4.40.2" datasets accelerate timm \
        pytest-cov pytest_mock nltk wrapt \
        zarr "tensorstore==0.1.50" \
        black isort "click==8.0.2" \
        pycocoevalcap megatron-energon mistral-common tiktoken && \
        pip install --no-cache-dir /opt/CLIP && \
        pip install --no-cache-dir open_clip_torch "open-flamingo[eval]" --no-deps
