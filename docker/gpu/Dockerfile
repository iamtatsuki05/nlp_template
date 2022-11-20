FROM python:3.9-slim

WORKDIR /main

RUN apt-get update && apt-get install -y\
    vim \
    neovim \
    nano \
    curl \
    git \
    git-lfs \
    zip \
    unzip \
    make \
    xz-utils \
    file \
    sudo \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    && pip install --upgrade pip

# install mecab-ipadic-neologd
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git && \
    cd mecab-ipadic-neologd && \
    ./bin/install-mecab-ipadic-neologd -n -y && \
    echo dicdir = `mecab-config --dicdir`"/mecab-ipadic-neologd">/etc/mecabrc && \
    sudo cp /etc/mecabrc /usr/local/etc && \
    cd ..

ENV NVARCH x86_64
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.3 brand=tesla,driver>=418,driver<419 driver>=450"
ENV NV_CUDA_CUDART_VERSION 11.3.58-1
ENV NV_CUDA_COMPAT_PACKAGE cuda-compat-11-3

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH}/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*


# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
ENV CUDA_VERSION 11.3.0
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-3=${NV_CUDA_CUDART_VERSION} \
    ${NV_CUDA_COMPAT_PACKAGE} \
    && ln -s cuda-11.3 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# cuda install end

# install python packages
RUN pip install --upgrade pip

RUN pip install poetry && \
    poetry config virtualenvs.create false
COPY poetry.lock pyproject.toml ./
RUN poetry install

RUN pip3 install tensorflow
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113
