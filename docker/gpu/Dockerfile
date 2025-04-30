FROM  nvidia/cuda:12.8.1-devel-ubuntu24.04 AS base

ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive
ENV WORKDIR /app/

WORKDIR /opt

# install dev tools
RUN apt-get update && apt-get install -y \
    vim neovim nano \
    git git-lfs \
    zip unzip \
    curl wget make build-essential xz-utils file tree \
    sudo \
    dnsutils \
    tzdata language-pack-ja \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# for Japanese settings
# ENV TZ Asia/Tokyo
# ENV LANG ja_JP.utf8

# for US settings
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US

# install Python
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential libssl-dev libffi-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    # python${PYTHON_VERSION}-distutils \ # for python3.12
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && apt-get purge -y --auto-remove software-properties-common \
    && rm -rf /var/lib/apt/lists/*

## install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR ${WORKDIR}

# install python packages
COPY uv.lock pyproject.toml ./
COPY src ./src
RUN uv sync --frozen --no-cache
RUN uv run pip3 install deepspeed

FROM base AS dev
WORKDIR /opt

# install MeCab
RUN apt-get update && apt-get -yV upgrade && apt-get -yV install \
    mecab \
    libmecab-dev \
    mecab-utils \
    mecab-ipadic-utf8 \
    fontconfig \
    libjpeg-dev \
    zlib1g-dev \
    python3-webencodings \
    libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
## install Neologism dictionary for MeCab
RUN git clone https://github.com/neologd/mecab-ipadic-neologd.git \
    && mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -y -u \
    && rm -r mecab-ipadic-neologd
## compile IPA dic
RUN cp -r /usr/share/mecab/dic/ipadic/ `mecab-config --dicdir`/ \
    && cd `mecab-config --dicdir`/ipadic/ \
    && `mecab-config --libexecdir`/mecab-dict-index -f euc-jp -t utf8 \
    && cp /etc/mecabrc /usr/local/etc/mecabrc

WORKDIR ${WORKDIR}

# install python packages
COPY uv.lock pyproject.toml ./
COPY src ./src
RUN uv sync --frozen --no-cache
RUN uv run pip3 install deepspeed
