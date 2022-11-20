FROM python:3.9-slim

WORKDIR /main

RUN apt-get update && apt-get install -y\
  vim \
  neovim \
  nano \
  curl\
  git\
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

# install python packages
RUN pip install poetry && \
  poetry config virtualenvs.create false
COPY poetry.lock pyproject.toml ./
RUN poetry install
RUN pip3 install tensorflow
