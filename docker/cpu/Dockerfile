FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y\
  vim \
  neovim \
  nano \
  curl\
  git\
  git-lfs \
  zip\
  unzip\
  && pip install --upgrade pip

RUN pip install poetry && \
  poetry config virtualenvs.create false
COPY poetry.lock pyproject.toml ./
RUN poetry install
RUN pip3 install tensorflow
