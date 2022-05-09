FROM python:3.8-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=./

WORKDIR /app

RUN apt-get update && apt-get install -y\
  curl\
  git\
  zip\
  unzip\
  && pip install --upgrade pip

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
ENV PATH $PATH:/root/.poetry/bin
RUN poetry config virtualenvs.create false
COPY poetry.lock pyproject.toml ./
RUN poetry install
