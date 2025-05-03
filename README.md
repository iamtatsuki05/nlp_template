# docker+uv

## How to operate uv
### setup
1. Install with`git clone`
### uv configuration
1. `uv sync`
### run script
```shell
uv run python ...
```

## How to operate docker
### setup
1. Install with`git clone`
### docker configuration
1. `docker compose up -d --build <service name(ex:python-cpu)`
### Connect to and disconnect from docker
1. connect`docker compose exec <service name(ex:python-cpu)> bash`
2. disconect`exit`
### Using jupyterlab
1. Access with a browser http://localhost:8888/lab
### Starting and Stopping Containers
1. Starting`docker compose start`
2. Stopping`docker compose stop`

## Directory structure
```text
./
├── .dockerignore
├── .git
├── .gitattributes
├── .github
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile
├── README.md
├── compose.yaml
├── config
├── data
│   ├── datasets
│   ├── misc
│   ├── models
│   ├── outputs
│   └── raw
├── docker
│   ├── cpu
│   └── gpu
├── docs
├── env.sample
├── notebooks
├── uv.lock
├── pyproject.toml
├── scripts
│   └── main.py
├── src
│   ├── __init__.py
│   └── project
│       ├── common
│       ├── config
│       ├── env.py
│       └── main.py
└── tests
    └── project
```
