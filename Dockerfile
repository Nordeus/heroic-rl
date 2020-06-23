FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04 AS base

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu bionic main" \
    >>/etc/apt/sources.list
RUN echo "deb-src http://ppa.launchpad.net/deadsnakes/ppa/ubuntu bionic main" \
    >>/etc/apt/sources.list

RUN apt-key adv \
    --keyserver hkp://keyserver.ubuntu.com:80 \
    --recv-keys F23C5A6CF475977595C89F51BA6932366A755776

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.7 ssh

FROM base AS builder

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=off \
    POETRY_VERSION=1.0.3 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get install -y --no-install-recommends \
        python3-pip python3.7-dev python3.7-venv gcc libopenmpi-dev

RUN python3.7 -m pip install setuptools
RUN python3.7 -m pip install "poetry==$POETRY_VERSION"
RUN python3.7 -m venv /venv

COPY pyproject.toml poetry.lock ./
RUN poetry export -E gpu -f requirements.txt | /venv/bin/pip install -r /dev/stdin

COPY ./heroic_rl/ ./heroic_rl/
COPY ./images/ ./images/
COPY ./README.md ./decks.csv ./
RUN poetry build && /venv/bin/pip install dist/*.whl

FROM base AS final

ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    OMPI_MCA_btl_vader_single_copy_mechanism=none

ENV HEROIC_RL_TRAIN_EXP_NAME="experiment"
ENV HEROIC_RL_TRAIN_SERVERS="localhost:8081"
ENV HEROIC_RL_TRAIN_PLAN="utility"
ENV HEROIC_RL_TRAIN_EPOCHS="10000"
ENV HEROIC_RL_TRAIN_STEPS="48400"
ENV HEROIC_RL_TRAIN_CPUS="4"

RUN apt-get install -y --no-install-recommends \
            openmpi-common openmpi-bin && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv

RUN groupadd -r appuser && \
    useradd --no-log-init -r -M -g appuser appuser && \
    mkdir -p /app /app/data && \
    chown appuser:appuser /app && \
    chown appuser:appuser /app/data

WORKDIR /app

COPY docker-entrypoint.sh decks.csv ./

USER appuser

ENTRYPOINT ["./docker-entrypoint.sh"]
