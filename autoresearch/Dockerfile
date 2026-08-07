FROM ghcr.io/astral-sh/uv:0.11.29-python3.10-trixie-slim@sha256:2a82e6df28dd2d4590903b20af3028fd2e58051e5a8e525da9655189f63fd1fc

WORKDIR /workspace

ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY .python-version LICENSE.txt README.md pyproject.toml sigproc.py uv.lock ./
RUN uv sync --locked --python /usr/local/bin/python

COPY autoresearch ./autoresearch

USER 65532:65532
ENV PATH=/workspace/.venv/bin:$PATH

ENTRYPOINT ["python", "-m", "autoresearch.evaluate"]
