# Sewtrace test page. Build:   docker build -t sewtrace .
# With real SAM decoding:      docker build --build-arg INSTALL_REAL=1 -t sewtrace .
# Mount a persistent volume at /app/data: it holds the spend ledger (docs/DEPLOY.md).
FROM python:3.12-slim

ARG INSTALL_REAL=0
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 10001 sewtrace
WORKDIR /app

# Editable install: the code finds data/ and cache/ next to the package, so /app/data is the
# ledger folder (the ledger path is fixed by config.toml, not by an environment variable).
COPY pyproject.toml README.md ./
COPY sewtrace ./sewtrace
RUN pip install -e . \
    && if [ "$INSTALL_REAL" = "1" ]; then pip install -e '.[real]'; fi \
    && mkdir -p data cache \
    && chown -R sewtrace:sewtrace /app
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','8000'), timeout=4)"
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["sh", "-c", "exec sewtrace serve --host 0.0.0.0 --port ${PORT:-8000}"]
