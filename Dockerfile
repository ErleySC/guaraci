# GUARACI -- imagem do app web em MODO SERVIDOR (padrao seguro, ver SECURITY.md).
# Multi-stage: o compilador e o git ficam so' no estagio de build.

# ---- estagio 1: constroi as wheels (precisa de git p/ o `spcfile` fixado por commit)
FROM python:3.11-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /w
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ---- estagio 2: imagem final, sem compilador, sem root
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 guaraci
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl \
    && rm -rf /wheels
WORKDIR /app
COPY --chown=guaraci:guaraci . .
USER guaraci

# MODO SERVIDOR: so' upload, pasta de sessao isolada, modelo so' em JSON portatil.
# NAO defina GUARACI_LOCAL_MODE=1 numa imagem exposta em rede: ele libera
# caminhos livres do servidor e .joblib.
ENV GUARACI_LOCAL_MODE=0 \
    PYTHONPATH=/app/src \
    GUARACI_SESSION_DIR=/tmp/guaraci_sessoes \
    PYTHONUNBUFFERED=1

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "app_quimiometria.py", \
            "--server.address=0.0.0.0", "--server.port=8501", \
            "--server.maxUploadSize=50", "--client.showErrorDetails=none"]
