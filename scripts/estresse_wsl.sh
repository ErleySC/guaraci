#!/bin/bash
# Lado Linux (WSL) do estresse: clona o repositorio NO HASH pedido e roda o subconjunto com memoria virtual limitada (ulimit -v).
#   bash scripts/estresse_wsl.sh HASH CAMINHO_DO_REPO_VISTO_DO_WSL LIMITE_KB|unlimited SATURAR(0|1) arquivo_de_teste...
# SATURAR=1: um laco ocupado por nucleo (nproc) roda enquanto o pytest roda -- a CPU 100% do estresse no Linux.
set -u
H="${1:?hash}"; REPO="${2:?repo}"; LIM="${3:?limite}"; SAT="${4:?saturar}"; shift 4
W="$HOME/estresse_$H"
rm -rf "$W"
git clone -q "$REPO" "$W" && cd "$W" && git checkout -q "$H" || { echo "clone falhou"; exit 3; }
export CI=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src
ulimit -v "$LIM"
PIDS=""
if [ "$SAT" = "1" ]; then
  for _ in $(seq 1 "$(nproc)"); do bash -c 'while :; do :; done' & PIDS="$PIDS $!"; done
fi
"$HOME/venv_matriz_3.13/bin/python" -m pytest "$@" -q -p no:cacheprovider --no-header -o addopts="" 2>&1 | tail -n 3
if [ -n "$PIDS" ]; then kill $PIDS 2>/dev/null; fi
