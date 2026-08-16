#!/usr/bin/env bash
# =============================================================================
# ci_local.sh — roda LOCALMENTE os mesmos gates do CI (.github/workflows/test.yml)
#
# Por que existe: a cota de minutos do GitHub Actions da conta esgotou
# (2026-08-07), e nenhum job roda enquanto isso — nem `lint`, nem `typecheck`,
# nem `test`. Sem um substituto local, o projeto ficaria sem NENHUMA
# verificação automatizada até a cota voltar, que é justamente quando erros
# passam despercebidos.
#
# Este script replica os 4 gates do workflow, na mesma ordem (barato primeiro):
#   1. ruff check .                      (job `lint`)
#   2. mypy nos 7 módulos puros          (job `typecheck`)
#   3. pytest + cobertura total >= 60%   (job `test`, passo principal)
#   4. cobertura do núcleo científico >= 95%  (job `test`, gate do P4)
#
# NÃO substitui o CI por completo: o CI roda a matriz de SO/versão de Python
# (Ubuntu/Windows/macOS x 3.10-3.13); aqui roda só no seu ambiente. Um bug
# específico de plataforma/versão continua só aparecendo no CI. Use isto para
# não voar às cegas, não como prova de que a matriz passa.
#
# Uso:
#   bash scripts/ci_local.sh
#   PYTHON=~/.venvs/guaraci/Scripts/python.exe bash scripts/ci_local.sh
# =============================================================================
set -uo pipefail

PYTHON="${PYTHON:-python}"
FALHAS=0

cd "$(dirname "$0")/.." || exit 1

hr() { printf '%s\n' "-----------------------------------------------------------"; }
etapa() { hr; printf '>> %s\n' "$1"; hr; }

etapa "1/4  ruff check .   (job 'lint' do CI)"
if "$PYTHON" -m ruff check .; then
    echo "OK: ruff limpo"
else
    echo "FALHOU: ruff"; FALHAS=$((FALHAS + 1))
fi

etapa "2/4  mypy nos modulos puros   (job 'typecheck' do CI)"
# Mesma lista do workflow. Ao criar um modulo puro novo, adicionar nos DOIS.
if "$PYTHON" -m mypy \
        src/guaraci/preprocessamento.py \
        src/guaraci/chemometric_stats.py \
        src/guaraci/classificadores.py \
        src/guaraci/validacao_estatistica.py \
        src/guaraci/modos_analise.py \
        src/guaraci/design_tokens.py \
        src/guaraci/resumo_parse.py; then
    echo "OK: mypy limpo"
else
    echo "FALHOU: mypy"; FALHAS=$((FALHAS + 1))
fi

etapa "3/4  pytest + cobertura total >= 60%   (job 'test' do CI)"
if "$PYTHON" -m pytest tests/ --cov=. --cov-report=term-missing \
        --cov-report=xml --cov-fail-under=60 -q; then
    echo "OK: suite passou e cobertura total >= 60%"
else
    echo "FALHOU: pytest/cobertura total"; FALHAS=$((FALHAS + 1))
fi

etapa "4/4  cobertura do nucleo cientifico >= 95%   (gate do P4)"
# Reusa os dados de cobertura do passo 3 (nao re-roda a suite), igual ao CI.
if "$PYTHON" -m coverage report \
        --include="*/guaraci/chemometric_stats.py,*/guaraci/classificadores.py,*/guaraci/preprocessamento.py,*/guaraci/validacao_estatistica.py" \
        --fail-under=95; then
    echo "OK: nucleo cientifico >= 95%"
else
    echo "FALHOU: cobertura do nucleo cientifico caiu abaixo de 95% (P4)"
    FALHAS=$((FALHAS + 1))
fi

hr
if [ "$FALHAS" -eq 0 ]; then
    echo "TODOS OS 4 GATES PASSARAM (no ambiente local)."
    echo "Lembrete: a matriz de SO/versao do CI nao foi exercitada aqui."
    exit 0
fi
echo "$FALHAS gate(s) FALHARAM — ver a saida acima."
exit 1
