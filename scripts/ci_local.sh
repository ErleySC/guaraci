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
# Este script replica os gates do workflow, na mesma ordem (barato primeiro):
#   1. ruff check .                      (job `lint`)
#   2. mypy no pacote inteiro + app      (job `typecheck`)
#   3. pytest + cobertura do pacote >= 75%   (job `test`, passo principal)
#   4. cobertura do núcleo científico >= 95%  (job `test`, gate do P4)
#   5. bandit + semgrep                  (job `seguranca`)
#   6. pip-audit no lock e nos pisos     (job `dependencias`)
# As etapas 5 e 6 usam ferramentas que NAO sao dependencias do projeto: se
# nao estiverem no PATH (ou em $FERRAMENTAS), a etapa e' marcada como NAO
# EXECUTADA -- nunca como aprovada (achado de 2026-10-02: o script dizia
# replicar o CI mas pulava os jobs `seguranca` e `dependencias`).
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
FERRAMENTAS="${FERRAMENTAS:-}"   # pasta bin/ de um venv so' de ferramentas (opcional)
FALHAS=0
NAO_EXECUTADAS=0

cd "$(dirname "$0")/.." || exit 1

hr() { printf '%s\n' "-----------------------------------------------------------"; }
etapa() { hr; printf '>> %s\n' "$1"; hr; }

etapa "1/6  ruff check .   (job 'lint' do CI)"
if "$PYTHON" -m ruff check .; then
    echo "OK: ruff limpo"
else
    echo "FALHOU: ruff"; FALHAS=$((FALHAS + 1))
fi

etapa "2/6  mypy no pacote inteiro   (job 'typecheck' do CI)"
# Mesmo comando do workflow (pacote inteiro: modulo novo entra sozinho).
if "$PYTHON" -m mypy src/guaraci/ app_quimiometria.py; then
    echo "OK: mypy limpo"
else
    echo "FALHOU: mypy"; FALHAS=$((FALHAS + 1))
fi

etapa "3/6  pytest + cobertura do pacote >= 75%   (job 'test' do CI)"
if "$PYTHON" -m pytest tests/ --cov=src/guaraci --cov-report=term-missing \
        --cov-report=xml --cov-fail-under=75 -q; then
    echo "OK: suite passou e cobertura do pacote >= 75%"
else
    echo "FALHOU: pytest/cobertura total"; FALHAS=$((FALHAS + 1))
fi

etapa "4/6  cobertura do nucleo cientifico >= 95%   (gate do P4)"
# Reusa os dados de cobertura do passo 3 (nao re-roda a suite), igual ao CI.
if "$PYTHON" -m coverage report \
        --include="*/guaraci/chemometric_stats.py,*/guaraci/classificadores.py,*/guaraci/preprocessamento.py,*/guaraci/validacao_estatistica.py" \
        --fail-under=95; then
    echo "OK: nucleo cientifico >= 95%"
else
    echo "FALHOU: cobertura do nucleo cientifico caiu abaixo de 95% (P4)"
    FALHAS=$((FALHAS + 1))
fi

ferramenta() {   # caminho da ferramenta: $FERRAMENTAS/<nome> ou PATH
    if [ -n "$FERRAMENTAS" ] && [ -x "$FERRAMENTAS/$1" ]; then echo "$FERRAMENTAS/$1";
    elif command -v "$1" >/dev/null 2>&1; then command -v "$1"; fi
}

etapa "5/6  bandit -r src -ll -q  +  semgrep pickle   (job 'seguranca' do CI)"
BANDIT=$(ferramenta bandit); SEMGREP=$(ferramenta semgrep)
if [ -z "$BANDIT" ] || [ -z "$SEMGREP" ]; then
    echo "NAO EXECUTADA: bandit/semgrep fora do PATH (instale num venv de ferramentas e use FERRAMENTAS=.../bin)"
    NAO_EXECUTADAS=$((NAO_EXECUTADAS + 1))
elif "$BANDIT" -r src -ll -q && "$SEMGREP" --config .semgrep/pickle.yml --error --metrics=off src app_quimiometria.py scripts; then
    echo "OK: bandit e semgrep limpos"
else
    echo "FALHOU: bandit/semgrep"; FALHAS=$((FALHAS + 1))
fi

etapa "6/6  pip-audit no lock e nos pisos   (job 'dependencias' do CI)"
if ! "$PYTHON" -m pip_audit --version >/dev/null 2>&1; then
    echo "NAO EXECUTADA: pip-audit nao instalado neste Python"
    NAO_EXECUTADAS=$((NAO_EXECUTADAS + 1))
else
    PISOS=$(mktemp)
    if "$PYTHON" scripts/pisos_de_versao.py > "$PISOS"             && PYTHONUTF8=1 "$PYTHON" -m pip_audit -r "$PISOS" --no-deps --disable-pip             && PYTHONUTF8=1 "$PYTHON" -m pip_audit -r requirements-lock.txt --no-deps --disable-pip; then
        echo "OK: sem vulnerabilidade conhecida no lock e nos pisos"
    else
        echo "FALHOU: pip-audit"; FALHAS=$((FALHAS + 1))
    fi
    rm -f "$PISOS"
fi

hr
if [ "$FALHAS" -eq 0 ] && [ "$NAO_EXECUTADAS" -eq 0 ]; then
    echo "TODOS OS 6 GATES PASSARAM (no ambiente local)."
    echo "Lembrete: a matriz de SO/versao do CI nao foi exercitada aqui."
    exit 0
fi
[ "$NAO_EXECUTADAS" -gt 0 ] && echo "$NAO_EXECUTADAS etapa(s) NAO EXECUTADA(S) -- nao conta como aprovada."
[ "$FALHAS" -gt 0 ] && echo "$FALHAS gate(s) FALHARAM -- ver a saida acima."
exit 1
