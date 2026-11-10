#!/bin/bash
# Lado Linux (WSL) da verificacao final: clona o repositorio NO HASH pedido e roda, la, o que o Windows nao roda bem:
#   - a suite completa (com slow) em Python 3.10, 3.11 e 3.13, UMA VERSAO POR VEZ (xdist, metade dos nucleos no maximo), cada uma num venv;
#   - bandit, semgrep e pip-audit (pisos e lock), com as ferramentas do venv_ferramentas;
#   - a instalacao limpa (scripts/instalacao_limpa.sh, que tambem percorre a ajuda do wheel instalado) em 3.10 e 3.13.
#
#   wsl -d "${GUARACI_WSL_DISTRO:-Ubuntu}" -- bash scripts/verificar_wsl.sh HASH CAMINHO_DO_REPO_VISTO_DO_WSL [PASTA_DE_TRABALHO] [ETAPAS]
#
# Sempre com a distribuicao EXPLICITA (`-d`, como scripts/verificar_hash_final.py e scripts/estresse_testes.py): a distro PADRAO
# do WSL pode nem ter bash (ex.: docker-desktop), e entao o WSL sem `-d` ou o `bash.exe` do sistema falham com
# "execvpe(/bin/bash) failed".
#
# ETAPAS: lista separada por virgula (ex.: bandit,pytest_3.13); vazia = todas. O verificador so' pede as etapas cujas entradas mudaram.
# Imprime uma linha `RESUMO chave=valor ...` por item (rc 0 = ok) e sai com 0 se tudo deu 0. Nada e' enviado a lugar nenhum
# (so' o pip-audit e a instalacao usam a rede, para ler avisos e baixar pacotes).
set -u
H="${1:?hash}"; REPO="${2:?repo}"; W="${3:-}"; ETAPAS="${4:-}"
# uma pasta por execucao, nunca apagada: junit e logs de uma rodada reprovada sao a unica prova de qual teste falhou
[ -n "$W" ] || W="$HOME/vhf_${H}_$(date +%Y%m%d_%H%M%S)"
[ -e "$W" ] && { echo "RESUMO pasta_existente rc=5"; exit 5; }
mkdir -p "$W"
git clone -q "$REPO" "$W/repo" && cd "$W/repo" && git checkout -q "$H" || { echo "RESUMO clone rc=3"; exit 3; }
[ "$(git rev-parse HEAD | cut -c1-${#H})" = "$H" ] || { echo "RESUMO hash_diferente rc=4"; exit 4; }
export CI=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=src
# processos do xdist no WSL: GUARACI_TESTES_WORKERS_WSL so' REDUZ o padrao (4, nunca mais que metade dos nucleos)
N="${GUARACI_TESTES_WORKERS_WSL:-4}"; case "$N" in ''|*[!0-9]*) N=4;; esac
M=$(( $(nproc) / 2 )); [ "$M" -gt 4 ] && M=4; [ "$N" -gt "$M" ] && N=$M; [ "$N" -lt 1 ] && N=1

quer() { [ -z "$ETAPAS" ] || case ",$ETAPAS," in *",$1,"*) return 0;; *) return 1;; esac; }

F="$HOME/venv_ferramentas/bin"
cd "$W/repo"
if quer bandit; then "$F/bandit" -r src -ll -q > "$W/bandit.txt" 2>&1; echo "RESUMO bandit rc=$?"; fi
if quer semgrep; then
  "$F/semgrep" --config .semgrep/pickle.yml --error --metrics=off src app_quimiometria.py scripts > "$W/semgrep.txt" 2>&1; echo "RESUMO semgrep rc=$?"
fi
if quer pip_audit_pisos; then
  "$HOME/venv_matriz_3.13/bin/python" scripts/pisos_de_versao.py > "$W/pisos.txt" 2>/dev/null
  "$F/pip-audit" -r "$W/pisos.txt" --no-deps --disable-pip > "$W/audit_pisos.txt" 2>&1; echo "RESUMO pip_audit_pisos rc=$?"
fi
if quer pip_audit_lock; then
  "$F/pip-audit" -r requirements-lock.txt --no-deps --disable-pip > "$W/audit_lock.txt" 2>&1; echo "RESUMO pip_audit_lock rc=$?"
fi

for v in 3.10 3.13; do
  if quer "instalacao_limpa_$v"; then
    PYTHON="$HOME/venv_matriz_$v/bin/python" bash scripts/instalacao_limpa.sh "$W/inst_$v" > "$W/instalacao_$v.txt" 2>&1
    echo "RESUMO instalacao_limpa_$v rc=$?"
  fi
done

# um pytest por vez: o xdist ja usa os nucleos, e dois processos pesados juntos saturam a maquina
for v in 3.10 3.11 3.13; do
  if quer "pytest_$v"; then
    # `build` e' ferramenta de desenvolvimento (o teste do pacote construido de verdade a usa), nao dependencia do programa
    "$HOME/venv_matriz_$v/bin/python" -m pip install -q build > /dev/null 2>&1 || true
    "$HOME/venv_matriz_$v/bin/python" -m pytest tests/ -q -p no:cacheprovider --no-header -o addopts="" -n "$N" -rfE \
        --junitxml="$W/pytest_$v.xml" > "$W/pytest_$v.txt" 2>&1
    rc=$?
    # o nome de cada teste que falhou vai para o resultado (o verificador le as linhas FALHOU)
    grep -E '^(FAILED|ERROR) ' "$W/pytest_$v.txt" | cut -d' ' -f2 | head -n 50 | while read -r t; do echo "FALHOU pytest_$v $t"; done
    echo "RESUMO pytest_$v rc=$rc resumo=\"$(tail -n 1 "$W/pytest_$v.txt" | tr -d '\r')\""
  fi
done
echo "RESUMO fim rc=0 pasta=$W"
