#!/bin/bash
# Instalacao num ambiente virgem, exatamente como docs/usuario/roteiro_teste_usabilidade.md manda, e as 5 tarefas por comando.
#
#   bash scripts/instalacao_limpa.sh [PASTA_DE_TRABALHO]     (PYTHON=python3.12 para escolher o interprete)
#
# Passos: `git archive HEAD` (so' o que o git versiona) -> venv novo -> `pip install ".[reports]"` -> HOME vazio ->
# tarefas 1 a 5 do roteiro. Escreve `resultado.txt` na pasta de trabalho com codigo de saida e tempo de cada passo;
# a ultima linha e' `RESULTADO OK` ou `RESULTADO FALHOU`. Usa a rede so' para o pip. Nada e' enviado.
set -u
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
W="${1:-$(mktemp -d)}"
PY="${PYTHON:-python}"
mkdir -p "$W"
rm -rf "$W/pacote" "$W/venv" "$W/home" "$W/trabalho"
mkdir -p "$W/pacote" "$W/home" "$W/trabalho"
LOG="$W/resultado.txt"
: > "$LOG"
( cd "$RAIZ" && git archive HEAD | tar -x -C "$W/pacote" )
t0=$(date +%s)
"$PY" -m venv "$W/venv" >> "$LOG" 2>&1
if [ -d "$W/venv/Scripts" ]; then BIN="$W/venv/Scripts"; EXE=".exe"; else BIN="$W/venv/bin"; EXE=""; fi
( cd "$W/pacote" && "$BIN/python$EXE" -m pip install --quiet ".[reports]" >> "$LOG" 2>&1 )
echo "pip install rc=$? em $(( $(date +%s) - t0 )) s" >> "$LOG"
export HOME="$W/home" USERPROFILE="$W/home" PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
cd "$W/trabalho" || exit 1
G="$BIN/guaraci$EXE"
FALHAS=0
passo() { local nome="$1"; shift; local t=$(date +%s); "$@" > "$W/trabalho/saida_$nome.txt" 2>&1; local rc=$?
          echo "$nome rc=$rc $(( $(date +%s) - t )) s" >> "$LOG"; [ $rc -ne 0 ] && FALHAS=$((FALHAS + 1)); return 0; }
passo versao "$G" --version
# no Windows a 1a execucao do `doctor` pode ser barrada pelo Controle de Aplicativo; repetir uma vez, como o roteiro manda
"$G" doctor > "$W/trabalho/saida_doctor.txt" 2>&1 || true
passo doctor "$G" doctor
passo t2_exemplo "$G" exemplo classificar ex_cls
passo t2_run "$G" run ex_cls/config.yaml
passo t3_exemplo "$G" exemplo quantificar ex_qtf
passo t3_run "$G" run ex_qtf/config.yaml
passo t4_exemplo "$G" exemplo comparar ex_cmp
passo t4_anova "$G" anova ex_cmp/tabela.csv --resposta teor --grupo grupo --unidade amostra --saida ex_cmp/res --sem-bootstrap
passo t5_assistente "$G" assistente "por que meu Q2 e baixo"
passo t5_confianca "$G" confianca pls_da
passo t5_ajuda "$G" ajuda max_lvs
passo repeticao_exemplo "$G" exemplo explorar ex_exp
passo repeticao_run "$G" run ex_exp/config.yaml
grep -q "Isto significa" "$W/trabalho/saida_t5_assistente.txt" || FALHAS=$((FALHAS + 1))
# toda a ajuda do programa INSTALADO: nenhum topico aponta para fora do pacote ("nao encontrado", docs/, MANUAL.md ...)
"$BIN/python$EXE" -I "$RAIZ/scripts/percorrer_ajuda_instalada.py" > "$W/trabalho/saida_ajuda_instalada.txt" 2>&1
echo "ajuda_instalada rc=$? $(tail -n 1 "$W/trabalho/saida_ajuda_instalada.txt")" >> "$LOG"
tail -n 1 "$W/trabalho/saida_ajuda_instalada.txt" | grep -q "rc=0" || FALHAS=$((FALHAS + 1))
[ -n "$(find ex_cls -name resumo_modelo.txt)" ] || FALHAS=$((FALHAS + 1))
if [ "$FALHAS" -eq 0 ]; then echo "RESULTADO OK" >> "$LOG"; else echo "RESULTADO FALHOU ($FALHAS)" >> "$LOG"; fi
cat "$LOG" | grep -v -i "warning\|retrying"
[ "$FALHAS" -eq 0 ]
