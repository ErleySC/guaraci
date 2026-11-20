# -*- coding: utf-8 -*-
"""Console da execucao no idioma da sessao (limitacoes.md, secao 8: "idioma misturado").

Medido antes da correcao: num `guaraci demo` com GUARACI_IDIOMA=en, 71 de 192 linhas do console tinham
palavras so' do portugues; depois, nenhuma (a unica acusada e' "PLS-DA", falso positivo de "da"). O teste
abaixo usa as linhas reais daquele console; as etiquetas que o painel le por regex nao mudam."""
from __future__ import annotations

import io
import sys

import pytest

from guaraci import app_logic
from guaraci.idioma_console import FRASES_CONSOLE, saida_traduzida, traduzir_linha, traduzir_texto
from guaraci.idioma_relatorio import vazamentos

LINHAS_DO_DEMO = [
    "[MODO] Objetivo cientifico: Classificacao  (nivel N2, objetivo configurado: auto)",
    "[MODO] Figuras pertinentes a este objetivo (5): Scores PLS-DA; Matriz de confusao + metricas por classe; "
    "Curvas ROC/AUC multiclasse; DD-SIMCA (autenticacao de pureza); Avaliacao em holdout externo",
    "[INFO] Perfil de tecnica: generico (Matriz nao especificada -- nenhum padrao de matriz aplicado) | eixo "
    "[400, 4001] (unidade nao declarada) | pre-proc msc_sg_mc",
    "[INFO] Synthetic MODE — cenario 'nir': 216 espectros, 72 amostras fisicas, 3 lotes; variancia dentro das "
    "amostras (replicas) = 7.6% do total.",
    "[INFO] Truncamento espectral [400, 4001]: 0/300 variaveis removidas, restam 300 (400.0-4001.0).",
    "[INFO] Auditoria de delineamento: 2 critico(s), 3 aviso(s) -- ver model card para o detalhe por checagem.",
    "  [AUDITORIA][AVISO] validacao_externa: Os padroes do motor sao ponto de partida, nao universais, e nao "
    "validam o seu estudo: confirme as conclusoes com validacao externa propria (amostras independentes).",
    "[INFO] Hold-out (StratifiedGroup (replicas juntas, por classe) + 27 puros preservados): 36 amostras "
    "reservadas (frac=0.20). Pipeline rodara em 180 amostras.",
    "[INFO] Amostras : 180",
    "  [INFO] CV aninhada ativada (selecao_lv_cv_aninhada=True, default a partir da v1.0 -- corrige o vies de "
    "reusar o mesmo dado p/ escolher n_opt e avaliar a metrica reportada). Pode levar ~4-5x mais tempo nesta "
    "etapa; desative no config.yaml (selecao_lv_cv_aninhada: false) p/ iteracao rapida, aceitando a metrica otimista.",
    "[6/7] Gerando figuras...",
    "[DDSimca] Classe 'Esp_A': 2 amostra(s) de treino atipica(s) (indices [3, 5] de 9, deteccao robusta "
    "mediana/MAD). Nao removidas automaticamente -- considere investigar essas replicas.",
    "    [AVISO] Esp_A: Sensibilidade estimada por LOGO com apenas 3 grupos de replica. Incerteza alta; IC "
    "bootstrap nao e confiavel neste regime. Interpretar como exploratoria.",
    "  [AVISO] alpha=0.05 NAO e' alcancavel com 12 amostra(s) de calibracao independentes: exigiria k=13 escores "
    "ordenados, mas so' ha' 12. Menor alpha garantivel: 0.0769 (precisa de n>=19 para alpha=0.05). Limiar NAO "
    "definido -- reportar como nao-estimavel, nunca como 'aceita tudo'.",
    "[7/7] PLS regressao — PULADA: objective=Classificacao (level=N2). A regressao da resposta pertence ao Modo "
    "Quantificacao (N3); nos modos Exploratorio/Classificacao ela seria um resultado fora de escopo (e em N1 as 3 "
    "classes dominam o sinal da resposta, produzindo R2~0).",
]


@pytest.mark.parametrize("linha", LINHAS_DO_DEMO)
def test_linha_do_demo_sai_em_ingles(linha):
    en = traduzir_linha(linha, "en")
    assert en != linha
    assert vazamentos(en.replace("PLS-DA", ""), "en") == [], en
    assert en[: len(linha) - len(linha.lstrip())] == linha[: len(linha) - len(linha.lstrip())]


def test_em_portugues_nada_muda():
    for linha in LINHAS_DO_DEMO:
        assert traduzir_linha(linha, "pt") == linha


def test_etiquetas_do_painel_continuam_legiveis():
    bloco = "\n".join(traduzir_linha(l, "en") for l in LINHAS_DO_DEMO)
    avisos = app_logic.log_warnings(bloco)       # o painel acha os [AVISO] traduzidos
    assert len(avisos) == 2 and all("Sensitivity" in a or "alpha" in a for a in avisos)
    fold = "[2b/7] CV aninhada: fold externo 3/5 concluido"
    assert traduzir_linha(fold, "en") == fold
    assert "[AUDITORIA][AVISO]" in traduzir_linha(LINHAS_DO_DEMO[6], "en")


def test_frase_sem_par_sai_como_veio():
    assert traduzir_linha("[INFO] frase nova sem traducao", "en") == "[INFO] frase nova sem traducao"


def test_todos_os_padroes_compilam_e_tem_modelo():
    import re
    for p, en in FRASES_CONSOLE:
        re.compile(p)
        assert en.strip()


def test_saida_traduzida_troca_stdout_so_em_ingles(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    with saida_traduzida("pt"):
        print("[6/7] Gerando figuras...")
    with saida_traduzida("en"):
        print("[6/7] Gerando", end="")            # linha em pedacos: traduz quando completa
        print(" figuras...")
        sys.stdout.write("[INFO] Amostras : 9")    # sem quebra: sai no flush do fim do bloco
    assert sys.stdout is buf
    assert buf.getvalue() == "[6/7] Gerando figuras...\n[6/7] Generating figures...\n[INFO] Samples  : 9"


def test_ajuda_segue_o_idioma(monkeypatch):
    from guaraci import guaraci as g
    monkeypatch.setenv("GUARACI_IDIOMA", "en")
    en = g._texto_ajuda_sessao()
    import re
    sem_identificadores = re.sub(r"--\S+|\b[A-Z][A-Z_]+\b\S*", " ", en)    # flags e PLACEHOLDERS ficam em PT
    assert en.startswith("Usage:") and "Exit codes" in en and vazamentos(sem_identificadores, "en") == []
    monkeypatch.setenv("GUARACI_IDIOMA", "pt")
    assert g._texto_ajuda_sessao().startswith("Uso:")


def test_traduzir_texto_em_bloco():
    assert traduzir_texto("[6/7] Gerando figuras...\n[INFO] Amostras : 3", "en") == \
        "[6/7] Generating figures...\n[INFO] Samples  : 3"
