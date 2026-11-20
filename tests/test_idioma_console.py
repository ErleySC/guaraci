# -*- coding: utf-8 -*-
"""Console da execucao no idioma da sessao (limitacoes.md, secao 8: "idioma misturado").

Medido antes da correcao: num `guaraci demo` com GUARACI_IDIOMA=en, 71 de 192 linhas do console tinham
palavras so' do portugues; depois, nenhuma (a unica acusada e' "PLS-DA", falso positivo de "da"). O teste
abaixo usa as linhas reais daquele console; as etiquetas que o painel le por regex nao mudam."""
from __future__ import annotations

import io
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from guaraci import app_logic
from guaraci.idioma_console import FRASES_CONSOLE, saida_traduzida, traduzir_linha, traduzir_texto
from guaraci.idioma_relatorio import vazamentos
from guaraci import idioma_console as ic

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
    bloco = "\n".join(traduzir_linha(l, "en", painel=True) for l in LINHAS_DO_DEMO)
    avisos = app_logic.log_warnings(bloco)       # o painel acha os [AVISO] traduzidos
    assert len(avisos) == 2 and all("Sensitivity" in a or "alpha" in a for a in avisos)
    fold = "[2b/7] CV aninhada: fold externo 3/5 concluido"
    assert traduzir_linha(fold, "en", painel=True) == fold
    assert "[AUDITORIA][AVISO]" in traduzir_linha(LINHAS_DO_DEMO[6], "en", painel=True)
    # fora do painel (comando de linha) as etiquetas tambem saem em ingles
    assert traduzir_linha(LINHAS_DO_DEMO[6], "en").lstrip().startswith("[AUDIT][WARNING] ")


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


# ---- comandos de linha em ingles de ponta a ponta (rodada da CLI de 2026-11-20) ------------------------------

RAIZ = Path(__file__).resolve().parents[1]
#: nomes de coluna e de pasta que os exemplos geram (dados do usuario, nao texto do programa)
_NOMES_DOS_EXEMPLOS = ("'amostra'", "--unidade amostra", "--id amostra", "--grupo amostra", "--resposta teor",
                       "teor:")


#: palavras de ligacao que so' existem em portugues, alem das de idioma_relatorio.PALAVRAS_SO_PT (que pegam
#: termos tecnicos): sem elas uma frase inteira em portugues sem termo tecnico passaria.
_MAIS_PT = {"de", "uma", "um", "ao", "aos", "das", "dos", "nas", "nos", "que", "ou", "mas", "porque", "esta",
            "isso", "tecnica", "disponiveis", "arquivo", "arquivos", "rodar", "gerando", "concluido", "concluida",
            "pulado", "pulada", "ainda", "entre", "sao", "foi", "ha", "apenas", "tambem", "quando",
            "usando", "salvo", "salvos", "gravado", "gravados", "carregados", "carregadas", "erro"}


def _vazamentos_da_saida(texto: str):
    """Linhas com palavra so' do portugues, ignorando caminhos, nomes de arquivo e o valor de config ecoado."""
    achados = []
    for linha in texto.splitlines():
        limpa = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", linha)
        # nao e' texto: valor de config ecoado, id da checagem da auditoria, nome de arquivo
        limpa = re.sub(r"configured objective: \w+|\[AUDIT\]\[\w+\] \w+:|PLS-DA|\b[\w-]+\.(csv|yaml|json|txt|md|dx)\b",
                       " ", limpa)
        if limpa.strip().startswith(("guaraci ", "cd ")) or re.fullmatch(r"  [\w.-]+", limpa):
            # linha de comando para copiar ou arquivo criado pelo exemplo: nomes, nao texto
            continue
        # caminhos, opcoes (--dados), marcadores de sintaxe (PASTA, [CONFIG]) e o nome do autor
        limpa = limpa.replace("Erley Silva da Costa", " ")
        limpa = " ".join(t for t in limpa.split() if "/" not in t and "\\" not in t and not t.startswith(("-", "["))
                         and not t.strip("[]().,:").isupper())
        v = vazamentos(limpa, "en", ignorar=_NOMES_DOS_EXEMPLOS)
        palavras = {w for w in re.findall(r"[a-z_]+", limpa.lower()) if "_" not in w}   # com "_" = identificador
        v += sorted(palavras & _MAIS_PT)
        if v:
            achados.append((v, linha[:200]))
    return achados


def test_em_portugues_a_linha_volta_igual():
    linha = "[INFO] Validacao group-aware ATIVA: 24 grupos unicos via mae_id."
    assert ic.traduzir_linha(linha, "pt") == linha


@pytest.mark.parametrize("pt, en", [
    ("[INFO] Validacao group-aware ATIVA: 24 grupos unicos via mae_id.",
     "[INFO] Group-aware validation ON: 24 unique groups via mae_id."),
    ("[5/7] Metricas finais (CV):", "[5/7] Final metrics (CV):"),
    ("  [7d/7] Dominio de aplicabilidade: fold 3/18 concluido (0s)", "  [7d/7] Applicability domain: fold 3/18 done (0s)"),
    ("[4/7] Teste de permutacao — PULADO: objective=Quantificacao. Significancia de classificacao nao e' pertinente "
     "fora do modo Classificacao (economiza 200 refits de CV).",
     "[4/7] Permutation test — SKIPPED: objective=Quantification. Classification significance does not apply "
     "outside the Classification mode (saves 200 CV refits)."),
])
def test_traduz_mensagens_do_pipeline(pt, en):
    assert ic.traduzir_linha(pt, "en") == en


def test_auditoria_usa_a_tabela_dos_relatorios():
    linha = ("  [AUDITORIA][AVISO] faixa_validacao_uso: Sem dados de concentracao -- nao e' possivel comparar faixa "
             "de calibracao com faixa de trabalho declarada.")
    en = ic.traduzir_linha(linha, "en")
    assert en.startswith("  [AUDIT][WARNING] faixa_validacao_uso: ") and not vazamentos(en, "en")


def test_painel_do_menu_e_progresso_continuam_lendo_o_texto_original():
    """O painel do menu le o log capturado por regex, com as etiquetas em PT."""
    from guaraci.app_logic import log_progress, log_warnings
    log = "[2/7] LV selection\n  [AVISO] Faixa espectral: so' 88% das 200 variaveis carregam sinal (SNR>=3).\n"
    assert log_warnings(log) and log_progress(log)[0] > 0


def test_tradutor_guarda_a_linha_incompleta_ate_fechar(monkeypatch):
    monkeypatch.setenv("GUARACI_IDIOMA", "en")
    destino = io.StringIO()
    t = ic.TradutorSaida(destino)
    t.write("[5/7] Metricas ")
    assert destino.getvalue() == ""
    t.write("finais (CV):\nresto")
    assert destino.getvalue() == "[5/7] Final metrics (CV):\n"
    t.flush()
    assert destino.getvalue().endswith("resto")


def test_ativar_nao_faz_nada_em_portugues(monkeypatch):
    monkeypatch.setenv("GUARACI_IDIOMA", "pt")
    antes = sys.stdout
    desfazer = ic.ativar()
    try:
        assert sys.stdout is antes and not ic.ativo()
    finally:
        desfazer()


def _roda_em_ingles(tmp_path, *args, timeout=600):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
           "USERPROFILE": str(home), "HOME": str(home)}
    env.pop("GUARACI_IDIOMA", None)      # usuario novo: nunca escolheu idioma -> ingles
    return subprocess.run([sys.executable, "-m", "guaraci", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env, cwd=str(tmp_path), timeout=timeout)


_RAPIDOS = ["comparar", "vip", "monitorar", "transferir"]
_LENTOS = ["explorar", "classificar", "quantificar", "multielementar", "imagem", "hsi", "pares",
           "fluorescencia_eem", "cromatograma_gcms"]


@pytest.mark.parametrize("cenario", _RAPIDOS + [pytest.param(c, marks=pytest.mark.slow) for c in _LENTOS])
def test_exemplo_em_ingles_sem_portugues(tmp_path, cenario):
    r = _roda_em_ingles(tmp_path, "exemplo", cenario, "ex", "--rodar")
    assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
    sobras = _vazamentos_da_saida(r.stdout + "\n" + r.stderr)
    assert not sobras, "\n".join(f"{v}: {l}" for v, l in sobras[:15])


@pytest.mark.slow
def test_demo_em_ingles_sem_portugues(tmp_path):
    r = _roda_em_ingles(tmp_path, "demo")
    assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
    sobras = _vazamentos_da_saida(r.stdout + "\n" + r.stderr)
    assert not sobras, "\n".join(f"{v}: {l}" for v, l in sobras[:15])


@pytest.mark.parametrize("args", [["--help"], ["--version"], ["exemplo"], ["perfis"]])
def test_comandos_curtos_em_ingles(tmp_path, args):
    r = _roda_em_ingles(tmp_path, *args, timeout=120)
    assert r.returncode == 0
    sobras = _vazamentos_da_saida(r.stdout)
    assert not sobras, "\n".join(f"{v}: {l}" for v, l in sobras[:15])


def test_checar_segue_o_idioma_da_sessao(tmp_path):
    """`guaraci checar` caia em portugues enquanto o `run` mostrava as mesmas regras em ingles."""
    _roda_em_ingles(tmp_path, "exemplo", "classificar", "ex")
    r = _roda_em_ingles(tmp_path, "checar", str(tmp_path / "ex" / "config.yaml"), timeout=120)
    assert "[WARNING]" in r.stdout or "No problem found" in r.stdout, r.stdout[-800:]
    assert "[AVISO]" not in r.stdout
