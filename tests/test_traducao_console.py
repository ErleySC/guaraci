# -*- coding: utf-8 -*-
"""Console em ingles sem frase em portugues (problema 1 da rodada da CLI de 2026-11-20).

Numa sessao em ingles -- o padrao de quem nunca escolheu idioma -- o `guaraci run` mostrava o pre-voo em
ingles e o log do pipeline em portugues. O tradutor do console (traducao_console) traduz a saida linha a
linha. Aqui: a mecanica do tradutor e, rodando os exemplos guiados de verdade em ingles, que nenhuma palavra
so' do portugues sobra no que o usuario le.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from guaraci import traducao_console as tc
from guaraci.idioma_relatorio import vazamentos

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
    assert tc.traduzir_linha(linha, "pt") == linha


@pytest.mark.parametrize("pt, en", [
    ("[INFO] Validacao group-aware ATIVA: 24 grupos unicos via mae_id.",
     "[INFO] Group-aware validation ON: 24 unique groups via mae_id."),
    ("[5/7] Metricas finais (CV):", "[5/7] Final metrics (CV):"),
    ("  [7d/7] Dominio de aplicabilidade: fold 3/18 concluido (0s)", "  [7d/7] Applicability domain: fold 3/18 done (0s)"),
    ("[4/7] Teste de permutacao — PULADO: objective=Quantificacao. Significancia de classificacao nao e' pertinente "
     "fora do modo Classificacao (economiza 200 refits de CV).",
     "[4/7] Permutation test -- SKIPPED: objective=Quantification. Classification significance does not apply "
     "outside Classification mode (saves 200 CV refits)."),
])
def test_traduz_mensagens_do_pipeline(pt, en):
    assert tc.traduzir_linha(pt, "en") == en


def test_auditoria_usa_a_tabela_dos_relatorios():
    linha = ("  [AUDITORIA][AVISO] faixa_validacao_uso: Sem dados de concentracao -- nao e' possivel comparar faixa "
             "de calibracao com faixa de trabalho declarada.")
    en = tc.traduzir_linha(linha, "en")
    assert en.startswith("  [AUDIT][WARNING] faixa_validacao_uso: ") and not vazamentos(en, "en")


def test_painel_do_menu_e_progresso_continuam_lendo_o_texto_original():
    """O tradutor so' age na saida final: o painel do menu le o log capturado (em PT) por regex."""
    from guaraci.app_logic import log_progress, log_warnings
    log = "[2/7] LV selection\n  [AVISO] Faixa espectral: so' 88% das 200 variaveis carregam sinal (SNR>=3).\n"
    assert log_warnings(log) and log_progress(log)[0] > 0


def test_tradutor_guarda_a_linha_incompleta_ate_fechar(monkeypatch):
    monkeypatch.setenv("GUARACI_IDIOMA", "en")
    destino = io.StringIO()
    t = tc.TradutorSaida(destino)
    t.write("[5/7] Metricas ")
    assert destino.getvalue() == ""
    t.write("finais (CV):\nresto")
    assert destino.getvalue() == "[5/7] Final metrics (CV):\n"
    t.flush()
    assert destino.getvalue().endswith("resto")


def test_ativar_nao_faz_nada_em_portugues(monkeypatch):
    monkeypatch.setenv("GUARACI_IDIOMA", "pt")
    antes = sys.stdout
    desfazer = tc.ativar()
    try:
        assert sys.stdout is antes and not tc.ativo()
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
