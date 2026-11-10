"""Estrutura do manual que o usuario le: secao 5 numerada, secoes de ajuda ao usuario, tabela conferida, topico de suporte."""
from __future__ import annotations

import re
from pathlib import Path

from guaraci import cli_comandos as cmd
from guaraci import docs_pacote as dp

RAIZ = Path(__file__).resolve().parents[1]
_PROIBIDO = re.compile(r"Passo \d|Bloco ?\d|Grupo \d|\b[Ss]print|\b20\d\d-\d\d-\d\d\b")


def _secao5() -> list[str]:
    linhas = dp.ler("manual").splitlines()
    ini = next(i for i, ln in enumerate(linhas) if ln.startswith("## 5 "))
    fim = next(i for i, ln in enumerate(linhas) if i > ini and ln.startswith("## 6 "))
    return linhas[ini:fim]


def test_subsecoes_da_secao_5_sao_5_1_a_5_n_em_ordem():
    nums = [re.match(r"### (\S+) ", ln).group(1) for ln in _secao5() if ln.startswith("### ")]
    assert nums == [f"5.{i}" for i in range(1, len(nums) + 1)] and len(nums) >= 11


def test_secao_5_do_manual_nao_traz_passo_bloco_grupo_nem_data():
    achados = [ln for ln in _secao5() if _PROIBIDO.search(ln)]
    assert not achados, achados[:3]


def test_manual_tem_as_secoes_de_ajuda_ao_usuario():
    texto = dp.ler("manual")
    for titulo in ("## 10 Perguntas frequentes e problemas comuns", "## 11 Glossário", "## 12 Requisitos mínimos", "## 13 Como reportar erro"):
        assert titulo in texto
    glossario = dp.secao_do_manual("11")
    assert glossario is not None
    for termo in ("LV", "RMSEP", "`mae_id`", "LOGO", "DD-SIMCA"):
        assert termo in glossario, termo


def test_requisitos_do_manual_nao_contradizem_o_guia_de_instalacao():
    guia = (RAIZ / "docs" / "usuario" / "instalacao.md").read_text(encoding="utf-8")
    req = dp.secao_do_manual("12")
    assert "4 GB de RAM e 2 núcleos" in guia and "4 GB de RAM e 2 núcleos" in req
    assert "3.11 a 3.13" in req and "macOS" in req


def test_tabela_de_recomendacao_tem_os_oito_cenarios_do_texto():
    sec = dp.secao_do_manual("5.9")
    linhas = [ln for ln in sec.splitlines() if ln.startswith("| ") and "%" in ln]
    assert len(linhas) == 8                       # o texto diz "seis dos oito cenarios"
    assert "oito cenários" in sec
    for valor in ("0,072", "0,089"):              # os dois que o texto cita como fora da faixa
        assert any(valor in ln for ln in linhas), valor


def test_manual_cita_docs_e_scripts_do_repositorio_so_com_a_ressalva():
    texto = dp.ler("manual")
    problemas = []
    for m in re.finditer(r"`((?:docs|scripts)/[^`]+)`", texto):
        contexto = texto[m.end(): m.end() + 140]
        if "só na cópia do repositório" not in contexto:
            problemas.append(m.group(1))
    assert not problemas, problemas


def test_topico_suporte_existe_e_diz_como_reportar(capsys):
    assert cmd.subcomando("manual", ["suporte"]) == 0
    saida = capsys.readouterr().out
    assert "guaraci --version" in saida and "7 dias" in saida
    assert "suporte" in dp.lista()


def test_manual_nao_manda_ao_usuario_para_arquivos_so_do_repositorio_sem_alternativa():
    texto = dp.ler("manual")
    assert "ver `SECURITY.md`" not in texto and "`SECURITY.md` na raiz" not in texto
