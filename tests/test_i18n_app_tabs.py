"""Contra-prova: as 4 abas do app web (Data/Preprocessing/Prediction/
Reports) ficaram sempre em ingles ate 2026-09-01, mesmo com idioma=PT
selecionado -- nenhuma delas recebia a funcao de traducao `T`. Dois
contratos aqui:

1. Toda chamada `T("...")` (string literal) nas 4 abas tem entrada
   correspondente em `_TR` (app_quimiometria.py) -- pega string nova
   adicionada sem traducao.
2. Rodando o app de verdade com `session_state.lang = "PT"`, o texto
   traduzido aparece de fato (nao so' a chave existe, a troca funciona).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

_RAIZ = Path(__file__).resolve().parents[1]
_ARQUIVOS_ABAS = [
    _RAIZ / "src" / "guaraci" / "app_tabs" / "dados.py",
    _RAIZ / "src" / "guaraci" / "app_tabs" / "preprocessamento.py",
    _RAIZ / "src" / "guaraci" / "app_tabs" / "predicao.py",
    _RAIZ / "src" / "guaraci" / "app_tabs" / "relatorios.py",
]


def _chaves_t_usadas() -> set:
    chaves = set()
    for arq in _ARQUIVOS_ABAS:
        tree = ast.parse(arq.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "T" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                chaves.add(node.args[0].value)
    # preprocessamento.py tambem traduz valores de _PRESET_INFO via
    # T(_PRESET_INFO[...]) -- indireto, adiciona os valores do dict.
    prep_tree = ast.parse(
        (_RAIZ / "src" / "guaraci" / "app_tabs" / "preprocessamento.py")
        .read_text(encoding="utf-8"))
    for node in ast.walk(prep_tree):
        if (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_PRESET_INFO"):
            for v in node.value.values:
                if isinstance(v, ast.Constant):
                    chaves.add(v.value)
    return chaves


def _chaves_tr() -> set:
    tree = ast.parse((_RAIZ / "app_quimiometria.py").read_text(encoding="utf-8"))
    chaves = set()
    for node in ast.walk(tree):
        alvo = None
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == "_TR"):
            alvo = node.value
        elif (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_TR" for t in node.targets)):
            alvo = node.value
        if alvo is not None and isinstance(alvo, ast.Dict):
            for k in alvo.keys:
                if isinstance(k, ast.Constant):
                    chaves.add(k.value)
    return chaves


def test_toda_chave_t_das_4_abas_tem_traducao_em_tr():
    faltando = _chaves_t_usadas() - _chaves_tr()
    assert not faltando, (
        f"{len(faltando)} chave(s) usada(s) via T(...) nas abas Data/"
        f"Preprocessing/Prediction/Reports sem entrada em _TR "
        f"(app_quimiometria.py): {sorted(faltando)[:5]}...")


def test_placeholders_de_format_batem_entre_chave_e_traducao_pt():
    """`.format(x=...)` explode se a string traduzida perder um {x} que a
    chave original tinha -- verifica que PT preserva os mesmos nomes."""
    tree = ast.parse((_RAIZ / "app_quimiometria.py").read_text(encoding="utf-8"))
    problemas = []
    for node in ast.walk(tree):
        alvo = None
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == "_TR"):
            alvo = node.value
        elif (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_TR" for t in node.targets)):
            alvo = node.value
        if alvo is None or not isinstance(alvo, ast.Dict):
            continue
        for k, v in zip(alvo.keys, alvo.values):
            if not (isinstance(k, ast.Constant) and isinstance(v, ast.Dict)):
                continue
            ph_chave = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)", k.value))
            if not ph_chave:
                continue
            for kk, vv in zip(v.keys, v.values):
                if (isinstance(kk, ast.Constant) and kk.value == "PT"
                        and isinstance(vv, ast.Constant)):
                    ph_pt = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)", vv.value))
                    if ph_pt != ph_chave:
                        problemas.append((k.value, ph_chave, ph_pt))
    assert not problemas, problemas


def test_abas_traduzem_de_verdade_com_lang_pt():
    """Nao basta a chave existir em _TR -- roda o app de verdade com
    session_state.lang='PT' e confirma que o texto PT aparece (contra-prova
    de que T() esta' de fato recebendo _T, nao um no-op esquecido numa
    chamada de render())."""
    at = AppTest.from_file(str(_RAIZ / "app_quimiometria.py"), default_timeout=30)
    at.session_state["lang"] = "PT"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    textos = ("\n".join(m.value for m in at.markdown)
              + "\n".join(s.value for s in at.subheader)
              + "\n".join(c.value for c in at.caption)
              + "\n".join(i.value for i in at.info))
    assert "Entrada de Dados" in textos       # Data tab
    assert "Data Input" not in textos
    assert "Pré-processamento Espectral" in textos   # Preprocessing tab
    assert "Spectral Preprocessing" not in textos
    assert any("Predizer" in b.label for b in at.button)   # Prediction tab
    assert "Relatórios e Downloads" in textos   # Reports tab (early-return msg)
    assert "Execute o pipeline (aba Modelo) para gerar relatórios." in textos
