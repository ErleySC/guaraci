"""Guarda permanente (M-01/M-02, parecer 2026-09-25): nenhum ponto do codigo
de producao carrega pickle fora de `guaraci.predicao.load_model`.

Equivalente local da regra semgrep `.semgrep/pickle.yml` que roda no CI;
este teste nao depende de rede nem de semgrep.
"""
from __future__ import annotations

import ast
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
_PROIBIDAS = {("joblib", "load"), ("pickle", "load"), ("pickle", "loads"),
              ("dill", "load"), ("cloudpickle", "load")}
_PERMITIDO = _RAIZ / "src" / "guaraci" / "predicao.py"


def _arquivos():
    yield from (_RAIZ / "src" / "guaraci").rglob("*.py")
    yield _RAIZ / "app_quimiometria.py"
    yield from (_RAIZ / "scripts").rglob("*.py")


def test_nenhum_carregamento_de_pickle_fora_de_load_model():
    achados = []
    for arq in _arquivos():
        if arq == _PERMITIDO:
            continue
        tree = ast.parse(arq.read_text(encoding="utf-8"))
        for no in ast.walk(tree):
            if (isinstance(no, ast.Call) and isinstance(no.func, ast.Attribute)
                    and isinstance(no.func.value, ast.Name)
                    and (no.func.value.id, no.func.attr) in _PROIBIDAS):
                achados.append(f"{arq.relative_to(_RAIZ)}:{no.lineno}")
    assert not achados, (
        "pickle carregado fora de predicao.load_model (sem manifesto "
        f"assinado): {achados}")


def test_o_unico_ponto_permitido_verifica_o_manifesto_antes_de_carregar():
    fonte = _PERMITIDO.read_text(encoding="utf-8")
    i_verifica = fonte.index("dados = _verificar_manifesto(str(caminho))")
    # A1-06: carrega os MESMOS bytes conferidos, nunca reabre pelo caminho
    i_carrega = fonte.index("joblib.load(io.BytesIO(dados))")
    assert i_verifica < i_carrega
    assert "joblib.load(caminho)" not in fonte
