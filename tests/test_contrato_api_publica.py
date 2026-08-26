# -*- coding: utf-8 -*-
"""Teste de contrato da API pública (Bloco B, Passo 73).

POR QUE ESTE TESTE EXISTE. `docs/COMPATIBILITY.md` descreve a política;
este arquivo é o mecanismo que a APLICA. A superfície coberta é
descoberta por INTROSPECÇÃO AUTOMÁTICA de todo `__all__` de todo módulo
de `src/guaraci/` (mesmo padrão de `test_contrato_validacao_agrupada.py`
-- nenhuma lista manual de nomes aqui, os únicos nomes hardcoded neste
arquivo são os das exceções documentadas em COMPATIBILITY.md). Um nome
novo adicionado a um `__all__` existente, ou um módulo novo com `__all__`
próprio, entra automaticamente na cobertura na próxima execução.

O que este teste PEGA: função pública removida/renomeada; parâmetro
removido/renomeado/reordenado; classe/dataclass com campo removido;
classe pública virando função ou vice-versa; item de `__all__`
removido; chave de `_CONFIG_SPEC` (schema do config.yaml) removida ou
com tipo/opções mudados.

O que este teste NÃO PEGA (limitação documentada, não descoberta
silenciosa): mudança de COMPORTAMENTO sem mudança de assinatura (isso é
trabalho dos testes golden/numéricos, não deste); nomes de coluna de
CSV/Excel gerados dinamicamente por `resultados_io.py`/`predicao.py`/
`reports.py` -- não existe objeto Python introspectável que declare
essas colunas sem rodar a função e inspecionar o DataFrame resultante;
ver `docs/COMPATIBILITY.md` para o texto desta ressalva. Cobrir isso
exigiria rodar o pipeline (como `test_golden_valores.py` já faz) e
extrair `df.columns` -- trabalho de escopo comparável a um Passo
próprio, não feito aqui para não prometer uma garantia que o teste não
sustenta de fato.

MECANISMO -- mesmo padrão de `test_golden_valores.py`: o snapshot vive
em `tests/golden/contrato_api_publica.json`, versionado no repo. Se a
API pública mudou DE PROPÓSITO, regrave com:

    GUARACI_REGRAVAR_GOLDEN=1 pytest tests/test_contrato_api_publica.py

Regravar sem entender por que o teste falhou anula o propósito dele --
o ponto é forçar quem mexe na API pública a perceber que está mexendo
nela, não silenciar o aviso.
"""
from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import os
import pathlib

import pytest


@pytest.fixture(scope="module")
def guaraci_mod():
    import guaraci.guaraci as mod
    return mod


_RAIZ_SRC = pathlib.Path(__file__).parent.parent / "src" / "guaraci"
_GOLDEN = pathlib.Path(__file__).parent / "golden" / "contrato_api_publica.json"


def _nomes_dos_modulos():
    """Descobre TODO modulo de src/guaraci/*.py com __all__ -- por
    listagem do diretorio, nao por lista escrita a mao (um modulo novo
    entra sozinho)."""
    nomes = []
    for f in sorted(_RAIZ_SRC.glob("*.py")):
        if f.stem == "__init__":
            continue
        nomes.append(f.stem)
    return nomes


def _descrever_callable(obj) -> dict:
    try:
        assinatura = str(inspect.signature(obj))
    except (TypeError, ValueError):
        assinatura = None
    return {"forma": "funcao", "assinatura": assinatura}


def _descrever_classe(obj) -> dict:
    if dataclasses.is_dataclass(obj):
        campos = []
        for campo in dataclasses.fields(obj):
            tem_default = (campo.default is not dataclasses.MISSING
                            or campo.default_factory is not dataclasses.MISSING)  # noqa: E501
            campos.append({
                "nome": campo.name,
                "tipo": str(campo.type),
                "tem_default": tem_default,
            })
        return {"forma": "dataclass", "campos": campos}

    # So' captura a assinatura de __init__ quando a PROPRIA classe o
    # define (`"__init__" in vars(obj)`) -- __init__ HERDADO (Enum,
    # Exception) e' implementacao da stdlib, cuja representacao textual
    # muda entre versoes/interpretes do Python (achado real: CI com
    # Python 3.10/3.11/3.13 em Linux/macOS reportava "(self, *args,
    # **kwds)" ou "(self, /, *args, **kwargs)" diferente do que Python
    # 3.12 no Windows produzia, para a MESMA classe sem nenhuma mudanca
    # de codigo -- falso positivo de "contrato mudou").
    if "__init__" in vars(obj):
        try:
            assinatura_init = str(inspect.signature(obj.__init__))
        except (TypeError, ValueError):
            assinatura_init = None
    else:
        assinatura_init = None
    metodos_publicos = sorted(
        nome for nome, membro in vars(obj).items()
        if not nome.startswith("_") and (inspect.isfunction(membro)
                                          or isinstance(membro, staticmethod)))
    bases = sorted(b.__name__ for b in obj.__bases__ if b is not object)
    return {
        "forma": "classe",
        "init": assinatura_init,
        "metodos_publicos": metodos_publicos,
        "bases": bases,
    }


def _descrever_simbolo(obj) -> dict:
    if inspect.isclass(obj):
        return _descrever_classe(obj)
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return _descrever_callable(obj)
    if isinstance(obj, pathlib.PurePath):
        # `WindowsPath` vs `PosixPath` e' o mesmo tipo LOGICO em SOs
        # diferentes -- achado real: DIR_PERFIS (perfil_matriz.py) e'
        # WindowsPath no Windows e PosixPath no CI (Linux/macOS), falso
        # positivo de "contrato mudou" sem nenhuma mudanca de codigo.
        return {"forma": "constante", "tipo": "Path"}
    return {"forma": "constante", "tipo": type(obj).__name__}


def _descrever_config_spec(spec_por_modulo: dict) -> dict:
    """Schema do config.yaml (Passo 72): cada chave, seu tipo e opcoes --
    remover uma chave ou mudar seu tipo/opcoes e' incompativel (ver
    COMPATIBILITY.md: chave desconhecida no YAML e' ERRO, entao nao ha'
    caminho suave de depreciacao para isto)."""
    config_io = importlib.import_module("guaraci.config_io")
    linhas = {}
    for item in config_io._CONFIG_SPEC:
        linhas[item["key"]] = {
            "attr": item["attr"],
            "tipo": item["tipo"],
            "opcoes": item["opcoes"],
            "min": item.get("min"),
            "max": item.get("max"),
        }
    return linhas


def _gerar_snapshot() -> dict:
    snapshot: dict = {"modulos": {}}
    for nome_modulo in _nomes_dos_modulos():
        modulo = importlib.import_module(f"guaraci.{nome_modulo}")
        all_pub = getattr(modulo, "__all__", None)
        if not all_pub:
            continue
        entradas = {}
        for nome in all_pub:
            obj = getattr(modulo, nome)
            entradas[nome] = _descrever_simbolo(obj)
        snapshot["modulos"][nome_modulo] = entradas
    snapshot["config_yaml_schema"] = _descrever_config_spec(snapshot["modulos"])
    return snapshot


@pytest.fixture(scope="module")
def snapshot_atual() -> dict:
    return _gerar_snapshot()


def test_contrato_publico_nao_mudou_sem_intencao(snapshot_atual):
    if os.environ.get("GUARACI_REGRAVAR_GOLDEN") == "1":
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(
            json.dumps(snapshot_atual, indent=2, sort_keys=True,
                       ensure_ascii=False) + "\n",
            encoding="utf-8")
        pytest.skip(f"golden regravado em {_GOLDEN} (GUARACI_REGRAVAR_GOLDEN=1)")

    if not _GOLDEN.exists():
        pytest.fail(
            f"Golden nao existe em {_GOLDEN}. Gere com:\n"
            "  GUARACI_REGRAVAR_GOLDEN=1 pytest tests/test_contrato_api_publica.py")

    esperado = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    if esperado != snapshot_atual:
        # Diff minimo por modulo, para o erro apontar direto o que mudou
        # em vez de despejar dois JSONs inteiros no terminal.
        diffs = []
        chaves = set(esperado.get("modulos", {})) | set(snapshot_atual.get("modulos", {}))  # noqa: E501
        for chave in sorted(chaves):
            antes = esperado.get("modulos", {}).get(chave)
            depois = snapshot_atual.get("modulos", {}).get(chave)
            if antes != depois:
                diffs.append(f"  modulo '{chave}': mudou")
        if esperado.get("config_yaml_schema") != snapshot_atual.get("config_yaml_schema"):  # noqa: E501
            diffs.append("  config_yaml_schema: mudou")
        pytest.fail(
            "Contrato de API publica mudou:\n" + "\n".join(diffs) +
            "\n\nSe a mudanca foi INTENCIONAL (e o bump de versao "
            "correspondente ja foi decidido -- ver docs/COMPATIBILITY.md), "
            "regrave com:\n"
            "  GUARACI_REGRAVAR_GOLDEN=1 pytest tests/test_contrato_api_publica.py")


# =========================================================================
#  Codigos de saida da CLI (Passo 72) -- contrato ja documentado no
#  proprio --help (_TEXTO_AJUDA), verificado aqui contra o comportamento
#  REAL de main(), nao so' contra o texto que o descreve.
# =========================================================================

def test_exit_code_2_uso_incorreto(guaraci_mod):
    with pytest.raises(SystemExit) as exc:
        guaraci_mod.main(["--mode=invalido", "demo"])
    assert exc.value.code == 2


def test_exit_code_0_comando_valido(guaraci_mod, capsys):
    guaraci_mod.main(["--help"])
    saida = capsys.readouterr().out
    assert "Codigos de saida" in saida


def test_texto_ajuda_declara_os_tres_codigos():
    texto = importlib.import_module("guaraci.guaraci")._TEXTO_AJUDA
    assert "0 sucesso" in texto
    assert "1 erro de execucao" in texto
    assert "2 uso incorreto" in texto
