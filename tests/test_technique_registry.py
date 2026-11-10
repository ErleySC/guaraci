"""Rede de seguranca do catalogo de tecnicas do assistente `G` (Agente 6).

Dois contratos, cada um pegando um jeito diferente do catalogo mentir:

1. Toda `referencia` de TechniqueEntry precisa apontar pra um simbolo que
   EXISTE de verdade (pega entrada que sobrevive a uma renomeacao/remocao
   do simbolo real -- o catalogo afirmaria um metodo que nao existe mais).
2. Para os modulos de proposito UNICO (todo simbolo publico e' de fato uma
   tecnica, listados em MODULOS_COBERTURA_TOTAL), todo `__all__` precisa
   estar OU no registry OU na lista de excecoes documentada -- pega metodo
   novo adicionado sem atualizar o catalogo (o motivo desta rodada existir:
   `_guaraci_navegar_secoes` tinha virado uma segunda fonte, desatualizada,
   do que ja existia em `_CONFIG_SPEC`/`_HELP_DB` -- ver docs/projeto/identidade_visual.md).
"""
from __future__ import annotations

import importlib

from guaraci.technique_registry import MODULOS_COBERTURA_TOTAL, REGISTRY


def test_toda_referencia_do_catalogo_resolve_para_simbolo_real():
    faltando = []
    for entry in REGISTRY:
        modulo_nome, _, simbolo = entry.referencia.rpartition(".")
        modulo = importlib.import_module(modulo_nome)
        if not hasattr(modulo, simbolo):
            faltando.append(f"{entry.id}: {entry.referencia}")
    assert not faltando, (
        f"Entrada(s) do catalogo apontando pra simbolo inexistente: "
        f"{faltando}. O metodo foi renomeado/removido -- atualize a "
        f"`referencia` em technique_registry.py.")


def test_ids_do_catalogo_sao_unicos():
    ids = [e.id for e in REGISTRY]
    assert len(ids) == len(set(ids)), f"id duplicado em REGISTRY: {ids}"


def test_modulos_de_proposito_unico_tem_cobertura_completa():
    referenciados = {e.referencia.rsplit(".", 1)[-1] for e in REGISTRY}
    faltando = {}
    for modulo_nome, regra in MODULOS_COBERTURA_TOTAL.items():
        modulo = importlib.import_module(modulo_nome)
        todos = set(getattr(modulo, "__all__", ()))
        cobertos = regra["incluir"] | regra["excecoes"]
        sobra = todos - cobertos
        if sobra:
            faltando[modulo_nome] = sorted(sobra)
        # Contra-prova reversa: tudo que a regra diz "incluir" precisa
        # de fato estar referenciado por alguma entrada do catalogo --
        # senao a regra promete cobertura que o REGISTRY nao cumpre.
        nao_registrado = regra["incluir"] - referenciados
        assert not nao_registrado, (
            f"{modulo_nome}: simbolo(s) em 'incluir' sem entrada "
            f"correspondente no REGISTRY: {sorted(nao_registrado)}")
    assert not faltando, (
        f"Modulo(s) de proposito unico com simbolo novo em __all__ sem "
        f"entrada no REGISTRY nem na lista de excecoes de "
        f"MODULOS_COBERTURA_TOTAL (technique_registry.py): {faltando}. "
        f"Adicione uma TechniqueEntry para o metodo novo, ou -- se ele "
        f"genuinamente nao for uma 'tecnica' (dataclass de resultado, "
        f"excecao, helper) -- inclua o nome em 'excecoes'.")


def test_categorias_do_catalogo_sao_um_conjunto_pequeno_e_fechado():
    """Categoria e' o agrupamento que o assistente usa pra apresentar o
    catalogo -- uma categoria nova precisa ser deliberada, nao surgir de
    um typo numa entrada nova."""
    categorias_conhecidas = {
        "classificacao_deteccao", "quantificacao",
        "identificacao_conjunto_aberto", "selecao_amostras",
        "transferencia_calibracao", "figuras_de_merito",
        "robustez_linearidade", "perfis", "resolucao_mistura",
        # deliberada (2026-10-02, Fase 10): ANOVA de grupos sobre a unidade
        "comparacao_grupos",
        # deliberada (2026-10-03, FECHAR O CLI Fase 2): `guaraci monitorar`
        "monitoramento",
        # deliberada (2026-10-03, FECHAR O CLI Fase 8): tela T item 7 (com x sem, Wilcoxon pareado)
        "portao_aceite",
        # deliberada (2026-10-06, ultima adicao antes da v1.0.0): VIP com estabilidade
        "importancia_variaveis",
    }
    usadas = {e.categoria for e in REGISTRY}
    assert usadas <= categorias_conhecidas, (
        f"Categoria(s) nao reconhecida(s): {usadas - categorias_conhecidas}. "
        f"Se for deliberada, adicione a categoria_conhecidas neste teste.")


def _modulos_alcancaveis_pelas_interfaces() -> set:
    """Fecho de imports `guaraci.*` a partir das entradas que o usuario usa:
    CLI (`guaraci.guaraci`, `cli_assistente`), web (`app_quimiometria.py` e
    `app_tabs/`). Varredura por AST, sem importar nada."""
    import ast
    import pathlib

    raiz = pathlib.Path(__file__).resolve().parents[1]
    src = raiz / "src" / "guaraci"

    def arquivo(mod):
        p = src.joinpath(*mod.split(".")[1:])
        if p.with_suffix(".py").exists():
            return p.with_suffix(".py")
        return p / "__init__.py" if (p / "__init__.py").exists() else None

    def imports_de(arq):
        out = set()
        for no in ast.walk(ast.parse(arq.read_text(encoding="utf-8"))):
            if isinstance(no, ast.ImportFrom) and no.module and no.module.startswith("guaraci"):
                out.add(no.module)
                out |= {f"{no.module}.{a.name}" for a in no.names}
            elif isinstance(no, ast.Import):
                out |= {a.name for a in no.names if a.name.startswith("guaraci")}
        return out

    fila = ["guaraci.guaraci", "guaraci.cli_assistente"]
    fila += [f"guaraci.app_tabs.{p.stem}" for p in (src / "app_tabs").glob("*.py")]
    fila += list(imports_de(src / "app_web.py"))
    vistos: set = set()
    while fila:
        m = fila.pop()
        if m in vistos:
            continue
        vistos.add(m)
        a = arquivo(m)
        if a:
            fila += list(imports_de(a))
    return {m for m in vistos if arquivo(m)}


def test_toda_tecnica_do_catalogo_e_alcancavel_pela_cli_ou_web():
    """Fase 8 (2026-10-02): tecnica listada no catalogo cujo modulo nenhuma
    interface importa so' existiria pela API -- o catalogo prometeria ao
    usuario algo que o menu nao alcanca."""
    alcancaveis = _modulos_alcancaveis_pelas_interfaces()
    fora = [(e.id, e.referencia) for e in REGISTRY
            if ".".join(e.referencia.split(".")[:2]) not in alcancaveis]
    assert not fora, f"tecnicas do catalogo fora do alcance da CLI/web: {fora}"
