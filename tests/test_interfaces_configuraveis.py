"""Rede de segurança: toda chave do _CONFIG_SPEC (fonte única de verdade)
deve estar acessível ao operador nas DUAS interfaces interativas — o app web
(app_quimiometria.py) e o assistente de terminal (cli_assistente.py).

Motivação: encontramos, na auditoria de 2026-07-04, quatro campos que
existiam no motor mas não apareciam em uma das interfaces (modo_ddsimca
ausente no app; selecao_spa/selecao_ag/figuras_detalhadas/n_jobs_permutacao
ausentes do menu do CLI, apesar de terem metadados "ricos" cadastrados —
metadado sem entrada no menu real não aparece para o operador). Este teste
pega esse tipo de lacuna automaticamente na próxima vez que alguém adicionar
um campo novo ao Config/_CONFIG_SPEC e esquecer de registrá-lo em uma das
duas interfaces.

Não importa app_quimiometria.py (script Streamlit — top-level `st.*` exige
um ScriptRunContext ativo, não é seguro importar fora de `streamlit run`);
em vez disso faz parsing de texto de QUALQUER lista `_..._KEYS...` do
código-fonte (`_DADOS_KEYS`, `_PREPROC_KEYS`, `_MODELO_KEYS_*`, e futuras —
regex genérico, não uma lista hardcoded de nomes conhecidos, para não ficar
defasado quando uma aba nova for adicionada). Desde a quebra do app por aba
(item 18), essas listas moram nos módulos de guaraci/app_tabs/, não mais em
app_quimiometria.py — o scan cobre os dois lugares. cli_assistente.py é
seguro de importar (sem I/O bloqueante em nível de módulo) e expõe
`MENU_FIELDS` diretamente.
"""
import glob
import os
import re


def _chaves_no_app() -> set:
    raiz = os.path.join(os.path.dirname(__file__), "..")
    caminhos = [os.path.join(raiz, "app_quimiometria.py")]
    caminhos += glob.glob(os.path.join(raiz, "src", "guaraci", "app_tabs", "*.py"))
    chaves = set()
    for caminho in caminhos:
        with open(caminho, encoding="utf-8") as f:
            src = f.read()
        for m in re.finditer(r"_\w*KEYS\w*\s*=\s*\[(.*?)\]", src, re.S):
            chaves.update(re.findall(r'"(\w+)"', m.group(1)))
    return chaves


def _menu_fields_do_cli() -> dict:
    import guaraci.cli_assistente as mod
    return mod.MENU_FIELDS


def test_todo_campo_do_config_spec_aparece_no_app(pq):
    """Nenhuma chave do _CONFIG_SPEC pode ficar de fora das listas
    _MODELO_KEYS_* (ou das abas dedicadas) do app web."""
    chaves_spec = {s["key"] for s in pq._CONFIG_SPEC}
    chaves_app = _chaves_no_app()
    faltando = chaves_spec - chaves_app
    assert not faltando, (
        f"Campo(s) do _CONFIG_SPEC sem widget no app web: {sorted(faltando)}. "
        f"Adicione em _MODELO_KEYS_ANALISE/_VALID/_EXTRAS/_FIGURAS "
        f"(guaraci/app_tabs/modelo.py) ou na aba correspondente.")


def test_todo_campo_do_config_spec_aparece_no_menu_cli(pq):
    """Nenhuma chave do _CONFIG_SPEC pode ficar de fora de MENU_FIELDS —
    metadados "ricos" (RISK_CLASS, rótulos bilíngues, help) sem entrada em
    MENU_FIELDS não tornam o campo visível/editável no menu real do CLI."""
    chaves_spec = {s["key"] for s in pq._CONFIG_SPEC}
    menu_fields = _menu_fields_do_cli()
    chaves_cli = {k for lista in menu_fields.values() for k in lista}
    faltando = chaves_spec - chaves_cli
    assert not faltando, (
        f"Campo(s) do _CONFIG_SPEC sem entrada em MENU_FIELDS "
        f"(cli_assistente.py): {sorted(faltando)}. Adicione a chave na lista "
        f"da categoria correta em MENU_FIELDS.")
