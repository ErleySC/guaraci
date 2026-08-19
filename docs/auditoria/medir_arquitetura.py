"""BLOCO L -- diagnostico de arquitetura: grafo de imports internos,
ciclos, fan-in/fan-out e violacoes de camada.

MEDE, NAO REDESENHA. Nao move, nao renomeia, nao divide arquivo nenhum.

Usa AST da biblioteca padrao -- nao adiciona dependencia (pydeps/
import-linter) so' para gerar um diagnostico de uma rodada.
"""
import ast
import pathlib
import sys
from collections import defaultdict

RAIZ = pathlib.Path("src/guaraci")

# Camadas declaradas por intencao do projeto (do mais baixo ao mais alto).
# Uma aresta de camada BAIXA para camada ALTA e' violacao: significa que o
# nucleo de calculo depende de apresentacao/orquestracao.
CAMADA = {
    0: {"design_tokens", "paleta_cores", "log", "hardware", "io_registry",
        "model_registry", "guaraci_theme", "resumo_parse"},
    1: {"chemometric_stats", "preprocessamento", "conformal"},
    2: {"classificadores", "validacao_estatistica", "modos_analise",
        "config", "config_io"},
    3: {"dados_io", "dados_imagem", "selecao_variaveis", "avaliacao_modelos",
        "predicao", "spectra_preview", "resultados_io"},
    4: {"figuras", "reports"},
    5: {"pipeline"},
    6: {"app_logic", "cli_logic", "cli_assistente", "guaraci"},
}
NOME_CAMADA = {
    0: "utilitario", 1: "calculo puro", 2: "metodo", 3: "dados/analise",
    4: "apresentacao", 5: "orquestracao", 6: "interface",
}


def camada_de(mod: str) -> int:
    for n, mods in CAMADA.items():
        if mod in mods:
            return n
    return -1


def _e_type_checking(no) -> bool:
    """`if TYPE_CHECKING:` -- import que NAO existe em runtime."""
    t = no.test
    return ((isinstance(t, ast.Name) and t.id == "TYPE_CHECKING")
            or (isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"))


def imports_internos(caminho: pathlib.Path, incluir_type_checking=False) -> set:
    """Modulos de guaraci importados por este arquivo.

    Por padrao IGNORA imports sob `if TYPE_CHECKING:` -- eles existem so'
    para anotacao de tipo, nao criam dependencia em tempo de execucao nem
    ciclo real de import. Contar os dois juntos infla o diagnostico e
    inventa violacao de camada onde nao ha.
    """
    try:
        texto = caminho.read_text(encoding="utf-8")
        arvore = ast.parse(texto)
    except (OSError, SyntaxError):
        return set()

    tc_nos = set()
    if not incluir_type_checking:
        for no in ast.walk(arvore):
            if isinstance(no, ast.If) and _e_type_checking(no):
                for filho in ast.walk(no):
                    tc_nos.add(id(filho))

    out = set()
    for no in ast.walk(arvore):
        if id(no) in tc_nos:
            continue
        if isinstance(no, ast.ImportFrom) and no.module:
            if no.module.startswith("guaraci"):
                partes = no.module.split(".")
                if len(partes) >= 2:
                    out.add(partes[1])
        elif isinstance(no, ast.Import):
            for a in no.names:
                if a.name.startswith("guaraci."):
                    out.add(a.name.split(".")[1])
    return out


def achar_ciclos(grafo):
    """Todos os ciclos simples, por DFS com pilha."""
    ciclos, visitando, visitado = [], set(), set()

    def dfs(no, caminho):
        if no in visitando:
            i = caminho.index(no)
            c = caminho[i:] + [no]
            if c not in ciclos:
                ciclos.append(c)
            return
        if no in visitado:
            return
        visitando.add(no)
        for viz in sorted(grafo.get(no, ())):
            dfs(viz, caminho + [viz])
        visitando.discard(no)
        visitado.add(no)

    for no in sorted(grafo):
        dfs(no, [no])
    return ciclos


def main():
    arquivos = sorted(p for p in RAIZ.glob("*.py") if p.stem != "__init__")
    grafo = {p.stem: imports_internos(p) - {p.stem} for p in arquivos}
    # subpacote app_tabs
    for p in sorted((RAIZ / "app_tabs").glob("*.py")):
        if p.stem != "__init__":
            grafo[f"app_tabs.{p.stem}"] = imports_internos(p)

    fan_out = {m: len(d) for m, d in grafo.items()}
    fan_in = defaultdict(int)
    for _m, deps in grafo.items():
        for d in deps:
            fan_in[d] += 1

    print(f"modulos analisados: {len(grafo)}")
    print(f"arestas internas  : {sum(fan_out.values())}\n")

    print("=" * 68)
    print("FAN-IN / FAN-OUT  (in = quantos me importam; out = quantos importo)")
    print("=" * 68)
    print(f"{'modulo':<26} {'camada':<14} {'in':>4} {'out':>5} {'in+out':>7}")
    print("-" * 68)
    for m in sorted(grafo, key=lambda x: -(fan_in[x] + fan_out[x])):
        c = camada_de(m)
        rot = f"{c} {NOME_CAMADA.get(c, '?')}" if c >= 0 else "- (nao mapeado)"
        print(f"{m:<26} {rot:<14} {fan_in[m]:>4} {fan_out[m]:>5} "
              f"{fan_in[m] + fan_out[m]:>7}")

    print("\n" + "=" * 68)
    print("CICLOS DE IMPORT")
    print("=" * 68)
    ciclos = achar_ciclos(grafo)
    if not ciclos:
        print("  nenhum ciclo detectado.")
    for c in ciclos:
        print("  " + " -> ".join(c))

    print("\n" + "=" * 68)
    print("VIOLACOES DE CAMADA (modulo baixo importando modulo alto)")
    print("=" * 68)
    viol = []
    for m, deps in grafo.items():
        cm = camada_de(m)
        if cm < 0:
            continue
        for d in deps:
            cd = camada_de(d)
            if cd >= 0 and cd > cm:
                viol.append((m, cm, d, cd))
    if not viol:
        print("  nenhuma violacao.")
    for m, cm, d, cd in sorted(viol, key=lambda x: x[3] - x[1], reverse=True):
        print(f"  {m} (camada {cm} {NOME_CAMADA[cm]}) -> "
              f"{d} (camada {cd} {NOME_CAMADA[cd]})   salto +{cd - cm}")

    print("\n" + "=" * 68)
    print("ARESTAS (para o grafo)")
    print("=" * 68)
    for m in sorted(grafo):
        for d in sorted(grafo[m]):
            print(f"  {m} -> {d}")


if __name__ == "__main__":
    sys.exit(main())
