"""cli_comandos.py -- `guaraci --help` e subcomandos novos, derivados do catalogo.

Leve de proposito (sem numpy/sklearn no import): `guaraci --help` e
`guaraci --version` nao pagam o custo do pipeline. Os subcomandos pesados
importam o que precisam so' quando rodam.
"""
from __future__ import annotations

import os

from guaraci.config import pasta_execucao
import sys
import textwrap
from typing import Sequence

from guaraci import catalogo as cat

__all__ = ["texto_ajuda", "SUBCOMANDOS", "subcomando"]

#: Subcomandos tratados aqui (com argumentos proprios, antes da extracao de
#: `--chave=valor` do main). `run`, `demo`, `perfis`, `doctor` e `web` seguem em guaraci.py.
SUBCOMANDOS = ("anova", "vip", "monitorar", "relatorio", "transferir", "exemplo", "confianca", "assistente", "ajuda", "buscar", "manual", "checar")

_OPCOES_PT = """Opcoes (valem para os comandos de analise):
  --perfil=NOME     perfil da matriz analisada (padrao: generico). Define
                    faixa espectral, pre-processamento e o vocabulario da
                    saida. `guaraci perfis` lista os nomes; tambem aceita o
                    caminho de um YAML proprio.
  --sem-preflight   `run` executa mesmo com achado critico do pre-voo do
                    assistente (sem a opcao, `run` para antes de comecar,
                    como `guaraci checar`, que sai com 1).
  --mode=MODO       'cego' (PADRAO) ou 'controle'.
                    cego     : a quantificacao usa a classe PREDITA pelo
                               classificador -- o unico modo que corresponde
                               ao uso real, em que a classe da amostra e'
                               desconhecida.
                    controle : usa a classe VERDADEIRA. So' para diagnostico
                               interno (separar erro de quantificacao de erro
                               de classificacao). Os numeros obtidos assim
                               NAO representam desempenho de uso, e a saida
                               marca isso explicitamente.

Codigos de saida: 0 sucesso | 1 erro de execucao | 2 uso incorreto."""

_OPCOES_EN = _OPCOES_PT.replace("Opcoes (valem para os comandos de analise)", "Options (apply to the analysis commands)")


def texto_ajuda(lang: str = "PT") -> str:
    """`guaraci --help`: comandos agrupados pelos mesmos grupos do menu."""
    L = 1 if str(lang).upper() == "EN" else 0
    linhas = ["Uso: guaraci [COMANDO] [OPCOES]" if L == 0 else "Usage: guaraci [COMMAND] [OPTIONS]", ""]
    linhas.append("  (sem argumentos)  abre as telas interativas" if L == 0 else "  (no arguments)    opens the interactive screens")
    linhas.append("  --version         mostra a versao instalada" if L == 0 else "  --version         shows the installed version")
    linhas.append("  --help            mostra esta ajuda" if L == 0 else "  --help            shows this help")
    for g in cat.grupos():
        cmds = cat.comandos(g["id"])
        if not cmds:
            continue
        linhas.append("")
        linhas.append(f"{g['nome'][L]}:")
        for c in cmds:
            uso, faz = c["uso"], c["faz"][L]
            corpo = textwrap.wrap(faz, 60) or [""]
            if len(uso) <= 17:
                linhas.append(f"  {uso:<17} {corpo[0]}")
            else:
                linhas.append(f"  {uso}")
                linhas.append(f"  {'':<17} {corpo[0]}")
            linhas.extend(f"  {'':<17} {extra}" for extra in corpo[1:])
    linhas.append("")
    linhas.append(_OPCOES_PT if L == 0 else _OPCOES_EN)
    return "\n".join(linhas)


def _cmd_exemplo(args: Sequence[str]) -> int:
    from guaraci import exemplos as ex
    rodar = "--rodar" in args
    pos = [a for a in args if not a.startswith("--")]
    if not pos:
        print("Cenarios de exemplo (dados sinteticos; use: guaraci exemplo CENARIO [PASTA] [--rodar]):\n")
        for k, v in ex.EXEMPLOS.items():
            print(f"  {k:<20} {v['nome'][0]}\n  {'':<20} {v['faz'][0]}")
        return 0
    cenario = pos[0]
    if cenario not in ex.EXEMPLOS:
        print(f"Erro: cenario desconhecido {cenario!r}. Disponiveis: {', '.join(ex.EXEMPLOS)}.",
              file=sys.stderr)
        return 2
    pasta = pos[1] if len(pos) > 1 else os.path.join("GUARACI_Exemplo", cenario)
    r = ex.gerar(cenario, pasta)
    print(f"Exemplo '{cenario}' gerado em {os.path.abspath(r['pasta'])} (dados sinteticos).")
    for a in r["arquivos"]:
        print(f"  {a}")
    if cenario == "comparar":
        print(f"\nPara rodar:\n  cd {r['pasta']}\n  guaraci anova tabela.csv --resposta teor --grupo grupo --unidade amostra")
        if rodar:
            return _rodar_anova_exemplo(r["pasta"])
    elif cenario == "vip":
        print(f"\nPara rodar:\n  cd {r['pasta']}\n  {ex.COMANDO_VIP}")
        print(f"\n{ex.NOTA_VIP}")
        if rodar:
            return _rodar_vip_exemplo(r["pasta"])
    elif cenario == "monitorar":
        print(f"\nPara rodar:\n  cd {r['pasta']}\n  {ex.COMANDO_MONITORAR}")
        if rodar:
            return _rodar_monitorar_exemplo(r["pasta"])
    elif cenario == "transferir":
        print(f"\nPara rodar:\n  cd {r['pasta']}\n  {ex.COMANDO_TRANSFERIR}")
        if rodar:
            return _rodar_transferir_exemplo(r["pasta"])
    else:
        print(f"\nPara rodar:\n  guaraci run {os.path.join(r['pasta'], 'config.yaml')}")
        if rodar:
            from guaraci.guaraci import main as _main
            _main(["run", os.path.join(r["pasta"], "config.yaml")])
    return 0


def _rodar_anova_exemplo(pasta: str) -> int:
    from guaraci import anova_cli
    return anova_cli.executar([os.path.join(pasta, "tabela.csv"), "--resposta", "teor", "--grupo", "grupo",
                               "--unidade", "amostra", "--saida", pasta_execucao(pasta, "resultados_anova")])


def _rodar_vip_exemplo(pasta: str) -> int:
    from guaraci import vip_cli
    return vip_cli.executar([os.path.join(pasta, "tabela.csv"), "--classe", "classe", "--grupo", "amostra",
                             "--saida", pasta_execucao(pasta, "resultados_vip")])


def _rodar_monitorar_exemplo(pasta: str) -> int:
    from guaraci import monitorar_cli
    return monitorar_cli.executar(["--referencia", os.path.join(pasta, "referencia.csv"),
                                   "--novos", os.path.join(pasta, "novos.csv"), "--lote", "lote", "--id", "amostra",
                                   "--saida", pasta_execucao(pasta, "resultados_monitoramento")])


def _rodar_transferir_exemplo(pasta: str) -> int:
    return _transferir_com_figura(["--mestre", os.path.join(pasta, "mestre.csv"),
                                   "--escravo", os.path.join(pasta, "escravo.csv"),
                                   "--novos", os.path.join(pasta, "novos_escravo.csv"),
                                   "--saida", pasta_execucao(pasta, "resultados_transferencia")])


def _transferir_com_figura(args: Sequence[str]) -> int:
    """`guaraci transferir` e, se terminou bem, a figura mestre x escravo
    antes/depois (figuras_extras; a figura nunca muda o codigo de saida)."""
    from guaraci import transferir_cli
    codigo = transferir_cli.executar(list(args))
    if codigo == 0:
        try:
            from guaraci.figuras_extras import figuras_transferencia_de_argv
            figuras_transferencia_de_argv(list(args))
        except Exception as e:  # noqa: BLE001 -- figura complementar
            print(f"  [AVISO] figura da transferencia nao gerada: {e}")
    return codigo


# Exemplos rodados por comando proprio (sem config.yaml): a tela [1.1] do menu usa os mesmos.
EXECUTORES_SEM_CONFIG = {"comparar": _rodar_anova_exemplo, "vip": _rodar_vip_exemplo, "monitorar": _rodar_monitorar_exemplo,
                         "transferir": _rodar_transferir_exemplo}


def _legenda_selos() -> str:
    """Uma linha com o significado dos simbolos de selo da lista de `guaraci confianca`."""
    from guaraci.catalogo_validacao import ROTULO_ESTADO, SIMBOLO_ESTADO
    ascii_only = not _unicode()
    partes = [f"{SIMBOLO_ESTADO[e][1 if ascii_only else 0]} {ROTULO_ESTADO[e][0]}" for e in SIMBOLO_ESTADO]
    return "Selos: " + "   ".join(partes)


def _cmd_confianca(args: Sequence[str]) -> int:
    from guaraci import confianca as conf
    if not args:
        print(conf.lista(None, "PT", ascii_only=not _unicode()))
        print("\n" + _legenda_selos())
        print("\nUse: guaraci confianca METODO  (id ou parte do nome)")
        return 0
    achado = conf.buscar_entrada(" ".join(args))
    if not achado:
        print(f"Erro: nada encontrado para {' '.join(args)!r}. `guaraci confianca` aceita um metodo ou opcao do programa "
              "(id ou parte do nome), nao uma tecnica de medicao como FT-NIR. Rode `guaraci confianca` para listar.", file=sys.stderr)
        return 2
    print(conf.pagina(achado[0], achado[1], "PT", ascii_only=not _unicode()))
    return 0


def _mapa_de_ajuda() -> str:
    """`guaraci ajuda` sem argumento: grupos, com telas, comandos e tecnicas de cada um."""
    L = ["Mapa do GUARACI (use: guaraci ajuda TOPICO, guaraci buscar TERMO, guaraci assistente PERGUNTA)", ""]
    for g in cat.grupos():
        telas, cmds = cat.telas(g["id"]), cat.comandos(g["id"])
        tecs = [t for t in cat.tecnicas() if t["grupo"] == g["id"]]
        if not (telas or cmds or tecs):
            continue
        L.append(f"{g['nome'][0]} -- {g['faz'][0]}")
        for t in telas:
            L.append(f"  tela [{t['tecla']}] {t['nome'][0]}")
        for c in cmds:
            L.append(f"  comando guaraci {c['uso'].split()[0]}")
        for t in tecs:
            L.append(f"  tecnica {t['id']}: {t['nome']}")
        L.append("")
    return "\n".join(L).rstrip()


def _cmd_ajuda(args: Sequence[str]) -> int:
    from guaraci import assistente as ass
    if not args:
        print(_mapa_de_ajuda())
        return 0
    termo = " ".join(args)
    e = ass.explicar(termo)
    if e:
        print(ass.formatar_explicacao(e))
        return 0
    achados = ass.buscar(termo, "PT", 5)
    if achados:
        print(f"Nao ha topico exato {termo!r}. Parecidos:")
        for a in achados:
            print(f"  [{a['tipo']}] {a['titulo']} ({a['id']})")
        return 0
    print(f"Erro: nada no catalogo para {termo!r}. Rode `guaraci ajuda` para ver o mapa.", file=sys.stderr)
    return 2


def _cmd_manual(args: Sequence[str]) -> int:
    from guaraci import docs_pacote as dp
    if not args:
        print(dp.lista())
        return 0
    alvo = " ".join(args).strip().lower()
    if alvo in dp.TOPICOS:
        print(dp.ler(alvo))
        return 0
    texto = dp.secao_do_manual(alvo)
    if texto is None:
        print(f"Erro: {alvo!r} nao e' topico nem secao do manual. Rode `guaraci manual` para ver a lista.", file=sys.stderr)
        return 2
    print(texto)
    return 0


def _cmd_buscar(args: Sequence[str]) -> int:
    from guaraci import assistente as ass
    if not args:
        print("Erro: informe o termo. Ex.: guaraci buscar censura", file=sys.stderr)
        return 2
    achados = ass.buscar(" ".join(args), "PT", 10)
    if not achados:
        print("Nada encontrado.")
        return 0
    for a in achados:
        print(f"[{a['tipo']}] {a['titulo']} ({a['id']}): {a['trecho']}")
    return 0


def _cmd_assistente(args: Sequence[str]) -> int:
    from guaraci import assistente as ass
    if not args:
        print("Use: guaraci assistente PERGUNTA   (ou abra o assistente nas telas: tecla G)")
        return 0
    print(ass.responder(" ".join(args), "PT"))
    return 0


def _unicode() -> bool:
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in enc and os.environ.get("GUARACI_ASCII") != "1" and os.environ.get("TERM") != "dumb"


def subcomando(nome: str, args: Sequence[str]) -> int:
    """Executa um dos SUBCOMANDOS e devolve o codigo de saida."""
    if nome == "anova":
        from guaraci import anova_cli
        return anova_cli.executar(list(args))
    if nome == "vip":
        from guaraci import vip_cli
        return vip_cli.executar(list(args))
    if nome == "monitorar":
        from guaraci import monitorar_cli
        return monitorar_cli.executar(list(args))
    if nome == "relatorio":
        from guaraci import relatorio_cli
        return relatorio_cli.executar(list(args))
    if nome == "transferir":
        return _transferir_com_figura(list(args))
    if nome == "exemplo":
        return _cmd_exemplo(list(args))
    if nome == "confianca":
        return _cmd_confianca(list(args))
    if nome == "ajuda":
        return _cmd_ajuda(list(args))
    if nome == "buscar":
        return _cmd_buscar(list(args))
    if nome == "manual":
        return _cmd_manual(list(args))
    if nome == "assistente":
        return _cmd_assistente(list(args))
    if nome == "checar":
        # Pre-voo do assistente pela linha de comando (mesmos argumentos de
        # `python -m guaraci.assistente_checar`): 0 sem achado critico, 1 com, 2 se nao carrega.
        from guaraci import assistente_checar
        return assistente_checar.main(list(args))
    raise ValueError(f"subcomando desconhecido: {nome}")
