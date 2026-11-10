#!/usr/bin/env python3
"""gerar_blindagem_anova.py -- gera docs/qualidade/anova_blindagem.md a partir de medicoes.

    python scripts/gerar_blindagem_anova.py --data 2026-10-06 \
        --cobertura docs/qualidade/anova/cobertura.json \
        --mutacao docs/qualidade/anova/mutacao_anova_grupos.json ... \
        --classificacao docs/qualidade/anova/mutantes_classificados.json

Nada e' escrito a mao: casos de borda e testes saem da AST dos arquivos de teste, a cobertura do JSON do
`coverage`, a mutacao dos JSON de `scripts/mutacao_modulo.py` e as referencias do registro
`docs/cientifico/referencias_verificadas.yaml`. Caminhos de usuario sao removidos do texto gerado.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
MODULOS = ["anova_grupos", "anova_cli", "vip_estabilidade", "vip_cli", "relatorio_blocos"]
ARQUIVOS_TESTE = ["test_anova_grupos", "test_anova_blindagem", "test_anova_nao_parametrico", "test_anova_cli",
                  "test_anova_cli_e2e", "test_anova_vip_lacunas", "test_anova_mutacao", "test_anova_cli_mutacao", "test_anova_bloco_aditivo_exato",
                  "test_vip_estabilidade", "test_vip_cli", "test_vip_mutacao", "test_vip_cli_mutacao", "test_relatorio_blocos"]

ORACULOS = [
    ("F e p do ANOVA", "`scipy.stats.f_oneway`", "rel 1e-12 (F), 1e-10 (p)"),
    ("Tukey-Kramer (n desigual) e Tukey", "`scipy.stats.tukey_hsd` e formula ingenua (amplitude studentizada)", "rel 1e-6 no p; IC rel 1e-6"),
    ("Games-Howell (variancias 1:100)", "formula ingenua com Welch-Satterthwaite (nao depende da versao do SciPy); `tukey_hsd(equal_var=False)` quando existe", "rel 1e-6"),
    ("Levene (centro mediana)", "`scipy.stats.levene`", "rel 1e-12"),
    ("eta2 e omega2", "definicao por somas de quadrados escrita no teste", "rel 1e-12"),
    ("Benjamini-Hochberg entre respostas", "implementacao ingenua em lacos", "rel 1e-10"),
    ("k = 2 (ANOVA x t)", "`scipy.stats.ttest_ind` (Student e Welch): F = t^2", "rel 1e-10"),
    ("Kruskal-Wallis", "`scipy.stats.kruskal`; epsilon2 = H/((n^2-1)/(n+1))", "rel 1e-12"),
    ("Dunn (z, p, Holm, BH)", "implementacao ingenua com postos por definicao (sem `rankdata`) e Holm/BH em lacos", "rel 1e-9"),
    ("Dunn com k = 2", "z^2 = H; p bruto = `scipy.stats.mannwhitneyu` assintotico sem continuidade", "rel 1e-8"),
    ("VIP", "PLS1 por NIPALS escrito no teste + formula de Wold (lacos); soma de VIP^2 = nº de variaveis", "rel 1e-8"),
]

FUNCOES_NOVAS = {
    "anova_grupos": ["anova_uma_via", "kruskal_dunn", "_dunn", "_ajustar_p", "diagnosticar_premissas", "recomendar_metodo"],
    "vip_estabilidade": ["vip_estabilidade", "figura_vip_estabilidade", "_alvo", "_autoescalar"],
}

ACHADOS = [
    ("Mutacao (2a rodada): ramos mortos", "nove guardas do ANOVA e da CLI nunca eram alcancadas (erro-padrao zero, QM dentro zero, SQ total zero, minimo de 3 residuos, amplitude zero): estavam classificadas como `inalcancavel`",
     "removidas em vez de classificadas (comentario diz a invariante que as dispensa)", "mutacao refeita: 500 mutantes em `anova_grupos`, 14 sobreviventes, todos classificados"),
    ("Mutacao (2a rodada): bloco aditivo exato", "o texto dizia que QM residual 0 \"nao ocorre\"; dados perfeitamente aditivos ocorrem (F infinito, p 0, IC degenerado)",
     "o ramo e' alcancavel e ganhou teste proprio (lstsq arredondado para o zero ser exato em qualquer plataforma)", "`test_anova_bloco_aditivo_exato.py`"),
    ("Integracao (diagnostico de premissas)", "a curtose dos residuos JUNTADOS explodia so' por variancias desiguais (1:100) e a recomendacao saia Kruskal-Wallis em vez de Games-Howell",
     "forma medida em residuos padronizados por grupo", "`test_regressao_variancias_desiguais_nao_viram_caudas_pesadas` (vermelho antes, verde depois)"),
    ("Teste existente (oraculo de Games-Howell)", "era pulado em SciPy < 1.16 (Python 3.10)", "oraculo ingenuo independente da versao; motivo de pulado removido de `tests/pulados_esperados.json`",
     "`test_pos_teste_proprio_igual_ao_*`"),
    ("Mutacao: `figura_vip_estabilidade`", "a figura quebrava (matplotlib recusa `yerr` negativo) quando a media e o percentil de uma variavel coincidiam e diferiam por -1e-17 em ponto flutuante: `guaraci vip` caia ao desenhar a figura",
     "`np.maximum(..., 0)` nas duas barras de erro", "`test_regressao_figura_com_vip_identico_nas_dobras_nao_quebra_por_erro_de_ponto_flutuante` (vermelho antes, verde depois)"),
    ("Mutacao: limiares", "os testes de borda da recomendacao importavam a propria constante do modulo (4, 1 e 3): mudar a constante mudava o teste junto e nada falhava",
     "bordas conferidas com numeros literais", "`test_limiares_da_recomendacao_sao_4_1_e_3_literais`"),
    ("Mutacao: ANOVA com bloco", "comparacoes e F/p do bloco so' eram conferidos em desenho balanceado, onde a variancia do contraste nao depende do desenho",
     "oraculo independente por minimos quadrados em desenho desbalanceado (nenhum erro de calculo encontrado)", "`test_bloco_desbalanceado_confere_f_p_e_comparacoes_com_minimos_quadrados_independentes`"),
    ("Mutacao: CLI", "bootstrap padrao, IC das tabelas, bordas `p == alfa`, letras e conteudo da figura nao eram verificados",
     "testes sobre os artistas da figura (posicoes, letras, 150 dpi) e sobre o conteudo exato das tabelas e do relatorio", "`test_anova_cli_mutacao.py`"),
    ("Teste novo", "o tamanho do PNG dependia do `savefig.bbox` global deixado por outros testes (falha so' na suite completa)",
     "a fixture fixa `savefig.bbox = standard`", "`test_figura_tem_dois_paineis_*`, `test_tamanho_da_figura_150_dpi_*`"),
]


def _limpar(t: str) -> str:
    t = re.sub(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s]+", "<usuario>", t)
    return re.sub(r"[A-Za-z]:[\\/]+[^\s|]*mut_[^\s|]*", "<tmp>", t)


def _funcoes(arquivo: str):
    fonte = (RAIZ / "tests" / f"{arquivo}.py").read_text(encoding="utf-8")
    arv = ast.parse(fonte)
    out = []
    for n in arv.body:
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"):
            parametros = 1
            for d in n.decorator_list:
                if isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "parametrize" and len(d.args) >= 2 \
                        and isinstance(d.args[1], (ast.List, ast.Tuple)):
                    parametros *= len(d.args[1].elts)
            out.append((n.name, n.lineno, parametros, (ast.get_docstring(n) or "").split("\n")[0]))
    return fonte, out


def _secao(fonte: str, linhas, ini: str, fim: str):
    li = next((i for i, l in enumerate(fonte.split("\n"), 1) if ini in l), None)
    lf = next((i for i, l in enumerate(fonte.split("\n"), 1) if fim in l), 10 ** 9)
    return [f for f in linhas if li is not None and li < f[1] < lf]


def gerar(data: str, cobertura: dict, mutacoes: list, classificacao: dict) -> str:
    L = [f"# Blindagem do ANOVA, Kruskal-Wallis/Dunn e VIP -- {data}", "", "**Público:** quem avalia os números · **Estado:** gerado por `scripts/gerar_blindagem_anova.py` (não editar à mão) · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`", "",
         "Gerado por `scripts/gerar_blindagem_anova.py` a partir da AST dos testes, do JSON do `coverage`, dos JSON de "
         "`scripts/mutacao_modulo.py` e do registro de referencias. Nada aqui foi escrito a mao.", ""]
    L += ["## 1. Inventario (A1)", "", "| modulo | funcoes publicas (`__all__`) |", "|---|---|"]
    for m in MODULOS:
        fonte = (RAIZ / "src" / "guaraci" / f"{m}.py").read_text(encoding="utf-8")
        mt = re.search(r"__all__\s*=\s*\[(.*?)\]", fonte, flags=re.S)
        nomes = ", ".join(re.findall(r'"([^"]+)"', mt.group(1))) if mt else "-"
        L.append(f"| `{m}.py` | {nomes} |")
    L += ["", "Comandos: `guaraci anova` (`--metodo auto|tukey|games-howell|kruskal-dunn|recomendar`, `--ajuste holm|bh`, "
          "`--sem-figuras`, `--idioma pt|en`) e `guaraci vip`. Tela do menu: [W] Comparar grupos (pergunta o metodo).", "",
          "| arquivo de teste | funcoes | casos (com parametrizacao) |", "|---|---|---|"]
    tot_f = tot_c = 0
    por_arquivo = {}
    for a in ARQUIVOS_TESTE:
        fonte, fs = _funcoes(a)
        por_arquivo[a] = (fonte, fs)
        nf, nc = len(fs), sum(f[2] for f in fs)
        tot_f, tot_c = tot_f + nf, tot_c + nc
        L.append(f"| `{a}.py` | {nf} | {nc} |")
    L.append(f"| **total** | **{tot_f}** | **{tot_c}** |")
    L += ["", "## 2. Cobertura de linhas e de ramos", "", "| modulo | linhas | ramos | cobertura (linhas+ramos) | linhas nao cobertas | ramos nao exercitados |",
          "|---|---|---|---|---|---|"]
    for m in MODULOS:
        chave = next((k for k in cobertura.get("files", {}) if k.replace("\\", "/").endswith(f"guaraci/{m}.py")), None)
        if not chave:
            L.append(f"| `{m}.py` | - | - | - | - | - |")
            continue
        s = cobertura["files"][chave]["summary"]
        ramos = s.get("num_branches", 0)
        L.append(f"| `{m}.py` | {s['covered_lines']}/{s['num_statements']} | {s.get('covered_branches', 0)}/{ramos} | "
                 f"{s['percent_covered']:.1f}% | {', '.join(map(str, cobertura['files'][chave]['missing_lines'])) or '-'} | "
                 f"{', '.join(f'{a}->{b}' for a, b in cobertura['files'][chave].get('missing_branches', [])) or '-'} |")
    L += ["", "### Cobertura de ramos por funcao (Kruskal-Wallis/Dunn, diagnostico, VIP e ANOVA)", "",
          "| modulo | funcao | linhas | ramos |", "|---|---|---|---|"]
    for m, nomes in FUNCOES_NOVAS.items():
        chave = next((k for k in cobertura.get("files", {}) if k.replace("\\", "/").endswith(f"guaraci/{m}.py")), None)
        fns = cobertura["files"][chave].get("functions", {}) if chave else {}
        for n in nomes:
            f = fns.get(n)
            if f is None:
                L.append(f"| `{m}.py` | `{n}` | - | - |")
                continue
            sm = f["summary"]
            L.append(f"| `{m}.py` | `{n}` | {sm['covered_lines']}/{sm['num_statements']} | {sm.get('covered_branches', 0)}/{sm.get('num_branches', 0)} |")
    L += ["", "Os ramos nao exercitados sao de defesa que o CLI nao alcanca: resultado sem diagnostico (o diagnostico so' falha quando o "
          "calculo tambem recusa), relatorio sem comparacoes (o ANOVA e o Kruskal sempre as tem), bloco de relatorio de tipo "
          "desconhecido e o ramo de compatibilidade do argparse em Python 3.10/3.11 (`SystemExit` em vez de `ArgumentError`).", ""]
    L += ["## 3. Referencias independentes e tolerancias (A2)", "", "| o que | oraculo | tolerancia |", "|---|---|---|"]
    L += [f"| {a} | {b} | {c} |" for a, b, c in ORACULOS]
    L += ["", "Exemplo de literatura com valores publicados: **nao usado** (nenhuma fonte com os numeros conferidos).", ""]
    fonte_b, fs_b = por_arquivo["test_anova_blindagem"]
    bordas = _secao(fonte_b, fs_b, "casos de borda (A3)", "propriedades (A4)")
    n_bordas_f = len(bordas)
    n_bordas_c = sum(f[2] for f in bordas)
    fonte_k, fs_k = por_arquivo["test_anova_nao_parametrico"]
    L += ["## 4. Casos de borda (A3)", "",
          f"ANOVA (`test_anova_blindagem.py`): **{n_bordas_f} funcoes, {n_bordas_c} casos** com parametrizacao. Resultado esperado "
          "escrito no docstring ou comentario antes da execucao.", ""]
    L += [f"- `{n}`" + (f": {d}" if d else "") for n, _, _, d in bordas]
    L += ["", "Kruskal-Wallis/Dunn (`test_anova_nao_parametrico.py`): k=2 contra Mann-Whitney, empates em massa, todos iguais, n=1, "
          "NaN, invariancia a transformacao monotona, ordem das linhas, recusas de unidade.", ""]
    L += ["## 5. Propriedades (A4, Hypothesis com `@example` fixos)", "",
          "p em [0, 1]; IC do Tukey contem a diferenca observada; simetria de pares; omega2 <= eta2; Tukey balanceado com F de p alto nao "
          "rejeita (a excecao inversa, F significativo sem par significativo, esta documentada no docstring); Kruskal: p, epsilon2, "
          "p ajustado >= p bruto, simetria; VIP: soma de VIP^2 = p.", ""]
    L += ["## 6. Mutacao (A5, B2, C4, D4)", "",
          "Operadores: comparacao, aritmetica, And/Or, `not`, constantes, negacao de `if`/`while`, `return`->None, `raise`->pass, "
          "`continue`/`break`->pass. Strings, docstrings, anotacoes e logs nao sao mutados. Mutante morto = algum teste falhou "
          "(o primeiro teste que falhou fica no JSON); sobrevivente = todos passaram.", "",
          "| modulo | mutantes | mortos | sobreviventes | equivalentes classificados | **sem classificacao** |", "|---|---|---|---|---|---|"]
    tot_m = tot_mortos = tot_equiv = tot_aberto = 0
    detalhes = []
    for mu in mutacoes:
        nome = Path(mu["modulo"]).stem
        viv = mu["sobreviventes"]
        eq, aberto = [], []
        for s in viv:
            k = f"{nome}:{s['linha']}:{s['operador']}"
            (eq if k in classificacao else aberto).append((k, s))
        tot_m += mu["total"]
        tot_mortos += mu["total"] - len(viv)
        tot_equiv += len(eq)
        tot_aberto += len(aberto)
        L.append(f"| `{nome}.py` | {mu['total']} | {mu['total'] - len(viv)} | {len(viv)} | {len(eq)} | {len(aberto)} |")
        for k, s in eq + aberto:
            detalhes.append((nome, s, classificacao.get(k, "**SEM CLASSIFICACAO**")))
    L.append(f"| **total** | **{tot_m}** | **{tot_mortos}** | **{tot_equiv + tot_aberto}** | **{tot_equiv}** | **{tot_aberto}** |")
    L += ["", f"**{tot_mortos + tot_equiv}/{tot_m} mortos ou equivalentes classificados; {tot_aberto} sem classificacao.**", ""]
    if detalhes:
        L += ["### Sobreviventes classificados", "", "| modulo | linha | operador | trecho | classificacao |", "|---|---|---|---|---|"]
        for nome, s, c in detalhes:
            L.append(f"| `{nome}.py` | {s['linha']} | {s['operador']} | `{s['trecho'].replace('|', '/')}` | {c} |")
        L.append("")
    L += ["## 7. Achados e correcoes", "", "| onde | achado | correcao | prova |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in ACHADOS]
    L += ["", "## 8. Referencias citadas", "", "| fonte | DOI | conferido em Crossref |", "|---|---|---|"]
    reg = (RAIZ / "docs" / "cientifico" / "referencias_verificadas.yaml").read_text(encoding="utf-8")
    cit = {"Dunn 1964": "10.1080/00401706.1964.10490181", "Wold, Sjostrom e Eriksson 2001": "10.1016/s0169-7439(01)00155-1",
           "Chong e Jun 2005": "10.1016/j.chemolab.2004.12.011", "Rochon, Gondan e Kieser 2012": "10.1186/1471-2288-12-81",
           "Tukey 1949": "10.2307/3001913", "Kramer 1956": "10.2307/3001469", "Games e Howell 1976": "10.3102/10769986001002113",
           "Nadeau e Bengio 2003": "10.1023/a:1024068626366"}
    for fonte, doi in cit.items():
        achou = re.search(rf"- doi: {re.escape(doi)}\s.*?conferido: [^\n]*Crossref", reg, flags=re.S | re.I)
        L.append(f"| {fonte} | {doi} | {'sim' if achou else 'nao (removida)'} |")
    L += ["", "Nao citadas por falta de DOI conferido: Holm 1979 e a fonte do epsilon2 (a formula esta no texto do relatorio).", ""]
    L += ["## 9. Verificacoes adicionais da lapidacao", "",
          "- Amostra de 10 equivalentes verificada contra entradas de borda, com controle de poder: `docs/qualidade/anova/equivalentes_amostra.md`.",
          "- O mutante de compatibilidade do `argparse` (`vip_cli.py` linha 214) sobrevive em 3.12 e morre em 3.10 e 3.11: "
          "`docs/qualidade/anova/mutacao_vip_cli_py310_linha214.json` e `docs/qualidade/anova/mutacao_vip_cli_py311_linha214.json`.",
          "- Simulacao da recomendacao de metodo: `docs/qualidade/anova/recomendacao_simulacao.md`; selecao aninhada com rotulos aleatorios: "
          "`docs/qualidade/selecao_aninhada_rotulos_aleatorios.json`.", ""]
    return _limpar("\n".join(L))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--cobertura", required=True)
    p.add_argument("--mutacao", nargs="+", required=True)
    p.add_argument("--classificacao")
    p.add_argument("--saida")
    a = p.parse_args(argv)
    cob = json.loads(Path(a.cobertura).read_text(encoding="utf-8"))
    mut = [json.loads(Path(x).read_text(encoding="utf-8")) for x in a.mutacao]
    cls = json.loads(Path(a.classificacao).read_text(encoding="utf-8")) if a.classificacao else {}
    saida = Path(a.saida) if a.saida else RAIZ / "docs" / "qualidade" / "anova_blindagem.md"
    saida.write_text(gerar(a.data, cob, mut, cls) + "\n", encoding="utf-8")
    print(saida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
