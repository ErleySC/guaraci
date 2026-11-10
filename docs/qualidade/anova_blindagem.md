# Blindagem do ANOVA, Kruskal-Wallis/Dunn e VIP -- 2026-10-09

**Público:** quem avalia os números · **Estado:** gerado por `scripts/gerar_blindagem_anova.py` (não editar à mão) · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

Gerado por `scripts/gerar_blindagem_anova.py` a partir da AST dos testes, do JSON do `coverage`, dos JSON de `scripts/mutacao_modulo.py` e do registro de referencias. Nada aqui foi escrito a mao.

## 1. Inventario (A1)

| modulo | funcoes publicas (`__all__`) |
|---|---|
| `anova_grupos.py` | ResultadoANOVA, ResultadoKruskal, agregar_por_unidade, anova_uma_via, anova_varias_respostas, diagnosticar_premissas, kruskal_dunn, recomendar_metodo, N_MINIMO_PODER, RAZAO_VARIANCIA_LIMITE, ASSIMETRIA_LIMITE, CURTOSE_LIMITE, TEXTO_KRUSKAL, TEXTO_PREMISSAS |
| `anova_cli.py` | executar, REGRA_UNIDADE, METODOS, letras_de_grupos |
| `vip_estabilidade.py` | vip_estabilidade, figura_vip_estabilidade, TEXTO_VIP, LIMIAR_PADRAO, FREQ_ESTAVEL, FREQ_INSTAVEL |
| `vip_cli.py` | executar |
| `relatorio_blocos.py` | renderizar_md, renderizar_html, renderizar_tex |

Comandos: `guaraci anova` (`--metodo auto|tukey|games-howell|kruskal-dunn|recomendar`, `--ajuste holm|bh`, `--sem-figuras`, `--idioma pt|en`) e `guaraci vip`. Tela do menu: [W] Comparar grupos (pergunta o metodo).

| arquivo de teste | funcoes | casos (com parametrizacao) |
|---|---|---|
| `test_anova_grupos.py` | 44 | 48 |
| `test_anova_blindagem.py` | 24 | 30 |
| `test_anova_nao_parametrico.py` | 21 | 27 |
| `test_anova_cli.py` | 8 | 11 |
| `test_anova_cli_e2e.py` | 22 | 30 |
| `test_anova_vip_lacunas.py` | 17 | 17 |
| `test_anova_mutacao.py` | 18 | 19 |
| `test_anova_cli_mutacao.py` | 21 | 24 |
| `test_anova_bloco_aditivo_exato.py` | 2 | 2 |
| `test_vip_estabilidade.py` | 13 | 19 |
| `test_vip_cli.py` | 7 | 10 |
| `test_vip_mutacao.py` | 16 | 21 |
| `test_vip_cli_mutacao.py` | 12 | 15 |
| `test_relatorio_blocos.py` | 6 | 6 |
| **total** | **231** | **279** |

## 2. Cobertura de linhas e de ramos

| modulo | linhas | ramos | cobertura (linhas+ramos) | linhas nao cobertas | ramos nao exercitados |
|---|---|---|---|---|---|
| `anova_grupos.py` | 392/392 | 110/110 | 100.0% | - | - |
| `anova_cli.py` | 368/368 | 127/132 | 99.0% | - | 407->410, 509->511, 514->516, 516->524, 524->527 |
| `vip_estabilidade.py` | 115/115 | 44/44 | 100.0% | - | - |
| `vip_cli.py` | 176/178 | 58/58 | 99.2% | 213, 214 | - |
| `relatorio_blocos.py` | 55/55 | 28/30 | 97.6% | - | 25->18, 43->36 |

### Cobertura de ramos por funcao (Kruskal-Wallis/Dunn, diagnostico, VIP e ANOVA)

| modulo | funcao | linhas | ramos |
|---|---|---|---|
| `anova_grupos.py` | `anova_uma_via` | 75/75 | 38/38 |
| `anova_grupos.py` | `kruskal_dunn` | 39/39 | 20/20 |
| `anova_grupos.py` | `_dunn` | 13/13 | 0/0 |
| `anova_grupos.py` | `_ajustar_p` | 10/10 | 4/4 |
| `anova_grupos.py` | `diagnosticar_premissas` | 26/26 | 6/6 |
| `anova_grupos.py` | `recomendar_metodo` | 7/7 | 4/4 |
| `vip_estabilidade.py` | `vip_estabilidade` | 54/54 | 32/32 |
| `vip_estabilidade.py` | `figura_vip_estabilidade` | 19/19 | 2/2 |
| `vip_estabilidade.py` | `_alvo` | 15/15 | 8/8 |
| `vip_estabilidade.py` | `_autoescalar` | 6/6 | 2/2 |

Os ramos nao exercitados sao de defesa que o CLI nao alcanca: resultado sem diagnostico (o diagnostico so' falha quando o calculo tambem recusa), relatorio sem comparacoes (o ANOVA e o Kruskal sempre as tem), bloco de relatorio de tipo desconhecido e o ramo de compatibilidade do argparse em Python 3.10/3.11 (`SystemExit` em vez de `ArgumentError`).

## 3. Referencias independentes e tolerancias (A2)

| o que | oraculo | tolerancia |
|---|---|---|
| F e p do ANOVA | `scipy.stats.f_oneway` | rel 1e-12 (F), 1e-10 (p) |
| Tukey-Kramer (n desigual) e Tukey | `scipy.stats.tukey_hsd` e formula ingenua (amplitude studentizada) | rel 1e-6 no p; IC rel 1e-6 |
| Games-Howell (variancias 1:100) | formula ingenua com Welch-Satterthwaite (nao depende da versao do SciPy); `tukey_hsd(equal_var=False)` quando existe | rel 1e-6 |
| Levene (centro mediana) | `scipy.stats.levene` | rel 1e-12 |
| eta2 e omega2 | definicao por somas de quadrados escrita no teste | rel 1e-12 |
| Benjamini-Hochberg entre respostas | implementacao ingenua em lacos | rel 1e-10 |
| k = 2 (ANOVA x t) | `scipy.stats.ttest_ind` (Student e Welch): F = t^2 | rel 1e-10 |
| Kruskal-Wallis | `scipy.stats.kruskal`; epsilon2 = H/((n^2-1)/(n+1)) | rel 1e-12 |
| Dunn (z, p, Holm, BH) | implementacao ingenua com postos por definicao (sem `rankdata`) e Holm/BH em lacos | rel 1e-9 |
| Dunn com k = 2 | z^2 = H; p bruto = `scipy.stats.mannwhitneyu` assintotico sem continuidade | rel 1e-8 |
| VIP | PLS1 por NIPALS escrito no teste + formula de Wold (lacos); soma de VIP^2 = nº de variaveis | rel 1e-8 |

Exemplo de literatura com valores publicados: **nao usado** (nenhuma fonte com os numeros conferidos).

## 4. Casos de borda (A3)

ANOVA (`test_anova_blindagem.py`): **16 funcoes, 22 casos** com parametrizacao. Resultado esperado escrito no docstring ou comentario antes da execucao.

- `test_k2_classico_e_o_t_de_student_ao_quadrado`: k=2 com variancias compativeis: F == t^2 (Student) e p identico.
- `test_k2_variancias_desiguais_e_welch_e_diz_isso`
- `test_grupo_com_n1_e_excluido_com_aviso_e_o_resto_segue`
- `test_todos_os_grupos_com_n1_e_erro`
- `test_todos_os_valores_iguais_e_erro`
- `test_entrada_vazia_e_erro_e_nao_excecao_obscura`
- `test_grupo_com_variancia_zero_cai_na_classica_com_aviso`
- `test_nan_e_inf_sao_descartados_com_aviso_e_nao_contaminam`
- `test_censura_icp_nas_duas_politicas_e_celula_vazia_nao_e_censura`
- `test_rotulos_numericos_texto_espacos_e_acentos_dao_o_mesmo_resultado`
- `test_ordem_das_linhas_e_dos_grupos_nao_altera_o_resultado`
- `test_invariancia_a_deslocamento_e_escala`
- `test_determinismo_inclusive_do_bootstrap_com_a_mesma_semente`
- `test_k_grande_funciona_e_o_cache_do_valor_critico_nao_muda_o_resultado`: k=50 (1225 pares) leva ~4 min por causa da amplitude studentizada do SciPy:
- `test_pseudo_replicacao_avisa_quando_cada_linha_vira_unidade`: Unidade = linha (replicas tratadas como independentes): aviso aparece e
- `test_pseudo_replicacao_animal_x_fazenda_agregada_por_unidade`: Cada fazenda tem varios animais medidos: com a unidade = fazenda, as 4

Kruskal-Wallis/Dunn (`test_anova_nao_parametrico.py`): k=2 contra Mann-Whitney, empates em massa, todos iguais, n=1, NaN, invariancia a transformacao monotona, ordem das linhas, recusas de unidade.

## 5. Propriedades (A4, Hypothesis com `@example` fixos)

p em [0, 1]; IC do Tukey contem a diferenca observada; simetria de pares; omega2 <= eta2; Tukey balanceado com F de p alto nao rejeita (a excecao inversa, F significativo sem par significativo, esta documentada no docstring); Kruskal: p, epsilon2, p ajustado >= p bruto, simetria; VIP: soma de VIP^2 = p.

## 6. Mutacao (A5, B2, C4, D4)

Operadores: comparacao, aritmetica, And/Or, `not`, constantes, negacao de `if`/`while`, `return`->None, `raise`->pass, `continue`/`break`->pass. Strings, docstrings, anotacoes e logs nao sao mutados. Mutante morto = algum teste falhou (o primeiro teste que falhou fica no JSON); sobrevivente = todos passaram.

| modulo | mutantes | mortos | sobreviventes | equivalentes classificados | **sem classificacao** |
|---|---|---|---|---|---|
| `anova_grupos.py` | 500 | 486 | 14 | 14 | 0 |
| `anova_cli.py` | 302 | 282 | 20 | 20 | 0 |
| `vip_estabilidade.py` | 177 | 163 | 14 | 14 | 0 |
| `vip_cli.py` | 147 | 145 | 2 | 2 | 0 |
| `relatorio_blocos.py` | 78 | 78 | 0 | 0 | 0 |
| **total** | **1204** | **1154** | **50** | **50** | **0** |

**1204/1204 mortos ou equivalentes classificados; 0 sem classificacao.**

### Sobreviventes classificados

| modulo | linha | operador | trecho | classificacao |
|---|---|---|---|---|
| `anova_grupos.py` | 164 | const 256 | `@lru_cache(maxsize=256)` | sem efeito observavel: tamanho do cache LRU do quantil critico: so' muda memoria |
| `anova_grupos.py` | 198 | const 10 | `crit = _q_critico(k, round(float(gl[idx]), 10), 1 - alpha)` | sem efeito observavel: casas decimais da chave do cache do quantil critico (10 -> 11 muda gl em menos de 1e-10; o quantil muda abaixo da tolerancia de qualquer teste) |
| `anova_grupos.py` | 201 | const 1.0 | `"p_ajustado": float(min(1.0, p[idx]))})` | equivalente matematico: min(1,p) com p <= 1 por construcao (a funcao de sobrevivencia nunca passa de 1) |
| `anova_grupos.py` | 275 | Eq->NotEq | `cols += [(b == bl).astype(float) for bl in blocos[1:]]` | equivalente matematico: o complemento do indicador do bloco gera o mesmo espaco de colunas com o intercepto: mesmas SQ, F, p e contrastes |
| `anova_grupos.py` | 291 | Gt->GtE | `F = (sq_f / gl_f) / qm_res if qm_res > 0 else float("inf")` | equivalente matematico: com QM residual exatamente 0 (dados perfeitamente aditivos, alcancavel: tests/test_anova_bloco_aditivo_exato.py) o quociente (SQ/gl)/0.0 em numpy e' inf, o mesmo valor do ramo else |
| `anova_grupos.py` | 293 | Gt->GtE | `Fb = (sq_b / gl_b) / qm_res if (qm_res > 0 and gl_b > 0) else float("nan")` | equivalente matematico: com gl do bloco == 0 o quociente 0/0 e' NaN, o mesmo valor do ramo else (e com QM residual 0 o ramo QM e' testado em tests/test_anova_bloco_aditivo_exato.py) |
| `anova_grupos.py` | 312 | Gt->GtE | `q = abs(d) / se * np.sqrt(2) if se > 0 else float("inf")` | equivalente matematico: com erro-padrao do contraste 0 o quociente |d|/0 e' inf (ou NaN se d = 0); nos dois casos o p sai 0 pela guarda de finitude logo abaixo, igual ao ramo else |
| `anova_grupos.py` | 317 | const 1.0 | `"p_ajustado": min(1.0, p_ij)})` | equivalente matematico: min(1,p) com p <= 1 por construcao |
| `anova_grupos.py` | 356 | Gt->GtE | `if len(y_u) == n_linhas - n_nan and n_linhas - n_nan > 0:` | equivalente matematico: segunda condicao (linhas validas > 0): com 0 unidades validas o calculo termina em erro antes de o aviso ser usado |
| `anova_grupos.py` | 356 | Sub->Add | `if len(y_u) == n_linhas - n_nan and n_linhas - n_nan > 0:` | equivalente matematico: segunda condicao (linhas validas > 0): com 0 unidades validas o calculo termina em erro antes de o aviso ser usado |
| `anova_grupos.py` | 356 | const 0 | `if len(y_u) == n_linhas - n_nan and n_linhas - n_nan > 0:` | equivalente matematico: segunda condicao (linhas validas > 0): com 0 unidades validas o calculo termina em erro antes de o aviso ser usado |
| `anova_grupos.py` | 385 | Lt->LtE | `variancias_iguais = {"auto": not (levene_p < alpha), "iguais": True, "desiguais": False}[variancias]` | igualdade exata de ponto flutuante (probabilidade zero): levene_p == alfa so' ocorre com um p calculado igual ao alfa digitado, bit a bit |
| `anova_grupos.py` | 399 | Lt->LtE | `if shapiro_p < alpha:` | igualdade exata de ponto flutuante (probabilidade zero): shapiro_p == alfa so' ocorre com um p calculado igual ao alfa digitado, bit a bit |
| `anova_grupos.py` | 651 | const 1 | `or [np.zeros(1)])` | equivalente matematico: np.zeros(1) x np.zeros(2): so' o caso de todos os grupos constantes, em que a forma e' constante nos dois |
| `anova_cli.py` | 281 | And<->Or | `if i in col and j in col:` | equivalente matematico: equivalente verificado por enumeracao: mesmas letras em 4660 casos (todas as matrizes de significancia, k = 2 a 5) e em 40 sorteios com k ate 7 |
| `anova_cli.py` | 287 | Gt->GtE | `if c and not any(c < o or (c == o and k > idx) for idx, o in enumerate(novas) if idx != k)]` | equivalente matematico: equivalente verificado por enumeracao: mesmas letras em 4660 casos (todas as matrizes de significancia, k = 2 a 5) e em 40 sorteios com k ate 7 |
| `anova_cli.py` | 287 | Gt->Lt | `if c and not any(c < o or (c == o and k > idx) for idx, o in enumerate(novas) if idx != k)]` | equivalente matematico: equivalente verificado por enumeracao: mesmas letras em 4660 casos (todas as matrizes de significancia, k = 2 a 5) e em 40 sorteios com k ate 7 |
| `anova_cli.py` | 287 | Lt->LtE | `if c and not any(c < o or (c == o and k > idx) for idx, o in enumerate(novas) if idx != k)]` | equivalente matematico: equivalente verificado por enumeracao: mesmas letras em 4660 casos (todas as matrizes de significancia, k = 2 a 5) e em 40 sorteios com k ate 7 |
| `anova_cli.py` | 461 | const False | `ax.boxplot(dados, showfliers=False)` | estetico (nao muda dado, texto nem estrutura): pontos extremos do boxplot (ja desenhados como pontos por unidade) |
| `anova_cli.py` | 462 | const 0 | `rng = np.random.default_rng(0)` | estetico (nao muda dado, texto nem estrutura): semente do deslocamento horizontal dos pontos |
| `anova_cli.py` | 464 | Add->Sub | `ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color="#4C72B0", alpha=0.7, zorder=3)` | estetico (nao muda dado, texto nem estrutura): tamanho, transparencia, camada e sinal do deslocamento horizontal (distribuicao simetrica) |
| `anova_cli.py` | 464 | const 14 | `ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color="#4C72B0", alpha=0.7, zorder=3)` | estetico (nao muda dado, texto nem estrutura): tamanho, transparencia, camada e sinal do deslocamento horizontal (distribuicao simetrica) |
| `anova_cli.py` | 464 | const 3 | `ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color="#4C72B0", alpha=0.7, zorder=3)` | estetico (nao muda dado, texto nem estrutura): tamanho, transparencia, camada e sinal do deslocamento horizontal (distribuicao simetrica) |
| `anova_cli.py` | 466 | const 30 | `ax.set_xticklabels(r.grupos, rotation=30, ha="right", fontsize=8)` | estetico (nao muda dado, texto nem estrutura): rotacao e fonte dos rotulos |
| `anova_cli.py` | 466 | const 8 | `ax.set_xticklabels(r.grupos, rotation=30, ha="right", fontsize=8)` | estetico (nao muda dado, texto nem estrutura): rotacao e fonte dos rotulos |
| `anova_cli.py` | 474 | const 9 | `ax.text(i, float(np.max(dados[i - 1])) + folga, letras[n], ha="center", va="bottom", fontsize=9)` | estetico (nao muda dado, texto nem estrutura): tamanho da fonte |
| `anova_cli.py` | 475 | const 9 | `ax.set_title(f"{nome}  ({_metodo_texto(r)})", fontsize=9)` | estetico (nao muda dado, texto nem estrutura): tamanho da fonte |
| `anova_cli.py` | 476 | const 8 | `ax.set_ylabel(_m("fig_caixas"), fontsize=8)` | estetico (nao muda dado, texto nem estrutura): tamanho da fonte |
| `anova_cli.py` | 479 | const 12 | `bx.scatter(d["qq_teorico"], d["qq_amostral"], s=12, color="#4C72B0")` | estetico (nao muda dado, texto nem estrutura): tamanho dos pontos do grafico quantil-quantil |
| `anova_cli.py` | 481 | const 1 | `bx.plot(lim, lim, color="#999999", lw=1)` | estetico (nao muda dado, texto nem estrutura): espessura da linha de referencia |
| `anova_cli.py` | 482 | const 9 | `bx.set_title(_m("fig_qq"), fontsize=9)` | estetico (nao muda dado, texto nem estrutura): tamanho da fonte |
| `anova_cli.py` | 483 | const 8 | `bx.set_xlabel(_m("fig_teo"), fontsize=8)` | estetico (nao muda dado, texto nem estrutura): tamanho da fonte |
| `anova_cli.py` | 484 | const 8 | `bx.set_ylabel(_m("fig_res"), fontsize=8)` | estetico (nao muda dado, texto nem estrutura): tamanho da fonte |
| `anova_cli.py` | 603 | const 1 | `return "en" if a.split("=", 1)[1].lower().startswith("en") else "pt"` | equivalente matematico: `split('=', n)[1]` devolve o texto ate o PROXIMO '=' quando n >= 2 e o resto quando n = 1; so' o inicio ('en') e' testado e 'en' nao contem '=', logo o resultado e' o mesmo para qualquer valor |
| `vip_estabilidade.py` | 76 | const 1 | `return yv.reshape(-1, 1), np.searchsorted(cortes, yv, side="right")` | equivalente matematico: reshape(-2, 1): o numpy aceita qualquer dimensao negativa como 'inferir' |
| `vip_estabilidade.py` | 82 | Sub->Add | `return Xtr - media` | equivalente matematico: o PLSRegression centra X internamente: X + media e X - media dao o mesmo modelo |
| `vip_estabilidade.py` | 83 | const 1 | `dp = Xtr.std(axis=0, ddof=1)` | equivalente matematico: ddof 1 -> 2 multiplica todas as colunas pelo mesmo fator: o VIP nao muda |
| `vip_estabilidade.py` | 84 | const 1.0 | `dp[dp == 0] = 1.0` | equivalente matematico: coluna de variancia zero fica toda zero depois de centrada: o divisor nao importa |
| `vip_estabilidade.py` | 85 | Sub->Add | `return (Xtr - media) / dp` | equivalente matematico: o PLSRegression centra X internamente |
| `vip_estabilidade.py` | 188 | const 2 | `ax.bar(x, res["media"], width=0.8 if p <= max_barras else 1.0, color=cores, zorder=2)` | estetico (nao muda dado, texto nem estrutura): camada (zorder) das barras |
| `vip_estabilidade.py` | 193 | const 0.8 | `fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2, zorder=3)` | estetico (nao muda dado, texto nem estrutura): espessura, largura da tampa e camada das barras de erro |
| `vip_estabilidade.py` | 193 | const 2 | `fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2, zorder=3)` | estetico (nao muda dado, texto nem estrutura): espessura, largura da tampa e camada das barras de erro |
| `vip_estabilidade.py` | 193 | const 3 | `fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2, zorder=3)` | estetico (nao muda dado, texto nem estrutura): espessura, largura da tampa e camada das barras de erro |
| `vip_estabilidade.py` | 196 | const 7 | `ha="right", fontsize=7)` | estetico (nao muda dado, texto nem estrutura): tamanho da fonte |
| `vip_estabilidade.py` | 197 | const 1 | `ax.axhline(res["limiar"], color="#C44E52", lw=1, ls="--", zorder=4)` | estetico (nao muda dado, texto nem estrutura): espessura e camada da linha do limiar |
| `vip_estabilidade.py` | 197 | const 4 | `ax.axhline(res["limiar"], color="#C44E52", lw=1, ls="--", zorder=4)` | estetico (nao muda dado, texto nem estrutura): espessura e camada da linha do limiar |
| `vip_estabilidade.py` | 198 | const 9 | `ax.set_ylabel("VIP (media das dobras)", fontsize=9)` | estetico (nao muda dado, texto nem estrutura): tamanho da fonte |
| `vip_estabilidade.py` | 200 | const 9 | `f"em >= {int(FREQ_ESTAVEL * 100)}% das dobras; laranja = instavel)", fontsize=9)` | estetico (nao muda dado, texto nem estrutura): tamanho da fonte |
| `vip_cli.py` | 144 | const 1 | `return "en" if a.split("=", 1)[1].lower().startswith("en") else "pt"` | sem efeito observavel: maxsplit 1 -> 2 com um unico '=' no argumento |
| `vip_cli.py` | 214 | raise->pass | `raise _Uso(_m("arg", e="--componentes/--divisoes/--repeticoes/--semente/--limiar")) from e` | ramo de compatibilidade de versao (nao e' codigo morto): no Python 3.12 o argparse com exit_on_error=False levanta ArgumentError tambem para argumento desconhecido, entao o mutante sobrevive; em 3.10 e 3.11 vem como SystemExit e o mutante MORRE (verificado: docs/qualidade/anova/mutacao_vip_cli_py310_linha214.json e _py311_) |

## 7. Achados e correcoes

| onde | achado | correcao | prova |
|---|---|---|---|
| Mutacao (2a rodada): ramos mortos | nove guardas do ANOVA e da CLI nunca eram alcancadas (erro-padrao zero, QM dentro zero, SQ total zero, minimo de 3 residuos, amplitude zero): estavam classificadas como `inalcancavel` | removidas em vez de classificadas (comentario diz a invariante que as dispensa) | mutacao refeita: 500 mutantes em `anova_grupos`, 14 sobreviventes, todos classificados |
| Mutacao (2a rodada): bloco aditivo exato | o texto dizia que QM residual 0 "nao ocorre"; dados perfeitamente aditivos ocorrem (F infinito, p 0, IC degenerado) | o ramo e' alcancavel e ganhou teste proprio (lstsq arredondado para o zero ser exato em qualquer plataforma) | `test_anova_bloco_aditivo_exato.py` |
| Integracao (diagnostico de premissas) | a curtose dos residuos JUNTADOS explodia so' por variancias desiguais (1:100) e a recomendacao saia Kruskal-Wallis em vez de Games-Howell | forma medida em residuos padronizados por grupo | `test_regressao_variancias_desiguais_nao_viram_caudas_pesadas` (vermelho antes, verde depois) |
| Teste existente (oraculo de Games-Howell) | era pulado em SciPy < 1.16 (Python 3.10) | oraculo ingenuo independente da versao; motivo de pulado removido de `tests/pulados_esperados.json` | `test_pos_teste_proprio_igual_ao_*` |
| Mutacao: `figura_vip_estabilidade` | a figura quebrava (matplotlib recusa `yerr` negativo) quando a media e o percentil de uma variavel coincidiam e diferiam por -1e-17 em ponto flutuante: `guaraci vip` caia ao desenhar a figura | `np.maximum(..., 0)` nas duas barras de erro | `test_regressao_figura_com_vip_identico_nas_dobras_nao_quebra_por_erro_de_ponto_flutuante` (vermelho antes, verde depois) |
| Mutacao: limiares | os testes de borda da recomendacao importavam a propria constante do modulo (4, 1 e 3): mudar a constante mudava o teste junto e nada falhava | bordas conferidas com numeros literais | `test_limiares_da_recomendacao_sao_4_1_e_3_literais` |
| Mutacao: ANOVA com bloco | comparacoes e F/p do bloco so' eram conferidos em desenho balanceado, onde a variancia do contraste nao depende do desenho | oraculo independente por minimos quadrados em desenho desbalanceado (nenhum erro de calculo encontrado) | `test_bloco_desbalanceado_confere_f_p_e_comparacoes_com_minimos_quadrados_independentes` |
| Mutacao: CLI | bootstrap padrao, IC das tabelas, bordas `p == alfa`, letras e conteudo da figura nao eram verificados | testes sobre os artistas da figura (posicoes, letras, 150 dpi) e sobre o conteudo exato das tabelas e do relatorio | `test_anova_cli_mutacao.py` |
| Teste novo | o tamanho do PNG dependia do `savefig.bbox` global deixado por outros testes (falha so' na suite completa) | a fixture fixa `savefig.bbox = standard` | `test_figura_tem_dois_paineis_*`, `test_tamanho_da_figura_150_dpi_*` |

## 8. Referencias citadas

| fonte | DOI | conferido em Crossref |
|---|---|---|
| Dunn 1964 | 10.1080/00401706.1964.10490181 | sim |
| Wold, Sjostrom e Eriksson 2001 | 10.1016/s0169-7439(01)00155-1 | sim |
| Chong e Jun 2005 | 10.1016/j.chemolab.2004.12.011 | sim |
| Rochon, Gondan e Kieser 2012 | 10.1186/1471-2288-12-81 | sim |
| Tukey 1949 | 10.2307/3001913 | sim |
| Kramer 1956 | 10.2307/3001469 | sim |
| Games e Howell 1976 | 10.3102/10769986001002113 | sim |
| Nadeau e Bengio 2003 | 10.1023/a:1024068626366 | sim |

Nao citadas por falta de DOI conferido: Holm 1979 e a fonte do epsilon2 (a formula esta no texto do relatorio).

## 9. Verificacoes adicionais da lapidacao

- Amostra de 10 equivalentes verificada contra entradas de borda, com controle de poder: `docs/qualidade/anova/equivalentes_amostra.md`.
- O mutante de compatibilidade do `argparse` (`vip_cli.py` linha 214) sobrevive em 3.12 e morre em 3.10 e 3.11: `docs/qualidade/anova/mutacao_vip_cli_py310_linha214.json` e `docs/qualidade/anova/mutacao_vip_cli_py311_linha214.json`.
- Simulacao da recomendacao de metodo: `docs/qualidade/anova/recomendacao_simulacao.md`; selecao aninhada com rotulos aleatorios: `docs/qualidade/selecao_aninhada_rotulos_aleatorios.json`.

