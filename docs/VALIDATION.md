# VALIDAÇÃO — GUARACI vs. implementações e fórmulas de referência

> Cartão de visita técnico: por que confiar nos números que o Guaraci produz.
> Cada linha desta tabela corresponde a um teste automatizado, executado a
> cada commit pela suíte `pytest`. Os valores abaixo foram obtidos **rodando
> a suíte nesta revisão** (não são estimativas nem números de literatura
> copiados). Para instalação, ver `README.md`; para como usar cada
> funcionalidade, ver `docs/MANUAL.md`.

## Tabela de validação

| Método | Referência | O que é verificado | Resultado | Teste |
|---|---|---|---|---|
| PLS-DA (`PLSDAClassifier`) | `sklearn.PLSRegression` + argmax manual | max\|Δcoef\| entre o wrapper Guaraci e a reprodução manual sklearn | **0.0** (exato) | `test_plsda_classifier_vs_sklearn_manual` |
| SNV — normalização | Barnes, Dhanoa & Lister (1989) | cada espectro após SNV tem média 0 e desvio-padrão 1 (definição analítica do método) | \|média\| < 1e-9, \|desvio − 1\| < 1e-9 | `test_snv_normaliza_por_amostra` |
| SNV — invariância de espalhamento | Barnes, Dhanoa & Lister (1989) | SNV(a + b·x) = SNV(x) para b>0 (anula ganho/offset por espectro — a justificativa física do método) | \|Δ\| < 1e-10 | `test_snv_invariante_a_escala_e_offset` |
| VIP (Chong & Jun, 2005) | propriedade Σ VIP² = p (nº de variáveis) | `mean(VIP²)` deve ser exatamente 1.0 | 1.0 (tolerância relativa 1e-6) | `test_vip_propriedade_soma_igual_p` |
| MSC | *stateful* (referência do treino) | `ref_` é ajustado **somente** no conjunto de treino (nunca refeito no teste) — a propriedade que evita vazamento | verificado (shape e ausência de refit) | `test_msc_no_leakage` |
| DD-SIMCA — UCL de T² (`ucl_method="theoretical"`) | Tracy-Young-Mason (1992), fórmula de pequena amostra | limite calculado bate com `hotelling_t2_limit()` (mesma fórmula, computada independentemente no teste) | igual (tolerância relativa 1e-6) | `test_compute_t2_ucl_theoretical_usa_formula_tracy_young` |
| DD-SIMCA — UCL de T² (`ucl_method="chi2"`) | χ²(1−α, k) | limite calculado bate com `scipy.stats.chi2.ppf(0.95, k)` | igual (tolerância relativa 1e-6) | `test_compute_t2_ucl_chi2` |
| DD-SIMCA — UCL de Q-resíduos | Jackson & Mudholkar (1979), aproximação g·χ²(h) | limite bate com `g·χ²(1−α, h)` recomputado independentemente (g=var/2μ, h=2μ²/var) | igual (tolerância relativa 1e-12) | `test_q_residuos_limite_bate_com_formula_jackson_mudholkar` |
| DD-SIMCA — regra de decisão (aceitar/rejeitar) | Kucheryavskiy, Rodionova & Pomerantsev (2024) *J. Chemometrics* 38(7):e3556, Eq. 3–4: distância combinada f=(T²/h₀)·N_h+(Q/q₀)·N_q ≤ χ²(1−α, N_h+N_q) | **corrigido em 2026-08-08** — a versão anterior aceitava se T²≤UCL(T²) **e** Q≤UCL(Q) independentemente (região retangular, não a do método citado); com alpha independente por eixo a rejeição conjunta efetiva era ~1−(1−α)²≈0.0975, quase o dobro do declarado. Propriedade verificada: o "E" de dois testes só pode aceitar ≤ cada teste isolado (P(A∩B)≤min(P(A),P(B))), sempre verdadeiro; a regra combinada nova não tem essa penalidade estrutural | propriedade estrutural + teste de discordância mensurável entre as duas regras | `test_predict_usa_distancia_combinada_nao_regra_retangular`, `test_predict_e_score_matrix_f_concordam` |
| CV-ANOVA (Eriksson, Trygg & Wold, 2008) | Q² = 1 − PRESS/SS_total | caso com valores manualmente calculados (SS_total=20, PRESS=2 → Q²=0.90) | \|Δ\| < 1e-9 | `test_cv_anova_q2_formula` |
| Bootstrap BCa (Efron & Tibshirani, 1993) | propriedades do intervalo (não simulação de cobertura) | predição perfeita → IC=[1,1]; valor observado sempre dentro do IC e em [0,1]; reprodutível com a mesma seed; `n_boot` baixo devolve NaN em vez de um IC enganoso | 5/5 propriedades verificadas | `test_bca_*` (`tests/test_validacao_estatistica.py`) |
| Teste de permutação (*Y-randomization*) | discriminação sinal × ruído | classes separáveis → p baixo (acc=1.000, **p=0.024**); rótulos aleatórios → p alto (acc=0.475, **p=0.781**) | ambos verificados | `test_permutacao_da_p_baixo_com_sinal_real`, `test_permutacao_da_p_alto_com_rotulos_aleatorios` |
| OPLS-DA (Trygg & Wold, 2002; Bylesjö et al., 2006) | propriedades do Algoritmo 1 (sem Gram-Schmidt extra, **corrigido em 2026-10-14**): `t_orth ⟂ y`, `t_orth ⟂ t_pred`, `transform(X_treino)` = escores do `fit`, predição = PLS1 com k+1 componentes | produtos internos normalizados (k = 1, 2, 3); comparação com `sklearn.PLSRegression(k+1)` no treino e em X novo; ortogonalidade também em 14 classes (alvo via PLS2) | < 1e-10 (ortogonalidade), < 1e-8 (predição) | `test_oplsda_t_orth_ortogonal_a_y_e_a_t_pred`, `test_oplsda_transform_no_treino_reproduz_escores_do_fit`, `test_oplsda_predicao_igual_pls1_com_k_mais_1_componentes`, `test_opls_orthogonality_*` |

**Reproduzir:**
```bash
pytest tests/test_pipeline_smoke.py::test_plsda_classifier_vs_sklearn_manual \
       tests/test_pipeline_core.py::test_vip_propriedade_soma_igual_p \
       tests/test_pipeline_core.py::test_snv_normaliza_por_amostra \
       tests/test_pipeline_core.py::test_snv_invariante_a_escala_e_offset \
       tests/test_pipeline_core.py::test_q_residuos_limite_bate_com_formula_jackson_mudholkar \
       tests/test_pipeline_smoke.py::test_msc_no_leakage \
       tests/test_classificadores.py::test_compute_t2_ucl_theoretical_usa_formula_tracy_young \
       tests/test_classificadores.py::test_compute_t2_ucl_chi2 \
       tests/test_validacao_estatistica.py \
       tests/test_pipeline_smoke.py::test_opls_orthogonality_binary \
       tests/test_pipeline_smoke.py::test_opls_orthogonality_multiclass \
       -v
```
Ou, para a suíte inteira (inclui os testes acima entre os demais testes de
funcionalidade — não é uma suíte separada):
```bash
pytest -q
```

## O diferencial real: validação *group-aware*

Nenhuma linha da tabela acima testa isso diretamente porque não é uma
fórmula com "resposta certa" externa — é uma **decisão de arquitetura**: toda
validação cruzada do Guaraci (`GroupKFold`/`GroupShuffleSplit` por `mae_id`)
impede que réplicas físicas da mesma amostra (T1/T2/T3) caiam em treino e
teste ao mesmo tempo. Sem isso, a acurácia reportada mede o modelo
reconhecendo a *própria réplica* que já viu — não a capacidade de
generalizar para uma amostra nova. Esse vazamento é comum na literatura de
espectroscopia e nenhum pacote mainstream (scikit-learn, mdatools,
hyperSpec) protege contra ele por padrão.

Consequência prática, já corrigida nesta revisão (ver `CHANGELOG`/commits):
a sensibilidade do DD-SIMCA é estimada por **leave-one-group-out (LOGO)**
em vez de re-substituição — ver a seção "Limitações" abaixo.

## Oráculos externos (implementações de referência em R)

Além das fórmulas fechadas da tabela acima, as contas centrais são
comparadas **número a número** com pacotes de referência escritos por
terceiros, rodados em R 4.3.3. Os scripts geradores (`gen_oracle.R`) e as
saídas ficam em `tests/fixtures/oraculo_*/`. Tolerância relativa 1e-8 onde
o algoritmo é o mesmo; 1e-4 onde os dois lados iteram (NIPALS).

| Método | Referência externa | Dados | Teste |
|---|---|---|---|
| PLS-DA binário e multiclasse, VIP, SR | mdatools 0.16.0 (SIMPLS), ropls 1.45.0 (NIPALS) | sintéticos; sacurine (Thévenot et al. 2015); iris (Fisher 1936) | `test_oraculo_referencias.py`, `test_oraculo_dados_reais.py` |
| OPLS / OPLS-DA (escores, predição) | ropls 1.45.0 | sintéticos; sacurine | idem |
| DD-SIMCA (distâncias h e q, momentos, f, decisão) | mdatools 0.16.0 `ddsimca` | sintéticos; iris | idem |
| SNV, MSC, EMSC (grau 2), PQN, Savitzky-Golay (derivadas 0–2, bordas incluídas) | mdatools `prep.norm`, `prep.emsc`, `prep.savgol` | espectros sintéticos | `test_oraculo_prep_limites.py` |
| Limite T² de Hotelling | mdatools `hotelling.crit` | grade n × k × α | idem |
| Limite Q (χ² por momentos) | `stats::qchisq` | Q sintético | idem |
| IC bootstrap BCa | `boot` (Canty & Ripley) `bca.ci`, com as mesmas réplicas | AUC e acurácia | idem |

Diferenças de **convenção** encontradas (não são bugs; os testes travam o
fator exato):

- **SNV:** o Guaraci divide pelo desvio com ddof=0; o mdatools usa ddof=1.
  A diferença é o fator constante √((p−1)/p).
- **PQN:** com a mesma referência, o mdatools normaliza por área antes, o
  que dá o fator constante Σ|ref|. A referência padrão também difere: o
  Guaraci usa o espectro mediano do treino, o mdatools a média dos
  espectros normalizados por área.
- **T²:** o Guaraci usa o limite de Fase II de Tracy, Young & Mason (1992);
  o mdatools usa k(n−1)/(n−k)·F. A razão entre os dois é exatamente (n+1)/n.
- **VIP:** o Guaraci e o ropls usam os pesos W (Wold); o mdatools usa os
  pesos R = W(P'W)⁻¹ do SIMPLS. Os VIPs diferem por definição.
- **DD-SIMCA:** o Guaraci calibra q₀/N_q com Q leave-one-out; o mdatools
  usa o Q in-sample. As distâncias batem; os limites diferem por desenho.
- **BCa:** z₀, a aceleração e os alfas ajustados batem com o `boot`. O
  quantil final usa interpolação linear (`np.percentile`), e não a
  `norm.inter` do `boot`. A diferença fica entre estatísticas de ordem
  vizinhas.

**Bug encontrado por esta comparação:** o SR (`compute_selectivity_ratio`)
era calculado sobre X não centrado. O defeito foi corrigido em 2026-10-14.

## O que NÃO está validado aqui (limitações honestas)

- **DD-SIMCA comparado só com o mdatools.** A comparação cobre as
  distâncias, os momentos e a decisão (ver "Oráculos externos"). O
  *toolbox* MATLAB de Pomerantsev & Rodionova é proprietário e não entrou
  na comparação. Os limites por eixo continuam validados contra as
  fórmulas fechadas (Tracy-Young-Mason para T² e χ² por momentos para Q).
- **Sensibilidade DD-SIMCA depende dos dados disponíveis.** Com um único
  grupo de réplica pura por classe (`mae_id`), a sensibilidade **não é
  validável por LOGO** (retorna `n/a`, nunca um número inflado por
  re-substituição). Ver `docs/MANUAL.md`, seção 2.1.
- **Benchmark contra dataset público externo: feito (2026-07-13), parcial.**
  `docs/BENCHMARK_TECATOR.md` roda o motor real de pré-processamento +
  regressão PLS do GUARACI no dataset Tecator (NIR, teor de gordura em
  carne, Thodberg 1996) — RMSEP 2,0–2,3% / R²pred 0,97–0,98, dentro da
  faixa esperada da literatura. Cobre pré-processamento + PLS; **não**
  cobre DD-SIMCA, OPLS-DA, classificação, nem a validação *group-aware*
  via `mae_id` (Tecator não tem réplicas físicas — não aplicável a este
  dataset). Reproduzível: `python scripts/benchmark_tecator.py`.
- **Partição de validação cruzada: estável por implementação própria, não
  herdada do scikit-learn.** Achado em 2026-08-05 pelo golden test
  (`tests/test_golden_valores.py`): o `StratifiedGroupKFold` do scikit-learn
  **muda a partição entre versões mesmo com `random_state` fixo** — com dados
  idênticos (72 amostras, 3 classes, 24 grupos, `random_state=42`), **42% das
  amostras caíam em fold diferente** entre 1.7.2 e 1.9.0. Isso tornava `Q2`,
  `RMSECV`, acurácia, F1, kappa e o nº de LVs ótimas dependentes da versão
  instalada — incompatível com a alegação de reprodutibilidade que sustenta o
  projeto. O Guaraci passou a usar `StableStratifiedGroupKFold`
  (`guaraci.validacao_estatistica`), que congela a ordenação por hash
  determinístico do id do grupo. **Verificação:** a partição tem hash
  idêntico em scikit-learn 1.7.2 e 1.9.0 (antes: dois hashes diferentes).
  Testada por 9 testes de propriedade, incluindo o que garante que réplicas
  de um mesmo `mae_id` nunca se separam. Mesmo assim, continua sendo boa
  prática citar o ambiente (`requirements-lock.txt`) junto de números
  publicados — o golden grava numpy/scipy/scikit-learn junto dos valores.
- **Bootstrap BCa: propriedades, não cobertura empírica.** As 5 checagens
  confirmam que o intervalo se comporta como deveria (contém o valor
  observado, é reprodutível, degrada honestamente com poucas reamostragens)
  — não uma simulação de Monte Carlo medindo se o IC 95% nominal realmente
  cobre o valor verdadeiro em ~95% das repetições.
- **AG e SPA (Etapa 4, opcionais): nested-CV desde a v31.3.0.** A *fitness*
  do AG e a pontuação do SPA usavam a MESMA CV (`cv_indices`) cujo
  resultado era depois reportado como número final — double dipping por
  construção (a busca otimizava diretamente contra a partição que media o
  resultado). Corrigido com `_avaliar_busca_nested_cv`: a cada fold
  externo, a busca inteira roda de novo usando só os dados de treino
  daquele fold (com uma CV interna própria, não *group-aware*, só para
  guiar a otimização); a máscara resultante é avaliada no fold de teste
  externo, nunca visto pela busca. Custo: a busca roda ~N vezes mais
  (N = número de folds) — aceitável, pois ambos já são *opt-in* e
  documentados como mais lentos.

## Referências completas

BARNES, R.J.; DHANOA, M.S.; LISTER, S.J. Standard normal variate
transformation and de-trending of near-infrared diffuse reflectance
spectra. **Applied Spectroscopy**, v. 43, n. 5, p. 772-777, 1989.

BYLESJÖ, M. et al. OPLS discriminant analysis: combining the strengths of
PLS-DA and SIMCA classification. **Journal of Chemometrics**, v. 20, n.
8-10, p. 341-351, 2006.

CHONG, I.-G.; JUN, C.-H. Performance of some variable selection methods
when multicollinearity is present. **Chemometrics and Intelligent
Laboratory Systems**, v. 78, n. 1-2, p. 103-112, 2005.

EFRON, B.; TIBSHIRANI, R.J. **An Introduction to the Bootstrap**. Boca
Raton: CRC Press, 1993.

ERIKSSON, L.; TRYGG, J.; WOLD, S. CV-ANOVA for significance testing of PLS
and OPLS models. **Journal of Chemometrics**, v. 22, n. 11-12, p. 594-600,
2008.

JACKSON, J.E.; MUDHOLKAR, G.S. Control procedures for residuals associated
with principal component analysis. **Technometrics**, v. 21, n. 3, p.
341-349, 1979.

THODBERG, H.H. A review of Bayesian neural networks with an application to
near infrared spectroscopy. **IEEE Transactions on Neural Networks**, v. 7,
n. 1, p. 56-72, 1996. doi:10.1109/72.478392 — origem do dataset Tecator
usado em `docs/BENCHMARK_TECATOR.md`.

TRACY, N.D.; YOUNG, J.C.; MASON, R.L. Multivariate control charts for
individual observations. **Journal of Quality Technology**, v. 24, n. 2,
p. 88-95, 1992.

TRYGG, J.; WOLD, S. Orthogonal projections to latent structures (O-PLS).
**Journal of Chemometrics**, v. 16, n. 3, p. 119-128, 2002.

---
*Última revisão: 2026-07-12. Todos os valores desta página foram obtidos
rodando `pytest` nesta sessão de trabalho — não copiados de execuções
anteriores nem estimados. Ver `docs/MANUAL.md` para o manual de uso e
`CLAUDE.md` para o roadmap de robustez do projeto.*
