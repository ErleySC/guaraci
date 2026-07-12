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
| DD-SIMCA — UCL de T² (`ucl_method="theoretical"`) | Tracy-Young-Mason (1992), fórmula de pequena amostra | limite calculado bate com `hotelling_t2_limite()` (mesma fórmula, computada independentemente no teste) | igual (tolerância relativa 1e-6) | `test_compute_t2_ucl_theoretical_usa_formula_tracy_young` |
| DD-SIMCA — UCL de T² (`ucl_method="chi2"`) | χ²(1−α, k) | limite calculado bate com `scipy.stats.chi2.ppf(0.95, k)` | igual (tolerância relativa 1e-6) | `test_compute_t2_ucl_chi2` |
| DD-SIMCA — UCL de Q-resíduos | Jackson & Mudholkar (1979), aproximação g·χ²(h) | limite bate com `g·χ²(1−α, h)` recomputado independentemente (g=var/2μ, h=2μ²/var) | igual (tolerância relativa 1e-12) | `test_q_residuos_limite_bate_com_formula_jackson_mudholkar` |
| CV-ANOVA (Eriksson, Trygg & Wold, 2008) | Q² = 1 − PRESS/SS_total | caso com valores manualmente calculados (SS_total=20, PRESS=2 → Q²=0.90) | \|Δ\| < 1e-9 | `test_cv_anova_q2_formula` |
| Bootstrap BCa (Efron & Tibshirani, 1993) | propriedades do intervalo (não simulação de cobertura) | predição perfeita → IC=[1,1]; valor observado sempre dentro do IC e em [0,1]; reprodutível com a mesma seed; `n_boot` baixo devolve NaN em vez de um IC enganoso | 5/5 propriedades verificadas | `test_bca_*` (`tests/test_validacao_estatistica.py`) |
| Teste de permutação (*Y-randomization*) | discriminação sinal × ruído | classes separáveis → p baixo (acc=1.000, **p=0.024**); rótulos aleatórios → p alto (acc=0.475, **p=0.781**) | ambos verificados | `test_permutacao_da_p_baixo_com_sinal_real`, `test_permutacao_da_p_alto_com_rotulos_aleatorios` |
| OPLS-DA (Trygg & Wold, 2002; Bylesjö et al., 2006) | ortogonalidade de Gram-Schmidt: `t_orth ⟂ t_pred` | produto interno `t_pred · t_orth` — binário e 14 classes (LDA) | < 1e-6 em ambos os casos | `test_opls_orthogonality_binary`, `test_opls_orthogonality_multiclass` |

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

## O que NÃO está validado aqui (limitações honestas)

- **Sem comparação contra um pacote de referência de DD-SIMCA.** A
  implementação canônica de Pomerantsev & Rodionova é um *toolbox* MATLAB
  proprietário, sem equivalente open-source amplamente disponível para
  *diff* numérico. A validação aqui é **contra as fórmulas fechadas** que
  definem o método (Tracy-Young-Mason para T², χ² de Jackson-Mudholkar
  para Q-resíduos) — não uma comparação pacote-a-pacote.
- **Sensibilidade DD-SIMCA depende dos dados disponíveis.** Com um único
  grupo de réplica pura por classe (`mae_id`), a sensibilidade **não é
  validável por LOGO** (retorna `n/a`, nunca um número inflado por
  re-substituição). Ver `docs/MANUAL.md`, seção 2.1.
- **Sem *benchmark* contra um dataset público externo (Tecator, corn,
  etc.).** Todas as validações desta página usam dados sintéticos ou
  propriedades matemáticas — nenhuma delas roda o pipeline inteiro num
  dataset publicado por terceiros e compara com resultados da literatura
  para aquele dataset. Item de roadmap em aberto.
- **Bootstrap BCa: propriedades, não cobertura empírica.** As 5 checagens
  confirmam que o intervalo se comporta como deveria (contém o valor
  observado, é reprodutível, degrada honestamente com poucas reamostragens)
  — não uma simulação de Monte Carlo medindo se o IC 95% nominal realmente
  cobre o valor verdadeiro em ~95% das repetições.

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
