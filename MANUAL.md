# Manual do GUARACI — Plataforma de Quimiometria

> Manual de instruções das funcionalidades do projeto. Mantido atualizado a
> cada mudança relevante. Para instalação/citação/licença, ver `README.md`.

O **GUARACI** é uma plataforma de quimiometria multi-técnica para autenticação
e caracterização de matrizes (óleos amazônicos por FT-NIR, e qualquer dado
espectral/tabular: Raman, UV-Vis, FTIR, cromatografia). Faz desde a análise
exploratória até classificação, autenticação, quantificação e relatórios de
publicação — com validação estatística rigorosa e anti-vazamento de réplicas.

---

## 1. As três formas de usar

| Forma | Comando | Para quem |
|-------|---------|-----------|
| **Web (Streamlit)** | `streamlit run app_quimiometria.py` | Uso visual, 7 abas guiadas. Também no demo: https://guaraci.streamlit.app |
| **Assistente de terminal** | `python guaraci.py` | Menu interativo colorido, sem editar código |
| **Pipeline direto** | `python pipeline.py --rodar` | Execução automatizada a partir de `config.yaml` |

As três compartilham o mesmo motor (`pipeline.py`) e a mesma configuração
(`config.yaml` / classe `Config`) — não há divergência de resultado entre elas.

---

## 2. Modos de análise (N1 / N2 / N3)

O **modo de análise** define o que o pipeline faz. Na interface aparecem com
nomes amigáveis; internamente são N1/N2/N3.

- **N1 — Classificação (por espécie).** Identifica a qual espécie/classe cada
  amostra pertence (ex.: 13-14 óleos). Método: **PLS-DA** com GroupKFold
  anti-vazamento de réplicas (T1/T2/T3 do mesmo ponto ficam juntas).
- **N2 — Discriminação (puro vs. adulterado).** Autentica pureza **por espécie**
  via **DD-SIMCA** one-class (T² + Q-resíduos com limites por classe).
- **N3 — Quantificação (% de adulterante).** Estima o teor por **regressão PLS
  por espécie** (`pls_regressao_por_espécie`).

---

## 3. Funcionalidades científicas

**Pré-processamento espectral** (dentro do Pipeline, sem vazamento entre folds):
SNV, MSC, Savitzky-Golay (suavização/derivada), mean-centering, autoscaling.
Presets: `msc_sg_mc` (melhor no dataset), `snv_sg_mc`, `mc`, `autoscaling`,
`custom`.

**Análise exploratória:** PCA (scores/loadings), HCA (dendrograma Ward sobre PCs).

**Classificação/discriminação:** PLS-DA, OPLS-DA (com S-Plot), DD-SIMCA
(com Cooman's Plot).

**Seleção de variáveis (Etapa 4):** iPLS (por intervalos), VIP ≥ 1,
Selectivity Ratio (top-20%), sPLS-DA esparso.

**Validação estatística:** teste de permutação, teste de Wold (R²Y/Q²Y),
CV-ANOVA (Eriksson), bootstrap BCa (IC de acurácia), holdout externo
group-aware, Monte Carlo CV (IC95%).

**Comparação de modelos (Auto-Benchmark):** PLS-DA vs SVM RBF vs Random Forest
vs Gradient Boosting vs XGBoost, na mesma CV group-aware. Curvas DET e
interpretabilidade via **SHAP** (TreeExplainer).

**Figuras:** conjunto essencial por padrão (~8) + figuras detalhadas opcionais
(`figuras_detalhadas=True`). Formatos PNG/PDF/SVG, DPI configurável.

**Relatórios:** PDF, Word (.docx), Excel (5 abas), LaTeX e PowerPoint, com capa
de projeto (nome/autor/instituição/objetivo — o "tipo de estudo" é derivado
automaticamente do modo de análise).

---

## 4. Fluxo típico na interface web

1. **Projeto** — preencha nome/autor/instituição/objetivo (descritivo, vai nos
   relatórios).
2. **Dados** — faça upload de um CSV ou aponte a pasta de espectros `.dx`
   (uma subpasta por classe).
3. **Pré-processamento** — escolha o preset; veja o antes/depois.
4. **Modelo** — escolha o modo de análise, ajuste parâmetros (LVs, holdout,
   validação, módulos extras, figuras) e clique **▶️ Run pipeline**.
5. **Validação / Predição / Relatórios** — inspecione métricas por classe,
   figuras, e baixe os relatórios/ZIP.

Tema claro/escuro: menu ⋮ → Settings → Theme (segue o sistema por padrão).

---

## 5. Mapa dos módulos (para desenvolvedores)

Desde a Fase H, o motor está modularizado. `pipeline.py` é a **fachada**:
reexporta tudo, então `import pipeline as pq; pq.X` continua funcionando.

| Módulo | Responsabilidade |
|--------|------------------|
| `pipeline.py` | `Config`, `_CONFIG_SPEC`, orquestrador `executar()`, IO de config, CLI e fachada de reexport |
| `chemometric_stats.py` | VIP, Selectivity Ratio, Hotelling T², Q-resíduos, variância explicada |
| `paleta_cores.py` | Paleta/marcadores de máxima distintividade por classe |
| `dados_io.py` | Parsing JCAMP-DX/ASDF, CSV, sintético; metadados de TITLE |
| `preprocessamento.py` | Transformers SNV/SavGol/MSC + `construir_preprocessador` |
| `classificadores.py` | DD-SIMCA, OPLS-DA |
| `figuras.py` | Camada de plotagem (todas as figuras) |
| `validacao_estatistica.py` | BCa, CV-ANOVA, permutação, Wold, CV manual |
| `hardware.py` | Probe de RAM/CPU/disco, auto-ajuste, guarda de RAM |
| `selecao_variaveis.py` | Etapa 4 (iPLS, sPLS-DA) + figuras da etapa |
| `avaliacao_modelos.py` | PLS-DA, Auto-Benchmark, Monte Carlo CV, DET, SHAP |

Interfaces: `app_quimiometria.py` (web), `guaraci.py` (assistente CLI),
`cli_assistente.py` (menu detalhado). Tema compartilhado: `guaraci_theme.py`,
`design_tokens.py`.

---

## 6. Desenvolvimento

```bash
pytest tests/                 # suíte completa (inclui end-to-end 'slow')
pytest tests/ -m "not slow"   # só os testes rápidos
ruff check .                  # lint estático
```

- **Testes:** `test_pipeline_smoke.py`, `test_pipeline_core.py` (unidade),
  `test_figuras_regressao.py` (regressão de figuras), `test_fachada_reexport.py`
  (protege o contrato de reexport da fachada).
- **CI (GitHub Actions):** lint (ruff) + testes com cobertura, a cada push/PR.
  Dependabot abre PRs semanais de atualização de dependências.

---

*Última revisão do manual: modularização (Fase H) + higiene de CI (ruff,
cobertura, dependabot).*
