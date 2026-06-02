# Plataforma de Quimiometria — CLAUDE.md

Guia de desenvolvimento para a plataforma quimiométrica de óleos vegetais
amazônicos (PIBIC / GEAAp / UFPA). Leia antes de editar qualquer arquivo.

---

## Arquivos principais

| Arquivo | Papel | Linhas aprox. |
|---|---|---|
| `pipeline_quimiometria_14.py` | Motor do pipeline (classe Config + funções de análise + `executar()`) | ~5 800 |
| `app_quimiometria.py` | Interface Streamlit (7 abas) | ~2 450 |
| `requirements.txt` | Dependências pip | 26 |
| `.streamlit/config.toml` | Configuração do servidor Streamlit | 24 |

---

## Como executar

### Interface gráfica (recomendado)
```bash
streamlit run app_quimiometria.py
```
Abre em http://localhost:8501

### CLI / menu interativo
```bash
python pipeline_quimiometria_14.py
```

### Instalar dependências
```bash
pip install -r requirements.txt
```

---

## Ambiente Python testado

| Pacote | Versão mínima | Versão atual (2026-05-29) |
|---|---|---|
| Python | 3.10 | 3.14.3 |
| scikit-learn | 1.3 | 1.8.0 |
| numpy | 1.24 | 2.4.4 |
| pandas | 2.0 | 3.0.2 |
| scipy | 1.10 | 1.17.1 |
| streamlit | 1.30 | 1.58.0 |
| shap | 0.42 | 0.52.0 |
| xgboost | 1.7 | 3.2.0 |
| python-pptx | 1.1 | 1.0.2 |

---

## Dados de entrada

```
dados/
  ├── andiroba_T1.dx   # espectros FT-NIR JCAMP-DX
  ├── andiroba_T2.dx
  ├── andiroba_T3.dx
  ├── bacaba_T1.dx
  ...                  # 14 espécies × 3 tríplicas = 42 arquivos .dx
```

**Convenção de nome:** `<especie>_T<n>.dx`  
**mae_id** (anti-leakage GroupKFold): derivado do par `<especie>_T<n>`.

---

## Estrutura do pipeline (`pipeline_quimiometria_14.py`)

### Dataclass de configuração
```python
@dataclass
class Config:
    pasta_dados: str
    pasta_saida: str
    # pré-processamento
    preproc: str = "msc+sg+mc"   # melhor: bal.acc=0.923
    n_componentes_pls: int = 0   # 0 = auto (Wold)
    # opcionais
    executar_benchmark: bool = False
    executar_monte_carlo: bool = False
    n_monte_carlo: int = 100
    executar_shap: bool = False
    shap_max_amostras: int = 500
    ...
```

### Função orquestradora
```python
def executar(cfg: Config) -> None:
    # 0. hardware_probe() → auto_ajustar_config_hardware()
    # 1-3. Leitura de DX → pré-processamento → PCA+HCA
    # 4. PLS-DA (GroupKFold) → VIP → sPLS-DA
    # 5. OPLS-DA (deflação NIPALS) → S-Plot
    # 6. DD-SIMCA (UCL por classe) → Cooman's Plot
    # 7. Validação: permutação, Wold, CV-ANOVA, bootstrap BCa
    # 8. Holdout externo
    # 9a. Benchmark (PLS-DA/SVM/RF/GBM/XGBoost) → DET curves → SHAP
    # 9b. Monte Carlo CV (IC95% por percentil)
```

### Classes e funções chave

| Símbolo | Descrição |
|---|---|
| `PLSDAClassifier` | Wrapper sklearn-compatible para PLS-DA (`PLSRegression` + `LabelBinarizer`) |
| `hardware_probe()` | Detecta RAM/CPU; fallback ctypes no Windows se psutil ausente |
| `auto_ajustar_config_hardware(cfg, hw)` | Ajusta flags pesadas conforme RAM disponível (4 tiers) |
| `_verificar_ram(min_gb, op)` | Guard antes de operações pesadas; retorna False se insuficiente |
| `benchmark_classificadores(...)` | Loop manual GroupKFold; coleta scores + OOF proba; gera DET + SHAP |
| `monte_carlo_cv(...)` | N × StratifiedGroupShuffleSplit; IC95% percentil |
| `fig_det_curvas(...)` | DET linear + log (macro-OvR) |
| `fig_shap_benchmark(...)` | Barplot top-20 + dependence plots top-3 (TreeExplainer) |
| `limpar_resultados_antigos(pasta, n)` | Remove pastas de execuções antigas além de N mais recentes |

---

## Interface Streamlit (`app_quimiometria.py`)

| Aba | Conteúdo |
|---|---|
| 1 Projeto | Metadados + widget status de hardware (RAM/CPU colorido) |
| 2 Dados | Upload de espectros DX + CSV; pré-visualização |
| 3 Pré-proc | Comparação de pipelines de pré-processamento |
| 4 Modelo | Configuração e execução do pipeline; progresso em tempo real |
| 5 Validação | Galeria de figuras (19 categorias); tabela Benchmark; MC CV; Acc por classe |
| 6 Predição | Upload CSV → predição com T²/Q (aceito/rejeitado) |
| 7 Relatórios | Downloads: ZIP, PDF, Word, Excel, LaTeX, **PowerPoint**; limpeza de execuções antigas |

### Progresso em tempo real
O log do processo filho é capturado e parseado pelo padrão `[N/7]`.  
Sub-etapas `[7b/7]` (Benchmark) e `[7c/7]` (MC CV) exibem descrição específica.

---

## Relatórios gerados

| Formato | Função geradora | Conteúdo |
|---|---|---|
| PDF | `_gerar_pdf_relatorio` | Resumo + figuras inline (fpdf2) |
| Word | `_gerar_word_relatorio` | Resumo + tabelas editáveis (python-docx) |
| Excel | `_gerar_excel_relatorio` | 5 abas: Resumo, Métricas, Benchmark, MC CV, VIP |
| LaTeX | `_gerar_latex_template` | Template para Talanta / Food Chemistry / J. Chemom. |
| PowerPoint | `_gerar_pptx_relatorio` | 6+ slides: Capa → Metodologia → Métricas → Figuras → Benchmark → Conclusões |

---

## Boas práticas ao editar

1. **Não quebre o anti-leakage.** O `mae_id = <especie>_T<n>` deve sempre ficar
   inteiramente em treino ou teste — jamais dividido. Use `GroupKFold` ou
   `StratifiedGroupKFold` com `grupos=grupos_cv`.

2. **MSC é stateful.** O `MSCPreprocessor` ajusta `ref_spectrum_` no `fit`.
   Sempre use dentro de `sklearn.Pipeline` para evitar data leakage no CV.

3. **Limites de memória SHAP.** Com 14 classes e ~1 000 features, `shap_values`
   retorna lista de 14 arrays. Mantenha `shap_max_amostras ≤ 500` para RF.

4. **pandas 3.x.** Use `.style.map()` (não `.applymap()`, removido em 2.1+).

5. **openpyxl.** `wb.active` retorna `Optional`; adicione `assert ws is not None`
   antes de acessar células.

6. **fpdf2.** Parâmetro `border` aceita `str | Literal[0, 1]` — use `0`, não `False`.

7. **python-docx Pylance.** `style.font` não está nos type stubs de `BaseStyle`;
   adicione `# type: ignore[union-attr]` nas linhas de acesso a `.font`.

---

## Deploy (Streamlit Cloud)

1. Faça push do repositório para o GitHub (incluindo `.streamlit/config.toml`).
2. Em https://share.streamlit.io → "New app" → selecione o repo.
3. **Main file path:** `app_quimiometria.py`
4. **Python version:** 3.11 (mais estável no Cloud em 2026).
5. O arquivo `requirements.txt` é usado automaticamente pelo Cloud.
6. Pasta `dados/` com os `.dx` deve estar no repositório (42 arquivos, ~1 MB total).

### Execução local com ngrok (sem conta no Cloud)
```bash
pip install pyngrok
python -c "from pyngrok import ngrok; print(ngrok.connect(8501))"
# Em outro terminal:
streamlit run app_quimiometria.py
```

---

## Versões do pipeline

| Versão | Data | Principais adições |
|---|---|---|
| v27 | 2026-05 | `benchmark_classificadores` integrado ao `executar()` |
| v28 | 2026-05 | Monte Carlo CV, SHAP TreeExplainer, DET curves (linear+log), Excel "Benchmark" |
| v29 | 2026-05 | Hardware probe, auto-ajuste por RAM, RAM guards, limpeza de execuções, Acc por classe |
| v30 | 2026-05-29 | PowerPoint (.pptx), `.streamlit/config.toml`, `CLAUDE.md`, requirements atualizado |

---

## Resultado mais relevante

**Melhor pipeline encontrado:** `MSC → SG → MC`  
Balanced Accuracy = **0.923** (GroupKFold, 1 807 amostras, 14 classes de óleos amazônicos).  
Autoscaling isolado: 0.472 — confirma a importância do pré-processamento correto para FT-NIR.
