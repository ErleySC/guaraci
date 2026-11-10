# Cobertura e lacunas

**Público:** quem mantém a suíte · **Estado:** gerado por `scripts/gerar_cobertura_lacunas.py` (não editar à mão) · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

Medida pela suíte completa (`tests/`, ramos), a mesma da verificação final, em src/guaraci, com 50 linhas excluídas e registradas (≈99,90 % sem exclusões); app_quimiometria.py e scripts fora do escopo. 100% de cobertura **não prova correção**: a prova de qualidade dos testes do núcleo é a mutação (`scripts/mutacao_modulo.py`).

## 1. Medição

| Camada | Linhas (antes → depois) | Ramos (antes → depois) | Combinada (antes → depois) |
|---|---|---|---|
| Global | 85.22% → 100.0% | 75.95% → 99.9% | 83.06% → 99.98% |
| Núcleo científico | 90.9% → 100.0% | 82.0% → 100.0% | 88.8% → 100.0% |
| Web (app_tabs) | 61.1% → 100.0% | 41.4% → 99.53% | 56.4% → 99.89% |
| CLI e menus | 65.3% → 100.0% | 57.2% → 99.87% | 63.4% → 99.97% |
| Demais módulos | 91.6% → 100.0% | 83.7% → 99.79% | 89.8% → 99.96% |

Alvos: núcleo 100% de linhas e de ramos; global ≥ 95% (combinada).

## 2. Categorias

- **A — testável barato → teste escrito.** Arquivos `tests/test_lacunas_*.py` (núcleo, CLI, web, figuras, relatórios): 47 arquivo(s).
- **B — defensivo inalcançável → código removido** (prova: o teste da verificação anterior recusa a entrada; nunca `pragma`):

  - test(cobertura): fechamento dos ultimos ramos; remove guardas inalcancaveis em abas web, figuras e relatorio Word
  - test(cobertura): registro de exclusoes + guarda (mutacao 51/54, 3 equivalentes); remove o ultimo pragma evitavel; mutacao_modulo aceita modulo fora do pacote
  - test(cobertura): aba web Modelo; remove guardas mortas de spec e torna o tempo maximo uma constante
  - test(cobertura): figuras - elipse, contorno, bandas VIP, DD-SIMCA, ROC, mapa de calor e bordas; remove laco morto
  - test(cobertura): guaraci.py - ramos restantes; remove duas guardas redundantes
  - test(cobertura): ultimos ramos do nucleo (MC vazio, pasta HSI ja definida); remove guarda inalcancavel de _ci_str
  - test(cobertura): pipeline.executar com execucoes sinteticas e falhas injetadas; remove guardas inalcancaveis
  - test(cobertura): 22 modulos pequenos do nucleo; remove tres ramos inalcancaveis
  - test(cobertura): validacao_estatistica, dados_io e selecao_variaveis; remove ramos inalcancaveis
  - test(cobertura): ramos de borda do nucleo estatistico e dos leitores; remove duas guardas inalcancaveis

- **C — plataforma ou dependência opcional → injeção do ambiente em teste;** sem como injetar, `# pragma: no cover` com motivo (seção 4).
- **D — interface web → `streamlit.testing.v1.AppTest`** sem navegador: 6 arquivo(s) novo(s) mais os existentes. O que o AppTest não alcança fica declarado em `docs/cientifico/limitacoes.md`.

## 3. O que ainda falta

| Arquivo | Camada | Linhas sem execução | Linhas com ramo parcial | Categoria · motivo |
|---|---|---|---|---|
| `app_tabs/predicao.py` | Web (app_tabs) | — | 228, 440 | A (testavel, nao escrito) · dois ramos de borda da aba Predicao: nomes de amostra de tamanho diferente do resultado (linha 228) e legenda 'sem faixa de decisao' quando ha mais quantificadas que faixas (linha 440); sao AppTest baratos, ficam para a v1.1 |
| `catalogo.py` | CLI e menus | — | 256, 258 | A (testavel, nao escrito) · dois ramos do texto de limites de `confianca` (opcao sem 'nao use' / selo sem nota, linhas 256 e 258); a saida e' texto e o conteudo e' conferido por outro teste; ficam para a v1.1 |
| `reports.py` | Demais módulos | — | 613, 625 | A (testavel, nao escrito) · dois ramos da planilha Benchmark do Excel em que so' um dos CSVs (benchmark ou Monte Carlo) existe (linhas 613 e 625); ficam para a v1.1 |

## 4. Exclusões registradas

Cada linha que o coverage exclui (`# pragma: no cover`, `if TYPE_CHECKING:`, stub `...`) e cada `omit`/`exclude_lines` está em `tests/cobertura_excecoes.json`; `tests/test_cobertura_excecoes.py` falha se o número crescer sem entrada.

| Arquivo:função | Quantidade | Categoria | Motivo |
|---|---:|---|---|
| `src/guaraci/__init__.py:<modulo>` | 1 | C | ramo de ImportError durante um import parcial no bootstrap do pacote; so' ocorre com a ordem de importacao quebrada, que o interpretador de teste nao reproduz sem corromper o proprio pacote. |
| `src/guaraci/avaliacao_modelos.py:<modulo>` | 1 | C | bloco `if TYPE_CHECKING:` (excluido pelo padrao do coverage) so' existe para o verificador de tipos; nunca executa em tempo de execucao. |
| `src/guaraci/dados_io.py:<modulo>` | 1 | C | bloco `if TYPE_CHECKING:` (excluido pelo padrao do coverage) so' existe para o verificador de tipos; nunca executa em tempo de execucao. |
| `src/guaraci/figuras.py:<modulo>` | 1 | C | bloco `if TYPE_CHECKING:` (excluido pelo padrao do coverage) so' existe para o verificador de tipos; nunca executa em tempo de execucao. |
| `src/guaraci/hardware.py:<modulo>` | 1 | C | bloco `if TYPE_CHECKING:` (excluido pelo padrao do coverage) so' existe para o verificador de tipos; nunca executa em tempo de execucao. |
| `src/guaraci/hsi_figures.py:<modulo>` | 1 | C | bloco `if TYPE_CHECKING:` (excluido pelo padrao do coverage) so' existe para o verificador de tipos; nunca executa em tempo de execucao. |
| `src/guaraci/hsi_pipeline.py:<modulo>` | 1 | C | bloco `if TYPE_CHECKING:` (excluido pelo padrao do coverage) so' existe para o verificador de tipos; nunca executa em tempo de execucao. |
| `src/guaraci/hsi_uncertainty.py:<modulo>` | 1 | C | bloco `if TYPE_CHECKING:` (excluido pelo padrao do coverage) so' existe para o verificador de tipos; nunca executa em tempo de execucao. |
| `src/guaraci/io_registry.py:<modulo>` | 1 | C | bloco `if TYPE_CHECKING:` (excluido pelo padrao do coverage) so' existe para o verificador de tipos; nunca executa em tempo de execucao. |
| `src/guaraci/model_registry.py:<modulo>` | 1 | C | bloco `if TYPE_CHECKING:` (excluido pelo padrao do coverage) so' existe para o verificador de tipos; nunca executa em tempo de execucao. |
| `src/guaraci/modos_analise.py:<modulo>` | 1 | C | bloco `if TYPE_CHECKING:` (excluido pelo padrao do coverage) so' existe para o verificador de tipos; nunca executa em tempo de execucao. |
| `src/guaraci/perfil_matriz.py:<modulo>` | 1 | C | bloco `if TYPE_CHECKING:` so' existe para o verificador de tipos; nunca executa em tempo de execucao. |
| `src/guaraci/preprocessamento.py:<modulo>` | 1 | C | bloco `if TYPE_CHECKING:` (excluido pelo padrao do coverage) so' existe para o verificador de tipos; nunca executa em tempo de execucao. |
| `src/guaraci/resultados_io.py:<modulo>` | 1 | C | bloco `if TYPE_CHECKING:` (excluido pelo padrao do coverage) so' existe para o verificador de tipos; nunca executa em tempo de execucao. |
| `src/guaraci/robustness.py:<modulo>` | 1 | C | bloco `if TYPE_CHECKING:` (excluido pelo padrao do coverage) so' existe para o verificador de tipos; nunca executa em tempo de execucao. |
| `src/guaraci/selecao_variaveis.py:<modulo>` | 1 | C | bloco `if TYPE_CHECKING:` (excluido pelo padrao do coverage) so' existe para o verificador de tipos; nunca executa em tempo de execucao. |

