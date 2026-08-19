# Arquitetura atual — diagnóstico (2026-08-17)

**Medição, não redesenho.** Nenhum arquivo foi movido, renomeado ou
dividido. Script: [`medir_arquitetura.py`](medir_arquitetura.py) (AST da
biblioteca padrão — não adiciona `pydeps`/`import-linter` só para um
diagnóstico de uma rodada).

**38 módulos, 65 arestas internas.**

---

## Nota de método que muda o resultado pela metade

A primeira execução contava **todos** os imports e reportava 10 ciclos e
14 violações de camada. Errado: 6 dos imports para `pipeline` estão sob
`if TYPE_CHECKING:` — existem só para anotação de tipo, **não criam
dependência em tempo de execução nem ciclo real**. Contá-los infla o
diagnóstico e inventa violação onde não há.

Com `TYPE_CHECKING` excluído: **4 ciclos, 7 violações**. É o número
honesto, e é o que está abaixo.

---

## Fan-in / fan-out

`in` = quantos módulos me importam · `out` = quantos eu importo

| módulo | camada | in | out | in+out |
|---|---|---:|---:|---:|
| **pipeline** | 5 orquestração | 5 | **18** | **23** |
| avaliacao_modelos | 3 dados/análise | 2 | 7 | 9 |
| **config** | 2 método | **9** | 0 | 9 |
| app_logic | 6 interface | 6 | 2 | 8 |
| guaraci | 6 interface | 1 | 7 | 8 |
| **chemometric_stats** | 1 cálculo puro | **6** | 0 | 6 |
| figuras | 4 apresentação | 3 | 3 | 6 |
| dados_io | 3 dados/análise | 3 | 2 | 5 |
| predicao | 3 dados/análise | 3 | 2 | 5 |
| reports | 4 apresentação | 0 | 4 | 4 |
| conformal | 1 cálculo puro | 0 | 0 | 0 |

### Módulos "deus": um só, e é o esperado

**`pipeline` importa 18 dos 38 módulos.** É o único com fan-out alto. Mas
**isso não é acidente arquitetural** — `pipeline` é declaradamente a
fachada do projeto (Fase H), reexportando símbolos dos 10 módulos
extraídos para não quebrar `pipeline.DDSimca(...)` etc. Fan-out alto é a
consequência pretendida de ser fachada.

O que **não** é fachada e merece atenção: `pipeline` tem `executar()` com
~1450 linhas. O problema é o tamanho da função, não o número de imports —
e o P9 do `CLAUDE.md` já diz explicitamente para não tocar nisso antes de
haver rede de segurança.

**`config` (fan-in 9) e `chemometric_stats` (fan-in 6) com fan-out ZERO**
são o padrão saudável: muitos dependem deles, eles não dependem de
ninguém. Núcleo estável.

`conformal` com 0/0 é o módulo novo, ainda não integrado ao `executar()`.

---

## Ciclos de import (4 reais)

```
1. avaliacao_modelos -> model_registry -> avaliacao_modelos
2. app_logic -> pipeline -> guaraci -> app_logic
3. pipeline -> guaraci -> cli_assistente -> pipeline
4. pipeline -> guaraci -> pipeline
```

Todos passam por `pipeline` ou por `guaraci`, e todos funcionam hoje
porque os imports estão **dentro de função** ou porque a ordem de
carregamento resolve. São dívida latente: qualquer reordenação de import
no topo desses módulos pode transformá-los em `ImportError` circular.

**Correção mínima proposta (não aplicada):**

| ciclo | correção | esforço |
|---|---|---|
| 1 | `model_registry` recebe os construtores por parâmetro em vez de importar `avaliacao_modelos` | 1–2 h |
| 2, 3, 4 | extrair de `guaraci.py` o que `pipeline`/`app_logic` consomem para um módulo neutro de camada baixa | 4–6 h |

---

## Violações de camada (7 reais)

Camadas declaradas por intenção: 0 utilitário · 1 cálculo puro · 2 método
· 3 dados/análise · 4 apresentação · 5 orquestração · 6 interface.
Violação = módulo baixo importando módulo alto.

| origem | destino | salto | leitura |
|---|---|---:|---|
| `model_registry` (0) | `avaliacao_modelos` (3) | **+3** | utilitário depende de análise — é o ciclo 1 |
| `reports` (4) | `app_logic` (6) | **+2** | apresentação depende de interface |
| `spectra_preview` (3) | `pipeline` (5) | **+2** | import de runtime, não `TYPE_CHECKING` |
| `avaliacao_modelos` (3) | `figuras` (4) | +1 | análise chama plotagem direto |
| `pipeline` (5) | `guaraci` (6) | +1 | orquestração depende de interface |
| `reports` (4) | `pipeline` (5) | +1 | só para ler `__version__` |
| `selecao_variaveis` (3) | `figuras` (4) | +1 | análise chama plotagem direto |

O padrão dominante é **módulos de análise chamando `figuras.salvar()`
direto** (`avaliacao_modelos`, `selecao_variaveis`). Funciona, mas amarra
cálculo a apresentação: hoje não dá para rodar a Etapa 4 sem matplotlib
disponível.

---

## Backlog pós-defesa, priorizado

Nada aqui é para fazer agora. A três meses da defesa, com o gate
científico aberto, refatoração estrutural não tem retorno — e o próprio
prompt desta rodada diz isso.

| # | Item | Por que vale | Esforço | Risco |
|---|---|---|---|---|
| 1 | Quebrar o ciclo `avaliacao_modelos ↔ model_registry` | menor e mais isolado dos 4; some com 1 mudança de assinatura | 1–2 h | baixo |
| 2 | `reports` deixar de importar `pipeline` só por `__version__` | mover a versão para `guaraci/__init__.py` ou `_version.py` | 30 min | muito baixo |
| 3 | Análise devolver dados em vez de chamar `figuras.salvar()` | desacopla cálculo de matplotlib; permite usar o motor sem GUI | 6–10 h | médio — muda assinaturas públicas |
| 4 | Extrair de `guaraci.py` o núcleo consumido por `pipeline`/`app_logic` | resolve 3 dos 4 ciclos de uma vez | 4–6 h | médio |
| 5 | `executar()` em fases (P9 do CLAUDE.md) | manutenção futura | 3–4 semanas | **alto** — exige golden tests de valores primeiro |

**Recomendação de sequência:** itens 1 e 2 são quase gratuitos e podem
entrar em qualquer janela. O 3 e o 4 depois da defesa. O 5 só depois do
que o próprio P9 exige (golden test de valores, não de PNGs) — e o golden
que existe hoje cobre um subconjunto pequeno.

## O que NÃO é problema

- **Monolito modular já é o desenho certo** para este projeto: pacote
  único, deployável como unidade, fronteiras por módulo. Não há ganho em
  transformar em serviços ou em múltiplos pacotes com um mantenedor só.
- **Fan-out de `pipeline`** é consequência de ser fachada declarada, não
  acidente.
- **`config` e `chemometric_stats`** com fan-out zero são exatamente o que
  se quer de um núcleo estável.
