# CLAUDE.md — Instruções de trabalho no repositório GUARACI

> Este arquivo é lido automaticamente pelo Claude Code. Ele define **como você deve
> se comportar**, **o que o projeto é**, **o que está errado nele** e **em que ordem
> consertar**. Leia inteiro antes da primeira ação.

---

## 0. REGRAS DE COMPORTAMENTO (não negociáveis)

### 0.1 Tom
- **Nunca elogie por padrão.** Não escreva "ótima ideia", "excelente pergunta",
  "você está certo", "isso faz total sentido". Se algo está bom, prove com um
  argumento específico ou não diga nada.
- **Não repita meu enquadramento de volta.** Se eu disser "acho que X é o melhor
  caminho", não comece com "X é realmente o melhor caminho". Comece pelo contra-argumento.
- **Comece pela coisa mais útil.** Se a resposta é "não vai funcionar", diga isso
  na primeira frase.
- **Sem preâmbulo, sem resumo do que você vai fazer, sem "vou agora...".** Faça.
- Quanto mais confiante eu parecer, mais você deve procurar o furo.

### 0.2 Honestidade técnica
- **Nunca invente número.** Se você não rodou o teste, não afirme cobertura,
  contagem de linhas, ou resultado. Rode o comando ou diga "não verifiquei".
- **Nunca invente API.** Antes de chamar `guaraci.foo()`, leia o arquivo e confirme
  que `foo` existe com essa assinatura.
- **Nunca invente referência bibliográfica.** Toda citação deve ser real e verificável.
  Se não tiver certeza, escreva `[VERIFICAR]` e siga.
- Se um plano meu está errado, diga antes de executar. Não execute em silêncio um
  plano que você acha ruim.
- Se você quebrou algo, diga imediatamente. Não tente esconder com um patch por cima.

### 0.3 Antes de mexer em código
1. **Leia o arquivo inteiro** antes de editar. Não edite baseado em suposição.
2. **Rode a suíte de testes antes e depois.** Se a contagem de testes passando cair,
   pare e reporte.
3. **Não refatore o que não te pedi.** Refactor oportunista mistura mudança de
   comportamento com mudança de forma e torna o diff impossível de revisar.
4. **Um commit = uma mudança lógica.** Não empacote 5 correções em um commit.

### 0.4 Autoavaliação
- **Não gere notas** ("8,5/10", "nota geral"). Nota sem rubrica é ruído.
- **Não escreva que o projeto é "raro para graduação"** ou similar. Autoelogio de IA
  sobre trabalho que ela mesma ajudou a fazer é circular e sem valor.
- Quando eu pedir avaliação, dê: o que está quebrado, o que é risco, o que falta.
  Não dê medalha.

---

## 1. O QUE É O PROJETO

**GUARACI** é uma plataforma Python de quimiometria multitécnica, aberta e reprodutível,
para classificação, autenticação, exploração e quantificação de matrizes complexas.
Caso de uso âncora: autenticação de óleos fixos amazônicos por FT-NIR (ABB MB3600,
8192 pts, ~934 amostras, 14 classes, adulterantes soja/milho/algodão).

**Contexto do autor:** Erley, graduação em Química (UFPA), grupo GEAAp. Este é o
software do TCC, mas a ambição é que ele seja usável por terceiros.

### Diferencial real
**Validação group-aware (`GroupKFold` por `mae_id`)** — impede que réplicas físicas
da mesma amostra caiam em treino e teste ao mesmo tempo. Esse vazamento infla acurácia
e é comum na literatura de espectroscopia. Nenhum pacote mainstream protege contra isso
por padrão. **Este é o argumento central do projeto. Tudo que o enfraquece é grave.**

### Arquitetura
- Pacote real: `src/guaraci/` (~34 módulos, ~17.500 linhas)
- Três interfaces sobre o **mesmo motor** (`executar()`):
  - Web: `app_quimiometria.py` → `app_tabs/` (Streamlit, 8 abas)
  - CLI: `guaraci.py` (Rich)
  - Direto: `python -m guaraci.pipeline`
- Saída: `<Amostra>/<Objetivo>/<Execução>/{Graficos,Tabelas,Relatorios,Modelos}`

### Módulos-chave
| Módulo | Papel |
|---|---|
| `pipeline.py` | Motor central + `executar()` |
| `chemometric_stats.py` | Diagnósticos puros, figuras de mérito (FOM) |
| `classificadores.py` | DD-SIMCA, OPLS-DA, PLS-DA |
| `preprocessamento.py` | SNV, MSC (stateful), SavGol, autoscale |
| `validacao_estatistica.py` | Permutação, Wold, BCa, CV-ANOVA |
| `selecao_variaveis.py` | iPLS, VIP, SR, sPLS-DA, SPA, GA-PLS |
| `avaliacao_modelos.py` | Benchmark, Monte Carlo CV, SHAP, DET |
| `dados_io.py` | JCAMP-DX / CSV, agrupamento `mae_id` |
| `dados_imagem.py` | Colorimetria digital — **PROTÓTIPO, não validado** |
| `predicao.py` | Aplica modelo `.joblib` a amostra nova |
| `modos_analise.py` | Gating por objetivo (Explor./Classif./Quantif.) |
| `figuras.py` | Camada de plotagem |
| `reports.py` | PDF, Word, Excel, LaTeX, PPTX |

---

## 2. ESTADO ALEGADO (VERIFICAR ANTES DE CONFIAR)

Estes números vêm de uma auditoria anterior. **Sua primeira tarefa em qualquer sessão
nova é reverificá-los.** Se divergirem, o código vence, e você me avisa da divergência.

| Item | Valor alegado | Comando para verificar |
|---|---|---|
| Versão | 31.8.0 | `grep -r version pyproject.toml` |
| Testes | 562 pass, 1 skip | `pytest -q` |
| Cobertura | 64% | `pytest --cov=src/guaraci --cov-report=term-missing` |
| Lint | ruff limpo | `ruff check .` |
| `executar()` | 1363 linhas | `grep -n "def executar" src/guaraci/pipeline.py` |
| `print()` em pipeline | 164 | `grep -c "print(" src/guaraci/pipeline.py` |
| `except` amplos | 51 (100% com `noqa: BLE001` justificado) | `grep -rn "except Exception\|except:" src/guaraci/ \| wc -l` |
| `guaraci.py` | 3318 linhas | `wc -l src/guaraci/guaraci.py` |
| TODO/FIXME reais | 6 (todos em `reports.py`, placeholders de template LaTeX — não são dívida de código) | `grep -rn "TODO\|FIXME\|HACK" src/guaraci/ \| grep -v NOTAS_METODOLOGICAS` |

> Atualizado em 2026-07-13 após a auditoria de 15 etapas de 2026-07-12. A tabela
> anterior (114 `except`, `executar()` 1269 linhas, `guaraci.py` 3133 linhas)
> estava desatualizada em relação ao código desde a v31.2.0 — não foi
> reverificada por várias sessões seguidas. **Reverifique a cada sessão nova,
> não confie neste valor por mais de uma sessão de trabalho.**

---

## 3. OS PROBLEMAS, EM ORDEM DE GRAVIDADE

### ✅ P1 (RESOLVIDO em 2026-07-11) — Sensibilidade DD-SIMCA reportada como 100% é re-substituição
**O que acontece:** só existem 3–4 amostras puras por espécie, e **todas estão no treino**.
A sensibilidade reportada mede o modelo classificando dados que ele já viu. É a prova
que o próprio aluno corrigiu. O número é vazio.

**Por que é o pior problema:** é o único que pode derrubar a defesa do TCC, e é o único
que compromete a *ciência*, não a engenharia.

**Solução (não é escrever disclaimer — é mudar o cálculo):**
Implementar **leave-one-group-out por `mae_id`** para a sensibilidade.

```python
def sensibilidade_logo(X, mae_ids, construir_modelo):
    """Sensibilidade DD-SIMCA por leave-one-group-out (LOGO).

    Com poucos grupos (n<10), LOGO é a única estimativa honesta possível.
    Re-substituição (treinar e testar no mesmo dado) NÃO é aceitável.

    Returns
    -------
    dict com chaves: sensibilidade, n_grupos, aviso (str|None)
    """
    grupos = np.unique(mae_ids)
    acertos = []
    for g in grupos:
        treino, teste = mae_ids != g, mae_ids == g
        modelo = construir_modelo(X[treino])
        acertos.append(bool(modelo.predict(X[teste]).all()))
    n = len(grupos)
    aviso = None
    if n < 10:
        aviso = (f"Sensibilidade estimada por LOGO com apenas {n} grupos. "
                 f"Incerteza alta; IC bootstrap não é confiável neste regime. "
                 f"Interpretar como exploratória.")
    return {"sensibilidade": float(np.mean(acertos)), "n_grupos": n, "aviso": aviso}
```

**Critério de aceite:**
- Nenhuma figura, tabela ou linha de relatório mostra sensibilidade sem `n_grupos` ao lado.
- Quando `n_grupos < 10`, o aviso aparece na figura, na tabela e no relatório.
- Existe um teste que **falha** se a chave `n_grupos` estiver ausente do dict de resultado.
- O valor vai cair abaixo de 100%. **Isso é o objetivo, não um problema.**

---

### ✅ P2 (RESOLVIDO em 2026-07-11) — Regressão agrupando espécies esconde as falhas por adulterante
**O que acontece:** a regressão PLS agrupando espécies dá R² ≈ 0 — a matriz vegetal
domina o sinal. A leitura correta é **espécie × adulterante**: 21 de 37 combinações
funcionam (R²cv ≥ 0,70). **16 falham.** Hoje isso só existe em script externo e não
aparece em nenhuma saída oficial.

**Solução:** o heatmap espécie×adulterante vira figura **nativa** do `executar()` no
objetivo Quantificação, e as células que falham ficam **visualmente marcadas**, não em branco.

Especificação:
- Y = espécie, X = adulterante (soja/milho/algodão)
- Cor = R²cv, escala divergente ancorada em 0,70
- Célula com R²cv < 0,70: **hachurada** + valor escrito. Não some.
- Célula sem dado: cinza explícito com "n/a"
- Título/legenda contém: `"16/37 combinações abaixo de R²cv = 0,70"`

**Critério de aceite:**
- A figura é gerada por `executar()` — zero script externo.
- O contador de falhas aparece no título da figura E no relatório.
- Teste que verifica: se todas as células passassem, o contador seria 0/N.

---

### 🟢 P3 (MAJORITARIAMENTE RESOLVIDO) — 114 blocos `except Exception` / `except:` — dívida de CORREÇÃO, não de estilo

> Estado em 2026-07-13: já caiu para **51** blocos, **100% com `noqa: BLE001`
> justificado** (zero `except` nu sem tratamento). A trava de lint `BLE` da
> seção abaixo está ativa e sendo seguida de fato. Achado residual real na
> auditoria de 2026-07-12: um desses `except Exception` (`guaraci.py`, menu
> Visualização) mascarava um bug permanente — 4 opções de menu apontando
> para funções que nunca existiram no código. Corrigido em 2026-07-13
> removendo as opções quebradas (ver `docs/CHANGELOG.md`). O restante dos
> 51 continua categorizado A/B corretamente — não precisa de nova varredura
> completa, só reverificar quando o número subir de novo.
**Por que é pior do que parece:** um `except: pass` numa função de cálculo faz o Guaraci
**produzir um número errado sem avisar ninguém**. Em software científico isso é o pior
tipo de bug: não trava, não avisa, só mente. Pode haver um R² errado no TCC agora mesmo.

**Solução — política de 3 categorias, aplicada arquivo por arquivo:**

```python
# CATEGORIA A — best-effort intencional (relatório opcional, plot extra).
# PERMITIDO. Mas SEMPRE loga e SEMPRE comenta o porquê.
try:
    gerar_pptx(resultados, path)
except Exception:  # noqa: BLE001 — PPTX é extra; ausência não invalida a análise
    log.warning("PPTX não gerado (dependência opcional ausente?)", exc_info=True)

# CATEGORIA B — erro esperado. Capture o TIPO EXATO e re-erga com contexto.
try:
    cfg = carregar_yaml(path)
except FileNotFoundError:
    raise ConfigError(f"Config não encontrado: {path}") from None
except yaml.YAMLError as e:
    raise ConfigError(f"YAML inválido em {path}: {e}") from e

# CATEGORIA C — cálculo científico. NUNCA capture amplo. DEIXE ESTOURAR.
# Se o SVD não converge, o usuário PRECISA saber. NaN silencioso vira paper errado.
scores = pca.fit_transform(X)   # sem try/except
```

**Regra de ouro:** todo `except` amplo em módulo de cálculo (`chemometric_stats`,
`classificadores`, `preprocessamento`, `validacao_estatistica`, `selecao_variaveis`)
é **categoria C até prova em contrário**.

**Como executar sem enlouquecer:**
```bash
grep -rn "except Exception\|except:" src/guaraci/ > /tmp/excepts.txt
# classificar: arquivo | linha | categoria A/B/C | ação
```

**Travar com lint para não voltar:**
```toml
[tool.ruff.lint]
select = ["E", "F", "BLE", "TRY", "LOG", "G"]
# BLE001 = blind except | TRY = anti-padrões de exceção | LOG/G = logging correto
```

**Critério de aceite:**
- `ruff check` com `BLE` ativo passa limpo.
- Todo `noqa: BLE001` tem comentário justificando na mesma linha.
- **Zero `except: pass`** no repositório.

---

### 🟢 P4 (núcleo científico feito, 2026-07-13) — Cobertura 64% é baixa demais para software que gera números publicáveis
> Núcleo científico (a linha "≥95%" abaixo) fechado nesta sessão via testes de
> PROPRIEDADE (edge cases numéricos: NIPALS/OPLS degenerados, LDA-fallback,
> LOGO com fold impossível, bootstrap/jackknife/CV com iteração que falha) —
> não valor gravado. Cobertura TOTAL do projeto continua 64% (não era o alvo
> desta rodada — as outras camadas da tabela abaixo não foram tocadas).
> `figuras.py` também já está em 87% (o "43%" abaixo é o número antigo da
> auditoria original, desatualizado).

**Metas assimétricas (não perseguir 100% global — é desperdício):**

| Camada | Meta | Estado em 2026-07-13 |
|---|---|---|
| `chemometric_stats` | ≥ 95% | ✅ 98% |
| `classificadores` | ≥ 95% | ✅ 97% |
| `preprocessamento` | ≥ 95% | ✅ 100% |
| `validacao_estatistica` | ≥ 95% | ✅ 95% (exato) |
| `dados_io`, `config`, `config_io` | ≥ 90% | dados_io 83% (não atingida), config/config_io 100%/98% |
| `pipeline.executar()` | ≥ 75% | 71% (pipeline.py inteiro; não separado por função) |
| `figuras.py` | ≥ 60% | ✅ 87% |
| Menus CLI/UI (`guaraci.py`) | ≥ 30% | 15% (não atingida — maior gap real remanescente) |

**Como subir sem escrever teste inútil — teste PROPRIEDADE MATEMÁTICA, não valor gravado:**

```python
# ❌ RUIM — testa que o número não mudou. Se estava errado, continua errado.
def test_snv():
    assert snv(X)[0, 0] == pytest.approx(-1.234567)

# ✅ BOM — testa a DEFINIÇÃO do método.
def test_snv_propriedade():
    """SNV: cada espectro vira média 0, desvio-padrão 1."""
    Xs = snv(X)
    np.testing.assert_allclose(Xs.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(Xs.std(axis=1, ddof=1), 1, atol=1e-12)

# ✅ BOM — testa a INVARIÂNCIA que justifica o método existir.
def test_snv_invariante_escala():
    """SNV deve anular multiplicação por escalar (efeito de caminho óptico)."""
    np.testing.assert_allclose(snv(X), snv(3.7 * X), atol=1e-10)

# ✅ MELHOR — property-based: gera centenas de matrizes tentando quebrar o código.
from hypothesis import given
from hypothesis.extra.numpy import arrays
@given(arrays(np.float64, (10, 50), elements=floats(0.1, 10)))
def test_snv_nunca_produz_nan(X):
    assert np.isfinite(snv(X)).all()
```

**Adicionar `hypothesis` ao projeto.** É o teste que encontra o bug que ninguém pensou.

**Critério de aceite:** CI falha se a cobertura do núcleo científico cair abaixo de 95%.

---

### ✅ P5 (RESOLVIDO) — `joblib.load` é execução remota de código
`predicao.py` carrega modelos `.joblib`. Pickle **executa código arbitrário** ao carregar.
Se um usuário baixar um `.joblib` de terceiro, é RCE. Isso pode barrar a revisão JOSS.

**Solução:**
```python
def carregar_modelo(path, confiar=False):
    """Carrega modelo treinado.

    AVISO DE SEGURANÇA: .joblib usa pickle, que EXECUTA CÓDIGO ARBITRÁRIO
    ao ser carregado. Carregue apenas modelos que você treinou ou de origem
    confiável. Ver docs/SECURITY.md
    """
    if not confiar:
        raise SecurityError(
            f"Carregar {path} executa código contido no arquivo. "
            "Passe confiar=True apenas se confia na origem. Ver docs/SECURITY.md"
        )
    ...
```

**Salvar manifesto junto do modelo** — resolve segurança E reprodutibilidade de uma vez:
```python
manifesto = {
    "guaraci_version": __version__,
    "sklearn_version": sklearn.__version__,
    "numpy_version": np.__version__,
    "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    "treinado_em": datetime.now(timezone.utc).isoformat(),
    "n_amostras": len(y),
    "classes": sorted(map(str, classes)),
    "config_hash": hash_config(cfg),
}
```

**Criar `SECURITY.md` na raiz.** Todo projeto levado a sério tem um.

---

### 🟢 P6 (PARCIAL, 2026-07-13) — 164 `print()` em vez de `logging`
**Feito:** `pipeline.py` migrado por completo para `log.info()`
(`src/guaraci/log.py` novo, handler que escreve em `sys.stdout` NO
MOMENTO do emit — não uma referência capturada na importação — para
continuar funcionando dentro do `contextlib.redirect_stdout` do CLI/app
web). Verificado com teste de integração dedicado
(`tests/test_pipeline_log_parsing_integration.py`) que roda o pipeline
sintético de verdade e confirma que os 3 regex do painel
(`_RE_ETAPA`/`_RE_ARQUIVO_SALVO`/`_RE_AVISO` em `app_logic.py`) ainda
casam com o texto capturado.

**NÃO feito (escopo maior, ver abaixo):** o painel do CLI e do app web
**continuam fazendo parsing de texto por regex**, não consumindo
`logging.Handler`/registros estruturados como a solução completa abaixo
propõe. `log.info(msg)` preserva o MESMO texto que `print(msg)` produzia
— resolve a inconsistência entre módulos e dá logger/verbosidade
configurável, mas **não** resolve a fragilidade de fundo ("mude uma
string e o painel quebra silenciosamente") — só relocou onde o texto
precisa ser preservado (de `print()` para `log.info()`). A reescrita do
painel para consumir registros estruturados (`extra={...}`) é um projeto
à parte, não feito nesta rodada.

**Solução:**
```python
# src/guaraci/log.py — ÚNICO ponto de configuração
import logging, sys

def configurar(nivel="INFO", arquivo=None):
    root = logging.getLogger("guaraci")
    root.setLevel(nivel)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s | %(message)s")
    h = logging.StreamHandler(sys.stderr); h.setFormatter(fmt); root.addHandler(h)
    if arquivo:
        fh = logging.FileHandler(arquivo); fh.setFormatter(fmt); root.addHandler(fh)
    return root
```
```python
# em cada módulo
import logging
log = logging.getLogger(__name__)   # vira "guaraci.pipeline", "guaraci.figuras"...

# antes:  print(f"Rodando PLS-DA com {n} LVs...")
log.info("PLS-DA iniciado", extra={"n_lvs": n, "n_amostras": len(y)})
```
```python
# O painel do CLI passa a usar HANDLER, não regex:
class PainelHandler(logging.Handler):
    def __init__(self, painel):
        super().__init__(); self.painel = painel
    def emit(self, record):
        self.painel.atualizar(record.getMessage(), record.levelname)
```

**Bônus (ainda não aproveitado):** um `arquivo=` em `guaraci.log.configurar()`
já permite log de execução em disco — ninguém liga isso ainda por padrão.

**Migração:** `pipeline.py` (164 prints) migrado em 2026-07-13 — decisão
tomada de fazer JUSTO o módulo maior primeiro (contrariando a ordem
sugerida aqui) porque era o único item concreto pedido; substituição
`print(` → `log.info(` foi mecânica mas verificada (não um `sed` cego às
cegas): confirmado antes por grep que as 164 chamadas eram todas de
argumento único (sem `flush=`/`sep=`/múltiplos args posicionais), e
depois por um teste de integração dedicado que roda o pipeline de
verdade e confere que os regex do painel ainda casam. Os demais módulos
já usavam `logging` desde antes desta sessão.

---

### 🟡 P7 — Instalação e primeira experiência
Ninguém clona repositório para experimentar software. **As pessoas fazem `pip install`.**
Se `pip install guaraci` não funciona hoje, essa é a barreira nº 1 de adoção — mais
importante que qualquer refatoração.

**Teste dos 5 minutos — em VM limpa, cronometrado:**
```bash
pip install guaraci        # ou guaraci[web]
guaraci demo               # ← esse comando existe? Deveria.
# → roda com dados embutidos, gera figuras, abre a pasta. Sem o usuário ter dado nenhum.
```

**Checklist:**
- [ ] Publicado no PyPI
- [ ] Extras: `[web]`, `[reports]`, `[benchmark]`, `[imagem]`, `[all]`
- [ ] Dataset de demo **embutido no pacote** (~1 MB, licença livre)
- [ ] `guaraci demo` — fluxo completo sem dado do usuário
- [ ] `guaraci doctor` — checa deps, RAM, versões, escreve diagnóstico
- [ ] `guaraci --version`
- [ ] CI testa instalação em Linux + Windows + macOS, py3.10–3.13
- [ ] **Notebook Colab "Guaraci em 5 minutos"** ← maior alavancagem de adoção da lista.
      Um link que o professor manda para a turma e 40 pessoas rodam no navegador.

**Pins vs. faixas — os dois, com papéis diferentes:**
- `pyproject.toml` → **faixas** (para ser instalável junto com outros pacotes)
- `requirements-lock.txt` → **pins exatos** (`pip-compile` / `uv pip compile`)

O lock é o que se cita no TCC e no paper:
> "Resultados obtidos com Guaraci v31.2.0 no ambiente descrito em `requirements-lock.txt`."

Isso é reprodutibilidade real, não retórica.

---

### ✅ P8 (RESOLVIDO em 2026-07-13) — Vocabulário N1/N2/N3 vs. Objetivo

> Menus principais lideram com o nome amigável (`_rotulo_opcao`,
> `guaraci.py:583`), tooltip do assistente "G" corrigido para a mesma
> convenção, e o residual (nomes de pasta `PLSDA_OE_N2_...` → agora
> `PLSDA_OE_Autenticacao_...`, e arquivo `figN3_heatmap_...` → agora
> `fig_heatmap_...`) também corrigido — com aprovação explícita do autor,
> já que era mudança de formato de saída. `cfg.nivel` continua "N1"/"N2"/
> "N3" internamente (config.yaml, lógica); só o SLUG usado no nome de
> pasta/arquivo mudou (`_NIVEL_SLUG_PASTA`, `config.py`). Documentação
> (README, MANUAL) e testes (`test_gerar_nome_saida_contem_nivel_e_preproc`,
> `test_heatmap_gera_png_valido`) atualizados.
Duas linguagens vivas no projeto. Ninguém de fora entende "N1/N2/N3".

**Solução:** aposentar N1/N2/N3 da UI e de todo documento externo. Manter como apelido
interno se quiser. Uma tabela de equivalência no MANUAL, e nada mais.

| Interno | Objetivo na UI | O que faz |
|---|---|---|
| N1 | Exploratório | PCA, HCA, loadings |
| N2 | Classificação / Autenticação | PLS-DA, OPLS-DA, DD-SIMCA |
| N3 | Quantificação | Regressão PLS, teor de adulterante |

Esforço: 2 horas. Impacto desproporcional — é o tipo de coisa que faz um revisor pensar
"esse projeto foi feito para outra pessoa usar".

---

### ⚪ P9 — `executar()` com 1269 linhas — **NÃO FAÇA AINDA**
Você (Claude) vai querer refatorar isso. **Não refatore antes da rede de segurança existir.**

Refatorar código com cobertura insuficiente é trocar o motor com o carro andando. Hoje
`executar()` só tem teste end-to-end. Se você quebrar em fases e um efeito colateral sumir,
o teste end-to-end pode **continuar passando** (os PNGs são gerados) enquanto os *valores*
dentro deles mudaram silenciosamente.

**Ordem obrigatória:**

**1º — Golden test de VALORES (não de PNGs):**
```python
def test_regressao_valores_pipeline(tmp_path):
    """Golden test: valores numéricos não mudam sem justificativa explícita."""
    res = executar(cfg_demo, outdir=tmp_path)
    esperado = json.loads(Path("tests/golden/demo_v31.json").read_text())
    for chave in ["r2_cv", "rmsecv", "n_lvs", "acuracia", "vip_top10", "t2_limite"]:
        np.testing.assert_allclose(
            res[chave], esperado[chave], rtol=1e-9,
            err_msg=f"REGRESSÃO NUMÉRICA em '{chave}' — investigar antes de commitar"
        )
```

**2º — Extrair funções PURAS primeiro** (só calculam, não salvam nada). Seguras de mover.

**3º — Só então quebrar em fases:**
```python
def executar(cfg):
    ctx = Contexto.de_config(cfg)          # dados + estado
    ctx = _fase_carregamento(ctx)
    ctx = _fase_preprocessamento(ctx)
    plano = resolver_objetivo(cfg)
    if plano.exploratorio:  ctx = _fase_exploratoria(ctx)
    if plano.classificacao: ctx = _fase_classificacao(ctx)
    if plano.quantificacao: ctx = _fase_quantificacao(ctx)
    _fase_figuras(ctx, plano)
    _fase_relatorios(ctx, plano)
    return ctx.resultados
```

**Impacto na ciência: zero.** Impacto na manutenção futura: alto.
**Portanto: só depois do TCC.** Não é prioridade. Não sugira fazer isso quando eu pedir
"o que fazer agora".

---

## 4. DOCUMENTAÇÃO — QUATRO DOCUMENTOS, QUATRO PÚBLICOS

Erro comum: um MANUAL gigante único. Separe:

| Documento | Público | Pergunta que responde | Tamanho |
|---|---|---|---|
| `README.md` | Chegou pelo GitHub | "O que é isso e devo me importar?" | 1 tela |
| `docs/QUICKSTART.md` | Decidiu tentar | "Como rodo em 5 minutos?" | 2 páginas |
| `docs/MANUAL.md` | Vai usar de verdade | "Como faço X? O que significa Y?" | Longo, navegável |
| `docs/VALIDATION.md` | Revisor / cético | "Por que devo confiar nesses números?" | Tabelas |
| `docs/SECURITY.md` | Revisor JOSS | "É seguro carregar um modelo de terceiro?" | 1 página |

### 4.1 `VALIDATION.md` — o cartão de visita técnico
A validação contra referências **já existe**, mas está enterrada no diretório de testes.
Ninguém vê. Isso é o que faz um pesquisador confiar — **não** uma nota de auditoria.

```markdown
# Validação contra implementações de referência

| Método | Referência | Métrica | Resultado | Teste |
|---|---|---|---|---|
| PLS-DA | sklearn PLSRegression + argmax | max\|Δcoef\| | 0.0 | `test_plsda_vs_sklearn` |
| DD-SIMCA | Pomerantsev & Rodionova (2014) | max\|ΔT²\| | 1.2e-14 | `test_ddsimca_vs_ref` |
| SNV | Barnes et al. (1989) | max\|Δ\| | < 1e-15 | `test_snv_analitico` |
| VIP | Propriedade Σ VIP² = p | \|Σ − p\| | < 1e-12 | `test_vip_soma` |
| MSC | Stateful (ref. do treino) | — | verificado | `test_msc_stateful` |
| Bootstrap BCa | Efron & Tibshirani (1993) | cobertura em MC | 94.8% (nominal 95%) | `test_bca_cobertura` |

Cada linha corresponde a um teste executado no CI a cada commit.
Reproduzir: `pytest tests/ -k validacao -v`
```
**Preencha com os valores REAIS obtidos rodando os testes. Não copie os números acima.**

### 4.2 A seção que falta no MANUAL: **"Como INTERPRETAR cada figura"**
Não "como gerar" — **como ler**. É isso que transforma o software em ferramenta de ensino,
e ferramenta de ensino é o caminho mais rápido para adoção.

Nível de detalhe exigido:
```markdown
### Gráfico de scores PCA

**O que mostra:** cada ponto é uma amostra. Distância ≈ similaridade espectral.

**Como ler:**
- Grupos separados = classes têm espectros distinguíveis. Bom sinal.
- Grupos sobrepostos = PCA não separa. Isso NÃO significa que a classificação
  vai falhar — PCA é não-supervisionado. Vá para o PLS-DA.
- Ponto isolado = possível outlier. Confira no Hotelling T² antes de excluir.

**Armadilha comum:** interpretar "% de variância explicada" como "% de informação
relevante". PC1 pode explicar 95% da variância e ser puro espalhamento físico —
por isso SNV/MSC existem.

**Se PC1 sozinho explica >90%:** você provavelmente esqueceu o pré-processamento
de correção de espalhamento.
```

### 4.3 A seção que ninguém escreve e todo mundo respeita: **Limitações**
Contraintuitivo mas verdadeiro: declarar limites **aumenta** a confiança no software.

```markdown
## Limitações conhecidas

- **DD-SIMCA com n<10 amostras puras:** sensibilidade tem incerteza alta.
  Interpretar como exploratória.
- **Regressão agrupando espécies:** não funciona (R²≈0). A matriz vegetal domina
  o sinal. Use o modo espécie×adulterante.
- **Modo imagem (colorimetria):** protótipo. Não validado com dataset real.
  Não use para resultado publicável.
- **Validado apenas em FT-NIR.** O motor é agnóstico, mas MIR e Raman não foram
  testados com dado real. Relatos de uso são bem-vindos.
- **Classes desbalanceadas:** Babaçu e Açaí têm recall menor. Reporte a matriz de
  confusão completa, não só a acurácia.
- **68 espectros descartados por faixa espectral** — merecem análise, não só descarte.
```

### 4.4 Documentação hospedada, não Markdown no GitHub
`docs/MANUAL.md` no repo não indexa bem, não tem busca. Use **MkDocs Material** +
GitHub Pages. 30 minutos de setup, muda a percepção de profissionalismo.
```yaml
# mkdocs.yml
site_name: Guaraci
theme: {name: material, language: pt-BR}
plugins: [search, mkdocstrings]   # gera API docs dos docstrings automaticamente
```

---

## 5. FIGURAS QUE FALTAM — ✅ TODAS ENTREGUES (verificado 2026-07-12)
As 4 abaixo já existem em `figuras.py` e são chamadas por `executar()` — não
são mais um item de roadmap. Mantido aqui só como registro do que foi pedido
e resolvido:
1. ~~Heatmap espécie × adulterante (R²cv)~~ — `fig_heatmap_especie_adulterante`, ver P2.
2. ~~Curva RMSECV × nº de LVs~~ — `fig1_selecao_lvs`.
3. ~~Espectros médios por classe com banda de ±1 desvio~~ — `fig_espectros_medios_classe`.
4. ~~Biplot PCA~~ (scores + loadings sobrepostos) — `fig_biplot_pca`.

---

## 6. UX — O QUE ESTÁ ERRADO

> Atualizado em 2026-07-13.

- **Densidade de configuração.** ✅ **Resolvido.** 3 presets por objetivo
  científico — **Explorar Dados / Autenticar Pureza / Quantificar Teor**
  (nomes sem sufixo N1/N2/N3, P8) — implementados em `PROFILES`
  (`cli_assistente.py`), reaproveitando a mesma infraestrutura dos perfis de
  rigor já existentes (`menu_perfis`, CLI) e espelhados na aba Dados do app
  web (`app_tabs/dados.py`, botões 🔍/🛡️/📊). Aplicar um preset já ajusta
  `nivel`+`objetivo`+módulos pesados de uma vez.
  **Modo Iniciante/Avançado** implementado com o design híbrido pedido:
  toggle global `[M]` no menu principal (persistido em `.cli_modo_usuario`,
  mesmo padrão do idioma) + revelação LOCAL `[V]` por submenu (não muda o
  modo da sessão, só expande aquela visita). Implementado em
  `_print_submenu_compact` (retorna a lista de campos REALMENTE exibidos —
  chamador usa essa lista para indexação numérica, não o `fields` original)
  e propagado para `_loop_menu` (usado por `menu_modelagem`) e o loop
  próprio de `menu_validacao`. Campos escondidos por padrão:
  `menu_modelagem` → `opls_da`/`ddsimca`/`modo_ddsimca`/
  `selecao_variaveis_etapa4` (nível N2 já força DD-SIMCA automaticamente,
  então o beginner não precisa tocar nesses); `menu_validacao` →
  `n_permutacoes`/`teste_wold`/`teste_cv_anova` (testes estatísticos extras,
  tuning fino). **Não tocado, de propósito:** `menu_preproc` (só 2 campos,
  não vale a pena), `menu_avancado` (já é uma seção separada só de módulos
  pesados — a própria existência da seção já cumpre a função), `menu_tecnica`/
  `menu_codificacao` (não são listas de hiperparâmetro — são escolha de
  técnica analítica e consulta de código de espécie, não se encaixam no
  conceito "esconder avançado").
- **Cliques até resultado.** ✅ **Resolvido.** No CLI, `menu_perfis` agora
  pergunta "Rodar agora com essa configuração?" logo após aplicar qualquer
  perfil (`_rodar_pipeline(cfg)` direto, sem passar pelo menu principal). No
  app web, aplicar um preset na aba Dados já configura `nivel`/`objetivo`;
  falta só ir na aba Model e clicar "Run pipeline" (não há um botão de
  "rodar" cross-tab no Streamlit por design — cada aba é independente).
- **Mensagens de erro** poderiam linkar direto ao campo problemático.
- **Falta tour guiado / tooltip contextual** dentro do app.

**Achado técnico não previsto no escopo original (✅ resolvido):** o app
Streamlit tinha um bug de sincronização de estado — widgets com `key=`
estático (`"w_"+campo`) só honram `value=`/`index=` na primeira
renderização; depois disso, seguem mostrando o valor antigo mesmo que
`cfg_base` mude, **a menos que o valor seja escrito diretamente em
`st.session_state[key]`** (apagar a chave não basta — verificado
empiricamente). Corrigido nos presets novos E no botão pré-existente
"↺ Reload config.yaml" (`app_tabs/dados.py`), via helper compartilhado
`_sincronizar_widgets_com_cfg()` — commit `73bb522`.

---

## 7. ADOÇÃO — O QUE O RELATÓRIO DE AUDITORIA NÃO DISSE

Qualidade de código **não** gera adoção. Adoção em ciência vem de: citação, dependência
de terceiros, e presença em cursos. mdatools (R) e hyperSpec já ocupam esse espaço.

### 7.1 Rodar em um dataset público que não é o meu
Enquanto o único dataset for o do autor, o software é "o TCC do Erley". No dia em que
alguém rodar em dado diferente, vira software.

**Ação:** rodar o Guaraci num dataset NIR público e clássico (ex.: Tecator, corn dataset
da Eigenvector, trigo NIR). Publicar como notebook. **Comparar com os resultados
publicados na literatura para aquele dataset.**

Isso vale mais que 10 pontos de cobertura. É a prova de que funciona fora da máquina do autor.

### 7.2 JOSS — o revisor vai cobrar exatamente três coisas que hoje faltam
1. **State of the field** — tabela comparando com mdatools, hyperSpec, pyChemometrics,
   scikit-learn. O que o Guaraci faz que eles não fazem.
   **Resposta honesta: validação group-aware nativa.** Esse é o argumento. É bom.
2. **Statement of need afiado** — não "quimiometria é cara", mas:
   *"Réplicas físicas causam vazamento em validação cruzada e nenhum pacote atual
   protege contra isso por padrão."*
3. **Exemplo de uso executável** dentro do paper.

### 7.3 O nome tem um custo
"Guaraci" tem raiz cultural e é defensável. Mas um pesquisador alemão ou coreano não
lembra, não pronuncia, não acha no Google. **Isso tem preço.** Se o preço for aceito
(legítimo — identidade importa), **compense com SEO**: o README precisa ter
"chemometrics", "PLS-DA", "SIMCA", "NIR authentication", "adulteration detection"
nas primeiras linhas em inglês.

---

## 8. ORDEM DE EXECUÇÃO

**Status em 2026-07-13 — itens ✅ concluídos nesta sessão (pós-auditoria de 15 etapas):**
| # | Item | Prazo | Bloqueia |
|---|---|---|---|
| ~~1~~ | ~~P1 — Sensibilidade LOGO~~ ✅ | feito 2026-07-11 | Defesa |
| ~~2~~ | ~~P2 — Heatmap espécie×adulterante nativo~~ ✅ | feito 2026-07-11 | Defesa |
| ~~3~~ | ~~P8 — Vocabulário N1/N2/N3 (menus + tooltip)~~ ✅ parcial | feito 2026-07-13 — pastas/arquivos ainda vazam N1/N2/N3, ver P8 acima | Clareza |
| ~~6~~ | ~~P5 — `SECURITY.md` + guarda no joblib~~ ✅ | feito | JOSS |
| ~~9~~ | ~~Curva RMSECV × LVs + espectros médios + biplot~~ ✅ | feito, ver seção 5 | Banca vai perguntar |
| — | **Viés de seleção de variáveis (Etapa 4, achado da auditoria 2026-07-12)** ✅ | corrigido 2026-07-13 — nested-CV para VIP/SR/sPLS-DA (`selecao_variaveis.py`) | **Correção científica** |
| — | **Submenu Visualização quebrado (H/M/B/V)** ✅ | corrigido 2026-07-13 — opções removidas (funções nunca existiram) | Robustez |

**Ainda pendente antes do TCC (bloqueiam defesa ou credibilidade):**
| # | Item | Prazo | Bloqueia |
|---|---|---|---|
| 8 | P7 — publicar no PyPI (`guaraci demo`/`doctor`/Colab já prontos) | depende de conta do autor | **Adoção** |
| — | **Rodar o pipeline atual (pós-correções 07-13) contra o dataset real do TCC** | só o autor pode (dataset fora do checkout) | Defesa — números antigos citados na monografia não refletem mais o código |

~~4~~ ~~`docs/VALIDATION.md` — nota sobre nested-CV da Etapa 4~~ ✅ feito — ver seção "AG e SPA (Etapa 4, opcionais)" em `VALIDATION.md`.
~~5~~ ~~Seção "Limitações" no MANUAL~~ ✅ já existia (seção 9) e cobre DD-SIMCA/regressão agrupada/modo imagem/FT-NIR-only/joblib/`mae_id` órfão.
~~7~~ ~~P3 — 51 `except` restantes~~ ✅ concluído em essência, 100% justificados.

**Depois do TCC:**
| # | Item | Prazo |
|---|---|---|
| ~~10~~ | ~~P4 — Cobertura núcleo → 95%~~ ✅ | feito 2026-07-13 — os 4 módulos do núcleo, todos ≥95% |
| ~~11~~ | ~~P6 — `print` → `logging` em `pipeline.py`~~ ✅ parcial | feito 2026-07-13 — falta reescrever o painel p/ Handler estruturado (ver P6) |
| ~~12~~ | ~~MkDocs + GitHub Pages~~ ✅ | feito 2026-07-13 — `mkdocs.yml` (tema Material + mkdocstrings no núcleo), `docs/index.md`, `.github/workflows/docs.yml` (build `--strict` + deploy); falta só habilitar a fonte "GitHub Actions" em Settings → Pages do repo (ação de configuração do GitHub, feita uma vez pelo dono) |
| ~~13~~ | ~~Rodar em dataset público externo (Tecator)~~ ✅ | feito 2026-07-13 — `docs/BENCHMARK_TECATOR.md`; falta um 2º dataset (corn) e cobertura de classificação/DD-SIMCA externa |
| ~~14~~ | ~~Paper JOSS: state of the field~~ ✅ | feito 2026-07-13 — nova seção comparando com mdatools/hyperSpec/pyChemometrics/scikit-learn, citações verificadas por busca antes de escrever (não geradas de memória) |
| ~~—~~ | ~~Renomear pastas/arquivos de saída para tirar N1/N2/N3~~ ✅ | corrigido 2026-07-13, com aprovação explícita — ver P8 |
| ~~—~~ | ~~Viés de seleção do AG/SPA~~ ✅ | corrigido 2026-07-13 — nested-CV via `_avaliar_busca_nested_cv` |
| ~~—~~ | ~~3 presets Autenticar/Explorar/Quantificar + "rodar recomendada"~~ ✅ | corrigido 2026-07-13 (CLI + app web) — ver seção 6 |
| ~~—~~ | ~~Modo Iniciante/Avançado (esconder campos avançados)~~ ✅ | feito 2026-07-13 — ver seção 6 (toggle `[M]` global + revelação local `[V]`) |
| ~~—~~ | ~~Bug do botão "↺ Reload config.yaml" (`app_tabs/dados.py`)~~ ✅ | corrigido 2026-07-13 (commit `73bb522`) — mesmo desync de widget `key=` dos presets, resolvido via `_sincronizar_widgets_com_cfg()` compartilhado — ver seção 6 |

**Quando sobrar tempo:**
| 15 | P9 — Golden tests + quebrar `executar()` | 3–4 semanas |
| 16 | Quebrar `guaraci.py` (3133 linhas) | — |
| 17 | Type-checker (mypy/pyright) só nos módulos puros | — |
| 18 | Validar modo imagem com dataset real | Precisa de dado |

---

## 9. O ITEM MAIS SUBESTIMADO
Auditorias anteriores tratam os **114 `except Exception`** como dívida de *estilo*.
**São dívida de CORREÇÃO.** Um deles pode estar engolindo um erro numérico e produzindo
um R² errado no TCC agora mesmo, sem nenhuma forma de saber.

Depois da sensibilidade LOGO, é por aí que se começa.

---

## 10. COMANDOS DO PROJETO

```bash
# Testes
pytest -q
pytest --cov=src/guaraci --cov-report=term-missing
pytest -k validacao -v            # só os testes contra referência
pytest -m "not slow"              # rápido, durante desenvolvimento

# Lint
ruff check .
ruff format .

# Execução
python -m guaraci.pipeline --config exemplo.yaml
guaraci                           # CLI Rich interativo
streamlit run app_quimiometria.py # Web

# Auditoria rápida do estado
wc -l src/guaraci/*.py | sort -n
grep -rn "except Exception\|except:" src/guaraci/ | wc -l
grep -c "print(" src/guaraci/pipeline.py
grep -rn "TODO\|FIXME\|HACK" src/guaraci/
```

---

## 11. CHECKLIST ANTES DE QUALQUER COMMIT
- [ ] `pytest -q` — nº de testes passando não caiu
- [ ] `ruff check .` — limpo
- [ ] Cobertura não caiu (`--cov`)
- [ ] Nenhum `print()` novo (use `log`)
- [ ] Nenhum `except Exception` novo sem `noqa` + comentário justificando
- [ ] Nenhum número inventado em docs — todo valor foi rodado e verificado
- [ ] Toda referência bibliográfica é real e verificável
- [ ] O diff faz **uma** coisa
- [ ] Se o commit muda `version` em `pyproject.toml`/`config.py`, criar e enviar
      a tag Git correspondente (`git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`)
      **no mesmo momento** — nunca deixar acumular bumps sem tag. A aba Releases do
      GitHub é a linha do tempo pública do projeto; um salto (ex.: v31.1.0 → v31.8.0
      sem nada no meio) é confuso para quem audita o histórico de fora.
