# Validação pública — a única base de evidência do software

Desde 2026-08-18 o GUARACI é validado exclusivamente em datasets públicos.
Nenhuma métrica desta página vem de dado privado: todas são reproduzíveis
por qualquer pessoa a partir dos datasets citados.

Reproduzir: `pytest tests/test_validacao_publica.py` com
`GUARACI_DATASETS_DIR` apontando para a pasta que contém os arquivos
(o CI faz o download no job `validacao-publica`).

---

## 1. Tabela consolidada

| Dataset | Matriz | n | Canais / faixa | Alvo | Métrica obtida | Referência da literatura | Estado |
|---|---|---:|---|---|---|---|---|
| Eigenvector **Corn** (m5) | milho em grão | 80 | 700 · 1100–2498 nm | proteína | **RMSEP 0,144 %m/m**; R²val 0,912; 8 LVs | RMSEP típico de PLS: **0,1–0,2** | ✅ dentro da faixa |
| **Tecator** | carne moída | 240 | 100 · 850–1050 nm | gordura | RMSEP 2,001 (`autoscaling`) | ver `docs/BENCHMARK_TECATOR.md` | ✅ dentro do esperado |
| **Mel adulterado** (478 × 700, 4 classes) | mel | — | — | puro vs. 3 xaropes | — | — | ❌ **NÃO OBTIDO** |
| Mendeley `10.17632/ctgg7k4m5g.2` | azeite | 100 | NIR + MIR + Raman | adulteração/PV | — | — | 🟡 **ACESSÍVEL, não integrado** (reconfirmado 2026-08-26) |

O RMSEP do Corn está no meio da faixa publicada — nem baixo demais (o que
sugeriria vazamento) nem alto demais (bug de pré-processamento). É esse
resultado que sustenta a afirmação "o motor está correto", e é o único tipo
de afirmação que este repositório pode fazer sobre desempenho.

---

## 2. Estado exato dos datasets ainda não integrados

### Mendeley `10.17632/ctgg7k4m5g.2` (azeite adulterado) — RETRATAÇÃO de 2026-08-18

> **A entrada anterior desta seção afirmava "❌ NÃO OBTIDO" com a rota
> `?version=2` retornando HTTP 403 e exigindo sessão de navegador.
> Reconfirmado por comando direto em 2026-08-26 (Passo 78): o endpoint
> SEM o parâmetro `?version=2` funciona e nunca exigiu sessão.** A
> tentativa de 2026-08-18 usou a rota errada (`.../files?version=2`,
> que de fato devolve erro) em vez do endpoint correto de metadados do
> dataset. Não se sabe se o endpoint mudou de comportamento ou se a
> investigação original simplesmente não tentou essa rota — o registro
> anterior não detalha todas as variações testadas o suficiente para
> distinguir os dois casos.

| Tentativa (2026-08-18) | Resultado |
|---|---|
| `data.mendeley.com/public-api/datasets/ctgg7k4m5g/files?version=2` | **HTTP 403** (reconfirmado 2026-08-26: `{"error":400}`, ainda bloqueado) |
| `api.data.mendeley.com/datasets/ctgg7k4m5g/2` | **HTTP 404** (reconfirmado 2026-08-26: ainda 404) |
| `api.data.mendeley.com/datasets/ctgg7k4m5g/2/files` | **HTTP 404** (reconfirmado 2026-08-26: ainda 404) |
| `doi.org/api/handles/10.17632/ctgg7k4m5g.2` | **200** — o DOI resolve, o dataset existe |
| Página de destino (`data.mendeley.com/datasets/ctgg7k4m5g/2`) | **200**, 125 KB — SPA, links montados por JS |

| Tentativa NOVA (2026-08-26) | Resultado |
|---|---|
| `data.mendeley.com/public-api/datasets/ctgg7k4m5g` (SEM `?version=2`) | **HTTP 200**, JSON completo: metadados + lista de 10 arquivos + `download_url` por arquivo (`ATRAdulteration3.csv`, `ATRPure3.csv`, `MIR1A.csv`, `NIR24mm1A.csv`, `NIR24mm1B.csv`, `NIR2mm1B.csv`, `NIR8mm1A.csv`, `OilClassKey.csv`, `Raman1A.csv`, `Raman2.csv`; ~129,5 MB total) |
| `HEAD` no `download_url` do menor arquivo (`OilClassKey.csv`, 401 bytes) | **302** → redireciona para URL assinada S3 (`prod-dcd-datasets-public-files-eu-west-1.s3...`) — download real, sem autenticação, confirmado funcional |

**Licença confirmada no JSON da API (não só no HTML): CC BY 4.0**
(`data_licence.short_name`) — compatível com uso e redistribuição
mediante atribuição.

**Estado real agora: ACESSÍVEL programaticamente, NADA baixado nem
integrado ao pipeline ainda** — esta verificação foi só de
acessibilidade (Passo 78 pede levantamento, não integração). Nome do
alvo científico do dataset (peroxide value / classificação, não
"adulteração" propriamente — ver descrição completa do dataset) precisa
ser conferido contra o objetivo real do GUARACI antes de decidir como
integrar. Para integrar de fato: baixar os 10 CSVs pelos `download_url`
acima, decidir formato de carregamento (provavelmente novo leitor em
`io_registry.py`, os arquivos não são `.dx`), e apontar
`GUARACI_DATASETS_DIR` — trabalho de um Passo próprio, não feito aqui.

### Mel adulterado (478 amostras, 700 comprimentos de onda, 4 classes)

Nenhuma fonte verificável foi localizada com essas características exatas.
**Não foi substituído por um dataset de mel qualquer**: a alegação a
validar é o requisito multimatriz com *n* adequado para classificação
puro vs. adulterado, e um dataset diferente não a sustenta.

Consequência assumida: o perfil `mel_vis_nir` existe e é carregável, mas
está marcado no próprio YAML como **declarado, não validado com dado real**.
O requisito multimatriz foi provado com outro par de matrizes (§3).

---

## 3. Prova do requisito multimatriz

`tests/test_perfil_matriz.py::test_aceitacao_multimatriz_milho_e_oleo_sem_tocar_em_codigo`

Roda o pipeline completo em duas matrizes de naturezas diferentes — uma em
**nm**, outra em **cm⁻¹**, com vocabulários distintos — alterando **um único
campo de configuração** (`cfg.matrix_profile`). Verifica que:

- cada model card declara a sua matriz (`milho em grao` / `oleo vegetal`);
- **nenhum declara a da outra** — este era exatamente o defeito medido em
  2026-08-17, quando o card do milho afirmava "óleo vegetal amazônico";
- o vocabulário de classe acompanha (`variedade` / `especie`);
- o perfil usado fica registrado no card, para quem o lê depois.

E `test_perfil_inexistente_aborta_o_pipeline_antes_de_predizer`: matriz sem
perfil cadastrado levanta `UnknownProfileError` **antes** de qualquer
predição, com a lista de perfis disponíveis e a instrução de como escrever
um novo. Nunca cai num padrão de outra matriz em silêncio.

---

## 4. Licenças dos datasets

| Dataset | Licença / termos | Verificado em |
|---|---|---|
| Eigenvector Corn | distribuído publicamente pela Eigenvector Research para benchmarking; ver a página da fonte para os termos | página da fonte, 2026-08-17 |
| Tecator | domínio público (StatLib) | `docs/BENCHMARK_TECATOR.md` |
| Mendeley `ctgg7k4m5g` | **CC BY 4.0** | HTML da página do dataset, 2026-08-18 |

Nenhum destes arquivos é versionado neste repositório.

---

## 5. Métricas de quantificação agora reportadas

A tabela do §1 só é interpretável porque o RMSEP vem acompanhado. Desde
esta rodada o pipeline reporta, para toda regressão:

| Métrica | Onde | Observação |
|---|---|---|
| RMSEC / RMSECV / RMSEP | log, `resumo_modelo.txt` | já existiam |
| R²cal / R²val / bias | idem | já existiam |
| **SEP** | idem | erro-padrão de predição corrigido pelo bias |
| **RPD** | idem | `SD(y_ref) / SEP`, **com a faixa de uso ao lado** |
| **RER** | idem | `amplitude(y_ref) / SEP` |
| LOD / LOQ / SEN / seletividade | `figS3_merito_regressao` | Valderrama, Braga & Poppi (2009); exige réplicas físicas |

RPD e RER nunca saem nus: `interpret_rpd()` anexa a faixa publicada
(Williams 2014, em Williams, Dardenne & Flinn, *J. Near Infrared
Spectrosc.* 22(2):85-93; AACC 39-00.01). Um número cru convida a
comparações indevidas entre estudos; a faixa carrega a referência que a
define.

Nota honesta sobre **LOD/LOQ**: eles são calculados, mas dependem de
réplicas físicas para estimar o ruído instrumental. Em datasets sem
réplicas — o Corn é um deles — saem `N/A`, e é correto que saiam: um LOD
estimado sem base de repetibilidade seria um número inventado.
