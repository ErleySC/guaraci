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
| Mendeley `10.17632/ctgg7k4m5g.2` | azeite | — | NIR + MIR + Raman | adulteração | — | — | ❌ **NÃO OBTIDO** |

O RMSEP do Corn está no meio da faixa publicada — nem baixo demais (o que
sugeriria vazamento) nem alto demais (bug de pré-processamento). É esse
resultado que sustenta a afirmação "o motor está correto", e é o único tipo
de afirmação que este repositório pode fazer sobre desempenho.

---

## 2. NÃO OBTIDO — motivo exato

### Mendeley `10.17632/ctgg7k4m5g.2` (azeite adulterado)

| Tentativa | Resultado |
|---|---|
| `data.mendeley.com/public-api/datasets/ctgg7k4m5g/files?version=2` | **HTTP 403** |
| `api.data.mendeley.com/datasets/ctgg7k4m5g/2` | **HTTP 404** |
| `api.data.mendeley.com/datasets/ctgg7k4m5g/2/files` | **HTTP 404** |
| `doi.org/api/handles/10.17632/ctgg7k4m5g.2` | **200** — o DOI resolve, o dataset existe |
| Página de destino (`data.mendeley.com/datasets/ctgg7k4m5g/2`) | **200**, 125 KB — mas é uma SPA: os links de arquivo são montados por JavaScript e não estão no HTML |

**Licença confirmada no HTML da página: CC BY 4.0** — compatível com uso e
redistribuição mediante atribuição, caso venha a ser obtido.

O download exige sessão de navegador. **Não foi substituído por nenhum
proxy**: um dataset "parecido" de outra fonte não valida a mesma
alegação. Para integrá-lo, baixe manualmente e aponte
`GUARACI_DATASETS_DIR`; o carregador já aceita caminho de arquivo.

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
campo de configuração** (`cfg.perfil_matriz`). Verifica que:

- cada model card declara a sua matriz (`milho em grao` / `oleo vegetal`);
- **nenhum declara a da outra** — este era exatamente o defeito medido em
  2026-08-17, quando o card do milho afirmava "óleo vegetal amazônico";
- o vocabulário de classe acompanha (`variedade` / `especie`);
- o perfil usado fica registrado no card, para quem o lê depois.

E `test_perfil_inexistente_aborta_o_pipeline_antes_de_predizer`: matriz sem
perfil cadastrado levanta `PerfilDesconhecidoError` **antes** de qualquer
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

RPD e RER nunca saem nus: `interpretar_rpd()` anexa a faixa publicada
(Williams 2014, em Williams, Dardenne & Flinn, *J. Near Infrared
Spectrosc.* 22(2):85-93; AACC 39-00.01). Um número cru convida a
comparações indevidas entre estudos; a faixa carrega a referência que a
define.

Nota honesta sobre **LOD/LOQ**: eles são calculados, mas dependem de
réplicas físicas para estimar o ruído instrumental. Em datasets sem
réplicas — o Corn é um deles — saem `N/A`, e é correto que saiam: um LOD
estimado sem base de repetibilidade seria um número inventado.
