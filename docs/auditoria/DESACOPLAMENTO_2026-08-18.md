# Desacoplamento do dataset institucional — 2026-08-18

**Decisão executada:** os `.dx` do acervo cedido saem do escopo do
software. O GUARACI passa a ser validado exclusivamente em datasets
públicos; o TCC é entregável separado, com base de evidência própria.

Consequência aplicada em todo o repositório: nenhuma métrica obtida sobre
os `.dx` permanece em README, model card, paper, docs, código ou testes.

---

## 1. Vazamento por SAÍDA — o que era premissa e o que foi medido

A premissa de partida era que "qualquer pessoa que rode o software sobre
esses arquivos gera saída contendo operador e local". **Medido, isso é
falso** — e a parte verdadeira era outra.

`parse_dx` lê exatamente 9 campos do cabeçalho JCAMP
(`dados_io.py:546-556`): `XFACTOR`, `YFACTOR`, `FIRSTX`, `LASTX`,
`NPOINTS`, `XYDATA`, `XYPOINTS`, `END`, `TITLE`. **`AUDIT TRAIL` nunca é
lido**, nem `$Detector model`, nem `$Spectrometer model`. Operador e local
jamais entraram no fluxo de dados.

O que vazava era o `##TITLE` — que o parser *precisa* ler, porque é de onde
saem classe, teor e o agrupamento de réplicas. Ele existia em memória de
forma legítima e era **gravado em disco** por `metadados.csv`
(`pipeline.py:1204`), com as colunas `title_original`, `arquivo`, `cod`,
`data`, `mae_id`, `subpasta`.

### O que foi feito

| Ação | Onde |
|---|---|
| Teste de sanitização escrito **antes** da correção | `tests/test_sanitizacao_metadados.py` |
| `sanitizar_metadados()` — remove identificação, preserva o analítico | `dados_io.py` |
| `metadados.csv` passa a ser sanitizado antes de tocar o disco | `pipeline.py` |
| `grupo_replica` (`G000`, `G001`…) substitui `mae_id` no arquivo | `dados_io.py` |

O teste monta `.dx` **sintéticos** com ASDF válido e sentinelas no
`AUDIT TRAIL`, roda o pipeline de verdade e varre todos os artefatos
(`.txt`, `.csv`, `.md`, `.json`, `.tex`, `.log`) e os bytes do `.joblib`.
Roda no CI sem depender de dado local.

**Resultado da varredura, com o pipeline executado:**

| Campo | Vaza para artefato de texto? | Vaza para o `.joblib`? |
|---|---|---|
| Operador (`AUDIT TRAIL`) | não | não |
| Local (`AUDIT TRAIL`) | não | não |
| Modelo do detector | não | não |
| Nome do arquivo `.dx` | não | não |
| Identificador da amostra (`TITLE`) | **sim, antes** → não, agora | não |

`grupo_replica` existe por um motivo específico: sem ele o CSV perderia a
informação que sustenta a validação group-aware — o diferencial do projeto
— e ninguém de fora conseguiria auditar se as réplicas foram mantidas
juntas. Dois testes travam isso: réplicas do mesmo ponto continuam no mesmo
grupo, pontos distintos não se fundem, e o rótulo não carrega o
identificador original.

---

## 2. Identificadores de amostra versionados

Varredura no padrão `COD-DD-MM-AAAA[-Xteor]` sobre arquivos rastreados,
cruzada com os identificadores reais do acervo (a lista e a contagem
ficam em documentação local, fora deste repositório):

| | Distintos | Ocorrências |
|---|---:|---:|
| Identificadores versionados | 8 | 26 |
| **Correspondem a amostras reais** | **3** | **13** |
| Inventados (não existem no acervo) | 5 | 13 |

Os 3 reais foram substituídos por identificadores sintéticos de ano **2099**
— mesma estrutura, mesma lógica de parsing exercitada, colisão impossível
com o acervo. 13 ocorrências em 6 arquivos de `src/` e `tests/`, mais 3
documentos de auditoria (incluindo o relatório da rodada anterior, que
também os reproduzia). 169 testes das áreas afetadas passam depois da troca.

### Histórico publicado — registrado, não executado

Os identificadores reais aparecem em **3 commits**, e os três **já estão em
`origin/master`**:

| Commit | Situação |
|---|---|
| `2927b70` | publicado |
| `338c45f` | publicado |
| `e6384db` | publicado |

**Nenhuma reescrita foi executada.** Opções, para decisão:

- **(A) Não fazer nada.** São metadados de 2 pontos físicos, sem espectro
  junto. Exposição mínima, custo zero, risco zero de quebrar clones.
- **(B) Reescrever.** `git clone --mirror` para backup fora do projeto →
  `git filter-repo --replace-text` → force-push → **chamado ao suporte do
  GitHub** para purgar o cache de SHA (sem isso, os commits antigos
  continuam acessíveis pela URL do SHA). Fora do alcance disso: issues, PRs,
  wiki, logs de Actions, releases e o Software Heritage, que arquiva o
  GitHub automaticamente.

Recomendação: **(A)**. O ganho de (B) é marginal frente ao custo e ao risco,
e o resultado seria redução de exposição, nunca eliminação.

---

## 3. Métricas dos `.dx` removidas de artefatos públicos

| Onde estava | O que dizia | Ação |
|---|---|---|
| `docs/BENCHMARK_TECATOR.md:85` | métrica do dataset próprio | trocado por descrição do preset |
| `pipeline.py:15` (cabeçalho) | métrica do dataset próprio | removido |
| `pipeline.py:1124` (comentário) | métrica + nº de classes do dataset próprio | removido |
| `pipeline.py` (N2) | figuras de mérito do dataset próprio | removido |
| `config.py:126` | nº de amostras do dataset próprio | removido |
| `preprocessamento.py:144` | métrica do dataset próprio | removido |
| `app_tabs/preprocessamento.py:16` | métrica + nº de amostras, na UI web | removido |
| `guaraci.py:956` | métrica do dataset próprio, no CLI | removido |
| `classificadores.py:228` | nº de classes com uma única amostra pura | generalizado |
| `docs/CHANGELOG.md` (2 entradas) | métricas em log histórico | marcadas como retiradas, entrada preservada |

O CHANGELOG teve o número retirado mas a **entrada preservada com nota**:
apagar a linha falsificaria o log de uma versão que de fato existiu.

Varredura final pelas métricas e contagens obtidas sobre o acervo em
`src/`, `docs/`, `README*`, `paper/`, `app_quimiometria.py`: **zero
ocorrências**
(fora de `docs/auditoria/`, que é o registro da própria auditoria).

---

## 4. Dependência de dados institucionais

| Verificação | Resultado |
|---|---|
| Caminho institucional/pessoal em arquivo versionado | **nenhum** — só padrões em `.gitignore`/`.dockerignore` |
| `config.example.yaml` | `modo_entrada: sintetico` — não aponta para dado local |
| `Config` padrão | `pasta_entrada="dados"` (relativo, genérico); ausência falha com mensagem que sugere `modo='sintetico'` |
| CI | roda sobre sintético + o job novo de dataset público; zero dependência de dado local |
| Correções de metadado do acervo | já viviam fora da árvore, em `~/.guaraci_local/` |

**Ponto que fica em disco, fora do Git:** `resultados_tcc/` contém
`metadados.csv` gerados **antes** desta correção, com identificadores reais.
Está no `.gitignore`, então não é exposição no repositório — mas essas
pastas não devem ser compartilhadas como estão. Regerar com a versão atual
produz o CSV sanitizado.

---

## 5. O que continua no repositório, e por quê

| Item | Decisão |
|---|---|
| Afiliação do autor (link Lattes em `CITATION.cff`, README, `sobre.py`, `guaraci.py`) | **mantida** — decisão do usuário, não da auditoria |
| Perfil `oleo_nir` | mantido: descreve uma **matriz**, não um acervo. É conhecimento reutilizável sobre FT-NIR de óleos |
| `CODIGO_ESPECIE` / `ADULTERANTE_NOME` (`dados_io.py:35,43`) | mantidos por ora — são a convenção de nomenclatura do parser, não dados de amostra. Migrar para o perfil é trabalho aberto (ver §7) |
| Registro da proveniência (este documento) | **mantido** — a proveniência é documentada, nunca apagada; apagá-la seria misatribuição |

---

## 6. Créditos a terceiros — pendente de decisão do autor

Os metadados dos `.dx` registram, em **todos** os arquivos do acervo, o
local institucional e o operador de cada leitura: mais de um rótulo, ao
menos duas pessoas, nenhuma delas o autor. Isso é fato medido, não
declaração. As contagens exatas ficam em documentação local — publicá-las
dimensionaria quanto material de terceiro identificável existe.

Duas ações que dependem de você, e que a auditoria não pode tomar sozinha:

1. **`ACKNOWLEDGMENTS`** — quem operou o instrumento entra em
   agradecimentos, não em `AUTHORS` nem como coautor do software. Requer
   autorização das pessoas antes de nomeá-las.
2. **Sanitizar o `AUDIT TRAIL` antes de qualquer publicação dos brutos.**
   Um dos rótulos é um apelido informal e depreciativo aplicado a pessoa
   identificável, presente em 828 arquivos. Publicar os espectros sem
   sanitizar levaria isso junto.

---

## 7. Aberto

| Item | Por quê |
|---|---|
| `CODIGO_ESPECIE`/`ADULTERANTE_NOME` ainda em `dados_io.py` | migrar para o perfil exige mudar a assinatura de `parse_title`, que é chamada sem `cfg` em vários pontos. Trabalho contido, mas não trivial — não cabia nesta rodada sem risco |
| Vocabulário "adulterados/puros" no log de quantificação (`pipeline.py`) | o model card já usa o perfil; o log ainda não |
| Dataset de mel | não obtido (ver `VALIDACAO_PUBLICA.md`) |
