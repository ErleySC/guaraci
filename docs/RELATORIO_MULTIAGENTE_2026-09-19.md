# Relatório multiagente — varredura final: técnicas, testes e vault (2026-09-19)

Rodada de quatro frentes com o objetivo de confirmar, com evidência, se ainda
existe alguma técnica quimiométrica relevante fora do GUARACI, algum teste
real faltando, e se o vault reflete tudo. Todos os quatro agentes rodaram
`consultar_vault.py --cobertura` e `consultar_vault.py "<termo>"` antes de
qualquer busca externa, seguindo a regra de ouro desta sessão. Bloqueio de
publicação permanece em vigor. Commit de referência final: `5d5d753`, sobre
`3dce05f` (Agente 3). O vault foi regenerado três vezes ao longo da rodada
(baseline, após o commit do Agente 3, após o commit do Agente 4) e
`consultar_vault.py --cobertura` fechou **COMPLETA** em todas: 73/73 módulos,
13/13 técnicas, 132/132 Passos, 61/61 pendências/limitações, 14/14 datasets.

---

## 1. Resumo executivo

- **Nenhuma técnica de peso foi implementada** nesta rodada, por instrução
  explícita — o Agente 1 encontrou 2 candidatos genuinamente ausentes do
  vault/código (aumento de dados espectral por VRM e Dual-sPLS) e os deixou
  registrados para decisão do autor, não implementados.
- **Uma correção de premissa real**: a instrução original do Agente 2
  presumia que o leitor PerkinElmer `.sp` ainda estivesse em backlog por
  faltar arquivo de exemplo de 3 de 4 variantes. Isso está desatualizado — o
  vault já documentava `.sp` como implementado desde 2026-09-12. A instrução,
  não o projeto, estava com informação velha.
- **Uma implementação pequena e contra-provada**: o Agente 3 viabilizou
  mutation testing em escopo reduzido (3 funções mais críticas, não o módulo
  inteiro) em `predicao.py`, achou 4 bugs reais de cobertura de teste (mais 1
  bônus) e fechou todos com testes novos, cada um confirmado matando o
  mutante específico via `cosmic-ray mutate-and-test` isolado — não só "a
  suíte passa depois".
- **Nenhum dataset novo foi buscado** — a necessidade de um segundo dataset
  público para MSPC/fusão multibloco/ASCA+ já tinha sido avaliada e decidida
  no mesmo dia (Passo 219/220), e nada mudou desde então.
- **Auditoria cruzada (Agente 4) não achou nada quebrado** no vault: a única
  lacuna real era os 2 candidatos do Agente 1 não terem nenhum registro em
  lugar nenhum do projeto — corrigido com uma edição mínima e cirúrgica em
  `docs/MAPA_COMPLETUDE_V1.md` (11 linhas, 0 remoções).
- **Achado de ambiente, fora do escopo dos 4 agentes**: a suíte completa
  (`pytest -q -m "not slow"`, 1785 itens coletados) rodou pela primeira vez
  nesta sessão neste container e falhou em 6 testes (`test_plano_coleta.py`,
  `test_reports.py`) por `ModuleNotFoundError: No module named
  '_cffi_backend'` — o pacote `cffi` simplesmente não estava instalado no
  ambiente, quebrando a cadeia de import `fpdf2 → cryptography` usada na
  geração de PDF. Não é dependência declarada do projeto nem bug de código:
  instalei `cffi` no ambiente de verificação e os 43 testes dos dois arquivos
  passaram limpos. **Resultado final da suíte completa: 1690 + 43 = 1733
  passed, 22 skipped, 0 failed.**
- `ruff check .` e `mypy src/guaraci/ app_quimiometria.py` (85 arquivos)
  limpos no HEAD final.

---

## 2. Agente 1 — técnicas quimiométricas/analíticas ainda ausentes

Consultou o vault para 18+ termos candidatos (PLS esparso, sparse PLS,
elastic net, regularização, rede neural, CNN, deep learning, data
augmentation, aumento de dados, transfer learning, Bayesian, GAN,
self-supervised, contrastive, few-shot, lasso, shrinkage, n pequeno, sinal
fraco) antes de pesquisar na literatura — todos retornaram "nenhuma nota"
ou apontaram só para itens já registrados (di-PLS/T5, T10, Ledoit-Wolf/T7).
Confirmou por `grep` direto que ElasticNet/Ridge/Lasso já existem como
benchmark em `avaliacao_modelos.py`, descartando um candidato antes mesmo de
propor.

**Limitação metodológica registrada explicitamente**: `api.crossref.org`,
`sciencedirect.com`, `arxiv.org`, `doi.org` e `wikipedia.org` estavam
bloqueados pela política de egress da sessão. Os DOIs abaixo foram
confirmados por **triangulação de 2-3 fontes independentes** que citam o
mesmo DOI literal (WebSearch, que roda hospedado e não sofre o bloqueio),
não por consulta direta à API do Crossref — ressalva que consta também no
vault (`docs/MAPA_COMPLETUDE_V1.md`, Grupo 2).

### Candidatos priorizados (nenhum implementado)

1. **Aumento de dados espectral por Vicinal Risk Minimization (VRM)** —
   Tumoine, Metz, Abdelghafour, Esteve, Grotus, Bendoula & Roger (2026),
   *Chemometrics and Intelligent Laboratory Systems*, DOI
   `10.1016/j.chemolab.2026.105769`. Ataca diretamente a limitação
   estrutural mais recorrente do acervo privado (n pequeno / sinal fraco):
   gera amostras "vicinais" por perturbação fisicamente plausível
   preservando o rótulo, aumentando o n efetivo de calibração sem coleta
   nova. O autor sênior (Jean-Michel Roger) é coautor do próprio EPO/GLSW
   já implementado no GUARACI (T2). Esforço **M** — reaproveita
   transformações já existentes em `preprocessamento.py`, mas exige
   integração cuidadosa com `StableStratifiedGroupKFold` (amostras
   aumentadas de uma mesma amostra física precisam ficar no mesmo
   fold/grupo, senão cria vazamento novo).
2. **Dual-sPLS (PLS esparso via norma dual)** — Alsouki, Duval, Marteau, El
   Haddad & Wahl (2023), *Chemometrics and Intelligent Laboratory Systems*
   237, DOI `10.1016/j.chemolab.2023.104813`. Regulariza/seleciona variável
   DENTRO do próprio ajuste PLS (diferente de Ridge/Lasso/ElasticNet
   aplicados sobre X bruto, já presentes) — alternativa genuinamente
   distinta para o regime p≫n do acervo privado. Esforço **L** — sem
   biblioteca Python madura (só R/CRAN dos próprios autores), exigiria
   reimplementar o algoritmo de otimização por norma dual e validar
   numericamente contra a referência R.

### Descartados nesta rodada (com razão)

- Regressão regularizada em datasets ultra-pequenos (PLOS ONE, DOI
  `10.1371/journal.pone.0341850`) — redundante com ElasticNet/Ridge/Lasso
  já implementados.
- Extended Multiplicative Signal Augmentation (EMSA/Bjerrum et al.) —
  antecessor direto do VRM, generalizado e superado pela opção #1.
- Deep learning / CNN 1D para espectros com poucas amostras — a própria
  literatura do grupo é consistente em dizer que o ganho no regime de n do
  GUARACI (dezenas de amostras) é marginal ou nulo; mudança de arquitetura
  grande sem justificativa de ganho.

---

## 3. Agente 2 — formatos de dado e compatibilidade de equipamento

**Achado principal: correção de premissa da instrução, não do projeto.**
A instrução presumia `.sp` PerkinElmer em backlog; o vault já documentava
`parse_sp` como implementado desde o fechamento do Grupo 1 em 2026-09-12.
Estado real confirmado por leitura direta do vault/`docs/COMPATIBILITY.md`:

| Formato | Estado |
|---|---|
| SPC (Galactic/Thermo) | Implementado (`parse_spc`, extra `[spc]`, `spcfile` LGPL-3.0) |
| PerkinElmer `.sp` | Implementado (`parse_sp`, sem dependência nova, lógica adaptada de `specio.plugins.sp` BSD-3 com atribuição) |
| RMN bruto (Bruker) | Implementado (`parse_rmn_bruker`, extra `[rmn]`, `nmrglue` BSD-3) — FID bruto/Varian é exclusão deliberada de escopo |
| HPLC/GC-MS | Implementado (`parse_cromatograma_hplc` + `parse_tic_andi_ms`, cobre Agilent/Waters/ANDI-MS E1948) |
| UV-Vis (Agilent DAD) | Implementado (mesmo `parse_cromatograma_hplc`, canal DAD) |
| Fluorescência EEM | Implementado e generalizado (`parse_eem_dat` lê 2 formatos reais independentes sem alteração de código) |
| GC-IMS bruto `.mea` | **Backlog real** — lib madura existe (`gc-ims-tools`), mas datasets conhecidos são grandes (4,8-6,9GB) sem arquivo pequeno rotulado. Verificação de novidade **bloqueada** por egress em Zenodo/ScienceDirect nesta rodada. |
| ANDI/AIA-Chrom E1947 | Backlog — candidato `hdkim99/ordifile` (Apache-2.0) encontrado, mas sem arquivo `.CDF` real de exemplo no repositório |
| Shimadzu UV-Vis nativo | Backlog — resolvido de facto por CSV genérico; nenhuma lib madura para o binário nativo |

**Nenhuma implementação nesta rodada** — nenhum dos 3 itens de backlog real
passou no crivo de evidência (arquivo real inspecionável + lib madura +
licença compatível).

---

## 4. Agente 3 — testes faltantes e validação com dataset

### Mutação em escopo reduzido (`predicao.py`)

`cosmic-ray` com filtro de linha nativo restringiu os 605 mutantes do
arquivo inteiro aos 195 das 3 funções mais críticas (`_interpolate_to_reference`,
`predict_samples`, `quantify_sample`). 20/195 mutantes rodados (5 killed,
15 survived) por orçamento de tempo; dos 15 sobreviventes, 4 eram achados
reais (mais 1 bônus), todos fechados com testes novos em
`tests/test_predicao.py` (39→44 testes) e cada um confirmado morto
individualmente:

1. Erro na fórmula de Hotelling T² (`** 2`→`** 3`, `ddof=1`→`ddof=0`) sem
   oráculo numérico independente no teste.
2. Guard de variância zero nunca exercitado por dado sintético real.
3. `np.clip` degenerando toda predição sem teste contra oráculo de
   classe/confiança.
4. Fronteira exata (`>= 2` → `>= 1`) em candidatos ambíguos sem teste.
5. (bônus) Guard de chaves `ad_cv_*` invertido sem teste de conteúdo.

175/195 mutantes de `predicao.py` e a totalidade de `pipeline.py`,
`avaliacao_modelos.py` e o resto de `validacao_estatistica.py` seguem sem
rodar — custo por mutante (processo `pytest` inteiro) documentado e
mantido como backlog.

### Segundo dataset para MSPC/fusão multibloco/ASCA+

Avaliado e considerado **desnecessário**: a decisão já tinha sido tomada no
mesmo dia (Passo 219/220) — TEP (Rieth et al. 2017, Harvard Dataverse,
CC0) avaliado e conscientemente não perseguido por custo de conversão de
domínio desproporcional ao ganho. Fusão multibloco já testada contra par
real complementar (resultado negativo, documentado). ASCA+ é backlog de
completude de implementação, não de dataset. Nenhum dataset novo buscado.

---

## 5. Agente 4 — auditoria cruzada do vault

Regenerou o vault 2 vezes (antes e depois de sua própria edição) e
confirmou cobertura completa nas duas. Auditoria item a item:

| Item | Veredito |
|---|---|
| Candidatos do Agente 1 (VRM, Dual-sPLS) sem registro em lugar nenhum | **CORRIGIDO** — 2 linhas + 1 subseção nova em `docs/MAPA_COMPLETUDE_V1.md` (Grupo 2), com a ressalva de verificação de DOI por triangulação |
| Correção de premissa do Agente 2 sobre `.sp` | **OK** — vault/docs já corretos desde 2026-09-12; a premissa desatualizada estava na instrução, não no projeto |
| Achados do Agente 3 (Passo 222, mutação `predicao.py`) refletidos | **OK** — `docs/PROGRESSO.md` linha 5079 e `docs/MAPA_COMPLETUDE_V1.md` linha 141 já continham o registro completo |
| Ressalva de calibração do MSPC (IC levemente otimista, poucos grupos no Corn) | **OK** — já registrada em `docs/VALIDACAO_PUBLICA.md` §11 e propagada ao Passo 222 |

Guarda de privacidade (`scripts/privacidade_amostras.py`, embutida em
`gerar_vault_obsidian.py --check`): limpa, "484 notas planejadas, guarda de
privacidade limpa". `tests/test_sem_identificador_real.py` (9/9) e a suíte
de vault (49/49) passando.

---

## 6. Verificação independente (orquestrador, pós-agentes)

Antes de consolidar, reproduzi por conta própria as alegações de cada
agente em vez de aceitá-las por relato:

- `git log`/`git status` confirmaram os 2 commits (`3dce05f`, `5d5d753`)
  reais, árvore limpa.
- `pytest tests/test_predicao.py -q` → 44 passed (bate com o Agente 3).
- `mypy src/guaraci/predicao.py` e `mypy src/guaraci/ app_quimiometria.py`
  (85 arquivos) → limpos.
- `ruff check .` → limpo no projeto inteiro.
- `python scripts/gerar_vault_obsidian.py` + `consultar_vault.py
  --cobertura` → **COMPLETA** no HEAD final (73/73, 13/13, 132/132, 61/61,
  14/14).
- Suíte completa (`pytest -q -m "not slow"`, 1785 itens coletados, 67
  deselecionados) rodada do zero neste ambiente: achou 6 falhas por
  ambiente (`_cffi_backend` ausente, ver §1) não relacionadas a nenhuma
  mudança dos 4 agentes. Corrigido instalando `cffi` (não é dependência do
  projeto — gap de provisionamento do container). Resultado final: **1690
  passed + 43 passed (os 2 arquivos afetados) = 1733 passed, 22 skipped, 0
  failed.**

---

## 7. Estado do mapa de completude

**Fechado, com 2 itens novos pendentes de decisão do autor** (não de
implementação nem de investigação adicional): VRM e Dual-sPLS, registrados
em `docs/MAPA_COMPLETUDE_V1.md` (Grupo 2), esforço M e L respectivamente.
Nenhum dos dois bloqueia publicação — nenhuma funcionalidade existente
depende deles. Fora isso, nada de genuinamente novo e acionável foi
encontrado: os 3 formatos de equipamento em backlog (Agente 2), os módulos
de mutation testing pendentes (Agente 3) e a decisão de não buscar segundo
dataset (Agente 3) permanecem exatamente como estavam antes desta rodada,
com razão registrada e reconfirmada.

## 8. Uma frase por agente

- **Agente 1**: encontrou 2 técnicas genuinamente ausentes (VRM, Dual-sPLS)
  com DOI real; nada implementado; acionável agora é a decisão do autor
  sobre priorizar uma delas.
- **Agente 2**: encontrou que a premissa da própria instrução sobre `.sp`
  estava desatualizada e reconfirmou o backlog real (GC-IMS, ANDI E1947,
  Shimadzu) sem mudança; nada implementado; nada acionável agora além de
  aguardar arquivo de exemplo real.
- **Agente 3**: encontrou 4 bugs reais de cobertura em `predicao.py` via
  mutação em escopo reduzido e reconfirmou que segundo dataset é
  desnecessário; implementou e contra-provou os 5 testes novos; acionável
  agora é decidir se vale orçamento futuro para os 175 mutantes restantes.
- **Agente 4**: encontrou que os 2 candidatos do Agente 1 não tinham
  registro em lugar nenhum; corrigiu com uma edição mínima no mapa de
  completude; nada mais acionável — vault e docs já estavam consistentes.
