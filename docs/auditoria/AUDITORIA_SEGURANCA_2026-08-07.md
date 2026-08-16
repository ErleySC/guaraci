# Auditoria de segurança — 2026-08-07

Varredura sistemática do programa inteiro (CLI + app web), não só do diff
desta sessão. Método: grep dirigido por classe de vulnerabilidade (injeção
de comando, desserialização insegura, path traversal, segredos expostos,
SSRF) + leitura manual de cada ocorrência até confirmar exploração real ou
descartar.

## Resumo

| # | Achado | Severidade | Estado |
|---|---|---|---|
| S1 | Bypass da mitigação de RCE via pickle (`GUARACI_DISABLE_MODEL_UPLOAD`) | **CRÍTICA** | ✅ Corrigido |
| S2 | Condição de corrida em arquivo temp compartilhado (multi-usuário) | ALTA | ✅ Corrigido |
| S3 | Interpolação de string em `os.system()` (padrão de injeção de comando) | BAIXA | ✅ Corrigido |
| S4 | Branch `master` LOCAL ainda contém 48 espectros reais no histórico | **ALTA** (dado, não código) | ⚠️ Requer ação do autor |

**Verificado e correto, sem achado:** guarda de `joblib.load` (P5,
`carregar_modelo`/`SecurityError`/manifesto SHA-256), ausência de
`eval`/`exec`/`yaml.load` inseguro/`pickle` direto, `unsafe_allow_html`
ausente, sem segredos hardcoded, sem chamada de rede (SSRF não aplicável),
`.gitignore` cobre todo estado sensível, workflows do CI não interpolam
`github.event.*` em shell (vetor clássico de injeção de script em Actions).

---

## S1 (CRÍTICA) — Bypass da mitigação de RCE via pickle

`GUARACI_DISABLE_MODEL_UPLOAD=1` é a mitigação documentada em `SECURITY.md`
para deploys públicos: desabilita o uploader de `.joblib` na aba Predição,
porque `joblib.load()` executa código arbitrário no momento do carregamento
(pickle). **A mitigação tinha um desvio que a esvaziava por completo.**

**Cadeia de exploração, confirmada por leitura de código:**

1. Operador de um deploy público configura `GUARACI_DISABLE_MODEL_UPLOAD=1`,
   seguindo a própria orientação do projeto.
2. O uploader de `.joblib` desaparece da aba Predição — **mas o campo de
   texto livre "Or local path to model" continua visível**
   (`app_tabs/predicao.py`, não estava atrás do mesmo `if upload_bloqueado`).
3. O uploader de **CSV** na aba Dados (`app_tabs/dados.py`) **não é coberto
   por essa flag** — segue aceitando upload de qualquer visitante.
4. `st.file_uploader(type=["csv","txt"])` só filtra no **seletor de arquivo
   do navegador** — é trivialmente contornável renomeando um arquivo antes
   de selecioná-lo (ou via requisição HTTP direta). `joblib.load()` não
   liga para extensão, só para os bytes.
5. Um visitante remoto sobe um pickle malicioso disfarçado de
   `"modelo.csv"`. O arquivo cai em
   `{tempdir}/pq_uploads/modelo.csv` — caminho **previsível**, porque era a
   MESMA pasta compartilhada entre todas as sessões/visitantes, com o nome
   original do arquivo como está.
6. O visitante volta à aba Predição, cola esse mesmo caminho no campo
   "local path", marca a caixa "I trust the source" (confirmação que só
   verifica a intenção do PRÓPRIO visitante, não a origem real do arquivo)
   e clica em Predict.
7. `carregar_modelo(caminho, confiar=True)` → `joblib.load()` → **RCE remota,
   sem autenticação**, apesar de `GUARACI_DISABLE_MODEL_UPLOAD=1` estar
   corretamente configurado.

**Causa raiz de design:** o comentário original em `app_quimiometria.py`
explica a intenção — "aceitar apenas caminhos locais controlados pelo
próprio operador". Mas um campo de texto num app web não distingue
"o operador digitou isso" de "um visitante digitou isso" — qualquer um que
acesse a página alcança o campo. A suposição de que o campo seria
"operador-only" nunca foi verdadeira para uma aplicação web pública.

**Correção** (`src/guaraci/app_tabs/predicao.py`): quando
`upload_bloqueado=True`, o campo de caminho local também fica oculto, não
só o uploader — nesse modo, a aba Predição não carrega nenhum modelo pela
UI web, ponto. Quem precisa rodar predição num deploy público deve usar a
CLI localmente.

---

## S2 (ALTA) — Condição de corrida em caminho de upload compartilhado

Mesmo sem a cadeia de S1, o caminho de upload (CSV em `dados.py`, `.joblib`
em `predicao.py`) usava um **nome fixo numa pasta temporária compartilhada**
entre todas as sessões do processo Streamlit. Num deploy multi-usuário
(vários visitantes concorrentes no mesmo processo, o modelo padrão de
deploy do Streamlit), a sessão B pode sobrescrever o arquivo entre a sessão
A escrevê-lo e carregá-lo — A acaba processando o conteúdo de B sem saber,
uma falha de integridade mesmo sem intenção maliciosa de ninguém.

**Correção:** `app_logic.caminho_upload_temp(nome_original, session_id,
...)` — função pura nova, testada isoladamente. Duas garantias: (1) só o
basename do nome original é usado (bloqueia path traversal — um nome como
`"../../etc/passwd"` não escapa do diretório de destino); (2) isolado numa
subpasta por `session_id` aleatório (`uuid.uuid4().hex`, gerado uma vez por
sessão via `st.session_state`, nunca exposto ao cliente) — sessões
diferentes nunca mais colidem, e isso **também fecha o caminho previsível
que S1 explorava**, como segunda camada de defesa independente da correção
de S1.

Usada em `dados.py` (upload de CSV) e `predicao.py` (upload de `.joblib`).

---

## S3 (BAIXA) — Padrão de interpolação de string em `os.system()`

`guaraci.py` (comando `guaraci demo`, ao abrir a pasta de resultados no
Finder/file manager):

```python
os.system(f'open "{pasta_run}"')      # macOS
os.system(f'xdg-open "{pasta_run}"')  # Linux
```

`pasta_run` é sempre gerado internamente neste caminho de código (nome de
pasta do `guaraci demo`, nunca influenciado por input externo) — **não é
explorável hoje**. Mas é exatamente o padrão que vira injeção de comando
real se algum dia `pasta_run` passar a incluir um valor influenciado por
usuário (ex.: se o `tag` da execução real, digitável livremente na CLI,
fosse usado aqui em vez de só no caminho gerado pelo `guaraci demo`) — um
diretório com `"` no nome já quebraria a citação hoje.

**Correção:** `subprocess.run(["open", str(pasta_run)])` /
`["xdg-open", str(pasta_run)]` — lista de argumentos nunca passa por um
shell, elimina a classe de vulnerabilidade por completo, não só o caso de
uso atual.

---

## Verificado e correto (sem mudança)

- **`carregar_modelo()` (P5, `predicao.py`)**: guarda `confiar=False` por
  padrão, `SecurityError` explícito, verificação de hash SHA-256 via
  manifesto ANTES de `joblib.load` quando disponível. Os dois pontos de
  chamada (CLI e app web) exigem confirmação humana real antes de passar
  `confiar=True` — CLI via pergunta s/n explícita, app web via checkbox
  `value=False` que efetivamente bloqueia o carregamento se desmarcado.
- Nenhum `eval`/`exec` no projeto.
- Nenhum `yaml.load()` sem `safe_load` (a carga de config usa PyYAML de
  forma segura em todo o projeto).
- Nenhum uso direto de `pickle.load`/`pickle.loads` fora do que já passa
  pelo guard de `carregar_modelo`.
- Nenhum `unsafe_allow_html=True` no app Streamlit.
- Nenhuma chamada de rede (`requests`/`urllib`) no código do pacote — sem
  superfície de SSRF.
- Nenhum segredo/token/chave hardcoded (varredura heurística por padrão de
  atribuição de string longa a `api_key`/`secret`/`password`/`token`).
- `.gitignore` cobre `config.yaml`, `codigos_usuario.json`, `perfis/`,
  `.cli_wizard_done`, `.cli_modo_usuario`, `*.joblib`, saídas de execução —
  nenhum estado sensível versionado.
- Workflows do CI (`.github/workflows/*.yml`) não interpolam
  `github.event.*` (título de PR, nome de branch) diretamente em comandos
  de shell — o vetor clássico de injeção de script em GitHub Actions.

## S4 (ALTA — exposição de dado, não vulnerabilidade de código) — `master` local carrega os espectros reais

**Requer ação do autor. Não é corrigível por commit** (é estado de um ref
local, não conteúdo de arquivo).

Levantado ao avaliar se o repositório pode ser tornado **público** — a via
mais direta para resolver o esgotamento de cota do GitHub Actions (Actions
é gratuito e ilimitado em repositório público; a cota só é consumida em
repositório privado).

**Situação medida:**

| Ref | Arquivos `.dx` no histórico |
|---|---|
| `HEAD` (`historico-limpo-preview`) | 0 |
| `origin/master` (remoto) | 0 |
| `origin/historico-limpo-preview` (remoto) | 0 |
| Todas as 11 tags remotas (`v31.0.0`…`v31.9.0`) | 0 |
| **`refs/heads/master` (branch LOCAL)** | **48** |

Ou seja: **tudo que está no GitHub hoje está limpo.** O que carrega os 48
espectros reais (`dados/ACA-04-11-2020_T1.dx` etc. — o dataset FT-NIR não
publicado do TCC) é a branch `master` **local**, que nunca foi atualizada
após a reescrita de histórico (`26a8f5b` local vs `88caa27` no remoto).

**Por que isso é um risco e não uma curiosidade:** enquanto esse ref existir
na máquina, um `git push origin master`, um `git push --all`, ou um
`git checkout master` seguido de push publica o dataset — e se o
repositório estiver público nesse momento, publica para qualquer um. É
exatamente o tipo de acidente de um comando só que a reescrita de histórico
existia para evitar.

**Ação recomendada, antes de tornar o repositório público:**

1. Confirmar que a branch local não tem nada a salvar que já não esteja no
   remoto (`git log origin/master..master --oneline`), e então apagá-la ou
   realinhá-la: `git branch -D master` (ou
   `git branch -f master origin/master`).
2. `git reflog expire --expire=now --all && git gc --prune=now --aggressive`
   para remover os objetos órfãos do clone local.
3. **Independentemente do local:** objetos de um histórico reescrito podem
   permanecer acessíveis por SHA direto no GitHub até a coleta de lixo do
   servidor. Se o histórico antigo **chegou** a ser enviado ao GitHub em
   algum momento, abrir um chamado no GitHub Support pedindo GC do
   repositório é o passo que efetivamente os remove. Sem essa confirmação,
   a alternativa mais segura é **publicar um repositório novo** (criado do
   zero, com o histórico limpo importado), em vez de tornar público o
   repositório que já existiu como privado com o dado dentro.

**Escopo desta constatação:** os arquivos foram identificados por nome e
extensão (`.dx`/`.jdx`) e por qual ref os alcança. Não foi feita inspeção
de conteúdo dos espectros nem avaliação sobre a política de compartilhamento
de dados do grupo/instituição — a decisão sobre publicar ou não o dataset é
do autor e do orientador, não uma conclusão técnica desta auditoria.

---

## Não auditado nesta rodada

Dependências de terceiros (CVEs conhecidas em versões pinadas — precisaria
de uma ferramenta como `pip-audit`/`safety`, não rodada aqui); autenticação/
autorização do deploy web em si (o projeto não implementa login — depende
inteiramente de controles de acesso de infraestrutura, fora do escopo do
código); `dados_imagem.py` (protótipo, não coberto por esta varredura).
