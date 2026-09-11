# Procedimento — duplicação para repositório novo (v1.0.0)

> Documento de PREPARAÇÃO (Fase D da instrução de 2026-09-11). Nenhum
> passo abaixo foi executado além do explicitamente marcado como
> "rehearsal" (rodado sobre um clone descartável, fora do repositório
> real). Criar o repositório novo, dar push nele e criar a tag `v1.0.0`
> exigem autorização explícita do autor, dada especificamente para essa
> ação — mesma categoria de "bloqueio de publicação" em vigor a sessão
> inteira.

## 0. Decisão da Fase C (já tomada pelo autor)

**Histórico filtrado** (`git filter-repo --replace-text`), não histórico
completo nem squashed. Motivo: o repositório atual carrega, em 4
commits antigos (`22b5511`, `676e9c4`, `dd09c41`, `338c45f`), o caminho
absoluto de máquina do autor (`C:\Users\erley\...`) — corrigido no texto
atual desde o Passo 200, mas ainda legível em `git log -p` desses 4
commits. Um repositório novo não precisa herdar isso.

## 1. Rehearsal já executado (nesta sessão, sobre clone descartável)

Feito e verificado, não é mais teórico:

1. Clone limpo de `origin/master` (commit `16788ce`) em pasta descartável.
2. `pip install git-filter-repo` (não vem por padrão).
3. Regra de substituição (`replacements.txt`, formato `--replace-text`):
   ```
   C:\Users\erley==>C:\Users\REDACTED
   ```
4. `git filter-repo --replace-text replacements.txt --force` no clone.
5. **Verificado**: `git log --all -p -S "C:\Users\erley"` — **zero
   ocorrências** depois do filtro (antes: 4 commits). Os 4 commits
   afetados agora trazem `C:\Users\REDACTED\...` no lugar, conteúdo ao
   redor intacto (verificado lendo o diff de `docs/PROGRESSO.md` num
   desses commits pós-filtro).
6. **Verificado**: as 320 commits do histórico continuam todas
   presentes (nenhuma removida — `filter-repo` reescreve conteúdo, não
   descarta commits). As 11 tags existentes (`pibic-2026-08`,
   `v31.0.0`..`v31.9.0`) **todas preservadas**, remapeadas
   automaticamente para os novos hashes (comportamento padrão do
   `filter-repo`, não precisa de passo manual).
7. HEAD do clone filtrado ficou em `d5b1410` (hash novo — reescrita de
   histórico sempre muda todos os hashes a partir do primeiro commit
   tocado, é esperado, é o motivo de nunca fazer isso no repositório
   real sem necessidade).

**Conclusão do rehearsal**: o filtro funciona exatamente como esperado,
sem efeito colateral visível. Pronto para ser repetido sobre um clone
"de verdade" quando o autor autorizar a Fase D final.

## 2. Procedimento para quando o autor autorizar

Nenhum destes passos foi executado no repositório real.

```bash
# 2.1 Clone dedicado -- NUNCA o working directory do dia a dia
git clone https://github.com/ErleySC/guaraci.git guaraci-v1-filtrado
cd guaraci-v1-filtrado

# 2.2 Filtro de histórico (mesma regra já testada acima)
pip install git-filter-repo
printf 'C:\\Users\\erley==>C:\\Users\\REDACTED\n' > replacements.txt
git filter-repo --replace-text replacements.txt --force

# 2.3 Verificação pós-filtro (obrigatória antes de prosseguir)
git log --all -p -S 'C:\Users\erley' | wc -l   # tem que dar 0
git tag                                          # tem que listar as 11 tags
git log --oneline | wc -l                        # tem que dar 320 (ou mais, se
                                                   # houver commits novos até lá)

# 2.4 Repositório novo no GitHub -- AÇÃO DO AUTOR (ou autorização explícita
# a esta sessão para criar via `gh repo create`), nome a decidir

# 2.5 Remoto novo + push (histórico + tags)
git remote add origin-novo https://github.com/<owner>/<nome-novo>.git
git push origin-novo --all
git push origin-novo --tags

# 2.6 Tag v1.0.0 (mensagem preparada na seção 3 abaixo)
git tag -a v1.0.0 -F v1.0.0-tag-message.txt
git push origin-novo v1.0.0
```

## 3. Mensagem da tag `v1.0.0` (preparada, não criada)

```
v1.0.0 — primeiro release coberto pela política formal de SemVer

A partir daqui, mudar assinatura de função pública, formato de
dataclass de resultado, esquema de config.yaml ou nome de coluna de
saída de forma incompatível exige bump de major (ou minor com
depreciação) -- ver docs/COMPATIBILITY.md.

Correção científica destacada: selecao_lv_cv_aninhada=True passa a ser
o padrão -- a métrica de CV reportada (balanced accuracy, Q2, ROC AUC,
CV-ANOVA) agora vem de CV aninhada honesta em vez de reusar os mesmos
folds para escolher o número de variáveis latentes E avaliar o modelo
(achado #12, medido com 10 seeds independentes + Wilcoxon p=0,0020).

Ver docs/CHANGELOG.md para a lista completa de capacidades acumuladas
(HSI, 14 validações públicas multitécnica, DD-SIMCA/OPLS-DA, PARAFAC/
N-PLS multiway, MCR-ALS, ASCA, EPO/GLSW, PQN, predição conforme,
execução não-interativa via CLI, etc.).
```

(Texto sincronizado com a entrada `## v1.0.0 — 2026-09-11` já escrita em
`docs/CHANGELOG.md`, Passo 213/214.)

## 4. Portabilidade — o que precisa de ajuste manual no repositório novo

Buscado `ErleySC/guaraci` em todo arquivo versionado (`.py`/`.md`/
`.toml`/`.yml`/`.cff`), 11 arquivos:

- **3 strings de código, cosméticas** (nenhuma lógica depende delas —
  são só texto de exibição, não URLs que o código busca/usa em tempo de
  execução): `src/guaraci/app_tabs/sobre.py:12`, `src/guaraci/
  guaraci.py:4326`, `src/guaraci/resultados_io.py:477` (linha do model
  card/tela "Sobre").
- **8 ocorrências em docs/config**: `README.md`/`README.pt-br.md` (5),
  `pyproject.toml` (3, campos `[project.urls]`), `docs/index.md` (3,
  MkDocs), `mkdocs.yml` (2, `repo_url`), `.github/ISSUE_TEMPLATE/
  config.yml` (2), `CONTRIBUTING.md` (1).

Nenhum desses é um acoplamento ESCONDIDO (nada faz request HTTP pra'
essa URL nem depende dela pra' funcionar) -- é find-and-replace
mecânico de `ErleySC/guaraci` pelo owner/nome do repositório novo,
quando o autor decidir esse nome.

## 5. Scripts do vault — confirmado, não presumido

`scripts/gerar_vault_obsidian.py` e `scripts/consultar_vault.py` usam
`git rev-parse --short HEAD` (via `subprocess`) para o campo
`commit:`/`gerado_em:` de cada nota -- **nenhuma URL de repositório
hardcoded em nenhum dos dois**. Confirmado por leitura direta do código
(`_head_info()` em `gerar_vault_obsidian.py`), não presumido. Funcionam
sem alteração em qualquer repositório/clone.

## 6. Simulação de instalação limpa

Clone fresco de `origin/master` (`16788ce`) + venv novo (Python 3.12,
`pip install -r requirements.txt`) num caminho **fora** de qualquer
estrutura de projeto existente, sem nenhum cache/estado da máquina de
desenvolvimento.

**Achado real or não-óbvio**: a 1ª tentativa (venv dentro da pasta de
scratchpad desta sessão, caminho profundamente aninhado) falhou com
`OSError` instalando `lxml` -- Windows MAX_PATH (260 caracteres)
excedido por um recurso interno do pacote
(`iso_schematron_skeleton_for_xslt1.xsl`), não suporte a caminho longo
habilitado no SO. **Não é bug do projeto** -- é uma característica
conhecida do Windows sem "Long Path support" ligado, que afeta
QUALQUER instalação Python num caminho profundo o suficiente. Resolvido
usando um caminho curto (`C:\gtmp\...`) para o rehearsal -- registrado
aqui porque quem clonar este projeto numa pasta muito aninhada no
Windows pode bater no mesmo problema; não é algo o `pyproject.toml`
consiga evitar (é limitação do SO do usuário, não da dependência).

**Resultado da simulação** (clone `C:\gtmp\repo`, venv `C:\gtmp\venv`,
Python 3.12.10, `pip install -r requirements.txt` + `pip install -e
.[dev]`): `guaraci --version` → `GUARACI v1.0.0` (confirma o
empacotamento/entry point funcionam sem nenhum estado residual da
máquina de desenvolvimento); `import guaraci` limpo; suíte completa
**1533 passed, 0 failed, 43 skipped** em 526,9s (~8,8 min). Nenhum
cache do `~/.venvs/guaraci` nem do OneDrive envolvido.
