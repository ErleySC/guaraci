# Varredura de governança de dados (2026-08-17)

Escopo estrito: **referências a dados**. Atribuição institucional **não é
alvo** desta varredura — créditos a a instituição, o instituto, o grupo de pesquisa, o laboratório, a
agência de fomento e orientadores ficam e ficam corretos. Nada foi
removido nem ocultado nesse eixo.

Varredura sobre arquivos **rastreados pelo Git** (`git grep`/`git ls-files`),
excluindo `docs/auditoria/` (relatórios desta auditoria, tratados à parte).

---

## Resultado em uma linha

**Nenhuma ocorrência foi introduzida por mim.** Todas as 20 referências a
identificador de amostra são **pré-existentes e já públicas no remoto** —
a contagem local bate exatamente com a do `origin/master` em todos os
arquivos. Zero espectro, zero `.dx`, zero caminho institucional.

---

## M.1 — Identificadores de amostra

| Arquivo | Linhas | Já público? | Autoria | Natureza |
|---|---|---|---|---|
| `src/guaraci/cli_assistente.py` | 925, 926, 933, 934 | **sim** (4=4) | pré-existente | texto de ajuda do CLI: exemplo de nome de arquivo |
| `src/guaraci/dados_io.py` | 50, 52, 202, 203 | **sim** (4=4) | pré-existente | docstring do formato de `mae_id` |
| `src/guaraci/guaraci.py` | 2258, 2271 | **sim** (2=2) | pré-existente | texto de ajuda do CLI |
| `tests/test_dados_io_jcamp.py` | 47, 59, 62, 78, 83, 105 | **sim** (6=6) | pré-existente | fixtures de teste |
| `tests/test_dados_io_parsing.py` | 170 | **sim** (1=1) | pré-existente | fixture de teste |
| `tests/test_heatmap_especie_adulterante.py` | 17, 18, 23 | **sim** (3=3) | pré-existente | fixture de teste |

Exemplos do que aparece: `AND-10-06-2020_T1.dx`,
`BCB-03-03-2099_AD-S-20_T1.dx`, `CAP-04-11-2099-A1.03`.

**Contexto:** são espécie + data e, em alguns casos, adulterante + teor.
Aparecem como exemplos de documentação, texto de ajuda e fixtures — não
como dado de análise.

**Nota sobre `AND-01-01-2022-S5.00`** (`test_heatmap`, linha 18): a data
2022 não corresponde a nenhuma sessão do dataset (que vai de 03-2020 a
11-2020) — é identificador **inventado** para teste, não real.

### O que foi removido nesta rodada

Os 6 `##TITLE=` malformados e o alias de `mae_id` **saíram do código** no
BLOCO B e vivem em `~/.guaraci_local/`. Os 11 commits locais foram
reconstruídos antes de qualquer push, e a verificação
`git log -p origin/master..HEAD | grep '^+'` não encontra nenhum
identificador introduzido.

## M.2 — Caminhos absolutos ou institucionais

**Duas ocorrências, ambas benignas:** `iniciar_guaraci.bat` linhas 9 e 11
mencionam "OneDrive" em **comentário explicativo** (por que o venv fica
fora dele). Não são caminhos, são prosa técnica. Nenhuma referência a
`dados oleos/`, nenhum `C:\Users\...`, nenhum caminho de rede
institucional em arquivo rastreado.

## M.3 — Arquivos `.dx`/`.jdx` versionados

**Zero.** `git ls-files | grep -i '\.dx$|\.jdx$'` → 0. Consistente com a
purga de 2026-08-15 (achado S4) e com o `.gitignore` (`*.dx`, `dados/`).

## M.4 — Valores espectrais reais embutidos

**Nenhum.** Busca por blocos numéricos com formato de par
`wavenumber intensidade` em `tests/` não retorna nada. As fixtures geram
espectros **sintéticos** em tempo de execução (`_escrever_dx` com
`y_base` construído no próprio teste).

---

## Já público × só local

**Todas as ocorrências de M.1 estão no `origin/master`.** A contagem
remota é idêntica à local em cada arquivo, o que significa: nada a
"proteger" com reescrita local, e a decisão sobre o já publicado **não é
técnica**.

Como registrado em [`exposicao_dado_publico.md`](exposicao_dado_publico.md):
reescrever o histórico remoto **não despublica** o que já foi clonado ou
indexado — públicos desde `e6384db` (2026-07-04). Fica **apenas
registrado**, para a conversa sobre autoria e distribuição.

---

## Confirmação dos CSVs externos

Verificado por execução, não por leitura:

```
1) fonte unica dos rotulos corrigidos:
   correcoes_titulo   -> 6 entradas, de C:\Users\erley\.guaraci_local
   pureza_indeterm.   -> 1
   alias_mae_id       -> 1
2) ausencia e SILENCIOSA: True
3) malformacao ABORTA: <caminho>, linha 2: esperadas 2 colunas ...
```

- **São a única fonte** dos rótulos corrigidos — não há cópia no código.
- **Ausência é silenciosa e legítima** (outra máquina, outro dataset): o
  parser roda normalmente, só sem os casos particulares.
- **Malformação aborta** com erro nomeando arquivo e linha. Uma correção
  de rótulo aplicada pela metade produziria amostra com pureza errada —
  exatamente o achado A2-2. Falhar alto é o comportamento correto.
- Estão **fora da árvore do repositório** (`~/.guaraci_local`), logo fora
  do alcance de qualquer `git add`.

---

## O que NÃO foi tocado, de propósito

Nenhuma menção a instituição, instituto, grupo de pesquisa, laboratório, agência de
fomento ou nomes de orientadores/colaboradores foi removida, alterada ou
ocultada. A titularidade do software decorre do vínculo de bolsa e dos
recursos institucionais usados (Lei 9.609/98, art. 4º, §3º), não do que
está escrito nos cabeçalhos — apagar atribuição não muda titularidade e
piora a posição do autor. Assunto do NIT/Agência de Inovação, fora do
código.
