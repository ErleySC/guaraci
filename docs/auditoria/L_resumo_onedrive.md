# Fase L — Limpeza, backup e preparação para formatar (2026-08-17)

**Script único:** [`preparar_formatacao.ps1`](preparar_formatacao.ps1) —
dry-run por padrão. Substitui o antigo `mover.ps1`, que cobria só a saída
do OneDrive (removido para não existirem dois scripts fazendo metade do
trabalho cada um). Inventário detalhado em
[`inventario_onedrive.csv`](inventario_onedrive.csv).

```powershell
powershell -ExecutionPolicy Bypass -File docs/auditoria/preparar_formatacao.ps1
powershell -ExecutionPolicy Bypass -File docs/auditoria/preparar_formatacao.ps1 -Execute
```

*(o `-ExecutionPolicy Bypass` é necessário: a política desta máquina bloqueia
scripts `.ps1` por padrão — verificado.)*

---

## O risco nº 1 não é o OneDrive, é o Git

O script começa por aí de propósito. No momento em que este documento foi
escrito havia **25 arquivos com alterações não commitadas** — toda a
auditoria das Fases A/B, as correções de código e os relatórios. Formatar
sem commitar perde isso, e nenhum backup de arquivo cobre o que está só na
árvore de trabalho.

```
git add -A && git commit -m "..." && git push
```

Commits locais pendentes de push: **0** (verificado).

## Já limpo nesta sessão

Removidos por serem **duplicatas byte-idênticas** de `~/.guaraci/`
(verificado com `diff` antes de apagar, não presumido):

| Removido | Por quê |
|---|---|
| `src/guaraci/config.yaml` | cópia da config pessoal deixada dentro do **pacote** pela versão anterior a 2026-08-07 |
| `src/guaraci/.cli_modo_usuario` | idem |
| `visual_config.json` (raiz) | idem |

> Corrige de passagem uma afirmação errada de `docs/_CANDIDATOS_REMOCAO.md`
> (2026-07-13), que classificava `src/guaraci/config.yaml` como "default
> empacotado com a lib". **Não é**: `config.yaml` casa com o `.gitignore` em
> qualquer nível, logo nunca foi versionado, e um wheel construído do
> checkout limpo não o contém — confirmado inspecionando `build/`.

Removidos por serem **regeneráveis**: `build/`, `*.egg-info/`,
`.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/` (55 MB), `__pycache__/`,
`coverage.xml`, `.coverage`. O import editável continua funcionando
(verificado após a remoção do `egg-info`).

## O que existe SÓ nesta máquina

Nada disto está no GitHub. É o conteúdo da Etapa 2 do script.

| Item | Tamanho | Por que não está versionado |
|---|---|---|
| `~/.guaraci/` | — | estado do CLI: config, perfis, paleta, códigos de espécie |
| `scripts/gerar_relatorio_abnt.py` | **38 KB** | gitignored: contém 1 caminho absoluto pessoal |
| `scripts/run_benchmark_tcc.py` | 1,6 KB | idem |
| `run_N1/N2/N3.log` | ~33 KB | registro das execuções reais de 2026-07-10 |
| `sanity_out.txt`, `val_out.txt` | ~2,5 KB | diagnósticos das mesmas execuções |
| `config.yaml` (raiz) | 5 KB | config pessoal (o versionado é `config.example.yaml`) |
| `docs/_*.md` | ~16 KB | relatórios de sessões de auditoria anteriores |

**Sobre os dois scripts pessoais:** avaliei versioná-los, já que
`gerar_relatorio_abnt.py` tem 38 KB de lógica real (gerador de relatório
ABNT) e o motivo do gitignore é *uma única linha* — o caminho de saída.
**Mantidos fora do repositório mesmo assim**: o repo é público e ambos
carregam caminhos absolutos que expõem a estrutura de diretórios pessoal.
A decisão original estava certa; o que faltava era o backup, que o script
agora cobre.

**Sobre os logs de execução:** valem backup e não são "só log". São o
registro do que foi rodado (parâmetros, avisos, contagens) — e reexecutar
**não** os reproduz, porque o DD-SIMCA foi recalibrado em 2026-08-16 e os
números mudam.

## O que sai do OneDrive (Etapa 3)

| Item | Tamanho |
|---|---|
| `dados oleos/` (dataset de terceiro) | 122,5 MB |
| `guaraci_historico_antigo_*.bundle` | 11,9 MB |
| `resultados_tcc/` | 138,1 MB |
| `GUARACI_Demo/` | 24,1 MB |
| `.venv/` do repo | 867 MB (confirmar e excluir — o venv em uso é `~/.venvs/guaraci`) |
| repositório Git | passo **manual** no fim do script |

## Depois de mover

- Conferir os arquivos no destino **antes** de formatar.
- "Liberar espaço" do OneDrive **não** remove da nuvem — é preciso excluir
  ou desmarcar a pasta em Configurações → Escolher pastas.
- Esvaziar a lixeira do OneDrive na web, senão o espaço não é liberado.
- **Um disco só é zero backup**: manter uma segunda cópia do dataset e do
  `.bundle` em disco externo.
