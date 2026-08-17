# preparar_formatacao.ps1
#
# Script UNICO de preparacao para formatar o PC. Substitui o antigo
# mover.ps1 (que so' cobria a saida do OneDrive) -- consolidado aqui para
# nao existirem dois scripts fazendo metade do trabalho cada um.
#
# Cobre, em ordem de irreversibilidade:
#   ETAPA 1 - AVISA sobre trabalho em git nao salvo (nao mexe no git)
#   ETAPA 2 - BACKUP do que existe SO' nesta maquina (nao esta no GitHub)
#   ETAPA 3 - MOVE dados sensiveis e pesados para fora do OneDrive
#   ETAPA 4 - LISTA o que e' regeneravel (nenhuma acao necessaria)
#
# Dry-run por PADRAO: mostra o que faria, nao mexe em nada.
#   powershell -File preparar_formatacao.ps1              # so' mostra
#   powershell -File preparar_formatacao.ps1 -Execute      # aplica
#
# ANTES de rodar com -Execute: feche o Claude Code e qualquer terminal com
# o diretorio do repositorio aberto (nao da' pra mover pasta em uso).

param(
    [switch]$Execute,
    [string]$Destino = "C:\GuaraciLocal"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

function Write-Etapa($n, $txt) {
    Write-Host "`n=== ETAPA $n - $txt ===" -ForegroundColor Cyan
}

function Copiar($origem, $destino, $desc) {
    if (-not (Test-Path $origem)) {
        Write-Host "  [PULAR] $desc (nao existe)" -ForegroundColor DarkGray
        return
    }
    Write-Host "  $desc"
    Write-Host "      $origem  ->  $destino"
    if (-not $Execute) { return }
    $pai = Split-Path $destino -Parent
    if (-not (Test-Path $pai)) { New-Item -ItemType Directory -Force -Path $pai | Out-Null }
    Copy-Item -Path $origem -Destination $destino -Recurse -Force
}

function Mover($origem, $destino, $desc) {
    if (-not (Test-Path $origem)) {
        Write-Host "  [PULAR] $desc (nao existe)" -ForegroundColor DarkGray
        return
    }
    $mb = [math]::Round((Get-ChildItem $origem -Recurse -File -ErrorAction SilentlyContinue |
                         Measure-Object -Property Length -Sum).Sum / 1MB, 1)
    Write-Host "  $desc  (~$mb MB)"
    Write-Host "      $origem  ->  $destino"
    if (-not $Execute) { return }
    if (Test-Path $destino) {
        Write-Host "      [ERRO] destino ja existe -- pulando p/ nao sobrescrever" -ForegroundColor Red
        return
    }
    $pai = Split-Path $destino -Parent
    if (-not (Test-Path $pai)) { New-Item -ItemType Directory -Force -Path $pai | Out-Null }
    Move-Item -Path $origem -Destination $destino -Force
}

Write-Host "=== PREPARACAO PARA FORMATAR O PC ===" -ForegroundColor White
if (-not $Execute) {
    Write-Host "MODO DRY-RUN (padrao). Nada sera alterado. Use -Execute para aplicar." -ForegroundColor Yellow
}
Write-Host "Repositorio: $repo"
Write-Host "Destino    : $Destino"

# ─────────────────────────────────────────────────────────────────────────
Write-Etapa 1 "Trabalho em Git nao salvo (AVISO -- este script nao mexe no git)"

Push-Location $repo
try {
    $naoCommitado = @(git status --porcelain 2>$null).Count
    $naoEnviado   = @(git log origin/master..HEAD --oneline 2>$null).Count

    if ($naoCommitado -gt 0) {
        Write-Host "  [ATENCAO] $naoCommitado arquivo(s) com alteracoes NAO COMMITADAS." -ForegroundColor Red
        Write-Host "            Formatar agora PERDE essas alteracoes." -ForegroundColor Red
        Write-Host "            Resolva antes:  git add -A ; git commit -m '...' ; git push" -ForegroundColor Yellow
    } else {
        Write-Host "  OK - nada pendente para commitar." -ForegroundColor Green
    }
    if ($naoEnviado -gt 0) {
        Write-Host "  [ATENCAO] $naoEnviado commit(s) locais ainda NAO enviados ao GitHub." -ForegroundColor Red
        Write-Host "            Resolva antes:  git push" -ForegroundColor Yellow
    } else {
        Write-Host "  OK - nenhum commit local pendente de push." -ForegroundColor Green
    }
} finally { Pop-Location }

# ─────────────────────────────────────────────────────────────────────────
Write-Etapa 2 "Backup do que existe SO' nesta maquina (fora do GitHub)"

$bkp = Join-Path $Destino "backup_nao_versionado"

# Estado do usuario do CLI (config, perfis salvos, paleta, codigos de especie)
Copiar "$env:USERPROFILE\.guaraci" "$bkp\dot_guaraci" `
       "Estado do CLI (~/.guaraci: config, perfis, paleta, codigos)"

# Config pessoal na raiz do repo (gitignored -- config.example.yaml e' o versionado)
Copiar "$repo\config.yaml" "$bkp\config.yaml" `
       "config.yaml pessoal da raiz do repo"

# Scripts pessoais: gitignored DE PROPOSITO (contem caminhos absolutos
# pessoais e o repo e' publico). Sem backup, somem na formatacao --
# gerar_relatorio_abnt.py sozinho tem ~38 KB de logica de relatorio ABNT.
Copiar "$repo\scripts\gerar_relatorio_abnt.py" "$bkp\scripts\gerar_relatorio_abnt.py" `
       "scripts/gerar_relatorio_abnt.py (38 KB, gerador ABNT)"
Copiar "$repo\scripts\run_benchmark_tcc.py" "$bkp\scripts\run_benchmark_tcc.py" `
       "scripts/run_benchmark_tcc.py (driver do benchmark)"

# Logs das execucoes reais do TCC: sao o REGISTRO do que foi rodado em
# 2026-07-10 (parametros, avisos, contagens). Nao regeneraveis sem
# reexecutar tudo -- e a reexecucao dara' numeros DIFERENTES, porque o
# DD-SIMCA foi recalibrado em 2026-08-16.
foreach ($log in @("run_N1.log", "run_N2.log", "run_N3.log",
                   "sanity_out.txt", "val_out.txt")) {
    Copiar "$repo\$log" "$bkp\logs_execucao\$log" "Registro de execucao: $log"
}

# Documentos de sessoes de auditoria anteriores (gitignored por padrao docs/_*.md)
foreach ($doc in @("_AUDITORIA_ESTADO.md", "_CANDIDATOS_REMOCAO.md", "_RELATORIO_SESSAO.md")) {
    Copiar "$repo\docs\$doc" "$bkp\docs_sessao\$doc" "Doc de auditoria: $doc"
}

# ─────────────────────────────────────────────────────────────────────────
Write-Etapa 3 "Mover dados sensiveis/pesados para fora do OneDrive"

$od = "$env:USERPROFILE\OneDrive\Documentos\ERLEY"

Mover "$od\dados oleos" "$Destino\dados_oleos" `
      "Dataset de terceiro (.dx) -- nao deve ficar em nuvem pessoal"
Mover "$od\guaraci_historico_antigo_20260815.bundle" `
      "$Destino\backup_historico\guaraci_historico_antigo_20260815.bundle" `
      "Bundle git com os 48 espectros reais purgados do historico"
Mover "$repo\resultados_tcc" "$Destino\resultados_execucoes\resultados_tcc" `
      "Saidas completas de execucao"
Mover "$repo\GUARACI_Demo" "$Destino\demo_saidas\GUARACI_Demo" `
      "Saida do 'guaraci demo' (regeneravel)"

Write-Host "`n  [MANUAL] O repositorio Git inteiro -> C:\dev\guaraci" -ForegroundColor Yellow
Write-Host "  Nao automatizado: mover um .git em uso corrompe o repo." -ForegroundColor Yellow
Write-Host @"
    # com o Claude Code e terminais FECHADOS:
    robocopy "$repo" C:\dev\guaraci /E /XD .venv __pycache__ build
    cd C:\dev\guaraci ; git status ; git log --oneline -3
    # so' depois de conferir os dois comandos acima, apague a origem
"@ -ForegroundColor DarkGray

# ─────────────────────────────────────────────────────────────────────────
Write-Etapa 4 "Regeneravel -- NAO precisa de backup"

Write-Host @"
  .venv/                 -> recriar: uv venv ; uv pip install -e ".[all]"
  build/, *.egg-info/    -> recriar: python -m build
  .pytest_cache/, .ruff_cache/, .mypy_cache/, __pycache__/
  coverage.xml, .coverage-> recriar: pytest --cov
  resultados_tecator.json-> recriar: python scripts/benchmark_tecator.py
  GUARACI_Demo/          -> recriar: guaraci demo
  Todo o codigo-fonte    -> ja esta no GitHub (repo publico)
"@ -ForegroundColor DarkGray

Write-Host "`n=== FIM ===" -ForegroundColor White
if (-not $Execute) {
    Write-Host "Revise acima. Rode com -Execute quando estiver pronto." -ForegroundColor Yellow
} else {
    Write-Host "CONFIRME os arquivos no destino ANTES de formatar." -ForegroundColor Yellow
    Write-Host "Lembre: 'liberar espaco' do OneDrive NAO remove da nuvem --" -ForegroundColor Yellow
    Write-Host "e' preciso excluir ou desmarcar a pasta em Configuracoes -> Escolher pastas." -ForegroundColor Yellow
    Write-Host "E um disco so' e' zero backup: mantenha uma 2a copia em disco externo." -ForegroundColor Yellow
}
