# Política de dados públicos

**Nenhum arquivo de dado de terceiro é versionado neste repositório.**
CSVs, `.mat`, espectros, imagens de amostra — nada disso entra no `git`,
mesmo quando a licença do dataset permitiria redistribuição (ex.:
Mendeley CC BY 4.0). "Usar um dataset para validação" e "publicar o
dataset no nosso repositório" são coisas diferentes; este projeto só
faz a primeira.

## Como cada dataset público é obtido

| Dataset | Licença | Script de download | Onde os testes procuram o dado |
|---|---|---|---|
| Eigenvector Corn | ver a fonte (benchmarking) | inline no job `validacao-publica` do CI (`.github/workflows/test.yml`) | `$GUARACI_DATASETS_DIR/corn.mat` |
| Mendeley `ctgg7k4m5g` (óleos comestíveis, Ottaway et al. 2021) | CC BY 4.0 | `scripts/download_datasets/baixar_mendeley_oleos.py` | `$GUARACI_DATASETS_DIR/mendeley_ctgg7k4m5g/` |
| DeepHS Fruit / Kaki / câmera VIS (Varga, Makowski & Zell, IJCNN 2021) | não declarada formalmente (SPDX) no repositório — ver `docs/VALIDACAO_PUBLICA.md` §4, mesmo tratamento já dado ao Corn | `scripts/download_datasets/baixar_deephs_kaki.py` | `$GUARACI_DATASETS_DIR/deephs_kaki_vis/` |

Detalhes de cada dataset (n, técnica, referência, métricas obtidas) em
`docs/VALIDACAO_PUBLICA.md`.

## Mecanismo comum

1. Um script (ou passo de CI) baixa o dataset da fonte original **em
   tempo de execução**, nunca do repositório.
2. O conteúdo é verificado por **SHA256 + tamanho em bytes**, pinados
   no próprio script/workflow — se a fonte servir algo diferente do
   que foi auditado, o download falha alto em vez de seguir com um
   arquivo não verificado.
3. O destino é sempre `$GUARACI_DATASETS_DIR/<nome>` — uma pasta de
   **cache local, fora do controle de versão** (ver `.gitignore`,
   seção "Cache local de datasets públicos de terceiro"). Variável de
   ambiente não definida = os testes que dependem do dataset **pulam**
   com instrução de como obtê-lo, nunca falham por ausência nem tentam
   baixar nada por conta própria dentro da suíte de testes normal.
4. O CI baixa e roda contra o dado a cada execução dos jobs dedicados
   (`validacao-publica`, `validacao-publica-mendeley`) — isso é uso
   para validação externa, não redistribuição: o arquivo nunca é
   commitado, só existe no runner efêmero daquela execução.

## Se um subconjunto pequeno for necessário como fixture de teste

Gere uma versão **sintética** com as mesmas propriedades estatísticas
do dataset real (mesma faixa espectral, mesma ordem de grandeza de
ruído, mesma estrutura de classes) — nunca uma amostra literal extraída
do dataset de terceiro, mesmo que pequena. `dados_io.generate_synthetic_data`
já existe para isso (`mode="sintetico"` do `Config`).

## Scripts de aquisição (`scripts/download_datasets/`)

Cada script:

- É standalone (`python scripts/download_datasets/<nome>.py [destino]`)
  e também funciona chamado de dentro do CI.
- Pina SHA256+tamanho dos arquivos que baixa — nunca confia
  silenciosamente no que a fonte devolve.
- Detecta cache válido (mesmo hash) e pula o download de novo.
- Nunca grava um arquivo cujo hash não bateu com o esperado.

## Dados sintéticos (podem ser versionados)

Fixtures gerados programaticamente para teste (`mode="sintetico"`,
`generate_synthetic_data`) não são dado real de nenhuma fonte — não
têm restrição de licença e podem ser versionados livremente quando
úteis. Não confundir com os datasets públicos desta página.
