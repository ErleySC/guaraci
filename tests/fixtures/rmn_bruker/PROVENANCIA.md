# Proveniencia do experimento Bruker de teste

Diretorio real (nao sintetico) de um experimento RMN Bruker BioSpin,
baixado publicamente para testar `guaraci.importadores_proprietarios.
parse_rmn_bruker` contra dado de instrumento genuino (ver docstring do
modulo, secao RMN bruto Bruker).

## Origem

- Repositorio: `CIC-methods/FID-A` (https://github.com/CIC-methods/FID-A),
  `exampleData/Bruker/sample01_press/press/` -- ferramenta de
  processamento de MRS (Magnetic Resonance Spectroscopy), sem license
  file dedicado identificado no repositorio no momento da coleta (dado
  de exemplo, nao codigo executavel -- nao ha dependencia de codigo deste
  repositorio, so' o arquivo de dados).
- Arquivos versionados no repositorio de origem via Git LFS -- baixados
  em 2026-09-12 pelo endpoint de midia do GitHub
  (`media.githubusercontent.com/media/...`), nao pela API de conteudo
  (que so' devolve o ponteiro LFS, nao o binario).
- Experimento: PRESS (Point RESolved Spectroscopy, sequencia MRS comum),
  300MHz (`BF1`/`SF`=300.306160513697 MHz), adquirido em 2017
  (`##OWNER=neajam`, cabecalho `acqus` original).

## Arquivos incluidos

- `acqus`/`acqu`: parametros de aquisicao (JCAMP-DX-like, formato nativo
  Bruker).
- `fid`: FID bruto (dominio do tempo, 2048 pontos complexos) -- incluido
  para completude/possivel uso futuro, mas `parse_rmn_bruker` NAO le este
  arquivo (ver ESCOPO DELIBERADO na docstring do modulo).
- `pulseprogram`, `method`: metadados adicionais do experimento.
- `pdata/1/`: espectro JA PROCESSADO pelo TopSpin (`1r`/`1i` = parte
  real/imaginaria pos-FFT-e-fase, `procs`/`proc` = parametros de
  processamento) -- e' isto que `parse_rmn_bruker` le.

## Conferencia

Extraido com `nmrglue.bruker.read_pdata` + `guess_udic`/`uc_from_udic`
(API do proprio `nmrglue` para eixo ppm): 2048 pontos, eixo ppm de
+6,6705 a -6,6641 (decrescente, convencao padrao RMN), intensidade real
com amplitude maxima ~7,96e6 -- sinal nao-trivial, nao um array
degenerado de zeros.
