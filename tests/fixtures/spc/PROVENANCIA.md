# Proveniencia dos arquivos SPC de teste

Arquivos binarios reais (nao sinteticos), baixados publicamente para testar
`guaraci.importadores_proprietarios.parse_spc` contra dado de instrumento
genuino (ver docstring do modulo, secao SPC).

## `spectra.spc`

- Origem: `specio/datasets/data/spectra.spc` do pacote `specio`
  (https://github.com/paris-saclay-cds/specio), licenca BSD-3-Clause.
- Baixado em 2026-09-12 via API de blobs do GitHub (commit sha
  `93cc8e0e2660fc91f2b1f7abbb3672f19e6eac0c`).
- Conteudo: espectro Raman, subarquivo unico, 1911 pontos.

## `nir.spc`

- Origem: `test_data/nir.spc` do pacote `rohanisaac/spc`
  (https://github.com/rohanisaac/spc), licenca GPL-3.0.
- Baixado em 2026-09-12 via `raw.githubusercontent.com`.
- Conteudo: NIR multi-subarquivo (20 espectros, X compartilhado, 700 pontos
  cada) -- usado para testar o caminho `TMULTI` de `parse_spc` (que retorna
  so' o subarquivo 0, ver docstring da funcao).
