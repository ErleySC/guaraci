# Proveniencia do arquivo PerkinElmer .sp de teste

Arquivo binario real (nao sintetico), baixado publicamente para testar
`guaraci.importadores_proprietarios.parse_sp` contra dado de instrumento
genuino (ver docstring do modulo, secao PerkinElmer `.sp`).

## `spectra.sp`

- Origem: `specio/datasets/data/spectra.sp` do pacote `specio`
  (https://github.com/paris-saclay-cds/specio), licenca BSD-3-Clause.
- Baixado em 2026-09-12 via API de blobs do GitHub (commit sha do arquivo
  `93cc8e0e2660fc91f2b1f7abbb3672f19e6eac0c` -- ver historico do repo para
  o sha exato deste blob especifico, `spectra.sp`).
- Conteudo: espectro FT-IR, 3301 pontos, 4000-700 cm^-1. Valores conferidos
  contra o docstring publicado do `specio` (`spectra.wavelength`/
  `spectra.amplitudes`) durante o desenvolvimento de `parse_sp`.
