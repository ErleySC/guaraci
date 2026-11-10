# Avisos de terceiros

Copyright (C) 2026 Erley Silva da Costa

GUARACI e' distribuido sob GPL-3.0-or-later (arquivo `LICENSE`). Este arquivo lista o
codigo de terceiros que esta copiado ou adaptado dentro do repositorio ou do pacote, com a
licenca e o texto que acompanha cada um. Dependencias instaladas a parte (numpy, scikit-learn,
brukeropus, nmrglue etc.) nao estao aqui: cada uma traz a propria licenca ao ser instalada.

## specio (leitor PerkinElmer .sp) -- BSD-3-Clause

- O que: a logica de leitura do formato `.sp` foi adaptada de `specio.plugins.sp`
  (Copyright 2017 Guillaume Lemaitre), sem o uso de `six`, em
  `src/guaraci/importadores_proprietarios.py` (`_sp_percorrer_diretorio_de_blocos` e auxiliares).
- Licenca: BSD-3-Clause. As tres condicoes e a isencao de garantia estao no texto integral em
  [`LICENSES/BSD-3-Clause-specio.txt`](LICENSES/BSD-3-Clause-specio.txt), que acompanha o wheel
  e o sdist.

## dual.spls (oraculo numerico dos testes) -- MIT

- O que: o codigo R de `d.spls.norm.R` e `d.spls.lasso.R` (pacote `dual.spls` 0.1.4, Louna
  Alsouki e Francois Wahl) em `tests/fixtures/dual_spls_oracle/`, usado so' para gerar as saidas
  de referencia dos testes. Nao faz parte do wheel.
- Licenca: MIT; texto completo em `tests/fixtures/dual_spls_oracle/LICENSE.dual.spls`. O titular e
  o ano seguem o que o proprio pacote declara no arquivo `LICENSE` dele (ano 2021; Louna Alsouki,
  Francois Wahl).
