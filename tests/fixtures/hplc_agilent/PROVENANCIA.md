# Proveniencia do diretorio Agilent `.D` de teste

Diretorio real (nao sintetico), baixado publicamente para testar
`guaraci.importadores_proprietarios.parse_cromatograma_hplc` contra dado
de instrumento genuino (ver docstring do modulo, secao HPLC cromatograma
bruto / UV-Vis comercial).

## `pink.D`

- Origem: `tests/inputs/pink.D` do pacote `rainbow-api`
  (https://github.com/evanyeyeye/rainbow, PyPI `rainbow-api`), licenca
  LGPL-3.0-or-later.
- Baixado em 2026-09-12 via `raw.githubusercontent.com`.
- Conteudo: 2 canais de detector UV/DAD (Agilent ChemStation `.ch`) de
  comprimento de onda unico -- `DAD1A.ch` (210nm) e `DAD1B.ch` (230nm),
  9000 pontos cada, 0-60 minutos de tempo de retencao. So' os arquivos
  `.ch` foram baixados (o cubo completo `DAD1.uv`, ~7,9MB, nao foi
  necessario para provar o parser -- um canal unico ja' exercita o mesmo
  caminho de codigo).
- Usado para testar o caminho `by_detector["UV"]` de `parse_cromatograma_
  hplc` -- que retorna so' o PRIMEIRO arquivo desse detector (`DAD1A.ch`,
  210nm) por design (ver docstring da funcao).
