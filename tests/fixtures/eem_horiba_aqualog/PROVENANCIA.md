# Proveniencia do arquivo EEM Horiba Aqualog de teste

Arquivo real (nao sintetico), baixado publicamente para testar se
`guaraci.eem_io.parse_eem_dat` generaliza para um SEGUNDO formato de
exportacao de fluorimetro genuinamente independente do dataset Zenodo
`10.5281/zenodo.19755088` que motivou o parser original (Passo 149).

## `B1S12022-09-29-09-10-M3PEM.dat`

- Origem: `tests/sample_data/B1S12022-09-29-09-10-M3PEM.dat` do pacote
  `eempy` (https://github.com/YongminHu/eempy, PyPI `eem-python`),
  licenca MIT.
- Baixado em 2026-09-12 via `raw.githubusercontent.com`.
- Instrumento: Horiba Aqualog (fluorimetro comercial, formato de
  exportacao "PEM" = Excitation-Emission Matrix da amostra; ha tambem
  "BEM" = branco e "ABS" = absorbancia no mesmo pacote, nao usados aqui).
- Estrutura: 106 comprimentos de onda de excitacao (450-240nm, passo 2nm,
  DECRESCENTE) x 500 de emissao (243,173-824,234nm) -- grade MUITO mais
  fina que o dataset Zenodo original (35 excitacoes), confirmando que
  `parse_eem_dat` nao tem nenhuma suposicao hardcoded de tamanho de
  grade.

## Achado (2026-09-12)

`parse_eem_dat` (escrito originalmente so' para o dataset Zenodo, Passo
149) leu este arquivo Horiba Aqualog **sem nenhuma modificacao de
codigo**, 0 linhas descartadas (500/500 validas) -- a convencao de
3-linhas-de-cabecalho + linha de dado
`emissao<TAB>valor1<TAB>valor2...<TAB>valorN` acabou sendo comum aos dois
formatos (nao coincidencia forcada -- confirmado por leitura direta dos
dois arquivos reais, nao suposto). Ver `tests/test_eem_io.py` e a nota no
docstring de `eem_io.py`.
