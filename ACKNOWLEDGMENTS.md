# Agradecimentos / Acknowledgments

**Público:** quem usa ou cita o GUARACI · **Estado:** manual · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

O GUARACI é um software de autoria individual (ver `CITATION.cff`). Esta página atribui o software e a literatura em que ele se apoia.

## Software

O GUARACI é construído sobre trabalho de outras pessoas. Em particular **NumPy**, **SciPy**, **scikit-learn**, **pandas**, **matplotlib**, **Rich** e **Streamlit** e, indiretamente, sobre a literatura de quimiometria citada em [`docs/cientifico/referencias.md`](docs/cientifico/referencias.md) e nas docstrings dos módulos de cálculo, cada método atribuído à publicação que o define.

## Dados

O GUARACI não distribui dados de terceiros. Os valores padrão são ponto de partida, não universais, e ajustáveis pelas opções de cada técnica (`guaraci confianca`).

## Dados de instrumento e privacidade

Os leitores de arquivo de instrumento não leem campos de operador, local ou trilha de auditoria (`##AUDIT TRAIL`), e um teste varre os artefatos gerados atrás de identificadores de amostra e de caminhos de máquina. Antes de publicar qualquer espectro bruto, sanitize o cabeçalho com `python scripts/sanitizar_dx.py <entrada> <saída> --conferir`; o `--conferir` lista para inspeção humana todo campo que não reconhece, e não é um veredito.
