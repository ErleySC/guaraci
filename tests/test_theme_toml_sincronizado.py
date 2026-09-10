"""Contra-prova: .streamlit/config.toml precisa continuar sincronizado com
design_tokens.TOKENS.

O Streamlit le' [theme]/[theme.dark] do TOML no boot do processo, antes de
qualquer import Python rodar -- nao ha' como o config.toml importar
design_tokens.py de volta, entao os dois arquivos sao duas fontes de verdade
mantidas manualmente em sincronia (comentario explicito no proprio TOML).
Sem este teste, mudar TOKENS (ex.: migracao de paleta de 2026-09-01, forest
verde -> laranja da mascote) muda o cabecalho custom (CSS, le' design_tokens
via _tok()) sem mudar a cor real dos botoes/widgets nativos do Streamlit
(le' so' o TOML) -- os dois ficariam visualmente inconsistentes sem nenhum
erro, silenciosamente.
"""
from __future__ import annotations

from pathlib import Path

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from guaraci.design_tokens import TOKENS

_RAIZ = Path(__file__).resolve().parents[1]


def _config_toml() -> dict:
    with open(_RAIZ / ".streamlit" / "config.toml", "rb") as f:
        return tomllib.load(f)


def test_tema_claro_bate_com_tokens_light():
    cfg = _config_toml()["theme"]
    tok = TOKENS["light"]
    assert cfg["primaryColor"] == tok["primary"]
    assert cfg["backgroundColor"] == tok["bg"]
    assert cfg["secondaryBackgroundColor"] == tok["surface"]
    assert cfg["textColor"] == tok["text"]


def test_tema_escuro_bate_com_tokens_dark():
    cfg = _config_toml()["theme"]["dark"]
    tok = TOKENS["dark"]
    assert cfg["primaryColor"] == tok["primary"]
    assert cfg["backgroundColor"] == tok["bg"]
    assert cfg["secondaryBackgroundColor"] == tok["surface"]
    assert cfg["textColor"] == tok["text"]
