"""preferencias_visuais.py — Preferencias visuais persistentes do usuario
(paleta/fonte/grid/alpha), compartilhadas entre a CLI e o app web.

Extraido de `guaraci.py` (`_carregar_visual_cfg`/`_salvar_visual_cfg`) para
que as DUAS interfaces leiam e gravem o MESMO arquivo, em vez de a web ter um
estado de cor proprio em paralelo: a paleta escolhida na CLI aparece
selecionada no app e vice-versa.

Sem dependencia de `rich` nem de `streamlit` — o app Streamlit importa isto
sem arrastar a CLI inteira (guaraci.py tem ~5 mil linhas e depende de rich).
Falha de escrita LANCA `OSError`: quem chama decide como avisar (a CLI
imprime no console, a web usa `st.warning`) — nunca um "salvo" silencioso que
nao aconteceu, que foi exatamente o bug corrigido em 2026-08-16.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["USER_DIR", "VISUAL_PATH", "load_visual_config", "save_visual_config"]

USER_DIR = Path.home() / ".guaraci"
VISUAL_PATH = USER_DIR / "visual_config.json"


def load_visual_config(caminho: Optional[Path] = None) -> Dict[str, Any]:
    """Preferencias visuais gravadas, ou {} se o arquivo nao existe/esta
    corrompido — configuracao cosmetica nunca impede uma analise de rodar.

    `caminho` (default `VISUAL_PATH`) existe para o chamador poder apontar
    para outro arquivo em teste, sem tocar no estado real do usuario.
    """
    alvo = caminho if caminho is not None else VISUAL_PATH
    try:
        return (json.loads(alvo.read_text(encoding="utf-8"))
                if alvo.exists() else {})
    except (OSError, json.JSONDecodeError):
        return {}


def save_visual_config(d: Dict[str, Any],
                       caminho: Optional[Path] = None) -> None:
    """Grava as preferencias. Propaga `OSError` para o chamador avisar."""
    alvo = caminho if caminho is not None else VISUAL_PATH
    alvo.parent.mkdir(parents=True, exist_ok=True)
    alvo.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                    encoding="utf-8")
