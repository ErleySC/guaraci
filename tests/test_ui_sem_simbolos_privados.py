"""m-22 do parecer de 2026-09-25: as interfaces (web e CLI) nao usam simbolo
privado de `guaraci.pipeline` (`pq._algo`); usam os nomes publicos."""
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def test_interfaces_nao_usam_pq_privado():
    arquivos = list((RAIZ / "src/guaraci/app_tabs").glob("*.py")) + [
        RAIZ / "src/guaraci/app_web.py", RAIZ / "src/guaraci/guaraci.py",
        RAIZ / "src/guaraci/app_logic.py"]
    achados = []
    for p in arquivos:
        for i, linha in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for m in re.finditer(r"\bpq\._(?!_)[A-Za-z]\w*", linha):
                achados.append(f"{p.name}:{i}: {m.group(0)}")
    assert not achados, "\n".join(achados)
