"""Pasta temporaria curta para o pytest no Windows (achado A3-05 da vistoria de 2026-10).

O pipeline recusa cedo uma pasta de saida cujo caminho passaria de 260 caracteres (MAX_PATH do Windows). Os testes que rodam
`pipeline.executar` gravam em `<basetemp>/popen-gwN/<nome do teste>/saida/<amostra>/<modo>/<execucao>/...`: com o basetemp padrao
(dentro do perfil do usuario) ou um `--basetemp` dentro de uma pasta funda, dezenas de testes reprovavam sem defeito nenhum no
produto. No Windows, quando o basetemp passa de `LIMITE`, a sessao troca para uma pasta nova e unica sob `%SystemDrive%/gt`
(a mesma base curta de scripts/camadas_de_teste.py e scripts/verificar_hash_final.py).

Funcoes puras (testaveis sem pytest rodando); o hook em `conftest.py` so' aplica.
"""
from __future__ import annotations

import getpass
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Mapping, Optional

#: basetemp acima disto (em caracteres) e' trocado no Windows. 260 - 75 (subcaminho interno da saida, pipeline.py) - ~75
#: (amostra/modo/execucao) - ~45 (popen-gwN + nome do teste + "saida") deixa ~65; 40 da' folga a subpastas dos testes.
LIMITE = 40
PREFIXO = "auto_"
MANTER = 3   # como o pytest: guarda as 3 ultimas pastas automaticas (prova de uma rodada reprovada)
#: ... e NUNCA apaga uma pasta mais nova que isto: outra sessao de pytest rodando ao mesmo tempo pode estar usando-a (achado ao
#: provar este lote: sessoes paralelas apagavam a pasta viva uma da outra e 2021 testes erravam no setup).
IDADE_MINIMA_S = 48 * 3600


def basetemp_previsto(opcao: Optional[str]) -> str:
    """Caminho absoluto que o pytest usaria: o `--basetemp` dado, ou `<temp>/pytest-of-<usuario>/pytest-NNNN` (com 4 digitos)."""
    if opcao:
        return os.path.abspath(opcao)
    try:
        usuario = getpass.getuser()
    except Exception:  # noqa: BLE001 -- o pytest tambem cai em "unknown" sem usuario
        usuario = "unknown"
    return os.path.join(tempfile.gettempdir(), f"pytest-of-{usuario}", "pytest-9999")


def precisa_trocar(opcao: Optional[str], os_name: str = os.name, limite: int = LIMITE) -> bool:
    return os_name == "nt" and len(basetemp_previsto(opcao)) > limite


def raiz_curta(env: Mapping[str, str] = os.environ) -> Path:
    return Path(env.get("SystemDrive", "C:") + "/") / "gt"


def criar_basetemp_curto(raiz: Path, manter: int = MANTER, idade_minima_s: float = IDADE_MINIMA_S,
                         agora: Optional[float] = None) -> str:
    """Cria uma pasta nova e unica sob `raiz` e apaga as pastas automaticas alem das `manter` mais novas que tenham mais de
    `idade_minima_s` segundos (nenhuma sessao de teste dura tanto: a pasta de uma sessao viva nunca e' apagada)."""
    agora = time.time() if agora is None else agora
    raiz.mkdir(parents=True, exist_ok=True)
    nova = tempfile.mkdtemp(prefix=PREFIXO, dir=str(raiz))
    antigas = sorted((p for p in raiz.iterdir() if p.is_dir() and re.fullmatch(PREFIXO + r"\w+", p.name) and str(p) != nova),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    for velha in antigas[max(manter - 1, 0):]:
        if agora - velha.stat().st_mtime > idade_minima_s:
            shutil.rmtree(velha, ignore_errors=True)
    return nova
