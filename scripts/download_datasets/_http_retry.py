# -*- coding: utf-8 -*-
"""_http_retry.py -- Retry com backoff para `urllib.request.urlopen`,
compartilhado pelos scripts `baixar_zenodo_*.py`.

Achado de auditoria (rodada multiagente 2026-09-10, R1): os 3 jobs de CI
que baixam do Zenodo (`hplc-azeite`, `gcims-urina`, `eem-zenodo`) falharam
por HTTP 504/timeout em 2026-09-08 -- os scripts não tentavam de novo,
diferente do job do Corn (`curl --retry 3`).

Preserva a semântica do gate: falha depois de esgotar as tentativas
CONTINUA sendo falha real (a exceção original é relançada, não engolida) --
"fonte fora do ar" ainda derruba o job, só não mais na primeira soletrada
de instabilidade transitória.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import IO, Any


def urlopen_com_retry(req: "urllib.request.Request", *, timeout: float,
                       tentativas: int = 3, espera_base: float = 2.0
                       ) -> IO[Any]:
    """Como `urllib.request.urlopen(req, timeout=timeout)`, mas tenta de
    novo (backoff exponencial: `espera_base * 2**tentativa`) em erro de
    rede/timeout/HTTP 5xx. HTTP 4xx (ex.: 404) NÃO é retentado -- é erro
    do pedido, não instabilidade transitória da fonte."""
    ultimo_erro: Exception | None = None
    for tentativa in range(tentativas):
        try:
            return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise
            ultimo_erro = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            ultimo_erro = e
        if tentativa < tentativas - 1:
            espera = espera_base * (2 ** tentativa)
            print(f"[AVISO] Falha de rede ({ultimo_erro}) -- tentativa "
                  f"{tentativa + 1}/{tentativas}, nova tentativa em "
                  f"{espera:.0f}s...")
            time.sleep(espera)
    assert ultimo_erro is not None
    raise ultimo_erro
