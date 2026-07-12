"""log.py — ponto UNICO de configuracao de logging do GUARACI (CLAUDE.md P6).

Escopo desta versao (2026-07-13): migra `print()` -> `log.info()` em
`pipeline.py`, ganhando verbosidade controlavel, testabilidade via
`caplog`, e consistencia com os modulos que ja usam `logging`
(classificadores.py, dados_io.py, figuras.py, hardware.py, etc.).

O handler de console usa formato SEM PREFIXO (`%(message)s`) e escreve em
stdout -- de proposito identico ao `print()` que substitui. Isso preserva
o texto exato capturado por `contextlib.redirect_stdout` no CLI
(`guaraci.py:_rodar_pipeline`) e no worker do app web
(`app_tabs/modelo.py`), que hoje fazem parsing por REGEX desse texto para
alimentar o painel de progresso ao vivo (`app_logic.progresso_do_log` /
`figuras_concluidas` / `avisos_do_log`).

NAO faz (escopo maior, fica para depois): reescrever o painel para
consumir logging.Handler/registros estruturados em vez de regex sobre
texto (a visao completa do P6) -- isso muda a arquitetura do painel do
CLI e do app web, nao so' o pipeline, e precisa de projeto proprio.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional


class _StdoutHandler(logging.Handler):
    """Escreve em `sys.stdout` NO MOMENTO do emit, nao numa referencia
    capturada na construcao do handler.

    Necessario porque o CLI (`guaraci.py:_rodar_pipeline`) e o worker do
    app web (`app_tabs/modelo.py`) rodam `executar()` dentro de
    `contextlib.redirect_stdout(buffer)` para capturar o texto e alimentar
    o painel de progresso (parsing por regex em `app_logic.py`). Um
    `logging.StreamHandler(sys.stdout)` comum capturaria a referencia de
    stdout que existia ANTES do redirect (import time), entao o log
    escaparia da captura e o painel ficaria sempre vazio."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stdout.write(self.format(record) + "\n")
        except Exception:  # noqa: BLE001 -- handler de logging: nunca deve
            # derrubar o processo por falha ao formatar/escrever 1 registro;
            # comportamento padrao de logging.Handler.handleError.
            self.handleError(record)


def configurar(nivel: str = "INFO", arquivo: Optional[str] = None) -> logging.Logger:
    """Configura o logger raiz "guaraci". Idempotente (nao duplica handlers
    se chamado mais de uma vez na mesma sessao de processo)."""
    root = logging.getLogger("guaraci")
    root.setLevel(nivel)
    if not any(isinstance(h, _StdoutHandler) for h in root.handlers):
        h = _StdoutHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(h)
    if arquivo and not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        fh = logging.FileHandler(arquivo, encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s | %(message)s"))
        root.addHandler(fh)
    root.propagate = False
    return root
