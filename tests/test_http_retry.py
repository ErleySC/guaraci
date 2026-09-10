# -*- coding: utf-8 -*-
"""Testes de `scripts/download_datasets/_http_retry.py` (achado R1, rodada
multiagente 2026-09-10): retry com backoff para urlopen, usado pelos
scripts `baixar_zenodo_*.py`."""
from __future__ import annotations

import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

_RAIZ = Path(__file__).resolve().parents[1]
_SCRIPTS_DL = str(_RAIZ / "scripts" / "download_datasets")
if _SCRIPTS_DL not in sys.path:
    sys.path.insert(0, _SCRIPTS_DL)

from _http_retry import urlopen_com_retry  # noqa: E402


def test_sucesso_de_primeira_nao_espera(monkeypatch):
    chamadas = []
    monkeypatch.setattr("time.sleep", lambda s: chamadas.append(s))
    with patch("urllib.request.urlopen", return_value="ok") as m:
        resultado = urlopen_com_retry("req-qualquer", timeout=10)
    assert resultado == "ok"
    assert m.call_count == 1
    assert chamadas == []   # nenhuma espera -- sucesso de primeira


def test_recupera_apos_falha_transitoria(monkeypatch):
    """2 falhas de rede seguidas de sucesso -- tem que voltar 'ok', nao
    propagar a excecao (a falha era transitoria)."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    efeitos = [urllib.error.URLError("timeout"),
               urllib.error.URLError("timeout"),
               "ok"]

    def _fake_urlopen(req, timeout):
        efeito = efeitos.pop(0)
        if isinstance(efeito, Exception):
            raise efeito
        return efeito

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        resultado = urlopen_com_retry("req-qualquer", timeout=10, tentativas=3)
    assert resultado == "ok"


def test_falha_apos_esgotar_tentativas_continua_sendo_falha_real(monkeypatch):
    """A regra de ouro do gate ('fonte fora do ar = falha') nao pode virar
    verde silencioso so' porque agora existe retry."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError("fonte fora do ar")):
        with pytest.raises(urllib.error.URLError):
            urlopen_com_retry("req-qualquer", timeout=10, tentativas=3)


def test_http_4xx_nao_e_retentado():
    """404 e' erro do pedido (URL mudou), nao instabilidade transitoria --
    tentar de novo so' atrasaria uma falha que retry nao resolve."""
    erro_404 = urllib.error.HTTPError("url", 404, "Not Found", {}, None)
    chamadas = {"n": 0}

    def _fake_urlopen(req, timeout):
        chamadas["n"] += 1
        raise erro_404

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        with pytest.raises(urllib.error.HTTPError):
            urlopen_com_retry("req-qualquer", timeout=10, tentativas=3)
    assert chamadas["n"] == 1   # NAO tentou de novo


def test_http_5xx_e_retentado(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    erro_504 = urllib.error.HTTPError("url", 504, "Gateway Timeout", {}, None)
    efeitos = [erro_504, "ok"]

    def _fake_urlopen(req, timeout):
        efeito = efeitos.pop(0)
        if isinstance(efeito, Exception):
            raise efeito
        return efeito

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        resultado = urlopen_com_retry("req-qualquer", timeout=10, tentativas=3)
    assert resultado == "ok"
