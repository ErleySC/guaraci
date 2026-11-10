"""Ramos de borda de guaraci.predicao: chave local de assinatura, manifesto, leitura do CSV e fluxo cego sem identificacao.

A chave de assinatura SEMPRE fica numa pasta temporaria (variavel GUARACI_MODEL_KEY_FILE): nenhum teste toca o estado real do usuario.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

import guaraci.predicao as pr
from guaraci.identificacao import CoverageStatus, IdentificationResult


@pytest.fixture
def chave(tmp_path, monkeypatch):
    caminho = tmp_path / "chaves" / "model_signing.key"
    monkeypatch.setenv("GUARACI_MODEL_KEY_FILE", str(caminho))
    return caminho


# ── chave local e manifesto ─────────────────────────────────────────────────
def test_chave_ilegivel_ou_nao_gravavel_vira_none(chave, tmp_path):
    chave.parent.mkdir(parents=True)
    chave.write_text("isto nao e' hexadecimal", encoding="ascii")
    assert pr._chave_local(criar=True) is None                              # chave existente e corrompida: nao recria por cima
    bloqueio = tmp_path / "arquivo_no_lugar_da_pasta"
    bloqueio.write_text("x", encoding="utf-8")
    os.environ["GUARACI_MODEL_KEY_FILE"] = str(bloqueio / "sub" / "k.key")      # o "pai" e' um arquivo: mkdir falha
    assert pr._chave_local(criar=True) is None


def test_chmod_que_falha_no_windows_nao_impede_a_chave(chave, monkeypatch):
    def sem_chmod(*a, **k):
        raise OSError("chmod parcial")
    monkeypatch.setattr(os, "chmod", sem_chmod)
    k = pr._chave_local(criar=True)
    assert isinstance(k, bytes) and len(k) == 32


def test_modelo_sem_chave_gravavel_salva_sem_assinatura_e_assinar_recusa(tmp_path, monkeypatch, caplog):
    bloqueio = tmp_path / "arquivo"
    bloqueio.write_text("x", encoding="utf-8")
    monkeypatch.setenv("GUARACI_MODEL_KEY_FILE", str(bloqueio / "sub" / "k.key"))
    manifesto = {"sha256": "abc"}
    with caplog.at_level("WARNING"):
        assert pr._assinar_no_manifesto(manifesto) is False
    assert "SEM assinatura" in caplog.text
    modelo = tmp_path / "m.joblib"
    modelo.write_bytes(b"conteudo")
    with pytest.raises(OSError, match="chave local"):
        pr.assinar_modelo(str(modelo))


def test_assinar_modelo_inexistente_e_manifesto_previo_valido_ou_corrompido(tmp_path, chave):
    with pytest.raises(FileNotFoundError):
        pr.assinar_modelo(str(tmp_path / "nao_existe.joblib"))
    modelo = tmp_path / "m.joblib"
    modelo.write_bytes(b"conteudo")
    manifesto = tmp_path / "m.joblib.manifest.json"
    manifesto.write_text(json.dumps({"origem": "anotada pelo usuario"}), encoding="utf-8")
    pr.assinar_modelo(str(modelo))
    lido = json.loads(manifesto.read_text(encoding="utf-8"))
    assert lido["origem"] == "anotada pelo usuario" and lido["assinado_manualmente"] is True and pr._CAMPO_ASSINATURA in lido
    manifesto.write_text("{ isto nao e' json", encoding="utf-8")                # corrompido: recomeca do zero
    pr.assinar_modelo(str(modelo))
    assert pr._CAMPO_ASSINATURA in json.loads(manifesto.read_text(encoding="utf-8"))
    manifesto.write_text("[1, 2]", encoding="utf-8")                           # JSON valido mas nao e' dict
    pr.assinar_modelo(str(modelo))
    assert "sha256" in json.loads(manifesto.read_text(encoding="utf-8"))


# ── A1-07: a chave nasce com 0o600 (sem janela legivel antes do chmod) ─────
def test_chave_local_e_criada_ja_com_modo_600_e_exclusiva(chave, monkeypatch):
    chamadas = []
    os_open_real = os.open

    def espiao(caminho, flags, modo=0o777, *a, **k):
        chamadas.append((str(caminho), flags, modo))
        return os_open_real(caminho, flags, modo, *a, **k)
    monkeypatch.setattr(os, "open", espiao)
    k = pr._chave_local(criar=True)
    assert isinstance(k, bytes) and len(k) == 32
    assert chave.read_text(encoding="ascii") == k.hex()
    (cam, flags, modo), = [c for c in chamadas if c[0] == str(chave)]
    assert modo == 0o600 and flags & os.O_EXCL and flags & os.O_CREAT
    if os.name == "posix":
        assert (chave.stat().st_mode & 0o777) == 0o600


# ── A1-06: a carga usa os bytes conferidos, nunca reabre o arquivo ─────────
def test_arquivo_trocado_entre_conferencia_e_carga_nao_e_executado(chave, tmp_path, monkeypatch):
    import io

    import joblib
    modelo = tmp_path / "m.joblib"
    joblib.dump({"ok": 1}, modelo)
    pr.assinar_modelo(str(modelo))
    marcador = tmp_path / "MARCADOR.txt"

    class _Malicioso:
        def __reduce__(self):
            return (marcador.write_text, ("executou",))

    verificar_real = pr._verificar_manifesto

    def verifica_e_troca(caminho):
        dados = verificar_real(caminho)
        joblib.dump({"x": _Malicioso()}, modelo)     # troca DEPOIS da conferencia
        return dados
    monkeypatch.setattr(pr, "_verificar_manifesto", verifica_e_troca)
    recebidos = []
    load_real = joblib.load

    def espiao(arq, *a, **k):
        recebidos.append(arq)
        return load_real(arq, *a, **k)
    monkeypatch.setattr(joblib, "load", espiao)
    assert pr.load_model(str(modelo), confiar=True) == {"ok": 1}
    assert not marcador.exists()
    assert len(recebidos) == 1 and isinstance(recebidos[0], io.BytesIO)


# ── CSV de predicao ─────────────────────────────────────────────────────────
def test_csv_de_predicao_nao_interpretavel_so_com_cabecalho_e_com_valores_nao_numericos(tmp_path, monkeypatch):
    from guaraci import csv_robusto
    original = csv_robusto.ler_csv

    def parser_quebra(*a, **k):
        raise pd.errors.ParserError("linha 2 tem colunas demais")
    monkeypatch.setattr(csv_robusto, "ler_csv", parser_quebra)
    with pytest.raises(ValueError, match="interpretar o arquivo como CSV"):
        pr.load_prediction_csv(str(tmp_path / "x.csv"))
    monkeypatch.setattr(csv_robusto, "ler_csv", original)
    so_cab = tmp_path / "cab.csv"
    so_cab.write_text("4000,4001,4002\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nenhuma coluna de dado|so' cabecalho"):
        pr.load_prediction_csv(str(so_cab))
    texto = tmp_path / "txt.csv"
    texto.write_text("4000,4001,4002\n1,abc,3\n4,5,6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="converter os valores espectrais"):
        pr.load_prediction_csv(str(texto))


# ── predict_samples, quantify_sample e predict_blind ────────────────────────
def test_predizer_sem_eixo_de_numeros_de_onda_e_erro(modelo_e_dados):
    pkg, X_novos, wn = modelo_e_dados
    with pytest.raises(ValueError, match="wn_new nao pode ser None"):
        pr.predict_samples(pkg, X_novos, None)


def test_predizer_com_pacote_sem_referencia_de_cv_nao_traz_as_colunas_de_referencia(modelo_e_dados):
    pkg, X_novos, wn = modelo_e_dados
    pacote_antigo = {k: v for k, v in pkg.items() if k not in ("ad_cv_rejeitadas", "ad_cv_n")}
    df = pr.predict_samples(pacote_antigo, X_novos, wn)
    assert "AD_ref_cv_rejeitadas" not in df.columns


def test_quantificar_especie_sem_modelo_de_regressao_bloqueia_com_motivo(modelo_e_dados):
    pkg, X_novos, wn = modelo_e_dados
    ident = IdentificationResult(classe_identificada="EspecieSemModelo|S", candidatos_ambiguos=[],
                                 cobertura_status=CoverageStatus.VALIDATED, alpha_alcancavel=0.05, escores={})
    q = pr.quantify_sample({**pkg, "regressao_por_especie": {}}, np.zeros(X_novos.shape[1]), ident)
    assert q.motivo_bloqueio == "sem_modelo_de_regressao_para_especie" and q.especie_usada == "EspecieSemModelo"


def test_fluxo_cego_sem_ensemble_de_identificacao_nao_identifica_nada(modelo_e_dados):
    pkg, X_novos, wn = modelo_e_dados
    df, resultados = pr.predict_blind({**pkg, "identification_ensemble": None}, X_novos[:2], wn)
    assert len(resultados) == 2 and all(r.identificacao.classe_identificada is None for r in resultados)
