"""Contra-prova do achado S-NOVO-1 (auditoria de seguranca 2026-09-01):
campos de texto livre com caminho de servidor (CSV de predicao, pasta_dados,
arquivo_csv) precisam sumir quando GUARACI_DISABLE_MODEL_UPLOAD esta ativo --
o mesmo gate ja usado para o upload de `.joblib` -- porque um app publico sem
autenticacao expoe esses campos a qualquer visitante remoto, nao so' ao
operador que "sabe o caminho certo".
"""
from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _app(upload_bloqueado: bool) -> None:
    from guaraci.app_tabs import predicao

    def _tok_falso():
        return {
            "success_bg": "#000", "success": "#000",
            "error_bg": "#000", "error": "#000",
        }

    predicao.render(upload_bloqueado, _tok_falso)


def _render_predicao(upload_bloqueado: bool) -> AppTest:
    at = AppTest.from_function(_app, default_timeout=15,
                                kwargs={"upload_bloqueado": upload_bloqueado})
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_campo_caminho_csv_predicao_some_com_upload_bloqueado():
    at = _render_predicao(upload_bloqueado=True)
    rotulos = [ti.label for ti in at.text_input]
    assert "Or local path to CSV" not in rotulos
    # O uploader de CSV continua disponivel -- so' o campo de texto livre
    # some. (O uploader de MODELO some junto, mas por um gate mais antigo
    # ja existente -- upload_bloqueado esconde upld_jbl inteiro, nao so'
    # cam_jbl -- por isso so' 1 file_uploader sobra, nao 2.)
    rotulos_upload = [fu.label for fu in at.get("file_uploader")]
    assert rotulos_upload == ["Upload CSV with new spectra"]


def test_campo_caminho_csv_predicao_disponivel_sem_bloqueio():
    at = _render_predicao(upload_bloqueado=False)
    rotulos = [ti.label for ti in at.text_input]
    assert "Or local path to CSV" in rotulos
