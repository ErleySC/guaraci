"""guaraci.icp_multielementar: leitura, censura e QC (fixtures sinteticas)."""
import math

import numpy as np
import pytest

from guaraci.icp_multielementar import (
    aplicar_censura, ler_icp_csv, normalizar_elemento, relatorio_qc, sensibilidade_censura)

CSV_PONTO_VIRGULA = """Amostra;Tipo;Grupo;Sessao;Cd 111;Pb 208;Zn 66;In 115
A1;amostra;sul;s1;0,12;<LOD;5,1;100,0
A2;amostra;sul;s1;0,15;0,40;5,3;99,0
A3;amostra;norte;s2;<0,05;0,55;6,8;97,5
A4;amostra;norte;s2;0,30;<0,05;7,1;96,0
B1;branco;-;s1;<0,05;0,02;0,01;100,5
CRM;crm;-;s1;0,45;1,90;30,0;98,0
"""

CSV_VIRGULA = """sample,type,Cd111,Cd 228.802 (mg/kg)
x,sample,0.1,0.11
"""


def _escrever(tmp_path, texto, nome="icp.csv"):
    p = tmp_path / nome
    p.write_text(texto, encoding="utf-8")
    return str(p)


def _ler(tmp_path, texto=CSV_PONTO_VIRGULA, **kw):
    args = dict(unidade="mg/kg", coluna_amostra="Amostra", coluna_tipo="Tipo",
                coluna_grupo="Grupo", coluna_sessao="Sessao")
    args.update(kw)
    return ler_icp_csv(_escrever(tmp_path, texto), **args)


@pytest.mark.parametrize("cab, esperado", [
    ("Cd 111", "Cd 111"), ("Cd111", "Cd 111"), ("111Cd", "Cd 111"),
    ("Cd 228.802", "Cd 228.802"), ("Cd228,802 (mg/kg)", "Cd 228.802"),
    ("Ca 317.933 [ppm]", "Ca 317.933"), ("Pb", "Pb"),
    ("Amostra", None), ("Data", None), ("Xx 12", None), ("Sessao", None),
])
def test_normalizacao_do_nome_do_elemento(cab, esperado):
    assert normalizar_elemento(cab) == esperado


def test_le_csv_com_ponto_e_virgula_e_decimal_virgula(tmp_path):
    t = _ler(tmp_path)
    assert t.elementos == ["Cd 111", "Pb 208", "Zn 66", "In 115"]
    assert t.valores[0, 0] == pytest.approx(0.12)
    assert t.tipos == ["amostra"] * 4 + ["branco", "crm"]
    assert t.grupos[:4] == ["sul", "sul", "norte", "norte"]
    assert t.unidade == "mg/kg"


def test_tokens_censurados_ficam_marcados_com_o_limite(tmp_path):
    t = _ler(tmp_path)
    j_pb, j_cd = t.elementos.index("Pb 208"), t.elementos.index("Cd 111")
    assert t.censurado[0, j_pb] and math.isnan(t.limite_censura[0, j_pb])   # <LOD sem numero
    assert t.censurado[2, j_cd] and t.limite_censura[2, j_cd] == pytest.approx(0.05)
    assert np.isnan(t.valores[2, j_cd])
    assert t.fracao_censurada()["Pb 208"] == pytest.approx(0.5)


def test_csv_com_virgula_e_cabecalhos_variados(tmp_path):
    t = ler_icp_csv(_escrever(tmp_path, CSV_VIRGULA), unidade="mg/kg",
                    coluna_amostra="sample", coluna_tipo="type")
    assert t.elementos == ["Cd 111", "Cd 228.802"]


@pytest.mark.parametrize("unidade", ["", None, "gramas"])
def test_sem_unidade_reconhecida_recusa(tmp_path, unidade):
    with pytest.raises(ValueError, match="unidade obrigatoria"):
        _ler(tmp_path, unidade=unidade)


def test_cabecalhos_ambiguos_sao_erro_nao_adivinhacao(tmp_path):
    texto = "Amostra;Cd111;Cd 111\nA;1;2\n"
    with pytest.raises(ValueError, match="ambiguos"):
        ler_icp_csv(_escrever(tmp_path, texto), unidade="mg/kg", coluna_amostra="Amostra")


def test_valor_nao_numerico_e_nao_censura_e_erro(tmp_path):
    texto = "Amostra;Cd 111\nA;abc\n"
    with pytest.raises(ValueError, match="nao numerico"):
        ler_icp_csv(_escrever(tmp_path, texto), unidade="mg/kg", coluna_amostra="Amostra")


def test_tipo_de_linha_desconhecido_e_erro(tmp_path):
    with pytest.raises(ValueError, match="tipo de linha desconhecido"):
        _ler(tmp_path, texto=CSV_PONTO_VIRGULA.replace("crm;", "xyz;"))


def test_arquivo_grande_demais_e_recusado(tmp_path, monkeypatch):
    import guaraci.icp_multielementar as icp
    monkeypatch.setattr(icp, "_MAX_BYTES", 10)
    with pytest.raises(ValueError, match="excede o limite"):
        _ler(tmp_path)


def test_politica_substituir_usa_lod_sobre_raiz_de_2_e_exige_limite(tmp_path):
    t = _ler(tmp_path)
    with pytest.raises(ValueError, match="sem limite"):
        aplicar_censura(t, politica="substituir", limiar=0.6)    # Pb '<LOD' sem numero
    X, els, reg = aplicar_censura(t, politica="substituir", limiar=0.6, lod={"Pb 208": 0.08})
    i_cd = els.index("Cd 111")
    assert X[2, i_cd] == pytest.approx(0.05 / math.sqrt(2))
    assert X[0, els.index("Pb 208")] == pytest.approx(0.08 / math.sqrt(2))
    assert reg["elementos_excluidos"] == [] and X.shape == (4, 4)


def test_politica_excluir_tira_linhas_com_ausencia(tmp_path):
    X, els, reg = aplicar_censura(_ler(tmp_path), politica="excluir", limiar=0.6)
    assert X.shape[0] == 1 and reg["linhas_descartadas"] == 3      # so' A2 nao tem censura
    assert not np.isnan(X).any()


CSV_COM_VAZIA = """Amostra;Tipo;Grupo;Sessao;Zn 66;Cu 63
A1;amostra;a;s1;5,1;1,0
A2;amostra;a;s1;;1,1
A3;amostra;b;s1;6,8;<0,05
A4;amostra;b;s1;7,1;1,3
"""


@pytest.mark.parametrize("politica", ["substituir", "excluir"])
def test_celula_vazia_nao_e_censura_e_a_linha_sai_nas_duas_politicas(tmp_path, politica):
    t = _ler(tmp_path, CSV_COM_VAZIA)
    assert t.censurado.sum() == 1                      # so' '<0,05'; a vazia NAO e' censura
    X, els, reg = aplicar_censura(t, politica=politica, limiar=0.6)
    assert not np.isnan(X).any()                       # nada de NaN chegando ao modelo
    assert reg["celulas_ausentes"] == 1
    assert reg["linhas_descartadas_por_ausencia"] == 1  # A2
    assert reg["linhas_descartadas_por_censura"] == (1 if politica == "excluir" else 0)  # A3
    assert reg["linhas_descartadas"] == (2 if politica == "excluir" else 1)
    assert 1 not in reg["indices_amostra"]             # A2 fora
    if politica == "substituir":                       # a censurada vira LOD/raiz(2); a vazia nunca vira numero
        assert X[1, els.index("Cu 63")] == pytest.approx(0.05 / math.sqrt(2))


def test_limiar_exclui_o_elemento_muito_censurado(tmp_path):
    X, els, reg = aplicar_censura(_ler(tmp_path), politica="excluir", limiar=0.4)
    assert "Pb 208" in reg["elementos_excluidos"] and "Pb 208" not in els


def test_sensibilidade_roda_as_duas_politicas_e_diz_se_muda(tmp_path):
    t = _ler(tmp_path)

    def concl(X, els, reg):
        return {e: bool(np.nanmean(X[:, j]) > 0.2) for j, e in enumerate(els)}

    out = sensibilidade_censura(t, concl, limiar=0.6, lod={"Pb 208": 0.08})
    assert set(out) >= {"substituir", "excluir", "muda", "diferencas"}
    assert isinstance(out["muda"], bool)


def test_qc_recuperacao_brancos_e_padrao_interno(tmp_path):
    t = _ler(tmp_path, padrao_interno="In 115")
    qc = relatorio_qc(t, crm_certificado={"Cd 111": 0.50, "Zn 66": 30.0},
                      faixa_recuperacao=(95.0, 105.0), lod={"Pb 208": 0.01, "Zn 66": 0.05})
    assert qc["recuperacao_crm_pct"]["Cd 111"] == pytest.approx(90.0)
    assert qc["recuperacao_crm_pct"]["Zn 66"] == pytest.approx(100.0)
    assert any("Cd 111" in a and "fora de 95-105%" in a for a in qc["alertas"])
    assert qc["brancos_acima_lod"] == ["Pb 208"]
    assert qc["padrao_interno"]["elemento"] == "In 115"
    assert "conforme" not in str(qc).lower()


# --- checkup 2026-11 (familia elementar/RMN/IMS/generico) -------------------

CSV_LD_TEXTO = """Amostra;Tipo;Grupo;Sessao;Cd 111;Zn 66;Rh 103
LD;LD;;;0,005;0,01;
A1;amostra;sul;s1;<LD;5,1;100,0
A2;amostra;sul;s1;0,015;5,3;99,0
A3;amostra;norte;s2;0,020;6,8;97,5
A4;amostra;norte;s2;<LD;7,1;96,0
B1;branco;-;s1;0,009;<LD;100,5
"""


def test_ld_sem_numero_usa_a_linha_tipo_ld_do_arquivo(tmp_path):
    """'<LD' (texto, sem numero) e' o formato mais comum dos laudos; com a linha
    Tipo=LD no proprio CSV a politica 'substituir' deixa de falhar."""
    t = _ler(tmp_path, CSV_LD_TEXTO)
    assert t.lod == {"Cd 111": 0.005, "Zn 66": 0.01}
    assert t.tipos[0] == "lod" and t.fracao_censurada()["Cd 111"] == 0.5
    X, els, reg = aplicar_censura(t, politica="substituir", limiar=0.6)
    assert X.shape == (4, 3)
    assert X[0, els.index("Cd 111")] == pytest.approx(0.005 / math.sqrt(2))
    assert relatorio_qc(t)["brancos_acima_lod"] == ["Cd 111"]      # o LD do arquivo vale no QC


def test_padrao_interno_nunca_entra_na_matriz_modelada(tmp_path):
    """O PI (Rh/In) e' adicionado em concentracao constante: a sua variacao e'
    deriva do instrumento e correlaciona com a ordem de corrida, nao com a amostra."""
    t = _ler(tmp_path, CSV_LD_TEXTO, padrao_interno="Rh 103")
    for pol in ("substituir", "excluir"):
        X, els, reg = aplicar_censura(t, politica=pol, limiar=0.6)
        assert "Rh 103" not in els and X.shape[1] == 2
        assert reg["padrao_interno_fora_do_modelo"] == "Rh 103"
    assert relatorio_qc(t)["padrao_interno"]["elemento"] == "Rh 103"   # continua no QC


def test_excluir_por_elemento_nao_derruba_linha_de_outro_elemento(tmp_path):
    """Na ANOVA por elemento, a politica 'excluir' tira so' a celula censurada:
    Zn (sem censura) mantem as 4 amostras mesmo com Cd censurado em 2."""
    t = _ler(tmp_path, CSV_LD_TEXTO)
    X, els, reg = aplicar_censura(t, politica="excluir", limiar=0.6)
    assert X.shape[0] == 2                                   # modelo: linha inteira sai
    X, els, reg = aplicar_censura(t, politica="excluir", limiar=0.6, por_elemento=True)
    assert X.shape[0] == 4
    assert np.isnan(X[:, els.index("Cd 111")]).sum() == 2
    assert not np.isnan(X[:, els.index("Zn 66")]).any()
