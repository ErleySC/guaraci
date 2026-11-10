"""ICP multielementar na fase 2: valor certificado do MRC (linha Tipo=CRM_CERT), verificacoes de QC que dizem
quando NAO rodaram e a secao multielementar dos relatorios (ANOVA por elemento com a correcao de comparacoes
multiplas, sensibilidade a censura, alertas de QC; sem faixa espectral/SNR).

Antes: nao havia como informar o valor certificado (a recuperacao do MRC nunca era calculada pelo arquivo) e o
PDF/DOCX da execucao multielementar nao trazia a ANOVA por elemento nem os alertas de QC."""
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from guaraci import exemplos as ex
from guaraci import guaraci as g
from guaraci import reports
from guaraci.icp_multielementar import ler_icp_csv, relatorio_qc

CSV_BASE = """Amostra;Tipo;Grupo;Sessao;Cd 111;Pb 208;Zn 66;In 115
A1;amostra;sul;s1;0,12;0,20;5,1;100,0
A2;amostra;sul;s1;0,15;0,40;5,3;99,0
A3;amostra;norte;s2;0,10;0,55;6,8;97,5
A4;amostra;norte;s2;0,30;0,35;7,1;96,0
B1;branco;-;s1;0,001;0,002;0,01;100,5
CRM1;crm;-;s1;0,40;2,00;30,0;98,0
CRM2;crm;-;s2;0,44;1,80;28,0;97,0
"""
LD = "LD;ld;;;0,01;0,01;0,05;\n"
CERT = "CERT;crm_cert;;;0,40;2,00;29,0;\n"


def _ler(tmp_path, texto):
    p = tmp_path / "icp.csv"
    p.write_text(texto, encoding="utf-8")
    return ler_icp_csv(str(p), unidade="mg/kg", coluna_amostra="Amostra", coluna_tipo="Tipo",
                       coluna_grupo="Grupo", coluna_sessao="Sessao")


# ---------------------------------------------------------------- leitura e recuperacao (verdade conhecida)
def test_linha_crm_cert_e_lida_e_a_recuperacao_bate_com_a_conta_a_mao(tmp_path):
    t = _ler(tmp_path, CSV_BASE + LD + CERT)
    assert t.crm_certificado == {"Cd 111": 0.40, "Pb 208": 2.00, "Zn 66": 29.0}
    qc = relatorio_qc(t)
    rec = qc["recuperacao_crm_pct"]
    # media dos dois CRM / certificado: Cd (0,40+0,44)/2/0,40 = 105%; Pb (2,0+1,8)/2/2,0 = 95%; Zn (30+28)/2/29 = 100%
    assert rec["Cd 111"] == pytest.approx(105.0)
    assert rec["Pb 208"] == pytest.approx(95.0)
    assert rec["Zn 66"] == pytest.approx(100.0)
    assert qc["verificacoes"]["recuperacao_crm"] == "ok"
    assert not [a for a in qc["alertas"] if "recuperacao" in a]


def test_recuperacao_fora_de_80_120_vira_alerta(tmp_path):
    t = _ler(tmp_path, CSV_BASE + LD + "CERT;crm_cert;;;0,40;1,00;29,0;\n")      # Pb: (2,0+1,8)/2/1,0 = 190%
    qc = relatorio_qc(t)
    assert qc["recuperacao_crm_pct"]["Pb 208"] == pytest.approx(190.0)
    assert any("Pb 208" in a and "recuperacao" in a for a in qc["alertas"])


def test_sem_crm_cert_a_recuperacao_nao_e_inventada(tmp_path):
    t = _ler(tmp_path, CSV_BASE + LD)
    qc = relatorio_qc(t)
    assert "recuperacao_crm_pct" not in qc
    assert qc["verificacoes"]["recuperacao_crm"] == "sem_valor_certificado"


def test_crm_cert_sem_linhas_crm_medidas(tmp_path):
    sem_crm = "\n".join(l for l in CSV_BASE.splitlines() if ";crm;" not in l) + "\n"
    qc = relatorio_qc(_ler(tmp_path, sem_crm + LD + CERT))
    assert qc["verificacoes"]["recuperacao_crm"] == "sem_linhas_crm"
    assert not qc.get("recuperacao_crm_pct") or all(v is None for v in qc["recuperacao_crm_pct"].values())


def test_mais_de_uma_linha_crm_cert_e_erro(tmp_path):
    with pytest.raises(ValueError, match="CRM_CERT"):
        _ler(tmp_path, CSV_BASE + LD + CERT + CERT.replace("CERT;", "CERT2;"))


def test_branco_acima_do_ld_e_deriva_do_padrao_interno_viram_alerta_e_sem_ld_diz_que_nao_verificou(tmp_path):
    alto = CSV_BASE.replace("B1;branco;-;s1;0,001;0,002;0,01;100,5", "B1;branco;-;s1;0,05;0,002;0,01;100,5")
    t = _ler(tmp_path, alto + LD + CERT)
    qc = relatorio_qc(t)
    assert qc["brancos_acima_lod"] == ["Cd 111"]
    assert any("acima do LOD no branco" in a for a in qc["alertas"])
    assert qc["verificacoes"]["brancos_vs_lod"] == "ok"
    # sem a linha LD: o branco nao e' comparado e isso e' dito
    assert relatorio_qc(_ler(tmp_path, alto + CERT))["verificacoes"]["brancos_vs_lod"] == "sem_lod"
    assert relatorio_qc(_ler(tmp_path, alto + LD + CERT))["verificacoes"]["padrao_interno"] == "sem_padrao_interno"


# ---------------------------------------------------------------- o exemplo traz as linhas de QC
def test_exemplo_multielementar_traz_ld_branco_crm_e_crm_cert(tmp_path):
    ex.gerar("multielementar", str(tmp_path))
    t = ler_icp_csv(str(tmp_path / "dados.csv"), unidade="mg/kg", coluna_amostra="Amostra", coluna_tipo="Tipo",
                    coluna_grupo="Lote", coluna_sessao="Sessao")
    assert t.crm_certificado and t.lod
    qc = relatorio_qc(t)
    assert set(qc["verificacoes"].values()) >= {"ok"}
    assert qc["verificacoes"]["recuperacao_crm"] == "ok" and qc["verificacoes"]["brancos_vs_lod"] == "ok"
    assert all(abs(v - 99.0) < 1e-6 for v in qc["recuperacao_crm_pct"].values())      # (1,04 + 0,94)/2 = 0,99 do certificado
    assert qc["brancos_acima_lod"] == []


# ---------------------------------------------------------------- secao multielementar nos relatorios
def _texto_pdf(dados: bytes) -> str:
    from pypdf import PdfReader
    return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(dados)).pages)


def _texto_docx(dados: bytes) -> str:
    import docx
    d = docx.Document(io.BytesIO(dados))
    partes = [p.text for p in d.paragraphs]
    for t in d.tables:
        for lin in t.rows:
            partes += [c.text for c in lin.cells]
    return "\n".join(partes)


@pytest.fixture(scope="module")
def execucao_icp():
    pasta = tempfile.mkdtemp(prefix="gi")
    ex.gerar("multielementar", pasta)
    g.main(["run", os.path.join(pasta, "config.yaml")])
    rel = next(Path(pasta).rglob("multielementar.json")).parent
    yield str(rel.parent)
    shutil.rmtree(pasta, ignore_errors=True)


_PROJ = {"nome": "Proj", "autor": "A", "instituicao": "I", "tipo": "T"}


@pytest.mark.slow
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_pdf_e_docx_multielementar_trazem_anova_sensibilidade_e_qc_e_nao_a_faixa_espectral(execucao_icp, lang, monkeypatch):
    monkeypatch.setenv("GUARACI_IDIOMA", lang)
    for t in (_texto_pdf(reports.generate_pdf_report(execucao_icp, _PROJ).getvalue()),
              _texto_docx(reports.generate_word_report(execucao_icp, _PROJ).getvalue())):
        assert "Fe 56" in t and "Mn 55" in t                       # tabela da ANOVA por elemento
        assert "FDR" in t and "Benjamini" in t                      # correcao de comparacoes multiplas
        assert "Games-Howell" in t
        assert ("99.0" in t) or ("99,0" in t)                       # recuperacao do MRC
        assert ("Sensibilidade à censura" if lang == "pt" else "Censoring sensitivity") in t       # sensibilidade a censura
        assert "SNR" not in t and "cm-1" not in t and "cm⁻¹" not in t         # nada de faixa espectral


@pytest.mark.slow
def test_relatorio_diz_o_que_o_qc_nao_verificou(execucao_icp, tmp_path, monkeypatch):
    """Sem valor certificado e sem padrao interno o relatorio tem que dizer 'NAO verificado', nunca calar."""
    monkeypatch.setenv("GUARACI_IDIOMA", "pt")
    copia = tmp_path / "exec"
    shutil.copytree(execucao_icp, copia)
    jp = next(copia.rglob("multielementar.json"))
    d = json.loads(jp.read_text(encoding="utf-8"))
    d["qc"]["verificacoes"]["recuperacao_crm"] = "sem_valor_certificado"
    d["qc"].pop("recuperacao_crm_pct", None)
    jp.write_text(json.dumps(d), encoding="utf-8")
    t = _texto_pdf(reports.generate_pdf_report(str(copia), _PROJ).getvalue())
    assert "CRM_CERT" in t and ("NAO" in t or "NÃO" in t)
    assert "99.0" not in t and "99,0" not in t


@pytest.mark.slow
def test_xlsx_e_tex_multielementar(execucao_icp, monkeypatch):
    monkeypatch.setenv("GUARACI_IDIOMA", "pt")
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(reports.generate_excel_report(execucao_icp).getvalue()))
    assert any("ultielement" in n for n in wb.sheetnames), wb.sheetnames
    tex = reports.generate_latex_template(execucao_icp, _PROJ).decode("utf-8")
    assert "Fe 56" in tex and "Benjamini" in tex
    assert "cm-1" not in tex and "cm\\textsuperscript" not in tex
