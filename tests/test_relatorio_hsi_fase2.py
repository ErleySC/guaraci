"""HSI na fase 2: a ficha anti-vazamento diz so' o que o codigo HSI faz, e `guaraci relatorio` aceita a pasta
de uma execucao HSI (PDF/DOCX/XLSX/TEX; PPTX nao e' gerado para HSI e isso e' dito).

Antes: a ficha herdava o texto espectral (MSC/EMSC, validacao aninhada, holdout de 20%) que o HSI nao faz, e
`guaraci relatorio PASTA_HSI` recusava a pasta (nao ha resumo_modelo.txt)."""
from pathlib import Path

import pytest

import matplotlib
matplotlib.use("Agg")

from guaraci import assistente_vazamento as V
from guaraci import relatorio_cli
from guaraci.config import Config, NOME_RELATORIOS
from guaraci.resumo_parse import N_TESTE_HSI_INDICATIVO, parse_hsi_relatorio


def _montar_pasta_hsi_generica(tmp, **kw):
    """Reaproveita o gerador de cubos sinteticos de test_hsi_pipeline.py (2 classes x 3 objetos, 64x64)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_t_hsi_pipeline_cubos", Path(__file__).with_name("test_hsi_pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._montar_pasta_hsi_generica(tmp, **kw)


def _executar(tmp_path_factory, nome, n_por_classe, **cfg_kw):
    tmp = tmp_path_factory.mktemp(nome)
    raiz = _montar_pasta_hsi_generica(tmp, n_amostras_por_classe=n_por_classe)
    cfg = Config(mode="hsi", hsi_dataset_folder=str(raiz), output_root_folder=str(tmp / "saida"),
                 output_format="png", **cfg_kw)
    import guaraci.pipeline as pq
    from guaraci import guaraci as G
    pq.executar(cfg)                                  # modo hsi: pipeline + hsi_relatorio.txt/.json (o caminho do `guaraci run`)
    G._assistente_pos_execucao(cfg, None, "PT")       # e a ficha, como o `guaraci run` faz
    return cfg, Path(cfg.output_folder)


@pytest.fixture(scope="module")
def pasta_hsi(tmp_path_factory):
    """Padrao: CV por objeto, 2 classes x 3 objetos (6 objetos: poucos, metrica indicativa)."""
    return _executar(tmp_path_factory, "hsi_cv6", 3)


@pytest.fixture(scope="module")
def pasta_hsi_holdout(tmp_path_factory):
    return _executar(tmp_path_factory, "hsi_ho6", 3, hsi_validation="holdout")


@pytest.fixture(scope="module")
def pasta_hsi_cv12(tmp_path_factory):
    """12 objetos avaliados pela CV (>= 10): nao e' 'indicativa'."""
    return _executar(tmp_path_factory, "hsi_cv12", 6)


@pytest.fixture(scope="module")
def pasta_hsi_holdout12(tmp_path_factory):
    """Os mesmos 12 objetos, mas o holdout avalia so' uns poucos: 'indicativa'."""
    return _executar(tmp_path_factory, "hsi_ho12", 6, hsi_validation="holdout")


# ---------------------------------------------------------------- ficha
_IDS_HSI = {"separacao_por_objeto", "pixels", "calibracao_hsi", "preproc_hsi", "segmentacao", "selecao_lvs_hsi",
            "holdout_hsi", "confianca_in_sample", "holdout_intocado", "sessao_hsi", "populacao"}


def _por_id(cfg, lang, resultado=None):
    return {i.id: i for i in V.itens(cfg, resultado=resultado, lang=lang)}


def test_ficha_hsi_tem_itens_proprios_e_o_ramo_espectral_nao_mudou():
    hsi = set(_por_id(Config(mode="hsi"), "PT"))
    esp = set(_por_id(Config(), "PT"))
    assert hsi == _IDS_HSI
    assert esp == {"separacao_por_unidade", "replicas", "preproc_nas_dobras", "selecao_lvs", "holdout_externo",
                   "holdout_intocado", "confundimento", "populacao"}


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_ficha_hsi_nao_afirma_o_que_o_hsi_nao_faz(lang):
    for res in (None, {"hsi_tipo": "cv_objetos", "hsi_n_objetos_avaliados": 12, "hsi_n_objetos_total": 12, "hsi_n_dobras": 5,
                       "hsi_fora_da_amostra": True}):
        it = _por_id(Config(mode="hsi"), lang, res)
        tudo = " ".join(f"{i.titulo} {i.detalhe}" for i in it.values())
        assert "20%" not in tudo and "20 %" not in tudo           # nao ha holdout de 20% fixo
        # MSC/EMSC so' aparecem para dizer que NAO se aplicam
        assert ("nao se aplicam" if lang == "PT" else "do not apply") in it["preproc_hsi"].detalhe
        # decisao por objeto (voto), nao por pixel
        assert ("OBJETO" if lang == "PT" else "OBJECT") in it["pixels"].detalhe


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_ficha_hsi_descreve_o_holdout_quando_o_modo_e_holdout(lang):
    it = _por_id(Config(mode="hsi"), lang, {"hsi_tipo": "holdout", "hsi_n_objetos_avaliados": 3, "hsi_n_objetos_total": 12,
                                            "hsi_fora_da_amostra": False})
    assert "cv_objetos_hsi" not in it and "holdout_hsi" in it
    assert it["selecao_lvs_hsi"].estado == "atencao"
    assert ("sem validacao aninhada" if lang == "PT" else "no nested validation") in it["selecao_lvs_hsi"].titulo
    assert it["confianca_in_sample"].estado == "atencao"                 # dentro da amostra
    assert "3" in it["holdout_hsi"].detalhe and ("INDICATIV" in it["holdout_hsi"].detalhe)
    assert ("Nao ha validacao externa" if lang == "PT" else "no external validation") in it["holdout_hsi"].detalhe


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_ficha_hsi_descreve_a_cv_por_objeto_calibracao_e_eixo(lang):
    res = {"hsi_tipo": "cv_objetos", "hsi_n_dobras": 5, "hsi_n_objetos_avaliados": 12, "hsi_n_objetos_total": 12,
           "hsi_fora_da_amostra": True,
           "hsi_calibracao": {"modo": "branco_escuro", "n_cubos_calibrados": 12, "avisos": ["branco com saturacao parcial"]},
           "hsi_eixo_bandas": {"modo": "reamostrar", "n_bandas": 224, "faixa_nm": [400.0, 1000.0], "n_reamostrados": 2,
                               "desvio_max_nm": 0.4, "tolerancia_nm": 1.0}}
    it = _por_id(Config(mode="hsi"), lang, res)
    assert "holdout_hsi" not in it and "cv_objetos_hsi" in it
    cv = it["cv_objetos_hsi"]
    assert "Clopper-Pearson" in cv.detalhe and "12 de 12" in cv.detalhe.replace(" of ", " de ")
    assert cv.estado == "ok"                                              # 12 objetos: nao e' indicativa
    assert it["selecao_lvs_hsi"].estado == "ok"                           # VLs reescolhidas em cada treino
    assert it["confianca_in_sample"].estado == "ok"                       # fora da amostra
    cal = it["calibracao_hsi"]
    assert "(I - D)/(W - D)" in cal.detalhe and "saturacao" in cal.detalhe and cal.estado == "atencao"
    assert "2" in it["eixo_bandas_hsi"].detalhe and "224" in it["eixo_bandas_hsi"].detalhe
    # poucos objetos avaliados: a CV tambem vira indicativa
    res["hsi_n_objetos_avaliados"] = res["hsi_n_objetos_total"] = 6
    peq = _por_id(Config(mode="hsi"), lang, res)["cv_objetos_hsi"]
    assert peq.estado == "atencao" and "INDICATIV" in peq.detalhe


def test_ficha_gravada_na_execucao_le_o_modo_do_relatorio(pasta_hsi, pasta_hsi_holdout):
    for (_, pasta), esperado, ausente in ((pasta_hsi, "Clopper-Pearson", "holdout de"),
                                          (pasta_hsi_holdout, "holdout de", "Clopper-Pearson")):
        t = (pasta / NOME_RELATORIOS / "ficha_vazamento.md").read_text(encoding="utf-8")
        assert esperado in t and ausente not in t
        assert "20%" not in t


# ---------------------------------------------------------------- hsi_relatorio: aviso de n pequeno = objetos AVALIADOS
def _txt_json(pasta):
    import json
    rel = pasta / NOME_RELATORIOS
    return (rel / "hsi_relatorio.txt").read_text(encoding="utf-8"), json.loads((rel / "hsi_relatorio.json").read_text(encoding="utf-8"))


def test_aviso_indicativa_com_poucos_objetos_avaliados(pasta_hsi):
    _, pasta = pasta_hsi
    txt, j = _txt_json(pasta)
    assert j["validacao"]["tipo"] == "cv_objetos" and j["validacao"]["n_objetos_avaliados"] == 6
    d = parse_hsi_relatorio(txt)
    assert d["tipo"] == "cv_objetos" and d["n_objetos_avaliados"] == 6 and d.get("aviso_n_pequeno")
    assert "indicativa" in txt


def test_aviso_nao_dispara_quando_a_cv_avalia_todos_os_objetos_mas_dispara_no_holdout(pasta_hsi_cv12, pasta_hsi_holdout12):
    txt_cv, j_cv = _txt_json(pasta_hsi_cv12[1])
    assert j_cv["validacao"]["n_objetos_avaliados"] == 12 >= N_TESTE_HSI_INDICATIVO
    assert "indicativa" not in txt_cv and "Aviso: so'" not in txt_cv
    txt_ho, j_ho = _txt_json(pasta_hsi_holdout12[1])
    assert j_ho["validacao"]["tipo"] == "holdout" and j_ho["validacao"]["n_objetos_avaliados"] < N_TESTE_HSI_INDICATIVO
    assert "indicativa" in txt_ho


def test_ler_hsi_pasta_json_e_txt_antigo_concordam(pasta_hsi, tmp_path):
    from guaraci.resumo_parse import ler_hsi_pasta
    _, pasta = pasta_hsi
    h = ler_hsi_pasta(str(pasta / NOME_RELATORIOS))
    assert h["tipo"] == "cv_objetos" and h["fora_da_amostra"] is True
    assert {c["classe"] for c in h["classes"]} == {"madura", "verde"}
    assert all(c["n"] == 3 and c["sens_ic"] for c in h["classes"])
    # execucao antiga: so' o .txt (sem .json) ainda e' lida pelas mesmas chaves
    antigo = tmp_path / "Relatorios"
    antigo.mkdir()
    (antigo / "hsi_relatorio.txt").write_text((pasta / NOME_RELATORIOS / "hsi_relatorio.txt").read_text(encoding="utf-8"),
                                              encoding="utf-8")
    h2 = ler_hsi_pasta(str(antigo))
    assert h2["tipo"] == "cv_objetos" and h2["n_objetos_avaliados"] == h["n_objetos_avaliados"]
    assert [c["classe"] for c in h2["classes"]] == [c["classe"] for c in h["classes"]]


# ---------------------------------------------------------------- guaraci relatorio PASTA_HSI
def test_relatorio_cli_aceita_pasta_hsi_e_gera_os_formatos(pasta_hsi, capsys):
    _, pasta = pasta_hsi
    rc = relatorio_cli.executar([str(pasta), "--formato", "pdf,docx,xlsx,tex"])
    assert rc == 0, capsys.readouterr().out
    rel = pasta / NOME_RELATORIOS
    for ext in ("pdf", "docx", "xlsx", "tex"):
        f = rel / f"relatorio.{ext}"
        assert f.is_file() and f.stat().st_size > 1000, ext
    from pypdf import PdfReader
    t = "\n".join(p.extract_text() or "" for p in PdfReader(str(rel / "relatorio.pdf")).pages)
    assert "HSI" in t
    assert "indicativ" in t.lower()
    assert "Clopper-Pearson" in t and ("CV por objeto" in t or "per-object CV" in t)
    assert "holdout de" not in t.lower().replace("holdout de objetos", "")
    import openpyxl
    nomes = openpyxl.load_workbook(str(rel / "relatorio.xlsx")).sheetnames
    assert any("icha" in n or "eakage" in n for n in nomes)


def test_relatorio_cli_hsi_pptx_explicito_falha_e_todos_pula(pasta_hsi, capsys):
    _, pasta = pasta_hsi
    assert relatorio_cli.executar([str(pasta), "--formato", "pptx"]) != 0
    cap = capsys.readouterr()
    assert "pptx" in (cap.out + cap.err).lower()
    assert relatorio_cli.executar([str(pasta), "--formato", "todos"]) == 0


def test_relatorio_hsi_no_modo_holdout_descreve_o_holdout(pasta_hsi_holdout):
    _, pasta = pasta_hsi_holdout
    assert relatorio_cli.executar([str(pasta), "--formato", "pdf"]) == 0
    from pypdf import PdfReader
    t = "\n".join(p.extract_text() or "" for p in PdfReader(str(pasta / NOME_RELATORIOS / "relatorio.pdf")).pages)
    assert "holdout" in t.lower() and "indicativ" in t.lower()
    assert "CV por objeto (" not in t and "per-object CV (" not in t
