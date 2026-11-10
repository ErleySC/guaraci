# -*- coding: utf-8 -*-
"""Relatorios num idioma so' e figuras com no maximo 3 paineis (pedido de 2026-11-06).

O model card em portugues trazia "Total samples", "Accuracy (CV)", "N samples outside DModX" e
"Quantificacao de o teor ... em a matriz analisada"; em ingles, metade do card continuava em
portugues. Estes testes geram cada relatorio nos dois idiomas e procuram palavras do OUTRO idioma
(`idioma_relatorio.vazamentos`). Identificadores entre crases (chaves de config, ids de checagem),
nomes de classe e citacoes bibliograficas sao legitimos e ficam de fora da busca.
"""
from __future__ import annotations

import os
import re
import types

import numpy as np
import pytest

from guaraci import auditoria_delineamento as aud
from guaraci import idioma_relatorio as i18n
from guaraci.config import Config

# ---------------------------------------------------------------------------
#  Unidades: traducao, contracoes, rotulos
# ---------------------------------------------------------------------------


def test_idioma_padrao_e_o_do_menu_e_segue_a_escolha_gravada(monkeypatch, tmp_path):
    """Quem nunca escolheu idioma ve menu em ingles; relatorio e figuras tem de sair no mesmo."""
    monkeypatch.delenv("GUARACI_IDIOMA", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path)); monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert i18n.idioma() == "en"
    (tmp_path / ".guaraci").mkdir()
    (tmp_path / ".guaraci" / ".cli_wizard_done").write_text("PT", encoding="utf-8")
    assert i18n.idioma() == "pt"
    monkeypatch.setenv("GUARACI_IDIOMA", "en")
    assert i18n.idioma() == "en"
    assert i18n.idioma("pt") == "pt"


def test_menu_sem_escolha_e_relatorio_no_mesmo_idioma(monkeypatch, tmp_path):
    import guaraci.guaraci as g
    monkeypatch.delenv("GUARACI_IDIOMA", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path)); monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(g, "_LANG_FLAG", tmp_path / ".guaraci" / ".cli_wizard_done")
    monkeypatch.setitem(g._STATE, "lang", "EN")
    g._sincronizar_idioma()
    assert g._lang() == "EN" and i18n.idioma() == "en"


def test_contracoes_do_portugues():
    assert (i18n.contrair("Quantificacao de o teor do analito em a matriz analisada, por a classe")
            == "Quantificacao do teor do analito na matriz analisada, pela classe")
    assert i18n.contrair("dados de amostras em uma matriz") == "dados de amostras em uma matriz"


@pytest.mark.parametrize("chave", sorted(i18n.ROTULOS_RESUMO))
def test_rotulos_do_resumo_nos_dois_idiomas(chave):
    pt, en = i18n.rotulo_resumo(chave, "pt"), i18n.rotulo_resumo(chave, "en")
    assert not i18n.vazamentos(pt, "pt"), (chave, pt)
    assert not i18n.vazamentos(en, "en"), (chave, en)


def test_vocabulario_dos_perfis_embutidos_em_ingles():
    from guaraci.perfil_matriz import load_profile
    for nome in ("generico", "bancada", "celular", "scanner", "nir_ft_difuso"):
        voc = i18n.vocabulario(load_profile(nome).vocabulario, "en")
        assert not i18n.vazamentos(" ".join(voc.values()), "en"), (nome, voc)


# ---------------------------------------------------------------------------
#  Toda mensagem da auditoria de delineamento tem traducao
# ---------------------------------------------------------------------------
def _mae(sessoes_por_classe, niveis=1):
    """rotulos e mae_id no formato real (sessao = tudo antes do ultimo token)."""
    rot, mae = [], []
    for classe, sessoes in sessoes_por_classe.items():
        for s in range(sessoes):
            for n in range(niveis):
                rot.append(classe)
                mae.append(f"{classe}-0{s + 1}-01-2099-A{n + 1}.05")
    return np.array(rot), np.array(mae)


def _achados():
    cfg = lambda **kw: types.SimpleNamespace(**{"grouping_guarantee": "high", "level": "N1",  # noqa: E731
                                                "matrix_profile": "generico", **kw})
    rot2, mae2 = _mae({"A": 2, "B": 2})
    rot_conf, mae_conf = _mae({"A": 1, "B": 3})
    rot_fino, mae_fino = _mae({"A": 2, "B": 2}, niveis=3)
    rot_um, mae_um = _mae({"A": 1, "B": 1}, niveis=3)
    rot_20, mae_20 = _mae({"A": 20, "B": 20})
    misto_rot = np.array(["A", "B", "A", "B"])
    misto_mae = np.array(["g1", "g1", "g2", "g2"])
    X = np.random.default_rng(0).normal(size=(4, 30))
    out = [
        aud.check_grouping(cfg()), aud.check_grouping(cfg(grouping_guarantee="medium")),
        aud.check_grouping(cfg(grouping_guarantee="none")),
        aud.check_grupo_em_varias_classes(rot2, None), aud.check_grupo_em_varias_classes(rot2, mae2),
        aud.check_grupo_em_varias_classes(misto_rot, misto_mae),
        aud.check_class_session_confounding(rot2, None),
        aud.check_class_session_confounding(np.array(["A", "B"]), np.array(["S-01-01-2099-A1.0"] * 2)),
        aud.check_class_session_confounding(rot2, mae2),
        aud.check_class_session_confounding(rot_conf, mae_conf),
        aud.check_unidade_independente(rot2, None),
        aud.check_unidade_independente(rot2, mae2, "sessao"),
        aud.check_unidade_independente(np.array(["A", "B"]), np.array(["S-01-01-2099-A1.0", "S-02-01-2099-A1.0"]),
                                       "sessao"),
        aud.check_unidade_independente(misto_rot, np.array(["x1", "x2", "x3", "x4"]), "mae_id", "medium"),
        aud.check_unidade_independente(rot2, mae2),
        aud.check_unidade_independente(rot_fino, mae_fino),
        aud.check_unidade_independente(np.array(list(rot_um) + ["A"] * 3),
                                       np.array(list(mae_um) + [f"A-09-01-2099-A{n}.05" for n in range(3)])),
        aud.check_classe_unica(rot2, cfg()), aud.check_classe_unica(np.array(["A"] * 4), cfg(level="N3")),
        aud.check_classe_unica(np.array(["A"] * 4), cfg()),
        aud.check_insufficient_n(rot2, None), aud.check_insufficient_n(rot2, mae2),
        aud.check_insufficient_n(rot_20, mae_20),
        aud.check_validation_use_range(None, cfg()),
        aud.check_validation_use_range(np.array([np.nan]), cfg()),
        aud.check_validation_use_range(np.array([1.0, 2.0]), cfg()),
        aud.check_external_validation(),
        aud.check_duplicates(X, np.arange(30.0), misto_rot, None, None),
        aud.check_duplicates(np.vstack([X[:3], X[:1]]), np.arange(30.0), misto_rot, None, None),
    ]
    out += aud.run_audit(X, np.arange(30.0), misto_rot, cfg(), silenciar={"duplicatas": "medido a parte"})
    return out


def test_cada_ramo_da_auditoria_e_coberto():
    mensagens = {a.mensagem for a in _achados()}
    assert len(mensagens) >= 24     # os ramos acima produzem mensagens distintas


@pytest.mark.parametrize("achado", _achados(), ids=lambda a: f"{a.nome}-{a.severidade}")
def test_auditoria_traduzida_sem_portugues(achado):
    en = i18n.achado(achado.mensagem, "en")
    assert en != achado.mensagem, achado.mensagem
    assert not i18n.vazamentos(en, "en", ignorar=("medido a parte",)), en


@pytest.mark.parametrize("achado", _achados(), ids=lambda a: f"{a.nome}-{a.severidade}")
def test_auditoria_em_portugues_sem_ingles(achado):
    pt = i18n.achado(achado.mensagem, "pt")
    assert not i18n.vazamentos(pt, "pt", ignorar=("StratifiedKFold", "holdout")), pt


# ---------------------------------------------------------------------------
#  Model card e relatorios gerados de uma execucao de verdade
# ---------------------------------------------------------------------------
def _texto_relatorio(caminho: str) -> str:
    ext = caminho.rsplit(".", 1)[-1]
    if ext in ("md", "tex", "txt"):
        with open(caminho, encoding="utf-8") as fh:
            txt = fh.read()
        if ext == "tex":
            txt = txt.split("\\begin{thebibliography}")[0]
            txt = re.sub(r"\\[A-Za-z]+(\[[^\]]*\])?(\{[^}]*\})?", " ", txt)   # comandos LaTeX
        return txt
    if ext == "docx":
        import docx
        d = docx.Document(caminho)
        partes = [p.text for p in d.paragraphs]
        for tb in d.tables:
            partes += [c.text for r in tb.rows for c in r.cells]
        return "\n".join(partes)
    if ext == "pptx":
        import pptx
        return "\n".join(sh.text_frame.text for sl in pptx.Presentation(caminho).slides
                         for sh in sl.shapes if sh.has_text_frame)
    if ext == "xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(caminho)
        partes = []
        for ws in wb.worksheets:
            partes.append(ws.title)
            if ws.max_row > 40:          # resumo bruto / tabelas: so' o titulo e o cabecalho
                partes.append(str(ws.cell(1, 1).value))
                continue
            partes += [str(v) for row in ws.iter_rows(values_only=True) for v in row if v is not None]
        return "\n".join(partes)
    if ext == "pdf":
        pypdf = pytest.importorskip("pypdf")
        return "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(caminho).pages)
    raise AssertionError(ext)


def _sem_referencias(txt: str) -> str:
    """Tira as linhas de citacao bibliografica (titulos de artigo em ingles sao legitimos)."""
    return "\n".join(linha for linha in txt.splitlines()
                     if not re.search(r"doi|et al|\(\d{4}\)|^\s*\[\d+\]", linha))


_IGNORAR = ("PLS-DA", "OPLS-DA", "DD-SIMCA", "github.com", "StratifiedKFold", "GroupKFold",
            "holdout", "Model Card")


#: textos de figura gravados por idioma durante as execucoes do fixture `execucoes`.
_TEXTOS_FIGURA: dict = {"pt": [], "en": []}


@pytest.fixture(scope="module")
def execucoes(tmp_path_factory):
    """Uma execucao N3 (quantificacao, a do model card enviado) e uma N2 (classificacao com DD-SIMCA,
    OPLS-DA, Wold e figuras detalhadas) em cada idioma + todos os relatorios. Os textos de toda figura
    gravada ficam em _TEXTOS_FIGURA."""
    pytest.importorskip("fpdf")
    pytest.importorskip("docx")
    pytest.importorskip("pptx")
    pytest.importorskip("openpyxl")
    import matplotlib.figure as mfig
    from guaraci import exemplos, relatorio_cli
    from guaraci import pipeline as pl
    from guaraci.config_io import load_config
    from guaraci.idioma_figuras import textos_da_figura
    saidas = {}
    antes = os.environ.get("GUARACI_IDIOMA")
    savefig_original = mfig.Figure.savefig
    lang_atual = ["pt"]

    def _savefig(self, *a, **k):
        self.canvas.draw()
        _TEXTOS_FIGURA[lang_atual[0]].extend(textos_da_figura(self))
        return savefig_original(self, *a, **k)

    mfig.Figure.savefig = _savefig
    try:
        for lang in ("pt", "en"):
            os.environ["GUARACI_IDIOMA"] = lang
            lang_atual[0] = lang
            for cenario, extras in (("quantificar", {}),
                                    ("classificar", dict(run_ddsimca=True, run_opls=True, run_wold=True,
                                                         detailed_figures=True))):
                base = tmp_path_factory.mktemp(f"run_{cenario}_{lang}")
                res = exemplos.gerar(cenario, str(base))
                cfg = load_config(os.path.join(res["pasta"], "config.yaml"))
                cfg.n_permutations = 19
                for k, v in extras.items():
                    setattr(cfg, k, v)
                pl.executar(cfg)
                run = cfg.output_folder
                assert relatorio_cli.executar([run, "--formato", "todos", "--nome", "Teste"]) == 0
                saidas[lang if cenario == "quantificar" else f"{lang}_{cenario}"] = run
    finally:
        mfig.Figure.savefig = savefig_original
        if antes is None:
            os.environ.pop("GUARACI_IDIOMA", None)
        else:
            os.environ["GUARACI_IDIOMA"] = antes
    return saidas


@pytest.mark.slow
@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("arquivo", ["model_card.md", "resumo_modelo.txt", "relatorio.pdf", "relatorio.docx",
                                     "relatorio.xlsx", "relatorio.pptx", "relatorio.tex"])
def test_relatorio_num_idioma_so(execucoes, lang, arquivo):
    from guaraci.config import NOME_RELATORIOS, pasta_execucao
    caminho = os.path.join(pasta_execucao(execucoes[lang], NOME_RELATORIOS), arquivo)
    assert os.path.isfile(caminho), caminho
    txt = _sem_referencias(_texto_relatorio(caminho))
    # os rotulos dos dados aparecem tambem escapados no .tex ("especie\\_A")
    assert not i18n.vazamentos(txt, lang, ignorar=_IGNORAR + ("especie_A", "especie_B", "especie\\_A", "especie\\_B")), arquivo


@pytest.mark.slow
def test_model_card_pt_sem_os_vazamentos_relatados(execucoes):
    from guaraci.config import NOME_RELATORIOS, pasta_execucao
    with open(os.path.join(pasta_execucao(execucoes["pt"], NOME_RELATORIOS), "model_card.md"), encoding="utf-8") as fh:
        card = fh.read()
    for vazado in ("Total samples", "Exact duplicates", "Accuracy (CV)", "N samples outside DModX",
                   "de o teor", "em a matriz"):
        assert vazado not in card, vazado
    assert "Quantificacao do teor do analito de interesse na matriz analisada" in card


@pytest.mark.slow
def test_model_card_en_inteiro_em_ingles(execucoes):
    from guaraci.config import NOME_RELATORIOS, pasta_execucao
    with open(os.path.join(pasta_execucao(execucoes["en"], NOME_RELATORIOS), "model_card.md"), encoding="utf-8") as fh:
        card = fh.read()
    for secao in ("## 1. Model Details", "## 2. Intended Use", "## Design Audit", "## Read first"):
        assert secao in card, secao
    for vazado in ("Detalhes do Modelo", "Uso Pretendido", "Auditoria de Delineamento", " nao ", " sessao "):
        assert vazado not in card, vazado


# ---------------------------------------------------------------------------
#  Nenhuma imagem com mais de 3 paineis
# ---------------------------------------------------------------------------
def test_legendas_das_figuras_nos_dois_idiomas():
    from guaraci.grupos_figuras import GRUPOS_FIGURAS
    for prefixo, _pasta in GRUPOS_FIGURAS:
        nome = prefixo.rstrip("_") + "_x.png"
        pt, en = i18n.titulo_figura(nome, "pt"), i18n.titulo_figura(nome, "en")
        assert pt != nome.replace("_", " ")[:-4], nome      # tem titulo na tabela
        assert not i18n.vazamentos(en, "en", ignorar=_IGNORAR), (nome, en)
    assert i18n.titulo_figura("fig_ddsimca_aceitacao_02.png", "pt").endswith("(parte 2)")
    assert i18n.titulo_figura("figS2d_williams_03.png", "en") == "Williams plot (part 3)"


def test_paginas_e_nomes():
    from guaraci.figuras import nome_pagina, paginas
    assert paginas(range(8)) == [[0, 1, 2], [3, 4, 5], [6, 7]]
    assert nome_pagina("fig", 1, 1) == "fig" and nome_pagina("fig", 2, 3) == "fig_02"


def _classes(n):
    rot = np.repeat([f"C{i}" for i in range(n)], 6)
    rng = np.random.default_rng(0)
    return rot, rng


def test_ddsimca_e_loadings_um_grafico_por_arquivo_hca_em_3(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from sklearn.decomposition import PCA
    from guaraci import figuras as fg
    fg.resetar_falhas_salvamento()
    cfg = Config(output_format="png")
    rot, rng = _classes(7)
    cores = {c: "#336699" for c in set(rot)}
    scores = {c: {"T2_norm": rng.uniform(0.1, 3, rot.size), "Q_norm": rng.uniform(0.1, 3, rot.size),
                  "f": rng.uniform(0, 3, rot.size), "f_crit": 2.0, "T2_ucl": 10.0, "Q_ucl": 1.0,
                  "n_comp": 2, "n_train": 6, "h0": 2.0, "Nh": 3.0, "v0": 1.0, "Nv": 3.0}
              for c in sorted(set(rot))}
    fg.fig_sprint3_ddsimca_acceptance(scores, rot, cores, cfg, str(tmp_path))
    fg.fig_cooman_ddsimca(scores, rot, cores, cfg, str(tmp_path), max_pares=6)
    X = rng.normal(size=(rot.size, 60)) + np.arange(60)
    fg.fig_hca_comparacao_pipelines(X, rot, cores, cfg, str(tmp_path))
    fg.fig_loadings_pca(PCA(5).fit(X), np.arange(60.0), cfg, str(tmp_path), n_pcs=5)
    nomes = sorted(p.name for p in (tmp_path / "Graficos").rglob("*.png"))
    assert [n for n in nomes if n.startswith("fig_ddsimca_aceitacao")] == [
        f"fig_ddsimca_aceitacao_{i:02d}.png" for i in range(1, 8)]          # 7 classes, 7 arquivos
    assert sum(n.startswith("fig_hca_comparacao_pipelines_") for n in nomes) == 3   # comparacao: ate' 3 juntos
    assert sum(n.startswith("fig_cooman_ddsimca_") for n in nomes) == 6
    assert sum(n.startswith("fig_loadings_pca_") for n in nomes) == 5
    assert fg.obter_figuras_excedentes() == []


def test_williams_coeficientes_e_eem_um_grafico_por_arquivo(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from guaraci import figuras as fg
    from guaraci import figuras_extras as fx
    fg.resetar_falhas_salvamento()
    cfg = Config(output_format="png")
    rng = np.random.default_rng(1)
    modelos = {f"esp{i}": {"h_cal": rng.uniform(0, .3, 20), "r_cal": rng.normal(size=20), "n_lv": 2}
               for i in range(13)}
    primeiro = fx.fig_williams(modelos, cfg, str(tmp_path))
    assert primeiro.endswith("figS2d_williams_01.png")
    coefs = {f"esp{i}": rng.normal(size=50) for i in range(4)}
    fx.fig_coeficientes_regressao(np.arange(50.0), coefs, cfg, str(tmp_path))
    ex, em = np.arange(5.0), np.arange(4.0)
    rot = np.repeat(["a", "b", "c", "d"], 3)
    fx.fig_eem_media_classe(ex, em, rng.uniform(size=(12, 20)), rot, cfg, str(tmp_path))
    nomes = sorted(p.name for p in (tmp_path / "Graficos").rglob("*.png"))
    assert sum(n.startswith("figS2d_williams_") for n in nomes) == 13       # 13 especies
    assert sum(n.startswith("figS2c_coeficientes_regressao_") for n in nomes) == 4
    assert sum(n.startswith("fig0_espectros_medios_classe_") for n in nomes) == 4
    assert fg.obter_figuras_excedentes() == []


def test_contagem_de_paineis_ignora_legenda_e_barra_de_cor():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from guaraci.figuras import _legenda_lateral, contar_paineis
    fig = plt.figure()
    gs = fig.add_gridspec(1, 4)
    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    im = axs[0].imshow(np.eye(3))
    axs[0].plot([0, 1], label="a")
    fig.colorbar(im, ax=axs[0])
    _legenda_lateral(fig.add_subplot(gs[3]), axs[0])
    assert contar_paineis(fig) == 3
    plt.close(fig)


# ---------------------------------------------------------------------------
#  resumo_modelo.txt legivel + resumo_modelo.json canonico
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_resumo_txt_no_idioma_e_canonico_no_json(execucoes, lang):
    """O .txt sai no idioma; quem extrai numero le o .json (chaves fixas), nos dois idiomas igual."""
    import json
    from guaraci.app_logic import load_summary, load_summary_text
    from guaraci.config import NOME_RELATORIOS, pasta_execucao
    from guaraci.resumo_parse import parse_dataset_counts, parse_model_metrics
    run = execucoes[f"{lang}_classificar"]
    with open(os.path.join(pasta_execucao(run, NOME_RELATORIOS), "resumo_modelo.json"), encoding="utf-8") as fh:
        canon = json.load(fh)
    assert canon["formato"] == "guaraci-resumo/1" and "Total samples" in canon["texto"]
    assert load_summary(run) == canon["texto"]
    txt = load_summary_text(run)
    assert txt and txt != canon["texto"]
    assert ("Total samples" in txt) is (lang == "en")
    assert ("LEIA PRIMEIRO" in txt) is (lang == "pt")
    assert parse_model_metrics(load_summary(run))["Optimal LVs"] not in ("-", "")
    assert parse_dataset_counts(load_summary(run))["amostras"]
    assert not i18n.vazamentos(_sem_referencias(txt), lang,
                               ignorar=_IGNORAR + ("classe_1", "classe_2", "classe_3"))


def test_valor_de_enum_dos_parametros_efetivos_sai_no_idioma_e_o_canonico_fica_em_pt():
    """`normalizacao: nenhuma` (default gravado em PT) vazava em relatorio EN; o canonico (JSON) fica em PT."""
    from guaraci import resultados_io as rio
    info = {"Total samples": 3, "parametros_efetivos": {"normalizacao": "nenhuma", "pre_processamento": "msc_sg_mc"}}
    assert "    normalizacao: none" in rio._texto_resumo(info, "en")
    assert "    normalizacao: nenhuma" in rio._texto_resumo(info, "pt")
    assert "    normalizacao: nenhuma" in rio._texto_resumo(info)
    assert rio._valor_parametro("nenhuma", "en") == "none" and rio._valor_parametro("nenhuma", "pt") == "nenhuma"


def test_resumo_antigo_so_txt_continua_lido(tmp_path):
    from guaraci.resumo_parse import ler_resumo_canonico
    arq = tmp_path / "resumo_modelo.txt"
    arq.write_text("  Total samples : 12\n", encoding="utf-8")
    assert ler_resumo_canonico(str(arq)) == "  Total samples : 12\n"
    assert ler_resumo_canonico(str(tmp_path / "nada.txt")) is None


def test_rotulos_e_valores_do_resumo_nos_dois_idiomas():
    assert i18n.rotulo_linha_resumo("  T2-outliers ACA", "en") == "  T2 outliers ACA"
    assert i18n.rotulo_linha_resumo("DD-SIMCA ACA sens(LOGO)/esp", "pt").endswith("sens. (LOGO) / esp.")
    assert i18n.rotulo_linha_resumo("--- Accuracy por classe ---", "pt") == "--- Acuracia por classe ---"
    assert i18n.rotulo_resumo("BCa Holdout Accuracy", "pt") == "IC BCa: acuracia"
    v = i18n.valor_resumo("86.7% / n/a (grupos_LOGO=5, puros=15, adult=0)", "en")
    assert v == "86.7% / n/a (LOGO groups=5, pure=15, adulterated=0)"
    aviso = ("Sensibilidade estimada por LOGO com apenas 5 grupos de replica. Incerteza alta; IC bootstrap "
             "nao e confiavel neste regime. Interpretar como exploratoria.")
    assert i18n.valor_resumo(aviso, "en").startswith("Sensitivity estimated by LOGO with only 5")
    assert i18n.valor_resumo(aviso, "pt") == aviso
    assert i18n.valor_resumo(False, "pt") == "nao" and i18n.valor_resumo(0.5, "en") == "0.5000"
    assert i18n.linha_resumo("  Pooled metrics:", "pt") == "  Metricas do modelo unico (todas as especies):"
    assert i18n.linha_resumo("  Limiar de aceite: R2cv >= 0.70", "en") == "  Acceptance threshold: R2cv >= 0.70"


# ---------------------------------------------------------------------------
#  Texto dentro das figuras
# ---------------------------------------------------------------------------
def test_glossario_das_figuras_nos_dois_sentidos():
    from guaraci.idioma_figuras import traduzir_texto as tr
    assert tr("(a) Hotelling T$^2$ (8 LVs) — 95% limit", "pt") == "(a) T$^2$ de Hotelling (8 VLs) — limite de 95%"
    assert tr("Model: ACA  sens.=87% (LOGO n=5) | spec.=n/a", "pt") == "Modelo: ACA  sens.=87% (LOGO n=5) | esp.=n/a"
    assert tr("Validation (n=18)", "pt") == "Validação (n=18)"
    assert tr("(b) Top 20 by SR vs VIP", "en") == "(b) Top 20 by SR vs VIP"   # ingles fica como esta'
    assert tr("Tendencia por lote (vermelho = alarme)", "en") == "Trend per batch (red = alarm)"
    assert tr("Tendencia por lote (vermelho = alarme)", "pt") == "Tendência por lote (vermelho = alarme)"
    assert tr("especie_A (n=18)", "pt") == "especie_A (n=18)"   # nome de classe nao e' tocado
    assert tr("Accuracy", "pt") == "Acurácia" and tr("Balanced accuracy", "pt") == "Acurácia balanceada"


@pytest.mark.parametrize("lang,esperado", [("pt", ["Acurácia", "Valor"]), ("en", ["Accuracy", "Value"])])
def test_save_traduz_titulo_eixos_legenda_e_rotulos_de_tick(tmp_path, lang, esperado, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from guaraci import figuras as fg
    monkeypatch.setenv("GUARACI_IDIOMA", lang)
    fig, ax = plt.subplots()
    ax.bar([0, 1], [0.9, 0.8], label="Validation")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Accuracy", "Value"])
    ax.set_title("(b) Cross-validation metrics"); ax.set_ylabel("Predicted value"); ax.legend()
    fg.save(fig, "fig_teste_idioma", str(tmp_path), Config(output_format="png"))
    fig.canvas.draw()
    assert [t.get_text() for t in ax.get_xticklabels()] == esperado
    assert ax.get_title() == ("(b) Métricas da validação cruzada" if lang == "pt" else "(b) Cross-validation metrics")
    assert ax.get_legend().get_texts()[0].get_text() == ("Validação" if lang == "pt" else "Validation")
    plt.close(fig)


@pytest.mark.slow
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_texto_das_figuras_num_idioma_so(execucoes, lang):
    """Toda figura das execucoes (quantificacao + classificacao com DD-SIMCA, OPLS, Wold e detalhadas)."""
    textos = _TEXTOS_FIGURA[lang]
    assert len(textos) > 100
    classes = ("especie_A", "especie_B", "especie_C", "classe_1", "classe_2", "classe_3")
    vazou = sorted({t for t in textos
                    if i18n.vazamentos(re.sub(r"\$[^$]*\$", " ", t), lang, ignorar=_IGNORAR + classes)})
    assert not vazou, vazou[:20]


def test_contribuicao_de_score_com_menos_variaveis_que_o_top_n(tmp_path):
    """ICP com 10 elementos e imagem com 18 descritores derrubavam a execucao ("shape mismatch")."""
    from sklearn.cross_decomposition import PLSRegression
    from guaraci import figuras as fg
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 10))
    rot = np.array(["a"] * 10 + ["b"] * 10)
    y = np.column_stack([rot == "a", rot == "b"]).astype(float)
    pls = PLSRegression(n_components=2).fit(X, y)
    fg.fig_sprint3_score_contribution(pls, X, rot, np.arange(10.0), {"a": "C0", "b": "C1"}, 20,
                                      Config(output_format="png"), str(tmp_path))
    nomes = [f.name for f in tmp_path.rglob("*.png")]
    assert any(n.startswith("fig_score_contribution_top") for n in nomes), nomes


@pytest.mark.parametrize("n_paineis,esperado", [(2, ["fig_qualquer.png"]),
                                                (7, [f"fig_qualquer_{i:02d}.png" for i in range(1, 8)])])
def test_regra_central_nenhuma_imagem_com_mais_de_3_graficos(tmp_path, n_paineis, esperado):
    """Qualquer figura -- mesmo uma que nao se divide sozinha -- sai em imagens de ate' 3 graficos."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from guaraci import figuras as fg
    fg.resetar_falhas_salvamento()
    contagens = []
    original = plt.Figure.savefig

    def _espiao(self, *a, **k):
        contagens.append(fg.contar_paineis(self))
        return original(self, *a, **k)

    fig, axs = plt.subplots(1, n_paineis, figsize=(3 * n_paineis, 3), constrained_layout=True, squeeze=False)
    for i, ax in enumerate(axs[0]):
        ax.plot([0, 1], [0, i]); ax.set_title(f"painel {i}")
    plt.Figure.savefig = _espiao
    try:
        fg.save(fig, "fig_qualquer", str(tmp_path), Config(output_format="png"))
    finally:
        plt.Figure.savefig = original
    assert sorted(p.name for p in tmp_path.rglob("*.png")) == esperado
    assert contagens and max(contagens) <= (fg.GRAFICOS_POR_ITEM if n_paineis > 3 else fg.MAX_PAINEIS_POR_FIGURA)
    larguras = [imread(str(p)).shape[1] for p in sorted(tmp_path.rglob("*.png"))]
    if n_paineis > 3:   # cada pagina tem a largura de ate' 3 paineis, nao a da figura de 7
        dpi = matplotlib.rcParams["savefig.dpi"]
        dpi = fig.dpi if dpi == "figure" else float(dpi)   # outro teste pode ter mudado o dpi de gravacao
        assert max(larguras) / dpi <= 3 * 1.5   # um painel de 3 pol. por imagem, nao a figura de 7
    assert (fg.obter_figuras_excedentes() == []) is (n_paineis <= 3)


# ---------------------------------------------------------------------------
# Pastas no idioma da sessao (pedido de 2026-11-07): grava no idioma, le os dois nomes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang,esperado", [("pt", "Relatorios"), ("en", "Reports")])
def test_pasta_nova_sai_no_idioma_da_sessao(tmp_path, lang, esperado):
    from guaraci.config import NOME_RELATORIOS, pasta_execucao
    assert os.path.basename(pasta_execucao(str(tmp_path), NOME_RELATORIOS, lang)) == esperado


@pytest.mark.parametrize("existente,lang", [("Relatorios", "en"), ("Reports", "pt")])
def test_pasta_existente_vence_o_idioma(tmp_path, existente, lang):
    """Rodada feita no outro idioma (ou antiga) continua abrindo e recebe os arquivos novos."""
    from guaraci.config import NOME_RELATORIOS, pasta_execucao
    (tmp_path / existente).mkdir()
    assert pasta_execucao(str(tmp_path), NOME_RELATORIOS, lang) == str(tmp_path / existente)


@pytest.mark.parametrize("lang,grupo,serie", [("pt", "10_quantificacao", "williams_por_especie"),
                                              ("en", "10_quantification", "williams_per_species")])
def test_serie_vai_para_subpasta_do_grupo_no_idioma(tmp_path, monkeypatch, lang, grupo, serie):
    import matplotlib
    matplotlib.use("Agg")
    from guaraci import figuras_extras as fx
    from guaraci import grupos_figuras
    monkeypatch.setenv("GUARACI_IDIOMA", lang)
    monkeypatch.setattr(grupos_figuras, "ORGANIZAR", True)
    rng = np.random.default_rng(0)
    modelos = {f"esp{i}": {"h_cal": rng.uniform(0, .3, 15), "r_cal": rng.normal(size=15), "n_lv": 2}
               for i in range(3)}
    primeiro = fx.fig_williams(modelos, Config(output_format="png"), str(tmp_path))
    graf = "Figures" if lang == "en" else "Graficos"
    pasta = tmp_path / graf / grupo / serie
    assert primeiro == str(pasta / "figS2d_williams_01.png")
    assert sorted(p.name for p in pasta.iterdir()) == [f"figS2d_williams_0{i}.png" for i in (1, 2, 3)]
    fx.fig_williams({"so": modelos["esp0"]}, Config(output_format="png"), str(tmp_path),
                    nome="figS2d_williams_unica")
    assert (tmp_path / graf / grupo / "figS2d_williams_unica.png").exists()   # figura unica: sem subpasta


def test_ordem_dos_relatorios_ignora_subpasta_de_serie():
    from guaraci.grupos_figuras import chave_ordem_figura
    a = os.path.join("r", "Figures", "10_quantification", "williams_per_species", "figS2d_williams_01.png")
    b = os.path.join("r", "Graficos", "figS2d_williams_01.png")
    assert chave_ordem_figura(a).split(os.sep)[-1] == chave_ordem_figura(b).split(os.sep)[-1]
    assert "williams_per_species" not in chave_ordem_figura(a)
