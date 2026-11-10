"""mode="multielementar" de ponta a ponta (dados sinteticos gerados aqui)."""
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from guaraci.resultados_io import MARCADOR_MULTIELEMENTAR

ELEMENTOS = ["Na 23", "Mg 24", "K 39", "Ca 44", "Mn 55", "Fe 56", "Cu 63", "Zn 66", "Sr 88", "Cd 111"]


def _csv_sintetico(caminho: Path, seed: int = 0, efeito: float = 1.5) -> None:
    """4 origens x 8 amostras fisicas x 2 replicas; efeito de origem em Mn,
    Sr e Zn; Cd quase todo censurado; 6 linhas de QC em 2 sessoes; 1 branco."""
    rng = np.random.default_rng(seed)
    linhas = ["Amostra;Tipo;Origem;Lote;Sessao;" + ";".join(ELEMENTOS)]
    for o in range(4):
        for a in range(8):
            base = rng.lognormal(0, 0.3, len(ELEMENTOS))
            for j, e in enumerate(ELEMENTOS):
                if e in ("Mn 55", "Sr 88", "Zn 66"):
                    base[j] *= 1 + efeito * o / 3
            for r in range(2):
                vals = base * rng.lognormal(0, 0.05, len(ELEMENTOS))
                cel = [f"{v:.4f}".replace(".", ",") for v in vals]
                cel[ELEMENTOS.index("Cd 111")] = "<0,01" if rng.random() < 0.8 else "0,02"
                if rng.random() < 0.05:
                    cel[ELEMENTOS.index("Cu 63")] = "<0,05"
                linhas.append(f"O{o}A{a}R{r};amostra;origem{o};O{o}L{a};s{(a % 2) + 1};" + ";".join(cel))
    for q in range(6):
        vals = np.ones(len(ELEMENTOS)) * (1.0 + 0.02 * (q % 2)) * rng.lognormal(0, 0.01, len(ELEMENTOS))
        linhas.append(f"QC{q};qc;-;QC;s{(q % 2) + 1};" + ";".join(f"{v:.4f}".replace(".", ",") for v in vals))
    linhas.append("B1;branco;-;BR;s1;" + ";".join(["<0,01"] * len(ELEMENTOS)))
    caminho.write_text("\n".join(linhas), encoding="utf-8")


def _cfg(pq, csv, saida, **kw):
    cfg = pq.Config()
    cfg.mode = "multielementar"
    cfg.csv_file = str(csv)
    cfg.class_column = "Origem"
    cfg.group_column = "Lote"
    cfg.icp_unidade = "mg/kg"
    cfg.icp_coluna_amostra = "Amostra"
    cfg.icp_coluna_tipo = "Tipo"
    cfg.icp_coluna_sessao = "Sessao"
    cfg.default_preprocessing = "autoscaling"
    cfg.output_root_folder = saida
    cfg.show_plots = False
    cfg.n_permutations = 20
    cfg.max_lvs = 5
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture()
def saida_curta():
    d = tempfile.mkdtemp(prefix="gm")      # MAX_PATH (M-14) no Windows
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_leitor_entrega_unidades_e_elementos(pq, tmp_path):
    from guaraci.dados_io import load_data
    csv = tmp_path / "icp.csv"
    _csv_sintetico(csv)
    cfg = _cfg(pq, csv, str(tmp_path / "s"))
    wn, X, rot, conc, mae, meta = load_data(cfg)
    assert X.shape == (64, 9)                   # Cd 111 sai (80% censurado > 50%)
    assert conc is None and len(set(mae)) == 32  # 32 lotes (unidades), 2 replicas cada
    assert set(rot) == {f"origem{o}" for o in range(4)}
    assert "Cd 111" in meta.attrs["multielementar"]["registro_censura"]["elementos_excluidos"]
    assert cfg.grouping_guarantee == "medium"   # grupo declarado pelo usuario


@pytest.mark.parametrize("campo, valor, trecho", [
    ("group_column", None, "coluna_grupo"),
    ("class_column", "", "coluna_classe"),
    ("default_preprocessing", "msc_sg_mc", "espectral"),
    ("icp_unidade", "", "unidade obrigatoria"),
])
def test_leitor_recusa_sem_unidade_independente_ou_com_preproc_espectral(pq, tmp_path, campo, valor, trecho):
    from guaraci.dados_io import load_data
    csv = tmp_path / "icp.csv"
    _csv_sintetico(csv)
    cfg = _cfg(pq, csv, str(tmp_path / "s"), **{campo: valor})
    with pytest.raises(ValueError, match=trecho):
        load_data(cfg)


def test_validacao_do_caminho_no_menu(pq, tmp_path):
    from guaraci.config_io import _validar_pasta_dados as validate_data_path
    csv = tmp_path / "icp.csv"
    _csv_sintetico(csv)
    ok, msg = validate_data_path(_cfg(pq, csv, str(tmp_path)))
    assert ok and "ICP" in msg
    ok, msg = validate_data_path(_cfg(pq, csv, str(tmp_path), group_column=None))
    assert not ok and "coluna_grupo" in msg


@pytest.mark.slow
def test_pipeline_multielementar_de_ponta_a_ponta(pq, tmp_path, saida_curta):
    from conftest import achar_pastas_run
    csv = tmp_path / "icp.csv"
    _csv_sintetico(csv)
    cfg = _cfg(pq, csv, saida_curta)
    pq.executar(cfg)
    runs = achar_pastas_run(cfg.output_root_folder)
    assert runs
    run = Path(runs[0])
    tab = run / pq.NOME_TABELAS
    rel = run / pq.NOME_RELATORIOS
    anova = (tab / "multielementar_anova_por_elemento.csv").read_text(encoding="utf-8")
    assert "Mn 55" in anova and "Cd 111" not in anova
    assert (tab / "multielementar_comparacoes.csv").is_file()
    resumo = json.loads((rel / "multielementar.json").read_text(encoding="utf-8"))
    assert {"Mn 55", "Sr 88", "Zn 66"} <= set(resumo["anova"]["elementos_com_diferenca_entre_grupos"])
    assert resumo["sensibilidade_censura"]["muda"] in (True, False)
    assert resumo["efeito_de_sessao"]["testado"] is True
    assert resumo["unidade"] == "mg/kg" and "Cd 111" in resumo["elementos_excluidos"]
    assert list((run / pq.NOME_GRAFICOS).rglob("fig_multielementar_medias*.png"))   # um elemento por imagem
    card = (rel / "model_card.md").read_text(encoding="utf-8")
    assert MARCADOR_MULTIELEMENTAR in card and "EM VALIDACAO" in card
    portatil = list((run / pq.NOME_MODELOS).glob("*.portatil.json"))
    assert portatil, "exportacao portatil nao gerada"
    dados = json.loads(portatil[0].read_text(encoding="utf-8"))
    assert dados["multielementar"]["unidade"] == "mg/kg"
    assert dados["multielementar"]["politica_censura"] == "substituir"
    assert "Mn 55" in dados["multielementar"]["elementos_modelados"]
    # nenhum codigo de amostra vai para disco
    meta = (tab / "metadados.csv").read_text(encoding="utf-8")
    assert "O0A0R0" not in meta and "O0L0" not in meta
    for arq in run.rglob("*.csv"):
        assert "O0A0R0" not in arq.read_text(encoding="utf-8", errors="replace"), arq
    assert os.path.isfile(rel / "resumo_modelo.txt")
