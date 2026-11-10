"""Achados R2-01/03/04/05/06/07 da revisao cega de 2026-09-26, cada um reproduzido
pela entrada real do usuario (CSV -> executar / predicao)."""

import numpy as np
import pandas as pd
import pytest

import guaraci.pipeline as pq
from guaraci import predicao as pr


def _csv(tmp_path, nome, texto, encoding="utf-8"):
    p = tmp_path / nome
    p.write_bytes(texto.encode(encoding))
    return str(p)


def _base(n=8, p=12):
    rng = np.random.default_rng(0)
    wn = np.linspace(4000, 4110, p)
    linhas = []
    for c, k in (("A", 0), ("B", 1)):
        for i in range(n):
            linhas.append((c, np.exp(-((wn - 4030 - 40 * k) / 20) ** 2) + 0.01 * rng.standard_normal(p)))
    return wn, linhas


def _cfg(tmp_path, arq, **kw):
    return pq.Config(mode="csv", csv_file=arq, class_column="classe", show_plots=False,
                     output_root_folder=str(tmp_path / "o"), n_permutations=3,
                     wn_min=4000, wn_max=4110, **kw)


def _carrega(cfg):
    from guaraci.dados_io import load_data
    return load_data(cfg)


def test_bom_do_excel_nao_quebra_a_coluna_de_classe(tmp_path):
    wn, linhas = _base()
    cab = "classe," + ",".join(f"{w:.2f}" for w in wn)
    corpo = "\n".join(c + "," + ",".join(f"{v:.4f}" for v in y) for c, y in linhas)
    arq = _csv(tmp_path, "bom.csv", "\ufeff" + cab + "\r\n" + corpo + "\r\n")
    _wn, X, rot, *_ = _carrega(_cfg(tmp_path, arq))
    assert sorted(set(rot)) == ["A", "B"] and X.shape[0] == 16


@pytest.mark.parametrize("sep", [";", "\t"])
def test_separador_com_virgula_decimal_no_cabecalho_e_nos_valores(tmp_path, sep):
    wn, linhas = _base()
    cab = "classe" + sep + sep.join(f"{w:.2f}".replace(".", ",") for w in wn)
    corpo = "\n".join(c + sep + sep.join(f"{v:.4f}".replace(".", ",") for v in y) for c, y in linhas)
    arq = _csv(tmp_path, "sep.csv", cab + "\n" + corpo + "\n")
    _wn, X, rot, *_ = _carrega(_cfg(tmp_path, arq))
    assert X.shape == (16, 12) and sorted(set(rot)) == ["A", "B"]
    # predicao: o mesmo formato
    Xp, wnp, meta = pr.load_prediction_csv(arq.replace("sep.csv", "sep.csv"))
    assert Xp.shape == (16, 12) and np.isclose(wnp[0], 4000.0)


def test_classe_vazia_e_erro_acionavel_e_espaco_nas_pontas_e_removido(tmp_path):
    wn, linhas = _base()
    cab = "classe," + ",".join(f"{w:.2f}" for w in wn)
    corpo = [c + "," + ",".join(f"{v:.4f}" for v in y) for c, y in linhas]
    arq = _csv(tmp_path, "vazia.csv", cab + "\n" + "\n".join(["," + corpo[0].split(",", 1)[1]] + corpo[1:]) + "\n")
    with pytest.raises(ValueError, match=r"sem valor na coluna de classe.*linhas 2"):
        _carrega(_cfg(tmp_path, arq))
    corpo2 = [("A " if i == 0 else "") + l if i == 0 else l for i, l in enumerate(corpo)]
    corpo2[0] = "A " + corpo[0][1:]
    arq2 = _csv(tmp_path, "espaco.csv", cab + "\n" + "\n".join(corpo2) + "\n")
    _wn, _X, rot, *_ = _carrega(_cfg(tmp_path, arq2))
    assert sorted(set(rot)) == ["A", "B"]


def test_grupo_de_valor_unico_nao_e_garantia_media(tmp_path):
    wn, linhas = _base()
    cab = "classe,grupo," + ",".join(f"{w:.2f}" for w in wn)
    corpo = "\n".join(c + ",G," + ",".join(f"{v:.4f}" for v in y) for c, y in linhas)
    arq = _csv(tmp_path, "g1.csv", cab + "\n" + corpo + "\n")
    cfg = _cfg(tmp_path, arq, group_column="grupo")
    _carrega(cfg)
    assert cfg.grouping_guarantee == "none"


def test_dataset_minusculo_segue_sem_holdout_em_vez_de_indexerror(tmp_path):
    wn, linhas = _base(n=2)
    cab = "classe," + ",".join(f"{w:.2f}" for w in wn)
    corpo = "\n".join(c + "," + ",".join(f"{v:.4f}" for v in y) for c, y in linhas)
    arq = _csv(tmp_path, "mini.csv", cab + "\n" + corpo + "\n")
    cfg = _cfg(tmp_path, arq)
    cfg.frac_holdout = 0.3
    try:
        pq.executar(cfg)
    except IndexError as e:                       # o defeito antigo
        pytest.fail(f"IndexError opaco no holdout: {e}")
    except ValueError:                            # erro acionavel, nao o IndexError opaco
        pass


def test_coluna_amostra_do_usuario_nao_duplica_a_do_resultado():
    df_res = pd.DataFrame({"amostra": [1, 2], "classe_pred": ["A", "B"]})
    meta = pd.DataFrame({"amostra": ["x", "y"]})
    out = pr.anexar_metadados(meta, df_res)
    assert list(out.columns) == ["amostra_arquivo", "amostra", "classe_pred"]
    assert out.columns.is_unique


def test_predicao_recusa_eixo_de_tamanho_diferente():
    """R2-13: eixo com tamanho diferente do numero de colunas era aceito."""
    pkg = {"preprocessador": None, "pls_final": None, "label_binarizer": None}
    with pytest.raises(ValueError, match="numero de onda"):
        pr.predict_samples(pkg, np.zeros((3, 10)), np.arange(9.0))


def test_grupo_em_varias_classes_gera_aviso():
    """R2-17: grupo com espectros de classes diferentes passava calado."""
    from guaraci.auditoria_delineamento import check_grupo_em_varias_classes
    r = check_grupo_em_varias_classes(np.array(["A", "B", "B"]), np.array(["X", "X", "Y"]))
    assert r.severidade == "aviso" and "X" in r.mensagem
    ok = check_grupo_em_varias_classes(np.array(["A", "A", "B"]), np.array(["X", "X", "Y"]))
    assert ok.severidade == "ok"


def test_erros_de_valor_dizem_linha_e_coluna(tmp_path):
    """R2-11: texto no treino e NaN na predicao sem dizer onde."""
    arq = _csv(tmp_path, "t.csv", "classe,1000,1010\nA,1,2\nB,abc,3\nA,1,2\nB,2,3\n")
    from guaraci.dados_io import load_csv_com_grupos
    with pytest.raises(ValueError, match=r"linha 3, coluna '1000' = 'abc'"):
        load_csv_com_grupos(arq, "classe", None)
    arq2 = _csv(tmp_path, "p.csv", "amostra,1000,1010\na,1,2\nb,,3\n")
    with pytest.raises(ValueError, match=r"linhas 3"):
        pr.load_prediction_csv(arq2)
