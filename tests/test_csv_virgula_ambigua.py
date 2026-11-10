"""V89: CSV com virgula decimal E virgula separadora (o "CSV separado por virgulas" do Excel em pt-BR).

Cada numero "0,00123" vira dois campos, as linhas ficam com o dobro de campos do cabecalho e o pandas
joga o excedente para o indice. Antes, o arquivo era lido como lixo e a execucao morria bem depois com
uma explicacao errada sobre classes; agora a leitura recusa na hora e diz o que fazer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guaraci.csv_robusto import ler_csv


def _gerar(tmp_path, sep, dec, nome="d.csv", n_lin=12, n_col=8):
    rng = np.random.default_rng(3)
    cab = ["classe", "lote"] + [f"{4000 + 30 * j:.1f}" for j in range(n_col)]
    linhas = [sep.join(cab)]
    for i in range(n_lin):
        valores = [f"{v:.5f}".replace(".", dec) for v in rng.uniform(0.1, 0.9, n_col)]
        linhas.append(sep.join([("A" if i % 2 else "B"), str(i // 2)] + valores))
    caminho = tmp_path / nome
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return str(caminho)


def test_virgula_decimal_com_virgula_separadora_e_recusada_com_explicacao(tmp_path):
    with pytest.raises(ValueError) as exc:
        ler_csv(_gerar(tmp_path, ",", ","))
    msg = str(exc.value)
    assert "decimal" in msg and "separador" in msg
    assert ";" in msg                      # sugere salvar com ponto e virgula
    assert "campos" in msg                 # diz a contagem que nao bate


def test_a_mesma_recusa_vale_pelo_caminho_do_modo_csv(tmp_path):
    from guaraci.config import Config
    from guaraci.dados_io import load_data
    cfg = Config(mode="csv", csv_file=_gerar(tmp_path, ",", ","), class_column="classe",
                 show_plots=False, output_root_folder=str(tmp_path / "o"))
    with pytest.raises(ValueError, match="decimal"):
        load_data(cfg)


@pytest.mark.parametrize("sep, dec", [(",", "."), (";", ","), ("\t", ","), (";", "."), ("\t", ".")])
def test_formatos_nao_ambiguos_continuam_lendo_com_indice_simples(tmp_path, sep, dec):
    df = ler_csv(_gerar(tmp_path, sep, dec))
    assert isinstance(df.index, pd.RangeIndex)
    assert df.shape == (12, 10)


def test_quem_pede_index_col_explicitamente_nao_e_recusado(tmp_path):
    df = ler_csv(_gerar(tmp_path, ";", "."), index_col=0)
    assert df.index.name == "classe"



def test_a_validacao_da_fonte_de_dados_tambem_explica(tmp_path):
    from guaraci.config import Config
    from guaraci.config_io import _validar_pasta_dados
    cfg = Config(mode="csv", csv_file=_gerar(tmp_path, ",", ","), class_column="classe")
    ok, msg = _validar_pasta_dados(cfg)
    assert ok is False and "decimal" in msg and "ValueError" not in msg
