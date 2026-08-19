# -*- coding: utf-8 -*-
"""Validacao externa em dataset PUBLICO, em modulo isolado.

Nao altera `dados_io.py` nem qualquer modulo do pacote: converte o dataset
publico para o formato CSV que o GUARACI ja' aceita (`cfg.modo="csv"`) e
roda o pipeline por fora. E' de proposito -- o teste do requisito 3.4 do
PROMPT_AUDITORIA_MESTRE e' justamente "adicionar uma matriz nova exige
alterar codigo-fonte?".

Dataset: Eigenvector Research "Corn" -- 80 amostras de milho em grao,
3 espectrometros (m5/mp5/mp6), 700 canais, 1100-2498 nm, com 4
propriedades de referencia (moisture, oil, protein, starch).
Fonte: https://eigenvector.com/data/Corn/  (corn.mat, 1.445.616 bytes,
sha256 e28fd4be274a54ca...). Distribuido publicamente pela Eigenvector
Research para benchmarking; ver a pagina da fonte para os termos.

Benchmark de referencia (o que a literatura obtem neste dataset):
protein e' a propriedade mais previsivel; RMSEP tipico de PLS no
espectrometro m5 fica na casa de 0,1-0,2 %m/m com ~10-15 LVs.

Uso:
    python docs/auditoria/datasets_publicos.py <caminho/corn.mat>
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def corn_para_csv(caminho_mat, instrumento="m5spec", propriedade=2):
    """Converte corn.mat no CSV que `dados_io.carregar_csv` le: uma coluna
    por canal (cabecalho = eixo espectral), + 'classe' + 'conc'.

    propriedade: 0=moisture 1=oil 2=protein 3=starch
    """
    import scipy.io as sio
    m = sio.loadmat(caminho_mat)
    X = np.asarray(m[instrumento]["data"][0, 0], dtype=float)
    # axisscale/label sao celulas (modo, indice) no formato DataSet Object do
    # PLS_Toolbox: o eixo das COLUNAS (canais) fica em [1, 0], nao em [0, 0]
    # (que e' o eixo das linhas/amostras e vem vazio aqui).
    eixo = np.asarray(m[instrumento]["axisscale"][0, 0][1, 0],
                      dtype=float).ravel()
    Y = np.asarray(m["propvals"]["data"][0, 0], dtype=float)
    nomes = [str(x).strip()
             for x in np.asarray(m["propvals"]["label"][0, 0][1, 0]).ravel()]
    df = pd.DataFrame(X, columns=[("%.1f" % v) for v in eixo])
    df.insert(0, "classe", "corn")
    df.insert(1, "conc", Y[:, propriedade])
    return df, eixo, nomes[propriedade] if propriedade < len(nomes) else "?"


def main(caminho_mat):
    from guaraci.config import Config
    from guaraci.pipeline import executar

    df, eixo, nome_prop = corn_para_csv(caminho_mat)
    print("dataset: %d amostras x %d canais | eixo %.0f..%.0f | alvo=%s"
          % (df.shape[0], df.shape[1] - 2, eixo.min(), eixo.max(), nome_prop))

    tmp = Path(tempfile.mkdtemp(prefix="guaraci_corn_"))
    csv = tmp / "corn.csv"
    df.to_csv(csv, index=False)

    cfg = Config()
    cfg.modo = "csv"
    cfg.arquivo_csv = str(csv)
    cfg.coluna_classe = "classe"
    cfg.coluna_conc = "conc"
    # A faixa espectral E' externalizavel (config), diferente das tabelas de
    # especie/banda -- este e' o unico ajuste necessario para trocar a matriz.
    cfg.wn_min, cfg.wn_max = float(eixo.min()), float(eixo.max())
    cfg.objetivo = "quantificacao"
    cfg.nivel = "N3"
    cfg.pasta_saida_raiz = str(tmp / "saida")
    cfg.agrupar_por_mae_id = False
    cfg.mostrar_graficos = False
    cfg.n_permutacoes = 20
    cfg.frac_holdout = 0.25

    print("\n--- executar() no dataset publico, SEM alterar codigo-fonte ---")
    res = executar(cfg)
    print("\n--- resultado ---")
    if isinstance(res, dict):
        for k in ("r2_val", "rmsep", "rmsecv", "bias", "n_opt", "r2cv"):
            if k in res:
                print("   %-8s = %s" % (k, res[k]))
    print("saida em: %s" % (tmp / "saida"))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
