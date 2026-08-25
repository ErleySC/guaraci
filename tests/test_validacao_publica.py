# -*- coding: utf-8 -*-
"""Validacao contra dataset PUBLICO, como teste permanente.

O GUARACI e' validado exclusivamente em dados publicos: o dataset do TCC e'
material cedido ao autor, fica local e nao produz nenhuma metrica em
artefato deste repositorio (decisao de 2026-08-18). Sem um teste como este,
"o motor esta' correto" seria uma alegacao sem lastro.

Dataset: Eigenvector Research "Corn" -- 80 amostras de milho em grao, 3
espectrometros, 700 canais, 1100-2498 nm, com moisture/oil/protein/starch
de referencia. Fonte: https://eigenvector.com/data/Corn/

O arquivo NAO e' versionado (licenca de terceiro; ver
docs/VALIDACAO_PUBLICA.md). O teste procura por ele em
`$GUARACI_DATASETS_DIR/corn.mat` e, na ausencia, PULA com instrucao de como
obter -- nunca falha por falta de dado nem baixa nada por conta propria
dentro da suite.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

#: Faixa de RMSEP que um PLS bem calibrado atinge para PROTEINA no
#: espectrometro m5 deste dataset. Nao e' um valor gravado: e' o intervalo
#: publicado na literatura de benchmarking do Corn. Sair dele para BAIXO
#: sugere vazamento; para CIMA, bug de pre-processamento.
#:
#: APERTADO em 2026-08-20 de (0.05, 0.35) para a faixa que o texto
#: publicado promete. O gate frouxo deixaria o CI verde com RMSEP 0,34 --
#: o dobro do teto anunciado no README e no paper. Gate e texto sao
#: amarrados por test_gate_bate_com_a_faixa_publicada, abaixo.
RMSEP_PROTEINA_M5 = (0.10, 0.20)


def _caminho_corn():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    caminho = Path(raiz) / "corn.mat"
    return caminho if caminho.is_file() else None


requer_corn = pytest.mark.skipif(
    _caminho_corn() is None,
    reason=("dataset publico Corn ausente. Baixe corn.mat de "
            "https://eigenvector.com/data/Corn/ e aponte "
            "GUARACI_DATASETS_DIR para a pasta que o contem."))


@requer_corn
@pytest.mark.slow
def test_guaraci_reproduz_a_literatura_no_corn(pq, tmp_path):
    """Regressao de proteina no Corn (m5) tem que cair na faixa publicada.

    Este e' o teste que separa BUG de LIMITACAO: se o motor reproduz o
    dataset publico e falha no dataset proprio, o problema esta' nos dados,
    nao no algoritmo.
    """
    import pandas as pd
    import scipy.io as sio
    from conftest import achar_pastas_run

    m = sio.loadmat(str(_caminho_corn()))
    X = np.asarray(m["m5spec"]["data"][0, 0], dtype=float)
    eixo = np.asarray(m["m5spec"]["axisscale"][0, 0][1, 0],
                      dtype=float).ravel()
    Y = np.asarray(m["propvals"]["data"][0, 0], dtype=float)
    proteina = Y[:, 2]

    assert X.shape == (80, 700), f"corn.mat inesperado: {X.shape}"
    assert eixo.min() == pytest.approx(1100.0)
    assert eixo.max() == pytest.approx(2498.0)

    csv = tmp_path / "corn.csv"
    df = pd.DataFrame(X, columns=[f"{v:.1f}" for v in eixo])
    df.insert(0, "classe", "corn")
    df.insert(1, "conc", proteina)
    df.to_csv(csv, index=False)

    cfg = pq.Config(
        modo="csv", arquivo_csv=str(csv),
        coluna_classe="classe", coluna_conc="conc",
        perfil_matriz="milho_nir",          # perfil, nao edicao de codigo
        objetivo="quantificacao", nivel="N3",
        output_root_folder=str(tmp_path / "saida"),
        agrupar_por_mae_id=False, mostrar_graficos=False,
        n_permutacoes=10, frac_holdout=0.25,
    )
    pq.executar(cfg)

    runs = achar_pastas_run(cfg.output_root_folder)
    assert runs, "executar() nao criou saida"
    resumo = (Path(runs[0]) / pq.NOME_RELATORIOS / "resumo_modelo.txt")
    assert resumo.is_file()
    texto = resumo.read_text(encoding="utf-8", errors="replace")

    # O model card tem que declarar a matriz do PERFIL, nunca a de origem.
    card = (Path(runs[0]) / pq.NOME_RELATORIOS / "model_card.md")
    conteudo = card.read_text(encoding="utf-8", errors="replace")
    assert "milho em grao" in conteudo
    assert "oleo vegetal" not in conteudo

    import re
    achado = re.search(r"RMSEP[^\d\-]*([\d.,]+)", texto)
    assert achado, f"RMSEP nao encontrado no resumo:\n{texto[:600]}"
    rmsep = float(achado.group(1).replace(",", "."))
    lo, hi = RMSEP_PROTEINA_M5
    assert lo <= rmsep <= hi, (
        f"RMSEP={rmsep:.3f} fora da faixa publicada [{lo}, {hi}] para "
        f"proteina no Corn/m5. Abaixo do piso sugere vazamento; acima do "
        f"teto, bug de pre-processamento. Investigar ANTES de qualquer "
        f"outra coisa.")


@requer_corn
@pytest.mark.slow
def test_execucao_no_corn_e_deterministica(pq, tmp_path):
    """Duas execucoes com a mesma seed produzem o mesmo numero.

    Determinismo nao e' detalhe: sem ele, nenhuma metrica desta suite e'
    reproduzivel por terceiros, que e' a razao de o dataset ser publico.
    """
    import pandas as pd
    import scipy.io as sio
    from conftest import achar_pastas_run

    m = sio.loadmat(str(_caminho_corn()))
    X = np.asarray(m["m5spec"]["data"][0, 0], dtype=float)
    eixo = np.asarray(m["m5spec"]["axisscale"][0, 0][1, 0],
                      dtype=float).ravel()
    Y = np.asarray(m["propvals"]["data"][0, 0], dtype=float)

    def _rodar(destino):
        csv = destino / "corn.csv"
        df = pd.DataFrame(X, columns=[f"{v:.1f}" for v in eixo])
        df.insert(0, "classe", "corn")
        df.insert(1, "conc", Y[:, 2])
        df.to_csv(csv, index=False)
        cfg = pq.Config(
            modo="csv", arquivo_csv=str(csv),
            coluna_classe="classe", coluna_conc="conc",
            perfil_matriz="milho_nir", objetivo="quantificacao", nivel="N3",
            output_root_folder=str(destino / "saida"),
            agrupar_por_mae_id=False, mostrar_graficos=False,
            n_permutacoes=10, frac_holdout=0.25, seed=1234,
        )
        pq.executar(cfg)
        runs = achar_pastas_run(cfg.output_root_folder)
        return (Path(runs[0]) / pq.NOME_RELATORIOS
                / "resumo_modelo.txt").read_text(encoding="utf-8",
                                                  errors="replace")

    a = (tmp_path / "a"); a.mkdir()
    b = (tmp_path / "b"); b.mkdir()
    txt_a, txt_b = _rodar(a), _rodar(b)

    import re
    def _numeros(t):
        # ignora linhas com data/hora e caminhos, que mudam entre execucoes
        util = [ln for ln in t.splitlines()
                if not re.search(r"\d{4}-\d{2}-\d{2}|[A-Za-z]:\\|/tmp", ln)]
        return re.findall(r"[-+]?\d+\.\d+", "\n".join(util))

    assert _numeros(txt_a) == _numeros(txt_b), (
        "duas execucoes com a mesma seed produziram numeros diferentes")


def test_gate_bate_com_a_faixa_publicada():
    """Gate do CI e texto publicado nao podem ser duas fontes de verdade.

    O gate aceitava (0,05; 0,35) enquanto README, paper e VALIDACAO_PUBLICA
    prometiam 0,1-0,2: o build ficaria verde com o DOBRO do teto anunciado, e
    ninguem perceberia, porque nada ligava os dois. Este teste le' a faixa do
    texto publicado e falha se o gate divergir -- mexer num obriga a mexer no
    outro.
    """
    import re
    from pathlib import Path

    raiz = Path(__file__).resolve().parents[1]
    fontes = {
        "README.md": r"RMSEP\s+0\.144\s*%w/w\*\*\s*\|\s*([\d.]+)\s*[–-]\s*([\d.]+)",
        "docs/VALIDACAO_PUBLICA.md": r"RMSEP t[íi]pico de PLS:\s*\*\*([\d,]+)\s*[–-]\s*([\d,]+)\*\*",
        "paper/paper.md": r"within the\s+([\d.]+)\s*[–-]\s*([\d.]+)\s+range",
    }
    faixas = {}
    for arquivo, padrao in fontes.items():
        texto = (raiz / arquivo).read_text(encoding="utf-8")
        m = re.search(padrao, texto)
        assert m, f"faixa publicada nao encontrada em {arquivo}"
        faixas[arquivo] = (float(m.group(1).replace(",", ".")),
                           float(m.group(2).replace(",", ".")))

    distintas = set(faixas.values())
    assert len(distintas) == 1, (
        f"o texto publicado anuncia faixas diferentes entre si: {faixas}")
    publicada = distintas.pop()
    assert RMSEP_PROTEINA_M5 == publicada, (
        f"gate do CI {RMSEP_PROTEINA_M5} != faixa publicada {publicada}. "
        "Um dos dois esta' mentindo; alinhe os dois no mesmo commit.")
