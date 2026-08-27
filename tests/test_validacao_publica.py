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
        mode="csv", csv_file=str(csv),
        class_column="classe", conc_column="conc",
        matrix_profile="milho_nir",          # perfil, nao edicao de codigo
        objective="quantificacao", level="N3",
        output_root_folder=str(tmp_path / "saida"),
        group_by_mae_id=False, show_plots=False,
        n_permutations=10, frac_holdout=0.25,
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
            mode="csv", csv_file=str(csv),
            class_column="classe", conc_column="conc",
            matrix_profile="milho_nir", objective="quantificacao", level="N3",
            output_root_folder=str(destino / "saida"),
            group_by_mae_id=False, show_plots=False,
            n_permutations=10, frac_holdout=0.25, seed=1234,
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


@requer_corn
@pytest.mark.slow
def test_transferencia_de_calibracao_reduz_erro_entre_instrumentos_do_corn(pq):
    """Passo 86: o Corn tem as MESMAS 80 amostras medidas em 3
    espectrometros (m5, mp5, mp6) -- o caso de uso real de transferencia de
    calibracao. Calibra PLS de proteina so' no m5, aplica o modelo direto
    no mp5 (sem transferencia -- deve degradar MUITO) e depois com PDS
    (`piecewise_direct_standardization`) treinado num pequeno subconjunto
    de amostras medidas nos dois instrumentos.

    Hiperparametros (janela=5, alpha=0.001, PLS n_components=7) medidos
    diretamente contra o corn.mat local antes de fixar o teste -- nao
    adivinhados. Com esses valores: RMSEP sem transferencia ~0.51,
    com PDS ~0.16 (quase o mesmo nivel do RMSEP so'-no-m5 ~0.148 do teste
    `test_guaraci_reproduz_a_literatura_no_corn` acima).
    """
    import scipy.io as sio
    from sklearn.cross_decomposition import PLSRegression

    from guaraci.transferencia_calibracao import (
        apply_standardization, piecewise_direct_standardization)

    m = sio.loadmat(str(_caminho_corn()))
    X_m5 = np.asarray(m["m5spec"]["data"][0, 0], dtype=float)
    X_mp5 = np.asarray(m["mp5spec"]["data"][0, 0], dtype=float)
    Y = np.asarray(m["propvals"]["data"][0, 0], dtype=float)
    proteina = Y[:, 2]

    rng = np.random.default_rng(0)
    idx = rng.permutation(80)
    # 3 grupos DISJUNTOS: transferencia (aprende F), calibracao (treina o
    # PLS no m5), teste (avaliado so' no mp5, nunca visto por nenhum dos
    # dois ajustes) -- sem isso o numero nao provaria generalizacao.
    idx_transferencia, idx_calibracao, idx_teste = idx[:15], idx[15:55], idx[55:]

    pls = PLSRegression(n_components=7, scale=False)
    pls.fit(X_m5[idx_calibracao], proteina[idx_calibracao])
    y_teste = proteina[idx_teste]

    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    # Sem transferencia: o modelo calibrado no m5 recebe espectro do mp5.
    pred_sem = pls.predict(X_mp5[idx_teste]).ravel()
    rmsep_sem = _rmse(pred_sem, y_teste)
    assert rmsep_sem > 0.35, (
        f"sem transferencia o RMSEP deveria ser claramente ruim (achado "
        f"{rmsep_sem:.3f}) -- se ja' e' bom sem correcao, o Corn nao "
        f"exercita o problema que a transferencia resolve, e o teste "
        f"abaixo nao prova nada")

    transform = piecewise_direct_standardization(
        X_m5[idx_transferencia], X_mp5[idx_transferencia], janela=5, alpha=0.001)
    X_mp5_padronizado = apply_standardization(X_mp5[idx_teste], transform)
    pred_com = pls.predict(X_mp5_padronizado).ravel()
    rmsep_com = _rmse(pred_com, y_teste)

    assert rmsep_com < rmsep_sem * 0.5, (
        f"PDS deveria reduzir o RMSEP entre instrumentos por pelo menos "
        f"metade no Corn real (sem={rmsep_sem:.3f}, com PDS={rmsep_com:.3f})")
    assert rmsep_com < 0.25, (
        f"com PDS o RMSEP deveria chegar perto do nivel so'-no-m5 (~0.148, "
        f"ver test_guaraci_reproduz_a_literatura_no_corn) -- achado "
        f"{rmsep_com:.3f}")
