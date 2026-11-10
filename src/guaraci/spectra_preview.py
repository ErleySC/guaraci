"""spectra_preview.py — Carregamento/plotagem de amostra de espectros para
prévia na UI web (abas Data e Preprocessing).

Extraído de app_quimiometria.py (item 18 da auditoria): usado por duas abas,
então vira módulo de serviço próprio em vez de duplicado ou passado por
parâmetro entre elas. `preview_spectra_dx`/`preview_spectra_csv` mantêm
o cache do Streamlit (`st.cache_data`) — não são puras no sentido de
app_logic.py, mas não fazem I/O de widget algum, só leitura/parsing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import guaraci.pipeline as pq


@st.cache_data(show_spinner=False, ttl=120)
def preview_spectra_dx(pasta: str, wn_min: float, wn_max: float,
                          max_por_classe: int = 5):
    """Loads up to max_por_classe samples per subfolder for visualization."""
    try:
        subpastas = sorted(
            p for p in Path(pasta).iterdir()
            if p.is_dir() and list(p.glob("*.dx"))
        )
        if not subpastas:
            # flat folder with .dx files
            subpastas = [Path(pasta)]
        wn_ref, specs, labs = None, [], []
        for sp in subpastas:
            arqs = sorted(sp.glob("*.dx"))[:max_por_classe]
            for arq in arqs:
                try:
                    wn_a, sp_a = pq.parse_dx(str(arq))
                    mask = (wn_a >= wn_min) & (wn_a <= wn_max)
                    wn_a, sp_a = wn_a[mask], sp_a[mask]
                    if wn_ref is None:
                        wn_ref = wn_a
                    else:
                        # np.interp replaces deprecated scipy.interpolate.interp1d
                        # Requires INCREASING xp and does not sort on its own:
                        # a .dx written in decreasing order (common FTIR
                        # convention) would silently render a wrong preview.
                        _ord = np.argsort(wn_a)
                        sp_a = np.interp(wn_ref, wn_a[_ord], sp_a[_ord])
                    specs.append(sp_a)
                    labs.append(sp.name)
                except Exception:  # noqa: BLE001 -- 1 arquivo de ate
                    # max_por_classe numa PREVIA visual (nao a analise real);
                    # excluido do preview, os demais continuam.
                    continue
        if not specs or wn_ref is None:
            return None, None, None
        return wn_ref, np.array(specs), np.array(labs)
    except Exception:  # noqa: BLE001 -- previa da UI, best-effort por
        # design: (None, None, None) e' o contrato documentado de "sem
        # previa disponivel"; nunca afeta o pipeline real.
        return None, None, None


@st.cache_data(show_spinner=False, ttl=120)
def preview_spectra_csv(caminho: str, col_cls: str,
                          wn_min: float, wn_max: float,
                          max_n: int = 50):
    """Loads up to max_n CSV rows for visualization."""
    try:
        from guaraci.csv_robusto import ler_csv
        df = ler_csv(caminho, nrows=max_n)
        # R5-02: so' colunas de NOME numerico sao espectro (coluna-alvo numerica,
        # ex. PeroxideValue, ou de grupo nao entram); virgula decimal no cabecalho ok.
        def _num(c):
            try:
                float(str(c).strip().replace(",", "."))
                return True
            except ValueError:
                return False
        num_cols = [c for c in df.columns if c != col_cls and _num(c)]
        if not num_cols:
            return None, None, None
        wn = np.array([float(str(c).strip().replace(",", ".")) for c in num_cols])
        mask = (wn >= wn_min) & (wn <= wn_max)
        X = df[num_cols].values[:, mask].astype(float)
        labs = (df[col_cls].astype(str).values if col_cls in df.columns
                else np.array(["?"] * len(df)))
        return wn[mask], X, labs
    except Exception:  # noqa: BLE001 -- mesmo contrato de preview_spectra_dx
        # acima: (None, None, None) = "sem previa", nunca afeta o pipeline real.
        return None, None, None


def plot_mean_spectra(wn: np.ndarray, X: np.ndarray,
                          rotulos: np.ndarray, titulo: str = ""):
    """Plots mean ± std per class.

    As cores vem de `paleta_cores.map_class_colors` -- a MESMA funcao que
    colore as figuras do pipeline. Ate' 2026-09-08 esta previa usava
    `tab10` fixo, entao ela nao refletia a paleta escolhida pelo usuario e a
    "pre-visualizacao de cor" da tela Visualizacao teria sido uma simulacao,
    nao o resultado de verdade.
    """
    from guaraci.eixo_espectral import rotulo_do_eixo
    from guaraci.paleta_cores import map_class_colors

    classes = np.unique(np.asarray(rotulos))
    mapa_cores = map_class_colors(classes)
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    for i, cls in enumerate(classes[:10]):
        cor = mapa_cores[str(cls)]
        mask = rotulos == cls
        med = X[mask].mean(axis=0)
        std = X[mask].std(axis=0)
        from guaraci.paleta_cores import estilo_linha
        ax.plot(wn, med, color=cor, lw=1.3, ls=estilo_linha(i), label=f"{cls} (n={mask.sum()})")
        ax.fill_between(wn, med - std, med + std, color=cor, alpha=0.15)
    if len(wn) > 1 and wn[0] > wn[-1]:
        ax.invert_xaxis()
    # A previa nao conhece perfil nem tecnica: eixo neutro (achado V75).
    ax.set_xlabel(rotulo_do_eixo(None))
    ax.set_ylabel("Intensity")
    if titulo:
        ax.set_title(titulo, fontsize=9)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(color="0.93", lw=0.5)
    return fig


def remover_faixas_excluidas(wn, X, faixas):
    """Tira da previa as variaveis dentro de `faixas_excluir` (as mesmas que o pipeline exclui). Devolve
    (wn, X, n_removidas). Nunca levanta: faixa ilegivel, fora do eixo ou que apagaria quase tudo deixa a previa
    como esta (o erro de verdade aparece na execucao, com mensagem propria)."""
    if wn is None or X is None or not faixas:
        return wn, X, 0
    try:
        from guaraci.preprocessamento import mascara_fora_das_faixas
        eixo = np.asarray(wn, dtype=float)
        fica = mascara_fora_das_faixas(eixo, faixas)
    except (ValueError, TypeError):
        return wn, X, 0
    n_rem = int((~fica).sum())
    if n_rem == 0 or int(fica.sum()) < 2:
        return wn, X, 0
    return eixo[fica], np.asarray(X)[:, fica], n_rem


__all__ = ["preview_spectra_dx", "preview_spectra_csv", "plot_mean_spectra", "remover_faixas_excluidas"]
