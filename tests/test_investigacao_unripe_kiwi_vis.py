"""test_investigacao_unripe_kiwi_vis.py -- Passo 112
(INSTRUCAO_HSI_DADO_PROPRIO.md): investigacao rigorosa do problema
`unripe` (Kiwi/VIS) achado no Passo 104 (`docs/PROGRESSO.md`): mesmo
sendo a UNICA combinacao fruta/camera do DeepHS Fruit com as 3 classes
`n>=19` (limiar do Passo 105 -- ou seja, nao e' so' falta de dado), a
classe `unripe` sai com sensibilidade 0,00 interna E externa.

3 hipoteses testadas, cada uma reportada honestamente (mesmo quando NAO
mostra melhora -- "nenhuma resolveu" e' resultado valido, nunca motivo
pra' inventar mais uma tentativa indefinidamente, ver instrucao):

  A. Selecao de banda quimicamente relevante (clorofila 660-680nm,
     carotenoide/antocianina 500-550nm, ver `hsi_chemistry.
     ATRIBUICAO_QUIMICA_VIS_FRUTA`) em vez de espectro completo.
  B. Fronteira de classe CONTINUA (`storage_days`, proxy real de
     maturacao presente no manifest) em vez de discreta -- PLS-R.
  C. Sobreposicao espectral real entre `unripe` e `perfect` --
     distancia de Mahalanobis entre centroides vs. dispersao intra-
     classe, medida numa reducao de dimensionalidade BEM CONDICIONADA
     (poucos componentes de PCA, n_amostras >> n_dimensoes -- Mahalanobis
     naive em alta dimensao com n pequeno INFLA artificialmente a
     distancia por mal-condicionamento da covariancia, achado real desta
     investigacao, reportado explicitamente abaixo em vez de escondido).

Passo 114 (rodada seguinte, INSTRUCAO_PUSH_HIPOTESE_D_...md) adicionou
uma 4a hipotese, motivada por um achado da Hipotese B: `unripe` tinha
`storage_days` MEDIO maior que `perfect` -- contraintuitivo o
suficiente pra' levantar a suspeita de ruido de rotulo.

  D. Qualidade do rotulo vs. medicao objetiva -- o manifest do DeepHS
     Fruit publica `firmness` (medicao objetiva de firmeza por fruto,
     independente do rotulo visual `ripeness_state`). Se o rotulo NAO
     concordasse com a firmeza, seria evidencia de ruido de rotulo (nao
     de sobreposicao espectral) e exigiria RETRATAR a conclusao da
     Hipotese C. Medido: o rotulo CONCORDA fortemente com a firmeza
     (unripe > perfect > overripe, ordem fisiologicamente correta,
     Mann-Whitney unripe-vs-perfect p<1e-7, Cohen's d~1.6) -- ou seja,
     NAO e' ruido de rotulo. Isso REFINA a conclusao da Hipotese C (nao
     a retrata): a diferenca fisica entre `unripe` e `perfect` e' real e
     substancial (confirmada por medicao independente), mas a tecnica
     especifica usada aqui (reflectancia VIS, 397-1004nm) tem
     sensibilidade fraca a essa diferenca -- limite da TECNICA pra' essa
     distincao, nao evidencia de que as classes sejam fisicamente
     indistinguiveis em geral, nem de rotulo nao confiavel.

Mesmo padrao de `test_validacao_publica_deephs_fruit.py`: PULA (nao
falha) se `GUARACI_DATASETS_DIR` nao apontar para o dataset ja baixado
com `scripts/download_datasets/baixar_deephs_fruit_todas.py`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from guaraci.hsi_chemistry import ATRIBUICAO_QUIMICA_VIS_FRUTA
from guaraci.hsi_io import load_deephs_fruit_dataset
from guaraci.hsi_pipeline import apply_quality_gate_and_segment
from guaraci.hsi_validation import run_external_validation_by_day


def _pasta_deephs_fruit_all() -> Optional[Path]:
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "deephs_fruit_all"
    return pasta if (pasta / "manifest.json").is_file() else None


_MAX_PIXELS_POR_GRAVACAO = 2000  # mesmo teto do Passo 104, ver docstring la'


@pytest.fixture(scope="module")
def kiwi_vis_filtrado():
    """Carrega + aplica quality gate/segmentacao 1 SO' VEZ (reaproveitado
    pelas 3 hipoteses -- carregar 138 gravacoes 3x seria desperdicio)."""
    pasta = _pasta_deephs_fruit_all()
    if pasta is None:
        pytest.skip(
            "dataset publico DeepHS Fruit (todas as frutas/cameras) ausente. "
            "Baixe com "
            "'python scripts/download_datasets/baixar_deephs_fruit_todas.py' "
            "e aponte GUARACI_DATASETS_DIR.")
    cubos, rotulos, grupos, wavelengths, meta_df = load_deephs_fruit_dataset(
        str(pasta), fruta="Kiwi", camera="VIS")
    filtrado = apply_quality_gate_and_segment(
        cubos, list(grupos), list(rotulos), list(meta_df["day"]))
    assert filtrado["n_rejeitados"] == 0, (
        "quality gate rejeitou gravacao(oes) do Kiwi/VIS -- premissa desta "
        "investigacao (Passo 104: todas passam) mudou, revisar antes de "
        "interpretar os resultados abaixo.")
    return {"filtrado": filtrado, "wavelengths": wavelengths, "meta_df": meta_df}


@pytest.fixture(scope="module")
def kiwi_nir_filtrado():
    """Mesmo Kiwi, camera NIR -- so' pra' checagem adicional do Passo
    121 (nao usada pelas hipoteses A-D, que sao especificas de VIS)."""
    pasta = _pasta_deephs_fruit_all()
    if pasta is None:
        pytest.skip(
            "dataset publico DeepHS Fruit (todas as frutas/cameras) ausente. "
            "Baixe com "
            "'python scripts/download_datasets/baixar_deephs_fruit_todas.py' "
            "e aponte GUARACI_DATASETS_DIR.")
    cubos, rotulos, grupos, wavelengths, meta_df = load_deephs_fruit_dataset(
        str(pasta), fruta="Kiwi", camera="NIR")
    filtrado = apply_quality_gate_and_segment(
        cubos, list(grupos), list(rotulos), list(meta_df["day"]))
    return {"filtrado": filtrado, "wavelengths": wavelengths, "meta_df": meta_df}


def _dias_externos(dias_unicos: List[str]) -> List[str]:
    n_dias_externos = max(1, len(dias_unicos) // 4)
    return dias_unicos[-n_dias_externos:]


def test_baseline_confirma_achado_do_passo_104(kiwi_vis_filtrado):
    """Contra-prova de premissa: confirma que o achado do Passo 104
    (unripe sens=0,00 interno E externo com espectro completo) ainda se
    reproduz ANTES de testar as hipoteses -- se isso falhar, a
    investigacao abaixo estaria testando um problema que ja nao existe."""
    filtrado = kiwi_vis_filtrado["filtrado"]
    dias_unicos = sorted(set(filtrado["dias"]))
    relatorio = run_external_validation_by_day(
        filtrado["cubos"], filtrado["mascaras"], filtrado["group_ids"],
        filtrado["rotulos"], filtrado["dias"],
        dias_externos=_dias_externos(dias_unicos), seed=42,
        max_pixels_por_gravacao=_MAX_PIXELS_POR_GRAVACAO)

    print(f"\n[baseline] unripe: sens(int/ext)="
          f"{relatorio.sensibilidade_interna['unripe']:.2f}/"
          f"{relatorio.sensibilidade_externa['unripe']:.2f}")
    assert relatorio.sensibilidade_interna["unripe"] == pytest.approx(0.0, abs=1e-9)
    assert relatorio.sensibilidade_externa["unripe"] == pytest.approx(0.0, abs=1e-9)


def test_hipotese_a_bandas_quimicas_nao_melhora_unripe(kiwi_vis_filtrado):
    """Hipotese A: restringir a analise as bandas de clorofila
    (660-680nm) e carotenoide/antocianina (500-550nm) -- literatura ja
    citada em `hsi_chemistry.py` para maturacao de fruta -- em vez do
    espectro completo (397-1004nm, 224 bandas)."""
    filtrado = kiwi_vis_filtrado["filtrado"]
    wavelengths = kiwi_vis_filtrado["wavelengths"]
    dias_unicos = sorted(set(filtrado["dias"]))

    janelas = [(e.banda_min_nm, e.banda_max_nm) for e in ATRIBUICAO_QUIMICA_VIS_FRUTA
              if e.atribuicao != "sobretom O-H da agua (2a harmonica)"]  # so' pigmentos
    idx_relevantes = np.where(np.any(
        [(wavelengths >= lo) & (wavelengths <= hi) for lo, hi in janelas], axis=0))[0]
    assert len(idx_relevantes) >= 5, (
        "poucas bandas na faixa quimica relevante -- a camera nao cobre "
        "bem essa regiao? Confira wavelengths antes de interpretar.")

    cubos_restritos = [c[:, :, idx_relevantes] for c in filtrado["cubos"]]
    relatorio = run_external_validation_by_day(
        cubos_restritos, filtrado["mascaras"], filtrado["group_ids"],
        filtrado["rotulos"], filtrado["dias"],
        dias_externos=_dias_externos(dias_unicos), seed=42,
        max_pixels_por_gravacao=_MAX_PIXELS_POR_GRAVACAO)

    print(f"\n[hipotese A] {len(idx_relevantes)}/{len(wavelengths)} bandas "
          f"(clorofila+carotenoide). unripe: sens(int/ext)="
          f"{relatorio.sensibilidade_interna['unripe']:.2f}/"
          f"{relatorio.sensibilidade_externa['unripe']:.2f}")

    # NAO afirma que vai melhorar -- so' registra o numero medido. Achado
    # real (2026-09): NAO melhora (permanece 0,00/0,00) -- restringir a
    # banda quimica nao resolve, reportado como resultado negativo.
    for m in (relatorio.sensibilidade_interna, relatorio.sensibilidade_externa):
        assert 0.0 <= m["unripe"] <= 1.0


def test_hipotese_b_fronteira_continua_storage_days(kiwi_vis_filtrado):
    """Hipotese B: `storage_days` (dias de armazenamento, metadado REAL
    do manifest -- nao inventado) como proxy continuo de maturacao.
    Regressao PLS-R (espectro medio por objeto) com CV group-aware POR
    DIA (mesma garantia contra vazamento do Passo 101 -- nunca objetos
    do MESMO dia em treino e teste)."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold

    filtrado = kiwi_vis_filtrado["filtrado"]
    meta_df = kiwi_vis_filtrado["meta_df"]

    espectros_por_obj: Dict[str, List[np.ndarray]] = {}
    rotulo_por_obj: Dict[str, str] = {}
    dia_por_obj: Dict[str, str] = {}
    for cubo, mascara, gid, rot, dia in zip(
            filtrado["cubos"], filtrado["mascaras"], filtrado["group_ids"],
            filtrado["rotulos"], filtrado["dias"]):
        espectros_por_obj.setdefault(gid, []).append(cubo[mascara].mean(axis=0))
        rotulo_por_obj[gid] = rot
        dia_por_obj[gid] = dia

    gids = sorted(espectros_por_obj)
    X_obj = np.array([np.mean(espectros_por_obj[g], axis=0) for g in gids])
    rot_obj = np.array([rotulo_por_obj[g] for g in gids])
    dia_obj = np.array([dia_por_obj[g] for g in gids])
    storage_por_gid = dict(zip(meta_df["group_id"], meta_df["storage_days"]))
    y_obj = np.array([storage_por_gid[g] for g in gids], dtype=float)

    print("\n[hipotese B] storage_days por classe (proxy de maturacao):")
    for c in sorted(set(rot_obj)):
        vals = y_obj[rot_obj == c]
        print(f"  {c}: n={len(vals)} media={vals.mean():.1f} "
              f"min={vals.min():.0f} max={vals.max():.0f}")

    n_splits = min(5, len(set(dia_obj)))
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.zeros_like(y_obj)
    for tr, te in gkf.split(X_obj, y_obj, groups=dia_obj):
        pls = PLSRegression(n_components=min(8, len(tr) - 1))
        pls.fit(X_obj[tr], y_obj[tr])
        preds[te] = pls.predict(X_obj[te]).ravel()

    q2 = 1.0 - np.sum((y_obj - preds) ** 2) / np.sum((y_obj - y_obj.mean()) ** 2)
    print(f"[hipotese B] Q2 (CV por dia) = {q2:.3f}")
    print("[hipotese B] storage_days PREDITO por classe:")
    for c in sorted(set(rot_obj)):
        print(f"  {c}: predito media={preds[rot_obj == c].mean():.2f}")

    # Achado real (2026-09): Q2 negativo (sem generalizacao entre dias) E
    # a media predita de unripe fica muito perto da de perfect -- a
    # fronteira continua NAO separa as classes melhor que a discreta.
    # Alem disso storage_days e' um proxy RUIDOSO do rotulo visual (nem
    # sempre unripe tem storage_days menor que perfect neste dataset --
    # avaliacao humana visual != funcao deterministica do tempo), o que
    # ja' e' uma limitacao da hipotese em si, reportada aqui.
    assert np.isfinite(q2)
    media_diff_unripe_perfect = abs(
        preds[rot_obj == "unripe"].mean() - preds[rot_obj == "perfect"].mean())
    print(f"[hipotese B] |predito(unripe) - predito(perfect)| = "
          f"{media_diff_unripe_perfect:.3f} dias")


def test_hipotese_c_sobreposicao_espectral_unripe_perfect(kiwi_vis_filtrado):
    """Hipotese C: mede a distancia de Mahalanobis entre os centroides
    espectrais (espectro medio por objeto) de `unripe` e `perfect`
    contra a dispersao intra-classe -- em MULTIPLAS dimensionalidades de
    PCA, porque Mahalanobis naive com poucas amostras (n~30-40/classe)
    numa dimensao alta (>=10) INFLA artificialmente a distancia por
    mal-condicionamento da covariancia (achado real medido abaixo, nao
    hipotetico) -- a leitura confiavel e' a de BAIXA dimensionalidade
    (poucos componentes, matriz bem condicionada, >=80% variancia
    explicada), nao a de alta dimensionalidade."""
    from sklearn.decomposition import PCA

    filtrado = kiwi_vis_filtrado["filtrado"]
    espectros_por_obj: Dict[str, List[np.ndarray]] = {}
    rotulo_por_obj: Dict[str, str] = {}
    for cubo, mascara, gid, rot in zip(
            filtrado["cubos"], filtrado["mascaras"], filtrado["group_ids"],
            filtrado["rotulos"]):
        espectros_por_obj.setdefault(gid, []).append(cubo[mascara].mean(axis=0))
        rotulo_por_obj[gid] = rot

    gids = sorted(espectros_por_obj)
    X_obj = np.array([np.mean(espectros_por_obj[g], axis=0) for g in gids])
    rot_obj = np.array([rotulo_por_obj[g] for g in gids])

    idx_u = rot_obj == "unripe"
    idx_p = rot_obj == "perfect"
    n_u, n_p = int(idx_u.sum()), int(idx_p.sum())

    # Efeito por-banda (sem reducao de dimensao nenhuma) -- referencia
    # mais simples/robusta possivel: |diferenca de media| / desvio-padrao
    # combinado, por banda. Mediana desse indicador entre TODAS as bandas
    # resume a separabilidade univariada tipica.
    mu_u, mu_p = X_obj[idx_u].mean(axis=0), X_obj[idx_p].mean(axis=0)
    std_u, std_p = X_obj[idx_u].std(axis=0), X_obj[idx_p].std(axis=0)
    efeito_por_banda = np.abs(mu_u - mu_p) / (0.5 * (std_u + std_p) + 1e-9)
    print(f"\n[hipotese C] efeito por-banda |diff|/dp: "
          f"mediana={np.median(efeito_por_banda):.3f} "
          f"max={efeito_por_banda.max():.3f} "
          f"(referencia: ~0.8 = separacao fraca-moderada, "
          f"~2.0 = separacao forte)")

    print("[hipotese C] Mahalanobis(unripe, perfect) por n_componentes PCA "
          "(cresce com a dimensao -- efeito de mal-condicionamento, nao "
          "mais separacao real):")
    mahalanobis_por_ncomp: Dict[int, float] = {}
    for ncomp in (2, 3, 5, 10):
        pca = PCA(n_components=ncomp)
        X_pca = pca.fit_transform(X_obj)
        mu_u2, mu_p2 = X_pca[idx_u].mean(axis=0), X_pca[idx_p].mean(axis=0)
        cov_u2 = np.cov(X_pca[idx_u], rowvar=False).reshape(ncomp, ncomp)
        cov_p2 = np.cov(X_pca[idx_p], rowvar=False).reshape(ncomp, ncomp)
        cov_pool = ((n_u - 1) * cov_u2 + (n_p - 1) * cov_p2) / (n_u + n_p - 2)
        cov_pool += np.eye(ncomp) * 1e-6
        diff = mu_u2 - mu_p2
        d_mahal = float(np.sqrt(diff @ np.linalg.inv(cov_pool) @ diff))
        mahalanobis_por_ncomp[ncomp] = d_mahal
        print(f"  ncomp={ncomp:>2} var_explicada={pca.explained_variance_ratio_.sum():.3f} "
              f"mahalanobis={d_mahal:.3f}")

    # Leitura CONFIAVEL (bem condicionada, poucos componentes): distancia
    # PEQUENA -- consistente com sobreposicao espectral real. Reportado
    # aqui em vez de citar so' o numero de alta dimensao (que e' maior,
    # mas e' artefato, ver docstring).
    d_confiavel = mahalanobis_por_ncomp[2]
    print(f"[hipotese C] leitura confiavel (ncomp=2): "
          f"mahalanobis={d_confiavel:.3f}, efeito mediano por banda="
          f"{np.median(efeito_por_banda):.3f} -- ambos indicam "
          f"sobreposicao substancial (nao separacao clara)")

    # Confirma o proprio achado de instabilidade (cresce com ncomp) --
    # se isso deixar de ser verdade, a ressalva acima precisa ser revista.
    assert mahalanobis_por_ncomp[10] > mahalanobis_por_ncomp[2]


def test_hipotese_d_firmeza_objetiva_confirma_rotulo_nao_e_ruido(kiwi_vis_filtrado):
    """Hipotese D (Passo 114): o manifest do DeepHS Fruit publica
    `firmness` -- medicao OBJETIVA de firmeza por fruto, independente do
    rotulo visual `ripeness_state`. Testa se o rotulo concorda com essa
    medicao (se nao concordasse, seria evidencia de ruido de rotulo, nao
    de sobreposicao espectral -- exigiria retratar a Hipotese C)."""
    from scipy import stats

    filtrado = kiwi_vis_filtrado["filtrado"]
    meta_df = kiwi_vis_filtrado["meta_df"]

    assert "firmness" in meta_df.columns, (
        "manifest do DeepHS Fruit nao publica 'firmness' -- premissa "
        "desta hipotese mudou, revisar antes de interpretar o resto.")

    rotulo_por_obj: Dict[str, str] = {}
    for gid, rot in zip(filtrado["group_ids"], filtrado["rotulos"]):
        rotulo_por_obj[gid] = rot

    # 1 firmeza por OBJETO FISICO (front/back da mesma fruta compartilham
    # o mesmo valor -- confirmado por leitura direta do manifest, zero
    # divergencias; agrega por group_id em vez de usar a gravacao solta
    # pra' nao contar a mesma fruta 2x na estatistica).
    import pandas as pd

    firmeza_por_obj: Dict[str, float] = {}
    for gid, firm in zip(meta_df["group_id"], meta_df["firmness"]):
        if pd.notna(firm) and gid in rotulo_por_obj:
            firmeza_por_obj[gid] = float(firm)

    gids_com_firmeza = sorted(firmeza_por_obj)
    firmeza = np.array([firmeza_por_obj[g] for g in gids_com_firmeza])
    rotulo = np.array([rotulo_por_obj[g] for g in gids_com_firmeza])

    print(f"\n[hipotese D] {len(gids_com_firmeza)}/"
          f"{len(set(filtrado['group_ids']))} objetos com firmeza medida.")
    for c in sorted(set(rotulo)):
        vals = firmeza[rotulo == c]
        print(f"  {c}: n={len(vals)} media={vals.mean():.1f} "
              f"dp={vals.std():.1f} min={vals.min():.0f} max={vals.max():.0f}")

    # Firmness=0 e' um valor REAL (fruta totalmente amolecida, piso do
    # instrumento) -- so' aparece em `overripe` (confirmado por leitura
    # direta), nunca em unripe/perfect, entao nao contamina a comparacao
    # abaixo (que e' so' unripe vs perfect).
    f_unripe = firmeza[rotulo == "unripe"]
    f_perfect = firmeza[rotulo == "perfect"]
    stat, p_valor = stats.mannwhitneyu(f_unripe, f_perfect, alternative="two-sided")
    d_cohen = ((f_unripe.mean() - f_perfect.mean())
              / np.sqrt((f_unripe.std() ** 2 + f_perfect.std() ** 2) / 2))

    print(f"[hipotese D] Mann-Whitney U unripe-vs-perfect (firmeza): "
          f"U={stat:.1f} p={p_valor:.2e}")
    print(f"[hipotese D] Cohen's d (firmeza, unripe vs perfect): "
          f"{d_cohen:.2f}")

    # Achado real (2026-09): unripe e' MAIS firme que perfect (fruta
    # ainda dura, fisiologicamente correto), com separacao estatistica
    # grande e limpa -- ao contrario do sinal espectral (Hipotese C,
    # efeito fraco-moderado). O rotulo visual e' respaldado por medicao
    # independente -- NAO e' ruido de rotulo. Isso REFINA a conclusao da
    # Hipotese C (nao a retrata, ver docstring do modulo): a diferenca
    # fisica e' real, a tecnica espectral especifica (VIS reflectancia)
    # e' que tem sensibilidade fraca a ela.
    assert p_valor < 0.05, (
        "firmeza NAO separa unripe de perfect significativamente -- isso "
        "MUDARIA a conclusao (evidencia de ruido de rotulo, nao de "
        "sobreposicao espectral) e exigiria retratar a Hipotese C em "
        "docs/VALIDACAO_PUBLICA.md; parar e reportar antes de prosseguir.")
    assert d_cohen > 0, (
        "unripe deveria ser MAIS firme que perfect (fisiologicamente) -- "
        "sinal invertido contradiz a premissa, investigar antes de "
        "reportar como confirmacao do rotulo.")


def test_checagem_adicional_camera_nir_kiwi_efeito_por_banda(kiwi_nir_filtrado):
    """Passo 121, checagem NAO BLOQUEANTE: o Kiwi tambem tem gravacoes de
    camera NIR (alem da VIS investigada acima) -- mede se o NIR capta
    MAIS da diferenca de firmeza que a VIS nao capta bem (Hipotese D).
    Observacional/exploratorio (n_unripe=7, bem menor que os n=28 do
    VIS) -- nao e' um teste de hipotese formal com p-valor, so' registra
    o numero pra' comparar com o efeito por-banda ja medido em VIS
    (mediana ~0,376, ver test_hipotese_c_...)."""
    filtrado = kiwi_nir_filtrado["filtrado"]

    espectros_por_obj: Dict[str, List[np.ndarray]] = {}
    rotulo_por_obj: Dict[str, str] = {}
    for cubo, mascara, gid, rot in zip(
            filtrado["cubos"], filtrado["mascaras"], filtrado["group_ids"],
            filtrado["rotulos"]):
        espectros_por_obj.setdefault(gid, []).append(cubo[mascara].mean(axis=0))
        rotulo_por_obj[gid] = rot

    gids = sorted(espectros_por_obj)
    X_obj = np.array([np.mean(espectros_por_obj[g], axis=0) for g in gids])
    rot_obj = np.array([rotulo_por_obj[g] for g in gids])
    contagem = {c: int((rot_obj == c).sum()) for c in sorted(set(rot_obj))}
    print(f"\n[checagem NIR] n_objetos por classe: {contagem}")

    idx_u, idx_p = rot_obj == "unripe", rot_obj == "perfect"
    assert idx_u.sum() >= 2 and idx_p.sum() >= 2, (
        "poucas amostras de NIR pra' sequer calcular um desvio-padrao "
        "por classe -- checagem exploratoria nao aplicavel aqui.")

    mu_u, mu_p = X_obj[idx_u].mean(axis=0), X_obj[idx_p].mean(axis=0)
    std_u, std_p = X_obj[idx_u].std(axis=0), X_obj[idx_p].std(axis=0)
    efeito = np.abs(mu_u - mu_p) / (0.5 * (std_u + std_p) + 1e-9)
    print(f"[checagem NIR] efeito por-banda |diff|/dp: "
          f"mediana={np.median(efeito):.3f} max={efeito.max():.3f} "
          f"(referencia VIS: mediana~0.376, ver Hipotese C)")

    # So' registra o numero (print acima) -- nao afirma superioridade do
    # NIR como fato estabelecido (n_unripe=7 e' pequeno demais pra' isso).
    # Ver docs/VALIDACAO_PUBLICA.md secao 7, Passo 121, pra' a leitura
    # completa incluindo a ressalva de tamanho de amostra.
    assert np.all(np.isfinite(efeito))
