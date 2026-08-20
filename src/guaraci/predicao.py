# -*- coding: utf-8 -*-
"""Predicao em amostras desconhecidas a partir de um modelo salvo (.joblib).

Modulo PURO (numpy/pandas apenas) extraido de app_quimiometria.py para que
o app web e o CLI (guaraci.py) usem exatamente a mesma logica cientifica --
mesmo padrao de extracao da Fase H (chemometric_stats.py, dados_io.py etc.):
mover codigo coeso + reexportar por nome, nunca duplicar.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from guaraci.chemometric_stats import dominio_aplicabilidade_amostras_novas
from guaraci.config import __version__ as _guaraci_version

_CHAVES_PACOTE_REQUERIDAS = {
    "preprocessador", "pls_final", "label_binarizer", "wavenumbers"}
# Chaves OPCIONAIS do Dominio de Aplicabilidade (AD, PCA exploratorio) --
# pacotes salvos por versoes antigas do pipeline nao tem essas chaves, e a
# predicao continua funcionando normalmente (so' sem as colunas AD_*).
_CHAVES_AD = {"pca", "ad_var_t", "ad_h0", "ad_q0", "ad_Nh", "ad_Nq", "ad_f_crit"}


class SecurityError(Exception):
    """Operacao bloqueada por falta de confirmacao explicita de confianca,
    ou por falha de integridade detectada (arquivo alterado apos o manifesto
    ter sido gerado). Ver docs/SECURITY.md."""


# =========================================================================
#  Manifesto de proveniencia/integridade (P5 -- CLAUDE.md)
# =========================================================================
def gerar_manifesto(caminho_joblib: str, pkg: Dict[str, Any]) -> Dict[str, Any]:
    """Monta o manifesto de proveniencia de um pacote de modelo ja salvo em
    disco. Resolve seguranca (hash p/ detectar arquivo trocado/corrompido)
    E reprodutibilidade (versoes exatas de biblioteca) na mesma estrutura.
    """
    import sklearn
    sha256 = hashlib.sha256(Path(caminho_joblib).read_bytes()).hexdigest()
    return {
        "guaraci_version": _guaraci_version,
        "sklearn_version": sklearn.__version__,
        "numpy_version":   np.__version__,
        "python_version":  sys.version.split()[0],
        "sha256":          sha256,
        "gerado_em":       datetime.now(timezone.utc).isoformat(),
        "classes":         [str(c) for c in pkg.get("classes", [])],
        "n_variaveis":     int(len(pkg["wavenumbers"]))
                           if "wavenumbers" in pkg else None,
    }


def salvar_manifesto(caminho_joblib: str, pkg: Dict[str, Any]) -> str:
    """Gera e grava `<caminho_joblib>.manifest.json` ao lado do modelo.
    Chamado por pipeline.executar() logo apos joblib.dump(). Retorna o
    caminho do manifesto escrito."""
    manifesto = gerar_manifesto(caminho_joblib, pkg)
    caminho_manifesto = caminho_joblib + ".manifest.json"
    with open(caminho_manifesto, "w", encoding="utf-8") as f:
        json.dump(manifesto, f, ensure_ascii=False, indent=2)
    return caminho_manifesto


def carregar_modelo(caminho: str, *, confiar: bool = False) -> Dict[str, Any]:
    """Carrega um pacote de modelo `.joblib` treinado pelo Guaraci.

    AVISO DE SEGURANCA: `.joblib` usa pickle, que EXECUTA CODIGO ARBITRARIO
    contido no arquivo NO MOMENTO DO CARREGAMENTO -- antes de qualquer
    validacao de conteudo ser sequer possivel (`validar_pacote_modelo` so'
    roda DEPOIS, tarde demais para impedir a execucao). Carregue apenas
    modelos que voce mesmo treinou ou de origem CONHECIDA e CONFIAVEL.
    Ver `docs/SECURITY.md`.

    `confiar=True` e' uma confirmacao EXPLICITA da origem -- nao existe
    verificacao automatica de "isto e' seguro" para pickle (nao e'
    sandboxed). Se um manifesto (`<caminho>.manifest.json`, gerado por
    `salvar_manifesto` no momento do treino) existir ao lado do arquivo, o
    hash SHA-256 e' conferido ANTES de chamar joblib.load -- isso E' uma
    protecao real (deteccao de arquivo trocado/corrompido acontece antes da
    execucao do pickle, nao depois). Sem manifesto, nao ha' nada a conferir;
    a unica garantia continua sendo a decisao humana em `confiar=True`.
    """
    if not confiar:
        raise SecurityError(
            f"Carregar '{caminho}' executa qualquer codigo contido no "
            "arquivo (formato pickle) -- inclusive antes de qualquer "
            "validacao de conteudo. Passe confiar=True somente se voce "
            "confia na origem deste arquivo (voce mesmo treinou, ou fonte "
            "conhecida e verificada). Ver docs/SECURITY.md.")

    caminho_manifesto = str(caminho) + ".manifest.json"
    if os.path.isfile(caminho_manifesto):
        try:
            with open(caminho_manifesto, encoding="utf-8") as f:
                manifesto = json.load(f)
            sha_esperado = manifesto.get("sha256")
            if sha_esperado:
                sha_atual = hashlib.sha256(Path(caminho).read_bytes()).hexdigest()
                if sha_atual != sha_esperado:
                    raise SecurityError(
                        f"Integridade falhou: '{caminho}' nao bate com o "
                        f"hash SHA-256 registrado em '{caminho_manifesto}' "
                        "-- o arquivo foi alterado desde que o manifesto foi "
                        "gerado (ou pertence a outro modelo). Carregamento "
                        "BLOQUEADO antes de executar o pickle.")
        except (OSError, json.JSONDecodeError):
            pass   # manifesto ilegivel -- nao ha' nada a conferir, segue

    import joblib
    return joblib.load(caminho)


def validar_pacote_modelo(pkg: Dict) -> None:
    """Validacao minima de estrutura do pacote .joblib carregado.

    Nao valida CONTEUDO (nao ha como, com pickle) -- so' confirma que as
    chaves esperadas existem, para dar um erro claro em vez de deixar um
    AttributeError/KeyError confuso estourar mais adiante. Evita tambem
    tentar "usar" um pickle qualquer como se fosse um pacote de modelo.
    """
    if not _CHAVES_PACOTE_REQUERIDAS.issubset(pkg.keys()):
        raise ValueError(
            f"Modelo invalido: esperado as chaves {_CHAVES_PACOTE_REQUERIDAS}, "
            f"encontrado {set(pkg.keys())}")


def carregar_csv_predicao(caminho_ou_buffer) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Le um CSV de espectros novos (colunas=numeros de onda, sem coluna de
    classe) e separa as colunas numericas (espectro) das colunas de
    metadados (ex.: nome da amostra), se houver.

    Retorna (X, wavenumbers, metadados_df). `metadados_df` pode ter 0
    colunas (nenhuma coluna nao-numerica encontrada).
    """
    df = pd.read_csv(caminho_ou_buffer, sep=None, engine="python")
    num_cols = []
    for c in df.columns:
        try:
            float(c)
            num_cols.append(c)
        except ValueError:
            pass
    if not num_cols:
        raise ValueError(
            "Nenhuma coluna com nome numerico (numero de onda) encontrada. "
            "Garanta que os cabecalhos das colunas espectrais sejam numeros "
            "de onda (ex.: 4000.5, 4001.0...).")
    wn = np.array([float(c) for c in num_cols])
    X = df[num_cols].values.astype(float)
    meta_df = df.drop(columns=num_cols, errors="ignore")
    return X, wn, meta_df


def predizer_amostras(pkg: Dict, X_new_raw: np.ndarray,
                       wn_new: Optional[np.ndarray]) -> pd.DataFrame:
    """Aplica o pacote de modelo salvo a espectros novos.

    Interpola para o eixo de referencia do treino, aplica o pre-processador
    ajustado, calcula classe predita (scores PLS softmax-normalizados),
    e dois diagnosticos complementares de "esta amostra e' confiavel?":

    - T2/Q no espaco PLS (colunas T2/Q/aceito): mede o quanto a amostra se
      afasta do que o MODELO DE CLASSIFICACAO capturou -- ja existia.
    - Dominio de Aplicabilidade no espaco PCA (colunas AD_*, Jaworska et
      al. 2005): mede o quanto a amostra e' um espectro atipico frente ao
      dataset de calibracao em geral, INDEPENDENTE da classe -- reaproveita
      `chemometric_stats.dominio_aplicabilidade_amostras_novas` com os
      artefatos leves salvos no pacote (`pca`, `ad_var_t`, `ad_h0`, `ad_q0`,
      `ad_Nh`, `ad_Nq`, `ad_f_crit` -- distancia combinada do DD-SIMCA desde
      a correcao do achado A3, auditoria 2026-08-07). So' aparece se o
      pacote foi salvo por uma versao do pipeline que exporta esses campos
      (opcional, retrocompativel).

    Retorna um DataFrame com o diagnostico por amostra.
    """
    preproc = pkg["preprocessador"]
    pls     = pkg["pls_final"]
    lb      = pkg["label_binarizer"]
    wn_train = np.asarray(pkg["wavenumbers"], dtype=float)
    if wn_new is None:
        raise ValueError("wn_new nao pode ser None")
    wn_min   = float(pkg.get("wn_min", wn_train.min()))
    wn_max   = float(pkg.get("wn_max", wn_train.max()))

    # Eixo de referencia do treino (faixa usada durante o treinamento)
    mask_ref = (wn_train >= wn_min) & (wn_train <= wn_max)
    wn_ref   = wn_train[mask_ref]

    # Interpola espectros novos para o eixo de treino
    X_interp = np.zeros((X_new_raw.shape[0], len(wn_ref)))
    wn_new_f = wn_new.astype(float)
    # np.interp exige eixo CRESCENTE e nao ordena sozinho. Um .dx de terceiro
    # gravado em ordem decrescente (convencao comum em FTIR) produziria aqui
    # um espectro reamostrado errado -- e, como nada estoura, a PREDICAO sairia
    # errada em silencio. Este e' o caminho "aplicar modelo a amostra nova":
    # e' exatamente onde um resultado errado sem aviso e' mais grave.
    ordem = np.argsort(wn_new_f)
    wn_new_f = wn_new_f[ordem]
    for i in range(X_new_raw.shape[0]):
        X_interp[i] = np.interp(wn_ref, wn_new_f,
                                X_new_raw[i].astype(float)[ordem])

    # Aplica o pre-processamento do treino
    X_proc = preproc.transform(X_interp)

    # Scores PLS (aplica a centragem interna do modelo)
    T_new  = np.asarray(pls.transform(X_proc), dtype=float)
    P      = np.asarray(pls.x_loadings_, dtype=float)   # (p, k)
    P_T    = P.T                                          # (k, p)

    # Hotelling T2 -- mesma formula do pipeline (escalado pela variancia de treino)
    T_train = np.asarray(pls.x_scores_, dtype=float)
    var_t   = T_train.var(axis=0, ddof=1)
    var_t[var_t == 0] = 1.0
    T2_new  = np.sum((T_new ** 2) / var_t, axis=1)

    # Q-residuos -- mesma convencao do pipeline (X_proc nao subtraido)
    X_rec  = T_new @ P_T                                  # (n_new, p)
    Q_new  = np.sum((X_proc - X_rec) ** 2, axis=1)

    # UCL do pacote (gerado pelo pipeline v25+). O T2 tem fallback legitimo
    # -- e' derivavel dos scores de TREINO, que viajam dentro do modelo.
    t2_ucl = float(pkg.get("t2_ucl", np.percentile(
        np.sum((T_train ** 2) / var_t, axis=1), 95)))
    # O Q, nao: o fallback anterior era `percentile(Q_new, 99) * 1.5`, isto
    # e', o limite de aceitacao calculado a partir das PROPRIAS amostras que
    # estavam sendo julgadas. Um lote inteiro fora do dominio elevava o
    # limite junto e era aceito -- circular, e sem nenhum aviso. Nao ha'
    # substituto honesto para um limite que precisa vir do treino: se o
    # pacote nao traz `q_ucl`, o certo e' recusar, nao inventar.
    if "q_ucl" not in pkg:
        raise ValueError(
            "Pacote de modelo sem 'q_ucl' (limite de Q-residuos derivado do "
            "conjunto de TREINO). Modelos exportados por versoes anteriores "
            "a v25 nao o incluem. Retreine o modelo com a versao atual do "
            "pipeline -- estimar esse limite a partir das amostras que estao "
            "sendo julgadas daria um criterio circular.")
    q_ucl = float(pkg["q_ucl"])

    # Classe predita via scores PLS softmax-normalizados
    Y_soft  = np.asarray(pls.predict(X_proc), dtype=float)
    Y_clip  = np.clip(Y_soft, 0.0, 1.0)
    totais  = Y_clip.sum(axis=1, keepdims=True)
    totais[totais < 1e-12] = 1.0
    Y_norm  = Y_clip / totais

    classes    = list(lb.classes_)
    idx_pred   = Y_norm.argmax(axis=1)
    classe_pred = [classes[i] if i < len(classes) else "?" for i in idx_pred]
    confianca  = Y_norm.max(axis=1)

    n = X_new_raw.shape[0]
    resultado = pd.DataFrame({
        "amostra":    [f"S{i+1:03d}" for i in range(n)],
        "classe_pred": classe_pred,
        "confianca_%": np.round(confianca * 100, 1),
        "T2":          np.round(T2_new, 3),
        "T2_ucl":      round(t2_ucl, 3),
        "Q":           np.round(Q_new, 6),
        "Q_ucl":       round(q_ucl, 6),
        "T2_ok":       T2_new <= t2_ucl,
        "Q_ok":        Q_new  <= q_ucl,
    })

    # Decisao de aceitacao: distancia combinada f=(T2/h0)*Nh+(Q/q0)*Nq <=
    # chi2(1-alpha, Nh+Nq) -- Eq. 3-4 de Kucheryavskiy, Rodionova &
    # Pomerantsev (2024), J. Chemometrics 38(7):e3556.
    #
    # A versao anterior usava a regra RETANGULAR (T2_ok E Q_ok), com alpha
    # independente em cada eixo: alpha conjunto efetivo ~0,0975 em vez de
    # 0,05. E' a mesma regra ja corrigida no DD-SIMCA (2026-08-08) e no
    # dominio de aplicabilidade (achado A3, 2026-08-07) -- ficou aqui porque
    # esta e' uma quarta copia da mesma decisao, em outro modulo. As colunas
    # T2_ok/Q_ok seguem expostas como DIAGNOSTICO por eixo (util para saber
    # QUAL das duas distancias disparou), nunca como criterio.
    #
    # Pacotes salvos antes de 2026-08-17 nao trazem pls_h0/q0/Nh/Nq: nesse
    # caso cai na regra por eixo e diz isso na coluna `criterio`, em vez de
    # aplicar em silencio um alpha diferente do declarado.
    _chaves_comb = {"pls_h0", "pls_q0", "pls_Nh", "pls_Nq", "pls_f_crit"}
    if _chaves_comb.issubset(pkg.keys()):
        from guaraci.chemometric_stats import distancia_combinada
        f_new = distancia_combinada(T2_new, Q_new, float(pkg["pls_h0"]),
                                    float(pkg["pls_q0"]), float(pkg["pls_Nh"]),
                                    float(pkg["pls_Nq"]))
        f_crit = float(pkg["pls_f_crit"])
        resultado["f"] = np.round(f_new, 3)
        resultado["f_crit"] = round(f_crit, 3)
        resultado["aceito"] = f_new <= f_crit
        resultado["criterio"] = "distancia combinada (alpha=0.05)"
    else:
        resultado["aceito"] = (T2_new <= t2_ucl) & (Q_new <= q_ucl)
        resultado["criterio"] = (
            "regra por eixo (pacote antigo; alpha conjunto ~0.0975)")

    # Dominio de Aplicabilidade (opcional -- so' se o pacote tiver os
    # artefatos leves do PCA exploratorio; retrocompativel com pacotes
    # antigos, que simplesmente nao ganham essas colunas).
    if _CHAVES_AD.issubset(pkg.keys()):
        ad = dominio_aplicabilidade_amostras_novas(
            pkg["pca"], X_proc, pkg["ad_var_t"],
            pkg["ad_h0"], pkg["ad_q0"], pkg["ad_Nh"], pkg["ad_Nq"],
            pkg["ad_f_crit"])
        resultado["AD_T2"] = np.round(ad["t2"], 3)
        resultado["AD_Q"] = np.round(ad["q"], 6)
        resultado["AD_f"] = np.round(ad["f"], 3)
        resultado["AD_f_crit"] = round(float(ad["f_crit"]), 3)
        resultado["AD_dentro_dominio"] = ad["dentro_dominio"]

    return resultado
