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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from guaraci.chemometric_stats import (
    applicability_domain_new_samples,
    combined_distance,
)
from guaraci.config import __version__ as _guaraci_version
from guaraci.identificacao import (
    CoverageStatus,
    IdentificationResult,
    combine_alpha_bonferroni,
    identify_sample,
)

_CHAVES_PACOTE_REQUERIDAS = {
    "preprocessador", "pls_final", "label_binarizer", "wavenumbers"}
# Chaves OPCIONAIS do Dominio de Aplicabilidade (AD, PCA exploratorio) --
# pacotes salvos por versoes antigas do pipeline nao tem essas chaves, e a
# predicao continua funcionando normalmente (so' sem as colunas AD_*).
_CHAVES_AD = {"pca", "ad_var_t", "ad_h0", "ad_q0", "ad_Nh", "ad_Nq", "ad_f_crit"}

__all__ = [
    "SecurityError",
    "generate_manifest",
    "save_manifest",
    "load_model",
    "validate_model_package",
    "load_prediction_csv",
    "predict_samples",
    "QuantificationResult",
    "PurityResult",
    "detect_purity",
    "BlindPredictionResult",
    "quantify_sample",
    "predict_blind",
]


class SecurityError(Exception):
    """Operacao bloqueada por falta de confirmacao explicita de confianca,
    ou por falha de integridade detectada (arquivo alterado apos o manifesto
    ter sido gerado). Ver docs/SECURITY.md."""


# =========================================================================
#  Manifesto de proveniencia/integridade (P5 -- CLAUDE.md)
# =========================================================================
def generate_manifest(caminho_joblib: str, pkg: Dict[str, Any]) -> Dict[str, Any]:
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
        # Bloco 8 (2026-08-25): ausente em pacotes de versoes anteriores --
        # tratar ausencia como "high" seria uma afirmacao nao verificavel
        # sobre um modelo antigo, entao o default aqui e' explicitamente
        # "unknown", nao "high".
        "grouping_guarantee": pkg.get("grouping_guarantee", "unknown"),
        # Passo 57: sem esta marca, uma execucao sintetica (metricas quase
        # sempre perfeitas por construcao) vira template de referencia
        # indistinguivel de dado real. Ausente (False) em pacotes sem a
        # chave -- pacotes anteriores a este passo eram, em sua maioria,
        # de dado real (o proprio motivo de nao terem sido marcados antes).
        "dados_sinteticos": bool(pkg.get("dados_sinteticos", False)),
        # Bloco 9b: resumo de cobertura do ensemble de Identificacao (D5 --
        # a ressalva de nao-validacao tem que aparecer AQUI tambem, nao so'
        # no log/model card). Ausente (None) em pacotes sem o ensemble
        # (versoes anteriores ao Bloco 9b, ou dataset sem adulterante).
        "identification_coverage": _summarize_identification_coverage(pkg),
        # Bloco 9b (fechamento do gap do Detectar): resumo do DD-SIMCA de
        # pureza por especie -- terceiro lugar da MESMA ressalva (D5),
        # agora tambem para o portao de pureza, nao so' o de identificacao.
        "purity_coverage": _summarize_purity_coverage(pkg),
    }


def _summarize_purity_coverage(pkg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resumo do DD-SIMCA de pureza por especie (`detect_purity`) -- `None`
    se o pacote nao tem esses modelos persistidos (versoes anteriores a
    este fechamento do gap do Detectar)."""
    modelos = pkg.get("ddsimca_por_especie")
    if not modelos:
        return None
    n_confiavel = sum(1 for m in modelos.values()
                       if int(m.get("n_grupos_calibracao", 0)) >= 3)
    return {
        "n_especies": len(modelos),
        "n_confiavel": n_confiavel,
        "n_grupos_calibracao_por_especie": {
            esp: int(m.get("n_grupos_calibracao", 0))
            for esp, m in modelos.items()},
    }


def _summarize_identification_coverage(pkg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resumo, por status de cobertura, do ensemble de Identificacao
    (Bloco 9b) -- usado pelo manifesto e pelo model card. `None` se o
    pacote nao tem ensemble (versoes anteriores, ou dataset sem
    adulterante nomeavel).

    CORRIGIDO (achado na revisao com o usuario): a versao anterior tinha
    um UNICO booleano `quantificacao_disponivel` calculado so' de
    `bool(pkg.get("regressao_por_especie"))` -- "existe ALGUM pipeline de
    regressao no pacote", INDEPENDENTE de qualquer combinacao ter
    cobertura validada. Com 0/N combinacoes validadas (o caso comum),
    `quantify_sample` NUNCA roda de fato (gated por `identify_sample`,
    D4) -- mas o manifesto dizia `true`, escondendo exatamente a
    granularidade que o addendum do model card ja expõe por extenso. Quem
    le' o manifesto programaticamente (esse e' o proposito dele) leria
    `true` e trataria como garantido.

    Dois campos agora, nunca um booleano ambiguo so':
      quantificacao_disponivel_com_garantia -- True so' se PELO MENOS UMA
        combinacao validada tiver, para a MESMA especie, um pipeline de
        regressao persistido (a combinacao real que `predict_blind`
        conseguiria de fato quantificar sem bloquear).
      quantificacao_possivel_sem_garantia -- True se existe ALGUM
        pipeline de regressao no pacote, mesmo sem nenhuma combinacao
        validada (o caso "candidato informacional" que o model card ja
        descreve -- a maquinaria existe, so' nao tem garantia estatistica
        por tras da escolha de especie).
    """
    ensemble = pkg.get("identification_ensemble")
    if not ensemble:
        return None
    contagem = {s.value: 0 for s in CoverageStatus}
    for info in ensemble.values():
        status = info.get("cobertura_status")
        chave = status.value if hasattr(status, "value") else str(status)
        contagem[chave] = contagem.get(chave, 0) + 1

    regressao_especies = set(pkg.get("regressao_por_especie") or {})
    especies_validadas = {
        esp for (esp, _adult), info in ensemble.items()
        if info.get("cobertura_status") == CoverageStatus.VALIDATED}
    return {
        "n_combinacoes": len(ensemble),
        "por_status": contagem,
        "quantificacao_disponivel_com_garantia": bool(
            especies_validadas & regressao_especies),
        "quantificacao_possivel_sem_garantia": bool(regressao_especies),
    }


def save_manifest(caminho_joblib: str, pkg: Dict[str, Any]) -> str:
    """Gera e grava `<caminho_joblib>.manifest.json` ao lado do modelo.
    Chamado por pipeline.executar() logo apos joblib.dump(). Retorna o
    caminho do manifesto escrito."""
    manifesto = generate_manifest(caminho_joblib, pkg)
    caminho_manifesto = caminho_joblib + ".manifest.json"
    with open(caminho_manifesto, "w", encoding="utf-8") as f:
        json.dump(manifesto, f, ensure_ascii=False, indent=2)
    return caminho_manifesto


def load_model(caminho: str, *, confiar: bool = False) -> Dict[str, Any]:
    """Carrega um pacote de modelo `.joblib` treinado pelo Guaraci.

    AVISO DE SEGURANCA: `.joblib` usa pickle, que EXECUTA CODIGO ARBITRARIO
    contido no arquivo NO MOMENTO DO CARREGAMENTO -- antes de qualquer
    validacao de conteudo ser sequer possivel (`validate_model_package` so'
    roda DEPOIS, tarde demais para impedir a execucao). Carregue apenas
    modelos que voce mesmo treinou ou de origem CONHECIDA e CONFIAVEL.
    Ver `docs/SECURITY.md`.

    `confiar=True` e' uma confirmacao EXPLICITA da origem -- nao existe
    verificacao automatica de "isto e' seguro" para pickle (nao e'
    sandboxed). Se um manifesto (`<caminho>.manifest.json`, gerado por
    `save_manifest` no momento do treino) existir ao lado do arquivo, o
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
    try:
        return joblib.load(caminho)
    except Exception as e:  # noqa: BLE001 -- pickle corrompido/truncado/de
        # formato errado lanca varias classes diferentes (ModuleNotFoundError,
        # UnpicklingError, EOFError, ValueError...) dependendo de COMO esta
        # quebrado -- todas viram a mesma mensagem acionavel aqui, em vez de
        # vazar o erro cru do pickle pro usuario (achado de auditoria,
        # 2026-09-01: "ModuleNotFoundError: No module named 'sto nao e um
        # pickle valido...'" nao diz o que aconteceu nem o que fazer).
        raise ValueError(
            f"'{caminho}' nao pode ser carregado como modelo .joblib -- "
            f"arquivo corrompido, truncado, ou nao e' um .joblib de verdade "
            f"(detalhe tecnico: {type(e).__name__}: {e}). Confirme o caminho "
            "e re-exporte o modelo se necessario.") from e


def validate_model_package(pkg: Dict) -> None:
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


def load_prediction_csv(caminho_ou_buffer) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Le um CSV de espectros novos (colunas=numeros de onda, sem coluna de
    classe) e separa as colunas numericas (espectro) das colunas de
    metadados (ex.: nome da amostra), se houver.

    Retorna (X, wavenumbers, metadados_df). `metadados_df` pode ter 0
    colunas (nenhuma coluna nao-numerica encontrada).
    """
    import csv as _csv_mod

    try:
        df = pd.read_csv(caminho_ou_buffer, sep=None, engine="python")
    except (pd.errors.EmptyDataError, _csv_mod.Error) as e:
        # csv.Error ("Could not determine delimiter") e' o que o sniffer
        # (sep=None, engine="python") de fato lanca pra' um arquivo VAZIO --
        # EmptyDataError so' cobre outros casos de "sem coluna pra' ler".
        raise ValueError(
            "O arquivo CSV esta vazio -- nao ha' nenhuma linha (nem "
            "cabecalho) pra' ler. Confirme que o caminho/upload aponta pro "
            "arquivo certo.") from e
    except pd.errors.ParserError as e:
        raise ValueError(
            f"Nao foi possivel interpretar o arquivo como CSV (detalhe "
            f"tecnico: {e}). Confirme que e' mesmo um CSV de texto (nao "
            "um Excel/binario renomeado) e que as linhas tem o mesmo "
            "numero de colunas.") from e
    if df.empty or len(df.columns) == 0:
        raise ValueError(
            "O CSV nao tem nenhuma coluna de dado -- so' cabecalho, ou "
            "arquivo vazio. Confirme o conteudo do arquivo.")

    def _colunas_numericas(frame: pd.DataFrame) -> list:
        cols = []
        for c in frame.columns:
            try:
                float(c)
                cols.append(c)
            except ValueError:
                pass
        return cols

    num_cols = _colunas_numericas(df)
    if not num_cols:
        raise ValueError(
            "Nenhuma coluna com nome numerico (numero de onda) encontrada. "
            "Garanta que os cabecalhos das colunas espectrais sejam numeros "
            "de onda (ex.: 4000.5, 4001.0...).")
    wn = np.array([float(c) for c in num_cols])
    try:
        X = df[num_cols].values.astype(float)
    except ValueError:
        # Valores com decimal "," (padrao BR/Excel PT-BR) -- o proprio CSV
        # de SAIDA do Guaraci usa sep=";", decimal="," (ver _menu_prediction
        # em guaraci.py), entao rejeitar esse formato na ENTRADA seria o
        # pipeline nao aceitar o proprio formato que ele produz (achado de
        # auditoria, 2026-09-01). Cabecalho (numero de onda) ja' e' numerico
        # dos dois jeitos -- so' os VALORES precisam reparse com decimal=",".
        try:
            if hasattr(caminho_ou_buffer, "seek"):
                caminho_ou_buffer.seek(0)
            df_virgula = pd.read_csv(caminho_ou_buffer, sep=None,
                                      engine="python", decimal=",")
            X = df_virgula[num_cols].values.astype(float)
            df = df_virgula
        except (pd.errors.EmptyDataError, pd.errors.ParserError,
                _csv_mod.Error, KeyError, ValueError) as e:
            raise ValueError(
                "Nao foi possivel converter os valores espectrais para "
                "numero, nem com ponto nem com virgula decimal (padrao "
                "BR/Excel). Confirme o formato numerico do arquivo.") from e
    meta_df = df.drop(columns=num_cols, errors="ignore")
    return X, wn, meta_df


def _interpolate_to_reference(pkg: Dict, X_new_raw: np.ndarray,
                              wn_new: np.ndarray) -> np.ndarray:
    """Interpola espectros novos para o eixo de numero de onda do TREINO
    (faixa `wn_min`/`wn_max` usada na calibracao). Extraida de
    `predict_samples` (Bloco 9b) para ser reaproveitada tambem por
    `predict_blind` -- a quantificacao por especie precisa do espectro
    interpolado em RAW (antes do preprocessador de classificacao), nao do
    `X_proc` que `predict_samples` calcula para o modelo PLS-DA.
    """
    wn_train = np.asarray(pkg["wavenumbers"], dtype=float)
    wn_min   = float(pkg.get("wn_min", wn_train.min()))
    wn_max   = float(pkg.get("wn_max", wn_train.max()))
    mask_ref = (wn_train >= wn_min) & (wn_train <= wn_max)
    wn_ref   = wn_train[mask_ref]

    X_interp = np.zeros((X_new_raw.shape[0], len(wn_ref)))
    wn_new_f = np.asarray(wn_new, dtype=float)
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
    return X_interp


def predict_samples(pkg: Dict, X_new_raw: np.ndarray,
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
      `chemometric_stats.applicability_domain_new_samples` com os
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
    if wn_new is None:
        raise ValueError("wn_new nao pode ser None")

    X_interp = _interpolate_to_reference(pkg, X_new_raw, wn_new)

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
        f_new = combined_distance(T2_new, Q_new, float(pkg["pls_h0"]),
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
        ad = applicability_domain_new_samples(
            pkg["pca"], X_proc, pkg["ad_var_t"],
            pkg["ad_h0"], pkg["ad_q0"], pkg["ad_Nh"], pkg["ad_Nq"],
            pkg["ad_f_crit"])
        resultado["AD_T2"] = np.round(ad["t2"], 3)
        resultado["AD_Q"] = np.round(ad["q"], 6)
        resultado["AD_f"] = np.round(ad["f"], 3)
        resultado["AD_f_crit"] = round(float(ad["f_crit"]), 3)
        resultado["AD_dentro_dominio"] = ad["dentro_dominio"]

    return resultado


# =========================================================================
#  Bloco 9b -- fluxo completo do mode cego: Detectar -> Identificar ->
#  Quantificar (D6: estende a predicao existente em vez de um comando novo).
# =========================================================================

@dataclass
class QuantificationResult:
    """Resultado estruturado de `quantify_sample` (D4, Bloco 9b). Nunca
    lanca excecao quando bloqueado -- `teor_estimado` fica `None` e
    `motivo_bloqueio` diz por que."""

    teor_estimado: Optional[float] = None
    especie_usada: Optional[str] = None
    motivo_bloqueio: Optional[str] = None


@dataclass
class PurityResult:
    """Resultado de `detect_purity` -- segundo sinal de Detectar (Bloco 9b),
    complementar ao dominio de aplicabilidade (AD).

    AD e' ajustado em TODA a amostragem (pura + adulterada): responde "isto
    e' parecido com algo que vimos no treino", nao "isto e' puro". Uma
    amostra adulterada passa tranquilamente pelo AD -- ela FAZ parte do
    treino do AD. `detect_purity` usa o DD-SIMCA ajustado SO' nos puros da
    especie predita, que e' quem responde a pergunta de pureza de fato.

    `aceito=None` quando a especie predita nao tem modelo DD-SIMCA
    persistido (pacote antigo, ou especie sem amostra pura suficiente) --
    nunca um `True`/`False` fabricado. `alpha_nominal` so' vem preenchido
    (0.05) quando `confiavel` (`n_grupos_calibracao>=3`, o MESMO limiar ja
    usado no aviso de `DDSimca.fit` para "regiao larga/conservadora por
    construcao") -- com calibracao mais fraca (o caso comum: 1 amostra pura
    por especie), o metodo AINDA decide aceitar/rejeitar (chi2 parametrico,
    nao conformal -- nao se recusa como `ConformalOneClass`), mas o alpha
    declarado nao teria base solida, entao fica de fora da soma de
    Bonferroni em vez de inflar `alpha_total` com um numero sem lastro.
    """

    aceito: Optional[bool]
    f: Optional[float]
    f_crit: Optional[float]
    n_grupos_calibracao: Optional[int]
    confiavel: bool
    alpha_nominal: Optional[float]


def detect_purity(pkg: Dict[str, Any], especie: Optional[str],
                   X_proc_amostra: np.ndarray) -> PurityResult:
    """Aplica o DD-SIMCA da `especie` (persistido em `pkg["ddsimca_por_
    especie"]`, ver `pipeline.executar()`) a UMA amostra ja pre-processada
    (mesmo preprocessador da classificacao). `especie` tipicamente vem de
    `classe_pred` (predicao N1) -- e' a especie que o classificador disse
    que a amostra e', e o que se testa aqui e' "essa amostra e' pura PARA
    essa especie", nao uma alegacao de identidade nova.
    """
    modelos = pkg.get("ddsimca_por_especie") or {}
    m = modelos.get(especie) if especie is not None else None
    if m is None:
        return PurityResult(aceito=None, f=None, f_crit=None,
                             n_grupos_calibracao=None, confiavel=False,
                             alpha_nominal=None)

    X = np.asarray(X_proc_amostra, dtype=float).reshape(1, -1)
    pca = m["pca"]
    T = np.asarray(pca.transform(X), dtype=float)
    X_rec = np.asarray(pca.inverse_transform(T), dtype=float)
    Q = np.sum((X - X_rec) ** 2, axis=1)
    var_t = np.asarray(m["var_t"], dtype=float)
    T2 = np.sum((T ** 2) / var_t, axis=1)
    f = float(combined_distance(T2, Q, m["h0"], m["q0"], m["Nh"], m["Nq"])[0])
    f_crit = float(m["f_crit"])
    n_grupos = int(m["n_grupos_calibracao"])
    confiavel = n_grupos >= 3
    return PurityResult(
        aceito=bool(f <= f_crit), f=f, f_crit=f_crit,
        n_grupos_calibracao=n_grupos, confiavel=confiavel,
        alpha_nominal=(0.05 if confiavel else None))


@dataclass
class BlindPredictionResult:
    """Resultado de UMA amostra no fluxo completo Detectar -> Identificar
    -> Quantificar (D6). Detectar tem DOIS sinais complementares (AD +
    pureza DD-SIMCA por especie) -- ver docstring de `PurityResult`."""

    detectado_no_dominio: Optional[bool]
    pureza: PurityResult
    identificacao: IdentificationResult
    quantificacao: QuantificationResult
    alpha_total: Optional[float]


def quantify_sample(pkg: Dict[str, Any], X_interp_amostra: np.ndarray,
                     identificacao: IdentificationResult
                     ) -> QuantificationResult:
    """Quantifica o teor de adulterante de UMA amostra -- gated pelo
    resultado de Identificar (D4). `X_interp_amostra` e' o espectro
    interpolado para o eixo de referencia (RAW, ANTES do preprocessador de
    classificacao -- os pipelines de regressao por especie tem o proprio
    preprocessador interno, ajustado separadamente em
    `pipeline.pls_regression_by_species`).

    NUNCA lanca excecao e NUNCA forca uma especie: sem identificacao com
    cobertura validada, devolve um resultado estruturado com o motivo
    (`identificacao_desconhecida` ou `identificacao_ambigua`), nunca um
    numero (D4).
    """
    if identificacao.classe_identificada is None:
        if (identificacao.cobertura_status == CoverageStatus.VALIDATED
                and len(identificacao.candidatos_ambiguos) >= 2):
            motivo = "identificacao_ambigua"
        else:
            motivo = "identificacao_desconhecida"
        return QuantificationResult(motivo_bloqueio=motivo)

    especie, _adulterante = identificacao.classe_identificada.split("|", 1)
    modelos = pkg.get("regressao_por_especie") or {}
    info = modelos.get(especie)
    if info is None:
        return QuantificationResult(
            especie_usada=especie,
            motivo_bloqueio="sem_modelo_de_regressao_para_especie")

    X = np.asarray(X_interp_amostra, dtype=float).reshape(1, -1)
    pred = np.asarray(info["pipeline"].predict(X)).flatten()
    return QuantificationResult(
        teor_estimado=float(pred[0]), especie_usada=especie)


def predict_blind(pkg: Dict[str, Any], X_new_raw: np.ndarray,
                   wn_new: np.ndarray) -> Tuple[pd.DataFrame,
                                                 List[BlindPredictionResult]]:
    """Fluxo completo do mode cego (D6): Detectar -> Identificar ->
    Quantificar, para um lote de espectros novos.

    - Detectar: DOIS sinais complementares -- `predict_samples` (Dominio de
      Aplicabilidade global, coluna `AD_dentro_dominio`, D do Bloco 9a, NAO
      alterado) + `detect_purity` (DD-SIMCA da especie predita por N1, novo
      -- ver docstring de `PurityResult` para por que os dois nao sao
      redundantes: AD responde "parecido com o treino", DD-SIMCA responde
      "puro para a especie predita").
    - Identificar: `identificacao.identify_sample` contra
      `pkg["identification_ensemble"]` (Bloco 9b) -- so' preenche uma
      classe quando a cobertura e' VALIDADA.
    - Quantificar: `quantify_sample`, gated pelo resultado de Identificar
      (D4) -- nunca forca especie/numero.
    - `alpha_total`: limite de uniao (Bonferroni) sobre os alpha NOMINAIS
      declarados de cada portao com base solida -- AD (0,05, quando
      disponivel), DD-SIMCA de pureza (0,05, so' quando `confiavel`) e
      `alpha_alcancavel` do Identificar. Um portao sem alpha confiavel
      (`None`) faz a soma inteira virar `None` (`combine_alpha_bonferroni`)
      -- nunca um numero inflado por um portao sem lastro estatistico. A
      Quantificacao (regressao PLS) nao tem um alpha de cobertura proprio
      neste design -- nao entra na soma.

    Retorna (DataFrame de `predict_samples` com as colunas ja existentes,
    lista de `BlindPredictionResult` -- uma por amostra, na mesma ordem).
    """
    df = predict_samples(pkg, X_new_raw, wn_new)
    X_interp = _interpolate_to_reference(pkg, X_new_raw, wn_new)

    ensemble = pkg.get("identification_ensemble") or {}
    tem_ad = _CHAVES_AD.issubset(pkg.keys())
    tem_pca_identificacao = "pca" in pkg and "ad_var_t" in pkg

    resultados: List[BlindPredictionResult] = []
    for i in range(X_interp.shape[0]):
        detectado = (bool(df.loc[i, "AD_dentro_dominio"])
                     if tem_ad else None)
        alpha_detectar = 0.05 if tem_ad else None

        X_proc_i = pkg["preprocessador"].transform(X_interp[i:i + 1])

        especie_pred = (str(df.loc[i, "classe_pred"])
                         if "classe_pred" in df.columns else None)
        pureza = detect_purity(pkg, especie_pred, X_proc_i[0])
        alpha_pureza = pureza.alpha_nominal

        if ensemble and tem_pca_identificacao:
            ident = identify_sample(
                ensemble, pkg["pca"], pkg["ad_var_t"], X_proc_i[0])
        else:
            ident = IdentificationResult(
                classe_identificada=None, candidatos_ambiguos=[],
                cobertura_status=None, alpha_alcancavel=None, escores={})

        quant = quantify_sample(pkg, X_interp[i], ident)
        alpha_total = combine_alpha_bonferroni(
            alpha_detectar, alpha_pureza, ident.alpha_alcancavel)

        resultados.append(BlindPredictionResult(
            detectado_no_dominio=detectado, pureza=pureza,
            identificacao=ident, quantificacao=quant,
            alpha_total=alpha_total))

    return df, resultados
