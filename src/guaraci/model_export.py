# -*- coding: utf-8 -*-
"""model_export.py -- Exportacao portatil de modelo (Grupo 2, JSON puro).

POR QUE ESTE MODULO EXISTE
---------------------------
Hoje o unico formato de exportacao e' `.joblib` (pickle) -- `pacote_
modelo` guarda os objetos sklearn AJUSTADOS (`Pipeline`, `PLSRegression`,
`PCA`, `LabelBinarizer`) diretamente. Isso tem dois problemas registrados
em `SECURITY.md`:

1. **Reprodutibilidade**: um `.joblib` so' e' legivel por outro processo
   Python com as MESMAS classes disponiveis (mesma versao de sklearn
   compativel, mesmo `guaraci` instalado) -- nao e' um formato de dados,
   e' um dump de objetos.
2. **Seguranca**: pickle EXECUTA CODIGO ao carregar (`joblib.load` ==
   `pickle.load` por baixo) -- o app web ja trata isso como risco de RCE
   (checkbox `confia_modelo`, `GUARACI_DISABLE_MODEL_UPLOAD`, manifesto
   SHA256 em `predicao.py`). Um export em JSON puro nao pode executar
   codigo nenhum ao ser lido -- elimina essa classe de risco por
   construcao, nao so' mitiga.

ONNX/PMML foram avaliados e REJEITADOS: nao cabem na estrutura multi-especie/multi-artefato do pacote
GUARACI sem redesenho grande. Este modulo e' a alternativa "formato
simples" proposta no lugar deles -- ESCOPO DELIBERADAMENTE REDUZIDO: so'
o caminho de predicao PRINCIPAL (pre-processamento + PLS-DA/PLS-R +
classes), NAO o pacote inteiro (DD-SIMCA por especie, ensemble de
identificacao, dominio de aplicabilidade, conjunto conforme continuam
so' em `.joblib` -- exporta-los portavel fica de fora desta rodada,
registrado como extensao futura, nao como lacuna escondida).

RECUSA EM VEZ DE EXPORT INCOMPLETO/ERRADO
-------------------------------------------
`export_portable_json` levanta `ValueError` para qualquer preset de
pre-processamento ou `PLSRegression(scale=True)` que ainda nao sabe
serializar -- nunca produz um JSON que pareceria completo mas prediria
errado. Ver `_PRESETS_SUPORTADOS`.

EXATIDAO NUMERICA
------------------
`predict_portable` replica a formula EXATA de
`sklearn.cross_decomposition._pls._PLS.predict` (sklearn 1.9):
`Y = (X - _x_mean) @ coef_.T + intercept_` (sem divisao por `_x_std`
porque o pipeline SEMPRE usa `scale=False` -- unica convencao do
projeto, ver `chemometric_stats.py`/achado de auditoria 2026-09-12) --
confirmado numericamente contra `pls_final.predict()` em
`tests/test_model_export.py` (diff < 1e-9 em dado sintetico E no modelo
real produzido por `pipeline.executar()`).
"""
from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
from scipy.signal import savgol_filter

from guaraci.preprocessamento import _msc_aplicar, _pqn_aplicar, aplicar_no_eixo_do_modelo

__all__ = [
    "UnsupportedModelError",
    "export_portable_json",
    "save_portable_json",
    "load_portable_json",
    "predict_portable",
]

#: Presets de `preprocessamento.build_preprocessor` que este modulo sabe
#: serializar -- compostos so' de SNV/MSC/SavGol/StandardScaler (with_std=False)
#: e, desde a fase 2, AirPLS ("airpls_sg_mc", "airpls_snv_sg_mc") e a remocao de
#: raios cosmicos (etapa `despike`, que a [2.4] liga no Raman). Nenhum dos dois
#: aprende nada no `fit`: o AirPLS ajusta uma linha de base POR AMOSTRA (otimizacao
#: iterativa por espectro, sem estado entre amostras) e o despike olha um espectro por
#: vez; o JSON leva so' os PARAMETROS (lam, porder, itermax, tol; limiar, janela...) e
#: o consumidor refaz a otimizacao em cada espectro novo, com o MESMO codigo do treino.
#: Predicao bit a bit igual a do .joblib (testada). "custom" (combinacao arbitraria de
#: EMSC/OSC/PQN/EPO/GLSW) segue de fora -- ver docstring do modulo.
_PRESETS_SUPORTADOS = {"snv_sg_mc", "msc_sg_mc", "mc", "autoscaling", "snv_mc",
                       "airpls_sg_mc", "airpls_snv_sg_mc", "pqn_mc"}

_TIPOS_PASSO_SUPORTADOS = {"SNV", "MSC", "PQN", "SavGol", "StandardScaler", "AirPLS", "RemocaoRaiosCosmicos", "AlinhadorCOW"}


class UnsupportedModelError(ValueError):
    """Pacote de modelo usa um preset/passo/config que este modulo ainda
    nao sabe exportar de forma portatil -- ver docstring do modulo."""


def _serializar_passo(nome: str, step: Any) -> Dict[str, Any]:
    tipo = type(step).__name__
    if tipo == "SNV":
        return {"tipo": "SNV"}
    if tipo == "MSC":
        return {"tipo": "MSC", "ref": np.asarray(step.ref_, dtype=float).tolist()}
    if tipo == "PQN":
        return {"tipo": "PQN", "ref": np.asarray(step.ref_, dtype=float).tolist()}
    if tipo == "AlinhadorCOW":
        # Alinhamento de retencao: leva a referencia ESCOLHIDA NO TREINO e os parametros, para
        # alinhar amostras novas exatamente como o modelo foi treinado.
        return {"tipo": "AlinhadorCOW",
                "referencia": np.asarray(step.referencia_, dtype=float).tolist(),
                "n_segmentos": int(step.n_segmentos), "folga": int(step.folga)}
    if tipo == "SavGol":
        return {"tipo": "SavGol",
                "window_length": int(step.window_length),
                "polyorder": int(step.polyorder),
                "deriv": int(step.deriv)}
    if tipo == "StandardScaler":
        return {
            "tipo": "StandardScaler",
            "mean": (np.asarray(step.mean_, dtype=float).tolist()
                     if step.mean_ is not None else None),
            "scale": (np.asarray(step.scale_, dtype=float).tolist()
                      if step.scale_ is not None else None),
        }
    if tipo == "AirPLS":
        return {"tipo": "AirPLS", "lam": float(step.lam), "porder": int(step.porder),
                "itermax": int(step.itermax), "tol": float(step.tol)}
    if tipo == "RemocaoRaiosCosmicos":
        return {"tipo": "RemocaoRaiosCosmicos", "limiar": float(step.limiar), "janela": int(step.janela),
                "largura_max": int(step.largura_max), "razao_isolamento": float(step.razao_isolamento)}
    raise UnsupportedModelError(
        f"Passo de pre-processamento '{nome}' (tipo {tipo}) ainda nao e' "
        f"serializavel de forma portatil -- tipos suportados: "
        f"{sorted(_TIPOS_PASSO_SUPORTADOS)}. Export recusado (nunca "
        "incompleto/silenciosamente errado).")


def _aplicar_passo(passo: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    tipo = passo["tipo"]
    if tipo == "SNV":
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        return (X - mu) / sd
    if tipo == "MSC":
        return _msc_aplicar(X, np.asarray(passo["ref"], dtype=float))
    if tipo == "PQN":
        return _pqn_aplicar(X, np.asarray(passo["ref"], dtype=float))
    if tipo == "AlinhadorCOW":
        from guaraci.alinhamento_retencao import alinhar_com_referencia
        return alinhar_com_referencia(np.asarray(passo["referencia"], dtype=float), X,
                                      int(passo["n_segmentos"]), int(passo["folga"]))
    if tipo == "SavGol":
        return savgol_filter(X, window_length=passo["window_length"],
                              polyorder=passo["polyorder"],
                              deriv=passo["deriv"], axis=1)
    if tipo == "AirPLS":
        from guaraci.preprocessamento import AirPLS
        return AirPLS(lam=passo["lam"], porder=passo["porder"], itermax=passo["itermax"],
                      tol=passo["tol"]).transform(X)
    if tipo == "RemocaoRaiosCosmicos":
        from guaraci.preprocessamento import RemocaoRaiosCosmicos
        return RemocaoRaiosCosmicos(limiar=passo["limiar"], janela=passo["janela"],
                                    largura_max=passo["largura_max"],
                                    razao_isolamento=passo["razao_isolamento"]).transform(X)
    if tipo == "StandardScaler":
        Xt = X
        if passo["mean"] is not None:
            Xt = Xt - np.asarray(passo["mean"], dtype=float)
        if passo["scale"] is not None:
            Xt = Xt / np.asarray(passo["scale"], dtype=float)
        return Xt
    raise UnsupportedModelError(f"Passo desconhecido no JSON portatil: {tipo!r}")


def export_portable_json(pkg: Dict[str, Any]) -> Dict[str, Any]:
    """Extrai do pacote `.joblib` (dict de `pipeline.executar`) um dict
    JSON-serializavel com tudo que o caminho de predicao PRINCIPAL
    precisa: eixo de comprimento de onda, pre-processamento (passo a
    passo, so' arrays/escalares), coeficientes do PLS e classes.

    Levanta `UnsupportedModelError` para presets/passos/config que ainda
    nao sabe serializar -- ver docstring do modulo.
    """
    preset = str(pkg.get("preset", ""))
    if preset not in _PRESETS_SUPORTADOS:
        raise UnsupportedModelError(
            f"Exportacao portatil nao suporta o preset {preset!r} ainda "
            f"(suportados: {sorted(_PRESETS_SUPORTADOS)}).")

    pls = pkg["pls_final"]
    if bool(getattr(pls, "scale", False)):
        raise UnsupportedModelError(
            "Exportacao portatil so' suporta PLSRegression(scale=False) "
            "-- unica convencao usada pelo pipeline; pacote com "
            "scale=True nao e' suportado.")

    preproc = pkg["preprocessador"]
    passos = [_serializar_passo(nome, step)
              for nome, step in preproc.named_steps.items()]

    lb = pkg["label_binarizer"]
    wn = np.asarray(pkg["wavenumbers"], dtype=float)

    dados: Dict[str, Any] = {
        "formato": "guaraci-modelo-portatil",
        "versao_formato": 2,
        "wavenumbers": wn.tolist(),
        "wn_min": float(pkg.get("wn_min", wn.min())),
        "wn_max": float(pkg.get("wn_max", wn.max())),
        "preset": preset,
        # Exclusao de faixas e normalizacao por espectro usadas no treino: `wavenumbers` acima ja'
        # e' o eixo SEM as faixas; a predicao as reaplica (idempotente) para nao depender disso.
        "faixas_excluir": [str(f) for f in pkg.get("faixas_excluir", [])],
        "normalizacao": str(pkg.get("normalizacao", "nenhuma") or "nenhuma"),
        "ims_rip_janela": pkg.get("ims_rip_janela"),
        "passos_preprocessamento": passos,
        "pls": {
            "x_mean": np.asarray(pls._x_mean, dtype=float).tolist(),  # noqa: SLF001
            "coef": np.asarray(pls.coef_, dtype=float).tolist(),
            "intercept": np.asarray(pls.intercept_, dtype=float).tolist(),
        },
        "classes": [str(c) for c in lb.classes_],
    }
    lda = getattr(pls, "lda_", None)
    if hasattr(pls, "predict_pls"):
        # PLS-LDA (classificador: plslda): a decisao e' a da LDA sobre os escores, nao a do PLS
        # linear. Sem isto o portatil preveria como PLS-DA e divergiria do modelo treinado.
        if lda is None:
            raise UnsupportedModelError(
                "PLS-LDA com uma unica classe no treino nao tem o que exportar de forma portatil.")
        dados["versao_formato"] = 3   # 3: decisao por LDA nos escores (campo "plslda")
        dados["plslda"] = {
            "x_rotations": np.asarray(pls.x_rotations_, dtype=float).tolist(),
            "classes_lda": [int(c) for c in lda.classes_],
            "coef": np.asarray(lda.coef_, dtype=float).tolist(),
            "intercept": np.asarray(lda.intercept_, dtype=float).tolist(),
            "n_classes": int(len(lb.classes_)),
        }
    # R2-02 (revisao cega 2026-09-26): o portatil nao tinha dominio de
    # aplicabilidade e dava "confianca 100%" para ruido puro. Leva os MESMOS
    # artefatos leves do .joblib (PCA do treino + h0/q0/Nh/Nq/f_crit).
    _ad = ("pca", "ad_var_t", "ad_h0", "ad_q0", "ad_Nh", "ad_Nq", "ad_f_crit")
    if all(k in pkg for k in _ad):
        pca = pkg["pca"]
        dados["dominio_aplicabilidade"] = {
            "pca_mean": np.asarray(pca.mean_, dtype=float).tolist(),
            "pca_components": np.asarray(pca.components_, dtype=float).tolist(),
            "var_t": np.asarray(pkg["ad_var_t"], dtype=float).tolist(),
            "h0": float(pkg["ad_h0"]), "q0": float(pkg["ad_q0"]),
            "Nh": float(pkg["ad_Nh"]), "Nq": float(pkg["ad_Nq"]),
            "f_crit": float(pkg["ad_f_crit"]),
        }
    if pkg.get("multielementar"):
        dados["multielementar"] = dict(pkg["multielementar"])
    if pkg.get("imagem"):   # modo imagem: recorte, cartao de referencia e segmentacao do treino
        dados["imagem"] = dict(pkg["imagem"])
    dados["conteudo_sha256"] = _sha256_conteudo(dados)
    return dados


def _sha256_conteudo(dados: Dict[str, Any]) -> str:
    """SHA-256 do JSON canonico (sem o proprio campo). Detecta corrupcao/edicao
    acidental (R2-18); NAO e' assinatura contra adulteracao intencional."""
    import hashlib
    corpo = {k: v for k, v in dados.items() if k != "conteudo_sha256"}
    return hashlib.sha256(json.dumps(corpo, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def save_portable_json(pkg: Dict[str, Any], caminho: str) -> str:
    """`export_portable_json` + grava em disco (UTF-8, indentado)."""
    dados = export_portable_json(pkg)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    return caminho


def load_portable_json(caminho: str) -> Dict[str, Any]:
    """Le um JSON portatil -- SEM `pickle`/`joblib`, nenhum codigo pode
    executar ao carregar (elimina a classe de risco de RCE de `.joblib`,
    ver docstring do modulo)."""
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)
    if dados.get("formato") != "guaraci-modelo-portatil":
        raise ValueError(
            f"Arquivo nao e' um modelo portatil GUARACI "
            f"(campo 'formato' ausente/incorreto: {dados.get('formato')!r}).")
    verificar_integridade(dados)
    return dados


def verificar_integridade(dados: Dict[str, Any]) -> None:
    """Confere `conteudo_sha256` quando presente (formato >= 2). Levanta
    ValueError se o conteudo nao bate (arquivo corrompido ou editado)."""
    esperado = dados.get("conteudo_sha256")
    if esperado is not None and _sha256_conteudo(dados) != esperado:
        raise ValueError(
            "O modelo portatil foi alterado depois de exportado (o SHA-256 do "
            "conteudo nao confere). Exporte de novo a partir do modelo original.")


def _posteriores_plslda(p: Dict[str, Any], Xc: np.ndarray) -> np.ndarray:
    """Probabilidades a posteriori do PLS-LDA (mesma conta de sklearn LDA.predict_proba): escores
    = Xc @ rotacoes; decisao linear; sigmoide (2 classes) ou softmax (>2)."""
    T = Xc @ np.asarray(p["x_rotations"], dtype=float)
    d = T @ np.asarray(p["coef"], dtype=float).T + np.asarray(p["intercept"], dtype=float)
    if d.shape[1] == 1:
        p1 = 1.0 / (1.0 + np.exp(-d[:, 0]))
        prob = np.column_stack([1.0 - p1, p1])
    else:
        e = np.exp(d - d.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
    saida = np.zeros((T.shape[0], int(p["n_classes"])))
    saida[:, [int(c) for c in p["classes_lda"]]] = prob
    return saida


def predict_portable(portable: Dict[str, Any], X_new_raw: np.ndarray,
                      wn_new: np.ndarray) -> Dict[str, Any]:
    """Aplica um modelo portatil (de `export_portable_json`/`load_
    portable_json`) a espectros novos -- SO' numpy/scipy, nenhum sklearn/
    joblib necessario em tempo de predicao (o proprio ponto do formato:
    um consumidor externo minimo consegue reimplementar isto em outra
    linguagem a partir do JSON).

    Returns: dict com `classe_pred` (List[str]) e `confianca_%`
    (List[float]) -- mesma convencao de nome de `predicao.predict_
    samples` (nao um contrato novo).
    """
    wn_train = np.asarray(portable["wavenumbers"], dtype=float)
    wn_min = float(portable["wn_min"])
    wn_max = float(portable["wn_max"])
    mask_ref = (wn_train >= wn_min) & (wn_train <= wn_max)
    wn_ref = wn_train[mask_ref]

    X_new_raw = np.asarray(X_new_raw, dtype=float)
    wn_new = np.asarray(wn_new, dtype=float)
    # M-06: mesma recusa de extrapolacao do caminho .joblib.
    from guaraci.eixo_espectral import (COBERTURA_EIXO_MINIMA,
                                        exigir_cobertura_do_eixo)
    exigir_cobertura_do_eixo(
        wn_ref, wn_new, float(portable.get("cobertura_eixo_minima",
                                           COBERTURA_EIXO_MINIMA)))
    ordem = np.argsort(wn_new)
    wn_new_sorted = wn_new[ordem]

    X_interp = np.zeros((X_new_raw.shape[0], len(wn_ref)))
    for i in range(X_new_raw.shape[0]):
        X_interp[i] = np.interp(wn_ref, wn_new_sorted, X_new_raw[i][ordem])

    # mesma exclusao de faixas e normalizacao por espectro do treino (modelos antigos: nada a fazer)
    X, _ = aplicar_no_eixo_do_modelo(
        X_interp, wn_ref, portable.get("faixas_excluir") or [],
        portable.get("normalizacao", "nenhuma"), portable.get("ims_rip_janela"))
    for passo in portable["passos_preprocessamento"]:
        X = _aplicar_passo(passo, X)

    x_mean = np.asarray(portable["pls"]["x_mean"], dtype=float)
    coef = np.asarray(portable["pls"]["coef"], dtype=float)
    intercept = np.asarray(portable["pls"]["intercept"], dtype=float)
    Y_soft = (X - x_mean) @ coef.T + intercept
    if "plslda" in portable:
        Y_soft = _posteriores_plslda(portable["plslda"], X - x_mean)

    classes = portable["classes"]
    Y_clip = np.clip(Y_soft, 0.0, 1.0)
    totais = Y_clip.sum(axis=1, keepdims=True)
    totais = np.where(totais < 1e-12, 1.0, totais)
    Y_norm = Y_clip / totais
    idx_pred = Y_norm.argmax(axis=1)

    saida: Dict[str, Any] = {
        "classe_pred": [classes[i] for i in idx_pred],
        "confianca_%": (Y_norm.max(axis=1) * 100.0).tolist(),
    }
    ad = portable.get("dominio_aplicabilidade")
    if ad:
        mean = np.asarray(ad["pca_mean"], dtype=float)
        P = np.asarray(ad["pca_components"], dtype=float)
        var_t = np.asarray(ad["var_t"], dtype=float)
        Xc = X - mean
        T = Xc @ P.T
        t2 = np.sum(T ** 2 / var_t, axis=1)
        q = np.sum((Xc - T @ P) ** 2, axis=1)
        f = (t2 / max(ad["h0"], 1e-12)) * ad["Nh"] + (q / max(ad["q0"], 1e-12)) * ad["Nq"]
        saida["AD_f"] = np.round(f, 3).tolist()
        saida["AD_f_crit"] = round(float(ad["f_crit"]), 3)
        saida["AD_dentro_dominio"] = (f <= ad["f_crit"]).tolist()
    return saida
