# -*- coding: utf-8 -*-
"""model_export.py -- Exportacao portatil de modelo (Grupo 2, JSON puro).

POR QUE ESTE MODULO EXISTE
---------------------------
Hoje o unico formato de exportacao e' `.joblib` (pickle) -- `pacote_
modelo` guarda os objetos sklearn AJUSTADOS (`Pipeline`, `PLSRegression`,
`PCA`, `LabelBinarizer`) diretamente. Isso tem dois problemas registrados
em `docs/MAPA_COMPLETUDE_V1.md`/`docs/SECURITY.md`:

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

ONNX/PMML foram avaliados e REJEITADOS (ver `docs/MAPA_COMPLETUDE_V1.md`,
Grupo 2): nao cabem na estrutura multi-especie/multi-artefato do pacote
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

from guaraci.preprocessamento import _msc_aplicar

__all__ = [
    "UnsupportedModelError",
    "export_portable_json",
    "save_portable_json",
    "load_portable_json",
    "predict_portable",
]

#: Presets de `preprocessamento.build_preprocessor` que este modulo sabe
#: serializar -- todos compostos so' de SNV/MSC/SavGol/StandardScaler
#: (with_std=False). "airpls_sg_mc" (AirPLS tem estado de otimizacao por
#: amostra mais complexo) e "custom" (combinacao arbitraria de
#: EMSC/OSC/PQN/EPO/GLSW) ficam de fora nesta rodada -- ver docstring do
#: modulo.
_PRESETS_SUPORTADOS = {"snv_sg_mc", "msc_sg_mc", "mc", "autoscaling"}

_TIPOS_PASSO_SUPORTADOS = {"SNV", "MSC", "SavGol", "StandardScaler"}


class UnsupportedModelError(ValueError):
    """Pacote de modelo usa um preset/passo/config que este modulo ainda
    nao sabe exportar de forma portatil -- ver docstring do modulo."""


def _serializar_passo(nome: str, step: Any) -> Dict[str, Any]:
    tipo = type(step).__name__
    if tipo == "SNV":
        return {"tipo": "SNV"}
    if tipo == "MSC":
        return {"tipo": "MSC", "ref": np.asarray(step.ref_, dtype=float).tolist()}
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
    if tipo == "SavGol":
        return savgol_filter(X, window_length=passo["window_length"],
                              polyorder=passo["polyorder"],
                              deriv=passo["deriv"], axis=1)
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

    return {
        "formato": "guaraci-modelo-portatil",
        "versao_formato": 1,
        "wavenumbers": wn.tolist(),
        "wn_min": float(pkg.get("wn_min", wn.min())),
        "wn_max": float(pkg.get("wn_max", wn.max())),
        "preset": preset,
        "passos_preprocessamento": passos,
        "pls": {
            "x_mean": np.asarray(pls._x_mean, dtype=float).tolist(),  # noqa: SLF001
            "coef": np.asarray(pls.coef_, dtype=float).tolist(),
            "intercept": np.asarray(pls.intercept_, dtype=float).tolist(),
        },
        "classes": [str(c) for c in lb.classes_],
    }


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
    return dados


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
    ordem = np.argsort(wn_new)
    wn_new_sorted = wn_new[ordem]

    X_interp = np.zeros((X_new_raw.shape[0], len(wn_ref)))
    for i in range(X_new_raw.shape[0]):
        X_interp[i] = np.interp(wn_ref, wn_new_sorted, X_new_raw[i][ordem])

    X = X_interp
    for passo in portable["passos_preprocessamento"]:
        X = _aplicar_passo(passo, X)

    x_mean = np.asarray(portable["pls"]["x_mean"], dtype=float)
    coef = np.asarray(portable["pls"]["coef"], dtype=float)
    intercept = np.asarray(portable["pls"]["intercept"], dtype=float)
    Y_soft = (X - x_mean) @ coef.T + intercept

    classes = portable["classes"]
    Y_clip = np.clip(Y_soft, 0.0, 1.0)
    totais = Y_clip.sum(axis=1, keepdims=True)
    totais = np.where(totais < 1e-12, 1.0, totais)
    Y_norm = Y_clip / totais
    idx_pred = Y_norm.argmax(axis=1)

    return {
        "classe_pred": [classes[i] for i in idx_pred],
        "confianca_%": (Y_norm.max(axis=1) * 100.0).tolist(),
    }
