"""resumo_parse.py — Parsing PURO do resumo_modelo.txt (item 19 da auditoria:
lógica testável extraída da UI).

O pipeline grava um resumo de texto livre (logs/resumo_modelo.txt); tanto a
aba Validation quanto os 5 geradores de relatório (guaraci.reports) extraíam
métricas dele por regex — cada gerador com sua PRÓPRIA cópia do helper `_ex`
(5 cópias idênticas) e PDF/Word repetindo o MESMO dicionário de 12 métricas.
Este módulo centraliza esse parsing: nada de I/O, só texto → dados, testável
em isolamento (ver tests/test_resumo_parse.py).
"""
from __future__ import annotations

import re
from typing import Dict, Optional

# Métricas padrão exibidas no cabeçalho de relatório (PDF/Word). Cada valor é
# (padrão_regex) aplicado ao resumo com IGNORECASE|MULTILINE; group(1) é o valor.
_PADROES_METRICAS = {
    "Balanced Accuracy (CV)":   r"[Bb]alanced[_ ]?[Aa]ccuracy.*?[:=]\s*([\d.]+)",
    "AUC macro OvR":            r"ROC AUC macro.*?[:=]\s*([\d.]+)",
    "R2Y":                      r"\bR2Y\b.*?[:=]\s*([\d.]+)",
    "Q2Y":                      r"\bQ2\b.*?[:=]\s*([\d.E+-]+)",
    "R2X":                      r"\bR2X\b.*?[:=]\s*([\d.]+)",
    "Optimal LVs":              r"Optimal LVs.*?[:=]\s*(\d+)",
    "p-value (permutation)":    r"p.?value.*?[:=]\s*([\d.E+-]+)",
    "Preprocessing":            r"[Pp]re.?[Pp]rocessamento.*?[:=]\s*([A-Za-z0-9_+]+)",
    "Hotelling T2 UCL (95%)":   r"[Hh]otelling.*?[:=]\s*([\d.]+)",
    "Q-residual UCL (95%)":     r"Q.residual.*?[:=]\s*([\d.E+-]+)",
    "n samples (training)":     r"[Nn]\s+treino.*?[:=]\s*(\d+)",
    "n classes":                r"[Nn]\.?\s*[Cc]lasses.*?[:=]\s*(\d+)",
}


# Contagens do conjunto de dados da execucao. As chaves do resumo mudaram de
# PT para EN ao longo do projeto ("Total de amostras" -> "Total samples"), e
# runs antigos continuam em disco -- os dois nomes sao aceitos, senao o painel
# de status mostraria "-" para uma execucao que tem o numero gravado.
_PADROES_CONTAGEM = {
    "amostras":  r"^\s*Total (?:de )?(?:samples|amostras)\s*[:=]\s*(\d+)",
    "variaveis": r"^\s*Total (?:de )?(?:variables|vari[aá]veis)\s*[:=]\s*(\d+)",
    "classes":   r"^\s*Total (?:de )?classes\s*[:=]\s*(\d+)",
}


def parse_dataset_counts(resumo: str) -> Dict[str, Optional[int]]:
    """Contagens reais da execucao (amostras/variaveis/classes) lidas do
    resumo_modelo.txt. Valor `None` quando a linha nao existe -- nunca zero,
    que se leria como "nenhuma amostra" em vez de "nao informado"."""
    contagens: Dict[str, Optional[int]] = {}
    for nome, padrao in _PADROES_CONTAGEM.items():
        bruto = extract_metric(resumo, padrao, default="")
        contagens[nome] = int(bruto) if bruto.isdigit() else None
    return contagens


def extract_metric(resumo: str, padrao: str, default: str = "-") -> str:
    """Extrai group(1) do primeiro casamento de `padrao` no resumo.

    IGNORECASE|MULTILINE, `.strip()` no valor; devolve `default` se nada casar.
    É o núcleo único do antigo `_ex` (que existia em 5 cópias nos geradores).
    """
    m = re.search(padrao, resumo or "", re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else default


def parse_model_metrics(resumo: str, default: str = "-") -> Dict[str, str]:
    """Dicionário padrão de 12 métricas (Balanced Accuracy, R2Y/Q2Y, LVs,
    p-valor, etc.) para o cabeçalho dos relatórios PDF/Word."""
    return {nome: extract_metric(resumo, padrao, default)
            for nome, padrao in _PADROES_METRICAS.items()}


def parse_accuracy_by_class(resumo: str) -> Dict[str, float]:
    """Extrai a acurácia (recall) por classe das linhas 'Acc <classe>: <val>'
    do resumo. Usado pela aba Validation para a tabela colorida por classe.
    Retorna {classe: valor_float}; dict vazio se não houver nenhuma linha."""
    acc: Dict[str, float] = {}
    for linha in (resumo or "").splitlines():
        m = re.match(r"\s*Acc\s+(.+?)\s*[:=]\s*([\d.]+)", linha)
        if m:
            acc[m.group(1).strip()] = float(m.group(2))
    return acc


__all__ = ["extract_metric", "parse_model_metrics", "parse_accuracy_by_class",
           "parse_dataset_counts"]
