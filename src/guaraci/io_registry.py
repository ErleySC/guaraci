"""io_registry.py — Registry de leitores de dados (item 20 da auditoria).

Formaliza o despacho por `cfg.modo` que antes era um if/elif fixo dentro de
`dados_io.carregar_dados()`. Cada leitor e' um `Callable[[Config], DadosCarregados]`
registrado sob uma chave `modo` (ex.: "dx", "csv", "imagem", "sintetico").

Adicionar um novo formato de entrada (uma nova tecnica analitica, um novo
layout de arquivo) passa a ser: escrever a funcao de leitura e chamar
`registrar_leitor("meu_modo", minha_funcao)` — de qualquer modulo, inclusive
fora do pacote guaraci — sem editar o dispatch de `carregar_dados()` nem
tocar em if/elif espalhados pelo nucleo cientifico.

Contrato do leitor (o mesmo que `carregar_dados()` sempre devolveu):
    (wavenumbers, X, rotulos, conc, mae_id, metadados_df)
    - wavenumbers: eixo espectral/variavel (1D)
    - X:           matriz de amostras x variaveis (2D)
    - rotulos:     classe/especie de cada amostra (1D, string)
    - conc:        concentracao/teor do adulterante, se aplicavel (ou None)
    - mae_id:      chave de agrupamento anti-vazamento (replicas fisicas),
                   None se o formato nao tiver essa nocao (ex.: CSV generico)
    - metadados_df: colunas extras nao-espectrais, se houver (ou None)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from guaraci.pipeline import Config

DadosCarregados = Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame],
]
LeitorDados = Callable[["Config"], DadosCarregados]

_LEITORES: Dict[str, LeitorDados] = {}


def registrar_leitor(modo: str, leitor: LeitorDados) -> None:
    """Registra (ou substitui) o leitor de dados para `cfg.modo == modo`.

    Chamar de novo com o mesmo `modo` SUBSTITUI o leitor anterior (util para
    testes ou para trocar a implementacao de um modo built-in).
    """
    _LEITORES[modo] = leitor


def obter_leitor(modo: str) -> LeitorDados:
    """Devolve o leitor registrado para `modo`.

    Levanta ValueError com a lista de modos disponiveis se `modo` nao foi
    registrado — mensagem de erro acionavel em vez de um KeyError cru.
    """
    try:
        return _LEITORES[modo]
    except KeyError:
        disponiveis = ", ".join(modos_registrados()) or "(nenhum registrado)"
        raise ValueError(
            f"Modo de entrada desconhecido: '{modo}'. "
            f"Modos disponiveis: {disponiveis}.") from None


def modos_registrados() -> Tuple[str, ...]:
    """Lista (ordenada) dos modos atualmente registrados."""
    return tuple(sorted(_LEITORES))


__all__ = ["DadosCarregados", "LeitorDados", "registrar_leitor",
           "obter_leitor", "modos_registrados"]
