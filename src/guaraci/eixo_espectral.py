# -*- coding: utf-8 -*-
"""eixo_espectral.py -- cobertura do eixo espectral na PREDICAO.

`np.interp` fora do intervalo dos dados REPETE o valor da borda: um espectro
que cobre so' metade do eixo do modelo era "completado" com uma reta e o
modelo respondia com 99,3% de confianca, "aceito" e "dentro do dominio"
(medido em `revisor2/pred_test.py`). Isto e' extrapolacao silenciosa -- o
tipo de erro que o projeto existe para impedir. Aqui a cobertura e' medida
ANTES de interpolar e, abaixo do minimo, a predicao e' RECUSADA com uma
mensagem que diz o que faltou.
"""
from __future__ import annotations

import numpy as np

__all__ = ["EixoForaDoModeloError", "COBERTURA_EIXO_MINIMA",
           "cobertura_do_eixo", "exigir_cobertura_do_eixo",
           "ROTULO_EIXO_NEUTRO", "rotulo_do_eixo", "unidade_do_eixo",
           "tecnica_do_perfil", "chave_faixa_espectral",
           "UNIDADES_DO_EIXO", "unidade_declarada", "unidade_dimensional",
           "eixo_decrescente_por_unidade", "campos_faixa_espectral",
           "LIMITE_CORTE_FAIXA", "aviso_de_corte_de_faixa"]

#: Rotulo usado quando NENHUM perfil informa o que o eixo representa. O motor
#: serve NIR, MIR, Raman, UV-Vis, fluorescencia, RMN...: presumir "numero de
#: onda (cm-1)" rotulava errado qualquer outra tecnica (achado V75).
ROTULO_EIXO_NEUTRO = "Variável espectral"

_PERFIL_NEUTRO = "generico"

#: unidade declarada (perfil ou `cfg.axis_unit`) -> (rotulo, unidade para texto), em notacao mathtext.
#: "raman_shift" e' cm-1 (dimensao de numero de onda) mas com rotulo proprio; "min", "ppm" e "ms"
#: vem da tela [2.4] (cromatografia, RMN, IMS). "indice" nao e' unidade fisica: o rotulo
#: "Variable index" vale so' quando o usuario/[2.4] o declara (ver `rotulo_do_eixo`).
_UNIDADES_CONHECIDAS = {
    "cm-1": ("Wavenumber (cm$^{-1}$)", "cm⁻¹"),
    "nm": ("Wavelength (nm)", "nm"),
    "raman_shift": ("Raman shift (cm$^{-1}$)", "cm⁻¹"),
    "min": ("Retention time (min)", "min"),
    "ppm": ("Chemical shift (ppm)", "ppm"),
    "ms": ("Drift time (ms)", "ms"),
    "indice": ("Variable index", ""),
}

#: unidades que `cfg.axis_unit` aceita (a de `eixo_unidade` no config.yaml); "" = nao declarada.
UNIDADES_DO_EIXO = tuple(_UNIDADES_CONHECIDAS)

#: Unidades cujo eixo se escreve do MAIOR para o MENOR: numero de onda no IR/NIR (convencao de
#: OPUS, OMNIC, Spectrum...) e deslocamento quimico em RMN (ppm cresce para a esquerda). Raman
#: (deslocamento), nm, min, ms e indice vao do menor para o maior.
_UNIDADES_DECRESCENTES = frozenset({"cm-1", "ppm"})

#: fracao minima dos pontos do eixo de TREINO que o espectro novo precisa
#: cobrir (sem extrapolar) para a predicao ser aceita.
COBERTURA_EIXO_MINIMA = 0.95


class EixoForaDoModeloError(ValueError):
    """O espectro novo nao cobre o eixo usado no treino: predizer exigiria
    extrapolar, e a resposta do modelo nao teria nenhum significado."""


def cobertura_do_eixo(wn_ref: np.ndarray, wn_novo: np.ndarray) -> float:
    """Fracao dos pontos de `wn_ref` (eixo do modelo) que caem DENTRO do
    intervalo coberto por `wn_novo` -- pontos fora exigiriam extrapolacao."""
    wn_ref = np.asarray(wn_ref, dtype=float)
    wn_novo = np.asarray(wn_novo, dtype=float)
    if wn_ref.size == 0 or wn_novo.size == 0:
        return 0.0
    lo, hi = float(np.nanmin(wn_novo)), float(np.nanmax(wn_novo))
    dentro = (wn_ref >= lo) & (wn_ref <= hi)
    return float(np.mean(dentro))


def exigir_cobertura_do_eixo(wn_ref: np.ndarray, wn_novo: np.ndarray,
                             minima: float = COBERTURA_EIXO_MINIMA) -> float:
    """Levanta `EixoForaDoModeloError` se a cobertura < `minima`; devolve a
    cobertura medida."""
    cob = cobertura_do_eixo(wn_ref, wn_novo)
    if cob + 1e-12 < minima:
        wn_ref = np.asarray(wn_ref, dtype=float)
        wn_novo = np.asarray(wn_novo, dtype=float)
        raise EixoForaDoModeloError(
            f"O espectro novo cobre so' {cob * 100:.0f}% do eixo do modelo "
            f"(modelo: {wn_ref.min():.0f}-{wn_ref.max():.0f}; espectro novo: "
            f"{np.nanmin(wn_novo):.0f}-{np.nanmax(wn_novo):.0f}). Predizer "
            f"exigiria extrapolar {100 - cob * 100:.0f}% dos pontos e o "
            f"resultado nao teria significado (minimo exigido: "
            f"{minima * 100:.0f}%). Meça a amostra na mesma faixa do treino.")
    return cob


def _perfil_do_cfg(cfg):
    """Perfil declarado em `cfg.matrix_profile`, ou None se ausente/ilegivel."""
    import yaml
    from guaraci.perfil_matriz import load_profile
    nome = str(getattr(cfg, "matrix_profile", "") or "")
    if not nome or nome == _PERFIL_NEUTRO:
        return None                      # perfil neutro nao informa nada
    try:
        return load_profile(nome)
    except (ValueError, OSError, TypeError, yaml.YAMLError):  # sem informacao
        return None


def _unidade_do_perfil(perfil) -> str:
    """Unidade efetiva declarada por um perfil ("" se desconhecida)."""
    un = str(perfil.unidade_eixo)
    if un == "cm-1" and "raman" in str(perfil.rotulo_eixo or "").lower():
        return "raman_shift"
    return un if un in _UNIDADES_CONHECIDAS else ""


def unidade_declarada(cfg=None) -> str:
    """Unidade do eixo declarada em `cfg.axis_unit` (a tela [2.4] a grava); "" se ausente ou desconhecida.

    Nunca presume: valor fora de `UNIDADES_DO_EIXO` conta como nao declarado."""
    u = str(getattr(cfg, "axis_unit", "") or "").strip() if cfg is not None else ""
    return u if u in _UNIDADES_CONHECIDAS else ""


def unidade_dimensional(cfg=None) -> str:
    """Unidade efetiva do eixo para decidir orientacao e texto: a de `cfg.axis_unit` (se declarada)
    e, senao, a do perfil de `cfg`; "" se nenhuma informa.

    Um perfil de unidade "cm-1" cujo `rotulo_eixo` diz "Raman" conta como "raman_shift" (o rotulo
    do perfil e' a declaracao do usuario sobre a tecnica; Raman nao se escreve decrescente)."""
    u = unidade_declarada(cfg)
    if u:
        return u
    perfil = _perfil_do_cfg(cfg) if cfg is not None else None
    return _unidade_do_perfil(perfil) if perfil is not None else ""


def eixo_decrescente_por_unidade(cfg=None):
    """True/False se a unidade declarada (cfg ou perfil) fixa a orientacao do eixo; None se nenhuma
    unidade foi declarada (quem chama cai no comportamento antigo, por modo de entrada)."""
    u = unidade_dimensional(cfg)
    if not u:
        return None
    return u in _UNIDADES_DECRESCENTES


def rotulo_do_eixo(cfg=None) -> str:
    """Rotulo do eixo espectral para as figuras.

    1. `cfg.axis_unit` (gravada pela tela [2.4]): o rotulo da unidade ("Wavelength (nm)",
       "Raman shift (cm-1)", "Retention time (min)", "Chemical shift (ppm)", "Drift time (ms)",
       "Variable index"); se o perfil tem `rotulo_eixo` proprio e a mesma unidade, vale o do perfil;
    2. perfil com `rotulo_eixo` proprio (ex. "Raman shift (cm$^{-1}$)"): esse;
    3. unidade do perfil conhecida (cm-1, nm, ppm...): o rotulo dela;
    4. perfil neutro, ausente, ou unidade sem sentido de eixo ("indice" num perfil):
       `ROTULO_EIXO_NEUTRO` -- nunca presume numero de onda.
    """
    perfil = _perfil_do_cfg(cfg) if cfg is not None else None
    u_cfg = unidade_declarada(cfg)
    if u_cfg:
        if (perfil is not None and perfil.rotulo_eixo
                and _unidade_do_perfil(perfil) == u_cfg):
            return str(perfil.rotulo_eixo)
        return _UNIDADES_CONHECIDAS[u_cfg][0]
    if perfil is None:
        return ROTULO_EIXO_NEUTRO
    if perfil.rotulo_eixo:
        return str(perfil.rotulo_eixo)
    if str(perfil.unidade_eixo) == "indice":      # num perfil, "indice" nao diz nada sobre o eixo
        return ROTULO_EIXO_NEUTRO
    conhecido = _UNIDADES_CONHECIDAS.get(str(perfil.unidade_eixo))
    return conhecido[0] if conhecido else ROTULO_EIXO_NEUTRO


def tecnica_do_perfil(cfg=None) -> str:
    """Tecnica do resumo ("" se nenhuma informa): a escolhida na tela [2.4] (`cfg.technique`, nome
    do catalogo) e, senao, a descricao do perfil de `cfg`.

    Tira a unidade entre parenteses no fim da descricao (ex. "... (cm-1)")."""
    import re
    chave = str(getattr(cfg, "technique", "") or "").strip() if cfg is not None else ""
    if chave:
        from guaraci.catalogo_tecnicas import INSTRUMENTOS
        ins = INSTRUMENTOS.get(chave)
        if ins:
            return str(ins["nome"][0]).split(" (")[0].strip()
    perfil = _perfil_do_cfg(cfg) if cfg is not None else None
    if perfil is None or not perfil.descricao:
        return ""
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(perfil.descricao)).strip()


def chave_faixa_espectral(cfg=None, prefixo: str = "Faixa espectral") -> str:
    """Rotulo do campo de faixa no resumo: `prefixo (unidade)`, sem unidade se nada a informa."""
    u = unidade_do_eixo(cfg)
    return f"{prefixo} ({u})" if u else prefixo


def unidade_do_eixo(cfg=None) -> str:
    """Unidade (texto simples) do eixo de `cfg` (a declarada, ou a do perfil); "" se nao informada."""
    u = unidade_dimensional(cfg)
    return _UNIDADES_CONHECIDAS[u][1] if u else ""


def campos_faixa_espectral(cfg) -> dict:
    """Campos da faixa espectral para `parametros_efetivos` do resumo.

    `faixa_espectral` e `unidade_eixo` dizem a faixa e a unidade DECLARADA (`nao_declarada` se
    nenhuma). O campo antigo `faixa_espectral_cm1` fica quando a unidade e' cm-1 (ou Raman, que e'
    cm-1) e tambem quando nenhuma unidade foi declarada (resumos antigos: nada a corrigir); some
    quando a unidade declarada e' outra (nm, min, ppm, ms, indice), porque ali o nome mentiria."""
    faixa = [float(cfg.wn_min), float(cfg.wn_max)]
    u = unidade_dimensional(cfg)
    campos = {"faixa_espectral": faixa, "unidade_eixo": u or "nao_declarada"}
    if u in ("", "cm-1", "raman_shift"):
        campos["faixa_espectral_cm1"] = faixa
    return campos


#: fracao de variaveis removida pela faixa a partir da qual o corte vira aviso (V81).
LIMITE_CORTE_FAIXA = 0.20


def aviso_de_corte_de_faixa(n_orig: int, n_removidas: int, faixa, eixo_dados,
                            unidade: str = "", limite: float = LIMITE_CORTE_FAIXA) -> str:
    """Aviso quando a faixa configurada remove MAIS de `limite` das variaveis; "" caso contrario.

    A faixa padrao (4000-10000, herdada do FT-NIR em cm-1) vale para qualquer arquivo
    csv/dx; em outra tecnica ela corta uma parte grande do espectro sem que ninguem
    tenha escolhido isso. So' texto: o corte em si nao muda."""
    if n_orig <= 0 or n_removidas / n_orig <= limite:
        return ""
    u = f" {unidade}" if unidade else ""
    return (f"A faixa espectral [{faixa[0]:.0f}, {faixa[1]:.0f}]{u} removeu "
            f"{n_removidas} de {n_orig} variaveis ({100.0 * n_removidas / n_orig:.0f}%; "
            f"aviso acima de {limite * 100:.0f}%). O eixo dos dados vai de "
            f"{eixo_dados[0]:.1f} a {eixo_dados[1]:.1f}{u}. Se a faixa padrao nao e' a da sua "
            f"tecnica, ajuste faixa_min_cm/faixa_max_cm (ou wn_min/wn_max) ou use um perfil "
            f"de tecnica.")
