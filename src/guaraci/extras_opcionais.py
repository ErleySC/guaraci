# -*- coding: utf-8 -*-
"""extras_opcionais.py -- pacotes opcionais que alguns formatos e recursos exigem.

A instalacao base (`pip install .`) NAO traz os leitores de OPUS, SPC, RMN
Bruker e HPLC/GC de fabricante, nem PARAFAC (tensorly), textura de imagem (scikit-image) e o
diagnostico PCV do DD-SIMCA (prcv). Sem este modulo o usuario so' descobria a falta ao rodar
("Pacote opcional ... nao instalado", depois "Nenhuma amostra valida"). Aqui fica o registro
unico do que e' opcional, de quais formatos/tecnicas/recursos cada pacote habilita e do comando
de instalacao; a tela [2.4], `guaraci doctor`, `guaraci checar` e o pre-voo leem daqui.

A deteccao usa `importlib.util.find_spec` (nao importa o pacote: barato e sem efeito colateral).
ASCII por convencao do projeto.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["ExtraOpcional", "EXTRAS", "instalado", "ausentes", "extra_do_modo", "comando_pip",
           "texto_instalar", "sufixo_menu", "aviso_de_escolha", "extras_exigidos_por",
           "extras_de_recurso_da_tecnica", "aviso_de_recurso", "COMANDO_TUDO",
           "EXTRAS_DE_CONFORTO", "plano_instalacao", "requisitos_do_extra"]

#: Comando que instala TUDO de uma vez (documentado no README e no manual).
COMANDO_TUDO = 'pip install ".[all]"'


@dataclass(frozen=True)
class ExtraOpcional:
    #: nome do MODULO a importar (nao o do pip: `rainbow-api` instala o modulo `rainbow`)
    modulo: str
    #: nome do pacote no pip
    pacote: str
    #: nome do extra em pyproject.toml
    extra: str
    #: valores de `modo_entrada` que dependem do pacote para LER os dados (falta = nao roda)
    modos: Tuple[str, ...]
    #: chaves de tecnica ([2.4]) afetadas
    tecnicas: Tuple[str, ...]
    #: o que o pacote habilita (PT, EN)
    afeta: Tuple[str, str]
    #: atributo booleano da Config que liga um RECURSO que depende do pacote ("" = nenhum)
    flag_cfg: str = ""
    #: versao minima do Python em que o pacote deixa de existir (None = sempre disponivel)
    python_ate: Optional[Tuple[int, int]] = None


EXTRAS: Tuple[ExtraOpcional, ...] = (
    ExtraOpcional("brukeropus", "brukeropus", "opus", ("opus",), ("ft-nir", "mir"),
                  ("arquivos Bruker OPUS (FT-NIR, MIR)", "Bruker OPUS files (FT-NIR, MIR)")),
    ExtraOpcional("spcfile", "spcfile", "spc", ("spc",), ("ft-nir", "nir", "mir", "raman", "uv-vis", "fluorescencia"),
                  ("arquivos SPC Thermo/Galactic (NIR, MIR, Raman, UV-Vis, fluorescencia)",
                   "Thermo/Galactic SPC files (NIR, MIR, Raman, UV-Vis, fluorescence)")),
    ExtraOpcional("nmrglue", "nmrglue", "rmn", ("rmn",), ("nmr",),
                  ("pasta Bruker de RMN ja processada", "processed Bruker NMR folder")),
    ExtraOpcional("rainbow", "rainbow-api", "hplc", ("hplc",), ("hplc",),
                  ("pasta Agilent/Waters de HPLC/GC", "Agilent/Waters HPLC/GC folder"), python_ate=(3, 14)),
    ExtraOpcional("tensorly", "tensorly", "multiway", (), ("eem", "hsi"),
                  ("PARAFAC da fluorescencia 3D (EEM) e da HSI multiway", "PARAFAC for 3D fluorescence (EEM) and multiway HSI")),
    ExtraOpcional("skimage", "scikit-image", "imagem", (), ("imagem",),
                  ("textura da imagem (GLCM)", "image texture (GLCM)"), flag_cfg="include_image_texture"),
    ExtraOpcional("prcv", "prcv", "robusto", (), (),
                  ("diagnostico PCV do DD-SIMCA", "DD-SIMCA PCV diagnostic"), flag_cfg="ddsimca_pcv"),
)


def instalado(e: ExtraOpcional) -> bool:
    """True se o modulo do extra pode ser importado (find_spec, sem importar)."""
    try:
        return importlib.util.find_spec(e.modulo) is not None
    except (ImportError, ValueError):     # pacote quebrado ou nome invalido: conta como ausente
        return False


def ausentes() -> List[ExtraOpcional]:
    """Extras de `EXTRAS` que NAO estao instalados, na ordem do registro."""
    return [e for e in EXTRAS if not instalado(e)]


def extra_do_modo(modo: str) -> Optional[ExtraOpcional]:
    """Extra que o formato `modo` (valor de `modo_entrada`) exige para ler os dados, ou None."""
    for e in EXTRAS:
        if modo in e.modos:
            return e
    return None


def extras_exigidos_por(cfg: Any) -> List[Tuple[ExtraOpcional, str]]:
    """Extras AUSENTES de que `cfg` depende: (extra, "formato" | "recurso").

    "formato": o `modo_entrada` precisa do pacote para ler (a execucao falha). "recurso": uma opcao
    ligada na Config precisa dele (a execucao segue sem o recurso, com aviso)."""
    modo = str(getattr(cfg, "mode", "") or "")
    achados: List[Tuple[ExtraOpcional, str]] = []
    for e in EXTRAS:
        if instalado(e):
            continue
        if modo in e.modos:
            achados.append((e, "formato"))
        elif e.flag_cfg and bool(getattr(cfg, e.flag_cfg, False)):
            achados.append((e, "recurso"))
    return achados


def comando_pip(e: ExtraOpcional) -> str:
    """Comando que instala o extra (com as faixas de versao do projeto)."""
    return f'pip install ".[{e.extra}]"'


def _indisponivel_nesta_versao(e: ExtraOpcional) -> bool:
    return e.python_ate is not None and sys.version_info[:2] >= e.python_ate


def texto_instalar(e: ExtraOpcional, lang: str = "PT") -> str:
    """Como instalar, em uma frase; avisa quando o pacote nao existe para este Python."""
    en = str(lang).upper() == "EN"
    if _indisponivel_nesta_versao(e):
        v = f"{e.python_ate[0]}.{e.python_ate[1]}" if e.python_ate else ""
        return (f"{e.pacote} has no release for Python {v}+; use Python {e.python_ate[0]}.{e.python_ate[1] - 1} "
                f"or export the data to CSV" if en else
                f"{e.pacote} nao tem versao para Python {v}+; use Python {e.python_ate[0]}.{e.python_ate[1] - 1} "
                f"ou exporte os dados para CSV")
    return comando_pip(e)


def sufixo_menu(modo: str, lang: str = "PT") -> str:
    """Texto "(extra: pip install ...)" para o formato `modo` na tela [2.4]; "" se nao precisa de extra
    ou se o pacote ja' esta instalado."""
    e = extra_do_modo(modo)
    if e is None or instalado(e):
        return ""
    return f"(extra: {texto_instalar(e, lang)})"


def aviso_de_escolha(modo: str, lang: str = "PT") -> str:
    """Aviso mostrado NA HORA em que o usuario escolhe o formato sem o pacote ("" se nada falta)."""
    e = extra_do_modo(modo)
    if e is None or instalado(e):
        return ""
    if str(lang).upper() == "EN":
        return (f"This format needs the optional package '{e.pacote}', which is not installed: reading "
                f"{e.afeta[1]} will fail. Install it with {texto_instalar(e, 'EN')} "
                f"(or everything at once: {COMANDO_TUDO}), or export the data to CSV.")
    return (f"Este formato exige o pacote opcional '{e.pacote}', que nao esta instalado: a leitura de "
            f"{e.afeta[0]} vai falhar. Instale com {texto_instalar(e, 'PT')} "
            f"(ou tudo de uma vez: {COMANDO_TUDO}), ou exporte os dados para CSV.")


def extras_de_recurso_da_tecnica(tecnica: str) -> List[ExtraOpcional]:
    """Extras AUSENTES que habilitam um RECURSO (nao um formato) da tecnica `tecnica` da tela [2.4]
    (ex.: PARAFAC na fluorescencia 3D, textura na imagem)."""
    return [e for e in EXTRAS if not e.modos and tecnica in e.tecnicas and not instalado(e)]


def aviso_de_recurso(e: ExtraOpcional, lang: str = "PT") -> str:
    """Uma linha dizendo o que o extra ausente deixa de habilitar."""
    if str(lang).upper() == "EN":
        return f"Optional feature unavailable: {e.afeta[1]} (extra: {texto_instalar(e, 'EN')})."
    return f"Recurso opcional indisponivel: {e.afeta[0]} (extra: {texto_instalar(e, 'PT')})."


#: Extras de conforto que quase todo usuario quer (navegador e relatorios): modulos a importar de cada um.
#: Os de formato/recurso vem de `EXTRAS`; `benchmark` (xgboost, shap) fica de fora do plano padrao: e' grande e so'
#: serve a comparacao com outros algoritmos.
EXTRAS_DE_CONFORTO: Dict[str, Tuple[str, ...]] = {
    "web": ("streamlit", "psutil"),
    "reports": ("fpdf", "docx", "openpyxl", "pptx"),
}

_NOME_DISTRIBUICAO = "guaraci-chemometrics"


def requisitos_do_extra(extra: str) -> List[str]:
    """Requisitos pip (com as faixas de versao do projeto) do extra `extra`, lidos dos metadados do pacote instalado,
    ja' filtrados pelo marcador do ambiente (ex.: `rainbow-api` so' para Python < 3.14). Vazio se o GUARACI nao esta
    instalado como pacote (rodando direto da pasta de fontes): ai' o pyproject.toml da pasta e' a fonte."""
    from importlib import metadata
    try:
        brutos = metadata.requires(_NOME_DISTRIBUICAO) or []
    except metadata.PackageNotFoundError:
        return []
    saida: List[str] = []
    for bruto in brutos:
        if _NOME_DISTRIBUICAO in bruto.split(";")[0].lower():          # o extra `all` aponta para os outros
            continue
        corpo, _, marcador = bruto.partition(";")
        if not re.search(r"""extra\s*==\s*['"]""" + re.escape(extra) + r"""['"]""", marcador):
            continue
        try:
            from packaging.requirements import Requirement
            req = Requirement(bruto)
            if req.marker is not None and not req.marker.evaluate({"extra": extra}):
                continue
            saida.append(str(req).split(";")[0].strip())
        except ImportError:                                            # sem `packaging`: ignora o marcador de Python
            saida.append(corpo.strip())
    return saida


def plano_instalacao(incluir_benchmark: bool = False) -> Dict[str, List[str]]:
    """O que falta instalar para o GUARACI funcionar por inteiro neste Python: {extra: [requisitos pip]}.

    Entram `web` e `reports` (se algum modulo falta), todo extra de formato/recurso ausente e disponivel para este Python
    (`rainbow-api` nao existe para o 3.14+) e, so' se pedido, `benchmark`. Extra sem requisito legivel nos metadados
    (GUARACI rodando da pasta de fontes) fica de fora: o chamador cai no comando `pip install ".[extra]"`."""
    plano: Dict[str, List[str]] = {}
    extras: List[str] = []
    for nome, modulos in EXTRAS_DE_CONFORTO.items():
        if any(importlib.util.find_spec(m) is None for m in modulos):
            extras.append(nome)
    extras += [e.extra for e in ausentes() if not _indisponivel_nesta_versao(e)]
    if incluir_benchmark and any(importlib.util.find_spec(m) is None for m in ("xgboost", "shap")):
        extras.append("benchmark")
    for nome in extras:
        reqs = requisitos_do_extra(nome)
        plano[nome] = reqs
    return plano
