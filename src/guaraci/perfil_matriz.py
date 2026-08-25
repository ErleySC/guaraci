# -*- coding: utf-8 -*-
"""Perfis de matriz — o que muda quando a amostra deixa de ser oleo.

O GUARACI e' uma plataforma multimatriz, mas nasceu num caso de uso unico
(FT-NIR de oleos vegetais) e carregava esse caso de uso espalhado pelo
codigo-fonte: faixa espectral, pre-processamento padrao, unidade do eixo e
-- o mais insidioso -- o VOCABULARIO. Rodar o pipeline sobre milho em grao
produzia um model card afirmando "quantificacao de adulterante em oleo
vegetal amazonico" e um log dizendo "60 adulterados + 0 puros" (medido na
auditoria de 2026-08-17). O numero estava certo; a frase, errada.

Um perfil junta num so' lugar tudo que e' propriedade da MATRIZ, e nada
que seja propriedade do METODO. Trocar de matriz passa a ser trocar de
perfil -- nunca editar codigo-fonte.

Perfis embutidos vivem em `perfis_matriz/*.yaml`, dentro do pacote. Um
perfil de usuario e' um YAML com o mesmo formato, passado por caminho.

    from guaraci.perfil_matriz import load_profile, apply_profile
    perfil = load_profile("milho_nir")
    cfg = apply_profile(cfg, perfil)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml

if TYPE_CHECKING:                                          # pragma: no cover
    from guaraci.config import Config

log = logging.getLogger(__name__)

#: Perfis embutidos, distribuidos junto com o pacote.
DIR_PERFIS = Path(__file__).parent / "perfis_matriz"


class UnknownProfileError(ValueError):
    """Matriz sem perfil cadastrado.

    Levantado em vez de rodar com o perfil de outra matriz: um espectro de
    mel analisado com a faixa e o vocabulario de oleo produz numeros que
    parecem validos e afirmacoes quimicas que nao sao.
    """


@dataclass(frozen=True)
class Vocabulary:
    """Como esta matriz chama as coisas, na saida voltada ao usuario.

    O motor nunca usa estes termos para decidir nada -- eles so' aparecem
    em texto. Manter separado do calculo e' o que impede o vocabulario de
    uma matriz de contaminar os resultados de outra.
    """
    #: Como se chama uma unidade de classificacao ("espécie", "variedade").
    classe: str = "classe"
    classe_plural: str = "classes"
    #: A matriz por extenso, para o model card ("óleo vegetal amazônico").
    matriz: str = "a matriz analisada"
    #: O que a quantificacao mede ("teor de adulterante", "teor de proteína").
    alvo: str = "o teor do analito de interesse"
    #: Rotulos das duas pontas da autenticacao one-class.
    conforme: str = "conforme"
    nao_conforme: str = "não conforme"


@dataclass(frozen=True)
class MatrixProfile:
    """Tudo que depende da MATRIZ, e nada que dependa do METODO."""
    nome: str
    descricao: str = ""
    #: Unidade do eixo espectral: "cm-1" (numero de onda) ou "nm".
    unidade_eixo: str = "cm-1"
    eixo_min: Optional[float] = None
    eixo_max: Optional[float] = None
    default_preprocessing: Optional[str] = None
    vocabulario: Vocabulary = field(default_factory=Vocabulary)
    #: Codigo -> nome legivel da classe (o que `CODIGO_ESPECIE` era para
    #: oleos). Vazio quando a matriz nao usa codificacao no nome do arquivo.
    codigos_classe: Dict[str, str] = field(default_factory=dict)
    #: Faixa de trabalho esperada do analito, em unidade do proprio analito.
    #: Serve para avisar quando uma predicao sai fora do que foi calibrado.
    faixa_trabalho: Optional[List[float]] = None
    #: Referencia da literatura para esta matriz (nunca inventar).
    referencia: str = ""
    # ---- Campos de TECNICA DE AQUISICAO (Bloco 8a, 2026-08-25) -----------
    # So' fazem sentido p/ mode="imagem" -- None em todo perfil espectral
    # (dx/csv/sintetico). Informativos, NUNCA restritivos: o nivel real de
    # garantia de agrupamento e' decidido pelos DADOS fornecidos (estrutura
    # de pasta/CSV), nunca pelo perfil -- ver dados_imagem.py.
    #: Resolucao minima recomendada p/ esta tecnica (texto livre, ex. "1024x768").
    resolucao_esperada: Optional[str] = None
    #: Extensoes de arquivo tipicas desta tecnica. None = usa o default do
    #: modulo (.jpg/.jpeg/.png/.bmp/.tif/.tiff).
    formatos_aceitos: Optional[List[str]] = None
    #: Nivel de garantia de agrupamento que este FLUXO DE TRABALHO
    #: tipicamente sustenta na pratica ("high"/"medium"/"none") -- so'
    #: informativo (aparece em texto/ajuda), nao restringe nem substitui a
    #: deteccao automatica real feita sobre os dados.
    nivel_agrupamento_tipico: Optional[str] = None

    def fora_da_faixa_de_trabalho(self, valor: float) -> bool:
        """True se `valor` cai fora da faixa calibrada declarada no perfil.

        Sem faixa declarada, devolve False -- ausencia de declaracao nao e'
        licenca para afirmar que esta dentro, mas tambem nao inventa um
        limite que ninguem mediu; quem consome deve tratar `faixa_trabalho
        is None` como "nao declarado".
        """
        if not self.faixa_trabalho or len(self.faixa_trabalho) != 2:
            return False
        lo, hi = self.faixa_trabalho
        return not (lo <= float(valor) <= hi)


def _perfis_disponiveis() -> List[str]:
    if not DIR_PERFIS.is_dir():
        return []
    return sorted(p.stem for p in DIR_PERFIS.glob("*.yaml"))


def load_profile(nome_ou_caminho: str) -> MatrixProfile:
    """Carrega um perfil embutido pelo nome, ou um YAML de usuario pelo caminho.

    Matriz sem perfil cadastrado levanta `UnknownProfileError` com a
    lista do que existe -- nunca cai num perfil padrao em silencio.
    """
    alvo = Path(nome_ou_caminho)
    if alvo.suffix in (".yaml", ".yml") and alvo.is_file():
        caminho = alvo
    else:
        caminho = DIR_PERFIS / f"{nome_ou_caminho}.yaml"
        if not caminho.is_file():
            raise UnknownProfileError(
                f"Nenhum perfil de matriz chamado '{nome_ou_caminho}'. "
                f"Perfis disponiveis: {', '.join(_perfis_disponiveis()) or '(nenhum)'}. "
                f"Para uma matriz nova, escreva um YAML com o mesmo formato "
                f"(veja {DIR_PERFIS}/generico.yaml) e passe o caminho do "
                f"arquivo. Rodar com o perfil de outra matriz produziria "
                f"faixa espectral e vocabulario errados.")

    with open(caminho, encoding="utf-8") as f:
        bruto: Dict[str, Any] = yaml.safe_load(f) or {}

    voc = Vocabulary(**(bruto.pop("vocabulario", None) or {}))
    bruto.pop("nome", None)
    return MatrixProfile(nome=caminho.stem, vocabulario=voc, **bruto)


def apply_profile(cfg: "Config", perfil: MatrixProfile) -> "Config":
    """Escreve no `cfg` o que o perfil define, sem tocar no que ja' foi
    escolhido explicitamente pelo usuario.

    Regra: o perfil e' um PADRAO da matriz, nao uma imposicao. Se o usuario
    definiu `wn_min` na configuracao, o valor dele vence -- o perfil so'
    preenche o que esta no default. Isso mantem o perfil util sem tirar o
    controle de quem sabe o que esta fazendo.
    """
    from guaraci.config import Config as _Config

    padrao = _Config()
    if perfil.eixo_min is not None and cfg.wn_min == padrao.wn_min:
        cfg.wn_min = float(perfil.eixo_min)
    if perfil.eixo_max is not None and cfg.wn_max == padrao.wn_max:
        cfg.wn_max = float(perfil.eixo_max)
    if (perfil.default_preprocessing
            and cfg.default_preprocessing == padrao.default_preprocessing):
        cfg.default_preprocessing = perfil.default_preprocessing
    log.info("[INFO] Perfil de matriz: %s (%s) | eixo [%.4g, %.4g] %s | "
             "pre-proc %s", perfil.nome, perfil.descricao or "-",
             cfg.wn_min, cfg.wn_max, perfil.unidade_eixo,
             cfg.default_preprocessing)
    return cfg


def cfg_profile(cfg: "Config") -> MatrixProfile:
    """Perfil declarado em `cfg.matrix_profile`. Erro claro se nao existir."""
    return load_profile(getattr(cfg, "matrix_profile", "generico"))
