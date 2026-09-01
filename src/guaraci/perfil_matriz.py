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

__all__ = [
    "DIR_PERFIS",
    "PERFIS_TECNICA",
    "UnknownProfileError",
    "Vocabulary",
    "MatrixProfile",
    "load_profile",
    "apply_profile",
    "cfg_profile",
    "perfis_disponiveis",
    "combine_profiles",
    "save_profile",
]

#: Perfis embutidos, distribuidos junto com o pacote.
DIR_PERFIS = Path(__file__).parent / "perfis_matriz"

#: Nomes de perfil embutido cujo foco principal e' TECNICA DE AQUISICAO de
#: imagem (campos resolucao_esperada/formatos_aceitos/nivel_agrupamento_
#: tipico preenchidos), distintos dos perfis de MATRIZ quimica (foco em
#: eixo espectral/vocabulario). "generico" nao entra aqui -- serve aos dois
#: papeis (ver `perfis_disponiveis`). Convencao por nome de arquivo: os 3
#: perfis de tecnica hoje existentes (Bloco 8a, 2026-08-25) sao estes.
PERFIS_TECNICA = frozenset({"bancada", "celular", "scanner"})


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

    def outside_working_range(self, valor: float) -> bool:
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


def perfis_disponiveis(*, apenas: Optional[str] = None) -> List[str]:
    """Nomes de perfil embutido, opcionalmente filtrados por dimensao.

    `apenas="tecnica"`: so' os de tecnica de aquisicao de imagem (+
    "generico", que serve de fallback para as duas dimensoes).
    `apenas="matriz"`: todos os que NAO sao de tecnica (inclui "generico").
    `apenas=None` (default): lista completa, sem filtro.
    """
    todos = _perfis_disponiveis()
    if apenas == "tecnica":
        return [p for p in todos if p in PERFIS_TECNICA or p == "generico"]
    if apenas == "matriz":
        return [p for p in todos if p not in PERFIS_TECNICA]
    return todos


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


def combine_profiles(nome: str, matriz: MatrixProfile,
                      tecnica: Optional[MatrixProfile]) -> MatrixProfile:
    """Funde um perfil de MATRIZ (o que e' a amostra) com um perfil de
    TECNICA de aquisicao (como ela foi capturada) num perfil novo, pronto
    pra salvar com `save_profile` e reusar (Agente 5B, "criar/salvar
    perfil combinado").

    Regra de precedencia -- cada campo vem de UMA fonte, nunca misturado
    campo-a-campo dentro do mesmo conceito:
    - vocabulario/codigos_classe/faixa_trabalho/eixo/unidade_eixo/
      referencia: sempre da MATRIZ (sao propriedade quimica da amostra,
      tecnica de captura nao muda o que ela e').
    - resolucao_esperada/formatos_aceitos/nivel_agrupamento_tipico: sempre
      da TECNICA (matriz nao declara esses campos -- ver
      `MatrixProfile.__doc__`).
    - default_preprocessing: da TECNICA quando ela declarar um (o
      pre-processamento de colorimetria digital -- ex. autoscaling -- e'
      sobre COMO o dado foi extraido, nao sobre qual e' a matriz); cai pro
      da MATRIZ quando a tecnica nao declarar nada.
    - descricao: concatena as duas, pra' o perfil combinado deixar claro
      as duas dimensoes de uma vez.

    `tecnica=None` funde so' com os defaults de tecnica (equivalente a nao
    combinar nada -- existe pra' nao forcar o chamador a checar None antes).
    """
    tec = tecnica if tecnica is not None else MatrixProfile(nome="")
    descricao = matriz.descricao
    if tecnica is not None and tecnica.descricao:
        descricao = f"{matriz.descricao} — {tecnica.descricao}" if descricao else tecnica.descricao
    return MatrixProfile(
        nome=nome,
        descricao=descricao,
        unidade_eixo=matriz.unidade_eixo,
        eixo_min=matriz.eixo_min,
        eixo_max=matriz.eixo_max,
        default_preprocessing=tec.default_preprocessing or matriz.default_preprocessing,
        vocabulario=matriz.vocabulario,
        codigos_classe=dict(matriz.codigos_classe),
        faixa_trabalho=list(matriz.faixa_trabalho) if matriz.faixa_trabalho else None,
        referencia=matriz.referencia,
        resolucao_esperada=tec.resolucao_esperada,
        formatos_aceitos=list(tec.formatos_aceitos) if tec.formatos_aceitos else None,
        nivel_agrupamento_tipico=tec.nivel_agrupamento_tipico,
    )


def save_profile(perfil: MatrixProfile, caminho: str) -> None:
    """Grava `perfil` como YAML no mesmo formato que `load_profile` le' --
    inverso de `load_profile`. `caminho` e' um arquivo completo (nao um
    diretorio); o chamador decide onde (perfil de usuario, fora de
    `DIR_PERFIS` -- esse diretorio vive DENTRO do pacote instalado e pode
    nao ser gravavel, ver `test_perfis_sao_empacotados_com_o_pacote`).
    """
    dados: Dict[str, Any] = {
        "descricao": perfil.descricao,
        "unidade_eixo": perfil.unidade_eixo,
        "eixo_min": perfil.eixo_min,
        "eixo_max": perfil.eixo_max,
        "default_preprocessing": perfil.default_preprocessing,
        "vocabulario": {
            "classe": perfil.vocabulario.classe,
            "classe_plural": perfil.vocabulario.classe_plural,
            "matriz": perfil.vocabulario.matriz,
            "alvo": perfil.vocabulario.alvo,
            "conforme": perfil.vocabulario.conforme,
            "nao_conforme": perfil.vocabulario.nao_conforme,
        },
        "codigos_classe": dict(perfil.codigos_classe),
        "faixa_trabalho": list(perfil.faixa_trabalho) if perfil.faixa_trabalho else None,
        "referencia": perfil.referencia,
        "resolucao_esperada": perfil.resolucao_esperada,
        "formatos_aceitos": list(perfil.formatos_aceitos) if perfil.formatos_aceitos else None,
        "nivel_agrupamento_tipico": perfil.nivel_agrupamento_tipico,
    }
    caminho_p = Path(caminho)
    caminho_p.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_p, "w", encoding="utf-8") as f:
        yaml.safe_dump(dados, f, allow_unicode=True, sort_keys=False)
