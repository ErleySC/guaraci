# -*- coding: utf-8 -*-
"""aumento_dados.py -- Aumento de dados espectral (data augmentation)
inspirado em Vicinal Risk Minimization (VRM), OPCIONAL e desligado por
padrao.

FONTE E LIMITACAO DE FIDELIDADE (leia antes de usar ou modificar)
------------------------------------------------------------------
Inspirado em:

    Tumoine, I., Metz, M., Abdelghafour, F. Y., Esteve, D., Grotus, N.,
    Bendoula, R. & Roger, J.-M. (2026). "A new framework for spectral data
    augmentation based on vicinal risk minimization: Application to the
    prediction of mango dry matter content." Chemometrics and Intelligent
    Laboratory Systems. DOI 10.1016/j.chemolab.2026.105769.

O texto completo do artigo (ScienceDirect, DOI resolver, o preprint SSRN
"A Vicinal View about Spectral Data Augmentation", abstract_id=6389102, e
o arXiv) esta' bloqueado pela politica de saida de rede desta sessao
(egress proxy nega api.crossref.org, doi.org, sciencedirect.com,
arxiv.org e papers.ssrn.com -- confirmado com erro explicito de politica,
nao timeout transitorio). O DOI e os autores/periodico foram confirmados
por WebSearch (que roda hospedado, fora do proxy bloqueado); NAO foi
possivel ler o mecanismo exato do artigo.

O que a busca (WebSearch, 2026-09-20) permitiu confirmar com razoavel
confianca, via resumos/citacoes de terceiros (nao o texto completo):
  - O framework "explora metodos classicos de correcao quimiometrica para
    quantificar e analisar distorcoes espectrais, depois gera amostras
    'vicinais' amostrando dentro dessas distorcoes, garantindo
    correspondencia com os valores de referencia originais" (ou seja: o
    ROTULO/alvo e' preservado por construcao -- e' a premissa central de
    VRM, nao um detalhe deste artigo especifico).
  - A implementacao testada no artigo usa "transformacoes ADITIVAS" e
    mede ganho de generalizacao num MODELO CONVOLUCIONAL (CNN) para teor
    de materia seca de manga (Vis-NIR).
  - Na literatura mais ampla de aumento de dados para NIR (nao este artigo
    especifico, mas o vocabulario padrao da area, ex. Bjerrum et al. 2017,
    "Data Augmentation of Spectral Data for CNN-based Deep Chemometrics"):
    as transformacoes tipicas sao OFFSET (deslocamento aditivo de linha de
    base), MULTIPLICATION (escala multiplicativa, path-length) e SLOPE
    (inclinacao linear aditiva, espalhamento/temperatura) -- exatamente a
    classe de distorcao que MSC/EMSC (`preprocessamento.py`) foram feitos
    para REMOVER. Isso da' fundamentacao fisica solida para a escolha de
    parametrizacao abaixo, mesmo sem o texto exato do artigo-alvo.

O QUE ESTE MODULO **NAO** E': nao e' uma replicacao literal do algoritmo
do artigo (que provavelmente estima a distribuicao das distorcoes
diretamente dos DADOS, via ajuste de MSC/EMSC por amostra, e amostra
dentro dela -- mas isso e' inferencia razoavel, nao confirmada). Este
modulo implementa uma versao GENERICA e PARAMETRICA da mesma FAMILIA de
perturbacao (aditiva: offset + inclinacao de linha de base; multiplicativa:
ganho; e ruido ESTRUTURADO/correlacionado espectralmente, nao ruido
i.i.d. generico -- reaproveitando `SavGol` de `preprocessamento.py` para
suavizar o ruido branco gerado, simulando ruido de detector correlacionado
entre canais vizinhos em vez de puramente independente). E' "inspirado
no framework geral referenciado pelo DOI", nao "a implementacao do
Tumoine et al. (2026)".

CONDICAO DE SEGURANCA NAO-NEGOCIAVEL (group-aware)
---------------------------------------------------
Toda amostra aumentada herda EXATAMENTE o `group_id`/`mae_id` da amostra
original da qual foi derivada -- nunca cria grupo novo. Isso e' o que
garante que um split group-aware (`StableStratifiedGroupKFold`) nunca
separa uma amostra original de suas variantes aumentadas entre treino e
validacao (se isso acontecesse, a "validacao" veria uma variante quase
identica a uma amostra de treino, inflando a metrica de forma artificial
-- o mesmo tipo de vazamento que agrupamento por `mae_id` ja existe para
prevenir em replicas de laboratorio). Ver
`tests/test_aumento_dados_hypothesis.py` para a prova por teste de
propriedade (com contra-prova deliberada).

USO -- SEMPRE OPCIONAL, NUNCA DEFAULT
--------------------------------------
`AumentoVRM` so' deve ser aplicado ao FOLD DE TREINO de um split
group-aware ja calculado, nunca ao dado completo antes do split (isso
vazaria treino->validacao mesmo com heranca de grupo correta, se o split
for calculado DEPOIS do aumento). Ver
`scripts/medicoes/portao_vrm_tecator.py` para o padrao correto de uso
dentro do portao de aceite (Bloco 20).

VEREDITO DO PORTAO DE ACEITE (Bloco 20, medido em 2026-09-20, ver
`scripts/medicoes/portao_vrm_tecator.py` -- `avaliar_correcao_sinal`
generico, PLS-R com `n_componentes=10`, `StableStratifiedGroupKFold`
`n_splits=5`, dataset REAL Tecator publico via sktime, 215 amostras --
ver ressalva sobre nao ser o cenario do acervo privado no docstring do
script): **APROVADO, mas com efeito PEQUENO** -- ganho e' consistente
e estatisticamente real, NAO um resultado forte:

  - Config default (`n_aumentos=2`, `amplitude_multiplicativa=0.01`,
    `amplitude_baseline=0.01`, ruido estruturado OFF, `n_seeds=20`):
    RMSEP 2,845 -> 2,839 (**~0,2% de reducao relativa**), p=0,001,
    tamanho de efeito padronizado (~Cohen's d pareado) 0,865 -- a
    consistencia de DIRECAO entre seeds e' alta (a maioria das 20
    particoes melhora, so' uma minoria piora), mas a MAGNITUDE da
    melhora por particao e' pequena.
  - Amplitude maior (`amplitude_multiplicativa=amplitude_baseline=0.03`):
    efeito sobe para ~1,1% (RMSEP 2,848 -> 2,818, p=0,020, n=10) --
    sugere que o ganho escala com a amplitude da perturbacao dentro da
    faixa testada, mas nao foi variado alem disso (risco de, em algum
    ponto, a perturbacao deixar de ser "vicinal" e comecar a distorcer
    demais o dado -- nao testado).
  - `n_aumentos` (1 a 4): efeito cresce levemente com mais copias por
    amostra (0,27% -> 0,37%), mas o ganho marginal por copia extra e'
    pequeno.
  - **Ruido estruturado LIGADO** (`amplitude_ruido_estruturado=0.01`,
    resto default): **REJEITADO** -- RMSEP PIOROU (2,848 -> 2,917,
    ~-2,4%, p=0,037, n=10). Por isso o default de
    `amplitude_ruido_estruturado` e' `0.0` (desligado) -- a perturbacao
    de ruido correlacionado, do jeito que foi parametrizada aqui, NAO
    ajuda neste dataset/config; so' offset+inclinacao (baseline) e ganho
    (multiplicativa) mostraram ganho real.

CONCLUSAO HONESTA: o mecanismo de baseline+multiplicativo (a classe
"aditiva" citada na busca sobre o artigo-fonte) tem suporte empirico real
mas MODESTO no Tecator -- nao e' um resultado do porte do EMSC (6-7% de
reducao de RMSEP, ver `preprocessamento.EMSC`). Fica disponivel como
OPCIONAL/experimental (nunca default), com o ruido estruturado desligado
por padrao ate' que uma parametrizacao melhor seja encontrada e testada
pelo mesmo portao. Nao substitui validacao contra o acervo privado real
(n pequeno por classe), unico cenario que motivou a proposta original --
essa validacao so' pode ser rodada localmente pelo usuario.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from guaraci.preprocessamento import SavGol

__all__ = ["AumentoVRM", "aumentar_vrm"]


def _eixo_normalizado(n_canais: int) -> np.ndarray:
    """Indice de canal normalizado (media 0, desvio 1) -- mesma convencao
    de normalizacao usada em `EMSC.fit` (nao importado dali para nao
    acoplar a uma classe de CORRECAO a uma funcao de GERACAO; a formula e'
    trivial o bastante para nao configurar duplicacao substantiva)."""
    eixo = np.arange(n_canais, dtype=float)
    desvio = eixo.std()
    return (eixo - eixo.mean()) / (desvio if desvio > 1e-300 else 1.0)


@dataclass
class AumentoVRM:
    """Aumento de dados espectral inspirado em VRM (ver docstring do
    modulo para a ressalva de fidelidade ao artigo original).

    Para cada amostra original, gera `n_aumentos` variantes combinando:

      1. Ganho MULTIPLICATIVO por amostra (~ path-length/instrumento):
         `X * (1 + eps_mult)`, `eps_mult ~ N(0, amplitude_multiplicativa)`.
      2. Linha de base ADITIVA polinomial (offset + inclinacao + curvatura,
         ate' ordem `ordem_polinomial_baseline`, no eixo de indice de canal
         normalizado -- mesma convencao de `EMSC`): coeficientes
         `~ N(0, amplitude_baseline * escala_i)`, onde `escala_i` e' o
         desvio-padrao do proprio espectro original (mantem a perturbacao
         proporcional a escala do dado, funciona tanto em reflectancia
         0-1 quanto em absorbancia).
      3. Ruido ESTRUTURADO (correlacionado entre canais vizinhos, nao
         i.i.d.): ruido branco suavizado por `SavGol` (janela
         `janela_suavizacao_ruido`, `polyorder=1`, `deriv=0`), escalado
         por `amplitude_ruido_estruturado * escala_i`. Amplitude default
         0.0 (desligado) -- as duas primeiras transformacoes ja cobrem a
         classe "aditiva" citada na busca sobre o artigo-fonte; o ruido
         estruturado e' oferecido como opcao extra, nao like parte
         obrigatoria do mecanismo.

    O ROTULO (`y`) e' preservado por construcao (premissa central de VRM:
    a perturbacao fica "na vizinhanca" do ponto original, sem mudar o
    valor de referencia) -- cada variante recebe o MESMO `y` da amostra de
    origem.

    O GRUPO (`group_id`/`mae_id`) e' herdado EXATAMENTE -- condicao de
    seguranca nao-negociavel, ver docstring do modulo.

    Nunca aplicado por default em nenhum pipeline do Guaraci -- so' quando
    explicitamente chamado, e so' dentro do fold de treino de um split
    ja calculado (ver `scripts/medicoes/portao_vrm_tecator.py`)."""

    n_aumentos: int = 1
    amplitude_multiplicativa: float = 0.01
    amplitude_baseline: float = 0.01
    amplitude_ruido_estruturado: float = 0.0
    ordem_polinomial_baseline: int = 2
    janela_suavizacao_ruido: int = 11
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_aumentos < 1:
            raise ValueError(f"n_aumentos deve ser >= 1, recebido {self.n_aumentos}")
        if self.amplitude_multiplicativa < 0 or self.amplitude_baseline < 0 \
                or self.amplitude_ruido_estruturado < 0:
            raise ValueError("amplitudes devem ser >= 0")
        if self.ordem_polinomial_baseline < 0:
            raise ValueError("ordem_polinomial_baseline deve ser >= 0")

    def gerar(
            self, X: np.ndarray, y: np.ndarray, grupos: np.ndarray,
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gera SO' as amostras NOVAS (nao inclui as originais) -- use
        `aumentar` para o array concatenado (original + novo), pronto para
        `fit`. Retorna `(X_novo, y_novo, grupos_novo)`, cada um com
        `len(X) * n_aumentos` linhas, na ordem
        [amostra_0_var_0, amostra_0_var_1, ..., amostra_1_var_0, ...] --
        cada bloco de `n_aumentos` linhas herda o `y`/grupo da amostra
        original correspondente."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        grupos = np.asarray(grupos)
        if len(X) != len(y) or len(X) != len(grupos):
            raise ValueError(
                f"X ({len(X)}), y ({len(y)}) e grupos ({len(grupos)}) "
                "devem ter o mesmo numero de linhas")
        if X.ndim != 2:
            raise ValueError(f"X deve ser 2D (n_amostras, n_canais), recebido shape {X.shape}")

        n, p = X.shape
        k = self.n_aumentos
        rng = np.random.default_rng(self.seed)

        # Repete cada amostra original k vezes ANTES de perturbar --
        # `np.repeat` (nao `np.tile`) da a ordem
        # [amostra_0]*k + [amostra_1]*k + ... exigida pelo contrato acima.
        X_rep = np.repeat(X, k, axis=0)                    # (n*k, p)
        y_novo = np.repeat(y, k, axis=0)
        grupos_novo = np.repeat(grupos, k, axis=0)          # HERANCA DE GRUPO
        escala = np.repeat(X.std(axis=1), k, axis=0)        # (n*k,) escala por amostra original
        escala_segura = np.where(escala > 1e-300, escala, 1.0)

        # 1. Ganho multiplicativo por variante.
        eps_mult = rng.normal(0.0, self.amplitude_multiplicativa, size=n * k)
        X_aug = X_rep * (1.0 + eps_mult)[:, None]

        # 2. Linha de base polinomial aditiva por variante.
        eixo_norm = _eixo_normalizado(p)
        n_termos = self.ordem_polinomial_baseline + 1
        coefs = rng.normal(0.0, self.amplitude_baseline, size=(n * k, n_termos)) \
            * escala_segura[:, None]
        base = np.column_stack([eixo_norm ** grau for grau in range(n_termos)])  # (p, n_termos)
        X_aug = X_aug + coefs @ base.T

        # 3. Ruido estruturado (opcional, default desligado).
        if self.amplitude_ruido_estruturado > 0:
            janela = min(self.janela_suavizacao_ruido, p if p % 2 == 1 else p - 1)
            janela = max(janela, 3)
            if janela % 2 == 0:
                janela -= 1
            ruido_branco = rng.normal(0.0, 1.0, size=(n * k, p))
            if janela >= 3 and p >= janela:
                ruido = SavGol(window_length=janela, polyorder=1, deriv=0).fit_transform(ruido_branco)
            else:
                # Poucos canais para suavizar de forma significativa --
                # cai pro ruido branco (ainda assim aditivo/vicinal, so'
                # sem a correlacao espectral desejavel).
                ruido = ruido_branco
            desvio_ruido = ruido.std(axis=1, keepdims=True)
            desvio_ruido_seguro = np.where(desvio_ruido > 1e-300, desvio_ruido, 1.0)
            ruido_normalizado = ruido / desvio_ruido_seguro
            X_aug = X_aug + ruido_normalizado * (self.amplitude_ruido_estruturado
                                                  * escala_segura[:, None])

        return X_aug, y_novo, grupos_novo

    def aumentar(
            self, X: np.ndarray, y: np.ndarray, grupos: np.ndarray,
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Conveniencia: devolve o dado ORIGINAL concatenado com as
        variantes geradas por `gerar` -- pronto para `fit` direto."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        grupos = np.asarray(grupos)
        X_novo, y_novo, grupos_novo = self.gerar(X, y, grupos)
        return (np.vstack([X, X_novo]),
                np.concatenate([y, y_novo]),
                np.concatenate([grupos, grupos_novo]))


def aumentar_vrm(
        X: np.ndarray, y: np.ndarray, grupos: np.ndarray, *,
        n_aumentos: int = 1,
        amplitude_multiplicativa: float = 0.01,
        amplitude_baseline: float = 0.01,
        amplitude_ruido_estruturado: float = 0.0,
        ordem_polinomial_baseline: int = 2,
        janela_suavizacao_ruido: int = 11,
        seed: int = 0,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Atalho funcional para `AumentoVRM(...).aumentar(X, y, grupos)` --
    devolve `(X, y, grupos)` JA' concatenados (original + variantes),
    pronto para `fit`. Ver `AumentoVRM` para os parametros e a ressalva de
    fidelidade ao artigo original no docstring do modulo."""
    return AumentoVRM(
        n_aumentos=n_aumentos,
        amplitude_multiplicativa=amplitude_multiplicativa,
        amplitude_baseline=amplitude_baseline,
        amplitude_ruido_estruturado=amplitude_ruido_estruturado,
        ordem_polinomial_baseline=ordem_polinomial_baseline,
        janela_suavizacao_ruido=janela_suavizacao_ruido,
        seed=seed,
    ).aumentar(X, y, grupos)
