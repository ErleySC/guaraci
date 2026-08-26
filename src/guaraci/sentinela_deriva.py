# -*- coding: utf-8 -*-
"""sentinela_deriva.py -- Monitoramento continuo de deriva do dominio de
aplicabilidade (Bloco 13b).

POR QUE ESTE MODULO EXISTE
---------------------------
O dominio de aplicabilidade (`chemometric_stats.applicability_domain_new_
samples`, ja' usado em `predicao.predict_samples`/`predict_blind`) julga
UMA amostra por vez: "esta amostra esta' dentro do dominio calibrado?".
O que ele NAO responde sozinho e' uma pergunta que so' faz sentido ao
longo do TEMPO, em producao real: "a taxa de rejeicao ESTA' SUBINDO --
o instrumento/matriz/processo derivou desde a calibracao, e o modelo
precisa ser recalibrado?"

Uma amostra isolada fora do dominio pode ser so' ruido de amostragem
(o alpha nominal ja' preve' uma fracao de rejeicoes legitimas). Uma
SEQUENCIA de amostras com taxa de rejeicao muito acima do nominal e' outra
coisa -- e' o sinal de que a populacao mudou. Este modulo distingue as
duas coisas com um TESTE ESTATISTICO (binomial exato), nao um limiar
arbitrario -- evita tanto o falso alarme (janela pequena, flutuacao
normal) quanto o alarme tardio (limiar frouxo demais).

USO TIPICO
-----------
    estado = EstadoSentinela(alpha_nominal=0.05)
    # a cada lote de predicoes (predict_samples/predict_blind ja devolve
    # a coluna AD_dentro_dominio quando o pacote tem os artefatos de AD):
    atualizar_com_predicoes(estado, df_predicoes)
    alerta = checar_deriva(estado)
    if alerta.alerta:
        print(alerta.mensagem)  # recomenda recalibracao

    # persistencia entre execucoes (LIMS/producao real, nao 1 processo so'):
    salvar_estado(estado, "sentinela.json")
    estado = carregar_estado("sentinela.json")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from guaraci.conformal import n_minimum_for_alpha

__all__ = [
    "EstadoSentinela",
    "AlertaDeriva",
    "atualizar_com_predicoes",
    "checar_deriva",
    "salvar_estado",
    "carregar_estado",
]


@dataclass
class EstadoSentinela:
    """Estado acumulado da sentinela -- 1 booleano por amostra julgada
    (`True` = dentro do dominio, `False` = rejeitada).

    `janela`: None (padrao) = ESTATISTICA CUMULATIVA, sem limite -- nunca
    descarta dado silenciosamente. Um inteiro ativa uma JANELA DESLIZANTE
    das ultimas `janela` amostras -- use quando quiser detectar deriva
    RECENTE especificamente (dados antigos, de antes da deriva comecar,
    diluiriam o sinal numa estatistica puramente cumulativa). O trade-off
    (cumulativo = mais poder estatistico mas mistura periodos; janela =
    mais sensivel a mudanca recente mas menos poder com N pequeno) e'
    deliberadamente exposto ao chamador, nao escondido atras de um
    default magico.
    """

    alpha_nominal: float = 0.05
    janela: Optional[int] = None
    historico: List[bool] = field(default_factory=list)

    def registrar(self, dentro_dominio: bool) -> None:
        self.historico.append(bool(dentro_dominio))
        if self.janela is not None and len(self.historico) > self.janela:
            self.historico = self.historico[-self.janela:]

    @property
    def n(self) -> int:
        return len(self.historico)

    @property
    def n_fora_do_dominio(self) -> int:
        return sum(1 for d in self.historico if not d)

    @property
    def taxa_rejeicao_observada(self) -> float:
        if self.n == 0:
            return float("nan")
        return self.n_fora_do_dominio / self.n


@dataclass
class AlertaDeriva:
    alerta: bool
    taxa_rejeicao_observada: float
    alpha_nominal: float
    n: int
    p_valor: float
    mensagem: str


def atualizar_com_predicoes(estado: EstadoSentinela, df_predicoes: Any
                             ) -> int:
    """Registra na sentinela cada linha de um DataFrame de predicoes que
    tenha a coluna `AD_dentro_dominio` (`predicao.predict_samples`/
    `predict_blind`, so' presente se o pacote de modelo tem os artefatos
    de dominio de aplicabilidade). Retorna quantas linhas foram
    registradas -- 0 se a coluna nao existir (nao lanca excecao: um
    pacote sem AD simplesmente nao alimenta a sentinela)."""
    if "AD_dentro_dominio" not in getattr(df_predicoes, "columns", []):
        return 0
    n_antes = estado.n
    for valor in df_predicoes["AD_dentro_dominio"]:
        estado.registrar(bool(valor))
    return estado.n - n_antes


def checar_deriva(estado: EstadoSentinela, significancia: float = 0.05,
                   n_minimo: Optional[int] = None) -> AlertaDeriva:
    """Testa H0: taxa de rejeicao verdadeira = `alpha_nominal` contra
    H1: taxa > `alpha_nominal` (deriva -- populacao rejeitando MAIS que o
    esperado), via teste binomial exato (`scipy.stats.binomtest`,
    unilateral) -- nao um limiar cru tipo "2x o nominal", que teria taxa
    de falso alarme dependente de `n` e sem justificativa formal.

    `n_minimo`: abaixo disso, nao ha' poder estatistico suficiente para
    testar (sentinela devolve `alerta=False` com aviso explicito, nunca
    finge ter testado). Default (None) usa
    `conformal.n_minimum_for_alpha(alpha_nominal)` -- MESMO minimo
    pratico ja' usado em todo o projeto para o gate conformal atingir
    aquele alpha (ex.: alpha=0.05 -> 19); reaproveitado aqui por
    consistencia, nao escolhido a dedo para este modulo.

    `significancia`: nivel do teste de deriva em si -- 0.05 por padrao,
    o MESMO alpha nominal usado em todo o projeto para os proprios gates
    (DD-SIMCA, AD, conformal) -- consistencia com a convencao ja
    estabelecida, nao uma escolha nova.
    """
    from scipy.stats import binomtest

    if n_minimo is None:
        n_minimo = n_minimum_for_alpha(estado.alpha_nominal)

    n = estado.n
    taxa = estado.taxa_rejeicao_observada
    if n < n_minimo:
        return AlertaDeriva(
            alerta=False, taxa_rejeicao_observada=taxa,
            alpha_nominal=estado.alpha_nominal, n=n, p_valor=float("nan"),
            mensagem=f"n={n} amostra(s) registrada(s), abaixo do minimo "
                     f"({n_minimo}) para testar deriva com poder "
                     f"estatistico -- continue registrando predicoes.")

    resultado = binomtest(estado.n_fora_do_dominio, n,
                           estado.alpha_nominal, alternative="greater")
    p_valor = float(resultado.pvalue)
    alerta = p_valor < significancia
    if alerta:
        mensagem = (
            f"DERIVA PROVAVEL: taxa de rejeicao observada "
            f"({taxa:.3f}, {estado.n_fora_do_dominio}/{n}) "
            f"significativamente acima do nominal "
            f"({estado.alpha_nominal:.3f}), p={p_valor:.4g} < "
            f"{significancia:.3f}. Recomenda-se investigar "
            "(instrumento/matriz/processo) e considerar recalibracao "
            "do modelo antes de confiar nas predicoes seguintes.")
    else:
        mensagem = (
            f"Sem evidencia de deriva: taxa de rejeicao observada "
            f"({taxa:.3f}, {estado.n_fora_do_dominio}/{n}) nao difere "
            f"significativamente do nominal ({estado.alpha_nominal:.3f}), "
            f"p={p_valor:.4g}.")
    return AlertaDeriva(
        alerta=alerta, taxa_rejeicao_observada=taxa,
        alpha_nominal=estado.alpha_nominal, n=n, p_valor=p_valor,
        mensagem=mensagem)


def salvar_estado(estado: EstadoSentinela, caminho: str) -> str:
    """Persiste o estado em JSON -- monitoramento continuo so' faz
    sentido entre EXECUCOES (LIMS/producao real chama o pipeline muitas
    vezes ao longo de dias/semanas, nao mantem o processo Python vivo)."""
    dados: Dict[str, Any] = {
        "alpha_nominal": estado.alpha_nominal,
        "janela": estado.janela,
        "historico": estado.historico,
    }
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f)
    return caminho


def carregar_estado(caminho: str) -> EstadoSentinela:
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)
    return EstadoSentinela(
        alpha_nominal=float(dados["alpha_nominal"]),
        janela=dados.get("janela"),
        historico=[bool(v) for v in dados.get("historico", [])])
