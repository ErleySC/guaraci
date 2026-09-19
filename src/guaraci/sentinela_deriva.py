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
SEQUENCIA de amostras com taxa de rejeicao muito acima da esperada e'
outra coisa -- e' o sinal de que a populacao mudou. Este modulo distingue
as duas coisas com um TESTE ESTATISTICO, nao um limiar arbitrario --
evita tanto o falso alarme (janela pequena, flutuacao normal) quanto o
alarme tardio (limiar frouxo demais).

O TESTE (Passo 220, correcao R2 do achado do Passo 219)
-------------------------------------------------------
Hipotese nula: a taxa de rejeicao em producao e' IGUAL a taxa que a
PROPRIA calibracao rejeita em dados que nao viu (validacao cruzada,
`chemometric_stats.ad_rejection_rate_cv`, guardada no pacote de modelo).
Teste exato de Fisher unilateral, DUAS amostras: (rejeitadas, aceitas) em
producao contra (rejeitadas, aceitas) na CV da calibracao -- a incerteza
da referencia entra no teste.

Por que NAO comparar contra o `alpha` nominal (5%), como esta versao
fazia antes: um Dominio de Aplicabilidade calibrado com n finito, em
espectros reais, rejeita em controle uma taxa diferente do nominal e
diferente a cada calibracao -- e' o "efeito da estimacao de parametros da
Fase I" em controle estatistico de processo (Jensen, Jones-Farmer, Champ &
Woodall 2006, J. Qual. Technol. 38(4):349-364, DOI
10.1080/00224065.2006.11918623). Medido no Corn (n_cal=40): rejeicao por
amostra em controle 6,5% (nao 5%) e falso alarme do sentinela ~21% contra
o nominal; com a referencia de CV, o falso alarme cai para ~3% sem perda
de poder (`docs/VALIDACAO_PUBLICA.md` §11).

Modelos SEM a referencia (pacotes salvos antes do Passo 220) caem no teste
binomial contra `alpha_nominal` -- comportamento antigo, sinalizado no
campo `DriftAlert.teste` e na mensagem (falso alarme inflado); retreinar o
modelo grava a referencia.

USO TIPICO
-----------
    estado = SentinelState(alpha_nominal=0.05)
    # a cada lote de predicoes (predict_samples/predict_blind ja devolve
    # a coluna AD_dentro_dominio quando o pacote tem os artefatos de AD):
    update_with_predictions(estado, df_predicoes)
    alerta = check_drift(estado)
    if alerta.alerta:
        print(alerta.mensagem)  # recomenda recalibracao

    # persistencia entre execucoes (LIMS/producao real, nao 1 processo so'):
    save_state(estado, "sentinela.json")
    estado = load_state("sentinela.json")
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from guaraci.conformal import n_minimum_for_alpha

__all__ = [
    "SentinelState",
    "DriftAlert",
    "update_with_predictions",
    "check_drift",
    "save_state",
    "load_state",
    "hook_apos_predicao",
]


@dataclass
class SentinelState:
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

    `ref_rejeitadas`/`ref_n`: referencia de calibracao (rejeicoes / amostras
    avaliadas na validacao cruzada do dominio, ver
    `chemometric_stats.ad_rejection_rate_cv`). Presente => teste de 2
    amostras (R2); ausente (`None`) => teste binomial legado contra
    `alpha_nominal`.
    """

    alpha_nominal: float = 0.05
    janela: Optional[int] = None
    historico: List[bool] = field(default_factory=list)
    ref_rejeitadas: Optional[int] = None
    ref_n: Optional[int] = None

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
class DriftAlert:
    alerta: bool
    taxa_rejeicao_observada: float
    alpha_nominal: float
    n: int
    p_valor: float
    mensagem: str
    teste: str = "binomial_nominal"          # ou "duas_amostras_cv" (R2)
    taxa_referencia: Optional[float] = None  # taxa da CV da calibracao (R2)


def update_with_predictions(estado: SentinelState, df_predicoes: Any
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


def check_drift(estado: SentinelState, significancia: float = 0.05,
                 n_minimo: Optional[int] = None) -> DriftAlert:
    """Testa se a taxa de rejeicao em producao SUBIU (deriva).

    Com referencia de calibracao (`estado.ref_rejeitadas`/`ref_n`, teste
    R2 -- ver docstring do modulo): H0 = a taxa em producao e' igual a taxa
    que a calibracao rejeita em validacao cruzada; H1 = maior. Teste exato
    de Fisher unilateral, duas amostras (`scipy.stats.fisher_exact`,
    `alternative="greater"`).

    Sem referencia (modelo antigo): H0 = taxa = `alpha_nominal`, teste
    binomial exato unilateral (`scipy.stats.binomtest`) -- comportamento
    anterior ao Passo 220, sinalizado em `DriftAlert.teste` e na mensagem
    (falso alarme inflado em espectros reais; medido ~21% no Corn).
    Nunca um limiar cru tipo "2x o nominal", que teria taxa de falso alarme
    dependente de `n` e sem justificativa formal.

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
    from scipy.stats import binomtest, fisher_exact

    if n_minimo is None:
        n_minimo = n_minimum_for_alpha(estado.alpha_nominal)

    ref_r, ref_m = estado.ref_rejeitadas, estado.ref_n
    taxa_ref: Optional[float] = None
    if ref_r is not None and ref_m is not None and ref_m > 0:
        taxa_ref = ref_r / ref_m
    teste = "duas_amostras_cv" if taxa_ref is not None else "binomial_nominal"

    n = estado.n
    taxa = estado.taxa_rejeicao_observada
    if n < n_minimo:
        return DriftAlert(
            alerta=False, taxa_rejeicao_observada=taxa,
            alpha_nominal=estado.alpha_nominal, n=n, p_valor=float("nan"),
            mensagem=f"n={n} amostra(s) registrada(s), abaixo do minimo "
                     f"({n_minimo}) para testar deriva com poder "
                     f"estatistico -- continue registrando predicoes.",
            teste=teste, taxa_referencia=taxa_ref)

    k = estado.n_fora_do_dominio
    if taxa_ref is not None and ref_r is not None and ref_m is not None:
        p_valor = float(fisher_exact([[k, n - k], [ref_r, ref_m - ref_r]],
                                     alternative="greater")[1])
        alvo = (f"taxa de rejeicao da propria calibracao em validacao "
                f"cruzada ({taxa_ref:.3f}, {ref_r}/{ref_m})")
        aviso = ""
    else:
        p_valor = float(binomtest(k, n, estado.alpha_nominal,
                                  alternative="greater").pvalue)
        alvo = f"nominal ({estado.alpha_nominal:.3f})"
        aviso = (" [teste LEGADO contra o nominal: este modelo nao guarda a "
                 "referencia de calibracao -- falso alarme inflado em "
                 "espectros reais (~21% medido no Corn); retreine o modelo "
                 "para o teste calibrado]")

    alerta = p_valor < significancia
    if alerta:
        mensagem = (
            f"DERIVA PROVAVEL: taxa de rejeicao observada "
            f"({taxa:.3f}, {k}/{n}) significativamente acima da {alvo}, "
            f"p={p_valor:.4g} < {significancia:.3f}. Recomenda-se investigar "
            "(instrumento/matriz/processo) e considerar recalibracao "
            "do modelo." + aviso)
    else:
        mensagem = (
            f"Sem evidencia de deriva: taxa de rejeicao observada "
            f"({taxa:.3f}, {k}/{n}) nao difere significativamente da "
            f"{alvo}, p={p_valor:.4g}." + aviso)
    return DriftAlert(
        alerta=alerta, taxa_rejeicao_observada=taxa,
        alpha_nominal=estado.alpha_nominal, n=n, p_valor=p_valor,
        mensagem=mensagem, teste=teste, taxa_referencia=taxa_ref)


def save_state(estado: SentinelState, caminho: str) -> str:
    """Persiste o estado em JSON -- monitoramento continuo so' faz
    sentido entre EXECUCOES (LIMS/producao real chama o pipeline muitas
    vezes ao longo de dias/semanas, nao mantem o processo Python vivo)."""
    dados: Dict[str, Any] = {
        "alpha_nominal": estado.alpha_nominal,
        "janela": estado.janela,
        "historico": estado.historico,
        "ref_rejeitadas": estado.ref_rejeitadas,
        "ref_n": estado.ref_n,
    }
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f)
    return caminho


def load_state(caminho: str) -> SentinelState:
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)
    ref_r, ref_n = dados.get("ref_rejeitadas"), dados.get("ref_n")
    return SentinelState(
        alpha_nominal=float(dados["alpha_nominal"]),
        janela=dados.get("janela"),
        historico=[bool(v) for v in dados.get("historico", [])],
        ref_rejeitadas=None if ref_r is None else int(ref_r),
        ref_n=None if ref_n is None else int(ref_n))


def _referencia_do_lote(df_predicoes: Any) -> Optional[Tuple[int, int]]:
    """Referencia de CV da calibracao que `predicao.predict_samples`
    anexa como colunas constantes (`AD_ref_cv_rejeitadas`/`AD_ref_cv_n`),
    ou `None` se o pacote de modelo e' anterior ao Passo 220."""
    colunas = getattr(df_predicoes, "columns", [])
    if ("AD_ref_cv_rejeitadas" not in colunas or "AD_ref_cv_n" not in colunas
            or len(df_predicoes) == 0):
        return None
    r = df_predicoes["AD_ref_cv_rejeitadas"].iloc[0]
    m = df_predicoes["AD_ref_cv_n"].iloc[0]
    if r != r or m != m or int(m) <= 0:      # NaN ou referencia vazia
        return None
    return int(r), int(m)


def hook_apos_predicao(caminho_modelo: str, df_predicoes: Any
                        ) -> Optional[DriftAlert]:
    """Orquestra a sentinela junto de UMA rodada de predicao -- carrega o
    estado persistido (ou cria um novo), registra o lote, salva de volta,
    e devolve o alerta de deriva. MESMA logica para o menu do CLI
    (`guaraci.py`, Bloco 13b) e a aba do app web (`app_tabs/predicao.py`)
    -- extraida para as duas superficies chamarem UMA funcao em vez de
    duplicar o fluxo carregar->atualizar->salvar->checar (mesmo padrao ja'
    usado por `app_logic.log_progress` para a barra de progresso).

    "Em linha" aqui significa execucao POR LOTE de predicao (nao um
    servidor continuo -- ver `docs/ESCOPO_FUSAO_MULTIBLOCO_E_MSPC.md` §2.2):
    cada chamada desta funcao e' 1 lote processado, o estado acumula entre
    chamadas via o arquivo `<caminho_modelo>.sentinela.json`.

    A referencia de calibracao (teste R2) vem do proprio lote de predicoes
    (colunas `AD_ref_cv_*`). Se o modelo foi RECALIBRADO sob o mesmo
    caminho (referencia diferente da guardada no estado), o historico e'
    zerado: rejeicoes julgadas por outra calibracao nao podem ser
    comparadas com a referencia da nova.

    Devolve `None` (sem lancar) quando `df_predicoes` nao tem a coluna
    `AD_dentro_dominio` -- pacote de modelo antigo, sem Dominio de
    Aplicabilidade, simplesmente nao alimenta a sentinela.
    """
    if "AD_dentro_dominio" not in getattr(df_predicoes, "columns", []):
        return None
    caminho_estado = str(caminho_modelo) + ".sentinela.json"
    estado = (load_state(caminho_estado) if os.path.isfile(caminho_estado)
              else SentinelState(alpha_nominal=0.05))
    ref = _referencia_do_lote(df_predicoes)
    if ref is not None:
        ref_estado = (None if estado.ref_rejeitadas is None
                      or estado.ref_n is None
                      else (estado.ref_rejeitadas, estado.ref_n))
        if ref_estado is not None and ref_estado != ref:
            estado.historico = []            # nova calibracao, novo baseline
        estado.ref_rejeitadas, estado.ref_n = ref
    update_with_predictions(estado, df_predicoes)
    save_state(estado, caminho_estado)
    return check_drift(estado)
