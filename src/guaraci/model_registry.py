"""model_registry.py — Registry de modelos do Auto-Benchmark / Monte Carlo CV
(item 20 da auditoria).

Antes desta extração, a MESMA lista de classificadores de comparação
(PLS-DA, SVM RBF, Random Forest, Gradient Boosting, XGBoost opcional) estava
hardcoded DUAS VEZES — em `benchmark_classifiers()` e `monte_carlo_cv()`
(avaliacao_modelos.py) — e havia divergido silenciosamente: o Gradient
Boosting do Monte Carlo CV não tinha `subsample=0.8` (o do benchmark tinha),
apesar da docstring de `monte_carlo_cv` afirmar "mesmos hiperparâmetros do
benchmark". Este módulo é a fonte ÚNICA de verdade; as duas funções agora
chamam `build_benchmark_list()` — a divergência foi corrigida
alinhando ao benchmark (fonte mais completa/documentada).

Adicionar, remover ou re-parametrizar um modelo do Auto-Benchmark e do
Monte Carlo CV (que sempre usam o MESMO conjunto de modelos "extras") passa
a ser: editar `_REGISTRO` aqui, em um único lugar.
"""
from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from guaraci.pipeline import Config

Construtor = Callable[[int, "Config"], Any]
log = logging.getLogger(__name__)


def _construir_pls_da(n_opt: int, cfg: "Config"):
    from guaraci.avaliacao_modelos import PLSDAClassifier
    return PLSDAClassifier(n_components=n_opt)


def _construir_svm_rbf(n_opt: int, cfg: "Config"):
    from sklearn.svm import SVC
    return SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
               random_state=cfg.seed, class_weight="balanced")


def _construir_random_forest(n_opt: int, cfg: "Config"):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                   class_weight="balanced_subsample",
                                   n_jobs=1, random_state=cfg.seed)


def _construir_grad_boost(n_opt: int, cfg: "Config"):
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                       max_depth=3, subsample=0.8,
                                       random_state=cfg.seed)


def _construir_xgboost(n_opt: int, cfg: "Config"):
    from xgboost import XGBClassifier  # ImportError se nao instalado (extra [benchmark])
    return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="mlogloss", verbosity=0,
                          n_jobs=1, random_state=cfg.seed)


# Fonte UNICA de verdade: (nome, construtor, obrigatorio).
# obrigatorio=True (so' PLS-DA) -> sempre incluido, mesmo com
# incluir_opcionais=False (caso de monte_carlo_cv quando
# cfg.monte_carlo_include_all=False: so' PLS-DA participa).
_REGISTRO: List[Tuple[str, Construtor, bool]] = [
    ("PLS-DA",        _construir_pls_da,        True),
    ("SVM RBF",       _construir_svm_rbf,       False),
    ("Random Forest", _construir_random_forest, False),
    ("Grad. Boost.",  _construir_grad_boost,    False),
    ("XGBoost",       _construir_xgboost,       False),
]


#: Excecoes que PODEM indicar dependencia opcional que nao carrega:
#: ImportError (pacote ausente), OSError (biblioteca nativa bloqueada ou
#: corrompida) e ValueError -- este so' quando for `XGBoostError` (a xgboost
#: captura o OSError da DLL internamente e relanca como XGBoostError, que
#: herda de ValueError). A classe nao e' importavel justamente quando a DLL
#: falha, por isso a checagem e' pelo nome (ver `_e_dependencia_indisponivel`).
_ERROS_DEPENDENCIA_OPCIONAL = (ImportError, OSError, ValueError)


def _e_dependencia_indisponivel(erro: BaseException) -> bool:
    """True so' para falha de CARREGAMENTO de dependencia opcional.
    Qualquer outro ValueError (ex.: parametro invalido) e' bug real e deve
    propagar, nao virar "tecnica indisponivel"."""
    if isinstance(erro, (ImportError, OSError)):
        return True
    return isinstance(erro, ValueError) and type(erro).__name__ == "XGBoostError"


def _motivo_indisponivel(nome: str, erro: Exception) -> str:
    """Mensagem em PT para uma tecnica opcional que nao pode ser construida:
    o que falhou, a causa mais provavel, e a acao sugerida. Nunca generica
    demais a ponto de esconder uma causa diferente de 'pacote ausente'."""
    texto_erro = str(erro)
    if "WinError 4551" in texto_erro or "Controle de Aplicativo" in texto_erro:
        causa = ("o Windows (Controle de Aplicativo/Smart App Control) "
                 "bloqueou a biblioteca nativa do pacote")
        acao = ("libere o arquivo bloqueado em Configuracoes > Privacidade "
                "e seguranca > Seguranca do Windows > Controle de aplicativos "
                "e navegador, ou reinstale o pacote fora de pasta sincronizada com a nuvem")
    elif isinstance(erro, ImportError):
        causa = "o pacote nao esta instalado neste ambiente"
        acao = "instale o extra opcional correspondente (ex.: pip install -e '.[benchmark]')"
    else:
        causa = f"falha ao carregar o pacote ({type(erro).__name__})"
        acao = "veja o detalhe tecnico abaixo; as demais tecnicas continuam disponiveis"
    return (f"{nome} indisponivel neste ambiente: {causa}. {acao}. "
            f"Detalhe tecnico: {texto_erro.strip().splitlines()[0][:200]}")


def _tentar_construir(nome: str, construtor: Construtor, n_opt: int,
                      cfg: "Config") -> Tuple[Optional[Any], Optional[str]]:
    """(instancia, None) se construiu; (None, motivo_em_PT) se a tecnica
    opcional nao pode ser usada neste ambiente -- NUNCA deixa a excecao
    propagar e derrubar o benchmark inteiro por causa de UMA tecnica.

    Captura ImportError, OSError e XGBoostError (nao so' ImportError): um pacote
    OPCIONAL instalado mas bloqueado pelo sistema (achado real: xgboost.dll
    barrada pelo Controle de Aplicativo do Windows, WinError 4551) levanta
    uma excecao propria do pacote (ex.: `xgboost.core.XGBoostError`, que
    herda de ValueError, nao de ImportError/OSError) -- so' capturar
    ImportError deixava a tecnica indisponivel CRASHAR o benchmark inteiro
    em vez de so' nao aparecer, com as demais tecnicas intactas (achado da
    revisao cega de 2026-09-26/instrucao de 2026-10-02)."""
    try:
        return construtor(n_opt, cfg), None
    except _ERROS_DEPENDENCIA_OPCIONAL as e:
        if not _e_dependencia_indisponivel(e):
            raise
        return None, _motivo_indisponivel(nome, e)


def build_benchmark_list(n_opt: int, cfg: "Config",
                               incluir_opcionais: bool = True
                               ) -> List[Tuple[str, Any]]:
    """Monta a lista (nome, instância) de classificadores para comparação.

    incluir_opcionais=False: só o modelo obrigatório (PLS-DA) — usado por
    monte_carlo_cv quando cfg.monte_carlo_include_all=False. Uma técnica
    opcional (ex.: xgboost) que não pode ser construída neste ambiente —
    pacote ausente OU presente mas bloqueado pelo sistema — é PULADA, nunca
    derruba as demais; o motivo vai para o log (`WARNING`), nunca em
    silêncio total. Mesma assinatura/retorno de sempre (ver
    `tests/golden/contrato_api_publica.json`).
    """
    lista: List[Tuple[str, Any]] = []
    for nome, construtor, obrigatorio in _REGISTRO:
        if not obrigatorio and not incluir_opcionais:
            continue
        instancia, motivo = _tentar_construir(nome, construtor, n_opt, cfg)
        if instancia is not None:
            lista.append((nome, instancia))
        elif motivo is not None:
            log.warning("[Benchmark] %s", motivo)
    return lista


def tecnicas_indisponiveis(n_opt: int, cfg: "Config") -> Dict[str, str]:
    """{nome: motivo_em_PT} das tecnicas OPCIONAIS que nao podem ser
    construidas agora neste ambiente -- aditivo (nao muda o retorno de
    `build_benchmark_list`), para a CLI/web avisarem o usuario ANTES de
    rodar o benchmark, em vez de so' a tecnica sumir da saida."""
    motivos: Dict[str, str] = {}
    for nome, construtor, obrigatorio in _REGISTRO:
        if obrigatorio:
            continue
        _instancia, motivo = _tentar_construir(nome, construtor, n_opt, cfg)
        if motivo is not None:
            motivos[nome] = motivo
    return motivos


def nomes_modelos_benchmark(incluir_opcionais: bool = True) -> Tuple[str, ...]:
    """Nomes registrados, sem instanciar (útil para UI/documentação)."""
    return tuple(nome for nome, _, obrig in _REGISTRO
                 if obrig or incluir_opcionais)


__all__ = ["build_benchmark_list", "nomes_modelos_benchmark", "Construtor",
          "tecnicas_indisponiveis"]
