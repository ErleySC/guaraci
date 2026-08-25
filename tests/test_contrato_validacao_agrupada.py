# -*- coding: utf-8 -*-
"""Contrato: os metodos de classificacao e quantificacao expostos
publicamente pelo pipeline usam validacao agrupada por padrao.

POR QUE ESTE TESTE EXISTE. O subtitulo do projeto -- "Chemometrics platform
with leakage-safe validation by default" -- e' uma alegacao sobre o
software, nao um rotulo. Regra permanente (2026-08-20): nenhuma prosa de
auditoria ou de README afirma "corrigido"/"zero"/"limpo"; so' teste
executavel protege uma alegacao contra regressao futura. Este arquivo e'
essa protecao para o subtitulo.

CLASSIFICACAO e' descoberta por INTROSPECCAO do registry
(`guaraci.model_registry.nomes_modelos_benchmark`), nao por lista manual:
um metodo novo adicionado la' entra automaticamente na cobertura deste
teste, sem editar este arquivo.

QUANTIFICACAO nao tem um registry equivalente hoje: `benchmark_regressao_
por_especie` mantem sua propria lista local (PLS-R + Ridge/Lasso/Elastic
Net/SVR/Random Forest). Testado diretamente contra essa lista, lida do
DataFrame que a funcao retorna (nao hardcoded aqui) -- se um metodo novo
for adicionado la', ele aparece na coluna "Modelo" do resultado e entra na
verificacao automaticamente. Se um dia existir um registry dedicado para
quantificacao, troque a leitura da coluna pela introspeccao do registry.

ARQUITETURA QUE ISTO ASSUME (verificada por leitura de codigo em
2026-08-20): o agrupamento nao e' propriedade de CADA metodo -- e' aplicado
UMA VEZ pela funcao que orquestra a comparacao (`benchmark_classifiers`
para classificacao; o split cal/val de `benchmark_regression_by_species`
para quantificacao), que constroi o splitter group-aware e o REUTILIZA para
todos os metodos registrados. Por isso a contra-prova abaixo nao injeta um
"classificador ficticio" no registry -- isso nao provaria nada, porque
qualquer classificador que entrasse ali herdaria o mesmo splitter seguro. A
contra-prova ataca o ponto que de fato pode regredir: o DETECTOR de
vazamento de grupo. Ela roda esse detector contra um split deliberadamente
NAO agrupado e confirma que ele acusa vazamento -- prova de que o detector
distingue split seguro de split vazado, e nao so' confirma o que ja' se
sabia.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import sklearn.model_selection as skms
from sklearn.preprocessing import LabelBinarizer

from guaraci import model_registry


# =========================================================================
#  Detector de vazamento de grupo -- usado pelas duas categorias e pela
#  contra-prova.
# =========================================================================

def _nenhum_grupo_vazado(capturas) -> bool:
    """True se, em NENHUM fold capturado, um grupo aparecer nos indices de
    treino E de teste ao mesmo tempo.

    `capturas` e' uma lista de `(tr, te, groups)` -- `groups` e' o array
    efetivamente passado NAQUELA chamada de `.split(...)`, nao um array
    global assumido de fora. Isto importa de verdade aqui: o caminho de
    quantificacao faz split POR ESPECIE (`X_c = X_raw[idx]`), entao os
    indices `tr`/`te` capturados sao relativos ao subconjunto da especie
    (0..len(idx)-1), NAO ao array completo. Checar contra um `mae_id`
    global com esses indices locais compara a especie errada para
    qualquer especie que nao seja a primeira -- bug real, encontrado e
    corrigido nesta verificacao (2026-08-20): o detector "funcionava" nos
    testes anteriores so' porque a 1a especie do fixture comeca no indice
    global 0, mascarando o erro."""
    for tr, te, groups in capturas:
        if groups is None:
            continue  # split sem grupo -- nada a checar aqui
        groups = np.asarray(groups)
        if set(groups[tr]) & set(groups[te]):
            return False
    return True


# =========================================================================
#  Dados sinteticos com REPLICAS FISICAS reais por grupo -- sem replica de
#  verdade (>=2 amostras por grupo) o detector de vazamento nao tem o que
#  detectar, e as duas categorias de teste seriam vacuas.
# =========================================================================

def _dados_classificacao_com_replicas(seed=0, n_grupos_por_classe=6,
                                      n_replicas=3, p=25, n_classes=3):
    rng = np.random.default_rng(seed)
    X_list, y_list, grp_list = [], [], []
    for c in range(n_classes):
        centro_classe = rng.normal(loc=c * 5.0, size=p)
        for g in range(n_grupos_por_classe):
            centro_grupo = centro_classe + rng.normal(scale=0.3, size=p)
            grupo_id = f"C{c}G{g}"
            for _r in range(n_replicas):
                X_list.append(centro_grupo + rng.normal(scale=0.05, size=p))
                y_list.append(c)
                grp_list.append(grupo_id)
    X = np.array(X_list)
    y_int = np.array(y_list)
    grupos = np.array(grp_list)
    lb = LabelBinarizer().fit(y_int)
    return X, y_int, grupos, lb


def _dados_regressao_com_replicas(seed=0, n_especies=2, n_pontos=8,
                                  n_replicas=3, p=30):
    rng = np.random.default_rng(seed)
    X_list, conc_list, rot_list, mae_list = [], [], [], []
    especies = [f"Esp_{chr(65 + i)}" for i in range(n_especies)]
    for e_idx, especie in enumerate(especies):
        w_true = rng.normal(size=p)
        centro = rng.normal(loc=e_idx * 6.0, size=p)
        concs = np.linspace(0, 40, n_pontos)
        for p_idx, c in enumerate(concs):
            espectro_base = centro + w_true * (c / 40.0) * 3.0
            # nome de grupo GLOBALMENTE unico -- inclui a especie inteira,
            # nao so' as 3 primeiras letras (que colidem: "Esp_A" e "Esp_B"
            # dao o mesmo prefixo "ESP"). Nao explorado como bug hoje
            # porque cada split e' verificado com o `groups` local daquela
            # chamada (ver `_nenhum_grupo_vazado`), mas evita qualquer
            # colisao silenciosa se o codigo mudar para agrupar entre
            # especies no futuro.
            grupo_id = f"{especie}_{p_idx:02d}"
            for _r in range(n_replicas):
                X_list.append(espectro_base + rng.normal(scale=0.10, size=p))
                conc_list.append(c)
                rot_list.append(especie)
                mae_list.append(grupo_id)
    X = np.array(X_list)
    conc = np.array(conc_list)
    rotulos = np.array(rot_list)
    mae_id = np.array(mae_list)
    classes_unicas = np.array(especies)
    return X, conc, rotulos, mae_id, classes_unicas


# =========================================================================
#  Espiao: subclasse que delega ao splitter real e grava cada
#  (treino, teste, groups) produzido, sem alterar o comportamento.
#
#  Grava `groups` JUNTO com os indices -- nao so' os indices -- porque o
#  caminho de quantificacao chama `.split()` uma vez POR ESPECIE, cada vez
#  com um array de grupos diferente (subconjunto local daquela especie).
#  Guardar so' os indices e checar depois contra um array global daria
#  falso resultado para qualquer especie que nao seja a primeira.
# =========================================================================

def _espiao(classe_real, destino: list):
    class _Espiao(classe_real):
        def split(self, X, y=None, groups=None):
            for tr, te in super().split(X, y, groups):
                destino.append((np.asarray(tr), np.asarray(te), groups))
                yield tr, te
    _Espiao.__name__ = f"Espiao{classe_real.__name__}"
    return _Espiao


# =========================================================================
#  1. CLASSIFICACAO -- introspeccao do registry + comportamento real
# =========================================================================

@pytest.mark.slow
def test_classificacao_cada_metodo_registrado_usa_cv_group_aware(
        pq, tmp_path, monkeypatch):
    """Para cada metodo em `model_registry` (descoberto por introspeccao,
    nao listado a mao): roda de verdade atraves de `benchmark_classifiers`
    e confirma que o splitter usado foi `StratifiedGroupKFold`, com grupos
    nao-vazios, e que nenhum grupo cruzou treino/teste em fold nenhum."""
    nomes_esperados = set(model_registry.nomes_modelos_benchmark(
        incluir_opcionais=True))
    assert nomes_esperados, "registry vazio -- nada para este teste cobrir"

    X, y_int, grupos, lb = _dados_classificacao_com_replicas()

    capturados: list = []
    monkeypatch.setattr(
        skms, "StratifiedGroupKFold",
        _espiao(skms.StratifiedGroupKFold, capturados))

    cfg = pq.Config(n_splits_cv=3, seed=0, run_shap=False)
    pasta = str(tmp_path)
    os.makedirs(os.path.join(pasta, pq.NOME_TABELAS), exist_ok=True)
    os.makedirs(os.path.join(pasta, pq.NOME_GRAFICOS), exist_ok=True)

    df = pq.benchmark_classifiers(X, y_int, grupos, lb, n_opt=2,
                                      cfg=cfg, pasta=pasta)

    # (a) o splitter group-aware foi de fato instanciado e usado --
    #     nao um caminho alternativo silenciosamente diferente.
    assert capturados, (
        "StratifiedGroupKFold nunca foi chamado -- benchmark_classifiers "
        "nao passou por este splitter, ou o import mudou de lugar e este "
        "teste precisa ser atualizado.")

    # (b) TODO metodo introspectado de fato rodou (nenhum foi pulado).
    assert nomes_esperados == set(df["Classificador"]), (
        f"registry declara {nomes_esperados}, mas so' rodaram "
        f"{set(df['Classificador'])}")

    # (c) nenhum fold, de nenhum metodo, vazou grupo entre treino e teste.
    assert _nenhum_grupo_vazado(capturados), (
        "um grupo (replica fisica) apareceu em treino E teste no mesmo "
        "fold -- e' o vazamento que o subtitulo declara nao existir por "
        "padrao.")


def test_deteccao_de_vazamento_pega_metodo_ficticio_sem_agrupamento():
    """Contra-prova. `_nenhum_grupo_vazado` e' o mesmo detector usado no
    teste acima -- sem provar que ele DISTINGUE split seguro de split
    vazado, o teste acima poderia estar passando por vacuidade (sempre
    True, indiferente ao que aconteceu).

    O "metodo ficticio" e' um `StratifiedKFold` comum: exatamente o que um
    metodo de classificacao adicionado sem passar por
    `benchmark_classifiers` produziria se ignorasse agrupamento --
    plausivel, porque e' a chamada mais obvia de se fazer por engano no
    lugar de `StratifiedGroupKFold`.
    """
    X, y_int, grupos, _lb = _dados_classificacao_com_replicas()

    cv_ficticio = skms.StratifiedKFold(n_splits=3, shuffle=True,
                                       random_state=0)
    splits_ficticios = [(tr, te, grupos)
                        for tr, te in cv_ficticio.split(X, y_int)]

    assert not _nenhum_grupo_vazado(splits_ficticios), (
        "o metodo ficticio (StratifiedKFold, sem agrupamento) deveria "
        "vazar grupo entre treino e teste; se nao vazou, os dados "
        "sinteticos nao tem replica suficiente para exercitar o caso, e "
        "o teste anterior estaria testando um detector vazio.")


# =========================================================================
#  2. QUANTIFICACAO -- benchmark_regression_by_species nao tem registry
#  dedicado; a lista de metodos e' lida do proprio resultado.
# =========================================================================

@pytest.mark.slow
def test_quantificacao_cada_metodo_usa_split_group_aware(pq, tmp_path,
                                                          monkeypatch):
    """PLS-R + Ridge/Lasso/Elastic Net/SVR/Random Forest compartilham o
    MESMO split cal/val por especie (comentario da propria funcao: "MESMO
    split... usado em pls_regression_by_species"). Confirma que esse split
    e' `GroupShuffleSplit` com `groups=mae_id`, capturando a chamada real, e
    que nenhum grupo aparece dos dois lados de cal/val -- nem no split
    cal/val, nem na CV interna (`GroupKFold`) usada so' para escolher o
    numero de LVs.

    ARQUITETURA -- ao contrario da classificacao, a quantificacao NAO tem um
    unico ponto orquestrador. Ha' TRES pontos de producao com split
    group-aware independente, verificado por leitura de codigo em
    2026-08-20:

      1. `pls_regression_by_species` (pipeline.py) -- cal/val
         (`GroupShuffleSplit`, import de MODULO) + CV interna de LV
         (`GroupKFold`, import de MODULO).
      2. `benchmark_regression_by_species` (avaliacao_modelos.py) -- MESMA
         logica de split REIMPLEMENTADA (comentario no proprio codigo:
         "para evitar acoplamento circular com pipeline.py"), com
         `GroupShuffleSplit` importado LOCALMENTE dentro da funcao.
      3. `pls_regressao_pooled` (pipeline.py) -- PLS-R sobre TODAS as
         amostras de uma vez (sem separar por especie); e' o caminho que
         roda quando (1) devolve `None` -- dataset de especie unica, nivel
         N1, ou nenhuma especie com amostras adulteradas suficientes. Ate'
         2026-08-20 vivia inline dentro de `executar()` (~150 linhas, sem
         funcao propria, sem teste). Extraida nesta data (Passo 27): mesmo
         corpo, mesmo comportamento -- so' ganhou nome e assinatura. Ver
         `test_quantificacao_pooled_usa_split_group_aware` abaixo.

    Este teste cobre (1) e (2) por completo -- inclusive a CV interna de
    LV-selection de (1), fechada em 2026-08-20 depois que a verificacao
    independente mostrou que o teste passava mesmo com essa CV interna
    quebrada (o `GroupShuffleSplit` do cal/val continuava correto e
    escondia o problema). (3) tem teste dedicado separado, abaixo -- os
    tres caminhos de quantificacao estao cobertos.
    """
    X, conc, rotulos, mae_id, classes_unicas = _dados_regressao_com_replicas()
    cfg = pq.Config(seed=0, max_lvs=5, frac_cal=0.7)
    pasta = str(tmp_path)
    os.makedirs(os.path.join(pasta, pq.NOME_TABELAS), exist_ok=True)
    os.makedirs(os.path.join(pasta, pq.NOME_GRAFICOS), exist_ok=True)

    capturados_split: list = []
    capturados_cv: list = []
    # `pls_regression_by_species` importa GroupShuffleSplit/GroupKFold no
    # TOPO de pipeline.py (nome ja' vinculado no modulo); precisa de patch
    # direto no modulo, nao no atributo de sklearn.model_selection.
    monkeypatch.setattr(
        pq, "GroupShuffleSplit",
        _espiao(skms.GroupShuffleSplit, capturados_split))
    monkeypatch.setattr(
        pq, "GroupKFold",
        _espiao(skms.GroupKFold, capturados_cv))
    # `benchmark_regression_by_species` (avaliacao_modelos.py) importa
    # GroupShuffleSplit DENTRO da funcao -- o patch no atributo do modulo
    # sklearn.model_selection e' o que alcanca esse import local.
    monkeypatch.setattr(
        skms, "GroupShuffleSplit",
        _espiao(skms.GroupShuffleSplit, capturados_split))

    reg_esp = pq.pls_regression_by_species(
        X, conc, rotulos, mae_id, classes_unicas, cfg, pasta, n_splits=3)
    assert reg_esp is not None, "fixture nao gerou dados suficientes p/ PLS-R"

    df = pq.benchmark_regression_by_species(
        X, conc, rotulos, mae_id, classes_unicas, cfg, pasta, reg_esp)
    assert df is not None

    nomes_rodados = set(df["Modelo"])
    assert {"PLS-R", "Ridge", "Lasso", "Elastic Net", "SVR (RBF)",
            "Random Forest"}.issubset(nomes_rodados)

    assert capturados_split, (
        "GroupShuffleSplit nunca foi chamado -- a quantificacao nao passou "
        "por um split group-aware, ou o import mudou de lugar.")
    assert _nenhum_grupo_vazado(capturados_split), (
        "um grupo (replica fisica) apareceu em calibracao E validacao no "
        "mesmo split -- e' o vazamento que o subtitulo declara nao existir "
        "por padrao, agora do lado de quantificacao.")

    assert capturados_cv, (
        "GroupKFold nunca foi chamado -- a CV interna de selecao de LV nao "
        "passou por um splitter group-aware, ou o import mudou de lugar. "
        "Achado real em 2026-08-20: com essa CV quebrada, o cal/val "
        "continuava correto e a checagem acima nao via nada de errado.")
    assert _nenhum_grupo_vazado(capturados_cv), (
        "um grupo apareceu nos dois lados de um fold da CV INTERNA de "
        "selecao de numero de LVs -- vaza informacao da mesma replica "
        "fisica entre treino e teste ao escolher o hiperparametro, mesmo "
        "que o split cal/val esteja correto.")


# =========================================================================
#  3. QUANTIFICACAO POOLED -- terceiro ponto de split, extraido de
#  executar() em 2026-08-20 (Passo 27) especificamente para ganhar este
#  teste. Mesmo padrao dos dois anteriores: split cal/val (GroupShuffleSplit)
#  + CV interna de LV (GroupKFold), ambos capturados e checados.
# =========================================================================

def _dados_regressao_uma_especie_com_replicas(seed=0, n_pontos=8,
                                              n_replicas=3, p=30):
    """Uma UNICA especie -- e' o gatilho real do caminho pooled:
    `pls_regression_by_species` so' roda por especie quando ha' mais de
    uma; com uma so', a orquestracao em executar() cai para
    `pls_regressao_pooled`."""
    rng = np.random.default_rng(seed)
    X_list, conc_list, mae_list = [], [], []
    w_true = rng.normal(size=p)
    concs = np.linspace(0, 40, n_pontos)
    for p_idx, c in enumerate(concs):
        espectro_base = w_true * (c / 40.0) * 3.0
        grupo_id = f"Especie_Unica_{p_idx:02d}"
        for _r in range(n_replicas):
            X_list.append(espectro_base + rng.normal(scale=0.10, size=p))
            conc_list.append(c)
            mae_list.append(grupo_id)
    X = np.array(X_list)
    conc = np.array(conc_list)
    mae_id = np.array(mae_list)
    rotulos = np.array(["Especie_Unica"] * len(X))
    return X, conc, rotulos, mae_id


@pytest.mark.slow
def test_quantificacao_pooled_usa_split_group_aware(pq, tmp_path,
                                                     monkeypatch):
    """`pls_regressao_pooled` -- terceiro ponto de split da quantificacao,
    extraido de dentro de `executar()` especificamente para ganhar este
    teste (Passo 27, 2026-08-20). Mesma verificacao dos dois primeiros:
    split cal/val (`GroupShuffleSplit`) e CV interna de selecao de LV
    (`GroupKFold`) capturados e checados quanto a vazamento de grupo."""
    X, conc, rotulos, mae_id = _dados_regressao_uma_especie_com_replicas()
    cfg = pq.Config(seed=0, max_lvs=5, frac_cal=0.7)
    pasta = str(tmp_path)
    for d in (pq.NOME_TABELAS, pq.NOME_GRAFICOS, pq.NOME_RELATORIOS):
        os.makedirs(os.path.join(pasta, d), exist_ok=True)
    pasta_logs = os.path.join(pasta, pq.NOME_RELATORIOS)

    capturados_split: list = []
    capturados_cv: list = []
    monkeypatch.setattr(
        pq, "GroupShuffleSplit",
        _espiao(skms.GroupShuffleSplit, capturados_split))
    monkeypatch.setattr(
        pq, "GroupKFold",
        _espiao(skms.GroupKFold, capturados_cv))

    resultado = pq.pls_regressao_pooled(
        X, conc, rotulos, mae_id, cfg, pasta, pasta_logs, n_splits=3)

    assert resultado is not None
    assert "rmsep" in resultado and "r2v" in resultado

    assert capturados_split, (
        "GroupShuffleSplit nunca foi chamado no caminho pooled -- ou nao "
        "passou por split group-aware, ou o import mudou de lugar.")
    assert _nenhum_grupo_vazado(capturados_split), (
        "um grupo (replica fisica) apareceu em calibracao E validacao no "
        "split cal/val do caminho pooled.")

    assert capturados_cv, (
        "GroupKFold nunca foi chamado no caminho pooled -- a CV interna de "
        "selecao de LV nao passou por splitter group-aware.")
    assert _nenhum_grupo_vazado(capturados_cv), (
        "um grupo apareceu nos dois lados de um fold da CV interna de "
        "selecao de LV, no caminho pooled.")
