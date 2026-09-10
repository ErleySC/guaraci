"""test_identificacao_generica.py -- Passo 122
(INSTRUCAO_HIPOTESE_D_IDENTIFICACAO_GENERICA...md): contra-prova de que
a Identificacao (Bloco 9b) funciona com uma convencao de nome de
"segundo fator" DIFERENTE da original do dataset de oleo (letras A/M/S
-> algodao/milho/soja), sem alterar codigo-fonte -- so' declarando
`codigos_adulterante` num perfil de matriz.

Achado do Passo 117 (auditoria de adaptabilidade): `train_
identification_ensemble` chamava `dados_io.adulterant_from_mae_id` com
o mapa GLOBAL `ADULTERANTE_NOME` fixo -- qualquer letra fora de A/M/S
(mesmo com a MESMA estrutura de mae_id `{cod}-{data_ou_ponto}-{letra}
{teor}`) retornava `None`, entao NUNCA formava uma combinacao. Achado
diagnosticado (Passo 122): e' um problema de PARSING (mapa hardcoded),
nao de conceito -- a estrutura do token (1 letra + digitos no ultimo
segmento) ja' era generica, so' o DICIONARIO letra->nome nao tinha como
ser trocado sem editar `dados_io.py`.

Corrigido: `MatrixProfile.codigos_adulterante` (novo campo, generalizando
o mesmo padrao ja' usado por `codigos_classe`) + `mapa_adulterante`
repassado explicitamente por `train_identification_ensemble`/
`r2cv_species_by_adulterant`/`adulterant_from_mae_id`. Mapa vazio
(default) preserva EXATAMENTE o comportamento historico -- nenhum
`.joblib` ja persistido muda de forma (mesmas chaves `(especie,
adulterante)`, mesmos campos internos).
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import yaml

from conftest import achar_pastas_run
from guaraci.identificacao import CoverageStatus


def test_ensemble_generico_com_convencao_de_nome_diferente_da_original(
        pq, tmp_path):
    """Dataset sintetico com `synthetic_adulterants=("X", "Y")` -- letras
    DELIBERADAMENTE diferentes de A/M/S (o dataset original de oleo) --
    e um perfil de matriz FICTICIO declarando `codigos_adulterante=
    {"X": "quitosana", "Y": "amido"}`. Roda `executar()` ponta-a-ponta
    (mesmo padrao de `pkg_bloco9b` em test_fluxo_cego_bloco9b.py) e
    confirma que o ensemble de Identificacao NAO fica vazio -- prova
    direta de que a convencao de nome e' pluggable via perfil, sem
    tocar em `dados_io.py`/`identificacao.py`."""
    caminho_perfil = tmp_path / "matriz_ficticia_quitosana_amido.yaml"
    caminho_perfil.write_text(yaml.safe_dump({
        "descricao": "Matriz ficticia p/ teste de convencao de adulterante nova",
        "unidade_eixo": "cm-1",
        "vocabulario": {
            "classe": "especie", "classe_plural": "especies",
            "matriz": "matriz ficticia", "alvo": "o teor do analito",
            "conforme": "puro", "nao_conforme": "adulterado",
        },
        "codigos_adulterante": {"X": "quitosana", "Y": "amido"},
    }), encoding="utf-8")

    cfg = pq.Config(
        input_folder=str(tmp_path / "dados"),
        output_root_folder=str(tmp_path / "saida"),
        mode="sintetico", level="N3",
        matrix_profile=str(caminho_perfil),
        n_per_class=12, n_synthetic_points=60, n_synthetic_replicates=3,
        synthetic_adulterants=("X", "Y"),  # NAO e' A/M/S -- convencao nova
        wn_min=400.0, wn_max=4001.0,
        n_splits_cv=2, n_repeats_cv=1, n_permutations=5,
        n_permutations_wold=5, n_bootstrap_vip=3, n_bootstrap_bca=20,
        n_monte_carlo=3, max_lvs=5,
    )
    os.makedirs(cfg.input_folder, exist_ok=True)
    pq.executar(cfg)

    runs = achar_pastas_run(cfg.output_root_folder)
    assert runs, "executar() nao criou pasta de saida"
    cam_modelo = os.path.join(runs[0], pq.NOME_MODELOS, "modelo_plsda.joblib")
    pkg = joblib.load(cam_modelo)

    ensemble = pkg["identification_ensemble"]
    assert ensemble, (
        "ensemble de Identificacao VAZIO com convencao de nome diferente "
        "da original -- a generalizacao do Passo 122 nao funcionou.")
    adulterantes_vistos = {ad for _esp, ad in ensemble}
    assert adulterantes_vistos == {"quitosana", "amido"}, (
        f"esperava {{'quitosana', 'amido'}} (do perfil ficticio), "
        f"achou {adulterantes_vistos!r}")
    # nunca deve inventar/vazar os nomes do dataset ORIGINAL de oleo
    assert "algodão" not in adulterantes_vistos
    assert "milho" not in adulterantes_vistos
    assert "soja" not in adulterantes_vistos

    # Detectar -> Identificar -> Quantificar ponta-a-ponta, contra o
    # pacote real (nao um pkg montado a mao) -- mesmo padrao de
    # test_fluxo_cego_bloco9b.py.
    from guaraci.predicao import predict_blind

    wn = np.asarray(pkg["wavenumbers"], dtype=float)
    X_novos = np.random.default_rng(0).normal(loc=0.5, scale=0.05,
                                              size=(3, len(wn)))
    _df, resultados = predict_blind(pkg, X_novos, wn)
    assert len(resultados) == 3
    # nao afirma que a identificacao vai VALIDAR (n_grupos pequeno no
    # dataset sintetico, mesmo regime ja documentado em
    # test_mae_id_sintetico_sem_data_cai_em_nao_validado_n1) -- so'
    # confirma que o fluxo roda ponta-a-ponta sem excecao, com o
    # ensemble genuinamente calibrado (nao vazio) por tras dele.
    for r in resultados:
        ident = r.identificacao.classe_identificada
        assert ident is None or isinstance(ident, str)


def test_mapa_adulterante_vazio_preserva_comportamento_historico(pq, tmp_path):
    """Contra-prova de retrocompatibilidade: SEM `codigos_adulterante`
    no perfil (perfil "generico", igual ao usado sempre), o dataset
    sintetico com letras A/M/S (convencao ORIGINAL) continua calibrando
    exatamente como antes -- a generalizacao e' aditiva, nunca muda o
    resultado de quem nao declarou nada novo."""
    cfg = pq.Config(
        input_folder=str(tmp_path / "dados"),
        output_root_folder=str(tmp_path / "saida"),
        mode="sintetico", level="N3",
        n_per_class=12, n_synthetic_points=60, n_synthetic_replicates=3,
        synthetic_adulterants=("S", "M"),  # convencao ORIGINAL do oleo
        wn_min=400.0, wn_max=4001.0,
        n_splits_cv=2, n_repeats_cv=1, n_permutations=5,
        n_permutations_wold=5, n_bootstrap_vip=3, n_bootstrap_bca=20,
        n_monte_carlo=3, max_lvs=5,
    )
    os.makedirs(cfg.input_folder, exist_ok=True)
    pq.executar(cfg)

    runs = achar_pastas_run(cfg.output_root_folder)
    cam_modelo = os.path.join(runs[0], pq.NOME_MODELOS, "modelo_plsda.joblib")
    pkg = joblib.load(cam_modelo)
    ensemble = pkg["identification_ensemble"]

    assert ensemble, "nenhuma combinacao especie x adulterante calibrada"
    assert {"Esp_A", "Esp_B", "Esp_C"} == {esp for esp, _ in ensemble}
    assert {"soja", "milho"} == {ad for _, ad in ensemble}
    for info in ensemble.values():
        assert info["cobertura_status"] == CoverageStatus.NOT_VALIDATED_N1
