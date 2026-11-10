"""B1 do parecer de 2026-09-25: `config.example.yaml` é o ponto de partida
documentado no README e no próprio comentário de cabeçalho do arquivo
(`guaraci run config.yaml`). Faltava uma verificação automatizada de que ele
de fato carrega e roda -- achado da reconferência por amostragem de
2026-10-02 (a tabela de correções marcava "falta conferir", sem teste)."""
from pathlib import Path

import pytest

from guaraci.config_io import load_config

_RAIZ = Path(__file__).resolve().parents[1]


def test_config_example_yaml_carrega_sem_erro():
    cfg = load_config(str(_RAIZ / "config.example.yaml"))
    assert cfg.mode == "sintetico"   # é o modo de demonstração, sem dado do usuário


@pytest.mark.slow
def test_config_example_yaml_roda_ate_o_fim(pq, tmp_path):
    cfg = load_config(str(_RAIZ / "config.example.yaml"))
    # overrides so' de velocidade de teste -- nao muda o que esta' sendo
    # verificado (o ARQUIVO carrega e produz uma config executavel).
    cfg.output_root_folder = str(tmp_path / "saida")
    cfg.n_permutations = 3
    cfg.show_plots = False
    pq.executar(cfg)
    from conftest import achar_pastas_run
    runs = achar_pastas_run(cfg.output_root_folder)
    assert runs, "config.example.yaml nao produziu nenhuma pasta de execucao"
    assert (Path(runs[0]) / pq.NOME_RELATORIOS / "resumo_modelo.txt").is_file()
