"""Tela T itens 7 (portao de correcao de sinal) e 8 (modelo unico x por especie): runners e tarefa pelo menu."""
import numpy as np
import pytest

from guaraci.tecnicas_avancadas import rodar_pooled_local, rodar_portao_correcao


def _dados(seed=0, n_grupos=12, rep=3, p=60):
    rng = np.random.default_rng(seed)
    grupos = np.repeat([f"g{i:02d}" for i in range(n_grupos)], rep)
    especie = np.where(np.arange(n_grupos * rep) < n_grupos * rep // 2, "A", "B")
    conc = np.repeat(rng.uniform(0, 20, n_grupos), rep)
    base = np.sin(np.linspace(0, 6, p))
    X = conc[:, None] * base[None, :] / 20 + rng.normal(0, 0.02, (n_grupos * rep, p))
    X = X * rng.uniform(0.8, 1.2, (n_grupos * rep, 1))           # efeito multiplicativo
    return X, conc, especie, grupos


def test_portao_devolve_veredito_com_poder_e_recusa_sem_grupo():
    X, conc, _esp, grupos = _dados()
    v = rodar_portao_correcao(X, conc, grupos, "emsc", classificacao=False, n_componentes=2, n_seeds=5)
    assert v.metodo == "EMSC" and v.metrica == "RMSEP" and v.n_pares == 5
    assert isinstance(v.poder_suficiente, bool) and v.veredito
    with pytest.raises(ValueError, match="mae_id"):
        rodar_portao_correcao(X, conc, None, "emsc", classificacao=False)
    with pytest.raises(ValueError, match="metodo"):
        rodar_portao_correcao(X, conc, grupos, "snv", classificacao=False)


def test_pooled_local_uma_decisao_por_especie_e_conservadora():
    X, conc, esp, grupos = _dados()
    decisoes = rodar_pooled_local(X, conc, esp, grupos, n_componentes=2, n_seeds=5)
    assert [d.especie for d in decisoes] == ["A", "B"]
    for d in decisoes:
        assert d.recomendacao in ("pooled", "local")
        if d.recomendacao == "local":
            assert d.veredito is not None and d.veredito.poder_suficiente
    with pytest.raises(ValueError, match="concentracao"):
        rodar_pooled_local(X, np.full(len(conc), np.nan), esp, grupos)


@pytest.mark.slow
def test_tarefa_portao_e_pooled_pela_tela_t(tmp_path):
    from tests.test_tarefas_cli import _roda
    destino = tmp_path / "q"
    # a lista de tecnicas reabre depois de cada uma: um Enter a mais para voltar dela
    roteiro = ["1.1", "quantificar", str(destino), "s", "", "v", "3.5", "7", "emsc", "", "", "3.5", "8", "", "", "q", "S"]
    assert len(roteiro) <= 17
    saida, _, rc = _roda(roteiro, tmp_path, timeout=600)
    assert rc == 0, saida[-800:]
    assert "EMSC" in saida and "nada foi alterado na configuracao" in saida
    assert "especie_A ->" in saida and "Recomendacao conservadora" in saida
