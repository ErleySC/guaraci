"""As nove guardas do ANOVA que a mutacao mostrou inalcancaveis foram REMOVIDAS (commit cae6619). Cada teste abaixo prova, para uma delas,
QUAL verificacao anterior recusa a entrada que a guarda cobriria -- e qual e' a mensagem que o usuario recebe. Se a verificacao anterior
sumir, a guarda volta a ser necessaria e estes testes caem.

Guardas removidas (anova_grupos.py / anova_cli.py):
  G1 amplitude zero da figura (`folga = (topo - base) * 0.06 or 0.1`)      -> dados constantes recusados antes da figura
  G2 erro-padrao do Tukey-Kramer igual a zero (`errstate` + `isfinite`)      -> variancia dentro zero recusada antes
  G3 p do Tukey-Kramer com q nao finito (`np.where(isfinite(q), ...)`)       -> idem
  G4 `qm_d > 0` antes de F = QM_entre / QM_dentro                           -> idem
  G5 `np.isfinite(F)` antes do p do F                                        -> idem
  G6 `sst > 0` em eta2                                                       -> SQ total >= SQ dentro > 0
  G7 `sst > 0` em omega2                                                     -> idem
  G8 `3 <= len(residuos)` no quantil-quantil/Shapiro                         -> minimo de 2 grupos com >= 2 unidades (>= 4 residuos)
  G9 `np.ptp(residuos) > 0` no Shapiro                                       -> residuos constantes = variancia dentro zero, recusada antes
"""
from __future__ import annotations

import csv
import math

import pytest

from guaraci import anova_cli
from guaraci.anova_grupos import anova_uma_via, diagnosticar_premissas, kruskal_dunn

MENSAGEM_VARIANCIA_ZERO = "variancia dentro dos grupos e' zero"
MENOS_DE_DOIS_GRUPOS = "menos de 2 grupos com >= 2 unidades independentes"


def _minimo():
    """Menor desenho aceito: 2 grupos x 2 unidades."""
    return [1.0, 2.0, 3.0, 4.0], ["A", "A", "B", "B"], ["a", "b", "c", "d"]


CONSTANTES_POR_GRUPO = [
    ([1.0, 1.0, 2.0, 2.0], "dentro zero em todos os grupos"),
    ([5.0, 5.0, 5.0, 5.0], "todos os valores iguais"),
]


@pytest.mark.parametrize("y, descricao", CONSTANTES_POR_GRUPO)
def test_G2_G3_G4_G5_variancia_dentro_zero_e_recusada_antes_de_qualquer_divisao(y, descricao):
    _y, g, u = _minimo()
    with pytest.raises(ValueError) as e:
        anova_uma_via(y, g, u, n_boot=0)
    assert MENSAGEM_VARIANCIA_ZERO in str(e.value), descricao
    assert "teste F nao e' definido" in str(e.value)


def test_G2_G3_o_que_passa_da_recusa_tem_erro_padrao_positivo_e_p_finito():
    y, g, u = _minimo()
    r = anova_uma_via(y, g, u, n_boot=0)
    for c in r.comparacoes:
        assert math.isfinite(c["p_ajustado"]) and 0.0 <= c["p_ajustado"] <= 1.0


def test_G4_G5_F_e_p_do_menor_desenho_sao_finitos():
    y, g, u = _minimo()
    r = anova_uma_via(y, g, u, n_boot=0)
    assert math.isfinite(r.p) and 0.0 < r.p < 1.0


def test_G6_G7_eta2_e_omega2_do_menor_desenho_sao_finitos_e_no_intervalo():
    y, g, u = _minimo()
    r = anova_uma_via(y, g, u, n_boot=0)
    assert math.isfinite(r.eta2) and 0.0 <= r.eta2 <= 1.0
    assert math.isfinite(r.omega2) and r.omega2 <= 1.0


def test_G8_menos_de_dois_grupos_com_duas_unidades_e_recusado_antes_dos_residuos():
    with pytest.raises(ValueError) as e:
        anova_uma_via([1.0, 2.0], ["A", "B"], ["a", "b"], n_boot=0)
    assert MENOS_DE_DOIS_GRUPOS in str(e.value)
    with pytest.raises(ValueError) as e2:
        diagnosticar_premissas([1.0, 2.0], ["A", "B"], ["a", "b"])
    assert MENOS_DE_DOIS_GRUPOS in str(e2.value)


def test_G8_o_menor_desenho_aceito_ja_tem_pelo_menos_quatro_residuos_e_o_shapiro_roda():
    y, g, u = _minimo()
    d = diagnosticar_premissas(y, g, u)
    assert len(d["residuos"]) >= 4 and d["shapiro_p"] is not None and math.isfinite(d["shapiro_p"])


def test_G9_residuos_constantes_equivalem_a_variancia_dentro_zero_e_ja_foram_recusados():
    # grupos constantes, residuos todos zero: o ANOVA recusa antes de chegar ao Shapiro
    with pytest.raises(ValueError) as e:
        anova_uma_via([3.0, 3.0, 7.0, 7.0], ["A", "A", "B", "B"], ["a", "b", "c", "d"], n_boot=0)
    assert MENSAGEM_VARIANCIA_ZERO in str(e.value)


def test_G1_kruskal_com_todos_os_valores_iguais_e_recusado_com_mensagem():
    y, g, u = [5.0] * 4, ["A", "A", "B", "B"], ["a", "b", "c", "d"]
    with pytest.raises(ValueError) as e:
        kruskal_dunn(y, g, u)
    assert "todos os valores sao iguais" in str(e.value)


def _tabela(caminho, y):
    with open(caminho, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["amostra", "grupo", "teor"])
        for i, (v, grp) in enumerate(zip(y, "AABB")):
            w.writerow([f"s{i}", grp, v])


@pytest.mark.parametrize("lang, trecho", [("pt", "variancia dentro dos grupos e' zero"), ("en", "within-group variance is zero")])
def test_G1_a_cli_recusa_dados_constantes_com_mensagem_ao_usuario_e_sem_figura(tmp_path, capsys, lang, trecho):
    """O usuario ve a recusa (PT ou EN), nunca uma figura com eixo de amplitude zero."""
    t = tmp_path / "t.csv"
    _tabela(t, [5, 5, 5, 5])
    rc = anova_cli.executar([str(t), "--resposta", "teor", "--grupo", "grupo", "--unidade", "amostra", "--sem-bootstrap",
                             "--saida", str(tmp_path / "r"), "--idioma", lang])
    err = capsys.readouterr().err
    assert rc == 2 and trecho in err, err
    assert not (tmp_path / "r" / "figuras").exists()
