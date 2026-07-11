"""Testes da lógica pura extraída da CLI de terminal (guaraci.cli_logic, item 19).

Estas funções não dependem de Rich/console/input, então são testáveis em
isolamento — mesmo objetivo do test_app_logic.py para a UI web.
"""

from guaraci.cli_logic import (
    trunc, truncar_desc_por_frase, fmt_bool, validar_faixas, contar_dx,
)


# ── trunc ─────────────────────────────────────────────────────────────────
def test_trunc_string_curta_nao_corta():
    assert trunc("abc", 10) == "abc"


def test_trunc_corta_em_borda_de_palavra():
    assert trunc("uma frase razoavelmente longa aqui", 15) == "uma frase…"


def test_trunc_sem_espaco_corta_no_limite():
    assert trunc("umapalavramuitogrande", 10) == "umapalavr…"


def test_trunc_limite_exato_nao_corta():
    assert trunc("exato", 5) == "exato"


# ── truncar_desc_por_frase ───────────────────────────────────────────────────
def test_truncar_desc_vazia():
    assert truncar_desc_por_frase("", 42) == ""
    assert truncar_desc_por_frase(None, 42) == ""


def test_truncar_desc_usa_primeira_frase_se_couber():
    assert truncar_desc_por_frase("Primeira frase. Segunda frase mais longa.", 42) == "Primeira frase."


def test_truncar_desc_cabe_inteira():
    assert truncar_desc_por_frase("curta", 42) == "curta"


def test_truncar_desc_corta_em_palavra_sem_ponto_proximo():
    desc = "uma descricao razoavelmente longa sem pontuacao no comeco dela"
    out = truncar_desc_por_frase(desc, 20)
    assert out.endswith("…")
    assert len(out) <= 21  # 20 + reticencias
    assert not out[:-1].endswith(" ")  # sem espaco colado nas reticencias


# ── fmt_bool ─────────────────────────────────────────────────────────────────
def test_fmt_bool_pt():
    assert fmt_bool(True, "PT") == "[g]Sim[/g]"
    assert fmt_bool(False, "PT") == "[m]Nao[/m]"


def test_fmt_bool_en():
    assert fmt_bool(True, "EN") == "[g]Yes[/g]"
    assert fmt_bool(False, "EN") == "[m]No[/m]"


def test_fmt_bool_nao_booleano_retorna_str():
    assert fmt_bool("abc", "PT") == "abc"
    assert fmt_bool(42, "EN") == "42"


# ── validar_faixas ───────────────────────────────────────────────────────────
def test_validar_faixas_valida_sem_avisos():
    assert validar_faixas(4000, 10000) == []


def test_validar_faixas_min_maior_que_max():
    avisos = validar_faixas(10000, 4000)
    assert len(avisos) == 1
    assert "invalido" in avisos[0]


def test_validar_faixas_iguais_e_invalido():
    avisos = validar_faixas(5000, 5000)
    assert len(avisos) == 1


# ── contar_dx ────────────────────────────────────────────────────────────────
def test_contar_dx_pasta_inexistente():
    assert contar_dx("/caminho/que/nao/existe/nunca") == 0


def test_contar_dx_arquivos_na_raiz(tmp_path):
    (tmp_path / "a.dx").write_text("x")
    (tmp_path / "b.dx").write_text("x")
    (tmp_path / "c.txt").write_text("x")
    assert contar_dx(str(tmp_path)) == 2


def test_contar_dx_um_nivel_abaixo(tmp_path):
    sub1 = tmp_path / "especie1"
    sub2 = tmp_path / "especie2"
    sub1.mkdir(); sub2.mkdir()
    (sub1 / "a.dx").write_text("x")
    (sub2 / "b.dx").write_text("x")
    (sub2 / "c.dx").write_text("x")
    assert contar_dx(str(tmp_path)) == 3


def test_contar_dx_prioriza_raiz_sobre_subpastas(tmp_path):
    (tmp_path / "raiz.dx").write_text("x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "outro.dx").write_text("x")
    # se ha .dx na raiz, subpastas nao sao somadas (layout "flat")
    assert contar_dx(str(tmp_path)) == 1
