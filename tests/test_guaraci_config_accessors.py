"""Rede de segurança dos acessores de config da CLI (guaraci.py): _get_val/
_set_val/_cfgv/_rotulo_opcao/_risco_hex/_risco_icon/_fmt_bool.

Estas funções gatekeepeiam o que o operador VÊ e EDITA no menu numerado do
CLI — uma regressão aqui mostra ou grava o campo errado silenciosamente
(complementa tests/test_guaraci_cli.py, que já cobre o caso especial
modo_ddsimca; aqui cobrimos o caminho comum + os demais casos especiais).
"""
import pytest


@pytest.fixture(scope="module")
def g():
    import guaraci.guaraci as mod
    return mod


# ── _get_val / _set_val: caminho comum (sem alias especial) ─────────────────
def test_get_val_campo_comum_le_atributo_real(g):
    cfg = g.Config()
    cfg.max_lvs = 25
    assert g._get_val(cfg, "max_lvs") == 25


def test_set_val_campo_comum_grava_atributo_real(g):
    cfg = g.Config()
    g._set_val(cfg, "max_lvs", "17")
    assert cfg.max_lvs == 17


def test_get_val_chave_desconhecida_usa_getattr_bruto(g):
    cfg = g.Config()
    assert g._get_val(cfg, "chave_totalmente_inexistente") == "?"


def test_set_val_chave_desconhecida_levanta_keyerror(g):
    cfg = g.Config()
    with pytest.raises(KeyError):
        g._set_val(cfg, "chave_totalmente_inexistente", "1")


# ── _rotulo_opcao: caso "nivel" (nome amigável lidera; N1/N2/N3 e' referencia
# tecnica secundaria — P8, aposentar o codigo interno como termo PRIMARIO) ────
def test_rotulo_opcao_nivel_lidera_com_nome_amigavel(g):
    rotulo = g._rotulo_opcao("nivel", "N1")
    assert rotulo.startswith("Classificacao")   # nome amigavel e' o rotulo principal
    assert rotulo.endswith("(N1)")              # codigo interno so' como referencia
    assert "N1" in rotulo and len(rotulo) > len("N1")  # tem o nome anexado


def test_rotulo_opcao_nivel_desconhecido_devolve_so_o_codigo(g):
    assert g._rotulo_opcao("nivel", "N9") == "N9"


def test_rotulo_opcao_campo_sem_alias_especial_devolve_str(g):
    assert g._rotulo_opcao("max_lvs", 30) == "30"


def test_get_val_nivel_usa_rotulo_opcao(g):
    cfg = g.Config()
    cfg.nivel = "N2"
    assert g._get_val(cfg, "nivel") == g._rotulo_opcao("nivel", "N2")


# ── _cfgv: resolve key -> attr real (evita getattr direto errado) ────────────
def test_cfgv_resolve_attr_diferente_da_key(g):
    # key "benchmark" no _CONFIG_SPEC mapeia para o atributo executar_benchmark
    cfg = g.Config()
    cfg.executar_benchmark = True
    assert g._cfgv(cfg, "benchmark") is True


def test_cfgv_key_sem_spec_cai_no_default(g):
    cfg = g.Config()
    assert g._cfgv(cfg, "chave_inexistente", default="fallback") == "fallback"


def test_cfgv_default_none_por_padrao(g):
    cfg = g.Config()
    assert g._cfgv(cfg, "chave_inexistente") is None


# ── _risco_hex / _risco_icon: classificação de risco por campo ──────────────
def test_risco_icon_visual_analitico_avancado_sao_distintos(g):
    # uma chave conhecida de cada categoria (RISK_CLASS) deve mapear pro
    # icone certo, e as 3 categorias tem icones DIFERENTES entre si.
    icone_visual    = g._risco_icon("dpi")               # VISUAL
    icone_analitico = g._risco_icon("max_lvs")            # ANALITICO
    icone_avancado  = g._risco_icon("benchmark")          # AVANCADO
    assert icone_visual == "●"
    assert icone_analitico == "◆"
    assert icone_avancado == "▲"
    assert len({icone_visual, icone_analitico, icone_avancado}) == 3


def test_risco_hex_chave_desconhecida_usa_default_analitico(g):
    # chave fora de RISK_CLASS cai no default "ANALITICO"
    assert g._risco_hex("campo_que_nao_existe_no_risk_class") == g._RISK_HEX["ANALITICO"]


def test_risco_icon_chave_desconhecida_usa_default_diamante(g):
    assert g._risco_icon("campo_que_nao_existe_no_risk_class") == "◆"


# ── _fmt_bool ────────────────────────────────────────────────────────────────
def test_fmt_bool_true_false_pt(g):
    assert "Sim" in g._fmt_bool(True, "PT")
    assert "Nao" in g._fmt_bool(False, "PT") or "Não" in g._fmt_bool(False, "PT")


def test_fmt_bool_true_false_en(g):
    assert "Yes" in g._fmt_bool(True, "EN")
    assert "No" in g._fmt_bool(False, "EN")


def test_fmt_bool_nao_booleano_vira_str(g):
    assert g._fmt_bool(42, "EN") == "42"


def test_fmt_bool_nao_booleano_escapa_markup_rich(g):
    # colchetes sao a sintaxe de markup do Rich ([red]...[/red]) -- precisam
    # vir escapados (\\[) para nao serem interpretados como cor/estilo.
    out = g._fmt_bool("[red]perigoso[/red]", "EN")
    assert out == "\\[red]perigoso\\[/red]"


# ── _nome_campo ──────────────────────────────────────────────────────────────
def test_nome_campo_existente_retorna_rotulo_bilingue(g):
    nome_pt = g._nome_campo("nivel")
    assert isinstance(nome_pt, str) and nome_pt != ""


def test_nome_campo_desconhecido_devolve_a_propria_key(g):
    assert g._nome_campo("chave_sem_rotulo_cadastrado") == "chave_sem_rotulo_cadastrado"
