"""Rede de segurança da camada de configuração (guaraci.config_io).

Config é user-facing e crítica: uma regressão em coerção/validação corrompe
SILENCIOSAMENTE os parâmetros de toda corrida (ou faz o pipeline quebrar tarde,
depois de minutos processando). Estes testes travam o comportamento de
_coagir_valor / _checar_faixa / _validar_semantico / _fmt_yaml e o roundtrip
salvar_config↔carregar_config.
"""
import pytest

from guaraci.config import Config
from guaraci import config_io as cio


def _spec(**kw):
    """Monta um spec mínimo do _CONFIG_SPEC para exercitar as funções puras."""
    base = {"key": "campo", "attr": "campo", "tipo": "str", "opcoes": None}
    base.update(kw)
    return base


# ── _checar_faixa ────────────────────────────────────────────────────────────
def test_checar_faixa_dentro_ok():
    assert cio._checar_faixa(_spec(min=1, max=10), 5) == 5


def test_checar_faixa_abaixo_do_min_levanta():
    with pytest.raises(ValueError, match="minimo"):
        cio._checar_faixa(_spec(min=1), 0)


def test_checar_faixa_acima_do_max_levanta():
    with pytest.raises(ValueError, match="maximo"):
        cio._checar_faixa(_spec(max=10), 11)


def test_checar_faixa_sem_limites_passa():
    assert cio._checar_faixa(_spec(), -999.0) == -999.0


def test_checar_faixa_nos_limites_e_inclusivo():
    s = _spec(min=0.0, max=0.5)
    assert cio._checar_faixa(s, 0.0) == 0.0
    assert cio._checar_faixa(s, 0.5) == 0.5


# ── _coagir_valor: bool ──────────────────────────────────────────────────────
@pytest.mark.parametrize("entrada,esperado", [
    (True, True), (False, False),
    ("true", True), ("Sim", True), ("1", True), ("s", True), ("v", True), ("yes", True),
    ("false", False), ("nao", False), ("0", False), ("qualquer", False),
])
def test_coagir_bool(entrada, esperado):
    assert cio._coagir_valor(_spec(tipo="bool"), entrada) is esperado


# ── _coagir_valor: int / float (com faixa) ───────────────────────────────────
def test_coagir_int_ok():
    assert cio._coagir_valor(_spec(tipo="int", min=1, max=40), "12") == 12


def test_coagir_int_fora_da_faixa_levanta():
    with pytest.raises(ValueError):
        cio._coagir_valor(_spec(tipo="int", min=1, max=40), 99)


def test_coagir_float_ok():
    assert cio._coagir_valor(_spec(tipo="float", min=0.0, max=0.5), "0.2") == 0.2


def test_coagir_int_nao_numerico_levanta():
    with pytest.raises(ValueError):
        cio._coagir_valor(_spec(tipo="int"), "abc")


# ── _coagir_valor: str_opcional ──────────────────────────────────────────────
def test_coagir_str_opcional_vazio_vira_none():
    assert cio._coagir_valor(_spec(tipo="str_opcional"), "   ") is None
    assert cio._coagir_valor(_spec(tipo="str_opcional"), "") is None


def test_coagir_str_opcional_com_valor():
    assert cio._coagir_valor(_spec(tipo="str_opcional"), "  classe ") == "classe"


# ── _coagir_valor: list ──────────────────────────────────────────────────────
def test_coagir_list_de_string_csv():
    assert cio._coagir_valor(_spec(tipo="list"), "Copaiba, Andiroba ,") == ("Copaiba", "Andiroba")


def test_coagir_list_de_lista():
    assert cio._coagir_valor(_spec(tipo="list"), ["A", " B ", ""]) == ("A", "B")


def test_coagir_list_none_vira_tupla_vazia():
    assert cio._coagir_valor(_spec(tipo="list"), None) == ()


# ── _coagir_valor: choice ────────────────────────────────────────────────────
def test_coagir_choice_valido():
    assert cio._coagir_valor(_spec(tipo="choice", opcoes=["N1", "N2", "N3"]), "N2") == "N2"


def test_coagir_choice_invalido_levanta():
    with pytest.raises(ValueError, match="invalido"):
        cio._coagir_valor(_spec(tipo="choice", opcoes=["N1", "N2"]), "N9")


# ── _coagir_valor: preproc (nome amigável ↔ interno) ─────────────────────────
def test_coagir_preproc_nome_amigavel():
    # "MSC+SG+MC" (amigável) → código interno "msc_sg_mc"
    assert cio._coagir_valor(_spec(tipo="preproc"), "MSC+SG+MC") == "msc_sg_mc"


def test_coagir_preproc_codigo_interno_aceito():
    assert cio._coagir_valor(_spec(tipo="preproc"), "msc_sg_mc") == "msc_sg_mc"


def test_coagir_preproc_invalido_levanta():
    with pytest.raises(ValueError, match="pre-processamento"):
        cio._coagir_valor(_spec(tipo="preproc"), "inexistente")


# ── _fmt_yaml ────────────────────────────────────────────────────────────────
def test_fmt_yaml_bool():
    assert cio._fmt_yaml(True) == "true"
    assert cio._fmt_yaml(False) == "false"


def test_fmt_yaml_lista():
    assert cio._fmt_yaml(["a", "b"]) == "[a, b]"


def test_fmt_yaml_caminho_windows_usa_aspas_simples():
    # backslash literal exige aspas SIMPLES em YAML (senão \\U vira escape)
    out = cio._fmt_yaml(r"C:\Users\erley\dados")
    assert out.startswith("'") and out.endswith("'")
    assert "\\U" not in out.replace("'", "") or "C:\\Users" in out


def test_fmt_yaml_string_simples_sem_aspas():
    assert cio._fmt_yaml("N1") == "N1"


def test_fmt_yaml_string_reservada_recebe_aspas():
    assert cio._fmt_yaml("true").startswith("'")
    assert cio._fmt_yaml("").startswith("'")


# ── _validar_semantico ───────────────────────────────────────────────────────
def test_validar_semantico_config_padrao_sem_erros():
    assert cio._validar_semantico(Config()) == []


def test_validar_semantico_faixa_invertida():
    cfg = Config()
    cfg.wn_min, cfg.wn_max = 10000.0, 4000.0
    erros = cio._validar_semantico(cfg)
    assert any("Faixa espectral" in e for e in erros)


def test_validar_semantico_holdout_fora_de_faixa():
    cfg = Config()
    cfg.frac_holdout = 0.9
    erros = cio._validar_semantico(cfg)
    assert any("holdout" in e.lower() for e in erros)


# ── roundtrip salvar_config ↔ carregar_config ────────────────────────────────
def test_roundtrip_preserva_valores(tmp_path):
    cfg = Config()
    cfg.nivel = "N2"
    cfg.max_lvs = 25
    cfg.frac_holdout = 0.15
    cfg.preprocessamento_padrao = "snv_sg_mc"
    caminho = str(tmp_path / "config.yaml")

    cio.salvar_config(cfg, caminho)
    lido = cio.carregar_config(caminho)

    assert lido.nivel == "N2"
    assert lido.max_lvs == 25
    assert lido.frac_holdout == 0.15
    assert lido.preprocessamento_padrao == "snv_sg_mc"


def test_carregar_config_inexistente_levanta(tmp_path):
    with pytest.raises(FileNotFoundError):
        cio.carregar_config(str(tmp_path / "nao_existe.yaml"))


def test_carregar_config_ignora_chave_desconhecida(tmp_path):
    caminho = tmp_path / "c.yaml"
    caminho.write_text("chave_inexistente: 123\nnivel: N1\n", encoding="utf-8")
    cfg = cio.carregar_config(str(caminho))
    assert cfg.nivel == "N1"


def test_carregar_config_valor_invalido_reune_erro(tmp_path):
    caminho = tmp_path / "c.yaml"
    # holdout_fracao tem faixa 0..0.5; 2.0 deve ser rejeitado na coerção
    caminho.write_text("holdout_fracao: 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Problemas no config"):
        cio.carregar_config(str(caminho))


# ── _validar_pasta_dados ─────────────────────────────────────────────────────
def test_validar_pasta_sintetico_sempre_ok():
    ok, msg = cio._validar_pasta_dados(Config(modo="sintetico"))
    assert ok and "sintetico" in msg


def test_validar_pasta_csv_inexistente():
    cfg = Config(modo="csv")
    cfg.arquivo_csv = "/caminho/que/nao/existe.csv"
    ok, _ = cio._validar_pasta_dados(cfg)
    assert not ok


def test_validar_pasta_dx_inexistente():
    cfg = Config(modo="dx")
    cfg.pasta_entrada = "/pasta/que/nao/existe"
    ok, _ = cio._validar_pasta_dados(cfg)
    assert not ok


def test_validar_pasta_dx_conta_arquivos(tmp_path):
    (tmp_path / "a.dx").write_text("x")
    (tmp_path / "b.dx").write_text("x")
    cfg = Config(modo="dx")
    cfg.pasta_entrada = str(tmp_path)
    ok, msg = cio._validar_pasta_dados(cfg)
    assert ok and "2" in msg


def test_validar_pasta_dx_pasta_vazia_falha(tmp_path):
    cfg = Config(modo="dx")
    cfg.pasta_entrada = str(tmp_path)  # existe mas sem .dx
    ok, _ = cio._validar_pasta_dados(cfg)
    assert not ok


def test_validar_pasta_csv_existente_ok(tmp_path):
    csv = tmp_path / "dados.csv"
    csv.write_text("a,b\n1,2\n")
    cfg = Config(modo="csv")
    cfg.arquivo_csv = str(csv)
    ok, msg = cio._validar_pasta_dados(cfg)
    assert ok and "dados.csv" in msg


def test_validar_pasta_imagem_conta_imagens(tmp_path):
    (tmp_path / "a.png").write_text("x")
    (tmp_path / "b.jpg").write_text("x")
    cfg = Config(modo="imagem")
    cfg.pasta_entrada = str(tmp_path)
    ok, msg = cio._validar_pasta_dados(cfg)
    assert ok and "2" in msg


def test_validar_pasta_imagem_sem_imagens_falha(tmp_path):
    (tmp_path / "leiame.txt").write_text("x")
    cfg = Config(modo="imagem")
    cfg.pasta_entrada = str(tmp_path)
    ok, _ = cio._validar_pasta_dados(cfg)
    assert not ok


# ── robustez: _validar_semantico com valores não-numéricos ───────────────────
def test_validar_semantico_wn_nao_numerico_nao_quebra():
    cfg = Config()
    cfg.wn_min = "abc"  # config.yaml editado à mão com lixo
    # não deve lançar — o try/except engole e segue (retorna lista, possivelmente vazia)
    assert isinstance(cio._validar_semantico(cfg), list)


def test_validar_semantico_holdout_nao_numerico_nao_quebra():
    cfg = Config()
    cfg.frac_holdout = "xyz"
    assert isinstance(cio._validar_semantico(cfg), list)
