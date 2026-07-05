"""Rede de segurança do auto-ajuste por hardware (guaraci.hardware).

auto_ajustar_config_hardware decide, por faixa de RAM livre, o que DESLIGAR
(SHAP/benchmark/Monte Carlo) para não travar máquinas modestas nem o demo
hospedado. Uma regressão de limiar aqui = freeze silencioso em produção.
Testável por completo passando um `hw` dict sintético (sem depender da RAM
real da máquina de teste).
"""
from guaraci.config import Config
from guaraci import hardware


def _cfg_pesado():
    """Config com TODAS as operações pesadas ligadas, para ver o que é cortado."""
    c = Config()
    c.executar_shap = True
    c.executar_benchmark = True
    c.executar_monte_carlo = True
    c.monte_carlo_incluir_todos = True
    c.n_splits_cv = 5
    c.n_monte_carlo = 200
    c.shap_max_amostras = 500
    return c


def _hw(ram_gb):
    return {"ram_livre_gb": ram_gb}


# ── faixa crítica (< 2 GB): desliga tudo pesado ──────────────────────────────
def test_ram_critica_desliga_tudo():
    c = _cfg_pesado()
    avisos = hardware.auto_ajustar_config_hardware(c, _hw(1.5))
    assert c.executar_shap is False
    assert c.executar_benchmark is False
    assert c.executar_monte_carlo is False
    assert c.n_splits_cv == 3
    assert len(avisos) >= 4


# ── faixa baixa (2–4 GB): SHAP e benchmark off, MC limitado ──────────────────
def test_ram_baixa_desliga_shap_benchmark_limita_mc():
    c = _cfg_pesado()
    hardware.auto_ajustar_config_hardware(c, _hw(3.0))
    assert c.executar_shap is False
    assert c.executar_benchmark is False
    assert c.executar_monte_carlo is True     # MC segue ligado, só limitado
    assert c.n_monte_carlo == 30


# ── faixa média (4–6 GB): reduz amostragem, MC multi-modelo off ──────────────
def test_ram_media_reduz_shap_e_mc():
    c = _cfg_pesado()
    hardware.auto_ajustar_config_hardware(c, _hw(5.0))
    assert c.executar_shap is True            # SHAP fica, com amostragem menor
    assert c.shap_max_amostras == 150
    assert c.n_monte_carlo == 60
    assert c.monte_carlo_incluir_todos is False


# ── faixa moderada (6–8 GB): reduções brandas ────────────────────────────────
def test_ram_moderada_reducoes_brandas():
    c = _cfg_pesado()
    hardware.auto_ajustar_config_hardware(c, _hw(7.0))
    assert c.shap_max_amostras == 300
    assert c.n_monte_carlo == 80
    assert c.executar_benchmark is True       # benchmark permitido nesta faixa


# ── RAM suficiente (>= 8 GB): nada muda ──────────────────────────────────────
def test_ram_suficiente_nao_altera_nada():
    c = _cfg_pesado()
    avisos = hardware.auto_ajustar_config_hardware(c, _hw(16.0))
    assert avisos == []
    assert c.executar_shap is True
    assert c.executar_benchmark is True
    assert c.executar_monte_carlo is True
    assert c.n_monte_carlo == 200
    assert c.shap_max_amostras == 500


def test_hw_sem_chave_ram_usa_default_seguro():
    # hw dict vazio -> assume 16 GB (default) -> não corta nada
    c = _cfg_pesado()
    avisos = hardware.auto_ajustar_config_hardware(c, {})
    assert avisos == []


def test_config_leve_nao_gera_avisos_em_ram_baixa():
    # Se nada pesado está ligado, mesmo em RAM crítica não há o que cortar.
    c = Config()
    c.executar_shap = False
    c.executar_benchmark = False
    c.executar_monte_carlo = False
    c.n_splits_cv = 3
    avisos = hardware.auto_ajustar_config_hardware(c, _hw(1.0))
    assert avisos == []


# ── hardware_probe / _verificar_ram (fail-safe) ──────────────────────────────
def test_hardware_probe_retorna_chaves_esperadas():
    hw = hardware.hardware_probe()
    for chave in ("ram_total_gb", "ram_livre_gb", "cpu_fisicos",
                  "cpu_logicos", "disco_livre_gb", "psutil_ok"):
        assert chave in hw


def test_verificar_ram_zero_sempre_ok():
    # 0 GB exigido: sempre há memória suficiente (ou psutil ausente = fail-safe True)
    assert hardware._verificar_ram(0.0, "op trivial") is True


def test_verificar_ram_absurdo_retorna_false_ou_failsafe():
    # Exigir 1e9 GB: se psutil presente, False; se ausente, True (fail-safe).
    # Nunca lança exceção — é a garantia principal da função.
    resultado = hardware._verificar_ram(1e9, "op impossivel")
    assert resultado in (True, False)
