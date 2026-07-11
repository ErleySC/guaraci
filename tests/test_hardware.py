"""Rede de segurança do auto-ajuste por hardware (guaraci.hardware).

auto_ajustar_config_hardware decide, por faixa de RAM livre, o que DESLIGAR
(SHAP/benchmark/Monte Carlo) para não travar máquinas modestas nem o demo
hospedado. Uma regressão de limiar aqui = freeze silencioso em produção.
Testável por completo passando um `hw` dict sintético (sem depender da RAM
real da máquina de teste).
"""
import builtins

from guaraci.config import Config
from guaraci import hardware


def _bloquear_psutil(monkeypatch):
    """Força ImportError em `import psutil`, exercitando os caminhos de
    fallback (ctypes/os.cpu_count) de hardware_probe/_verificar_ram sem
    depender de psutil estar de fato ausente no ambiente de teste."""
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("psutil bloqueado no teste")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", _fake_import)


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


# ── Fallback sem psutil (ctypes/os.cpu_count) ────────────────────────────────

def test_hardware_probe_sem_psutil_usa_fallback_ctypes(monkeypatch):
    """Sem psutil, hardware_probe() ainda retorna todas as chaves esperadas
    (via GlobalMemoryStatusEx no Windows / os.cpu_count), com psutil_ok=False."""
    _bloquear_psutil(monkeypatch)
    hw = hardware.hardware_probe()
    assert hw["psutil_ok"] is False
    for chave in ("ram_total_gb", "ram_livre_gb", "cpu_fisicos",
                  "cpu_logicos", "disco_livre_gb"):
        assert chave in hw
    assert hw["cpu_logicos"] >= 1
    assert hw["cpu_fisicos"] >= 1


def test_verificar_ram_sem_psutil_e_fail_safe(monkeypatch):
    """Sem psutil, _verificar_ram nunca lança e assume que há memória
    suficiente (fail-safe True), mesmo exigindo uma quantidade absurda."""
    _bloquear_psutil(monkeypatch)
    assert hardware._verificar_ram(1e9, "op impossivel") is True


def test_hardware_probe_disco_livre_com_disk_usage_falhando(monkeypatch):
    """Se psutil.disk_usage() lançar, hardware_probe() não quebra e mantém
    o default conservador de disco_livre_gb (guard de exceção isolado)."""
    import psutil as _ps

    def _quebra(*a, **k):
        raise OSError("disco inacessivel")
    monkeypatch.setattr(_ps, "disk_usage", _quebra)
    hw = hardware.hardware_probe()
    assert hw["psutil_ok"] is True
    assert hw["disco_livre_gb"] == 5.0


# ── cgroup (containers Docker/Streamlit Cloud/Kubernetes) ───────────────────
def test_cgroup_ram_limit_ausente_retorna_none(monkeypatch):
    """Sem arquivos de cgroup (ambiente sem container, ex.: Windows/macOS
    bare metal), retorna None e nao afeta o resultado do psutil."""
    def _sem_arquivo(*a, **k):
        raise FileNotFoundError()
    monkeypatch.setattr("builtins.open", _sem_arquivo)
    assert hardware._cgroup_ram_limit_gb() is None


def test_cgroup_ram_limit_sentinela_sem_limite_retorna_none(monkeypatch):
    """cgroup v2 com 'max' (sem limite real) e cgroup v1 com o sentinela de
    ~2^63 bytes (praticamente ilimitado) sao ambos ignorados."""
    import builtins
    _real_open = builtins.open

    def _fake_open(caminho, *a, **k):
        if caminho == "/sys/fs/cgroup/memory.max":
            import io
            return io.StringIO("max")
        raise FileNotFoundError()
    monkeypatch.setattr("builtins.open", _fake_open)
    assert hardware._cgroup_ram_limit_gb() is None


def test_cgroup_ram_limit_com_valor_real_retorna_gb(monkeypatch):
    """cgroup v2 com limite real (ex.: 2GB, tipico de containers de nuvem
    gratuitos) e convertido corretamente para GB."""
    def _fake_open(caminho, *a, **k):
        if caminho == "/sys/fs/cgroup/memory.max":
            import io
            return io.StringIO(str(2 * 1024**3))  # 2 GiB
        raise FileNotFoundError()
    monkeypatch.setattr("builtins.open", _fake_open)
    assert hardware._cgroup_ram_limit_gb() == 2.0


def test_hardware_probe_usa_limite_de_cgroup_quando_menor_que_host(monkeypatch):
    """Caso real do bug reportado: psutil.virtual_memory() reporta a RAM do
    HOST (ex.: 128GB, maquina fisica compartilhada de um provedor de nuvem),
    mas o container so tem 2GB alocados via cgroup. hardware_probe() deve
    reportar o limite do container, nao o do host, e marcar a flag."""
    import psutil as _ps

    class _MemFake:
        total = 128 * 1024**3       # 128 GiB do host
        available = 100 * 1024**3   # 100 GiB "livres" do host (irreal p/ o container)

    monkeypatch.setattr(_ps, "virtual_memory", lambda: _MemFake())
    monkeypatch.setattr(hardware, "_cgroup_ram_limit_gb", lambda: 2.0)

    hw = hardware.hardware_probe()
    assert hw["ram_total_gb"] == 2.0
    assert hw["ram_limitada_por_container"] is True
    assert hw["ram_livre_gb"] <= 2.0


def test_hardware_probe_ignora_cgroup_quando_maior_que_host(monkeypatch):
    """Se o valor lido do cgroup for MAIOR que a RAM total do host (cgroup
    nao esta de fato limitando, ou leitura invalida), o total do host
    prevalece e a flag de container permanece False."""
    monkeypatch.setattr(hardware, "_cgroup_ram_limit_gb", lambda: 999.0)
    hw = hardware.hardware_probe()
    assert hw["ram_limitada_por_container"] is False
