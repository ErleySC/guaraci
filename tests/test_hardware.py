"""Rede de segurança do auto-ajuste por hardware (guaraci.hardware).

auto_adjust_hardware_config decide, por faixa de RAM livre, o que DESLIGAR
(SHAP/benchmark/Monte Carlo) para não travar máquinas modestas nem o demo
hospedado. Uma regressão de limiar aqui = freeze silencioso em produção.
Testável por completo passando um `hw` dict sintético (sem depender da RAM
real da máquina de teste).
"""
import builtins
import os

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
    c.run_shap = True
    c.run_benchmark = True
    c.run_monte_carlo = True
    c.monte_carlo_include_all = True
    c.n_splits_cv = 5
    c.n_monte_carlo = 200
    c.shap_max_samples = 500
    return c


def _hw(ram_gb):
    return {"ram_livre_gb": ram_gb}


# M-03 (parecer 2026-09-25): o hardware NUNCA altera parametro cientifico.
# Antes, n_splits_cv caia para 3 com RAM < 2 GB e a metrica publicada mudava.
_CIENTIFICOS = ("n_splits_cv", "n_permutations", "n_repeats_cv",
                "n_bootstrap_bca", "max_lvs", "seed", "frac_holdout",
                "n_monte_carlo")


def _cientificos(c):
    return {k: getattr(c, k) for k in _CIENTIFICOS}


# ── RAM < 4 GB: extras pesados DESLIGADOS (nunca "pela metade") ──────────────
def test_ram_critica_desliga_extras_e_nao_toca_na_ciencia():
    c = _cfg_pesado()
    antes = _cientificos(c)
    avisos = hardware.auto_adjust_hardware_config(c, _hw(1.5))
    assert c.run_shap is False and c.run_benchmark is False
    assert c.run_monte_carlo is False
    assert _cientificos(c) == antes          # n_splits_cv segue 5, MC segue 200
    assert c.n_splits_cv == 5
    assert len(avisos) == 4                  # 3 extras + dpi das figuras


def test_ram_baixa_tambem_desliga_e_nunca_reduz_repeticoes():
    c = _cfg_pesado()
    antes = _cientificos(c)
    hardware.auto_adjust_hardware_config(c, _hw(3.0))
    assert c.run_shap is False and c.run_benchmark is False
    assert c.run_monte_carlo is False
    assert _cientificos(c) == antes


# ── 4-6 GB: reduz so' a amostragem do SHAP e o MC multi-modelo ────────────────
def test_ram_media_reduz_shap_e_mc_multimodelo_sem_tocar_na_ciencia():
    c = _cfg_pesado()
    antes = _cientificos(c)
    hardware.auto_adjust_hardware_config(c, _hw(5.0))
    assert c.run_shap is True
    assert c.shap_max_samples == 150
    assert c.monte_carlo_include_all is False
    assert c.run_monte_carlo is True
    assert _cientificos(c) == antes


def test_ram_moderada_reducoes_brandas_so_no_shap():
    c = _cfg_pesado()
    antes = _cientificos(c)
    hardware.auto_adjust_hardware_config(c, _hw(7.0))
    assert c.shap_max_samples == 300
    assert c.run_benchmark is True
    assert _cientificos(c) == antes


# ── RAM suficiente (>= 8 GB): nada muda ──────────────────────────────────────
def test_ram_suficiente_nao_altera_nada():
    c = _cfg_pesado()
    avisos = hardware.auto_adjust_hardware_config(c, _hw(16.0))
    assert avisos == [] and c.ajustes_hardware == []
    assert c.run_shap is True
    assert c.run_benchmark is True
    assert c.run_monte_carlo is True
    assert c.n_monte_carlo == 200
    assert c.shap_max_samples == 500


def test_cada_ajuste_fica_registrado_para_o_model_card_e_o_manifesto():
    c = _cfg_pesado()
    hardware.auto_adjust_hardware_config(c, _hw(3.0))
    nomes = {a["parametro"] for a in c.ajustes_hardware}
    assert nomes == {"run_shap", "run_benchmark", "run_monte_carlo", "save_dpi"}
    for a in c.ajustes_hardware:
        assert {"parametro", "pedido", "efetivo", "motivo"} <= set(a)
        assert "RAM" in a["motivo"]


def test_o_registro_de_ajustes_e_reiniciado_a_cada_chamada():
    c = _cfg_pesado()
    hardware.auto_adjust_hardware_config(c, _hw(1.0))
    assert c.ajustes_hardware
    hardware.auto_adjust_hardware_config(c, _hw(16.0))
    assert c.ajustes_hardware == []


def test_hw_sem_chave_ram_usa_default_seguro():
    # hw dict vazio -> assume 16 GB (default) -> não corta nada
    c = _cfg_pesado()
    avisos = hardware.auto_adjust_hardware_config(c, {})
    assert avisos == []


def test_config_leve_nao_gera_avisos_em_ram_baixa():
    # Se nada pesado está ligado, mesmo em RAM crítica não há o que cortar.
    c = Config()
    c.run_shap = False
    c.run_benchmark = False
    c.run_monte_carlo = False
    c.n_splits_cv = 3
    c.save_dpi = 300
    avisos = hardware.auto_adjust_hardware_config(c, _hw(1.0))
    assert avisos == []


# ── 2026-10-15: dpi e processos paralelos por RAM (nao mudam resultado) ──────
def test_dpi_alto_cai_para_300_com_pouca_ram_e_fica_com_ram_folgada():
    c = Config()
    assert c.save_dpi == 600
    hardware.auto_adjust_hardware_config(c, _hw(5.0))
    assert c.save_dpi == 300
    assert any(a["parametro"] == "save_dpi" for a in c.ajustes_hardware)
    c = Config()
    hardware.auto_adjust_hardware_config(c, _hw(12.0))
    assert c.save_dpi == 600


def test_processos_paralelos_limitados_por_ram_e_nucleos():
    c = Config()
    c.n_jobs_permutation = 8
    antes = _cientificos(c)
    hardware.auto_adjust_hardware_config(c, {"ram_livre_gb": 2.0, "cpu_fisicos": 4})
    assert c.n_jobs_permutation == 2           # (2,0 - 1,0) / 0,4 = 2
    assert _cientificos(c) == antes
    c = Config()
    c.n_jobs_permutation = -1                  # "todos os nucleos"
    hardware.auto_adjust_hardware_config(c, {"ram_livre_gb": 32.0, "cpu_fisicos": 4})
    assert c.n_jobs_permutation == -1          # cabe: nao mexe
    c.n_jobs_permutation = 1
    hardware.auto_adjust_hardware_config(c, {"ram_livre_gb": 1.0, "cpu_fisicos": 4})
    assert c.n_jobs_permutation == 1


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


# ── V86: RAM desconhecida e' None, nunca um numero inventado ─────────────────
def _sem_ctypes_windows(monkeypatch):
    """Torna o teste igual em Windows e Linux: a fonte ctypes nao mede nada."""
    monkeypatch.setattr(hardware, "_ram_por_ctypes_windows", lambda: (None, None))


def test_sem_psutil_a_ram_vem_do_proc_meminfo(monkeypatch, tmp_path):
    """psutil ausente + /proc/meminfo falso (7 GB totais, 5 GB disponiveis):
    os valores LIDOS aparecem, nao os padroes 4,0/2,0 do codigo antigo."""
    _bloquear_psutil(monkeypatch)
    _sem_ctypes_windows(monkeypatch)
    falso = tmp_path / "meminfo"
    falso.write_text("MemTotal:        7340032 kB\nMemFree:          100000 kB\n"
                     "MemAvailable:    5242880 kB\nBuffers:           1000 kB\n",
                     encoding="ascii")
    monkeypatch.setattr(hardware, "_CAMINHO_MEMINFO", str(falso))
    hw = hardware.hardware_probe()
    assert hw["psutil_ok"] is False
    assert hw["ram_total_gb"] == 7.0 and hw["ram_livre_gb"] == 5.0
    assert hw["ram_fonte"] == "meminfo"


def test_sem_psutil_e_sem_meminfo_usa_sysconf(monkeypatch, tmp_path):
    _bloquear_psutil(monkeypatch)
    _sem_ctypes_windows(monkeypatch)
    monkeypatch.setattr(hardware, "_CAMINHO_MEMINFO", str(tmp_path / "nao_existe"))
    valores = {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 2 * 1024**3 // 4096,
               "SC_AVPHYS_PAGES": 1024**3 // 4096}

    def _sysconf(nome):
        return valores[nome]
    monkeypatch.setattr(os, "sysconf", _sysconf, raising=False)
    hw = hardware.hardware_probe()
    assert hw["ram_total_gb"] == 2.0 and hw["ram_livre_gb"] == 1.0
    assert hw["ram_fonte"] == "sysconf"


def test_sem_nenhuma_fonte_a_ram_e_none_e_os_extras_nao_sao_desligados(monkeypatch, tmp_path):
    _bloquear_psutil(monkeypatch)
    _sem_ctypes_windows(monkeypatch)
    monkeypatch.setattr(hardware, "_CAMINHO_MEMINFO", str(tmp_path / "nao_existe"))

    def _sem_sysconf(nome):
        raise ValueError(nome)
    monkeypatch.setattr(os, "sysconf", _sem_sysconf, raising=False)
    hw = hardware.hardware_probe()
    assert hw["ram_total_gb"] is None and hw["ram_livre_gb"] is None
    assert hw["ram_fonte"] is None
    c = _cfg_pesado()
    avisos = hardware.auto_adjust_hardware_config(c, hw)
    assert avisos == [] and c.ajustes_hardware == []
    assert c.run_shap and c.run_benchmark and c.run_monte_carlo
    assert c.shap_max_samples == 500 and c.monte_carlo_include_all


def test_meminfo_corrompido_nao_inventa_numero(tmp_path):
    ruim = tmp_path / "meminfo"
    ruim.write_text("MemTotal: abc kB\n", encoding="ascii")
    assert hardware._ram_por_meminfo(str(ruim)) == (None, None)


def test_model_card_diz_ram_nao_medida_quando_o_probe_nao_leu(tmp_path):
    from guaraci import resultados_io as rio
    hw = {"ram_total_gb": None, "ram_livre_gb": None, "cpu_fisicos": 4, "cpu_logicos": 8}
    rio.generate_model_card(str(tmp_path), Config(level="N2"), {"Total samples": 10}, hw, ["A"])
    txt = (tmp_path / "model_card.md").read_text(encoding="utf-8")
    assert "RAM nao medida" in txt
    assert "None GB" not in txt


def test_ajustes_de_hardware_vao_para_o_mesmo_json_dos_ajustes_automaticos(tmp_path):
    import json
    from guaraci import contexto
    c = _cfg_pesado()
    hardware.auto_adjust_hardware_config(c, _hw(1.0))
    contexto.gravar_efetiva(c, str(tmp_path), str(tmp_path))
    dados = json.loads((tmp_path / "ajustes_automaticos.json").read_text(encoding="utf-8"))
    nomes = {a["parametro"] for a in dados["ajustes_hardware"]}
    assert {"run_shap", "run_benchmark", "run_monte_carlo"} <= nomes
