"""
hardware.py — Deteccao e adaptacao a hardware (RAM/CPU/disco): probe, auto-
ajuste de Config conforme recursos, e guarda de RAM antes de operacoes pesadas.

Extraido de pipeline.py (Fase H). Config so em type hint (TYPE_CHECKING).
pipeline.py e app_quimiometria.py usam via reexport (pipeline.hardware_probe()).
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from guaraci.pipeline import Config

__all__ = [
    "hardware_probe",
    "auto_adjust_hardware_config",
]


def _cgroup_ram_limit_gb() -> "float | None":
    """
    Le o limite de RAM do cgroup (Linux, containers Docker/Streamlit Cloud/
    Kubernetes), quando existir. psutil.virtual_memory() le /proc/meminfo,
    que reporta a RAM da maquina HOSPEDEIRA fisica, nao a fatia alocada ao
    container — em nuvens compartilhadas isso infla o "Total RAM" exibido
    (ex.: mostra 128GB do host quando o container so tem 1-2GB reservados).
    Retorna None se nao houver cgroup, se o valor for o sentinela de
    "sem limite" (cgroup v1: ~2^63-4096; v2: literal "max"), ou se for maior
    que a RAM total real (nesse caso o cgroup nao esta de fato limitando).
    """
    candidatos = [
        "/sys/fs/cgroup/memory.max",                      # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",     # cgroup v1
    ]
    for caminho in candidatos:
        try:
            with open(caminho, "r", encoding="ascii") as f:
                bruto = f.read().strip()
            if bruto == "max":
                continue
            limite_bytes = int(bruto)
            if limite_bytes <= 0 or limite_bytes >= 2**62:
                continue  # sentinela de "sem limite"
            return round(limite_bytes / 1024**3, 1)
        except (OSError, ValueError):
            continue
    return None


# Caminho do /proc/meminfo; modulo-nivel para os testes poderem apontar para um arquivo falso.
_CAMINHO_MEMINFO = "/proc/meminfo"


def _ram_por_meminfo(caminho: "str | None" = None) -> "tuple[float | None, float | None]":
    """(total_gb, livre_gb) lidos de /proc/meminfo (MemTotal, MemAvailable).
    Valor ausente ou ilegivel vira None; nunca inventa numero."""
    caminho = caminho or _CAMINHO_MEMINFO
    campos: Dict[str, float] = {}
    try:
        with open(caminho, "r", encoding="ascii", errors="replace") as f:
            for linha in f:
                nome, _, resto = linha.partition(":")
                if nome in ("MemTotal", "MemAvailable"):
                    partes = resto.split()
                    if partes:
                        campos[nome] = float(partes[0]) * 1024.0  # kB -> bytes
    except (OSError, ValueError):
        return None, None
    total = campos.get("MemTotal")
    livre = campos.get("MemAvailable")
    return (round(total / 1024**3, 1) if total else None,
            round(livre / 1024**3, 1) if livre is not None else None)


def _ram_por_sysconf() -> "tuple[float | None, float | None]":
    """(total_gb, livre_gb) por os.sysconf (Linux e macOS). A RAM livre so'
    existe onde SC_AVPHYS_PAGES existe (Linux); no macOS fica None."""
    def _bytes(nome_paginas: str) -> "float | None":
        try:
            sysconf = getattr(os, "sysconf")  # so existe fora do Windows
            pagina = sysconf("SC_PAGE_SIZE")
            paginas = sysconf(nome_paginas)
        except (AttributeError, ValueError, OSError):
            return None
        if pagina <= 0 or paginas <= 0:
            return None
        return float(pagina) * float(paginas)

    total = _bytes("SC_PHYS_PAGES")
    livre = _bytes("SC_AVPHYS_PAGES")
    return (round(total / 1024**3, 1) if total else None,
            round(livre / 1024**3, 1) if livre else None)


def _ram_por_ctypes_windows() -> "tuple[float | None, float | None]":
    """(total_gb, livre_gb) por GlobalMemoryStatusEx; (None, None) fora do Windows."""
    try:
        import ctypes

        class _MEMSTATUS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = _MEMSTATUS()
        ms.dwLength = ctypes.sizeof(ms)
        getattr(ctypes, "windll").kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return (round(ms.ullTotalPhys / 1024**3, 1),
                round(ms.ullAvailPhys / 1024**3, 1))
    except Exception as _e_win:  # noqa: BLE001 -- fallback best-effort; fora
        # do Windows (sem windll) ou com a chamada falhando, a RAM segue
        # DESCONHECIDA (None), nunca um numero padrao.
        logging.getLogger(__name__).debug(
            "hardware_probe: RAM (ctypes) nao detectada: %s", _e_win)
        return None, None


def _ram_sem_psutil() -> "tuple[str | None, float | None, float | None]":
    """Primeira fonte que mede a RAM: ctypes (Windows), /proc/meminfo, sysconf.
    Devolve (fonte, total_gb, livre_gb); (None, None, None) se nenhuma mede."""
    total, livre = _ram_por_ctypes_windows()
    if total is not None:
        return "ctypes", total, livre
    total, livre = _ram_por_meminfo()
    if total is not None:
        fonte = "meminfo"
        if livre is None:
            livre = _ram_por_sysconf()[1]
        return fonte, total, livre
    total, livre = _ram_por_sysconf()
    if total is not None:
        return "sysconf", total, livre
    return None, None, None


def hardware_probe() -> Dict[str, Any]:
    """
    Detecta RAM, CPU e disco disponiveis.
    RAM: psutil; sem ele, GlobalMemoryStatusEx (Windows), /proc/meminfo e
    os.sysconf (Linux/macOS). RAM que nenhuma fonte mede fica None
    (`ram_total_gb`/`ram_livre_gb`) e `ram_fonte` None: NUNCA um numero
    inventado (o padrao antigo de 4/2 GB desligava extras sem motivo e ia
    para o model card como se fosse medicao).
    Nunca lanca excecao.
    """
    info: Dict[str, Any] = {
        "ram_total_gb":  None,  # None = nao medida
        "ram_livre_gb":  None,
        "ram_fonte":     None,
        "cpu_logicos":   2,     # CPU/disco: padroes conservadores
        "cpu_fisicos":   1,
        "disco_livre_gb": 5.0,
        "psutil_ok":     False,
        "ram_limitada_por_container": False,
    }
    try:
        import psutil as _ps
        mem = _ps.virtual_memory()
        info["ram_total_gb"]   = round(mem.total   / 1024**3, 1)
        info["ram_livre_gb"]   = round(mem.available / 1024**3, 1)
        info["ram_fonte"]      = "psutil"
        info["cpu_logicos"]    = _ps.cpu_count(logical=True)  or 2
        info["cpu_fisicos"]    = _ps.cpu_count(logical=False) or 1
        try:
            info["disco_livre_gb"] = round(
                _ps.disk_usage(os.path.abspath(".")).free / 1024**3, 1)
        except Exception as _e_disco:  # noqa: BLE001 -- probe best-effort por
            # contrato da funcao (docstring: "nunca lanca excecao"); mantem o
            # default conservador de disco_livre_gb ja setado acima.
            logging.getLogger(__name__).debug(
                "hardware_probe: disco nao detectado: %s", _e_disco)
        info["psutil_ok"] = True
    except ImportError:
        try:
            info["cpu_logicos"] = os.cpu_count() or 2
            info["cpu_fisicos"] = max(1, (os.cpu_count() or 2) // 2)
        except Exception as _e_cpu:  # noqa: BLE001 -- mesmo contrato
            # best-effort; mantem os defaults conservadores de CPU.
            logging.getLogger(__name__).debug(
                "hardware_probe: CPU nao detectada: %s", _e_cpu)
    except Exception as _e_probe:  # noqa: BLE001 -- ultimo recurso do
        # probe inteiro (docstring: "nunca lanca excecao"); a RAM cai nas
        # fontes sem psutil logo abaixo e, se nenhuma medir, fica None.
        logging.getLogger(__name__).debug(
            "hardware_probe: deteccao de hardware falhou por completo: %s",
            _e_probe)

    if info["ram_total_gb"] is None:
        try:
            fonte, total, livre = _ram_sem_psutil()
            info["ram_fonte"], info["ram_total_gb"], info["ram_livre_gb"] = (
                fonte, total, livre)
        except Exception as _e_fb:  # noqa: BLE001 -- "nunca lanca excecao"
            logging.getLogger(__name__).debug(
                "hardware_probe: fontes de RAM sem psutil falharam: %s", _e_fb)

    try:
        limite_cgroup = _cgroup_ram_limit_gb()
        if (limite_cgroup is not None and info["ram_total_gb"] is not None
                and limite_cgroup < info["ram_total_gb"]):
            # Container com limite real menor que a RAM do host: o total
            # exibido passa a ser o limite do container (o que a analise
            # de fato pode usar), e a RAM livre e recalculada dentro desse
            # teto (a "disponivel" do SO pode ultrapassar o limite do cgroup).
            livre = info["ram_livre_gb"]
            usada_gb = max(0.0, info["ram_total_gb"] - livre) if livre is not None else 0.0
            info["ram_total_gb"] = limite_cgroup
            info["ram_livre_gb"] = round(max(0.0, limite_cgroup - usada_gb), 1)
            info["ram_limitada_por_container"] = True
    except Exception as _e_cg:  # noqa: BLE001 -- "nunca lanca excecao"
        logging.getLogger(__name__).debug(
            "hardware_probe: limite de cgroup nao aplicado: %s", _e_cg)
    return info


def auto_adjust_hardware_config(cfg: "Config",
                                  hw: Dict[str, Any]) -> List[str]:
    """
    Ajusta EXTRAS pesados (SHAP, benchmark, Monte Carlo) conforme a RAM livre,
    para nao travar maquinas com < 8 GB. Retorna as mensagens de aviso.

    REGRA: o hardware NUNCA altera um
    parametro CIENTIFICO. Antes, `n_splits_cv` caia para 3 com < 2 GB de RAM
    e a acuracia balanceada mudava so' porque a maquina estava ocupada, enquanto o
    model card afirmava "nao afeta os resultados". Agora so' se desliga ou se
    reduz o que e' EXTRA (SHAP, benchmark, Monte Carlo) e cada ajuste fica
    registrado em `cfg.ajustes_hardware` (model card e manifesto) e no log.
    `n_splits_cv`, `n_permutations` e o numero de repeticoes nao sao tocados.
    """
    avisos: List[str] = []
    cfg.ajustes_hardware = []
    ram_bruta = hw.get("ram_livre_gb", 16.0)
    if ram_bruta is None:
        # RAM nao medida (nenhuma fonte a le): nao ha base para desligar
        # nada. Os extras pedidos ficam como estao; o chamador registra
        # "RAM nao medida".
        return avisos
    ram = float(ram_bruta)

    def _ajustar(parametro: str, pedido: Any, efetivo: Any, motivo: str) -> None:
        cfg.ajustes_hardware.append({
            "parametro": parametro, "pedido": pedido, "efetivo": efetivo,
            "motivo": motivo})
        avisos.append(f"{parametro}: {pedido} -> {efetivo} ({motivo})")

    if ram < 4.0:
        # RAM < 4 GB: os tres extras pesados ficam DESLIGADOS (nada de
        # "rodar com metade das iteracoes": um resultado parcial dependente
        # da RAM e' pior que um extra ausente e declarado).
        if cfg.run_shap:
            cfg.run_shap = False
            _ajustar("run_shap", True, False, f"RAM livre {ram:.1f} GB < 4 GB")
        if cfg.run_benchmark:
            cfg.run_benchmark = False
            _ajustar("run_benchmark", True, False, f"RAM livre {ram:.1f} GB < 4 GB")
        if cfg.run_monte_carlo:
            cfg.run_monte_carlo = False
            _ajustar("run_monte_carlo", True, False, f"RAM livre {ram:.1f} GB < 4 GB")
    elif ram < 6.0:
        if cfg.run_shap and cfg.shap_max_samples > 150:
            _ajustar("shap_max_samples", cfg.shap_max_samples, 150,
                     f"RAM livre {ram:.1f} GB < 6 GB")
            cfg.shap_max_samples = 150
        if cfg.monte_carlo_include_all:
            cfg.monte_carlo_include_all = False
            _ajustar("monte_carlo_include_all", True, False,
                     f"RAM livre {ram:.1f} GB < 6 GB")
    elif ram < 8.0:
        if cfg.run_shap and cfg.shap_max_samples > 300:
            _ajustar("shap_max_samples", cfg.shap_max_samples, 300,
                     f"RAM livre {ram:.1f} GB < 8 GB")
            cfg.shap_max_samples = 300

    # Resolucao das figuras: nao muda nenhum numero, so' o arquivo. Medido
    # (2026-10-15, 243 x 1500): pico de memoria de 958 MB a 600 dpi contra
    # 434 MB a 300 dpi -- o maior risco de travar uma maquina de 4 GB.
    dpi = int(getattr(cfg, "save_dpi", 0) or 0)
    if ram < 6.0 and dpi > 300:
        cfg.save_dpi = 300
        _ajustar("save_dpi", dpi, 300, f"RAM livre {ram:.1f} GB < 6 GB")

    # Processos paralelos das permutacoes: cada processo do loky carrega o
    # interpretador, numpy/sklearn e uma copia de X (~0,2-0,4 GB medidos no
    # Windows). O resultado nao depende de n_jobs (sementes por permutacao).
    n_jobs = int(getattr(cfg, "n_jobs_permutation", 1) or 1)
    if n_jobs != 1:
        nucleos = int(hw.get("cpu_fisicos") or 1)
        teto = max(1, min(nucleos, int((ram - 1.0) / 0.4)))
        pedido = nucleos if n_jobs < 0 else n_jobs
        if pedido > teto:
            cfg.n_jobs_permutation = teto
            _ajustar("n_jobs_permutation", n_jobs, teto,
                     f"RAM livre {ram:.1f} GB, {nucleos} nucleos fisicos "
                     f"(~0,4 GB por processo)")
    return avisos


def _verificar_ram(min_gb: float, operacao: str) -> bool:
    """
    Verifica RAM livre antes de operacao pesada.
    Retorna True se seguro prosseguir, False se insuficiente.
    Nunca lanca excecao — fail-safe assume OK se psutil indisponivel.
    """
    try:
        import psutil as _ps
        livre_gb = _ps.virtual_memory().available / 1024**3
        if livre_gb < min_gb:
            print(f"  [AVISO RAM] '{operacao}' pulada: "
                  f"{livre_gb:.1f} GB livre < {min_gb:.1f} GB necessario. "
                  f"Reduza n_amostras, desabilite SHAP ou feche outros programas.")
            return False
    except ImportError:
        pass   # sem psutil: assume que ha memoria (falha graciosamente)
    except Exception as _e_ram:  # noqa: BLE001 -- fail-safe por contrato da
        # funcao (docstring: "nunca lanca excecao"); assume OK e prossegue.
        logging.getLogger(__name__).debug(
            "_verificar_ram('%s'): deteccao falhou, assumindo OK: %s",
            operacao, _e_ram)
    return True
