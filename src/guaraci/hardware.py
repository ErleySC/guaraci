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


def hardware_probe() -> Dict[str, Any]:
    """
    Detecta RAM, CPU e disco disponiveis.
    Usa psutil se disponivel; caso contrario retorna estimativas conservadoras.
    Nunca lanca excecao — falha silenciosamente com valores defaults seguros.
    """
    info: Dict[str, Any] = {
        "ram_total_gb":  4.0,   # defaults conservadores
        "ram_livre_gb":  2.0,
        "cpu_logicos":   2,
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

        limite_cgroup = _cgroup_ram_limit_gb()
        if limite_cgroup is not None and limite_cgroup < info["ram_total_gb"]:
            # Container com limite real menor que a RAM do host: o total
            # exibido passa a ser o limite do container (o que a analise
            # de fato pode usar), e a RAM livre e recalculada dentro desse
            # teto (psutil.available pode ultrapassar o limite do cgroup).
            usada_gb = max(0.0, info["ram_total_gb"] - info["ram_livre_gb"])
            info["ram_total_gb"] = limite_cgroup
            info["ram_livre_gb"] = round(max(0.0, limite_cgroup - usada_gb), 1)
            info["ram_limitada_por_container"] = True

        info["psutil_ok"] = True
    except ImportError:
        # Fallback Windows: GlobalMemoryStatusEx via ctypes
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
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))  # type: ignore
            info["ram_total_gb"] = round(ms.ullTotalPhys / 1024**3, 1)
            info["ram_livre_gb"] = round(ms.ullAvailPhys / 1024**3, 1)
        except Exception as _e_win:  # noqa: BLE001 -- fallback Windows
            # best-effort; mantem os defaults conservadores de RAM ja setados.
            logging.getLogger(__name__).debug(
                "hardware_probe: RAM (ctypes) nao detectada: %s", _e_win)
        # CPU via os
        try:
            info["cpu_logicos"] = os.cpu_count() or 2
            info["cpu_fisicos"] = max(1, (os.cpu_count() or 2) // 2)
        except Exception as _e_cpu:  # noqa: BLE001 -- mesmo contrato
            # best-effort; mantem os defaults conservadores de CPU.
            logging.getLogger(__name__).debug(
                "hardware_probe: CPU nao detectada: %s", _e_cpu)
    except Exception as _e_probe:  # noqa: BLE001 -- ultimo recurso do
        # probe inteiro (docstring: "nunca lanca excecao"); os defaults
        # conservadores setados no topo da funcao ja cobrem este caso.
        logging.getLogger(__name__).debug(
            "hardware_probe: deteccao de hardware falhou por completo: %s",
            _e_probe)
    return info


def auto_ajustar_config_hardware(cfg: "Config",
                                  hw: Dict[str, Any]) -> List[str]:
    """
    Ajusta automaticamente limites do cfg com base na RAM livre detectada.
    Previne travamentos em maquinas com < 8 GB RAM disponivel.
    Retorna lista de mensagens de aviso para impressao.
    """
    avisos: List[str] = []
    ram = float(hw.get("ram_livre_gb", 16.0))

    if ram < 2.0:
        # Modo minimo absoluto: desabilitar tudo pesado
        if cfg.executar_shap:
            cfg.executar_shap = False
            avisos.append("SHAP desabilitado (RAM livre < 2 GB)")
        if cfg.executar_benchmark:
            cfg.executar_benchmark = False
            avisos.append("Benchmark desabilitado (RAM livre < 2 GB)")
        if cfg.executar_monte_carlo:
            cfg.executar_monte_carlo = False
            avisos.append("Monte Carlo CV desabilitado (RAM livre < 2 GB)")
        if cfg.n_splits_cv > 3:
            cfg.n_splits_cv = 3
            avisos.append("CV reduzido para 3 folds (RAM livre < 2 GB)")

    elif ram < 4.0:
        # RAM 2-4 GB: desabilitar SHAP e benchmark, limitar MC CV
        if cfg.executar_shap:
            cfg.executar_shap = False
            avisos.append("SHAP desabilitado (RAM livre < 4 GB)")
        if cfg.executar_benchmark:
            cfg.executar_benchmark = False
            avisos.append("Benchmark desabilitado (RAM livre < 4 GB). "
                          "Habilite manualmente se necessario.")
        if cfg.n_monte_carlo > 30:
            cfg.n_monte_carlo = 30
            avisos.append("Monte Carlo CV limitado a 30 iteracoes (RAM livre < 4 GB)")

    elif ram < 6.0:
        # RAM 4-6 GB: SHAP com amostragem reduzida, benchmark sem XGBoost via flag
        if cfg.executar_shap and cfg.shap_max_amostras > 150:
            cfg.shap_max_amostras = 150
            avisos.append("SHAP max_amostras reduzido para 150 (RAM livre < 6 GB)")
        if cfg.n_monte_carlo > 60:
            cfg.n_monte_carlo = 60
            avisos.append("Monte Carlo CV limitado a 60 iteracoes (RAM livre < 6 GB)")
        if cfg.monte_carlo_incluir_todos:
            cfg.monte_carlo_incluir_todos = False
            avisos.append("MC CV multi-modelo desabilitado (RAM livre < 6 GB)")

    elif ram < 8.0:
        # RAM 6-8 GB: reducoes moderadas
        if cfg.executar_shap and cfg.shap_max_amostras > 300:
            cfg.shap_max_amostras = 300
            avisos.append("SHAP max_amostras reduzido para 300 (RAM livre < 8 GB)")
        if cfg.n_monte_carlo > 80:
            cfg.n_monte_carlo = 80
            avisos.append("Monte Carlo CV limitado a 80 iteracoes (RAM livre < 8 GB)")

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
