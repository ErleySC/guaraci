"""
hardware.py — Deteccao e adaptacao a hardware (RAM/CPU/disco): probe, auto-
ajuste de Config conforme recursos, e guarda de RAM antes de operacoes pesadas.

Extraido de pipeline.py (Fase H). Config so em type hint (TYPE_CHECKING).
pipeline.py e app_quimiometria.py usam via reexport (pipeline.hardware_probe()).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from pipeline import Config


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
        except Exception:
            pass
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
        except Exception:
            pass
        # CPU via os
        try:
            info["cpu_logicos"] = os.cpu_count() or 2
            info["cpu_fisicos"] = max(1, (os.cpu_count() or 2) // 2)
        except Exception:
            pass
    except Exception:
        pass
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
            avisos.append(f"SHAP max_amostras reduzido para 150 (RAM livre < 6 GB)")
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
            avisos.append(f"SHAP max_amostras reduzido para 300 (RAM livre < 8 GB)")
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
    except Exception:
        pass
    return True
