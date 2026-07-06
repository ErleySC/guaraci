"""Testes de paleta_cores.py — paleta e marcadores de máxima
distintividade perceptual. Modulo sem cobertura dedicada ate aqui
(reforço de margem de cobertura de CI, auditoria jul/2026 — piso 60%,
real 62%). Os testes de _paleta_externa cobrem os 3 caminhos (glasbey
disponivel / só colorcet / nenhum instalado) via monkeypatch em
builtins.__import__, sem depender de quais libs opcionais estao de fato
instaladas no ambiente que roda o teste."""
import builtins

import pytest


@pytest.fixture(scope="module")
def pc():
    import guaraci.paleta_cores as mod
    return mod


def _bloquear_imports(nomes_bloqueados):
    """Monkeypatch de __import__ que lança ImportError para os módulos
    dados, deixando os demais imports passarem normalmente."""
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name in nomes_bloqueados:
            raise ImportError(f"{name} bloqueado no teste")
        return real_import(name, *args, **kwargs)
    return _fake_import


def test_paleta_externa_usa_glasbey_quando_disponivel(pc, monkeypatch):
    pytest.importorskip("glasbey")
    cores = pc._paleta_externa(5)
    assert cores is not None
    assert len(cores) == 5


def test_paleta_externa_cai_para_colorcet_sem_glasbey(pc, monkeypatch):
    pytest.importorskip("colorcet")
    monkeypatch.setattr(builtins, "__import__",
                         _bloquear_imports({"glasbey"}))
    cores = pc._paleta_externa(3)
    assert cores is not None
    assert len(cores) == 3


def test_paleta_externa_retorna_none_sem_nenhuma_lib(pc, monkeypatch):
    monkeypatch.setattr(builtins, "__import__",
                         _bloquear_imports({"glasbey", "colorcet"}))
    assert pc._paleta_externa(3) is None


def test_cor_dentro_da_paleta_base(pc):
    assert pc.cor(0) == pc.PALETA[0]
    assert pc.cor(len(pc.PALETA) - 1) == pc.PALETA[-1]


def test_cor_alem_da_paleta_sem_lib_externa_usa_tab20(pc, monkeypatch):
    monkeypatch.setattr(builtins, "__import__",
                         _bloquear_imports({"glasbey", "colorcet"}))
    resultado = pc.cor(len(pc.PALETA) + 2)
    assert resultado.startswith("#") and len(resultado) == 7


def test_luminancia_preto_e_branco(pc):
    assert pc._luminancia("#000000") == pytest.approx(0.0, abs=1e-6)
    assert pc._luminancia("#FFFFFF") == pytest.approx(1.0, abs=1e-6)


def test_edge_para_cor_escolhe_contraste_correto(pc):
    assert pc.edge_para_cor("#FFFFFF") == "0.25"
    assert pc.edge_para_cor("#000000") == "white"


def test_mapear_cores_classes_e_deterministico_e_ordenado(pc):
    classes = ["Zebra", "Abelha", "Mono"]
    mapa1 = pc.mapear_cores_classes(classes)
    mapa2 = pc.mapear_cores_classes(list(reversed(classes)))
    assert mapa1 == mapa2
    ordenadas = sorted(classes)
    assert mapa1[ordenadas[0]] == pc.PALETA[0]
    assert mapa1[ordenadas[1]] == pc.PALETA[1]


def test_mapear_marcadores_classes_ciclo(pc):
    classes = [f"c{i}" for i in range(len(pc.MARCADORES) + 2)]
    mapa = pc.mapear_marcadores_classes(classes)
    ordenadas = sorted(classes)
    assert mapa[ordenadas[0]] == pc.MARCADORES[0]
    assert mapa[ordenadas[len(pc.MARCADORES)]] == pc.MARCADORES[0]
