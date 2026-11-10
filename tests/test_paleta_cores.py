"""Testes de paleta_cores.py — paleta e marcadores de máxima
distintividade perceptual. Modulo sem cobertura dedicada ate aqui
(reforço de margem de cobertura de CI, auditoria jul/2026 — piso 60%,
real 62%). Os testes de _paleta_externa cobrem os 3 caminhos (glasbey
disponivel / só colorcet / nenhum instalado) via monkeypatch em
builtins.__import__, sem depender de quais libs opcionais estao de fato
instaladas no ambiente que roda o teste."""
import builtins
import re

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
    assert all(re.fullmatch(r"#[0-9a-fA-F]{6}", c) for c in cores), cores


def test_paleta_externa_cai_para_colorcet_sem_glasbey(pc, monkeypatch):
    pytest.importorskip("colorcet")
    monkeypatch.setattr(builtins, "__import__",
                         _bloquear_imports({"glasbey"}))
    cores = pc._paleta_externa(3)
    assert cores is not None
    assert len(cores) == 3
    # hex de verdade: o ramo colorcet devolvia triplas RGB em float (bug real)
    assert all(isinstance(c, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", c) for c in cores), cores


def test_paleta_externa_retorna_none_sem_nenhuma_lib(pc, monkeypatch):
    monkeypatch.setattr(builtins, "__import__",
                         _bloquear_imports({"glasbey", "colorcet"}))
    assert pc._paleta_externa(3) is None


def test_cor_dentro_da_paleta_base(pc):
    assert pc.color(0) == pc.PALETA[0]
    assert pc.color(len(pc.PALETA) - 1) == pc.PALETA[-1]


def test_cor_alem_da_paleta_sem_lib_externa_usa_tab20(pc, monkeypatch):
    monkeypatch.setattr(builtins, "__import__",
                         _bloquear_imports({"glasbey", "colorcet"}))
    resultado = pc.color(len(pc.PALETA) + 2)
    assert resultado.startswith("#") and len(resultado) == 7


def test_luminancia_preto_e_branco(pc):
    assert pc._luminancia("#000000") == pytest.approx(0.0, abs=1e-6)
    assert pc._luminancia("#FFFFFF") == pytest.approx(1.0, abs=1e-6)


def test_edge_para_cor_escolhe_contraste_correto(pc):
    assert pc.get_edge_color("#FFFFFF") == "0.25"
    assert pc.get_edge_color("#000000") == "white"


def test_mapear_cores_classes_e_deterministico_e_ordenado(pc):
    classes = ["Zebra", "Abelha", "Mono"]
    mapa1 = pc.map_class_colors(classes)
    mapa2 = pc.map_class_colors(list(reversed(classes)))
    assert mapa1 == mapa2
    ordenadas = sorted(classes)
    assert mapa1[ordenadas[0]] == pc.PALETA[0]
    assert mapa1[ordenadas[1]] == pc.PALETA[1]


def test_mapear_marcadores_classes_ciclo(pc):
    classes = [f"c{i}" for i in range(len(pc.MARCADORES) + 2)]
    mapa = pc.map_class_markers(classes)
    ordenadas = sorted(classes)
    assert mapa[ordenadas[0]] == pc.MARCADORES[0]
    assert mapa[ordenadas[len(pc.MARCADORES)]] == pc.MARCADORES[0]


# ── Paleta ATIVA (escolha do usuario chega de fato a figura) ───────────────
# REGRESSAO (2026-09-08): escolher uma paleta so' mexia em
# `rcParams["axes.prop_cycle"]`, e NENHUMA figura do pipeline usa o ciclo
# padrao do matplotlib (todas passam `color=color(i)`/`map_class_colors()`).
# O menu confirmava a escolha e nao mudava cor nenhuma. Estes testes fixam o
# contrato oposto.

@pytest.fixture
def paleta_limpa(pc):
    """Garante que cada teste comeca e termina sem paleta ativa (estado de
    modulo global)."""
    pc.set_active_palette(None)
    yield pc
    pc.set_active_palette(None)


def test_sem_paleta_ativa_comportamento_historico_intacto(paleta_limpa):
    pc = paleta_limpa
    assert pc.get_active_palette() is None
    assert pc.color(0) == pc.PALETA[0]
    assert pc.map_class_colors(["a", "b"])["a"] == pc.PALETA[0]


def test_paleta_ativa_muda_cor_de_color_e_map_class_colors(paleta_limpa):
    pc = paleta_limpa
    pc.set_active_palette(["#111111", "#222222", "#333333"])
    assert pc.get_active_palette() == ["#111111", "#222222", "#333333"]
    assert pc.color(0) == "#111111"
    assert pc.map_class_colors(["b", "a"]) == {"a": "#111111", "b": "#222222"}


def test_paleta_ativa_menor_que_n_classes_e_recusada_sem_repetir_cor(
        paleta_limpa, caplog):
    """Duas especies com a MESMA cor na mesma figura e' um grafico que mente
    -- prefere-se abandonar a paleta escolhida e avisar."""
    pc = paleta_limpa
    pc.set_active_palette(["#111111", "#222222"])
    classes = ["a", "b", "c", "d"]
    with caplog.at_level("WARNING"):
        mapa = pc.map_class_colors(classes)
    assert len(set(mapa.values())) == len(classes)      # nenhuma cor repetida
    assert "#111111" not in mapa.values()               # nao usou a escolhida
    assert any("paleta escolhida" in r.message for r in caplog.records)


def test_set_active_palette_none_volta_ao_padrao(paleta_limpa):
    pc = paleta_limpa
    pc.set_active_palette(["#111111"])
    pc.set_active_palette(None)
    assert pc.get_active_palette() is None
    assert pc.color(0) == pc.PALETA[0]


def test_apply_palette_do_catalogo_ativa_as_cores_reais(paleta_limpa):
    """`apply_palette` e' a implementacao UNICA usada pela CLI e pelo app
    web; tem que ativar as cores do catalogo, nao so' mexer em rcParams."""
    pc = paleta_limpa
    from guaraci.cli_assistente import PALETAS_COR, apply_palette

    apply_palette("daltonismo_safe")
    assert pc.get_active_palette() == PALETAS_COR["daltonismo_safe"]["cores"]
    assert pc.color(1) == PALETAS_COR["daltonismo_safe"]["cores"][1]

    apply_palette("qualitativo")     # entrada sem lista de cores fixa
    assert pc.get_active_palette() is None
    assert pc.color(0) == pc.PALETA[0]


def test_apply_palette_nome_desconhecido_nao_explode(paleta_limpa):
    from guaraci.cli_assistente import apply_palette
    apply_palette("nao_existe_essa_paleta")
    assert paleta_limpa.get_active_palette() is None


def test_cli_e_web_usam_a_mesma_funcao_de_paleta():
    """Anti-duplicacao: a CLI (guaraci.py) e a aba Modelo do app web tem que
    apontar para o MESMO `apply_palette`, nunca cada uma com sua copia."""
    import guaraci.cli_assistente as cli
    import guaraci.guaraci as cli_menu
    from guaraci.app_tabs import modelo as aba_modelo

    assert cli_menu._aplicar_paleta is cli.apply_palette
    assert aba_modelo.apply_palette is cli.apply_palette


# ── ordem entre arquivos: o estado global da paleta nao vaza de um teste para outro ──────────────────────────────────────────────────

_VAZAVA = "tests/test_lacunas_modulos_pequenos_final.py::test_paleta_aplica_estilo_e_mapa_de_cor_do_catalogo_e_tolera_estilo_desconhecido"
_SOFRIA = "tests/test_paleta_cores.py::test_cor_dentro_da_paleta_base"


def _pytest_em_ordem(*alvos):
    import subprocess
    import sys
    from pathlib import Path
    raiz = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", "-p", "no:xdist", *alvos],
                       cwd=str(raiz), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    return r.returncode, r.stdout[-1500:]


@pytest.mark.slow
@pytest.mark.parametrize("ordem", [(_VAZAVA, _SOFRIA), (_SOFRIA, _VAZAVA)], ids=["ordem-que-vazava", "ordem-inversa"])
def test_paleta_ativada_por_um_teste_nao_chega_ao_seguinte_em_nenhuma_ordem(ordem):
    rc, saida = _pytest_em_ordem(*ordem)
    assert rc == 0, saida


_ARQUIVOS_DA_PALETA = ["tests/test_lacunas_modulos_pequenos_final.py", "tests/test_auditoria_ux_2026_09_24.py", "tests/test_paleta_cores.py"]


@pytest.mark.slow
@pytest.mark.parametrize("invertida", [False, True], ids=["ordem-direta", "ordem-invertida"])
def test_arquivos_que_mexem_na_paleta_passam_em_ordem_direta_e_invertida(invertida):
    arquivos = list(reversed(_ARQUIVOS_DA_PALETA)) if invertida else _ARQUIVOS_DA_PALETA
    rc, saida = _pytest_em_ordem(*arquivos, "-k", "not em_nenhuma_ordem and not ordem_direta_e_invertida")
    assert rc == 0, saida
