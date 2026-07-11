"""Testes da lógica pura extraída da UI web (guaraci.app_logic, item 19).

Estas funções não dependem de Streamlit, então são testáveis em isolamento —
o objetivo do item 19 é justamente tirar lógica dos monólitos de UI para cá.
"""

import pytest

from guaraci.app_logic import (
    progresso_do_log, fmt_tempo, coletar_config,
    listar_figuras, ler_resumo, ler_model_card,
)


# ── progresso_do_log ─────────────────────────────────────────────────────────
def test_progresso_vazio_retorna_inicio():
    frac, nome = progresso_do_log("")
    assert frac == 0.0
    assert nome == "Starting..."


def test_progresso_sem_marcador_retorna_inicio():
    frac, nome = progresso_do_log("linha qualquer sem marcador de etapa")
    assert frac == 0.0 and nome == "Starting..."


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6])
def test_progresso_usa_maior_etapa(n):
    # Mesmo com etapas antigas no log, usa a MAIOR vista (nunca regride).
    txt = "\n".join(f"[{i}/7] passo" for i in range(n + 1))
    frac, nome = progresso_do_log(txt)
    assert frac == pytest.approx(min(0.99, n / 7.0))
    assert nome  # rótulo não-vazio


def test_progresso_nunca_passa_de_099():
    frac, _ = progresso_do_log("[7/7] done")
    assert frac == 0.99


def test_progresso_substep_benchmark_e_mc():
    frac_b, nome_b = progresso_do_log("[7/7] fim\n[7b/7] rodando")
    assert "Benchmark" in nome_b and frac_b == 0.99
    _, nome_c = progresso_do_log("[7c/7] mc")
    assert "Monte Carlo" in nome_c


def test_progresso_ignora_marcador_malformado():
    # "[9/8]" não casa o padrão /7 → tratado como sem-marcador.
    frac, nome = progresso_do_log("[9/8] invalido")
    assert frac == 0.0 and nome == "Starting..."


# ── fmt_tempo ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("entrada,esperado", [
    (0, "0s"),
    (5, "5s"),
    (59, "59s"),
    (60, "1min 00s"),
    (61, "1min 01s"),
    (3600, "1h 00min"),
    (3661, "1h 01min"),
    (86400, "1d 0h"),
    (90000, "1d 1h"),
])
def test_fmt_tempo_faixas(entrada, esperado):
    assert fmt_tempo(entrada) == esperado


@pytest.mark.parametrize("ruim", [None, "abc", float("nan"), -5])
def test_fmt_tempo_robusto_a_entrada_ruim(ruim):
    out = fmt_tempo(ruim)
    assert out in ("—", "0s")


def test_fmt_tempo_arredonda():
    assert fmt_tempo(59.6) == "1min 00s"


# ── coletar_config ───────────────────────────────────────────────────────────
def test_coletar_config_aplica_valores(pq):
    base = pq.Config()
    cfg, erros = coletar_config(base, {"max_lvs": 12})
    assert erros == []
    assert cfg.max_lvs == 12


def test_coletar_config_nao_muta_base(pq):
    base = pq.Config()
    orig = base.max_lvs
    coletar_config(base, {"max_lvs": orig + 7})
    assert base.max_lvs == orig  # deepcopy: original intacto


def test_coletar_config_ignora_chave_desconhecida(pq):
    base = pq.Config()
    cfg, erros = coletar_config(base, {"chave_que_nao_existe": 1})
    assert erros == []  # chave fora do _CONFIG_SPEC é ignorada


def test_coletar_config_reporta_erro_de_coercao(pq):
    base = pq.Config()
    # max_lvs espera int; um valor não-coercível deve ir para `erros`, sem lançar.
    cfg, erros = coletar_config(base, {"max_lvs": "não-é-número"})
    assert any("max_lvs" in e for e in erros)


# ── listar_figuras / ler_resumo / ler_model_card ─────────────────────────────
def test_listar_figuras_encontra_png_jpg_recursivo(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.png").write_text("x")
    (tmp_path / "sub" / "b.jpg").write_text("x")
    (tmp_path / "nota.txt").write_text("x")  # ignorado (nao e figura)
    imgs = listar_figuras(str(tmp_path))
    assert len(imgs) == 2
    assert all(im.lower().endswith((".png", ".jpg")) for im in imgs)


def test_listar_figuras_pasta_sem_imagens_retorna_vazio(tmp_path):
    assert listar_figuras(str(tmp_path)) == []


def test_ler_resumo_prioriza_logs_subpasta(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "resumo_modelo.txt").write_text("conteudo em logs/")
    (tmp_path / "resumo_modelo.txt").write_text("conteudo na raiz")
    assert ler_resumo(str(tmp_path)) == "conteudo em logs/"


def test_ler_resumo_cai_para_raiz_sem_logs(tmp_path):
    (tmp_path / "resumo_modelo.txt").write_text("so' na raiz")
    assert ler_resumo(str(tmp_path)) == "so' na raiz"


def test_ler_resumo_arquivo_ausente_retorna_none(tmp_path):
    assert ler_resumo(str(tmp_path)) is None


def test_ler_model_card_prioriza_logs_subpasta(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "model_card.md").write_text("card em logs/")
    (tmp_path / "model_card.md").write_text("card na raiz")
    assert ler_model_card(str(tmp_path)) == "card em logs/"


def test_ler_model_card_ausente_retorna_none(tmp_path):
    assert ler_model_card(str(tmp_path)) is None
