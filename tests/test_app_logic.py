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


# ── progresso_do_log: "bug do progresso" (achado 2026-08-07) ────────────────
# A etapa "[6/7]" (figuras + DD-SIMCA + OPLS-DA + holdout) concentra a maior
# parte do tempo real de execução, mas só tinha 2 marcadores de texto
# opcionais entre início e fim -- sem eles, a fração ficava CRAVADA em
# 6/7=0.857 durante toda essa fase (medido: 96,1% das amostras de progresso
# num run real, ver docs/auditoria/medir_bug_progresso_cli.py). Estes testes
# travam a correção: com `total_figuras_planejadas`, a fração AVANÇA
# conforme cada figura é salva.

def test_progresso_etapa6_sem_total_planejado_comportamento_antigo():
    """Retrocompatibilidade: sem o parâmetro novo, a fração da etapa 6 é
    EXATAMENTE a mesma de antes da correção (6/7), mesmo com figuras já
    salvas no log -- ninguém que já chama `progresso_do_log(txt)` (1
    argumento) é afetado."""
    txt = "[6/7] Gerando figuras...\n" + "\n".join(
        f"  -> saida/fig{i}.png" for i in range(5))
    frac, _ = progresso_do_log(txt)
    assert frac == pytest.approx(6 / 7.0)


def test_progresso_etapa6_avanca_com_figuras_concluidas():
    """Com `total_figuras_planejadas`, a fração sobe conforme mais figuras
    aparecem no log -- não fica mais cravada num único número durante toda
    a etapa mais demorada."""
    base = "[6/7] Gerando figuras...\n"
    frac_0fig, _ = progresso_do_log(base, total_figuras_planejadas=10)
    frac_5fig, _ = progresso_do_log(
        base + "\n".join(f"  -> saida/fig{i}.png" for i in range(5)),
        total_figuras_planejadas=10)
    frac_10fig, _ = progresso_do_log(
        base + "\n".join(f"  -> saida/fig{i}.png" for i in range(10)),
        total_figuras_planejadas=10)
    # Nunca regride, sempre avança com mais figuras.
    assert frac_0fig == pytest.approx(6 / 7.0)
    assert frac_0fig < frac_5fig < frac_10fig
    # Nunca ULTRAPASSA o teto global 0.99 (com o plano 100% concluído,
    # pode alcançar o teto, mas nunca estourá-lo).
    assert frac_10fig <= 0.99


def test_progresso_etapa6_nao_afeta_outras_etapas():
    """O bônus de figuras só se aplica DENTRO da etapa 6 -- em qualquer
    outra etapa, `total_figuras_planejadas` não muda o resultado."""
    for n in (0, 1, 2, 3, 4, 5, 7):
        txt = f"[{n}/7] etapa\n  -> saida/fig0.png\n  -> saida/fig1.png"
        frac_sem, _ = progresso_do_log(txt)
        frac_com, _ = progresso_do_log(txt, total_figuras_planejadas=10)
        assert frac_sem == frac_com == pytest.approx(min(0.99, n / 7.0))


def test_progresso_etapa6_total_zero_nao_quebra():
    """total_figuras_planejadas=0 (plano vazio, caso degenerado) não deve
    causar ZeroDivisionError -- cai no comportamento sem bônus."""
    frac, _ = progresso_do_log("[6/7] etapa", total_figuras_planejadas=0)
    assert frac == pytest.approx(6 / 7.0)


def test_progresso_substep_holdout_e_comparacao_pipelines():
    """Sub-passos da etapa 6 (achado 2026-08-07: não eram reconhecidos --
    só a etapa 7 tinha rótulo específico para sub-passos) mostram rótulo
    específico em vez do genérico da etapa."""
    _, nome_holdout = progresso_do_log("[6/7] fig\n[6c/7] holdout rodando")
    assert "holdout" in nome_holdout.lower()
    _, nome_comp = progresso_do_log("[6/7] fig\n[6b/7] comparando")
    assert "preprocessing" in nome_comp.lower() or "pipelines" in nome_comp.lower()


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
