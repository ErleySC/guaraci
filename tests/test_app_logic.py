"""Testes da lógica pura extraída da UI web (guaraci.app_logic, item 19).

Estas funções não dependem de Streamlit, então são testáveis em isolamento —
o objetivo do item 19 é justamente tirar lógica dos monólitos de UI para cá.
"""

from pathlib import Path

import pytest

from guaraci.app_logic import (
    log_progress, fmt_time, collect_config,
    list_figures, load_summary, load_model_card,
    temp_upload_path,
)


# ── log_progress ─────────────────────────────────────────────────────────
def test_progresso_vazio_retorna_inicio():
    frac, nome = log_progress("")
    assert frac == 0.0
    assert nome == "Starting..."


def test_progresso_sem_marcador_retorna_inicio():
    frac, nome = log_progress("linha qualquer sem marcador de etapa")
    assert frac == 0.0 and nome == "Starting..."


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6])
def test_progresso_usa_maior_etapa(n):
    # Mesmo com etapas antigas no log, usa a MAIOR vista (nunca regride).
    txt = "\n".join(f"[{i}/7] passo" for i in range(n + 1))
    frac, nome = log_progress(txt)
    assert frac == pytest.approx(min(0.99, n / 7.0))
    assert nome  # rótulo não-vazio


def test_progresso_nunca_passa_de_099():
    frac, _ = log_progress("[7/7] done")
    assert frac == 0.99


def test_progresso_substep_benchmark_e_mc():
    frac_b, nome_b = log_progress("[7/7] fim\n[7b/7] rodando")
    assert "Benchmark" in nome_b and frac_b == 0.99
    _, nome_c = log_progress("[7c/7] mc")
    assert "Monte Carlo" in nome_c


def test_progresso_ignora_marcador_malformado():
    # "[9/8]" não casa o padrão /7 → tratado como sem-marcador.
    frac, nome = log_progress("[9/8] invalido")
    assert frac == 0.0 and nome == "Starting..."


# ── log_progress: "bug do progresso" (achado 2026-08-07) ────────────────
# A etapa "[6/7]" (figuras + DD-SIMCA + OPLS-DA + holdout) concentra a maior
# parte do tempo real de execução, mas só tinha 2 marcadores de texto
# opcionais entre início e fim -- sem eles, a fração ficava CRAVADA em
# 6/7=0.857 durante toda essa fase (medido: 96,1% das amostras de progresso
# num run real, ver scripts/medicoes/medir_bug_progresso_cli.py). Estes testes
# travam a correção: com `total_figuras_planejadas`, a fração AVANÇA
# conforme cada figura é salva.

def test_progresso_etapa6_sem_total_planejado_comportamento_antigo():
    """Retrocompatibilidade: sem o parâmetro novo, a fração da etapa 6 é
    EXATAMENTE a mesma de antes da correção (6/7), mesmo com figuras já
    salvas no log -- ninguém que já chama `log_progress(txt)` (1
    argumento) é afetado."""
    txt = "[6/7] Gerando figuras...\n" + "\n".join(
        f"  -> saida/fig{i}.png" for i in range(5))
    frac, _ = log_progress(txt)
    assert frac == pytest.approx(6 / 7.0)


def test_progresso_etapa6_avanca_com_figuras_concluidas():
    """Com `total_figuras_planejadas`, a fração sobe conforme mais figuras
    aparecem no log -- não fica mais cravada num único número durante toda
    a etapa mais demorada."""
    base = "[6/7] Gerando figuras...\n"
    frac_0fig, _ = log_progress(base, total_figuras_planejadas=10)
    frac_5fig, _ = log_progress(
        base + "\n".join(f"  -> saida/fig{i}.png" for i in range(5)),
        total_figuras_planejadas=10)
    frac_10fig, _ = log_progress(
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
        frac_sem, _ = log_progress(txt)
        frac_com, _ = log_progress(txt, total_figuras_planejadas=10)
        assert frac_sem == frac_com == pytest.approx(min(0.99, n / 7.0))


# ── log_progress: CV aninhada sem indicacao de progresso (Grupo 3, 2026-09-11) ─
# `pipeline.executar()` mediu ate' 16 minutos de silencio total na etapa
# "[2/7]" enquanto a CV aninhada de selecao de LVs roda (scripts/medicoes/
# medir_custo_cv_aninhada.py) -- mesma familia de bug de "[6/7]" acima, so'
# que sem NENHUM marcador intermediario (nem os 2 opcionais que "[6/7]" ja
# tinha). Estes testes travam a correcao: um log por fold externo
# ("[2b/7] ... fold externo K/N concluido") agora avanca a fracao.

def test_progresso_etapa2_sem_fold_concluido_fica_no_valor_base():
    """Sem nenhuma linha de fold (comportamento antes da correcao / etapa
    2 mal comecou): fracao e' exatamente 2/7, igual a qualquer outra
    etapa sem sub-passo."""
    frac, _ = log_progress("[2/7] LV selection by CV\n[INFO] CV aninhada ativada")
    assert frac == pytest.approx(2 / 7.0)


def test_progresso_etapa2_avanca_a_cada_fold_aninhado_concluido():
    """Ao menos uma atualizacao POR FOLD -- nao so' inicio/fim: a fracao
    sobe estritamente a cada fold externo reportado como concluido, nunca
    regride, nunca ultrapassa o teto 0.99."""
    base = "[2/7] LV selection by CV\n[INFO] CV aninhada ativada\n"
    fracoes = []
    for k in range(1, 6):
        txt = base + "\n".join(
            f"  [2b/7] CV aninhada: fold externo {i}/5 concluido"
            for i in range(1, k + 1))
        frac, nome = log_progress(txt)
        fracoes.append(frac)
        assert "Nested CV" in nome or "LV" in nome
    assert fracoes == sorted(fracoes)          # nunca regride
    assert len(set(fracoes)) == 5              # os 5 folds dao 5 valores DISTINTOS
    assert fracoes[0] > 2 / 7.0                 # 1o fold ja avanca alem do valor base
    assert fracoes[-1] <= 0.99


def test_progresso_etapa2_usa_o_maior_fold_visto_nunca_regride():
    """Mesmo principio do resto de log_progress: usa o MAIOR progresso ja
    visto no log, mesmo que uma linha de fold MAIOR apareca ANTES de uma
    de fold menor (buffering/ordem de escrita nao ideal) -- nao regride
    so' porque a ULTIMA linha do texto e' de um fold anterior."""
    txt = ("[2/7] etapa\n"
           "  [2b/7] CV aninhada: fold externo 4/5 concluido\n"
           "  [2b/7] CV aninhada: fold externo 2/5 concluido\n")  # fora de ordem
    frac, _ = log_progress(txt)
    assert frac == pytest.approx((2 + 4 / 5) / 7.0)


def test_progresso_etapa2_nao_afeta_outras_etapas():
    """A linha de fold da CV aninhada so' importa quando a etapa atual e'
    a 2 -- um marcador "[2b/7]" perdido no log de uma etapa POSTERIOR nao
    muda o resultado (mesmo padrao de test_progresso_etapa6_nao_afeta_
    outras_etapas). n < 2 fica de fora de proposito: "[2b/7]" contem "2"
    e legitimamente viraria a MAIOR etapa vista nesse caso -- nao e' o
    comportamento sob teste aqui."""
    for n in (3, 4, 5, 6, 7):
        txt = (f"[{n}/7] etapa\n"
               f"  [2b/7] CV aninhada: fold externo 1/5 concluido")
        frac, _ = log_progress(txt)
        assert frac == pytest.approx(min(0.99, n / 7.0))


def test_progresso_etapa6_total_zero_nao_quebra():
    """total_figuras_planejadas=0 (plano vazio, caso degenerado) não deve
    causar ZeroDivisionError -- cai no comportamento sem bônus."""
    frac, _ = log_progress("[6/7] etapa", total_figuras_planejadas=0)
    assert frac == pytest.approx(6 / 7.0)


def test_progresso_substep_holdout_e_comparacao_pipelines():
    """Sub-passos da etapa 6 (achado 2026-08-07: não eram reconhecidos --
    só a etapa 7 tinha rótulo específico para sub-passos) mostram rótulo
    específico em vez do genérico da etapa."""
    _, nome_holdout = log_progress("[6/7] fig\n[6c/7] holdout rodando")
    assert "holdout" in nome_holdout.lower()
    _, nome_comp = log_progress("[6/7] fig\n[6b/7] comparando")
    assert "preprocessing" in nome_comp.lower() or "pipelines" in nome_comp.lower()


# ── fmt_time ────────────────────────────────────────────────────────────────
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
    assert fmt_time(entrada) == esperado


@pytest.mark.parametrize("ruim", [None, "abc", float("nan"), -5])
def test_fmt_tempo_robusto_a_entrada_ruim(ruim):
    out = fmt_time(ruim)
    assert out in ("—", "0s")


def test_fmt_tempo_arredonda():
    assert fmt_time(59.6) == "1min 00s"


# ── collect_config ───────────────────────────────────────────────────────────
def test_coletar_config_aplica_valores(pq):
    base = pq.Config()
    cfg, erros = collect_config(base, {"max_lvs": 12})
    assert erros == []
    assert cfg.max_lvs == 12


def test_coletar_config_nao_muta_base(pq):
    base = pq.Config()
    orig = base.max_lvs
    collect_config(base, {"max_lvs": orig + 7})
    assert base.max_lvs == orig  # deepcopy: original intacto


def test_coletar_config_ignora_chave_desconhecida(pq):
    base = pq.Config()
    cfg, erros = collect_config(base, {"chave_que_nao_existe": 1})
    assert erros == []  # chave fora do _CONFIG_SPEC é ignorada


def test_coletar_config_reporta_erro_de_coercao(pq):
    base = pq.Config()
    # max_lvs espera int; um valor não-coercível deve ir para `erros`, sem lançar.
    cfg, erros = collect_config(base, {"max_lvs": "não-é-número"})
    assert any("max_lvs" in e for e in erros)


# ── list_figures / load_summary / load_model_card ─────────────────────────────
def test_listar_figuras_encontra_png_jpg_recursivo(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.png").write_text("x")
    (tmp_path / "sub" / "b.jpg").write_text("x")
    (tmp_path / "nota.txt").write_text("x")  # ignorado (nao e figura)
    imgs = list_figures(str(tmp_path))
    assert len(imgs) == 2
    assert all(im.lower().endswith((".png", ".jpg")) for im in imgs)


def test_listar_figuras_pasta_sem_imagens_retorna_vazio(tmp_path):
    assert list_figures(str(tmp_path)) == []


def test_ler_resumo_prioriza_logs_subpasta(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "resumo_modelo.txt").write_text("conteudo em logs/")
    (tmp_path / "resumo_modelo.txt").write_text("conteudo na raiz")
    assert load_summary(str(tmp_path)) == "conteudo em logs/"


def test_ler_resumo_cai_para_raiz_sem_logs(tmp_path):
    (tmp_path / "resumo_modelo.txt").write_text("so' na raiz")
    assert load_summary(str(tmp_path)) == "so' na raiz"


def test_ler_resumo_arquivo_ausente_retorna_none(tmp_path):
    assert load_summary(str(tmp_path)) is None


def test_ler_model_card_prioriza_logs_subpasta(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "model_card.md").write_text("card em logs/")
    (tmp_path / "model_card.md").write_text("card na raiz")
    assert load_model_card(str(tmp_path)) == "card em logs/"


def test_ler_model_card_ausente_retorna_none(tmp_path):
    assert load_model_card(str(tmp_path)) is None


# ── temp_upload_path (achado S1 da auditoria de seguranca, 2026-08-07) ───
# Um caminho de upload PREVISIVEL (nome fixo, pasta compartilhada entre
# sessoes/visitantes) era uma das pecas de um bypass de RCE via pickle num
# deploy publico (ver AUDITORIA_SEGURANCA_2026-08-07.md).
# Estes testes travam as DUAS propriedades que fecham essa peca.

def test_caminho_upload_temp_bloqueia_path_traversal():
    """So' o BASENAME do nome original e' usado -- um nome de arquivo
    malicioso como '../../etc/passwd' nao pode escapar do diretorio de
    destino."""
    p = temp_upload_path("../../etc/passwd", "sessao123",
                            base=Path("/base"))
    assert p == Path("/base/pq_uploads/sessao123/passwd")
    assert ".." not in p.parts


def test_caminho_upload_temp_isola_por_sessao():
    """Sessoes/visitantes DIFERENTES com o MESMO nome de arquivo devem cair
    em caminhos DIFERENTES -- e' a propriedade que fecha o bypass de RCE
    (visitante nao pode mais prever/reusar o caminho de outra sessao)."""
    p_a = temp_upload_path("modelo.csv", "sessao-A", base=Path("/base"))
    p_b = temp_upload_path("modelo.csv", "sessao-B", base=Path("/base"))
    assert p_a != p_b
    assert "sessao-A" in p_a.parts
    assert "sessao-B" in p_b.parts


def test_caminho_upload_temp_mesma_sessao_mesmo_nome_da_mesmo_caminho():
    """Propriedade complementar: DENTRO da mesma sessao, o mesmo nome de
    arquivo sempre resolve para o mesmo caminho -- preserva a otimizacao de
    'reusar copia ja salva' que dados.py usa entre reruns do Streamlit."""
    p1 = temp_upload_path("dados.csv", "sessao-X", base=Path("/base"))
    p2 = temp_upload_path("dados.csv", "sessao-X", base=Path("/base"))
    assert p1 == p2


def test_caminho_upload_temp_usa_gettempdir_por_padrao():
    """Sem `base` explicito, usa o diretorio temporario real do SO (nao
    lanca, nao exige o parametro)."""
    p = temp_upload_path("x.csv", "sessao1")
    assert "pq_uploads" in p.parts
    assert "sessao1" in p.parts
    assert p.name == "x.csv"


# ── Próxima ação sugerida (faixa do topo da tela Início) ──────────────────
# A tela só desenha o que esta função decide; a ordem das perguntas é a
# ordem do fluxo de trabalho, e o primeiro obstáculo é o que se sugere.

def test_proxima_acao_sem_dado_manda_para_a_tela_dados():
    from guaraci.app_logic import next_action
    acao = next_action(tem_dados=False, tem_execucao=False)
    assert acao.destino == "dados"
    assert acao.severidade == "info"


def test_proxima_acao_com_dado_sem_execucao_manda_rodar():
    from guaraci.app_logic import next_action
    acao = next_action(tem_dados=True, tem_execucao=False)
    assert acao.destino == "modelo"


def test_achado_critico_tem_prioridade_sobre_aviso_e_sobre_predicao():
    """Um crítico é o obstáculo mais grave: aparece antes de sugerir
    predição ou relatório, mesmo que o resto do fluxo esteja completo."""
    from guaraci.app_logic import next_action
    achados = [{"nome": "a", "severidade": "aviso", "mensagem": "cuidado"},
               {"nome": "b", "severidade": "critico", "mensagem": "grave"}]
    acao = next_action(tem_dados=True, tem_execucao=True,
                       achados_auditoria=achados, tem_predicao=True)
    assert acao.severidade == "critico"
    assert "grave" in acao.detalhe


def test_aviso_aparece_quando_nao_ha_critico():
    from guaraci.app_logic import next_action
    achados = [{"nome": "a", "severidade": "aviso", "mensagem": "1 sessão só"},
               {"nome": "b", "severidade": "ok", "mensagem": "tudo bem"}]
    acao = next_action(tem_dados=True, tem_execucao=True,
                       achados_auditoria=achados)
    assert acao.severidade == "aviso"
    assert "1 alerta(s)" in acao.titulo


def test_fluxo_completo_sem_achado_sugere_relatorio():
    from guaraci.app_logic import next_action
    acao = next_action(tem_dados=True, tem_execucao=True,
                       achados_auditoria=[{"nome": "x", "severidade": "ok",
                                           "mensagem": ""}],
                       tem_predicao=True)
    assert acao.destino == "relatorios"
    assert acao.severidade == "ok"


def test_proxima_acao_traduz_para_ingles():
    from guaraci.app_logic import next_action
    acao = next_action(tem_dados=False, tem_execucao=False, pt=False)
    assert "Load the spectra" in acao.titulo
