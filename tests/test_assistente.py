"""Assistente Guaraci: respostas vem do catalogo, sem inventar e sem relato de validacao."""
import pytest

from guaraci import assistente as ass
from guaraci import catalogo as cat
from guaraci.catalogo_ajuda import DADOS, DIAGNOSTICOS, OBJETIVOS, ROTAS, TERMOS
from guaraci.catalogo_duvidas import DUVIDAS
from guaraci.config import Config


def test_catalogo_com_rotas_e_glossario_cumpre_o_contrato():
    assert cat.validar_catalogo() == []


@pytest.mark.parametrize("par", list(ROTAS), ids=lambda p: f"{p[0]}-{p[1]}")
def test_toda_rota_so_cita_entidades_do_catalogo(par):
    r = ass.rota(*par)
    assert r and r["passos"] and r["motivo"]
    for k in r["config"]:
        assert k in cat.OPCOES
    for t in r["tecnicas"]:
        assert t in cat.TECNICAS
    txt = ass.formatar_rota(r)
    assert cat.texto_proibido(txt) is None, txt


def test_todo_objetivo_tem_ao_menos_uma_rota_e_toda_rota_tem_pergunta():
    objs = {o[0] for o in OBJETIVOS}
    assert objs == {o for o, _ in ROTAS}
    assert {d for _, d in ROTAS} <= {d[0] for d in DADOS}


def test_rota_inexistente_devolve_none_em_vez_de_inventar():
    assert ass.rota("quantificar", "cubo_hsi") is None


def test_monitorar_tem_tela_comando_e_exemplo():
    r = ass.rota("monitorar", "csv_espectral")
    assert r["exemplo"] == "monitorar" and "monitoramento_mspc" in r["tecnicas"]
    passos = [cat.passo_texto(p) for p in r["passos"]]
    assert any("guaraci monitorar" in p for p in passos) and any(p.startswith("[4.2]") for p in passos)


def test_todo_comando_citado_em_rota_existe_no_catalogo():
    for r in ROTAS.values():
        if r["comando"]:
            assert r["comando"].split()[1] in cat.COMANDOS


@pytest.mark.parametrize("oid", list(cat.OPCOES))
def test_explicar_acha_toda_opcao_por_id_e_por_nome_e_usa_o_gabarito(oid):
    for chave in (oid, cat.opcao(oid)["nome"][0], cat.opcao(oid)["nome"][0].upper()):
        e = ass.explicar(chave)
        assert e and e["tipo"] == "opcao" and e["id"] == oid
        assert "Faz:" in e["texto"] and "Use quando" in e["texto"] and "Nao use quando" in e["texto"]


def test_explicar_acha_tecnicas_graficos_telas_comandos_e_termos():
    assert ass.explicar("pls_da")["tipo"] == "tecnica"
    assert ass.explicar("confusao")["tipo"] == "grafico"
    assert ass.explicar("Pre-processamento")["tipo"] in ("tela", "opcao")
    assert ass.explicar("anova")["tipo"] in ("comando", "termo", "tecnica")
    assert ass.explicar("pseudo-replicacao")["tipo"] == "termo"
    assert ass.explicar("termo que nao existe") is None


@pytest.mark.parametrize("termo", list(TERMOS))
def test_todo_termo_do_glossario_e_explicavel(termo):
    e = ass.explicar(termo)
    assert e and "Exemplo:" in e["texto"] or e["tipo"] in ("opcao", "tecnica", "grafico", "tela", "comando")


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_nenhuma_resposta_relata_validacao(lang):
    textos = []
    for oid in cat.OPCOES:
        textos.append(ass.explicar(oid, lang)["texto"])
    for t in cat.TECNICAS:
        textos.append(ass.explicar(t, lang)["texto"])
    for g in cat.GRAFICOS:
        textos.append(ass.explicar(g, lang)["texto"])
    for k in TERMOS:
        textos.append(ass.explicar(k, lang)["texto"])
    for d in DIAGNOSTICOS.values():
        textos += [cat.tx(d["significa"], lang), cat.tx(d["fazer"], lang)]
    ruins = [(t[:60], cat.texto_proibido(t)) for t in textos if cat.texto_proibido(t)]
    assert not ruins, ruins[:5]


def test_buscar_acha_cada_opcao_pelo_nome_e_tolera_erro_de_digitacao():
    for oid, o in cat.OPCOES.items():
        ids = [r["id"] for r in ass.buscar(o["nome"][0], limite=5)]
        assert oid in ids, (oid, ids)
    assert ass.buscar("censura")[0]["id"] in ("censura", "icp_politica_censura", "icp_limiar_censura")
    assert any(r["id"] == "anova_tukey_unidade" for r in ass.buscar("compraracao de grupos anova"))
    assert ass.buscar("xyzqwk") == []
    assert ass.buscar("") == []


def test_buscar_so_devolve_entidades_do_catalogo():
    conhecidos = {("opcao", k) for k in cat.OPCOES} | {("tecnica", k) for k in cat.TECNICAS} | \
        {("grafico", k) for k in cat.GRAFICOS} | {("tela", k) for k in cat.TELAS} | \
        {("comando", k) for k in cat.COMANDOS} | {("termo", k) for k in TERMOS} | {("diagnostico", k) for k in DIAGNOSTICOS} | {("duvida", k) for k in DUVIDAS}
    for q in ("validacao", "grupo", "pre-processamento", "figura", "ICP", "erro", "conformal"):
        for r in ass.buscar(q, limite=20):
            assert (r["tipo"], r["id"]) in conhecidos


def test_diagnosticar_mapeia_os_codigos_reais_da_auditoria():
    from guaraci import auditoria_delineamento as ad
    import inspect
    fonte = inspect.getsource(ad)
    for codigo in ("agrupamento", "classe_unica", "confundimento_classe_sessao", "duplicatas",
                   "faixa_validacao_uso", "grupo_em_varias_classes", "n_insuficiente",
                   "unidade_independente", "validacao_externa"):
        assert f'"{codigo}"' in fonte, codigo                       # o codigo existe de verdade
        d = ass.diagnosticar(f"[AUDITORIA][AVISO] {codigo}: mensagem qualquer")
        assert d and d[0]["id"] == codigo and d[0]["fazer"], codigo


def test_diagnosticar_erros_conhecidos_e_desconhecido():
    assert ass.diagnosticar("ANOVA recusada: unidade independente nao informada")[0]["id"] == "anova_recusada"
    assert ass.diagnosticar("unidade obrigatoria e reconhecida")[0]["id"] == "unidade_icp"
    assert ass.diagnosticar("OSError: [WinError 206] caminho longo")[0]["id"] == "caminho_longo"
    assert ass.diagnosticar("nada a ver") == []


def test_sugerir_nunca_aplica_e_aponta_a_unidade_independente():
    cfg = Config()
    cfg.mode, cfg.group_column, cfg.n_permutations = "csv", None, 20
    antes = dict(vars(cfg))
    s = ass.sugerir(cfg)
    assert vars(cfg) == antes                                        # nada foi alterado
    textos = " ".join(x.texto for x in s)
    assert "coluna_grupo" in textos
    assert any(x.campo == "n_permutacoes" and x.valor == 200 for x in s)


def test_sugerir_configuracao_incompativel_diz_o_que_corrigir():
    cfg = Config()
    cfg.mode, cfg.level, cfg.conc_column = "csv", "N3", None
    s = ass.sugerir(cfg)
    assert any("coluna_concentracao" in x.texto for x in s)


def test_responder_encaminha_para_diagnostico_explicacao_ou_busca():
    assert "anova_recusada" in ass.responder("ANOVA recusada: falta unidade")
    assert ass.responder("pls_da").startswith("PLS-DA")
    assert "Resultados:" in ass.responder("carta t2 lotes")
    # busca fraca (palavra solta em id) nao vira resposta: vale o "nao sei" honesto
    assert ass.responder("separar classes").startswith("Nao sei responder")
    assert ass.responder("zzzzqq").startswith("Nao sei responder")


def test_assistente_responde_no_idioma_escolhido():
    en = ass.responder("pseudo-replicacao", "EN")
    assert "Example:" in en and "Exemplo" not in en
    assert "Recommended route:" in ass.formatar_rota(ass.rota("comparar", "tabela_resposta"), "EN")
    d = ass.diagnosticar("anova recusada", "EN")[0]
    assert d["fazer"].startswith("Give the physical sample column")
    assert "Use when" in ass.explicar("max_lvs", "EN")["texto"]
    assert any("Fix before running" in s.texto or "Declare the independent unit" in s.texto
               for s in ass.sugerir(Config(mode="csv", level="N3"), "EN"))


def test_confianca_em_ingles_traz_limites_e_evidencia_em_ingles():
    from guaraci import confianca as conf
    txt = conf.pagina("tecnica", "anova_tukey_unidade", "EN", ascii_only=True)
    assert "Limits:" in txt and "Evidence:" in txt and "Caveat:" in txt
    assert "the program refuses" in txt and "o programa recusa" not in txt
