"""Ramos de borda de guaraci.icp_multielementar (leitura, censura, QC, relatorio por elemento): um teste por ramo."""
from __future__ import annotations

import csv
import math
from types import SimpleNamespace

import numpy as np
import pytest

from guaraci import icp_multielementar as icp


def _escrever(tmp_path, texto, nome="icp.csv", binario=None):
    p = tmp_path / nome
    if binario is not None:
        p.write_bytes(binario)
    else:
        p.write_text(texto, encoding="utf-8")
    return str(p)


def _tabela_minima(tipos, valores, censurado=None, **kw):
    n, k = np.asarray(valores).shape
    return icp.TabelaMultielementar(
        amostras=[f"s{i}" for i in range(n)], tipos=list(tipos), grupos=kw.get("grupos"), unidades_fisicas=kw.get("unidades"),
        sessoes=None, elementos=[f"Cd {111 + j}" for j in range(k)], valores=np.asarray(valores, dtype=float),
        censurado=np.zeros((n, k), dtype=bool) if censurado is None else np.asarray(censurado, dtype=bool),
        limite_censura=np.full((n, k), np.nan), unidade="mg/kg", padrao_interno=kw.get("padrao_interno"))


# ── leitura ─────────────────────────────────────────────────────────────────
def test_fracao_censurada_sem_nenhuma_amostra_e_nan():
    t = _tabela_minima(["branco", "branco"], [[1.0], [2.0]])
    assert math.isnan(t.fracao_censurada()["Cd 111"])
    assert t.fracao_censurada(so_amostras=False)["Cd 111"] == 0.0


def test_formato_sem_delimitador_reconhecivel_cai_para_a_contagem_de_virgulas_e_ponto_e_virgula():
    assert icp._detectar_formato("abc")[0] == ","
    assert icp._detectar_formato("a;b;c\nxyz")[0] in (";", ",")        # o farejador recusa; vale a maioria


def test_celula_ausente_e_nan_nao_censurada():
    v, cens, lim = icp._para_float(None, ".")
    assert math.isnan(v) and cens is False and math.isnan(lim)
    assert math.isnan(icp._para_float(float("nan"), ".")[0])


def test_arquivo_em_latin1_e_lido(tmp_path):
    texto = "Amostra;Cd 111\nPaço;0,1\nB;0,2\n".encode("latin-1")
    t = icp.ler_icp_csv(_escrever(tmp_path, "", binario=texto), unidade="mg/kg", coluna_amostra="Amostra")
    assert t.amostras[0] == "Paço" and t.tipos == ["amostra", "amostra"]          # sem coluna de tipo: tudo e' amostra


def test_coluna_obrigatoria_ausente_coluna_ignorada_e_arquivo_sem_elemento(tmp_path):
    base = "Amostra;Obs;Cd 111\nA;x;0,1\nB;y;0,2\n"
    with pytest.raises(ValueError, match="coluna 'Grupo' nao encontrada"):
        icp.ler_icp_csv(_escrever(tmp_path, base), unidade="mg/kg", coluna_amostra="Amostra", coluna_grupo="Grupo")
    t = icp.ler_icp_csv(_escrever(tmp_path, base, "b.csv"), unidade="mg/kg", coluna_amostra="Amostra")
    assert t.colunas_ignoradas == ["Obs"]
    with pytest.raises(ValueError, match="nenhuma coluna reconhecida"):
        icp.ler_icp_csv(_escrever(tmp_path, "Amostra;Obs\nA;x\nB;y\n", "c.csv"), unidade="mg/kg", coluna_amostra="Amostra")


def test_padrao_interno_fora_dos_elementos_lidos_e_erro(tmp_path):
    with pytest.raises(ValueError, match="padrao interno"):
        icp.ler_icp_csv(_escrever(tmp_path, "Amostra;Cd 111\nA;0,1\nB;0,2\n"), unidade="mg/kg", coluna_amostra="Amostra", padrao_interno="Y 89")


# ── censura, sensibilidade, QC ──────────────────────────────────────────────
def test_censura_recusa_politica_e_limiar_invalidos():
    t = _tabela_minima(["amostra"] * 3, [[1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="politica"):
        icp.aplicar_censura(t, politica="outra")
    with pytest.raises(ValueError, match="limiar"):
        icp.aplicar_censura(t, politica="excluir", limiar=1.5)


def test_sensibilidade_com_uma_politica_inaplicavel_nao_conclui_se_muda():
    cens = [[True], [False], [False]]
    t = _tabela_minima(["amostra"] * 3, [[np.nan], [2.0], [3.0]], censurado=cens)        # censurado sem limite: 'substituir' exige LOD
    r = icp.sensibilidade_censura(t, lambda X, els, reg: {e: True for e in els})
    assert "erro" in r["substituir"] and r["muda"] is None and "nao pode ser aplicada" in r["diferencas"]


def test_qc_com_crm_de_outro_elemento_certificado_nao_positivo_deriva_do_padrao_interno():
    valores = np.array([[10.0 + 3 * i, 100.0 + 6 * i] for i in range(8)])
    t = _tabela_minima(["amostra"] * 7 + ["crm"], valores, padrao_interno="Cd 112")
    t.elementos = ["Cd 111", "Cd 112"]
    r = icp.relatorio_qc(t, crm_certificado={"Zn 66": 5.0, "Cd 111": 0.0})
    assert r["recuperacao_crm_pct"]["Zn 66"] is None and r["recuperacao_crm_pct"]["Cd 111"] is None
    assert any("deriva do padrao interno" in a for a in r["alertas"])


# ── transformacao e fluxo ───────────────────────────────────────────────────
def test_log_recusa_concentracao_negativa_e_transforma_o_resto():
    with pytest.raises(ValueError, match="negativa"):
        icp._transformar(np.array([[1.0, -0.5]]), True)
    np.testing.assert_allclose(icp._transformar(np.array([[0.0, 9.0]]), True), [[0.0, 1.0]])
    assert icp._transformar(np.array([[1.0]]), False).tolist() == [[1.0]]


def _csv_icp(tmp_path, *, censurar=None, efeito=0.0, nome="icp.csv"):
    """2 grupos x 4 unidades fisicas x 2 replicas; `censurar` = (elemento, token) coloca um token na primeira linha."""
    rng = np.random.default_rng(0)
    linhas = [["Amostra", "Grupo", "Unidade", "Cd 111", "Pb 208"]]
    for g, mu in (("A", 1.0), ("B", 1.0 + efeito)):
        for u in range(4):
            for r in range(2):
                linhas.append([f"{g}{u}{r}", g, f"{g}-{u}", f"{mu + rng.normal(0, .1):.3f}".replace(".", ","),
                               f"{2.0 + rng.normal(0, .1):.3f}".replace(".", ",")])
    if censurar:
        j = linhas[0].index(censurar[0])
        linhas[1][j] = censurar[1]
    p = tmp_path / nome
    with open(p, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter=";").writerows(linhas)
    return str(p)


def _cfg(arquivo, **kw):
    base = dict(class_column="Grupo", group_column="Unidade", default_preprocessing="Autoscaling", csv_file=arquivo, icp_unidade="mg/kg",
                icp_politica_censura="substituir", icp_limiar_censura=0.5, icp_log=False, icp_coluna_amostra="Amostra",
                icp_coluna_tipo="", icp_coluna_sessao="", icp_padrao_interno="")
    base.update(kw)
    return SimpleNamespace(**base)


def test_leitor_recusa_quando_todo_elemento_passa_do_limiar_de_censura(tmp_path):
    arq = _csv_icp(tmp_path, censurar=("Cd 111", "<0,05"))
    # o limiar de 0% tira todo elemento com ao menos uma celula censurada: so' o Cd tem censura aqui, entao censuramos o Pb tambem
    texto = open(arq, encoding="utf-8").read().replace("\nA01;A;A-0;", "\nA01;A;A-0;").replace(";2,", ";<0,05;2,", 0)
    linhas = texto.splitlines()
    campos = linhas[2].split(";")
    campos[4] = "<0,05"
    linhas[2] = ";".join(campos)
    (tmp_path / "icp2.csv").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="todos os elementos excederam"):
        icp._leitor_multielementar(_cfg(str(tmp_path / "icp2.csv"), icp_limiar_censura=0.0))


def test_relatorio_com_politica_principal_inaplicavel_e_erro_claro(tmp_path):
    arq = _csv_icp(tmp_path, censurar=("Cd 111", "<LOD"))                    # 'substituir' sem LOD: nao ha' como substituir
    *_, meta = icp._leitor_multielementar(_cfg(arq, icp_politica_censura="excluir"))
    ctx = dict(meta.attrs["multielementar"])
    ctx["contexto"] = dict(ctx["contexto"], politica="substituir")
    with pytest.raises(ValueError, match="ANOVA por elemento impossivel"):
        icp.relatorio_multielementar(ctx, str(tmp_path), str(tmp_path), str(tmp_path), n_boot=0)


def test_relatorio_com_a_politica_alternativa_inaplicavel_registra_o_motivo_e_sem_efeito_nao_faz_figura(tmp_path):
    arq = _csv_icp(tmp_path, censurar=("Cd 111", "<LOD"))
    *_, meta = icp._leitor_multielementar(_cfg(arq, icp_politica_censura="excluir"))
    pasta = tmp_path / "saida"
    pasta.mkdir()
    r = icp.relatorio_multielementar(meta.attrs["multielementar"], str(pasta), str(pasta), str(pasta), n_boot=0)
    assert r["sensibilidade_censura"]["muda"] is None and "substituir" in r["sensibilidade_censura"]["motivo"]
    assert icp._figura_medias({}, "mg/kg", str(pasta / "f.png"), 0.05) is None


def test_qc_com_padrao_interno_de_menos_de_tres_valores_nao_estima_deriva():
    t = _tabela_minima(["amostra"] * 4, [[1.0, 100.0], [2.0, np.nan], [3.0, np.nan], [4.0, 90.0]], padrao_interno="Cd 112")
    r = icp.relatorio_qc(t)
    assert "padrao_interno" not in r and not r["alertas"]
