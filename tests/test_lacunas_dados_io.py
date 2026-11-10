"""Ramos de borda de guaraci.dados_io (correcoes locais, JCAMP-DX, CSV de treino, mapa de grupos, carga de pasta): um teste por ramo.

Tudo sintetico e minusculo; o diretorio de correcoes locais e' sempre trocado por uma pasta temporaria (nunca toca o estado real).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from guaraci import csv_robusto
from guaraci import dados_io as dio

_SQZ = "@ABCDEFGHI"


def _y(v: int) -> str:
    return _SQZ[v] if v >= 0 else "abcdefghi"[-v - 1]


def escrever_dx(caminho: Path, titulo: str, pontos: dict, firstx: float, lastx: float, npoints: int, extras=()) -> Path:
    """.dx minimo: um ponto SQZ por linha ('x' + letra); `pontos` = {x: y_inteiro_de_-9_a_9}."""
    linhas = [f"##TITLE={titulo}" if titulo else "##JCAMP-DX=4.24", "##XFACTOR=1", "##YFACTOR=1", f"##FIRSTX={firstx}", f"##LASTX={lastx}",
              f"##NPOINTS={npoints}", *extras, "##XYDATA=(X++(Y..Y))"]
    linhas += [f"{int(x)}{_y(y)}" for x, y in pontos.items()]
    linhas.append("##END=")
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return caminho


@pytest.fixture
def locais(tmp_path, monkeypatch):
    """Pasta de correcoes locais temporaria; as correcoes voltam a ficar desligadas ao final."""
    monkeypatch.setattr(dio, "_DIR_LOCAL", tmp_path / "local")
    (tmp_path / "local").mkdir()
    yield tmp_path / "local"
    dio.definir_correcoes_locais(False)


# ── correcoes locais ────────────────────────────────────────────────────────
def test_csv_local_ausente_e_vazio_comentario_e_linha_em_branco_sao_ignorados(locais):
    assert dio._ler_csv_local("alias_mae_id.csv", 2) == []
    (locais / "alias_mae_id.csv").write_text("# motivo\n\na;b\n", encoding="utf-8")
    assert dio._ler_csv_local("alias_mae_id.csv", 2) == [["a", "b"]]


def test_csv_local_ilegivel_e_erro_claro(locais, monkeypatch):
    (locais / "alias_mae_id.csv").write_text("a;b\n", encoding="utf-8")

    def nao_le(self, *a, **k):
        raise OSError("sem permissao")
    monkeypatch.setattr(Path, "read_text", nao_le)
    with pytest.raises(RuntimeError, match="ilegiveis"):
        dio._ler_csv_local("alias_mae_id.csv", 2)


def test_correcao_de_titulo_com_alias_de_mae_id_conta_titulo_e_alias(locais):
    (locais / "correcoes_titulo.csv").write_text("ABC-03-01-2099-AD-A-5%-T1;ABC;03-01-2099;A;5;1\n", encoding="utf-8")
    (locais / "alias_mae_id.csv").write_text("ABC-03-01-2099-A5.00;ABC-ALIAS\n", encoding="utf-8")
    (locais / "pureza_indeterminada.csv").write_text("TITULO-RUIM;motivo registrado\n", encoding="utf-8")
    dio.definir_correcoes_locais(True)
    info = dio.parse_title("ABC-03-01-2099-AD-A-5%-T1")
    assert info["mae_id"] == "ABC-ALIAS" and info["teor"] == 5.0
    r = dio.resumo_correcoes_locais()
    assert r["ativo"] and r["titulos_corrigidos"] == 1 and r["alias_aplicados"] == 1 and len(r["arquivos"]) == 3


@pytest.mark.parametrize("valor, ligou", [("1", True), ("Sim", True), ("on", True), ("0", False), ("", False)])
def test_variavel_de_ambiente_liga_as_correcoes_locais(locais, monkeypatch, valor, ligou):
    monkeypatch.setenv("GUARACI_APLICAR_CORRECOES_LOCAIS", valor)
    assert dio._ativar_correcoes_por_ambiente() is ligou
    assert dio.resumo_correcoes_locais()["ativo"] is ligou


# ── titulo, selecao de amostras, divisao ────────────────────────────────────
def test_extract_dx_title_sem_titulo_com_dados_primeiro_e_arquivo_ausente(tmp_path):
    assert dio.extract_dx_title(str(tmp_path / "nao_existe.dx")) is None
    (tmp_path / "a.dx").write_text("##XYDATA=(X++(Y..Y))\n##TITLE=tarde demais\n", encoding="utf-8")
    assert dio.extract_dx_title(str(tmp_path / "a.dx")) is None
    (tmp_path / "b.dx").write_text("##JCAMP-DX=4.24\n##NPOINTS=3\n", encoding="utf-8")        # nem titulo nem dados: acaba o arquivo
    assert dio.extract_dx_title(str(tmp_path / "b.dx")) is None


def test_kennard_stone_sem_amostras_e_com_uma_so():
    assert dio.kennard_stone(np.zeros((0, 3)), 2).size == 0
    X = np.random.default_rng(0).normal(size=(6, 3))
    assert dio.kennard_stone(X, 1).tolist() == [int(np.argmax(np.sum((X - X.mean(axis=0)) ** 2, axis=1)))]


def test_divisao_por_especie_sem_amostras_para_o_splitter_devolve_none():
    cfg = SimpleNamespace(cal_val_split="kennard_stone", frac_cal=0.7, seed=0)
    X = np.random.default_rng(1).normal(size=(1, 4))
    assert dio._split_cal_val_por_especie(X, np.array([1.0]), None, cfg) is None


def test_sintetico_numa_faixa_que_nao_e_a_do_modelo_nao_remapeia_o_eixo():
    cfg = SimpleNamespace(seed=0, n_synthetic_points=40, wn_min=400.0, wn_max=400.0, n_per_class=3, n_synthetic_replicates=1,
                          synthetic_adulterants=())
    wn, X, *_ = dio.generate_synthetic_data(cfg)
    assert wn[0] == pytest.approx(4000.0) and wn[-1] == pytest.approx(400.0) and X.shape[1] == 40


# ── CSV de treino ───────────────────────────────────────────────────────────
def _csv(tmp_path, texto, nome="d.csv"):
    p = tmp_path / nome
    p.write_text(texto, encoding="utf-8")
    return p


def test_tabela_csv_inexistente_vazia_so_cabecalho_e_malformada(tmp_path):
    with pytest.raises(FileNotFoundError):
        dio._ler_tabela_csv(tmp_path / "nao_existe.csv")
    with pytest.raises(ValueError, match="esta vazio"):
        dio._ler_tabela_csv(_csv(tmp_path, ""))
    with pytest.raises(ValueError, match="so' cabecalho"):
        dio._ler_tabela_csv(_csv(tmp_path, "classe,4000,4001\n", "c.csv"))


def test_tabela_csv_que_nao_e_csv_vira_erro_de_interpretacao(tmp_path, monkeypatch):
    import pandas.errors as pe

    def quebra(*a, **k):
        raise pe.ParserError("linha 2 tem colunas demais")
    monkeypatch.setattr(csv_robusto, "ler_csv", quebra)
    with pytest.raises(ValueError, match="Nao foi possivel interpretar"):
        dio._ler_tabela_csv(_csv(tmp_path, "a,b\n1,2\n"))


def test_tabela_csv_com_virgula_decimal_que_nao_se_deixa_reler_mantem_a_primeira_leitura(tmp_path, monkeypatch):
    chamadas = []
    original = csv_robusto.ler_csv

    def instavel(caminho, **kw):
        chamadas.append(kw)
        if kw.get("decimal") == ",":
            raise ValueError("nao deu")
        return original(caminho, **kw)
    monkeypatch.setattr(csv_robusto, "ler_csv", instavel)
    df = dio._ler_tabela_csv(_csv(tmp_path, "classe;4000;4001\nA;1,5;2,5\nB;3,5;4,5\n"))
    assert len(chamadas) == 2 and list(df.columns)[0] == "classe"      # tentou o decimal "," e voltou ao que ja tinha lido


def test_localizar_texto_limita_a_tres_achados_e_conta_o_resto():
    bloco = pd.DataFrame({"4000": ["a", "b", "c", "d", "e"], "4001": [1, 2, 3, 4, 5]})
    texto = dio._localizar_texto(bloco)
    assert texto.count("linha") == 3 and "(+2 outras)" in texto


def test_mapa_de_grupos_recusa_uma_coluna_grupo_vazio_e_grupos_conflitantes(tmp_path, monkeypatch):
    monkeypatch.setattr(dio, "_ler_tabela_csv", lambda caminho: pd.DataFrame({"id": ["A", "B"]}))
    with pytest.raises(ValueError, match=">=2 colunas"):
        dio.carregar_mapa_grupos("qualquer.csv")
    monkeypatch.undo()
    with pytest.raises(ValueError, match="sem grupo"):
        dio.carregar_mapa_grupos(str(_csv(tmp_path, "id,grupo\nA,g1\nB,\n", "g2.csv")))
    with pytest.raises(ValueError, match="grupos diferentes"):
        dio.carregar_mapa_grupos(str(_csv(tmp_path, "id,grupo\nA,g1\nA,g2\n", "g3.csv")))
    assert dio.carregar_mapa_grupos(str(_csv(tmp_path, "id,grupo\nA,g1\nA,g1\n", "g4.csv"))) == ("id", {"A": "g1"})


_CABECA = "classe,teor,4000,4001,4002"


def test_csv_com_grupos_recusa_coluna_ausente_grupo_vazio_id_ausente_e_pouca_coluna_espectral(tmp_path):
    ok = f"{_CABECA}\nA,1,0.1,0.2,0.3\nB,2,0.4,0.5,0.6\n"
    with pytest.raises(ValueError, match="concentracao 'zzz'"):
        dio.load_csv_com_grupos(_csv(tmp_path, ok), "classe", "zzz")
    com_grupo = f"{_CABECA},grp\nA,1,0.1,0.2,0.3,g1\nB,2,0.4,0.5,0.6,\n"
    with pytest.raises(ValueError, match="valores vazios"):
        dio.load_csv_com_grupos(_csv(tmp_path, com_grupo, "g.csv"), "classe", "teor", col_grupo="grp")
    grupos = _csv(tmp_path, "amostra,grupo\nA,g1\nB,g2\n", "mapa.csv")
    with pytest.raises(ValueError, match="1a coluna do arquivo de grupos"):
        dio.load_csv_com_grupos(_csv(tmp_path, ok, "o.csv"), "classe", "teor", arquivo_grupos=str(grupos))
    so_uma = "classe,teor,4000\nA,1,0.1\nB,2,0.4\n"
    with pytest.raises(ValueError, match="menos de 2 colunas numericas de variaveis"):
        dio.load_csv_com_grupos(_csv(tmp_path, so_uma, "u.csv"), "classe", "teor")


def test_csv_com_celula_de_texto_no_espectro_e_erro_que_diz_onde(tmp_path):
    texto = f"{_CABECA}\nA,1,0.1,xyz,0.3\nB,2,0.4,0.5,0.6\n"
    with pytest.raises(ValueError, match="nao numericos"):
        dio.load_csv_com_grupos(_csv(tmp_path, texto), "classe", "teor")


# ── JCAMP-DX ────────────────────────────────────────────────────────────────
def test_decodificacao_dup_com_contagem_de_varios_digitos():
    # JCAMP-DX 4.24: DUP e' a contagem TOTAL de ocorrencias e continua nos digitos seguintes ('S5' = 15 ocorrencias).
    # O teste antigo fixava a leitura errada ('5' ignorado, 2 ocorrencias) -- achado do checkup de MIR/Raman.
    SQZ, DIF, DUP = {"A": 1}, {"J": 1}, {"S": 1}
    x, ys = dio._decodificar_linha_asdf("100A5S5", SQZ, DIF, DUP)          # A5 = 15 ; S5 = 15 ocorrencias de 15
    assert x == 100.0 and ys == [15.0] * 15


def test_parse_dx_reancora_linha_sem_abscissa_e_ignora_ponto_fora_da_grade(tmp_path):
    p = escrever_dx(tmp_path / "a.dx", "T", {100: 1, 101: 2, 102: 3, 103: 4}, 100, 103, 4)
    with open(p, "a", encoding="utf-8") as f:
        f.write("A5\n9999B\n")                                             # sem abscissa; e abscissa muito alem da grade
    texto = p.read_text(encoding="utf-8").replace("##END=\n", "") + "##END=\n"
    p.write_text(texto, encoding="utf-8")
    x, y = dio.parse_dx(str(p))
    assert len(x) == 4 and np.isfinite(y).all()


def test_parse_dx_interpola_buracos_com_eixo_decrescente_e_ignora_npoints_ilegivel(tmp_path):
    # firstx > lastx (convencao FTIR) e o ponto de x=107 faltando: a interpolacao precisa de eixo crescente
    pontos = {109: 1, 108: 2, 106: 4, 105: 5, 104: 6, 103: 7, 102: 8, 101: 9, 100: 0}
    p = escrever_dx(tmp_path / "d.dx", "T", pontos, 109, 100, 10)
    x, y = dio.parse_dx(str(p))
    assert x[0] == 109 and x[-1] == 100 and np.isfinite(y).all() and y[2] == pytest.approx(3.0)
    q = escrever_dx(tmp_path / "n.dx", "T", {100: 1, 101: 2, 102: 3}, 100, 102, 3)
    q.write_text(q.read_text(encoding="utf-8").replace("##NPOINTS=3", "##NPOINTS=abc"), encoding="utf-8")
    x2, y2 = dio.parse_dx(str(q))
    assert len(y2) == 3 and x2[0] == 100 and x2[-1] == 102             # sem NPOINTS valido: modo legado, grade pelos extremos


def test_parse_dx_modo_legado_pula_linha_sem_abscissa_e_para_na_diretiva_seguinte(tmp_path):
    p = tmp_path / "l.dx"
    p.write_text("##TITLE=T\n##XYDATA=(X++(Y..Y))\n100A\nB\n101C\n##PEAK TABLE=(XY..XY)\n102D\n##END=\n", encoding="utf-8")
    x, y = dio.parse_dx(str(p))
    assert y.tolist() == [1.0, 3.0] and x.tolist() == [100.0, 101.0]   # 'B' (sem abscissa) saiu; depois de '##PEAK' nao le mais dados


def test_parse_dx_modo_legado_com_extremos_sem_npoints(tmp_path):
    p = tmp_path / "e.dx"
    p.write_text("##TITLE=T\n##FIRSTX=100\n##LASTX=110\n##XYDATA=(X++(Y..Y))\n100A\n105B\n##END=\n", encoding="utf-8")
    x, y = dio.parse_dx(str(p))
    assert x.tolist() == [100.0, 110.0] and y.tolist() == [1.0, 2.0]


# ── sanitizacao e varredura rapida ──────────────────────────────────────────
def test_sanitizar_metadados_mantem_vazio_o_grupo_de_quem_nao_tinha_mae_id():
    df = pd.DataFrame({"mae_id": ["A", None, "A", "B"], "especie": list("wxyz"), "arquivo": list("abcd")})
    limpo = dio.sanitizar_metadados(df)
    assert limpo["grupo_replica"].tolist() == ["G000", "", "G000", "G001"] and "mae_id" not in limpo and "arquivo" not in limpo


def test_prescan_conta_arquivo_ilegivel_e_arquivo_sem_lastx(tmp_path):
    (tmp_path / "pasta.dx").mkdir()                                       # abrir diretorio como arquivo: OSError
    (tmp_path / "sem_lastx.dx").write_text("##TITLE=ABC-03-01-2099-T1\n##XYDATA=(X++(Y..Y))\n", encoding="utf-8")
    escrever_dx(tmp_path / "ok.dx", "ABC-03-01-2099-T2", {100: 1, 101: 2}, 100, 101, 2)
    r = dio.prescan_dx(str(tmp_path))
    assert r["n_sem_cabecalho"] == 2 and r["n_arquivos"] == 3


# ── load_dx ─────────────────────────────────────────────────────────────────
def test_load_dx_pasta_inexistente_e_pasta_sem_arquivo_espectral(tmp_path):
    with pytest.raises(FileNotFoundError, match="does NOT exist"):
        dio.load_dx(str(tmp_path / "nao_existe"))
    (tmp_path / "vazia").mkdir()
    (tmp_path / "vazia" / "leiame.md").write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no known spectral files"):
        dio.load_dx(str(tmp_path / "vazia"))


def test_load_dx_so_com_arquivos_que_falham_ou_sem_dado_e_erro(tmp_path, capsys):
    (tmp_path / "ruim.dx").write_text("##XFACTOR=abc\n", encoding="utf-8")        # float('abc') no cabecalho
    (tmp_path / "vazio.dx").write_text("##TITLE=T\n##XYDATA=(X++(Y..Y))\n##END=\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No valid spectra"):
        dio.load_dx(str(tmp_path))
    saida = capsys.readouterr().out
    assert "ruim.dx" in saida and "has no data" in saida


def _pasta_dx(tmp_path):
    p = tmp_path / "dados"
    p.mkdir()
    pts = {100: 1, 101: 2, 102: 3, 103: 4, 104: 5}
    escrever_dx(p / "ABC-03-01-2099-T1.dx", "ABC-03-01-2099-T1", pts, 100, 104, 5)                       # titulo valido (mae_id)
    escrever_dx(p / "ABC-03-01-2099-T2.dx", "ABC-03-01-2099-T2", pts, 100, 104, 5)
    escrever_dx(p / "XY_5%_solto.dx", "titulo fora do padrao", pts, 100, 104, 5)                        # titulo que nao casa: fallback pelo nome
    escrever_dx(p / "ZZZ-03-01-2099-T1.dx", "ZZZ-03-01-2099-T1", pts, 100, 104, 5)                      # cod nao mapeado
    escrever_dx(p / "faixa_outra.dx", "FAI-03-01-2099-T1", {1000 + i: 1 for i in range(5)}, 1000, 1004, 5)  # faixa espectral incompativel
    return p


def test_load_dx_titulo_valido_fallback_pelo_nome_faixa_incompativel_e_orfaos(tmp_path, capsys, locais):
    (locais / "pureza_indeterminada.csv").write_text("ZZZ-03-01-2099-T1;pureza nao determinavel\n", encoding="utf-8")
    dio.definir_correcoes_locais(True)
    wn, X, rot, conc, mae, meta = dio.load_dx(str(_pasta_dx(tmp_path)), extrair_conc=True)
    saida = capsys.readouterr().out
    assert X.shape[0] == 3                                  # 2 validos + 1 fallback; excluido por pureza e fora de faixa saem
    assert "EXCLUIDO" in saida and "DISCARDING 1 files" in saida and "do not follow the ##TITLE= pattern" in saida
    assert mae is not None and any(str(m).startswith("orfao_") for m in mae) and "1 isolated orphans" in saida
    assert conc is not None and 5.0 in conc.tolist()        # teor extraido do nome do arquivo no fallback


def test_load_dx_fallback_por_prefixo_de_codigo_conhecido_subpasta_e_partes_do_nome(tmp_path, monkeypatch):
    monkeypatch.setitem(dio.CODIGO_ESPECIE, "QQQ", "Especie Q")
    pts = {100: 1, 101: 2, 102: 3}
    raiz = tmp_path / "r"
    (raiz / "classeA").mkdir(parents=True)
    escrever_dx(raiz / "classeA" / "QQQ-1.dx", "x", pts, 100, 102, 3)          # prefixo conhecido -> nome canonico
    escrever_dx(raiz / "classeA" / "outro.dx", "x", pts, 100, 102, 3)          # sem prefixo -> nome da subpasta
    solta = tmp_path / "s"
    solta.mkdir()
    escrever_dx(solta / "alfa-beta-gama.dx", "x", pts, 100, 102, 3)            # sem subpasta -> parte do nome
    escrever_dx(solta / "unico.dx", "x", pts, 100, 102, 3)                     # indice fora do nome -> nome inteiro
    _, _, rot1, *_ = dio.load_dx(str(raiz), use_parse_title=False)
    assert sorted(rot1) == ["Especie Q", "classeA"]
    _, _, rot2, *_ = dio.load_dx(str(solta), class_part=1, use_parse_title=False)
    assert sorted(rot2) == ["beta", "unico"]
    _, _, rot3, conc3, mae3, _ = dio.load_dx(str(solta), extrair_conc=False, use_parse_title=False)
    assert conc3 is None and mae3 is None


def test_load_dx_avisa_arquivo_com_erro_de_leitura_e_ainda_carrega_os_demais(tmp_path, capsys):
    pasta = tmp_path / "m"
    pasta.mkdir()
    pts = {100: 1, 101: 2, 102: 3}
    escrever_dx(pasta / "ok1.dx", "x", pts, 100, 102, 3)
    escrever_dx(pasta / "ok2.dx", "x", pts, 100, 102, 3)
    (pasta / "ruim.dx").write_text("##XFACTOR=abc\n", encoding="utf-8")
    _, X, *_ = dio.load_dx(str(pasta), use_parse_title=False)
    assert X.shape[0] == 2 and "1 files with parsing errors" in capsys.readouterr().out


def test_divisao_por_especie_com_fracao_invalida_para_o_splitter_devolve_none():
    cfg = SimpleNamespace(cal_val_split="group_shuffle", frac_cal=1.0, seed=0)         # train_size=1.0: o GroupShuffleSplit recusa
    X = np.random.default_rng(2).normal(size=(12, 4))
    mae = np.repeat(np.arange(6), 2).astype(str)
    assert dio._split_cal_val_por_especie(X, np.arange(12.0), mae, cfg) is None


def test_csv_com_concentracao_em_texto_e_erro_claro(tmp_path):
    texto = f"{_CABECA}\nA,abc,0.1,0.2,0.3\nB,2,0.4,0.5,0.6\n"
    with pytest.raises(ValueError, match="concentracao 'teor' tem valores nao numericos"):
        dio.load_csv_com_grupos(_csv(tmp_path, texto), "classe", "teor")


def test_prescan_le_ate_o_fim_quando_o_arquivo_nao_tem_secao_de_dados(tmp_path):
    (tmp_path / "so_cabecalho.dx").write_text("##TITLE=ABC-03-01-2099-T1\n##LASTX=400\n", encoding="utf-8")
    r = dio.prescan_dx(str(tmp_path))
    assert r["n_sem_cabecalho"] == 0 and r["n_apos_descarte"] == 1 and r["faixa_dominante"] == 400.0


def test_load_dx_de_arquivo_sem_titulo_conta_titulo_fora_do_padrao(tmp_path, capsys):
    pasta = tmp_path / "sem_titulo"
    pasta.mkdir()
    escrever_dx(pasta / "a.dx", "", {100: 1, 101: 2, 102: 3}, 100, 102, 3)
    _, X, rot, *_ = dio.load_dx(str(pasta))
    assert X.shape[0] == 1 and "No ##TITLE= follows the optional GUARACI title pattern" in capsys.readouterr().out


def test_leitor_de_imagem_sem_metadados_nao_mexe_na_garantia_de_agrupamento(monkeypatch):
    from guaraci import dados_imagem
    resultado = (np.arange(3.0), np.zeros((1, 3)), np.array(["a"]), None, None, None)
    monkeypatch.setattr(dados_imagem, "load_images", lambda *a, **kw: resultado)
    cfg = SimpleNamespace(input_folder="x", image_crop=(0, 0, 1, 1), include_image_texture=False, grouping_guarantee="none")
    assert dio._leitor_imagem(cfg) is resultado and cfg.grouping_guarantee == "none"
