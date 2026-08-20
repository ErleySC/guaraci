# -*- coding: utf-8 -*-
"""Nenhum metadado de PROVENIENCIA do arquivo de entrada pode chegar a um
artefato de saida.

POR QUE ESTE TESTE EXISTE. Um `.dx` de FT-NIR carrega, no cabecalho JCAMP,
muito mais do que o espectro: `##AUDIT TRAIL` grava data/hora, **operador**
e **local** de cada leitura, e `##TITLE` grava o identificador da amostra.
Nada disso e' necessario para a analise quimiometrica, e tudo isso e'
dado de terceiro quando o arquivo vem de um acervo que nao e' do usuario
do software.

Tirar os arquivos do repositorio NAO fecha esse caminho: quem rodar o
GUARACI sobre eles gera saida. O que fecha e' o software nunca propagar
esses campos -- e um teste que falhe se algum dia propagar.

O teste NAO usa dado real: monta `.dx` sinteticos com strings-sentinela no
lugar dos campos sensiveis, roda o pipeline de verdade e varre TODOS os
artefatos gerados. Roda no CI sem depender de dado local (requisito do
desacoplamento, 2026-08-18).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Sentinelas: strings improvaveis, que so' podem aparecer na saida se
# tiverem vindo do cabecalho do arquivo de entrada.
SENTINELA_OPERADOR = "OPERADOR_SENTINELA_XYZZY"
SENTINELA_LOCAL = "INSTITUICAO_SENTINELA_XYZZY"
SENTINELA_DETECTOR = "DETECTOR_SENTINELA_XYZZY"

#: Extensoes de artefato que sao texto e podem ser varridas byte a byte.
_EXT_TEXTO = {".txt", ".csv", ".md", ".json", ".tex", ".yaml", ".yml", ".log"}


def _sqz(valor: int) -> str:
    """Codifica um inteiro em SQZ (o 1o digito vira letra), como um JCAMP-DX
    real gravado pelo instrumento. Sem isso o arquivo nao exercita o mesmo
    caminho de parsing dos `.dx` de verdade -- e o teste testaria outra coisa."""
    s = str(abs(int(valor)))
    tabela = "@ABCDEFGHI" if valor >= 0 else "@abcdefghi"
    return tabela[int(s[0])] + s[1:]


def _escrever_dx(caminho: Path, titulo: str, n_pontos: int = 64,
                 semente: int = 0) -> None:
    """Escreve um `.dx` ASDF valido, com `##AUDIT TRAIL` e `##$Detector
    model` preenchidos pelas sentinelas -- mesma estrutura de cabecalho dos
    arquivos gravados pelo ABB MB3600."""
    rng = np.random.default_rng(semente)
    x = np.linspace(4000.0, 10000.0, n_pontos)
    y = 0.5 + 0.05 * np.sin(x / 500.0) + rng.normal(0, 0.005, n_pontos)
    yfactor = 1e-6
    y_int = np.round(y / yfactor).astype(np.int64)
    linhas = [
        f"##TITLE={titulo}",
        "##JCAMP-DX=4.24",
        "##DATATYPE=INFRARED SPECTRUM",
        "##$Spectrometer model=MB3600",
        f"##$Detector model={SENTINELA_DETECTOR}",
        "##AUDIT TRAIL= $$ (NUMBER, WHEN, WHO, WHERE, WHAT)",
        f"( 1, <2021/01/06 14:15:37 -03>,  <{SENTINELA_OPERADOR}>,  "
        f"<{SENTINELA_LOCAL}>,  <Measured>)",
        "##XUNITS=1/CM",
        "##YUNITS=ABS",
        f"##FIRSTX={x[0]}",
        f"##LASTX={x[-1]}",
        f"##NPOINTS={n_pontos}",
        "##XFACTOR=1.0",
        f"##YFACTOR={yfactor}",
        "##XYDATA=(X++(Y..Y))",
    ]
    por_linha = 8
    for i in range(0, n_pontos, por_linha):
        linhas.append(str(int(round(x[i])))
                      + "".join(_sqz(v) for v in y_int[i:i + por_linha]))
    linhas.append("##END=")
    caminho.write_text("\n".join(linhas), encoding="latin-1")


@pytest.fixture(scope="module")
def saida_pipeline_com_sentinelas(pq, tmp_path_factory):
    """Roda executar() sobre .dx sinteticos contaminados e devolve a pasta
    de saida gerada."""
    from conftest import achar_pastas_run

    base = tmp_path_factory.mktemp("sanitizacao")
    entrada = base / "entrada"
    # Duas classes x 2 pontos fisicos x 3 replicas = 12 espectros: o minimo
    # para o pipeline classificar e para GroupKFold ter grupos.
    for cls, cod in (("ClasseA", "ACA"), ("ClasseB", "BAB")):
        pasta_cls = entrada / cls
        pasta_cls.mkdir(parents=True, exist_ok=True)
        for ponto in (1, 2):
            for trip in (1, 2, 3):
                titulo = f"{cod}-0{ponto}-01-2099-T{trip}"
                _escrever_dx(pasta_cls / f"{titulo}.dx", titulo,
                             semente=hash((cod, ponto, trip)) % 10_000)

    cfg = pq.Config(
        modo="dx", pasta_entrada=str(entrada),
        pasta_saida_raiz=str(base / "saida"),
        wn_min=4000.0, wn_max=10000.0,
        n_splits_cv=2, n_repeats_cv=1, n_permutacoes=5,
        n_permutacoes_wold=5, n_bootstrap_vip=3, n_bootstrap_bca=20,
        n_monte_carlo=3, max_lvs=3, frac_holdout=0.0,
    )
    pq.executar(cfg)
    runs = achar_pastas_run(cfg.pasta_saida_raiz)
    assert runs, "executar() nao criou pasta de saida"
    return Path(runs[0])


def _varrer(pasta: Path):
    """Devolve {sentinela: [arquivos onde apareceu]} varrendo todo artefato
    de texto sob `pasta`, recursivamente."""
    achados: dict = {SENTINELA_OPERADOR: [], SENTINELA_LOCAL: [],
                     SENTINELA_DETECTOR: []}
    for caminho in pasta.rglob("*"):
        if not caminho.is_file() or caminho.suffix.lower() not in _EXT_TEXTO:
            continue
        try:
            texto = caminho.read_text(encoding="utf-8", errors="replace")
        except OSError:                                    # noqa: PERF203
            continue
        for sentinela in achados:
            if sentinela in texto:
                achados[sentinela].append(str(caminho.relative_to(pasta)))
    return achados


def test_nenhum_metadado_de_proveniencia_chega_a_artefato_de_texto(
        saida_pipeline_com_sentinelas):
    """Operador, local e modelo de detector do cabecalho JCAMP nunca podem
    aparecer em relatorio, tabela, log ou model card."""
    achados = _varrer(saida_pipeline_com_sentinelas)
    vazando = {s: arqs for s, arqs in achados.items() if arqs}
    assert not vazando, (
        "metadado de proveniencia do arquivo de entrada vazou para a "
        f"saida: {vazando}")


def test_nenhum_metadado_de_proveniencia_no_modelo_serializado(
        saida_pipeline_com_sentinelas):
    """O `.joblib` e' distribuido junto com o modelo -- e' o artefato com
    maior chance de sair da maquina de quem treinou."""
    import joblib

    modelos = list(saida_pipeline_com_sentinelas.rglob("*.joblib"))
    assert modelos, "nenhum modelo .joblib foi salvo"
    for caminho in modelos:
        bruto = caminho.read_bytes()
        for sentinela in (SENTINELA_OPERADOR, SENTINELA_LOCAL,
                          SENTINELA_DETECTOR):
            assert sentinela.encode() not in bruto, (
                f"'{sentinela}' encontrado nos bytes de {caminho.name}")
        pkg = joblib.load(caminho)
        texto = repr(pkg)
        for sentinela in (SENTINELA_OPERADOR, SENTINELA_LOCAL,
                          SENTINELA_DETECTOR):
            assert sentinela not in texto, (
                f"'{sentinela}' encontrado no conteudo de {caminho.name}")


def test_parse_dx_nao_devolve_campos_de_proveniencia(tmp_path):
    """Barreira na origem: o proprio parser nao deve extrair esses campos.

    Testa o contrato de `parse_dx` diretamente -- se um dia alguem
    adicionar `##AUDIT TRAIL` a lista de campos lidos "porque pode ser
    util", este teste falha antes de o dado chegar a qualquer saida.
    """
    from guaraci.dados_io import parse_dx

    arq = tmp_path / "amostra.dx"
    _escrever_dx(arq, "ACA-01-01-2099-T1")
    x, y = parse_dx(str(arq))
    assert x.size == y.size > 0
    # parse_dx devolve so' os vetores numericos: nao ha' onde um campo de
    # texto se esconder. O teste abaixo garante que continua assim.
    assert np.issubdtype(x.dtype, np.floating)
    assert np.issubdtype(y.dtype, np.floating)


def test_metadados_exportados_nao_contem_identificador_de_amostra(
        saida_pipeline_com_sentinelas):
    """`metadados.csv` exportava `title_original`/`arquivo`/`mae_id` crus --
    isto e', o identificador da amostra do acervo de origem, incluindo
    codigo de especie, data de coleta e (quando adulterada) adulterante e
    teor. Sao metadados de terceiro; o pipeline nao precisa deles no disco
    para nada, e `amostras_identificadores.csv` ja usa IDs anonimos
    (S000, S001...).
    """
    import pandas as pd

    metas = list(saida_pipeline_com_sentinelas.rglob("metadados.csv"))
    if not metas:
        pytest.skip("pipeline nao exportou metadados.csv nesta configuracao")
    proibidas = {"title_original", "arquivo", "mae_id", "data", "cod"}
    for caminho in metas:
        cols = set(pd.read_csv(caminho, sep=";", decimal=",", nrows=1).columns)
        vazando = cols & proibidas
        assert not vazando, (
            f"{caminho.name} exporta colunas de identificacao da amostra de "
            f"origem: {sorted(vazando)}")


def test_amostras_identificadores_usa_id_anonimo(
        saida_pipeline_com_sentinelas):
    """Contra-prova do teste acima: a tabela que existe PARA identificar
    amostras usa ID sequencial anonimo, nao o identificador do acervo."""
    import pandas as pd

    csvs = list(saida_pipeline_com_sentinelas.rglob(
        "amostras_identificadores.csv"))
    assert csvs, "amostras_identificadores.csv nao foi gerado"
    df = pd.read_csv(csvs[0], sep=";", decimal=",")
    assert "ID" in df.columns
    assert all(str(v).startswith("S") for v in df["ID"].head(5)), (
        "coluna ID deveria ser sequencial anonima (S000, S001...)")


def test_log_do_pipeline_nao_ecoa_nome_de_arquivo_de_entrada(
        saida_pipeline_com_sentinelas):
    """Nome de arquivo tambem e' identificador: `CAP-04-11-2099-AD-S-1,03%_T1.dx`
    carrega especie, data, adulterante e teor no proprio nome."""
    for caminho in saida_pipeline_com_sentinelas.rglob("*"):
        if not caminho.is_file() or caminho.suffix.lower() not in _EXT_TEXTO:
            continue
        texto = caminho.read_text(encoding="utf-8", errors="replace")
        assert ".dx" not in texto.lower() or "metadados" in caminho.name, (
            f"{caminho.name} ecoa nome de arquivo .dx de entrada")


def test_sanitizar_metadados_e_idempotente_e_preserva_o_util():
    """A funcao de sanitizacao remove identificacao mas mantem o que a
    analise precisa (classe, teor, pureza, se e' replica)."""
    import pandas as pd

    from guaraci.dados_io import sanitizar_metadados

    df = pd.DataFrame([{
        "title_original": "CAP-04-11-2099-AD-S-1,03%_T1",
        "arquivo": "CAP-04-11-2099-AD-S-1,03%_T1.dx",
        "cod": "CAP", "data": "04-11-2099", "mae_id": "CAP-04-11-2099-S1.03",
        "especie": "Castanha do Para", "teor": 1.03, "puro": False,
        "adulterante": "S", "adulterante_nome": "soja", "triplicata": 1,
        "subpasta": "Castanha", "cod_conhecido": True,
    }])
    limpo = sanitizar_metadados(df)
    for col in ("title_original", "arquivo", "cod", "data", "mae_id",
                "subpasta"):
        assert col not in limpo.columns, f"'{col}' deveria ter sido removida"
    for col in ("especie", "teor", "puro", "adulterante_nome", "triplicata"):
        assert col in limpo.columns, f"'{col}' e' util e nao deveria sumir"
    assert "grupo_replica" in limpo.columns, (
        "o vinculo entre replicas do mesmo ponto fisico precisa sobreviver, "
        "de forma anonima -- e' o que sustenta a validacao group-aware")
    assert limpo.equals(sanitizar_metadados(limpo)), "nao e' idempotente"


def test_sanitizar_metadados_mantem_replicas_no_mesmo_grupo():
    """`grupo_replica` tem que agrupar exatamente como o `mae_id` agrupava:
    se ele quebrar, a validacao group-aware -- o diferencial do projeto --
    passa a vazar replica entre treino e teste sem ninguem perceber."""
    import pandas as pd

    from guaraci.dados_io import sanitizar_metadados

    df = pd.DataFrame([
        {"mae_id": "AAA-01-01-2099", "especie": "A", "triplicata": 1},
        {"mae_id": "AAA-01-01-2099", "especie": "A", "triplicata": 2},
        {"mae_id": "BBB-02-02-2099", "especie": "B", "triplicata": 1},
        {"mae_id": "AAA-01-01-2099", "especie": "A", "triplicata": 3},
    ])
    g = sanitizar_metadados(df)["grupo_replica"].tolist()
    assert g[0] == g[1] == g[3], "replicas do mesmo ponto se separaram"
    assert g[2] != g[0], "pontos fisicos distintos foram fundidos"
    assert not any("AAA" in str(v) or "BBB" in str(v) for v in g), (
        "grupo_replica ainda carrega o identificador original")
