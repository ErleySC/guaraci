"""Testes de `scripts/benchmark_tecator.py` — so' a logica de PARSING (pura,
sem rede). `rodar_benchmark()` baixa dado real da internet a cada chamada
e nao e' testada aqui (rede indisponivel/instavel em CI nao deve quebrar a
suite principal) -- roda manualmente via `python scripts/benchmark_tecator.py`.
"""
import os
import sys

import pytest

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import benchmark_tecator as bt  # noqa: E402


def _bloco_amostra(absorb100, moisture, fat, protein):
    """Monta as 25 linhas de UMA amostra no formato real do arquivo Tecator
    (20 linhas de 5 absorbancias, 4 linhas de PCs fake, 1 linha final com
    2 PCs fake + umidade/gordura/proteina)."""
    linhas = []
    for i in range(0, 100, 5):
        linhas.append(" ".join(f"{v:.5f}" for v in absorb100[i:i + 5]))
    for _ in range(4):
        linhas.append("0.1 0.2 0.3 0.4 0.5")
    linhas.append(f"0.1 0.2 {moisture} {fat} {protein}")
    assert len(linhas) == 25
    return "\n".join(linhas)


def _texto_tecator_fake(n_amostras: int) -> str:
    corpo = [
        "texto de descricao qualquer, extrapolation_examples=25 aparece aqui"
        " so' como prosa (deve ser ignorado -- ver rfind em _parsear_tecator)",
        "",
        "real_in=122",
        "real_out=3",
        "training_examples=172",
        "test_examples=43",
        "extrapolation_examples=25",
        "",
    ]
    for i in range(n_amostras):
        absorb = [2.0 + 0.001 * (i + j) for j in range(100)]
        corpo.append(_bloco_amostra(absorb, moisture=60.0 + i,
                                     fat=20.0 + i, protein=15.0 + i))
    return "\n".join(corpo)


def test_parsear_tecator_extrai_espectro_e_teor_gordura():
    """Com N amostras completas, o parser deve devolver um DataFrame com
    N linhas, 100 colunas espectrais numericas + teor_gordura + split."""
    texto = _texto_tecator_fake(3)
    df = bt._parsear_tecator(texto)
    assert len(df) == 3
    wn_cols = [c for c in df.columns if c not in ("teor_gordura", "split_original")]
    assert len(wn_cols) == 100
    # teor_gordura da amostra 0 foi montado como 20.0 (ver _texto_tecator_fake)
    assert df["teor_gordura"].iloc[0] == pytest.approx(20.0)
    assert df["teor_gordura"].iloc[1] == pytest.approx(21.0)


def test_parsear_tecator_marcador_duplicado_nao_confunde_parser():
    """O marcador de inicio dos dados aparece 1x na prosa descritiva do
    arquivo REAL do Tecator antes de aparecer de verdade no cabecalho --
    _parsear_tecator deve usar a ULTIMA ocorrencia (rfind), nao a primeira,
    senao tenta interpretar texto como numero e quebra (bug real encontrado
    ao integrar com o arquivo real em 2026-07-13)."""
    texto = _texto_tecator_fake(2)
    assert texto.count("extrapolation_examples=25") == 2  # prosa + cabecalho real
    df = bt._parsear_tecator(texto)  # nao deve levantar ValueError
    assert len(df) == 2


def test_parsear_tecator_split_train_test_extrapolation():
    """Amostras 0-171 = train, 172-214 = test, 215+ = extrapolation
    (convencao padrao da literatura para este dataset)."""
    texto = _texto_tecator_fake(216)
    df = bt._parsear_tecator(texto)
    assert (df["split_original"].iloc[:172] == "train").all()
    assert (df["split_original"].iloc[172:215] == "test").all()
    assert (df["split_original"].iloc[215:] == "extrapolation").all()


def test_parsear_tecator_amostra_com_absorbancias_faltando_levanta_erro():
    """Amostra corrompida (menos de 100 absorbancias) deve falhar alto e
    claro, nao silenciosamente -- dado externo nao confiavel, corrupcao
    parcial nao pode virar um espectro errado sem avisar (mesma filosofia
    dos guards de dados_io.py para o dataset proprio)."""
    texto_valido = _texto_tecator_fake(1)
    linhas = texto_valido.splitlines()
    # remove um valor da 1a linha de absorbancias (a ultima linha antes do
    # bloco de amostra, apos o cabecalho + linha em branco = indice 8)
    idx_primeira_linha_espectro = 8
    valores = linhas[idx_primeira_linha_espectro].split()[:-1]
    linhas[idx_primeira_linha_espectro] = " ".join(valores)
    with pytest.raises(ValueError, match="absorbancias"):
        bt._parsear_tecator("\n".join(linhas))
