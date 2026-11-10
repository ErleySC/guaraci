"""JCAMP-DX: casos que a comparacao com arquivos reais publicos (a suite de teste oficial do JCAMP-DX, de
R. J. Lancashire, e os espectros de gas do NIST) mostrou errados. Os arquivos reais nao entram no repositorio; cada caso
abaixo reproduz so' a esquisitice, em dados sinteticos.

Antes: (1) arquivos AFFN com os rotulos de abscissa das linhas deslocados em um ponto (o formato do NIST) eram ancorados
pelos rotulos e perdiam um valor por linha -- espectro distorcido em ate' 99% da faixa, sem erro; (2) um comentario `$$`
depois dos dados (o INTTODX grava "$$No of bytes of data = ...") fazia o arquivo AFFN ser lido como ASDF e falhar;
(3) `##NPOINTS = 8192` (espaco antes do `=`) nao era reconhecido e o espectro vinha vazio; (4) arquivo composto, tabela de
picos e NTUPLES eram lidos como se fossem um espectro, ou davam resultado vazio sem dizer por que."""
import numpy as np
import pytest

from guaraci.dados_io import parse_dx


def _escrever(tmp_path, texto, nome="a.dx"):
    p = tmp_path / nome
    p.write_text(texto, encoding="latin-1")
    return str(p)


def _cabecalho(n, x0, xn, yfac=1.0, extra=""):
    return (f"##TITLE=teste\n##JCAMP-DX=4.24\n##DATA TYPE=INFRARED SPECTRUM\n##XUNITS=1/CM\n##YUNITS=ABSORBANCE\n"
            f"##XFACTOR=1\n##YFACTOR={yfac}\n##FIRSTX={x0}\n##LASTX={xn}\n##NPOINTS={n}\n{extra}")


def _y(n):
    return np.round(np.sin(np.linspace(0, 6, n)) * 1000 + 2000).astype(int)


def test_rotulos_deslocados_do_nist_nao_perdem_pontos(tmp_path):
    """6 valores por linha; do 2o rotulo em diante a abscissa vem um ponto antes do certo (como nos arquivos do NIST)."""
    n, por_linha = 60, 6
    y = _y(n)
    x0, dx = 600.0, 2.0
    linhas = []
    for k in range(n // por_linha):
        idx = k * por_linha - (1 if k else 0)           # rotulo deslocado a partir da 2a linha
        vals = " ".join(str(v) for v in y[k * por_linha:(k + 1) * por_linha])
        linhas.append(f"{x0 + idx * dx:.2f} {vals}")
    f = _escrever(tmp_path, _cabecalho(n, x0, x0 + (n - 1) * dx, 1.0, "##XYDATA=(X++(Y..Y))\n") + "\n".join(linhas) + "\n##END=\n")
    x, yy = parse_dx(f)
    assert yy.size == n and np.array_equal(np.round(yy).astype(int), y)
    assert x[0] == x0 and np.isclose(x[-1], x0 + (n - 1) * dx)


def test_afnn_com_rotulos_certos_continua_lido_igual(tmp_path):
    n, por_linha = 60, 5
    y = _y(n)
    linhas = [f"{600 + k * por_linha * 2.0:.1f} " + " ".join(str(v) for v in y[k * por_linha:(k + 1) * por_linha])
              for k in range(n // por_linha)]
    f = _escrever(tmp_path, _cabecalho(n, 600, 600 + (n - 1) * 2.0, 1.0, "##XYDATA=(X++(Y..Y))\n") + "\n".join(linhas) + "\n##END=\n")
    assert np.array_equal(np.round(parse_dx(f)[1]).astype(int), y)


def test_comentario_depois_dos_dados_nao_vira_asdf(tmp_path):
    n = 12
    y = _y(n)
    corpo = "\n".join(f"{600 + k * 6 * 2.0:.1f} " + " ".join(str(v) for v in y[k * 6:(k + 1) * 6]) for k in range(2))
    f = _escrever(tmp_path, _cabecalho(n, 600, 622, 1.0, "##XYDATA=(X++(Y..Y))\n") + corpo
                  + "\n$$$No of bytes of data =  23200\n##END=\n")
    assert np.array_equal(np.round(parse_dx(f)[1]).astype(int), y)


def test_rotulos_com_espaco_antes_do_igual(tmp_path):
    n = 12
    y = _y(n)
    cab = (f"##TITLE = x\n##DATA TYPE = NMR SPECTRUM\n##NPOINTS = {n}\n##FIRSTX = 100.0\n##LASTX = 89.0\n"
           f"##XFACTOR = 1.0\n##YFACTOR = 2.0   $$ fator\n##XYDATA = (X++(Y..Y))\n")
    corpo = "\n".join(f"{100 - k * 6.0:.1f} " + " ".join(str(v) for v in y[k * 6:(k + 1) * 6]) for k in range(2))
    f = _escrever(tmp_path, cab + corpo + "\n##END = \n")
    x, yy = parse_dx(f)
    assert yy.size == n and np.allclose(yy, y * 2.0) and x[0] == 100.0 and x[-1] == 89.0


def test_arquivo_composto_e_recusado_com_motivo(tmp_path):
    bloco = _cabecalho(6, 600, 610, 1.0, "##XYDATA=(X++(Y..Y))\n600 1 2 3 4 5 6\n##END=\n")
    f = _escrever(tmp_path, "##TITLE=ligacao\n##DATA TYPE=LINK\n##BLOCKS=2\n" + bloco + bloco + "##END=\n")
    with pytest.raises(ValueError, match="composto"):
        parse_dx(f)


@pytest.mark.parametrize("rotulo, trecho", [("##PEAK TABLE=(XY..XY)\n100,5 200,9\n", "tabela de picos"),
                                            ("##NTUPLES=NMR SPECTRUM\n##VAR_NAME=A,B\n", "NTUPLES")])
def test_tabela_de_picos_e_ntuples_dizem_o_que_sao(tmp_path, rotulo, trecho):
    f = _escrever(tmp_path, _cabecalho(2, 100, 200, 1.0, rotulo) + "##END=\n")
    with pytest.raises(ValueError, match=trecho):
        parse_dx(f)
