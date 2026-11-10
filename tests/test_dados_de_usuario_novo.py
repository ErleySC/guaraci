"""Quem chega com os proprios dados .dx (sem a convencao de ##TITLE do autor): o log nao trata isso como defeito e diz
como informar o grupo; o manual explica de onde vem classe e grupo e so' cita chaves que existem."""
import re
from pathlib import Path

import numpy as np

from guaraci import dados_io
from guaraci.config_io import _CONFIG_SPEC


def _dx(path, titulo, n=40):
    wn = np.linspace(4000, 10000, n)
    y = np.sin(wn / 900.0) + 2
    corpo = "\n".join(f"{wn[k]:.4f} " + " ".join(f"{v:.6f}" for v in y[k:k + 5]) for k in range(0, n, 5))
    Path(path).write_text(
        f"##TITLE={titulo}\n##JCAMP-DX=4.24\n##DATA TYPE=NEAR INFRARED SPECTRUM\n##XUNITS=1/CM\n##YUNITS=ABSORBANCE\n"
        f"##XFACTOR=1\n##YFACTOR=1\n##FIRSTX={wn[0]}\n##LASTX={wn[-1]}\n##NPOINTS={n}\n##XYDATA=(X++(Y..Y))\n{corpo}\n##END=\n",
        encoding="latin-1")


def _carregar(pasta, capsys):
    dados_io.load_dx(str(pasta))
    return capsys.readouterr().out


def test_titulos_livres_nao_geram_warning_e_dizem_como_agrupar(tmp_path, capsys):
    for cls in ("controle", "tratado"):
        (tmp_path / cls).mkdir()
        for i in range(3):
            _dx(tmp_path / cls / f"a{i}.dx", f"Medida {cls} {i}")
    saida = _carregar(tmp_path, capsys)
    assert "non-conforming" not in saida and "[WARNING] 6 files" not in saida
    assert "[INFO] No ##TITLE= follows the optional GUARACI title pattern" in saida
    assert "arquivo_grupos" in saida


def test_titulos_misturados_avisam_quantos_destoam(tmp_path, capsys):
    (tmp_path / "A").mkdir()
    _dx(tmp_path / "A" / "um.dx", "ABC-01-02-2099-T1")
    _dx(tmp_path / "A" / "dois.dx", "ABC-01-02-2099-T2")
    _dx(tmp_path / "A" / "tres.dx", "qualquer coisa")
    saida = _carregar(tmp_path, capsys)
    assert "[WARNING] 1 of 3 files do not follow the ##TITLE= pattern" in saida


def test_secao_41_do_manual_so_cita_chaves_que_existem():
    manual = Path(__file__).resolve().parents[1] / "MANUAL.md"
    texto = manual.read_text(encoding="utf-8")
    ini = texto.index("### 4.1 Como organizar os seus dados")
    fim = texto.index("## 4b Perfis de")
    secao = texto[ini:fim]
    chaves = {c["key"] for c in _CONFIG_SPEC}
    citadas = set(re.findall(r"`((?:arquivo_grupos|coluna_[a-z_]+|dx_[a-z_]+))`", secao))
    assert {"arquivo_grupos", "coluna_classe", "coluna_grupo", "dx_parte_classe", "dx_teor_no_nome"} <= citadas
    faltam = citadas - chaves
    assert not faltam, f"chaves citadas no manual que nao existem: {sorted(faltam)}"


def test_pacote_leva_o_mesmo_manual():
    raiz = Path(__file__).resolve().parents[1]
    assert (raiz / "MANUAL.md").read_text(encoding="utf-8") == (raiz / "src/guaraci/dados_docs/manual.md").read_text(encoding="utf-8")
