"""Ramos de borda de guaraci.leitores_avancados (pastas de classe, arquivos e pastas-amostra, EEM, GC-MS): um teste por ramo."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from guaraci import leitores_avancados as la


def _par(caminho: str):
    """Parser de teste: arquivo texto 'x y' por linha."""
    a = np.loadtxt(caminho, ndmin=2)
    return a[:, 0], a[:, 1]


def _escrever(caminho, linhas="1 1\n2 2\n3 3\n"):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(linhas)
    return caminho


# ── utilidades ──────────────────────────────────────────────────────────────
def test_grade_comum_sem_sobreposicao_e_erro_claro():
    with pytest.raises(ValueError, match="sem sobreposicao"):
        la._grade_comum([(np.array([1.0, 2.0]), np.zeros(2)), (np.array([5.0, 6.0]), np.zeros(2))])


def test_rotulos_com_concentracao_so_em_algumas_amostras_completam_com_zero():
    _, conc, meta = la._rotulos_e_metadados([("A", "a1", 2.5), ("A", "a2", None)])
    assert conc.tolist() == [2.5, 0.0] and len(meta) == 2


def test_subpastas_de_raiz_inexistente_e_pasta_ilegivel(tmp_path, monkeypatch):
    assert la._subpastas_classe(str(tmp_path / "nao_existe"), lambda c: True) == []
    (tmp_path / "ilegivel").mkdir()
    original = os.listdir

    def lista(p):
        if str(p).endswith("ilegivel"):
            raise PermissionError("negado")
        return original(p)
    monkeypatch.setattr(os, "listdir", lista)
    assert la._subpastas_classe(str(tmp_path), lambda c: True) == []


# ── um arquivo = uma amostra ────────────────────────────────────────────────
def test_arquivos_pasta_inexistente_e_pasta_sem_arquivo_valido(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pasta nao existe"):
        la.load_generic_file_dataset(str(tmp_path / "nao_existe"), _par, "teste")
    (tmp_path / "vazia").mkdir()
    with pytest.raises(FileNotFoundError, match="Nenhum arquivo candidato"):
        la.load_generic_file_dataset(str(tmp_path / "vazia"), _par, "teste")
    _escrever(str(tmp_path / "todos_ruins" / "x.txt"), "nao numerico\n")
    with pytest.raises(ValueError, match="Nenhuma amostra teste valida"):
        la.load_generic_file_dataset(str(tmp_path / "todos_ruins"), _par, "teste")


def test_arquivos_soltos_na_raiz_ignoram_pasta_e_classe_com_subpasta_ignora_o_que_nao_e_arquivo(tmp_path, capsys):
    raiz = tmp_path / "soltos"
    _escrever(str(raiz / "a.txt"))
    _escrever(str(raiz / "b.txt"), "1 2\n2 3\n3 4\n")
    (raiz / "pasta_perdida").mkdir()
    grade, X, rotulos, *_ = la.load_generic_file_dataset(str(raiz), _par, "teste", extensoes=(".txt",))
    assert X.shape[0] == 2 and set(rotulos) == {"soltos"}
    cls = tmp_path / "porclasse"
    _escrever(str(cls / "A" / "a1.txt"))
    _escrever(str(cls / "B" / "b1.txt"))
    (cls / "A" / "subdir").mkdir()
    _, X2, rot2, *_ = la.load_generic_file_dataset(str(cls), _par, "teste", extensoes=(".txt",))
    assert sorted(rot2) == ["A", "B"] and X2.shape[0] == 2


def test_arquivo_vazio_e_arquivo_que_falha_sao_pulados_com_aviso(tmp_path, capsys):
    raiz = tmp_path / "mistos"
    _escrever(str(raiz / "ok1.txt"))
    _escrever(str(raiz / "ok2.txt"))
    _escrever(str(raiz / "ruim.txt"), "isto nao e' numero\n")

    def parser(caminho):
        if caminho.endswith("vazio.txt"):
            return np.array([]), np.array([])
        return _par(caminho)
    _escrever(str(raiz / "vazio.txt"))
    _, X, *_ = la.load_generic_file_dataset(str(raiz), parser, "teste", extensoes=(".txt",))
    saida = capsys.readouterr().out
    assert X.shape[0] == 2 and "sem dado -- pulado" in saida and "1 arquivos teste com erro de parsing" in saida


# ── uma pasta = uma amostra ─────────────────────────────────────────────────
def _eh_amostra(caminho: str) -> bool:
    return os.path.isfile(os.path.join(caminho, "dado.txt"))


def _identidade(caminho: str) -> str:
    return os.path.join(caminho, "dado.txt")


def test_pastas_amostra_inexistente_classes_com_amostras_e_raiz_que_ja_e_amostra(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pasta nao existe"):
        la.load_generic_folder_dataset(str(tmp_path / "nao_existe"), _par, "teste", _eh_amostra, _identidade)
    raiz = tmp_path / "classes"
    _escrever(str(raiz / "A" / "a1" / "dado.txt"))
    _escrever(str(raiz / "B" / "b1" / "dado.txt"))
    _escrever(str(raiz / "B" / "arquivo_solto.txt"))
    (raiz / "B" / "sem_dado").mkdir()
    _, X, rotulos, *_ = la.load_generic_folder_dataset(str(raiz), _par, "teste", _eh_amostra, _identidade)
    assert sorted(rotulos) == ["A", "B"]
    unica = tmp_path / "amostra_unica"
    _escrever(str(unica / "dado.txt"))
    *_, = la.load_generic_folder_dataset(str(unica), _par, "teste", _eh_amostra, _identidade)


def test_pasta_amostra_vazia_e_com_erro_sao_puladas_com_aviso(tmp_path, capsys):
    raiz = tmp_path / "mistas"
    for nome in ("ok1", "ok2", "vazio", "ruim"):
        _escrever(str(raiz / nome / "dado.txt"), "x\n" if nome == "ruim" else "1 1\n2 2\n3 3\n")

    def parser(caminho):
        if "vazio" in caminho:
            return np.array([]), np.array([])
        return _par(caminho)
    _, X, *_ = la.load_generic_folder_dataset(str(raiz), parser, "teste", _eh_amostra, _identidade)
    saida = capsys.readouterr().out
    assert X.shape[0] == 2 and "sem dado -- pulado" in saida and "1 pastas teste com erro de parsing" in saida


def test_pdata_sem_experimento_numerado_e_erro(tmp_path):
    (tmp_path / "exp" / "pdata").mkdir(parents=True)
    with pytest.raises(ValueError, match="sem nenhum subdiretorio numerado"):
        la._resolver_rmn(str(tmp_path / "exp"))


# ── GC-MS e EEM ─────────────────────────────────────────────────────────────
def test_leitor_gcms_converte_o_tic_do_arquivo(tmp_path):
    from scipy.io import netcdf_file
    pasta = tmp_path / "gcms" / "classeA"
    pasta.mkdir(parents=True)
    for i in range(2):
        with netcdf_file(str(pasta / f"a{i}.cdf"), "w") as f:
            f.createDimension("n", 5)
            f.createVariable("scan_acquisition_time", "d", ("n",))[:] = np.arange(5) * 60.0
            f.createVariable("total_intensity", "d", ("n",))[:] = np.arange(5) * (i + 1.0)
    grade, X, rotulos, *_ = la._leitor_gcms(SimpleNamespace(input_folder=str(tmp_path / "gcms")))
    assert X.shape == (2, 5) and set(rotulos) == {"classeA"} and grade[-1] == pytest.approx(4.0)


def _eem(caminho, n_emissao, n_exc=2):
    cab = "nome\t" + "\t".join(str(250 + 10 * j) for j in range(n_exc)) + "\n"
    corpo = "".join(f"{300 + 10 * i}\t" + "\t".join(str(float(i + j)) for j in range(n_exc)) + "\n" for i in range(n_emissao))
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(cab + "x\nx\n" + corpo)


def test_leitor_eem_sem_arquivo_e_so_com_arquivo_invalido(tmp_path):
    with pytest.raises(FileNotFoundError, match="Nenhum arquivo .dat"):
        la._leitor_eem(SimpleNamespace(input_folder=str(tmp_path)))
    (tmp_path / "ruim.dat").write_text("curto\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Nenhuma amostra EEM valida"):
        la._leitor_eem(SimpleNamespace(input_folder=str(tmp_path)))


def test_leitor_eem_pula_arquivo_invalido_e_recusa_grades_diferentes(tmp_path, capsys):
    # Antes: grades diferentes eram truncadas ao menor vetor e a amostra truncada virava lixo (correlacao 0,02 com a
    # original; checkup de fluorescencia). Agora o arquivo invalido e' pulado com aviso e a grade diferente e' erro.
    _eem(str(tmp_path / "c" / "a.dat"), 4)
    (tmp_path / "c" / "ruim.dat").write_text("curto\n", encoding="utf-8")
    _eem(str(tmp_path / "c" / "z.dat"), 4)
    grade, X, rotulos, *_ = la._leitor_eem(SimpleNamespace(input_folder=str(tmp_path)))
    assert X.shape[0] == 2 and "ruim.dat" in capsys.readouterr().out and len(grade) == X.shape[1]
    _eem(str(tmp_path / "c" / "b.dat"), 3)
    with pytest.raises(ValueError, match="grade"):
        la._leitor_eem(SimpleNamespace(input_folder=str(tmp_path)))
