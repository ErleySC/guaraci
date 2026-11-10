# -*- coding: utf-8 -*-
"""P6: o menu [2.4] oferecia formatos que dependem de pacotes opcionais (OPUS, SPC, RMN Bruker, HPLC Agilent/Waters,
PARAFAC, textura de imagem) sem avisar que a instalacao base nao os tem. Agora: a tela marca o formato com
"(extra: pip install ...)" quando o pacote falta, avisa na hora da escolha, e `guaraci doctor`/`guaraci checar`
listam o que falta e o que isso afeta."""
from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

from guaraci import assistente_checar as chk
from guaraci import assistente_regras as regras
from guaraci import catalogo_tecnicas as ct
from guaraci import extras_opcionais as ext
from guaraci.config import Config

RAIZ = Path(__file__).resolve().parents[1]


def _sem(monkeypatch, *modulos):
    """Faz `find_spec` nao achar os modulos dados (o que a instalacao base sem o extra veria)."""
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda nome, *a, **k: None if nome in modulos else real(nome, *a, **k))


def test_deteccao_por_find_spec_instalado_e_ausente(monkeypatch):
    e = next(x for x in ext.EXTRAS if x.extra == "opus")
    _sem(monkeypatch, "brukeropus")
    assert not ext.instalado(e) and e in ext.ausentes()
    monkeypatch.undo()
    # nome do MODULO, nao do pip: rainbow-api instala o modulo `rainbow`
    assert next(x for x in ext.EXTRAS if x.pacote == "rainbow-api").modulo == "rainbow"


def test_sufixo_so_aparece_quando_o_pacote_falta(monkeypatch):
    _sem(monkeypatch, "spcfile")
    assert ext.sufixo_menu("spc") == '(extra: pip install ".[spc]")'
    assert "[spc]" not in ext.sufixo_menu("opus") and ext.sufixo_menu("opus") in ("", ext.sufixo_menu("opus"))
    assert ext.sufixo_menu("csv") == "" and ext.sufixo_menu("dx") == ""      # formatos sem extra
    monkeypatch.undo()
    if ext.instalado(next(x for x in ext.EXTRAS if x.extra == "spc")):
        assert ext.sufixo_menu("spc") == ""                                  # instalado: normal


def test_python_sem_wheel_do_rainbow_diz_isso(monkeypatch):
    e = next(x for x in ext.EXTRAS if x.extra == "hplc")
    monkeypatch.setattr(ext, "_indisponivel_nesta_versao", lambda x: True)
    assert "nao tem versao" in ext.texto_instalar(e, "PT") and "no release" in ext.texto_instalar(e, "EN")


def _tela(monkeypatch, tk, formato):
    from guaraci import guaraci as G
    cfg = Config()
    respostas = iter([str(G._tecnica_ordem().index(tk) + 1), formato])
    monkeypatch.setattr(G, "_input", lambda *a, **k: next(respostas))
    monkeypatch.setattr(G, "_pause", lambda *a, **k: None)
    monkeypatch.setattr(G, "_cabecalho_tela", lambda *a, **k: None)
    monkeypatch.setattr(G, "_rodape_tela", lambda *a, **k: None)
    with G.console.capture() as cap:
        G._menu_technique(cfg)
    return cfg, " ".join(cap.get().split())             # o painel quebra linhas: compara o texto corrido


def test_tela_2_4_marca_os_formatos_sem_pacote_e_avisa_na_escolha(monkeypatch):
    _sem(monkeypatch, "brukeropus", "spcfile")
    cfg, saida = _tela(monkeypatch, "ft-nir", "2")                 # escolhe OPUS
    assert saida.count("(extra: pip install") >= 2                 # OPUS e SPC marcados na lista
    assert "[opus]" in saida and "[spc]" in saida
    assert ("exige o pacote opcional" in saida or "needs the optional package" in saida) and "brukeropus" in saida   # aviso na escolha
    assert cfg.mode == "opus"                                      # a escolha vale (o aviso nao a impede)


def test_tela_2_4_nao_marca_nada_quando_os_pacotes_estao_instalados(monkeypatch):
    for e in ext.EXTRAS:
        if not ext.instalado(e):
            pytest.skip("ambiente sem todos os extras")
    _, saida = _tela(monkeypatch, "ft-nir", "2")
    assert "extra: pip install" not in saida and "optional package" not in saida and "pacote opcional" not in saida


def test_tela_2_4_formato_sem_extra_nao_avisa_mesmo_sem_pacotes(monkeypatch):
    _sem(monkeypatch, *(e.modulo for e in ext.EXTRAS))
    _, saida = _tela(monkeypatch, "ft-nir", "4")                   # CSV
    assert "optional package" not in saida and "pacote opcional" not in saida


def test_tela_2_4_avisa_do_recurso_opcional_da_tecnica(monkeypatch):
    _sem(monkeypatch, "tensorly")
    _, saida = _tela(monkeypatch, "eem", "")
    assert "PARAFAC" in saida and "[multiway]" in saida
    _sem(monkeypatch, "skimage")
    _, saida = _tela(monkeypatch, "imagem", "")
    assert ("textura" in saida or "texture" in saida) and "[imagem]" in saida


def test_checar_aponta_formato_ausente_como_critico_e_recurso_como_aviso(monkeypatch):
    _sem(monkeypatch, "brukeropus", "skimage")
    achados = regras.avaliar(Config(mode="opus", include_image_texture=True), lang="PT")
    por_id = {a.id: a for a in achados}
    assert por_id["extra_formato_ausente"].severidade == "critico"
    assert "brukeropus" in por_id["extra_formato_ausente"].mensagem and "[opus]" in por_id["extra_formato_ausente"].fazer
    assert por_id["extra_recurso_ausente"].severidade == "aviso" and "scikit-image" in por_id["extra_recurso_ausente"].mensagem
    ok = {a.id for a in regras.avaliar(Config(mode="csv"), lang="PT")}
    assert "extra_formato_ausente" not in ok and "extra_recurso_ausente" not in ok


def test_checar_imprime_a_lista_de_extras_ausentes_e_o_que_afetam(monkeypatch, tmp_path, capsys):
    _sem(monkeypatch, "nmrglue")
    y = tmp_path / "c.yaml"
    y.write_text("n_permutacoes: 200\n", encoding="utf-8")
    assert chk.main([str(y)]) == 0                                  # config limpa: so' informa, nao reprova
    out = capsys.readouterr().out
    assert "nmrglue" in out and "nmr" in out and '[rmn]' in out and '".[all]"' in out
    assert "CRITICO" not in out
    monkeypatch.undo()
    if not ext.ausentes():
        assert chk.main([str(y)]) == 0 and "Pacotes opcionais" not in capsys.readouterr().out


def test_doctor_lista_extras_de_formato_com_o_que_afetam(monkeypatch, tmp_path, capsys):
    from guaraci import guaraci as G
    _sem(monkeypatch, "rainbow", "tensorly")
    monkeypatch.chdir(tmp_path)
    G._comando_doctor()
    txt = (tmp_path / "guaraci_doctor.txt").read_text(encoding="utf-8")
    assert "rainbow-api" in txt and "tecnicas afetadas: hplc" in txt.lower() or "affected techniques: hplc" in txt.lower()
    assert "tensorly" in txt and '".[all]"' in txt
    linha = next(l for l in txt.splitlines() if "brukeropus" in l)
    assert ("ausente" in linha or "missing" in linha) == (not ext.instalado(next(e for e in ext.EXTRAS if e.extra == "opus")))


def test_pyproject_all_inclui_todo_extra_do_registro():
    dados = tomllib.loads((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))
    extras = dados["project"]["optional-dependencies"]
    tudo = extras["all"][0]
    for e in ext.EXTRAS:
        assert e.extra in extras, e.extra
        assert any(e.pacote.split("[")[0] in d for d in extras[e.extra]), (e.extra, e.pacote)
        assert e.extra in tudo, f"o extra `all` nao inclui [{e.extra}]"
    # as opcoes de formato do menu que dependem de extra estao todas no registro
    modos = {m for e in ext.EXTRAS for m in e.modos}
    for k, ins in ct.INSTRUMENTOS.items():
        for m, _rot in ins["formatos"]:
            if m in ("opus", "spc", "rmn", "hplc"):
                assert m in modos, (k, m)
