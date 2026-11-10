"""Verificacao incremental: etapa so' e' reaproveitada se as entradas dela nao mudaram desde uma execucao em que ela passou."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verificacao_incremental as vi  # noqa: E402
import verificar_hash_final as vhf  # noqa: E402

RAIZ = Path(".")


def _git_fixo(saida: str):
    return lambda *a: saida


def _res(hash_, etapas, quando="2026-10-08T10:00:00+00:00"):
    return {"hash": hash_, "concluido_em": quando, "etapas": etapas}


def _etapa(nome, estado="ok", entradas="h1", **extra):
    return {"nome": nome, "estado": estado, "detalhe": "d", "entradas": entradas, **extra}


def test_hash_de_entradas_muda_quando_um_arquivo_de_entrada_muda():
    a = vi.hash_de_entradas(RAIZ, "ruff", _git_fixo("100644 aaa 0\tx.py\n"))
    b = vi.hash_de_entradas(RAIZ, "ruff", _git_fixo("100644 bbb 0\tx.py\n"))
    assert a and b and a != b
    assert a == vi.hash_de_entradas(RAIZ, "ruff", _git_fixo("100644 aaa 0\tx.py\n"))


def test_sem_arquivo_de_entrada_ou_etapa_fora_da_tabela_nao_ha_hash():
    assert vi.hash_de_entradas(RAIZ, "ruff", _git_fixo("")) is None            # git falhou: nunca vira "igual"
    assert vi.hash_de_entradas(RAIZ, "vault", _git_fixo("x")) is None
    assert vi.hash_de_entradas(RAIZ, "arvore_limpa", _git_fixo("x")) is None


def test_etapa_com_as_mesmas_entradas_e_ok_e_reaproveitada():
    ant = [_res("a" * 40, [_etapa("ruff", entradas="h1")])]
    origem, e = vi.reaproveitar("ruff", "h1", ant)
    assert origem == "a" * 12 and e["nome"] == "ruff"


def test_mudar_uma_entrada_invalida_a_etapa():
    ant = [_res("a" * 40, [_etapa("ruff", entradas="h1")])]
    assert vi.reaproveitar("ruff", "h2", ant) is None


def test_etapa_que_nao_passou_ou_sem_hash_nao_e_reaproveitada():
    assert vi.reaproveitar("ruff", "h1", [_res("a" * 40, [_etapa("ruff", estado="falhou")])]) is None
    assert vi.reaproveitar("ruff", "h1", [_res("a" * 40, [{"nome": "ruff", "estado": "ok", "detalhe": "d"}])]) is None
    assert vi.reaproveitar("ruff", None, [_res("a" * 40, [_etapa("ruff")])]) is None


def test_etapas_baratas_ou_do_repositorio_inteiro_nunca_sao_reaproveitadas():
    ant = [_res("a" * 40, [_etapa(n) for n in ("arvore_limpa", "exposicao", "links", "docs_gerados_em_dia", "vault", "cobertura_minima")])]
    for n in ("arvore_limpa", "exposicao", "links", "docs_gerados_em_dia", "vault", "cobertura_minima"):
        assert vi.reaproveitar(n, "h1", ant) is None


def test_vale_a_execucao_aprovada_mais_recente_e_a_origem_original_e_preservada():
    ant = [_res("a" * 40, [_etapa("ruff", entradas="h1")], "2026-10-01T00:00:00+00:00"),
           _res("b" * 40, [_etapa("ruff", entradas="h1", reaproveitada_de="a" * 12)], "2026-10-02T00:00:00+00:00")]
    assert vi.reaproveitar("ruff", "h1", ant)[0] == "a" * 12


def test_carregar_anteriores_ignora_arquivo_ilegivel_e_ordena_por_data(tmp_path):
    (tmp_path / "resultado_2.json").write_text(json.dumps(_res("2" * 40, [], "2026-10-02T00:00:00+00:00")), encoding="utf-8")
    (tmp_path / "resultado_1.json").write_text(json.dumps(_res("1" * 40, [], "2026-10-01T00:00:00+00:00")), encoding="utf-8")
    (tmp_path / "resultado_x.json").write_text("{nao e json", encoding="utf-8")
    (tmp_path / "resultado_y.json").write_text(json.dumps({"hash": "y"}), encoding="utf-8")
    assert [r["hash"][0] for r in vi.carregar_anteriores(tmp_path)] == ["1", "2"]


def test_decidir_reaproveitamento_respeita_a_flag_sem_reaproveitar():
    git = _git_fixo("100644 aaa 0\tx.py\n")
    h = vi.hash_de_entradas(RAIZ, "ruff", git)
    ant = [_res("a" * 40, [_etapa("ruff", entradas=h)])]
    assert vhf.decidir_reaproveitamento("ruff", ant, True, RAIZ, git)[1] is not None
    assert vhf.decidir_reaproveitamento("ruff", ant, False, RAIZ, git) == (h, None)


def test_toda_etapa_reaproveitavel_declara_entradas_e_a_suite_ignora_so_o_gerado():
    assert vi.REUTILIZAVEIS == set(vi.ENTRADAS)
    assert all(vi.ENTRADAS[n] for n in vi.ENTRADAS)
    assert ":(exclude)docs/qualidade" in vi.ENTRADAS["suite_completa"]
    assert vi.ENTRADAS["pytest_3.13"] == vi.ENTRADAS["suite_completa"]


def _main_com_etapas_falsas(monkeypatch, tmp_path, anteriores):
    """Roda `main` com duas etapas falsas e o git trocado por uma saida controlada; devolve (rc, resultado gravado)."""
    pasta = tmp_path / "docs" / "qualidade"
    pasta.mkdir(parents=True)
    for i, r in enumerate(anteriores):
        (pasta / f"resultado_{i}.json").write_text(json.dumps(r), encoding="utf-8")
    execucoes = []
    monkeypatch.setattr(vhf, "RAIZ", tmp_path)
    monkeypatch.setattr(vhf, "head", lambda: "c" * 40)
    monkeypatch.setattr(vhf, "_git", lambda *a: "t")
    monkeypatch.setattr(vhf, "e_arvore_limpa", lambda: ("ok", "limpa"))
    monkeypatch.setattr(vhf, "e_exposicao", lambda h=None: ("ok", "0"))
    monkeypatch.setattr(vhf, "ETAPAS_LOCAIS", [("arvore_limpa", vhf.e_arvore_limpa),
                                              ("ruff", lambda: (execucoes.append("ruff") or ("ok", "ruff limpo"))),
                                              ("links", lambda: (execucoes.append("links") or ("ok", "0 links")))])
    monkeypatch.setattr(vi, "_git_padrao", lambda raiz: _git_fixo("100644 aaa 0\tx.py\n"))
    monkeypatch.setattr(vhf, "rodar_wsl", lambda h, etapas=None: ({}, "sem WSL no teste"))
    monkeypatch.setattr(vhf, "nomes_wsl", lambda: [])
    rc = vhf.main(["--so", "arvore_limpa", "exposicao", "ruff", "links"])
    return rc, execucoes


def test_main_reaproveita_ruff_quando_as_entradas_nao_mudaram_e_sempre_roda_links(monkeypatch, tmp_path):
    h = vi.hash_de_entradas(RAIZ, "ruff", _git_fixo("100644 aaa 0\tx.py\n"))
    ant = [_res("a" * 40, [_etapa("ruff", entradas=h)])]
    rc, execucoes = _main_com_etapas_falsas(monkeypatch, tmp_path, ant)
    assert rc == 0 and execucoes == ["links"]


def test_main_reexecuta_ruff_quando_uma_entrada_mudou(monkeypatch, tmp_path):
    ant = [_res("a" * 40, [_etapa("ruff", entradas="hash-de-outra-versao")])]
    rc, execucoes = _main_com_etapas_falsas(monkeypatch, tmp_path, ant)
    assert rc == 0 and execucoes == ["ruff", "links"]


def test_reaproveitada_e_marcada_no_resultado(monkeypatch, tmp_path):
    h = vi.hash_de_entradas(RAIZ, "ruff", _git_fixo("100644 aaa 0\tx.py\n"))
    reg = {}
    monkeypatch.setattr(vhf, "montar_resultado", lambda *a, **k: reg.update(etapas=a[4]) or {"head_estavel": True, "falhas": 0, "hash": "c" * 40,
                                                                                            "pendentes_do_autor": [], "reaproveitadas": [],
                                                                                            "executadas": [], "versoes_python": []})
    _main_com_etapas_falsas(monkeypatch, tmp_path, [_res("a" * 40, [_etapa("ruff", entradas=h)])])
    ruff = next(e for e in reg["etapas"] if e["nome"] == "ruff")
    assert ruff["origem"] == "reaproveitada" and "reaproveitado de " + "a" * 12 in ruff["detalhe"]
    links = next(e for e in reg["etapas"] if e["nome"] == "links")
    assert links["origem"] == "executada"


def test_resultado_lista_o_que_foi_executado_e_o_que_foi_reaproveitado():
    et = [{"nome": "a", "estado": "ok", "origem": "reaproveitada"}, {"nome": "b", "estado": "ok", "origem": "executada"}]
    r = vhf.montar_resultado("a" * 40, "a" * 40, "t", "r", et, "3.12", "i", "f")
    assert r["reaproveitadas"] == ["a"] and r["executadas"] == ["b"]
