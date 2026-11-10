"""Conteudo permitido do wheel/sdist e varredura da ajuda: as regras puras (rapidas) e a construcao real (slow)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import auditar_pacote_construido as apc  # noqa: E402
import percorrer_ajuda_instalada as pai  # noqa: E402

BOM_WHEEL = ["guaraci/__init__.py", "guaraci/dados_docs/manual.md", "guaraci/perfis_matriz/generico.yaml",
             "guaraci/app_web.py", "guaraci/assets/guaraci_icon.png",
             "guaraci_chemometrics-1.0.0.dist-info/licenses/LICENSE",
             "guaraci_chemometrics-1.0.0.dist-info/licenses/LICENSES/BSD-3-Clause-specio.txt",
             "guaraci_chemometrics-1.0.0.dist-info/licenses/THIRD_PARTY_NOTICES.md"]


def test_wheel_bom_nao_tem_problema():
    assert apc.problemas_do_wheel(BOM_WHEEL) == []


@pytest.mark.parametrize("intruso", ["guaraci/dados/amostra.csv", "guaraci/x.xlsx", "tests/test_a.py", "docs/index.md",
                                     "validacao/a.py", "datasets_" "publicos/b.txt", "scripts/s.py", "guaraci/m.joblib"])
def test_wheel_com_dado_ou_arquivo_do_repositorio_e_apontado(intruso):
    assert apc.problemas_do_wheel(BOM_WHEEL + [intruso])


@pytest.mark.parametrize("faltando", ["guaraci/dados_docs/", "guaraci/perfis_matriz/"])
def test_wheel_sem_dados_do_pacote_e_apontado(faltando):
    assert apc.problemas_do_wheel([n for n in BOM_WHEEL if not n.startswith(faltando)])


@pytest.mark.parametrize("faltando", ["guaraci/app_web.py", "guaraci/assets/guaraci_icon.png"])
def test_wheel_sem_o_app_web_e_apontado(faltando):
    """V07: `guaraci web` a partir do wheel exige o app dentro do pacote."""
    assert apc.problemas_do_wheel([n for n in BOM_WHEEL if n != faltando])


@pytest.mark.parametrize("faltando", ["licenses/LICENSE", "licenses/LICENSES/BSD-3-Clause-specio.txt",
                                      "licenses/THIRD_PARTY_NOTICES.md"])
def test_wheel_sem_licenca_de_terceiros_e_apontado(faltando):
    """V104: o BSD-3 do porte do leitor .sp exige o texto da licenca junto com as copias."""
    assert apc.problemas_do_wheel([n for n in BOM_WHEEL if not n.endswith(faltando)])


@pytest.mark.parametrize("faltando", ["LICENSE", "LICENSES/BSD-3-Clause-specio.txt", "THIRD_PARTY_NOTICES.md"])
def test_sdist_sem_licenca_de_terceiros_e_apontado(faltando):
    ok = ["guaraci-1.0.0/src/guaraci/a.py", "guaraci-1.0.0/LICENSE",
          "guaraci-1.0.0/LICENSES/BSD-3-Clause-specio.txt", "guaraci-1.0.0/THIRD_PARTY_NOTICES.md"]
    assert apc.problemas_do_sdist(ok) == []
    assert apc.problemas_do_sdist([n for n in ok if not n.endswith(faltando)]) == [f"sdist sem {faltando}"]


def test_sdist_com_validacao_ou_relatorios_e_apontado():
    ok = ["guaraci-1.0.0/src/guaraci/a.py", "guaraci-1.0.0/MANUAL.md", "guaraci-1.0.0/LICENSE",
          "guaraci-1.0.0/LICENSES/BSD-3-Clause-specio.txt", "guaraci-1.0.0/THIRD_PARTY_NOTICES.md"]
    assert apc.problemas_do_sdist(ok) == []
    for ruim in ("validacao/x.py", "datasets_" "publicos/y.csv", "docs/qualidade/z.md", "docs/auditoria/w.md"):
        assert apc.problemas_do_sdist(ok + [f"guaraci-1.0.0/{ruim}"]) == [f"sdist traz {ruim}"]


def test_a_varredura_da_ajuda_passa_sobre_o_checkout():
    problemas, n = pai.percorrer()
    assert problemas == []
    assert n["ajuda"] > 100 and n["manual"] >= 5 and n["help"] >= 5 and n["assistente"] >= 10


def test_a_varredura_reprova_ponteiro_para_fora_do_pacote(monkeypatch):
    from guaraci import assistente
    monkeypatch.setattr(assistente, "responder", lambda q, lang="PT": "veja o MANUAL.md")
    problemas, _ = pai.percorrer()
    assert any("MANUAL.md" in p for p in problemas)


def test_a_varredura_recusa_rodar_sobre_o_checkout_sem_permissao(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x"])
    assert pai.main() == 2
    assert "nao esta instalado" in capsys.readouterr().out


@pytest.mark.slow
def test_pacote_construido_de_verdade_passa_a_auditoria_de_conteudo(capsys):
    assert apc.main([]) == 0
    assert "problemas de conteudo: 0" in capsys.readouterr().out
