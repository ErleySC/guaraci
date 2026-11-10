"""README de entrada: sem linha interna em português, sem frases absolutas, idioma da interface descrito como é (V38, V102, V23, V27)."""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import padronizar_cabecalhos as pc  # noqa: E402

EN = (RAIZ / "README.md").read_text(encoding="utf-8")
PT = (RAIZ / "README.pt-BR.md").read_text(encoding="utf-8")


def test_readme_em_ingles_nao_abre_com_linha_interna_em_portugues():
    primeiras = "\n".join(EN.splitlines()[:6])
    assert "Público" not in primeiras and "Estado:" not in primeiras
    assert "README.md" not in pc.alvos(["README.md", "README.pt-BR.md", "MANUAL.md"])      # o padronizador não o reinsere
    assert "README.pt-BR.md" in pc.alvos(["README.md", "README.pt-BR.md"])


def test_sem_frases_absolutas_que_nao_se_verificam():
    for trecho in ("Nothing was softened", "every metric ships"):
        assert trecho not in EN, trecho
    for trecho in ("Nada foi suavizado", "toda métrica vem"):
        assert trecho not in PT, trecho
    assert "Weak and null results are reported too" in EN and "Resultados fracos ou nulos também são relatados" in PT


def test_descreve_o_idioma_da_interface_como_ele_e():
    """O terminal abre em ingles e troca com `I` (CHANGELOG, "CLI antes do lancamento"); nomes de pasta e coluna ficam em portugues."""
    assert "opens in English and switches to Portuguese with the `I` key" in EN
    assert "abre em inglês e passa para o português com a tecla `I`" in PT
    assert "mostly in Portuguese" not in EN and "maioria em português" not in PT
    assert "i18n-PT%20%2F%20EN" not in EN + PT


def test_cita_a_variavel_do_terminal_de_fundo_claro_e_o_guia_de_instalacao():
    for texto in (EN, PT):
        assert "GUARACI_TERMINAL=claro" in texto
        assert "docs/usuario/instalacao.md" in texto
