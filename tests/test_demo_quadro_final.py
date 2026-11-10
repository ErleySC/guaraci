"""`guaraci demo` fecha com um quadro que explica os 2 avisos CRITICOS e a acuracia 1,000 (V25, lote 4).

O dado sem confundimento classe x sessao fica para a v1.1; aqui o demo so' passa a EXPLICAR o aviso.
"""
import os
import subprocess

from guaraci import guaraci as g


def test_demo_fecha_com_o_quadro_dos_avisos_criticos(tmp_path, monkeypatch, capsys):
    def executar_falso(cfg):
        os.makedirs(os.path.join(cfg.output_root_folder, "demo", "PLSDA_OE_falso"), exist_ok=True)

    monkeypatch.setattr(g.pq, "executar", executar_falso)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(os, "startfile", lambda *a, **k: None, raising=False)
    monkeypatch.chdir(tmp_path)
    g._comando_demo()
    saida = capsys.readouterr().out
    assert "[AUDITORIA][CRITICO] acima significam" in saida
    assert "NAO prova desempenho" in saida
    assert "cada classe apareca em varias sessoes" in saida
    assert saida.index("Demo concluida") < saida.index("acima significam")
