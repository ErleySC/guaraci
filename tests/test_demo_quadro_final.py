"""`guaraci demo` fecha com um quadro que explica os 2 avisos CRITICOS e por que a acuracia nao prova desempenho
(V25, lote 4; sem numero fixo desde que a acuracia do demo deixou de ser 1,000).

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
    assert "nenhuma acuracia deste demo prova desempenho" in saida and "1,000" not in saida
    assert "cada classe apareca em varias sessoes" in saida
    assert saida.index("Demo concluida") < saida.index("acima significam")


def test_demo_em_ingles_fecha_com_o_quadro_em_ingles(tmp_path, monkeypatch, capsys):
    def executar_falso(cfg):
        print("[INFO] Truncamento espectral [400, 4001]: 0/300 variaveis removidas, restam 300 (400.0-4001.0).")
        os.makedirs(os.path.join(cfg.output_root_folder, "demo", "PLSDA_OE_falso"), exist_ok=True)

    monkeypatch.setenv("GUARACI_IDIOMA", "en")
    monkeypatch.setattr(g.pq, "executar", executar_falso)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(os, "startfile", lambda *a, **k: None, raising=False)
    monkeypatch.chdir(tmp_path)
    g._comando_demo()
    saida = capsys.readouterr().out
    assert "Spectral truncation [400, 4001]: 0/300 variables removed" in saida
    assert "Demo finished." in saida and "no accuracy from this demo proves performance" in saida
    assert "Demo concluida" not in saida and "variaveis removidas" not in saida
