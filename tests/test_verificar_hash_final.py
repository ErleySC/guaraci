"""scripts/verificar_hash_final.py: a regra do veredito (falha, nao executada, HEAD instavel) e a leitura do lado WSL."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verificar_hash_final as vhf  # noqa: E402


def _e(nome, estado, det=""):
    return {"nome": nome, "estado": estado, "detalhe": det, "duracao_s": 0.0}


def test_falha_e_etapa_falhou_ou_nao_executada_nunca_nao_se_aplica():
    etapas = [_e("a", "ok"), _e("b", "nao_se_aplica", "motivo"), _e("c", "falhou"), _e("d", "nao_executada")]
    assert vhf.contar_falhas(etapas) == 2
    assert vhf.contar_falhas([_e("a", "ok"), _e("b", "nao_se_aplica")]) == 0
    assert vhf.contar_falhas([]) == 0


def test_nao_executada_nao_e_aprovacao():
    r = vhf.montar_resultado("a" * 40, "a" * 40, "t", "r", [_e("x", "nao_executada")], "3.12", "i", "f")
    assert r["falhas"] == 1 and vhf.veredito(r) == 1


def test_head_que_mudou_invalida_o_resultado_mesmo_sem_falha():
    r = vhf.montar_resultado("a" * 40, "b" * 40, "t", "r", [_e("x", "ok")], "3.12", "i", "f")
    assert r["head_estavel"] is False and r["falhas"] == 0
    assert vhf.veredito(r) == 2


def test_zero_falhas_com_head_estavel_e_o_unico_zero():
    r = vhf.montar_resultado("a" * 40, "a" * 40, "t", "r", [_e("x", "ok"), _e("v", "nao_se_aplica", "m")], "3.12", "i", "f")
    assert vhf.veredito(r) == 0
    assert r["hash"] == "a" * 40 and r["arvore"] == "t"


def test_versoes_de_python_verdes_so_contam_as_que_passaram():
    etapas = [_e("suite_completa", "ok"), _e("pytest_3.10", "ok"), _e("pytest_3.11", "falhou"), _e("pytest_3.13", "ok")]
    assert vhf.versoes_python_ok(etapas, "3.12") == ["3.10", "3.12", "3.13"]
    assert vhf.versoes_python_ok([_e("suite_completa", "falhou")], "3.12") == []
    assert vhf.versoes_python_ok([_e("suite_completa", "ok"), _e("pytest_3.10", "ok")], "3.10") == ["3.10"]   # sem duplicata, ordem numerica


def test_parse_resumo_le_rc_e_o_resumo_do_pytest():
    t = ('lixo\nRESUMO bandit rc=0\nRESUMO semgrep rc=1\n'
         'RESUMO pytest_3.10 rc=0 resumo="2917 passed, 3 skipped in 1500s"\nRESUMO fim rc=0 pasta=/x\n')
    r = vhf.parse_resumo(t)
    assert r["bandit"] == (0, "") and r["semgrep"] == (1, "")
    assert r["pytest_3.10"] == (0, "2917 passed, 3 skipped in 1500s")
    assert "lixo" not in r


def test_etapas_wsl_ausentes_viram_nao_executadas_com_o_motivo():
    out = dict(vhf.etapas_wsl({}, "WSL indisponivel"))
    assert set(out) == {"bandit", "semgrep", "pip_audit_pisos", "pip_audit_lock", "instalacao_limpa_3.10", "instalacao_limpa_3.13",
                        "pytest_3.10", "pytest_3.11", "pytest_3.13"}
    assert all(v[0] == "nao_executada" and "WSL" in v[1] for v in out.values())


def test_etapas_wsl_rc_diferente_de_zero_falha():
    out = dict(vhf.etapas_wsl({"bandit": (0, ""), "semgrep": (1, "")}, "x"))
    assert out["bandit"][0] == "ok" and out["semgrep"][0] == "falhou" and "rc=1" in out["semgrep"][1]


def test_caminho_do_windows_para_o_wsl():
    s = vhf._caminho_wsl(Path("C:/Users/alguem/proj"))
    if sys.platform == "win32":
        assert s.startswith("/mnt/c/")


def test_por_rc_traduz_o_codigo_de_saida():
    assert vhf._por_rc(0, "") == ("ok", "ok")
    assert vhf._por_rc(1, "erro x") == ("falhou", "erro x")
    assert vhf._por_rc(2, "") == ("falhou", "rc=2")


def test_vault_sem_a_variavel_e_pendencia_do_autor_e_nao_aprovacao(monkeypatch):
    monkeypatch.delenv("GUARACI_VAULT", raising=False)
    estado, motivo = vhf.e_vault()
    assert estado == "pendente_do_autor" and "GUARACI_VAULT" in motivo
    assert vhf.contar_falhas([_e("vault", estado, motivo)]) == 0          # nao e' falha ...
    r = vhf.montar_resultado("a" * 40, "a" * 40, "t", "r", [_e("vault", estado, motivo)], "3.12", "i", "f")
    assert r["pendentes_do_autor"] == ["vault"] and vhf.veredito(r) == 0   # ... mas aparece, separada, no resultado


def test_vault_com_a_variavel_roda_o_medidor_e_traduz_o_codigo(monkeypatch):
    monkeypatch.setenv("GUARACI_VAULT", "x")
    monkeypatch.delenv("GUARACI_VAULT_GERADOR", raising=False)
    chamadas = []
    monkeypatch.setattr(vhf, "_rodar", lambda cmd, **k: (chamadas.append(cmd) or (0, "vault: notas 3")))
    assert vhf.e_vault() == ("ok", "vault: notas 3") and "--gerar" not in chamadas[0]
    monkeypatch.setenv("GUARACI_VAULT_GERADOR", "g.py")
    vhf.e_vault()
    assert "--gerar" in chamadas[1]
    monkeypatch.setattr(vhf, "_rodar", lambda cmd, **k: (1, "vault: notas com dado sensivel 2"))
    assert vhf.e_vault()[0] == "falhou"
    monkeypatch.setattr(vhf, "_rodar", lambda cmd, **k: (3, "vault: nao executado"))
    assert vhf.e_vault()[0] == "pendente_do_autor"


CAM = {"nucleo": {"linhas": 100.0, "ramos": 100.0, "combinada": 100.0}, "global": {"linhas": 97.0, "ramos": 93.0, "combinada": 95.5}}


def test_cobertura_minima_exige_nucleo_cem_e_global_noventa_e_cinco():
    assert vhf.avaliar_cobertura(CAM)[0] == "ok"
    assert vhf.avaliar_cobertura({**CAM, "nucleo": {**CAM["nucleo"], "ramos": 99.9}})[0] == "falhou"
    assert vhf.avaliar_cobertura({**CAM, "nucleo": {**CAM["nucleo"], "linhas": 99.9}})[0] == "falhou"
    assert vhf.avaliar_cobertura({**CAM, "global": {**CAM["global"], "combinada": 94.9}})[0] == "falhou"
    assert vhf.avaliar_cobertura(None)[0] == "nao_executada"


def test_trabalhadores_no_windows_sao_dois_por_padrao_e_nunca_mais_que_metade_dos_nucleos(monkeypatch):
    monkeypatch.delenv("GUARACI_TESTES_WORKERS", raising=False)
    for n, esperado in ((16, 2), (8, 2), (4, 2), (2, 1), (1, 1), (None, 1)):
        monkeypatch.setattr(vhf.os, "cpu_count", lambda n=n: n)
        assert vhf.trabalhadores() == esperado


def test_trabalhadores_pedido_pela_variavel_do_windows_tem_teto_de_quatro(monkeypatch):
    monkeypatch.setattr(vhf.os, "cpu_count", lambda: 16)
    for pedido, esperado in (("2", 2), ("1", 1), ("4", 4), ("8", 4), ("0", 1), ("-3", 1), ("abc", 2), ("", 2)):
        monkeypatch.setenv("GUARACI_TESTES_WORKERS", pedido)
        assert vhf.trabalhadores() == esperado
    monkeypatch.setenv("GUARACI_TESTES_WORKERS_WSL", "1")       # a variavel do WSL nao mexe no lado Windows
    monkeypatch.delenv("GUARACI_TESTES_WORKERS")
    assert vhf.trabalhadores() == 2


def test_arvore_suja_reprova(monkeypatch):
    monkeypatch.setattr(vhf, "_git", lambda *a: " M x.py\n?? y.py")
    assert vhf.e_arvore_limpa()[0] == "falhou"
    monkeypatch.setattr(vhf, "_git", lambda *a: "")
    assert vhf.e_arvore_limpa()[0] == "ok"


@pytest.mark.parametrize("resumo", ["", "sem rc aqui"])
def test_linha_sem_rc_nao_vira_etapa_ok(resumo):
    assert "RESUMO" not in vhf.parse_resumo(resumo)


def test_rodada_local_dos_formatos_reais_so_passa_com_registro_aprovado_e_em_dia(monkeypatch):
    import registrar_rodada_local as rrl
    monkeypatch.setattr(rrl, "registro_atual_cobre_o_codigo", lambda raiz=None: None)
    assert vhf.e_rodada_local_formatos()[0] == "ok"
    monkeypatch.setattr(rrl, "registro_atual_cobre_o_codigo", lambda raiz=None: "os leitores mudaram depois da rodada local registrada")
    estado, motivo = vhf.e_rodada_local_formatos()
    assert estado == "falhou" and "leitores mudaram" in motivo
    assert "rodada_local_formatos" in [n for n, _ in vhf.ETAPAS_LOCAIS]


def test_r5_com_lista_vazia_e_pendencia_do_autor_e_nao_zero_ocorrencias(monkeypatch):
    import auditar_exposicao as ae
    monkeypatch.setattr(ae, "termos_locais", lambda caminho=None: [])
    estado, detalhe = vhf.e_r5_termos_locais()
    assert estado == "pendente_do_autor" and "NAO EXECUTADO" in detalhe and "0 ocorrencias" not in detalhe
    monkeypatch.setattr(ae, "termos_locais", lambda caminho=None: ["termo"])
    estado, detalhe = vhf.e_r5_termos_locais()
    assert estado == "ok" and "1 termo" in detalhe


def test_cobertura_lacunas_compara_o_documento_com_o_gerado_da_cobertura_medida(tmp_path, monkeypatch):
    import gerar_cobertura_lacunas as gcl
    saida = tmp_path / "cobertura_lacunas.md"
    monkeypatch.setattr(gcl, "SAIDA", saida)
    monkeypatch.setattr(gcl, "gerar", lambda xml: "texto gerado\n")
    saida.write_text("texto gerado\n", encoding="utf-8")
    assert vhf._lacunas_em_dia(tmp_path / "c.xml")[0] == "ok"
    saida.write_text("velho\n", encoding="utf-8")
    estado, detalhe = vhf._lacunas_em_dia(tmp_path / "c.xml")
    assert estado == "falhou" and "desatualizado" in detalhe
    saida.unlink()
    assert vhf._lacunas_em_dia(tmp_path / "c.xml")[0] == "falhou"
    monkeypatch.setattr(gcl, "gerar", lambda xml: (_ for _ in ()).throw(ValueError("xml quebrado")))
    assert "xml quebrado" in vhf._lacunas_em_dia(tmp_path / "c.xml")[1]


def test_etapa_cobertura_lacunas_sem_suite_nao_executa(monkeypatch):
    monkeypatch.setitem(vhf.METRICAS, "suite_completa", {"cobertura": None, "cobertura_lacunas": None})
    assert vhf.e_cobertura_lacunas()[0] == "nao_executada"
    monkeypatch.setitem(vhf.METRICAS, "suite_completa", {"cobertura_lacunas": ("ok", "em dia")})
    assert vhf.e_cobertura_lacunas() == ("ok", "em dia")


def test_lado_wsl_instala_build_no_venv_da_matriz_antes_de_cada_pytest():
    # sem `build` o teste lento do pacote construido falha so' na matriz WSL (causa de uma verificacao final reprovada)
    sh = (Path(__file__).resolve().parents[1] / "scripts" / "verificar_wsl.sh").read_text(encoding="utf-8")
    laco = sh[sh.index('for v in 3.10 3.11 3.13; do'):]
    instala, roda = laco.find('-m pip install -q build'), laco.find('-m pytest tests/')
    assert -1 < instala < roda


JUNIT_UMA_FALHA = ('<?xml version="1.0"?><testsuites><testsuite name="pytest">'
                   '<testcase classname="tests.test_a" name="test_ok"/>'
                   '<testcase classname="tests.test_b" name="test_quebra[x]"><failure message="m">boom</failure></testcase>'
                   '</testsuite></testsuites>')


def _suite_falsa(monkeypatch, tmp_path, junit=JUNIT_UMA_FALHA):
    monkeypatch.setenv("GUARACI_REGISTROS_VERIFICACAO", str(tmp_path / "reg"))
    monkeypatch.setattr(vhf, "head", lambda: "a" * 40)

    def rodar(cmd, log=None, **k):
        destino = next(c.split("=", 1)[1] for c in cmd if c.startswith("--junitxml="))
        Path(destino).write_text(junit, encoding="utf-8")
        log.write_text("FAILED tests/test_b.py::test_quebra[x] - boom\n1 failed, 1 passed\n", encoding="utf-8")
        return 1, "1 failed, 1 passed"
    monkeypatch.setattr(vhf, "_rodar", rodar)


def test_junit_e_log_da_suite_ficam_gravados_fora_do_repositorio_depois_da_etapa(monkeypatch, tmp_path):
    _suite_falsa(monkeypatch, tmp_path)
    vhf.e_suite_completa()
    (pasta,) = (tmp_path / "reg").iterdir()
    assert pasta.name.startswith("a" * 12 + "_")
    assert {p.name for p in pasta.iterdir()} >= {"suite_completa_junit.xml", "suite_completa.log"}


def test_rodar_com_log_grava_a_saida_inteira_e_devolve_a_ultima_linha(tmp_path):
    log = tmp_path / "x.log"
    rc, ult = vhf._rodar([sys.executable, "-c", "print('primeira'); print('ultima')"], log=log)
    assert (rc, ult) == (0, "ultima") and log.read_text(encoding="utf-8").split() == ["primeira", "ultima"]


def test_lado_wsl_nunca_apaga_a_pasta_de_uma_execucao_anterior():
    sh = (Path(__file__).resolve().parents[1] / "scripts" / "verificar_wsl.sh").read_text(encoding="utf-8")
    assert 'rm -rf "$W"' not in sh and 'vhf_${H}_$(date' in sh


def test_suite_reprovada_grava_no_detalhe_o_nome_de_cada_teste_que_falhou(monkeypatch, tmp_path):
    _suite_falsa(monkeypatch, tmp_path)
    estado, det = vhf.e_suite_completa()
    assert estado == "falhou" and "falharam: tests.test_b::test_quebra[x]" in det and "test_ok" not in det


def test_nomes_alem_do_limite_viram_contagem():
    assert vhf._com_nomes("d", [f"t{i}" for i in range(23)]).endswith("t19 (+3)")
    assert vhf._com_nomes("d", []) == "d"


def test_falha_de_pytest_no_wsl_traz_o_nome_do_teste_para_a_etapa():
    texto = ('FALHOU pytest_3.10 tests/test_b.py::test_quebra[x]\n'
             'RESUMO pytest_3.10 rc=1 resumo="1 failed, 9 passed"\nRESUMO pytest_3.13 rc=0 resumo="10 passed"\n')
    r = vhf.parse_resumo(texto)
    assert r["pytest_3.10"] == (1, "1 failed, 9 passed; falharam: tests/test_b.py::test_quebra[x]")
    assert r["pytest_3.13"] == (0, "10 passed")
    sh = (Path(__file__).resolve().parents[1] / "scripts" / "verificar_wsl.sh").read_text(encoding="utf-8")
    assert 'echo "FALHOU pytest_$v $t"' in sh and "-rfE" in sh


@pytest.mark.parametrize("valor, esperado", [("3", "export GUARACI_TESTES_WORKERS_WSL=3 && "), ("", ""), ("x", ""), ("0", "")])
def test_limite_de_processos_do_wsl_vem_da_variavel_propria(monkeypatch, valor, esperado):
    monkeypatch.setenv("GUARACI_TESTES_WORKERS_WSL", valor)
    monkeypatch.setenv("GUARACI_TESTES_WORKERS", "2")
    assert vhf._limite_xdist_wsl() == esperado


def test_rodar_wsl_exporta_so_o_limite_do_wsl_e_nunca_o_do_windows(monkeypatch):
    vistos = []
    monkeypatch.setattr(vhf.subprocess, "run", lambda cmd, **k: vistos.append(cmd) or type("R", (), {"stdout": ""})())
    monkeypatch.setenv("GUARACI_TESTES_WORKERS", "2")
    monkeypatch.delenv("GUARACI_TESTES_WORKERS_WSL", raising=False)
    vhf.rodar_wsl("a" * 40, ["pytest_3.10"])
    monkeypatch.setenv("GUARACI_TESTES_WORKERS_WSL", "3")
    vhf.rodar_wsl("a" * 40, ["pytest_3.10"])
    assert vistos[0][-1].startswith("tr -d") and "GUARACI_TESTES_WORKERS" not in vistos[0][-1]
    assert vistos[1][-1].startswith("export GUARACI_TESTES_WORKERS_WSL=3 && tr -d")


MOTIVO_SEM_BASH = "bash utilizavel ausente"


def comando_bash(os_name: str = os.name, env=os.environ, which=shutil.which) -> "list[str] | None":
    """Comando que roda um script bash lido da entrada padrao. No Windows, o `bash.exe` do sistema roda na distro WSL PADRAO (que pode
    nem ter bash, ex.: docker-desktop) -- entao vai pelo `wsl -d ${GUARACI_WSL_DISTRO:-Ubuntu}`, como o verificador de producao."""
    if os_name == "nt":
        wsl = which("wsl")
        return [wsl, "-d", env.get("GUARACI_WSL_DISTRO", "Ubuntu"), "--", "bash", "-s"] if wsl else None
    bash = which("bash")
    return [bash, "-s"] if bash else None


def _bash_ou_pular() -> "list[str]":
    """O comando bash, testado DE FATO (`true`); sem bash utilizavel o teste pula com o motivo cadastrado em pulados_esperados.json."""
    cmd = comando_bash()
    if cmd is None:
        pytest.skip(f"{MOTIVO_SEM_BASH}: nem wsl nem bash no PATH")
    try:
        r = subprocess.run(cmd, input=b"true\n", capture_output=True, timeout=60)   # bytes: texto no Windows viraria \r\n
    except (OSError, subprocess.TimeoutExpired) as e:
        pytest.skip(f"{MOTIVO_SEM_BASH}: {type(e).__name__}")
    if r.returncode != 0:
        pytest.skip(f"{MOTIVO_SEM_BASH}: {' '.join(cmd[:-1])} devolveu {r.returncode}")
    return cmd


def test_comando_bash_no_windows_usa_distro_explicita_e_nunca_o_bash_exe_do_sistema():
    achar = {"wsl": r"C:\Windows\System32\wsl.exe", "bash": r"C:\Windows\System32\bash.exe"}.get
    assert comando_bash("nt", {}, achar) == [achar("wsl"), "-d", "Ubuntu", "--", "bash", "-s"]
    assert comando_bash("nt", {"GUARACI_WSL_DISTRO": "Debian"}, achar)[1:3] == ["-d", "Debian"]
    assert comando_bash("nt", {}, {"bash": r"C:\Windows\System32\bash.exe"}.get) is None   # sem wsl: pula, nao usa bash.exe
    assert comando_bash("posix", {}, {"bash": "/bin/bash"}.get) == ["/bin/bash", "-s"]


def test_toda_chamada_ao_wsl_nos_scripts_tem_distribuicao_explicita():
    import re
    sem_distro = []
    for arq in sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*")):
        if arq.suffix not in (".py", ".sh", ".ps1", ".bat", ".cmd"):
            continue
        for n, linha in enumerate(arq.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            # lista de argumentos (["wsl", ...]) ou linha de comando (wsl -- / wsl bash / wsl.exe ...)
            if re.search(r"""["']wsl(\.exe)?["']\s*,(?!\s*["']-d["'])""", linha) \
                    or re.search(r"\bwsl(\.exe)?\s+(--|bash\b|sh\b|python|-e\b|-u\b|~)", linha):
                sem_distro.append(f"{arq.name}:{n}: {linha.strip()}")
    assert not sem_distro, sem_distro


def _n_do_wsl(nproc, valor):
    """Roda, em bash de verdade, as linhas do verificar_wsl.sh que escolhem N (com `nproc` simulado)."""
    sh = (Path(__file__).resolve().parents[1] / "scripts" / "verificar_wsl.sh").read_text(encoding="utf-8")
    linhas = [ln for ln in sh.splitlines() if ln.startswith(('N="${GUARACI_TESTES_WORKERS_WSL', "M=$(( $(nproc)"))]
    assert len(linhas) == 2
    bash = _bash_ou_pular()
    # as variaveis vao DENTRO do script: o bash do WSL nao herda o ambiente do Windows
    wsl = f"export GUARACI_TESTES_WORKERS_WSL='{valor}'" if valor is not None else "unset GUARACI_TESTES_WORKERS_WSL"
    script = "\n".join([f"nproc() {{ echo {nproc}; }}", wsl, "export GUARACI_TESTES_WORKERS=1",   # a do Windows nao mexe no WSL
                        *linhas, 'echo "$N"', ""])
    r = subprocess.run(bash, input=script.encode(), capture_output=True, timeout=60)
    assert r.returncode == 0, r.stderr.decode(errors="replace")
    return int(r.stdout.decode().strip())


@pytest.mark.parametrize("nproc, valor, esperado", [(16, None, 4), (16, "2", 2), (16, "8", 4), (16, "0", 1), (16, "abc", 4),
                                                    (16, "", 4), (4, None, 2), (1, None, 1)])
def test_lado_wsl_em_bash_real_padrao_quatro_e_a_variavel_so_reduz(nproc, valor, esperado):
    assert _n_do_wsl(nproc, valor) == esperado
