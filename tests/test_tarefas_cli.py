"""Teste de tarefas pelo CLI de verdade: um novo usuario cumpre cada tarefa so com o
que o programa mostra, dentro de um teto de passos, sem rede e sem material externo.

Cada tarefa e' um roteiro de teclas enviado ao `python -m guaraci` (subprocesso, HOME
isolado). "Passos" = linhas digitadas. Falha se exceder o teto, se a saida esperada nao
aparecer ou se o programa pedir algo fora do roteiro.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


def _roda(roteiro, tmp_path, *, timeout=300):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    # Os roteiros conferem os textos em portugues; um usuario novo abre em ingles, entao o idioma ja vem escolhido.
    (home / ".guaraci").mkdir(exist_ok=True)
    (home / ".guaraci" / ".cli_wizard_done").write_text("PT", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8",
           "USERPROFILE": str(home), "HOME": str(home), "NO_PROXY": "*", "HTTP_PROXY": "http://0.0.0.0:9",
           "HTTPS_PROXY": "http://0.0.0.0:9"}                       # qualquer tentativa de rede falha
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "guaraci"], input="\n".join(roteiro) + "\n", capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env, cwd=str(tmp_path), timeout=timeout)
    return r.stdout, time.perf_counter() - t0, r.returncode


@pytest.mark.parametrize("nome, roteiro, teto, esperado", [
    ("abrir_dados_e_voltar", ["2.3", "v", "q"], 3,
     ["Menu › 2 Planejar e preparar dados › 2.3 Dados", "? ajuda | G Guaraci | I idioma | C confianca | R rodar | V voltar"]),
    ("ajuda_de_um_campo", ["3.1", "?", "1", "", "v", "q"], 6, ["Nivel de analise"]),
    ("ver_a_confianca_de_uma_tecnica", ["C", "pls_da", "", "", "q"], 5, ["PLS-DA", "em validacao", "Evidencia:"]),
    ("assistente_rota_classificar_csv", ["G", "7", "2", "2", "", "q"], 6,
     ["Rota recomendada", "guaraci exemplo classificar"]),
    ("assistente_explica_termo", ["G", "8", "pseudo-replicacao", "", "q"], 5, ["Tratar replicas como se fossem"]),
    ("assistente_diagnostica_aviso", ["G", "10", "[AUDITORIA][AVISO] n_insuficiente: ...", "", "q"], 5,
     ["O que fazer:", "Colete mais amostras fisicas"]),
])
def test_tarefa_leve_dentro_do_teto_de_passos(tmp_path, nome, roteiro, teto, esperado):
    assert len(roteiro) <= teto, f"{nome}: {len(roteiro)} passos > teto {teto}"
    saida, _tempo, rc = _roda(roteiro, tmp_path)               # o tempo e' medido na documentacao; nao e' afirmado (depende da maquina)
    assert rc == 0, saida[-500:]
    for trecho in esperado:
        assert trecho in saida, f"{nome}: faltou {trecho!r}"


def test_tarefa_gerar_exemplo_pelo_menu(tmp_path):
    destino = tmp_path / "ex"
    roteiro = ["1.1", "2", str(destino), "n", "", "v", "q"]
    assert len(roteiro) <= 8
    saida, _, rc = _roda(roteiro, tmp_path)
    assert rc == 0 and (destino / "config.yaml").is_file() and (destino / "dados.csv").is_file()
    assert "guaraci run" in saida


def test_tarefa_exemplo_e_anova_so_por_comandos(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8",
           "USERPROFILE": str(tmp_path / "h"), "HOME": str(tmp_path / "h")}
    r1 = subprocess.run([sys.executable, "-m", "guaraci", "exemplo", "comparar", "cmp"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                        env=env, cwd=str(tmp_path), timeout=120)
    assert r1.returncode == 0
    r2 = subprocess.run([sys.executable, "-m", "guaraci", "anova", "cmp/tabela.csv", "--resposta", "teor", "--grupo", "grupo",
                         "--unidade", "amostra", "--saida", "cmp/res", "--sem-bootstrap"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                        env=env, cwd=str(tmp_path), timeout=120)
    assert r2.returncode == 0 and "diferenca entre grupos" in r2.stdout
    assert (tmp_path / "cmp" / "res" / "tabelas" / "anova_por_resposta.csv").is_file()


@pytest.mark.slow
@pytest.mark.parametrize("nome, tecla_cenario, achado", [
    ("primeiros_5_minutos_classificar", "2", "resumo_modelo.txt"),
    ("quantificar_pelo_menu", "3", "resumo_modelo.txt"),
    ("multielementar_pelo_menu", "4", "multielementar.json"),
])
def test_tarefa_do_exemplo_ate_o_resultado_pelo_menu(tmp_path, nome, tecla_cenario, achado):
    destino = tmp_path / "ex"
    roteiro = ["1.1", tecla_cenario, str(destino), "r", "", "s", "", "q", "S"]
    assert len(roteiro) <= 10, f"{nome}: {len(roteiro)} passos"
    saida, _, rc = _roda(roteiro, tmp_path, timeout=600)
    assert rc == 0, saida[-800:]
    assert list(destino.rglob(achado)), f"{nome}: a execucao nao gerou {achado}"
    assert (list(destino.rglob("config_efetiva.yaml"))), f"{nome}: sem config_efetiva.yaml"


def test_tarefa_monitorar_pelo_menu_com_o_exemplo(tmp_path):
    """Grupo Monitorar: tela [D], opcao x (exemplo sintetico) -- gera, roda e mostra os alarmes."""
    roteiro = ["4.2", "x", "", "q"]
    saida, _, rc = _roda(roteiro, tmp_path, timeout=300)
    assert rc == 0, saida[-800:]
    assert "Menu › 4 Comparar grupos e monitorar processos › 4.2 Monitorar lotes" in saida
    assert "lote N05" in saida and "ALARME" in saida
    assert list(tmp_path.rglob("monitor_lotes.csv"))


def test_tarefa_monitorar_pelo_exemplo_do_menu(tmp_path):
    destino = tmp_path / "mon"
    roteiro = ["1.1", "monitorar", str(destino), "r", "", "v", "q"]
    saida, _, rc = _roda(roteiro, tmp_path, timeout=300)
    assert rc == 0, saida[-800:]
    assert (destino / "resultados_monitoramento" / "tabelas" / "monitor_lotes.csv").is_file()


def test_tarefa_transferir_pelo_exemplo_do_menu(tmp_path):
    """Tela [E] com o exemplo de transferencia: nao ha config.yaml; roda o mesmo comando de `guaraci exemplo`."""
    destino = tmp_path / "tr"
    roteiro = ["1.1", "transferir", str(destino), "r", "", "v", "q"]
    saida, _, rc = _roda(roteiro, tmp_path, timeout=300)
    assert rc == 0, saida[-800:]
    assert "config.yaml" not in saida
    assert (destino / "resultados_transferencia" / "novos_no_espaco_do_mestre.csv").is_file()


@pytest.mark.slow
@pytest.mark.parametrize("nome, cenario, achado", [
    ("explorar_pelo_menu", "explorar", "resumo_modelo.txt"),
    ("imagem_pelo_menu", "imagem", "resumo_modelo.txt"),
    ("hsi_pelo_menu", "hsi", "hsi_relatorio"),
])
def test_tarefa_exemplo_por_nome_ate_o_resultado_pelo_menu(tmp_path, nome, cenario, achado):
    destino = tmp_path / "ex"
    roteiro = ["1.1", cenario, str(destino), "r", "", "s", "", "q", "S"]
    assert len(roteiro) <= 10, f"{nome}: {len(roteiro)} passos"
    saida, _, rc = _roda(roteiro, tmp_path, timeout=600)
    assert rc == 0, saida[-800:]
    assert list(destino.rglob(achado + "*")), f"{nome}: a execucao nao gerou {achado}"


@pytest.mark.parametrize("nome, roteiro, teto, esperado", [
    ("comparar_grupos_com_o_exemplo", ["4.1", "x", "", "q"], 4, ["Menu › 4 Comparar grupos e monitorar processos › 4.1 Comparar grupos", "diferenca entre grupos"]),
    ("perfil_explorar_rapido", ["1.2", "1", "1", "", "", "q"], 6, ["Passo 2 de 2", "Perfil aplicado"]),
    ("fluxo_planejar_selecionar_auditar_pulando", ["2.1", "2", "2", "2", "q"], 5,
     ["Planejar, selecionar e auditar", "1. Planejar a coleta", "3. Auditar o delineamento"]),
])
def test_tarefa_telas_novas_do_menu_por_objetivo(tmp_path, nome, roteiro, teto, esperado):
    assert len(roteiro) <= teto
    saida, _, rc = _roda(roteiro, tmp_path)
    assert rc == 0, saida[-500:]
    for trecho in esperado:
        assert trecho in saida, f"{nome}: faltou {trecho!r}"


@pytest.mark.slow
def test_tarefa_relatorio_pela_tela_z(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
           "USERPROFILE": str(tmp_path / "home"), "HOME": str(tmp_path / "home")}
    (tmp_path / "home").mkdir()
    subprocess.run([sys.executable, "-m", "guaraci", "exemplo", "classificar", "ex", "--rodar"], capture_output=True,
                   env=env, cwd=str(tmp_path), timeout=600, check=True)
    run = next((tmp_path / "ex").rglob("resumo_modelo.txt")).parents[1]
    saida, _, rc = _roda(["5.2", str(run), "xlsx", "", "q"], tmp_path)
    assert rc == 0 and (run / "Relatorios" / "relatorio.xlsx").is_file(), saida[-500:]


def test_tarefa_planejar_coleta_pela_tela_j(tmp_path):
    """plano_coleta/plano_amostral pelo menu: classes, sessoes, alvo, valor, prefixo, sem PDF, sem refinar."""
    prefixo = tmp_path / "plano"
    roteiro = ["2.7", "A,B,C", "3", "C", "0.10", str(prefixo), "n", "n", "", "q", "S"]
    assert len(roteiro) <= 11
    saida, _, rc = _roda(roteiro, tmp_path)
    assert rc == 0, saida[-800:]
    assert (tmp_path / "plano.md").is_file() and (tmp_path / "plano.xlsx").is_file()
    assert "REPLICA TECNICA != AMOSTRA FISICA" in saida


@pytest.mark.slow
def test_tarefa_extras_hsi_pela_tela_x(tmp_path):
    """hsi_extras/hsi_applicability/hsi_resampling pela tela X sobre o exemplo hsi."""
    env = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8",
           "USERPROFILE": str(tmp_path / "home"), "HOME": str(tmp_path / "home")}
    (tmp_path / "home").mkdir()
    subprocess.run([sys.executable, "-m", "guaraci", "exemplo", "hsi", "ex"], capture_output=True, env=env,
                   cwd=str(tmp_path), timeout=300, check=True)
    roteiro = ["3.6", str(tmp_path / "ex" / "cubos"), "1", "2", "3", "", "q", "S"]
    saida, _, rc = _roda(roteiro, tmp_path, timeout=600)
    assert rc == 0, saida[-800:]
    assert "Dentro do dominio" in saida and "objetos distintos" in saida and "N-PLS bal.acc=" in saida


@pytest.mark.slow
def test_tarefa_benchmark_de_regressao_com_dual_spls(tmp_path):
    """dual_spls pelo CLI: exemplo quantificar + benchmark_regressao: true + guaraci run."""
    env = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
           "USERPROFILE": str(tmp_path / "home"), "HOME": str(tmp_path / "home")}
    (tmp_path / "home").mkdir()
    subprocess.run([sys.executable, "-m", "guaraci", "exemplo", "quantificar", "q"], capture_output=True, env=env,
                   cwd=str(tmp_path), timeout=300, check=True)
    with open(tmp_path / "q" / "config.yaml", "a", encoding="utf-8") as fh:
        fh.write("benchmark_regressao: true\n")
    r = subprocess.run([sys.executable, "-m", "guaraci", "run", "q/config.yaml"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env, cwd=str(tmp_path), timeout=600)
    assert r.returncode == 0, r.stdout[-800:]
    assert "Dual-sPLS" in r.stdout
