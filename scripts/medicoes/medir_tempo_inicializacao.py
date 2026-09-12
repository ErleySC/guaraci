"""Mede o tempo de inicializacao do GUARACI -- CLI e Streamlit -- contra os
limiares de Nielsen (Usability Engineering, 1993, cap.5, consolidando Miller
1968): 0.1s = resposta instantanea, 1.0s = fluxo de pensamento preservado,
10s = limite de atencao (acima disso, indicacao de progresso obrigatoria).

Grupo 3 do docs/MAPA_COMPLETUDE_V1.md -- "nunca medido ate agora".

O QUE E MEDIDO (e como)
------------------------
1. CLI cold start: tempo de parede entre o spawn do processo `guaraci` (sem
   argumentos, ou seja, o loop interativo) e o momento em que a "primeira
   tela pronta" e alcancada. Como o codigo-fonte nao e' instrumentado (nao
   modificamos guaraci.py so' para medir), usamos um proxy honesto: stdin
   fechado (DEVNULL) faz o primeiro `input()` do loop levantar EOFError
   IMEDIATAMENTE apos a tela ser desenhada -- o processo entao imprime a
   despedida e sai. O tempo total do processo e', portanto, tempo-ate-
   primeira-tela + custo trivial de saida (sub-milissegundo). 10 repeticoes
   cold (processo novo a cada vez) para media/desvio-padrao, com HOME/
   USERPROFILE isolados num diretorio temporario (nao toca ~/.guaraci real).

2. Decomposicao de import: `python -X importtime` numa invocacao que aciona
   a MESMA cadeia de imports do menu interativo (guaraci.guaraci importa
   guaraci.pipeline no topo do modulo, independente do comando escolhido) --
   usamos o comando "version", que sai sem desenhar o menu nem dormir 1s,
   isolando o custo de IMPORT do resto.

3. Streamlit: (a) tempo ate o servidor aceitar conexao TCP (boot do
   processo) e (b) tempo ate a primeira resposta HTTP 200 em "/" (proxy de
   "pagina comecou a ser servida" -- NAO e' o mesmo que "app Python
   terminou de rodar e renderizou componentes", que so' acontece depois que
   o cliente JS abre um websocket; ver LIMITACAO abaixo). 5 repeticoes,
   porta livre nova a cada rodada, processo morto ao final de cada uma.

LIMITACAO CONHECIDA (documentada, nao escondida)
--------------------------------------------------
O item 3(b) mede a resposta HTTP do shell estatico do Streamlit, nao o
render completo do app apos a conexao websocket (que exigiria um browser
real -- Playwright nao e' dependencia do projeto e nao sera adicionado so'
para esta medicao, o que seria ironico num grupo que audita peso de
dependencias). Um cross-check manual via browser real foi feito
separadamente pelo agente na mesma rodada desta medicao e esta registrado
em prosa no relatorio, nao neste script.

Sem dado privado: nao le nenhum arquivo de dados do usuario.
"""
from __future__ import annotations

import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

LIMIAR_INSTANTANEO = 0.1
LIMIAR_FLUXO = 1.0
LIMIAR_ATENCAO = 10.0

REPO_ROOT = Path(__file__).resolve().parents[2]


def _classificar(segundos: float) -> str:
    if segundos <= LIMIAR_INSTANTANEO:
        return "INSTANTANEO (<=0.1s)"
    if segundos <= LIMIAR_FLUXO:
        return "ACEITAVEL, fluxo preservado (<=1.0s)"
    if segundos <= LIMIAR_ATENCAO:
        return "TOLERAVEL SE houver indicacao de progresso (<=10s)"
    return "ACIMA DO LIMITE DE ATENCAO -- indicacao de progresso obrigatoria (>10s)"


def _resumo(nome: str, tempos: list[float]) -> None:
    media = statistics.mean(tempos)
    desvio = statistics.stdev(tempos) if len(tempos) > 1 else 0.0
    print(f"\n=== {nome} ===")
    print(f"  n={len(tempos)}  media={media:.3f}s  desvio_padrao={desvio:.3f}s"
          f"  min={min(tempos):.3f}s  max={max(tempos):.3f}s")
    print(f"  classificacao (Nielsen, sobre a media): {_classificar(media)}")


# ---------------------------------------------------------------------------
# 1. CLI cold start
# ---------------------------------------------------------------------------
def medir_cli_cold_start(n: int = 10) -> list[float]:
    exe = shutil.which("guaraci")
    if exe is None:
        raise RuntimeError(
            "executavel 'guaraci' nao encontrado no PATH -- ative o venv "
            "do projeto antes de rodar este script.")

    tmp_home = tempfile.mkdtemp(prefix="guaraci_medicao_home_")
    env_isolado = {
        "HOME": tmp_home, "USERPROFILE": tmp_home,
        "PATH": __import__("os").environ.get("PATH", ""),
        "SYSTEMROOT": __import__("os").environ.get("SystemRoot", ""),
    }

    tempos = []
    for i in range(n):
        t0 = time.perf_counter()
        subprocess.run(
            [exe], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=env_isolado, timeout=30,
        )
        tempos.append(time.perf_counter() - t0)
        print(f"  rodada {i + 1}/{n}: {tempos[-1]:.3f}s")

    shutil.rmtree(tmp_home, ignore_errors=True)
    return tempos


# ---------------------------------------------------------------------------
# 2. Decomposicao de import (-X importtime)
# ---------------------------------------------------------------------------
def medir_importtime(top_n: int = 20) -> None:
    codigo = "from guaraci.guaraci import main; main(['version'])"
    resultado = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", codigo],
        capture_output=True, text=True, timeout=30,
    )
    linhas = resultado.stderr.splitlines()
    entradas = []  # (self_us, cumulative_us, nome_modulo)
    for linha in linhas:
        if not linha.startswith("import time:"):
            continue
        corpo = linha[len("import time:"):]
        partes = corpo.split("|")
        if len(partes) != 3:
            continue
        try:
            self_us = int(partes[0].strip())
            cum_us = int(partes[1].strip())
        except ValueError:
            continue
        nome = partes[2].strip()
        entradas.append((self_us, cum_us, nome))

    if not entradas:
        print("  [aviso] nao foi possivel parsear a saida de -X importtime "
              "(formato pode ter mudado entre versoes de Python).")
        print("  stderr bruto (ultimas 10 linhas):")
        for linha in linhas[-10:]:
            print(f"    {linha}")
        return

    tempo_total_us = max(cum for _, cum, nome in entradas if not nome.startswith(" "))
    print(f"\n  tempo total de import (raiz, cumulative): {tempo_total_us / 1e6:.3f}s")

    print(f"\n  top {top_n} modulos por SELF time (custo proprio, sem filhos):")
    for self_us, _cum, nome in sorted(entradas, key=lambda e: -e[0])[:top_n]:
        print(f"    {self_us / 1000:8.1f} ms   {nome}")

    alvos = ["numpy", "scipy", "pandas", "sklearn", "matplotlib", "rich",
             "PIL", "psutil", "streamlit", "yaml", "joblib", "tensorly",
             "brukeropus", "prcv", "shap", "xgboost", "skimage"]
    # MAX cumulative por raiz, nao a primeira ocorrencia: sys.modules cacheia
    # apos o 1o import real, entao a MESMA raiz aparece varias vezes na saida
    # -- um submodulo interno aninhado (self+cumulative pequenos, impresso
    # ANTES da linha-pai por causa da ordem postorder de -X importtime) e
    # reimports posteriores ja cacheados (~0). Pegar o maior valor visto
    # captura a carga real e completa do pacote, nao um fragmento dela.
    print("\n  cumulative das dependencias pesadas conhecidas (maior valor "
          "visto p/ raiz do pacote, se importadas):")
    maior_por_raiz: dict[str, int] = {}
    for self_us, cum_us, nome in entradas:
        raiz = nome.strip().split(".")[0]
        if raiz in alvos:
            maior_por_raiz[raiz] = max(maior_por_raiz.get(raiz, 0), cum_us)
    for raiz in sorted(maior_por_raiz, key=lambda r: -maior_por_raiz[r]):
        print(f"    {maior_por_raiz[raiz] / 1000:8.1f} ms   {raiz}")
    vistos = set(maior_por_raiz)
    ausentes = [a for a in alvos if a not in vistos]
    if ausentes:
        print(f"  NAO importados no boot do CLI (bom sinal, sao extras "
              f"opcionais ou carregados sob demanda): {', '.join(ausentes)}")


# ---------------------------------------------------------------------------
# 3. Streamlit boot
# ---------------------------------------------------------------------------
def _porta_livre() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def medir_streamlit_boot(n: int = 5, timeout_s: float = 60.0):
    app = REPO_ROOT / "app_quimiometria.py"
    if not app.is_file():
        raise RuntimeError(f"app nao encontrado: {app}")

    tempos_tcp = []
    tempos_http = []
    for i in range(n):
        porta = _porta_livre()
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(app),
            "--server.headless=true", f"--server.port={porta}",
            "--server.address=127.0.0.1", "--server.runOnSave=false",
            "--browser.gatherUsageStats=false",
            "--server.fileWatcherType=none",
        ]
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        try:
            # (a) aceita conexao TCP
            t_tcp = None
            while time.perf_counter() - t0 < timeout_s:
                try:
                    with socket.create_connection(("127.0.0.1", porta), timeout=0.2):
                        t_tcp = time.perf_counter() - t0
                        break
                except OSError:
                    time.sleep(0.05)
            if t_tcp is None:
                raise RuntimeError(f"streamlit nao abriu porta em {timeout_s}s")

            # (b) primeira resposta HTTP 200
            t_http = None
            while time.perf_counter() - t0 < timeout_s:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{porta}/", timeout=0.5
                    ) as resp:
                        if resp.status == 200:
                            t_http = time.perf_counter() - t0
                            break
                except (urllib.error.URLError, OSError, TimeoutError):
                    time.sleep(0.05)
            if t_http is None:
                raise RuntimeError(f"streamlit nao respondeu HTTP em {timeout_s}s")

            tempos_tcp.append(t_tcp)
            tempos_http.append(t_http)
            print(f"  rodada {i + 1}/{n}: TCP={t_tcp:.3f}s  HTTP_200={t_http:.3f}s")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    return tempos_tcp, tempos_http


if __name__ == "__main__":
    print("=" * 78)
    print("GUARACI -- medicao de tempo de inicializacao (Grupo 3, MAPA_COMPLETUDE_V1)")
    print("Limiares de Nielsen: 0.1s instantaneo | 1.0s fluxo | 10s atencao")
    print("=" * 78)

    print("\n[1/3] CLI cold start (10 execucoes, stdin fechado -> EOF na "
          "primeira tela)...")
    tempos_cli = medir_cli_cold_start(10)
    _resumo("CLI cold start (spawn -> primeira tela pronta, proxy)", tempos_cli)

    print("\n[2/3] Decomposicao de import (-X importtime, comando 'version')...")
    medir_importtime()

    print("\n[3/3] Streamlit boot (5 execucoes)...")
    tempos_tcp, tempos_http = medir_streamlit_boot(5)
    _resumo("Streamlit: boot ate aceitar TCP", tempos_tcp)
    _resumo("Streamlit: boot ate primeira resposta HTTP 200 (shell estatico)", tempos_http)

    print("\n" + "=" * 78)
    print("FIM. Ver docstring deste arquivo para a limitacao conhecida do "
          "item 3(b) (nao mede o render completo pos-websocket).")
