"""Mede o PICO de RSS (Resident Set Size -- memoria fisica real, inclui
buffers C/numpy, ao contrario de tracemalloc que so' ve' o heap Python) do
GUARACI processando os maiores dados praticos disponiveis nesta sessao.

Grupo 3 do docs/MAPA_COMPLETUDE_V1.md -- "nunca medido ate agora".

METODO: cada caso roda como um SUBPROCESSO filho (`python medir_uso_memoria.py
--worker <caso>`), monitorado pelo processo pai via `psutil.Process(pid).
memory_info().rss` a cada 0.05s ate' o filho terminar. Isolar em subprocesso
evita que os imports/estado do PROPRIO script de medicao contaminem o pico
medido (o interesse e' o pico do GUARACI, nao do medidor).

CASOS E LIMITACOES HONESTAS (nenhum dado privado, nenhum download de GB
nesta rodada -- ver texto abaixo para o porque de cada escolha):

1. HSI (cubo hiperespectral): o dataset publico real (DeepHS Fruit) NAO
   pode ser baixado nesta sessao -- `scripts/download_datasets/
   baixar_deephs_fruit_todas.py` depende de um sidecar de pins
   (`_deephs_fruit_todas_pins.json`) que nao existe no repositorio (achado
   colateral desta medicao, registrado a parte, nao investigado aqui).
   Usamos cubos ENVI SINTETICOS com resolucao espacial 64x64 -- a MESMA do
   dataset publico real Kaki/VIS, confirmada em
   tests/test_hsi_pipeline.py::_montar_pasta_hsi_generica -- e 224 bandas
   (estimativa comum de camera pushbroom VIS/NIR; a contagem real de bandas
   do DeepHS nao foi confirmada nesta sessao). 60 gravacoes (30/classe,
   2 classes) ~ 210 MB de dado bruto total, ordem de grandeza maior que os
   6 bandas x 3 amostras dos testes unitarios.

2. Tabular (FT-NIR): sem dado privado disponivel neste ambiente (pasta
   `dados/` vazia). Usamos dado SINTETICO nas dimensoes do maior dataset
   PUBLICO ja integrado ao projeto (Mendeley 10.17632/ctgg7k4m5g.2, NIR
   8mm: n=100 amostras, ~11500 canais espectrais -- ver
   docs/VALIDACAO_PUBLICA.md). Contagens de repeticao de CV/bootstrap/
   permutacao reduzidas ao minimo (isola o custo de MEMORIA da dimensao dos
   dados do custo de TEMPO das repeticoes, que e' o Passo seguinte do
   Grupo 3).

3. GC-IMS bruto: NAO MEDIDO. GUARACI nao tem leitor de arquivo bruto de
   GC-IMS (so' aceita tabela de pico ja extraida -- Grupo 1 do
   MAPA_COMPLETUDE_V1.md) -- nao existe caminho de codigo para processar
   "dataset GC-IMS bruto" porque esse caminho de codigo nao existe. Reportar
   "N/A" e' mais honesto que inventar um numero.
"""
from __future__ import annotations

import argparse
import os
import sys as _sys_early
# console do Windows costuma ser cp1252 -- o worker tabular as vezes emite
# um caractere que nao codifica nela (mojibake ja visivel no proprio CLI,
# achado separado, nao investigado aqui); sem isto, so' IMPRIMIR o log
# capturado pode derrubar este script de medicao com UnicodeEncodeError.
if hasattr(_sys_early.stdout, "reconfigure"):
    _sys_early.stdout.reconfigure(encoding="utf-8", errors="replace")
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# geradores de dado sintetico (independentes dos testes -- self-contained)
# ---------------------------------------------------------------------------
def _gravar_cubo_envi(caminho_hdr_sem_ext: Path, cubo: np.ndarray, n_bandas: int) -> None:
    n_lin, n_col, _ = cubo.shape
    caminho_hdr_sem_ext.parent.mkdir(parents=True, exist_ok=True)
    caminho_hdr_sem_ext.with_suffix(".bin").write_bytes(cubo.astype("<f4").tobytes())
    caminho_hdr_sem_ext.with_suffix(".hdr").write_text(
        f"ENVI\nsamples = {n_col}\nlines = {n_lin}\nbands = {n_bandas}\n"
        f"header offset = 0\nfile type = ENVI Standard\ndata type = 4\n"
        f"interleave = bip\nbyte order = 0\n", encoding="utf-8")


def _mascara_objeto_suave(n_lin, n_col, raio_frac=0.35, largura_borda=4.0):
    yy, xx = np.ogrid[:n_lin, :n_col]
    cy, cx = (n_lin - 1) / 2.0, (n_col - 1) / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    raio = raio_frac * min(n_lin, n_col)
    return np.clip((raio - dist) / largura_borda + 0.5, 0.0, 1.0)


def _montar_pasta_hsi_grande(raiz: Path, n_amostras_por_classe: int,
                              n_lin: int, n_col: int, n_bandas: int, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    alpha = _mascara_objeto_suave(n_lin, n_col)[..., None]
    for classe, nivel_objeto in (("madura", 0.85), ("verde", 0.45)):
        for a in range(n_amostras_por_classe):
            pasta_amostra = raiz / classe / f"amostra{a}"
            fundo = rng.normal(loc=0.05, scale=0.01, size=(n_lin, n_col, n_bandas))
            objeto = rng.normal(loc=nivel_objeto, scale=0.01, size=(n_lin, n_col, n_bandas))
            cubo = (alpha * objeto + (1.0 - alpha) * fundo).astype("float32")
            _gravar_cubo_envi(pasta_amostra / "vista0", cubo, n_bandas)
    return raiz


# ---------------------------------------------------------------------------
# workers (rodam DENTRO do subprocesso filho monitorado)
# ---------------------------------------------------------------------------
def _worker_hsi() -> None:
    from guaraci.config import Config
    from guaraci.hsi_pipeline import run_hsi_pipeline

    with tempfile.TemporaryDirectory() as tmp:
        raiz = _montar_pasta_hsi_grande(
            Path(tmp) / "hsi_grande", n_amostras_por_classe=30,
            n_lin=64, n_col=64, n_bandas=224)
        cfg = Config(mode="hsi", hsi_dataset_folder=str(raiz),
                     output_root_folder=str(Path(tmp) / "saida"),
                     output_format="png")
        resumo = run_hsi_pipeline(cfg, modo_dataset="generico")
        print(f"[worker_hsi] gravacoes aceitas: {resumo.get('n_gravacoes_aceitas')}")


def _worker_tabular() -> None:
    import guaraci.pipeline as pq

    with tempfile.TemporaryDirectory() as tmp:
        cfg = pq.Config(
            input_folder=os.path.join(tmp, "dados"),
            output_root_folder=os.path.join(tmp, "saida"),
            mode="sintetico", n_per_class=34, n_synthetic_points=11500,
            wn_min=400.0, wn_max=4001.0,
            n_splits_cv=2, n_repeats_cv=1, n_permutations=3,
            n_permutations_wold=3, n_bootstrap_vip=3, n_bootstrap_bca=10,
            max_lvs=5,
        )
        os.makedirs(cfg.input_folder, exist_ok=True)
        pq.executar(cfg)
        print("[worker_tabular] executar() concluido")


_WORKERS = {"hsi": _worker_hsi, "tabular": _worker_tabular}


# ---------------------------------------------------------------------------
# monitor (roda no processo PAI)
# ---------------------------------------------------------------------------
def medir_pico_rss(caso: str, timeout_s: float = 600.0):
    # stdout/stderr vao para um ARQUIVO, nao um PIPE: um PIPE tem buffer
    # limitado (~64KB no Windows) e o loop de monitoramento abaixo so' le'
    # o stdout DEPOIS que o processo termina -- se o worker imprimir mais
    # que o buffer (o worker tabular imprime bastante: caminho de cada
    # figura, model card, diagnostico DDSimca por classe...) ele TRAVA
    # esperando espaco no pipe, e o loop fica preso ate' o timeout, dando
    # um falso positivo de "processo lento" que na verdade e' um deadlock
    # do proprio script de medicao (achado real desta rodada, corrigido
    # aqui -- ver nota no relatorio).
    with tempfile.TemporaryDirectory() as tmp_log:
        log_path = Path(tmp_log) / "saida.txt"
        env = {**os.environ, "PYTHONIOENCODING": "utf-8:replace"}
        with open(log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker", caso],
                cwd=str(REPO_ROOT), stdout=log_f, stderr=subprocess.STDOUT, env=env,
            )
            ps_proc = psutil.Process(proc.pid)
            pico_rss = 0
            pico_t = 0.0
            t0 = time.perf_counter()
            while proc.poll() is None:
                try:
                    rss = ps_proc.memory_info().rss
                    for filho in ps_proc.children(recursive=True):
                        try:
                            rss += filho.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    if rss > pico_rss:
                        pico_rss = rss
                        pico_t = time.perf_counter() - t0
                except psutil.NoSuchProcess:
                    break
                if time.perf_counter() - t0 > timeout_s:
                    proc.kill()
                    raise RuntimeError(f"worker '{caso}' excedeu timeout de {timeout_s}s")
                time.sleep(0.05)
        duracao = time.perf_counter() - t0
        saida_capturada = log_path.read_text(encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            print(saida_capturada)
            raise RuntimeError(f"worker '{caso}' saiu com codigo {proc.returncode}")
        return pico_rss, pico_t, duracao, saida_capturada


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=list(_WORKERS), default=None,
                     help="uso interno: roda so' o worker (chamado como subprocesso)")
    args = ap.parse_args()

    if args.worker:
        _WORKERS[args.worker]()
        sys.exit(0)

    print("=" * 78)
    print("GUARACI -- pico de RSS com os maiores dados disponiveis (Grupo 3)")
    print("=" * 78)

    for nome, caso in (("HSI (cubo sintetico 64x64x224, 60 gravacoes)", "hsi"),
                        ("Tabular (sintetico, n=102 amostras x 11500 canais, "
                         "escala do Mendeley NIR 8mm real)", "tabular")):
        print(f"\n--- {nome} ---")
        pico_rss, pico_t, duracao, saida = medir_pico_rss(caso)
        print(saida.strip())
        print(f"  pico RSS: {pico_rss / (1024**2):.1f} MB, em t={pico_t:.1f}s "
              f"(duracao total do worker: {duracao:.1f}s)")

    print("\n--- GC-IMS bruto ---")
    print("  N/A -- sem leitor de arquivo bruto de GC-IMS implementado "
          "(Grupo 1 do MAPA_COMPLETUDE_V1.md). Nao ha' caminho de codigo a medir.")
