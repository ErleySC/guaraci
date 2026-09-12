"""Mede o custo real de parede de `selecao_lv_cv_aninhada` (CV aninhada p/
numero de variaveis latentes, default True desde v1.0 -- ver comentario em
config.py linha ~218 e achado #12 de docs/BACKLOG_MULTIAGENTE.md) contra os
limiares de Nielsen. O comentario no codigo ja estima "~4-5x mais tempo";
este script MEDE, nao aceita a estimativa de cabeca.

Escala de dado: n=102 amostras (34/especie x 3 especies) x 11500 canais
espectrais -- a MESMA escala do maior dataset publico ja integrado
(Mendeley NIR 8mm, ver docs/VALIDACAO_PUBLICA.md) e, pelo comentario do
achado #12 em config.py, tambem da ordem de grandeza do dataset PRIVADO do
autor (~11500 variaveis colineares). Sem dado privado nesta medicao --
sintetico nessas dimensoes.

Grupo 3 do docs/MAPA_COMPLETUDE_V1.md, ultimo Passo -- "custo da CV
aninhada vs. tolerancia real de UX".

NOTA METODOLOGICA (retratacao parcial -- Regra 5 da instrucao): a 1a versao
deste script usava n_splits_cv=2/n_repeats_cv=1/max_lvs=5 (reduzidos p/
velocidade, reaproveitando a config do script de memoria) e mediu razao
~1x -- CONTRADIZENDO a estimativa "~4-5x" do comentario em config.py. A
CV aninhada roda um loop externo (n_splits_cv x n_repeats_cv folds) que,
dentro de cada fold, busca o n_opt de LVs sobre ate' max_lvs valores --
com os parametros reduzidos esse trabalho extra e' pequeno demais p/
aparecer sobre o custo fixo (dominado por outro passo, `chemometric_stats.
training_applicability_domain`, ~60% do tempo total medido por profiling
separado) que NAO escala com este flag. Corrigido aqui p/ n_splits_cv=5/
n_repeats_cv=3/max_lvs=40 (defaults reais do Config) -- so' assim o
parametro sob teste de fato varia o suficiente p/ o efeito aparecer.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

LIMIAR_ATENCAO = 10.0


def _rodar(nested: bool) -> float:
    import guaraci.pipeline as pq

    with tempfile.TemporaryDirectory() as tmp:
        cfg = pq.Config(
            input_folder=os.path.join(tmp, "dados"),
            output_root_folder=os.path.join(tmp, "saida"),
            mode="sintetico", n_per_class=34, n_synthetic_points=11500,
            wn_min=400.0, wn_max=4001.0,
            n_splits_cv=5, n_repeats_cv=3, n_permutations=3,
            n_permutations_wold=3, n_bootstrap_vip=3, n_bootstrap_bca=10,
            max_lvs=40, selecao_lv_cv_aninhada=nested,
        )
        os.makedirs(cfg.input_folder, exist_ok=True)
        t0 = time.perf_counter()
        pq.executar(cfg)
        return time.perf_counter() - t0


if __name__ == "__main__":
    import contextlib
    import io

    print("Rodando SEM CV aninhada (selecao_lv_cv_aninhada=False, metrica "
          "otimista conhecida)...")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        t_sem = _rodar(False)
    print(f"  tempo: {t_sem:.1f}s")

    print("\nRodando COM CV aninhada (selecao_lv_cv_aninhada=True, default "
          "desde v1.0, metrica honesta)...")
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        t_com = _rodar(True)
    print(f"  tempo: {t_com:.1f}s")

    razao = t_com / t_sem if t_sem > 0 else float("nan")
    print("\n=== Resumo ===")
    print(f"  sem CV aninhada: {t_sem:.1f}s")
    print(f"  com CV aninhada: {t_com:.1f}s")
    print(f"  razao (com/sem): {razao:.2f}x  "
          f"(comentario no codigo estima '~4-5x')")
    if t_com > LIMIAR_ATENCAO:
        print(f"  {t_com:.1f}s > {LIMIAR_ATENCAO}s (limite de atencao de "
              f"Nielsen) -- indicacao de progresso e' OBRIGATORIA aqui.")
    else:
        print(f"  {t_com:.1f}s <= {LIMIAR_ATENCAO}s -- dentro do limite de "
              f"atencao de Nielsen mesmo com CV aninhada, nesta escala.")
