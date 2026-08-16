"""Reproduz o mecanismo exato do painel de progresso do CLI
(`_rodar_pipeline` em guaraci.py): thread em background rodando executar()
dentro de contextlib.redirect_stdout/redirect_stderr, thread principal
fazendo poll de app_logic.progresso_do_log a cada 0.3s -- sem Rich Live
(isolando so' a logica de progresso, nao a renderizacao no terminal).

Mede o "bug do progresso" relatado em 2026-08-07: a etapa "[6/7]" (figuras +
DD-SIMCA + OPLS-DA + holdout) concentra a maior parte do tempo real de
execucao, mas so' tinha 2 marcadores de texto OPCIONAIS entre o inicio e o
fim -- sem eles, a fracao reportada ficava CRAVADA em 6/7=0.857 durante toda
essa fase. Compara ANTES (sem total_figuras_planejadas) e DEPOIS (com) da
correcao em app_logic.progresso_do_log.
"""
import contextlib
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, "src")
import guaraci.pipeline as pq
from guaraci.app_logic import LogThreadSafe, progresso_do_log


def _rodar_e_medir(total_figuras_planejadas):
    with tempfile.TemporaryDirectory() as tmp:
        cfg = pq.Config(
            pasta_entrada=os.path.join(tmp, "dados"),
            pasta_saida_raiz=os.path.join(tmp, "saida"),
            modo="sintetico", n_por_classe=8, n_pontos_sint=50,
            wn_min=400.0, wn_max=4001.0,
            n_splits_cv=2, n_repeats_cv=1, n_permutacoes=5,
            n_permutacoes_wold=5, n_bootstrap_vip=3, n_bootstrap_bca=20,
            n_monte_carlo=3, max_lvs=5,
        )
        os.makedirs(cfg.pasta_entrada, exist_ok=True)

        _done = {"ok": False, "error": None}
        _logger = LogThreadSafe()

        def _run():
            try:
                with contextlib.redirect_stdout(_logger), \
                     contextlib.redirect_stderr(_logger):
                    pq.executar(cfg)
            except Exception as e:  # noqa: BLE001 -- script de diagnostico
                # standalone: qualquer falha do pipeline deve virar mensagem
                # no relatorio de medicao, nao um traceback que interrompe
                # o script antes de imprimir o historico ja coletado.
                _done["error"] = str(e)
            finally:
                _done["ok"] = True

        thr = threading.Thread(target=_run, daemon=True)
        t0 = time.time()
        thr.start()

        historico = []
        while not _done["ok"]:
            txt = _logger.text()
            frac, label = progresso_do_log(txt, total_figuras_planejadas)
            historico.append((time.time() - t0, frac, label))
            time.sleep(0.1)
        thr.join()

        if _done["error"]:
            raise RuntimeError(f"executar() falhou: {_done['error']}")
        return historico


def _resume(historico, titulo):
    duracao_total = historico[-1][0]
    valores = [round(f, 3) for _, f, _ in historico]
    moda = max(set(valores), key=valores.count)
    n_na_moda = sum(1 for v in valores if v == moda)
    pct_parado = 100 * n_na_moda / len(valores)
    print(f"=== {titulo} ===")
    print(f"  duracao total: {duracao_total:.1f}s ({len(historico)} amostras)")
    print(f"  fracao mais frequente (moda): {moda:.3f}")
    print(f"  % do tempo/amostras nessa fracao: {pct_parado:.1f}%")
    print("  progressao (t, frac, label) a cada ~1s:")
    ultimo_t = -1.0
    for t, f, label in historico:
        if t - ultimo_t >= 1.0 or t == historico[-1][0]:
            print(f"    {t:5.1f}s  {f:5.3f}  {label}")
            ultimo_t = t
    print()
    return pct_parado


if __name__ == "__main__":
    print("Rodando SEM a correcao (total_figuras_planejadas=None -- "
          "comportamento antigo)...\n")
    hist_antes = _rodar_e_medir(None)
    pct_antes = _resume(hist_antes, "ANTES (bug)")

    print("Rodando COM a correcao (total_figuras_planejadas=7, plano "
          "tipico deste cenario sintetico)...\n")
    hist_depois = _rodar_e_medir(7)
    pct_depois = _resume(hist_depois, "DEPOIS (corrigido)")

    print(f"Resumo: parado numa unica fracao {pct_antes:.1f}% do tempo "
          f"antes da correcao, {pct_depois:.1f}% depois.")
