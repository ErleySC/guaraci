"""Escritores de resumo e model card (`guaraci.resultados_io`): disco que falha, model card ausente e formatos de borda.

Falha de disco e' injetada trocando `open` para o arquivo escolhido; o escritor precisa avisar e seguir, nunca derrubar a execucao.
"""
from __future__ import annotations

import builtins
import types

import pytest

import guaraci.resultados_io as rio
from guaraci.config import Config
from tests._resumo import canonico


def _sem_escrita(monkeypatch, nome):
    real = builtins.open

    def falso(caminho, modo="r", *a, **k):
        if str(caminho).endswith(nome) and ("a" in modo or "w" in modo):
            raise OSError("disco cheio")
        return real(caminho, modo, *a, **k)

    monkeypatch.setattr(builtins, "open", falso)


def _arquivo(tmp_path, nome, texto="cabecalho\n"):
    (tmp_path / nome).write_text(texto, encoding="utf-8")
    return str(tmp_path)


def test_auditoria_de_delineamento_nao_gravada_vira_aviso(tmp_path, monkeypatch, caplog):
    _sem_escrita(monkeypatch, "auditoria_delineamento.json")
    rio.save_design_audit(str(tmp_path), [{"nome": "n", "severidade": "ok", "mensagem": "m"}])
    assert "nao persistida" in caplog.text


def test_resumo_do_modelo_lista_dicionarios_listas_e_notas_vazias(tmp_path, monkeypatch):
    monkeypatch.setattr(rio, "_NOTAS_METODOLOGICAS", [("Titulo", "uma nota curta"), ("Vazia", "")])
    info = {"Accuracy": 0.9, "Grupos": {"a": 1, "b": 2}, "Itens": ["x", "y"], "auditoria_delineamento": [{"severidade": "critico", "nome": "n", "mensagem": "m"}]}
    rio.save_model_summary(str(tmp_path), info)
    texto = canonico((tmp_path / "resumo_modelo.txt"))
    for trecho in ("a: 1", "- x", "CRITICO", "uma nota curta", "[Vazia]"):
        assert trecho in texto, trecho


def test_resumo_de_regressao_cobre_pooled_sem_vies_perda_de_amostras_e_disco_cheio(tmp_path, monkeypatch, capsys):
    pasta = _arquivo(tmp_path, "resumo_modelo.txt")
    rio.append_regression_summary(pasta, pooled={"rmsep": 1.0}, holdout={"rmsep": 1.2, "bias": 0.1, "n_sem_modelo": 2})
    rio.append_regression_summary(pasta, pooled={"rmsep": 1.0, "bias": 0.0})
    rio.append_regression_summary(pasta)
    texto = canonico((tmp_path / "resumo_modelo.txt"))
    assert "Bias" in texto and "2 amostra(s) do holdout" in texto
    _sem_escrita(monkeypatch, "resumo_modelo.txt")
    rio.append_regression_summary(pasta, pooled={"rmsep": 1.0})
    assert "Nao foi possivel anexar regressao ao resumo" in capsys.readouterr().out


def test_resumo_do_mapa_de_calor_ignora_celula_nao_numerica_e_avisa_disco_cheio(tmp_path, monkeypatch, capsys):
    pasta = _arquivo(tmp_path, "resumo_modelo.txt")
    resultado = {"limiar_r2": 0.7, "n_falhas": 1, "n_ok": 1, "n_na": 0, "n_total": 2,
                 "matriz": {("e", "x"): 0.4, ("e", "y"): "sem dado", ("e", "z"): 0.9}}
    rio.append_heatmap_summary(pasta, resultado)
    assert "abaixo" in canonico((tmp_path / "resumo_modelo.txt"))
    _sem_escrita(monkeypatch, "resumo_modelo.txt")
    rio.append_heatmap_summary(pasta, resultado)
    assert "Nao foi possivel anexar heatmap" in capsys.readouterr().out


def _resumo_min(**extra):
    base = {"Total samples": 10, "Total variables": 5, "Total classes": 2, "Pre-processamento": "MSC", "Faixa espectral (cm-1)": "[1, 2]", "Tag": "-",
            "Group-aware (mae_id)": "sim", "N grupos mae_id": 3, "Imbalance ratio": 1.0, "Accuracy (CV)": 0.9, "Balanced accuracy": 0.9, "F1 (macro)": 0.9,
            "Cohen's kappa": 0.8, "R2X": 0.9, "R2Y": 0.9, "Q2": 0.8, "Permutation p-value": 0.01, "Hotelling T2 (95%)": 5.0, "Q-residual (95%)": 0.1,
            "NaN integrity": 0, "Integridade Inf": 0, "Constant variables": 0, "Exact duplicates": 0, "  Acc A": 0.9}
    base.update(extra)
    return base


def test_model_card_lista_dd_simca_por_classe_e_avisa_disco_cheio(tmp_path, monkeypatch, capsys):
    hw = {"ram_total_gb": 16.0, "cpu_fisicos": 4, "cpu_logicos": 8}
    rio.generate_model_card(str(tmp_path), Config(level="N2"), _resumo_min(**{"DD-SIMCA A sens/esp": "90% / 80%"}), hw, ["A"])
    assert "sensibilidade/especificidade por classe" in (tmp_path / "model_card.md").read_text(encoding="utf-8")
    _sem_escrita(monkeypatch, "model_card.md")
    rio.generate_model_card(str(tmp_path), Config(level="N2"), _resumo_min(), hw, ["A"])
    assert "Nao foi possivel gerar model_card.md" in capsys.readouterr().out


def test_addendum_de_regressao_do_model_card_sem_pooled_e_com_disco_cheio(tmp_path, monkeypatch, capsys):
    pasta = _arquivo(tmp_path, "model_card.md")
    rio.append_regression_model_card(pasta, pooled=None, tabela_especie=[{"especie": "A", "rmsep": 1.0, "lod": 1.0, "loq": 3.0}])
    rio.append_regression_model_card(pasta, pooled={"rmsep": "indisponivel"})
    _sem_escrita(monkeypatch, "model_card.md")
    rio.append_regression_model_card(pasta, pooled={"rmsep": 1.0})
    assert "Nao foi possivel anexar regressao ao model card" in capsys.readouterr().out


@pytest.mark.parametrize("funcao, argumento", [
    (rio.append_identification_model_card, {}),
    (rio.append_purity_model_card, {}),
    (rio.append_correcao_sinal_model_card, []),
    (rio.append_politica_pooled_local_model_card, []),
])
def test_addendos_sem_model_card_ou_sem_conteudo_nao_escrevem_nada(tmp_path, funcao, argumento):
    funcao(str(tmp_path), argumento)
    assert not (tmp_path / "model_card.md").exists()
    pasta = _arquivo(tmp_path, "model_card.md")
    funcao(pasta, argumento)
    if funcao in (rio.append_correcao_sinal_model_card, rio.append_politica_pooled_local_model_card):
        assert (tmp_path / "model_card.md").read_text(encoding="utf-8") == "cabecalho\n"


def test_addendos_de_identificacao_pureza_correcao_e_politica_avisam_disco_cheio(tmp_path, monkeypatch, capsys):
    pasta = _arquivo(tmp_path, "model_card.md")
    veredito = types.SimpleNamespace(metodo="emsc", veredito="aprovado", metrica="rmse", valor_sem=1.0, valor_com=0.8, tamanho_efeito_padronizado=0.5,
                                     p_valor=0.01, n_pares=10, poder_suficiente=True)
    decisao = types.SimpleNamespace(especie="A", recomendacao="local", n_amostras=9, n_minimo=6, veredito=veredito)
    ensemble = {("A", "x"): {"n_grupos": 4, "cobertura_status": types.SimpleNamespace(value="validado"), "alpha_alcancavel": None}}
    _sem_escrita(monkeypatch, "model_card.md")
    rio.append_identification_model_card(pasta, ensemble)
    rio.append_purity_model_card(pasta, {"A": {"n_grupos_calibracao": 4}})
    rio.append_correcao_sinal_model_card(pasta, [veredito])
    rio.append_politica_pooled_local_model_card(pasta, [decisao])
    rio.append_linearity_robustness_model_card(pasta, robustez={"x": types.SimpleNamespace(baseline=1.0, minimo=0.5, maximo=1.5, mediana=1.0,
                                                                                             variacao_absoluta=0.2, n_replicas=3)})
    saida = capsys.readouterr().out
    for trecho in ("identificacao ao model card", "pureza ao model card", "portao de correcao", "politica pooled", "linearidade/robustez"):
        assert trecho in saida, trecho


def test_formatacao_de_numeros_ausentes_ou_invalidos_da_linearidade(tmp_path):
    pasta = _arquivo(tmp_path, "model_card.md")
    lof = types.SimpleNamespace(computavel=True, linear=False, F=None, p_value="invalido", df_lof=1, df_pe=2, n_niveis=3, n_niveis_com_replica=2,
                                alpha=float("nan"))
    rio.append_linearity_robustness_model_card(pasta, linearidade=lof)
    texto = (tmp_path / "model_card.md").read_text(encoding="utf-8")
    assert "n/a" in texto and "FALTA DE AJUSTE" in texto


def test_resumo_do_mapa_de_calor_sem_matriz_so_traz_os_totais(tmp_path):
    pasta = _arquivo(tmp_path, "resumo_modelo.txt")
    rio.append_heatmap_summary(pasta, {"limiar_r2": 0.7, "n_falhas": 0, "n_ok": 0, "n_na": 0, "n_total": 0, "matriz": {}})
    assert "0/0 combinacoes abaixo" in canonico((tmp_path / "resumo_modelo.txt"))
