"""Ajustes do CLI antes do lancamento: idioma padrao, textos em ingles, voltar um nivel e protecao contra travamento."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

import guaraci.guaraci as g
from tests._resumo import canonico

RAIZ = Path(__file__).resolve().parents[1]


def test_usuario_novo_abre_em_ingles():
    """Sem a marca de idioma gravada, o estado inicial e' ingles (o conftest troca para PT so' nos testes)."""
    codigo = f"import sys; sys.path.insert(0, {str(RAIZ / 'src')!r}); import guaraci.guaraci as g; print(g._STATE['lang'])"
    out = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip().splitlines()[-1] == "EN"


@pytest.mark.parametrize("pt, en", [
    ("imagem", "image"), ("multielementar", "multi-element"), ("sintetico", "synthetic"),
    ("qualitativo", "qualitative"), ("cinza", "grayscale"), ("3 a 15, impar", "3 to 15, odd"),
])
def test_valores_aparecem_em_ingles(monkeypatch, pt, en):
    monkeypatch.setitem(g._STATE, "lang", "EN")
    assert g._texto_para_exibir(pt) == en
    monkeypatch.setitem(g._STATE, "lang", "PT")
    assert g._texto_para_exibir(pt) == pt


def test_nomes_de_pasta_nao_sao_traduzidos(monkeypatch):
    """dados/resultados/classe sao nomes reais de pasta e coluna; traduzir mudaria o que o usuario precisa digitar."""
    monkeypatch.setitem(g._STATE, "lang", "EN")
    for nome in ("dados", "resultados", "classe", "Amostra"):
        assert g._texto_para_exibir(nome) == nome


def test_perfil_generico_em_ingles(monkeypatch):
    monkeypatch.setitem(g._STATE, "lang", "EN")
    rotulo = g._rotulo_opcao("matrix_profile", "generico")
    assert "generic" in rotulo and "nao especificada" not in rotulo


def test_mensagem_de_valor_em_ingles(monkeypatch):
    monkeypatch.setitem(g._STATE, "lang", "EN")
    msg = g._msg_valor("invalid literal for int() with base 10: 'x'")
    assert "x" in msg
    monkeypatch.setitem(g._STATE, "lang", "PT")
    assert "invalid literal" not in g._msg_valor("invalid literal for int() with base 10: 'x'")


def test_tela_de_passos_reabre_ate_voltar(monkeypatch):
    """Depois de rodar (ou de um erro), a tela de passos reabre; so' a tecla de voltar sai dela."""
    chamadas = []

    def falsa(tid, cfg):
        chamadas.append(tid)
        g._LEITURAS[0] += 1           # a tela leu o teclado (caminho, Enter...)
        g._VOLTOU[0] = len(chamadas) == 3

    monkeypatch.setattr(g, "_abrir_tela_uma_vez", falsa)
    g._abrir_tela("predicao", None)
    assert chamadas == ["predicao"] * 3


def test_tela_comum_abre_uma_vez(monkeypatch):
    chamadas = []
    monkeypatch.setattr(g, "_abrir_tela_uma_vez", lambda tid, cfg: chamadas.append(tid))
    g._abrir_tela(next(t for t in g._cat.TELAS if t not in g._TELAS_DE_PASSOS), None)
    assert len(chamadas) == 1


def test_tecla_de_voltar_marca_voltou(cli):
    g._VOLTOU[0] = False
    assert g._tecla_comum("0", "predicao", None) == "voltar"
    assert g._VOLTOU[0] is True


def test_sem_quickedit_nao_faz_nada_fora_do_windows():
    with g._sem_quickedit():
        pass


def test_diagnostico_de_travamento_grava_pilha(cli):
    caminho = g._gravar_diagnostico_travamento()
    assert caminho and Path(caminho).parent == g._USER_DIR
    texto = Path(caminho).read_text(encoding="utf-8")
    assert "Thread" in texto or "File" in texto


def test_doctor_em_ingles(cli, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(g._STATE, "lang", "EN")
    g._comando_doctor()
    saida = cli.saida()
    assert "environment check" in saida and "Report saved to" in saida
    assert "diagnostico de ambiente" not in saida


def test_formato_all_aceito():
    from guaraci import relatorio_cli
    assert relatorio_cli.formatos_pedidos("all") == relatorio_cli.formatos_pedidos("todos")


def test_tela_de_passos_sai_quando_a_entrada_acaba(cli, monkeypatch):
    """Com stdin fechado (EOF) a tela de passos rodava de novo com os padroes, sem fim."""
    chamadas = []

    def falsa(tid, cfg):
        chamadas.append(tid)
        g._input("Pasta: ")       # o roteiro vazio devolve EOF

    monkeypatch.setattr(g, "_abrir_tela_uma_vez", falsa)
    g._abrir_tela("hsi", None)
    assert chamadas == ["hsi"]


def test_tela_de_passos_que_nao_pede_nada_nao_reabre(monkeypatch):
    chamadas = []
    monkeypatch.setattr(g, "_abrir_tela_uma_vez", lambda tid, cfg: chamadas.append(tid))
    g._abrir_tela("comparar", None)
    assert chamadas == ["comparar"]


def test_tela_volta_ao_terminal_depois_da_analise_mesmo_se_a_analise_comeca_antes_do_painel(cli, monkeypatch):
    """Travamento relatado: a analise trocava sys.stdout pelo buffer do log antes de o painel (Live) entrar; o Live
    guardava o BUFFER e o devolvia ao sair. Painel de erro e "Enter para continuar" iam para o buffer: tela parada."""
    import io
    import threading
    import guaraci.dados_io as dio

    tela = io.StringIO()
    monkeypatch.setattr(sys, "stdout", tela)
    monkeypatch.setattr(g.console, "_file", None)          # como na CLI de verdade: o console segue sys.stdout
    monkeypatch.setattr(g, "_print_checklist", lambda cfg: True)
    monkeypatch.setattr(dio, "garantia_agrupamento_prevista", lambda cfg: "high")
    monkeypatch.setattr(g, "_bytes_da_entrada", lambda cfg: None)

    analise_entrou, painel_entrou = threading.Event(), threading.Event()

    def analise(cfg):
        analise_entrou.set()
        painel_entrou.wait(30)
        print("[AVISO] linha-do-log-da-analise")
        raise ValueError("Validacao cruzada impossivel: teste")
    monkeypatch.setattr(g, "_executar", analise)

    class LiveAtrasado(g.Live):                            # garante a ordem ruim: a analise entra primeiro
        def __enter__(self):
            analise_entrou.wait(30)
            r = super().__enter__()
            painel_entrou.set()
            return r
    monkeypatch.setattr(g, "Live", LiveAtrasado)
    cli.dizer("", "s", "")
    g._rodar_pipeline(g.Config())
    assert sys.stdout is tela
    saida = tela.getvalue()
    assert "Validacao cruzada impossivel" in saida and g._t("continuar") in saida
    assert "[AVISO] linha-do-log-da-analise" not in saida   # o log vai para o painel, nao solto na tela


def test_classe_com_uma_unidade_oferece_retirar_e_rodar(cli, monkeypatch):
    import types
    import guaraci.dados_io as dio
    monkeypatch.setattr(g, "_print_checklist", lambda cfg: True)
    monkeypatch.setattr(dio, "garantia_agrupamento_prevista", lambda cfg: "high")
    dados = types.SimpleNamespace(unidades_por_classe={"A": 10, "B": 1, "C": 1})
    monkeypatch.setattr(g, "_preflight", lambda cfg, lang: ([], dados))
    rodou = []
    monkeypatch.setattr(g, "_executar", lambda cfg: rodou.append(tuple(cfg.exclude_classes)))
    cfg = g.Config()
    cli.dizer("", "s", "s", "")                              # tag, rodar?, retirar e rodar?, Enter
    g._rodar_pipeline(cfg)
    assert rodou == [("B", "C")]
    cli.dizer("", "s", "n", "")                                   # recusar cancela: com elas a validacao nao roda
    rodou.clear()
    g._rodar_pipeline(g.Config())
    assert rodou == []


def test_validacao_impossivel_depois_do_holdout_oferece_retirar_e_roda_de_novo(cli, monkeypatch):
    """O pre-voo conta unidades antes do holdout; se a falta de grupos so' aparece depois, a falha oferece retirar
    as classes e roda de novo sem repetir as perguntas."""
    import guaraci.dados_io as dio
    from guaraci.pipeline import ValidacaoCruzadaImpossivel
    monkeypatch.setattr(g, "_print_checklist", lambda cfg: True)
    monkeypatch.setattr(dio, "garantia_agrupamento_prevista", lambda cfg: "high")
    monkeypatch.setattr(g, "_preflight", lambda cfg, lang: ([], None))
    rodadas = []

    def analise(cfg):
        rodadas.append(tuple(cfg.exclude_classes))
        if "B" not in cfg.exclude_classes:
            raise ValidacaoCruzadaImpossivel(["B"], "Validacao cruzada impossivel: B")
    monkeypatch.setattr(g, "_executar", analise)
    cli.dizer("", "s", "s", "")                              # tag, rodar?, retirar B e rodar de novo?, Enter
    g._rodar_pipeline(g.Config())
    assert rodadas == [(), ("B",)]
    assert g._t("exec_concluido").upper() in cli.saida()


def _csv_uma_amostra_em_d(tmp_path):
    import numpy as np
    wn = np.linspace(4000, 4110, 60)
    linhas = ["classe," + ",".join(f"{w:.2f}" for w in wn)]
    rng = np.random.default_rng(0)
    for c, n in (("A", 8), ("B", 8), ("C", 8), ("D", 1)):
        for _ in range(n):
            linhas.append(c + "," + ",".join(f"{v:.4f}" for v in rng.random(60)))
    arq = tmp_path / "d1.csv"
    arq.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return str(arq)


def test_exploratorio_roda_com_classe_de_uma_amostra_e_sem_holdout(tmp_path):
    """O exploratorio so' gera PCA/HCA: classe com uma unidade e holdout nao podem tirar amostras dos graficos."""
    import guaraci.pipeline as pq
    cfg = pq.Config(mode="csv", csv_file=_csv_uma_amostra_em_d(tmp_path), class_column="classe", show_plots=False,
                    output_root_folder=str(tmp_path / "o"), n_permutations=3, wn_min=4000, wn_max=4110,
                    frac_holdout=0.25, objective="exploratorio")
    pq.executar(cfg)
    resumo = canonico(next((tmp_path / "o").rglob("resumo_modelo.txt")))   # chaves fixas
    assert re.search(r"Total samples\s+:\s+25\b", resumo) and re.search(r"Total classes\s+:\s+4\b", resumo)
    assert "nao valida o modelo" in resumo


def test_classificacao_com_classe_de_uma_amostra_continua_recusada(tmp_path):
    import guaraci.pipeline as pq
    from guaraci.pipeline import ValidacaoCruzadaImpossivel
    cfg = pq.Config(mode="csv", csv_file=_csv_uma_amostra_em_d(tmp_path), class_column="classe", show_plots=False,
                    output_root_folder=str(tmp_path / "o"), n_permutations=3, wn_min=4000, wn_max=4110,
                    frac_holdout=0.0, objective="classificacao")
    with pytest.raises(ValidacaoCruzadaImpossivel) as e:
        pq.executar(cfg)
    assert e.value.classes == ["D"]


def test_exploratorio_nao_acusa_classe_de_uma_unidade():
    import types
    from guaraci import assistente_regras as ar
    dados = types.SimpleNamespace(unidades_por_classe={"A": 5, "D": 1}, n_unidades=6, leituras_por_unidade_max=1)

    def ids(obj):
        return {a.id for a in ar.avaliar(g.Config(objective=obj, frac_holdout=0.25), dados)}
    assert "classe_com_uma_unidade" in ids("classificacao")
    assert "classe_com_uma_unidade" not in ids("exploratorio") and "holdout_pequeno" not in ids("exploratorio")
