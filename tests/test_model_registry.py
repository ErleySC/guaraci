"""Testes de guaraci.model_registry (item 20: registry de modelos do
Auto-Benchmark / Monte Carlo CV).

Regressão-chave: antes da extração, benchmark_classifiers() e
monte_carlo_cv() tinham a MESMA lista de classificadores hardcoded duas
vezes, e haviam divergido — o Gradient Boosting do Monte Carlo CV não tinha
subsample=0.8 (o do benchmark tinha), apesar da docstring afirmar hiper-
parâmetros idênticos. O teste abaixo trava essa divergência.
"""
import pytest

from guaraci.model_registry import build_benchmark_list, nomes_modelos_benchmark


def test_pls_da_sempre_presente_mesmo_sem_opcionais(pq):
    cfg = pq.Config()
    lista = build_benchmark_list(n_opt=5, cfg=cfg, incluir_opcionais=False)
    nomes = [n for n, _ in lista]
    assert nomes == ["PLS-DA"]


def test_lista_completa_inclui_todos_os_modelos_core(pq):
    cfg = pq.Config()
    lista = build_benchmark_list(n_opt=5, cfg=cfg, incluir_opcionais=True)
    nomes = [n for n, _ in lista]
    # SVM RBF/Random Forest/Grad. Boost. nao tem dependencia opcional --
    # sempre presentes; XGBoost so' se o pacote estiver instalado.
    assert "PLS-DA" in nomes
    assert "SVM RBF" in nomes
    assert "Random Forest" in nomes
    assert "Grad. Boost." in nomes


def test_gradient_boosting_usa_subsample_08_regressao(pq):
    """Regressao: GB do Monte Carlo CV nao tinha subsample=0.8 (drift vs
    benchmark). Agora vem de UMA fonte so', entao os dois caminhos coincidem."""
    cfg = pq.Config()
    lista = build_benchmark_list(n_opt=5, cfg=cfg, incluir_opcionais=True)
    gb = dict(lista)["Grad. Boost."]
    assert gb.subsample == 0.8


def test_construtores_respeitam_seed_e_n_opt(pq):
    cfg = pq.Config()
    cfg.seed = 123
    lista = build_benchmark_list(n_opt=7, cfg=cfg, incluir_opcionais=True)
    d = dict(lista)
    assert d["PLS-DA"].n_components == 7
    assert d["Random Forest"].random_state == 123
    assert d["SVM RBF"].random_state == 123


def test_nomes_modelos_benchmark_sem_instanciar():
    assert nomes_modelos_benchmark(incluir_opcionais=False) == ("PLS-DA",)
    completo = nomes_modelos_benchmark(incluir_opcionais=True)
    assert "PLS-DA" in completo and "SVM RBF" in completo


def test_modelo_opcional_ausente_e_pulado_silenciosamente(monkeypatch, pq):
    """Se um pacote opcional (ex.: xgboost) nao estiver instalado, o
    registry pula esse modelo sem lancar excecao (comportamento historico)."""
    import guaraci.model_registry as mr

    def _falha_import(n_opt, cfg):
        raise ImportError("pacote nao instalado (simulado)")

    monkeypatch.setattr(mr, "_construir_xgboost", _falha_import)
    # Reconstroi o registro apontando para o construtor monkeypatched.
    registro_original = mr._REGISTRO
    mr._REGISTRO = [
        (nome, (_falha_import if nome == "XGBoost" else construtor), obrig)
        for nome, construtor, obrig in registro_original
    ]
    try:
        cfg = pq.Config()
        lista = build_benchmark_list(n_opt=5, cfg=cfg, incluir_opcionais=True)
        assert "XGBoost" not in dict(lista)
    finally:
        mr._REGISTRO = registro_original


def test_modelo_opcional_bloqueado_pelo_sistema_e_pulado_com_motivo_acionavel(monkeypatch, pq):
    """Achado real (revisao cega + instrucao de 2026-10-02): um pacote opcional
    PRESENTE mas com a biblioteca nativa BLOQUEADA pelo sistema (ex.: xgboost.dll
    barrada pelo Controle de Aplicativo do Windows, WinError 4551) levanta uma
    excecao do PROPRIO pacote (ex.: XGBoostError, que herda de ValueError) --
    NAO ImportError. So' capturar ImportError derrubava o benchmark inteiro.
    Aqui simulamos exatamente esse caso (nao so' 'pacote ausente')."""
    import guaraci.model_registry as mr

    # mesmo NOME da classe real (xgboost.core.XGBoostError): o helper so'
    # trata ValueError quando o nome e' esse (a classe nao e' importavel
    # quando a DLL falha).
    XGBoostError = type("XGBoostError", (ValueError,), {})

    def _falha_bloqueio(n_opt, cfg):
        raise XGBoostError(
            "[WinError 4551] Uma politica de Controle de Aplicativo bloqueou este arquivo")

    registro_original = mr._REGISTRO
    mr._REGISTRO = [
        (nome, (_falha_bloqueio if nome == "XGBoost" else construtor), obrig)
        for nome, construtor, obrig in registro_original
    ]
    try:
        cfg = pq.Config()
        # (1) nao crasha -- as demais tecnicas continuam disponiveis.
        lista = mr.build_benchmark_list(n_opt=5, cfg=cfg, incluir_opcionais=True)
        nomes = [n for n, _ in lista]
        assert "XGBoost" not in nomes
        assert {"PLS-DA", "SVM RBF", "Random Forest", "Grad. Boost."} <= set(nomes)

        # (2) o motivo e' acessivel (aditivo, nao muda build_benchmark_list)
        # e contem o que falhou, uma causa provavel e uma ACAO sugerida --
        # nunca so' "ImportError" generico.
        motivos = mr.tecnicas_indisponiveis(n_opt=5, cfg=cfg)
        assert "XGBoost" in motivos
        motivo = motivos["XGBoost"].lower()
        # texto que SO' existe no ramo Windows (nao e' eco da mensagem de erro
        # crua, que vem so' depois de "detalhe tecnico:") -- distingue de
        # verdade o ramo Windows dos ramos ImportError/generico (mata AddNot
        # na condicao inteira, nao so' confirma por coincidencia do eco).
        causa_e_acao = motivo.split("detalhe tecnico:")[0]
        assert "controle de aplicativo" in causa_e_acao
        assert "libere o arquivo" in causa_e_acao

        # (3) o app continua funcionando -- a falha de UMA tecnica opcional
        # nao impede nada do resto (ponto de entrada real, nao so' a funcao).
        import subprocess
        import sys
        r = subprocess.run([sys.executable, "-m", "guaraci", "--help"],
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 0 and "guaraci" in r.stdout.lower()
    finally:
        mr._REGISTRO = registro_original


def test_motivo_so_marca_windows_com_as_DUAS_pistas_separadas():
    """Mutante sobrevivente real (cosmic-ray, 2026-10-02): com as DUAS pistas
    ('WinError 4551' e 'Controle de Aplicativo') sempre juntas no mesmo texto
    de erro, `or` e `and` davam o mesmo resultado -- o teste anterior nao
    distinguia os dois. Aqui cada pista aparece SOZINHA, em mensagens
    separadas; as duas tem que cair no ramo Windows (e' OR, nao AND)."""
    import guaraci.model_registry as mr

    for msg in ("[WinError 4551] bloqueio generico sem o nome da politica",
               "Controle de Aplicativo vetou isto, sem o codigo de erro"):
        motivo = mr._motivo_indisponivel("XGBoost", ValueError(msg))
        causa_e_acao = motivo.lower().split("detalhe tecnico:")[0]
        assert "controle de aplicativo" in causa_e_acao, (msg, motivo)


def test_build_benchmark_list_avisa_no_log_quando_pula_uma_tecnica(pq, caplog):
    """Mutante sobrevivente real: `elif motivo is not None` virou
    `elif motivo is None` sem nenhum teste notar -- so' o RESULTADO (lista
    sem a tecnica) era checado, nunca o AVISO em si. Agora captura o log de
    verdade."""
    import logging

    import guaraci.model_registry as mr

    def _falha(n_opt, cfg):
        raise ImportError("pacote nao instalado (simulado)")

    registro_original = mr._REGISTRO
    mr._REGISTRO = [
        (nome, (_falha if nome == "XGBoost" else construtor), obrig)
        for nome, construtor, obrig in registro_original
    ]
    try:
        cfg = pq.Config()
        with caplog.at_level(logging.WARNING, logger="guaraci.model_registry"):
            mr.build_benchmark_list(n_opt=5, cfg=cfg, incluir_opcionais=True)
        assert any("XGBoost" in r.message and "indisponivel" in r.message
                  for r in caplog.records), caplog.text
    finally:
        mr._REGISTRO = registro_original


def test_build_benchmark_list_inclui_obrigatorio_em_qualquer_posicao_do_registro(pq):
    """Mutante sobrevivente real: `continue` virou `break` no loop principal
    sem nenhum teste notar -- so' e' alcancavel com `incluir_opcionais=False`
    (e' a guarda `if not obrigatorio and not incluir_opcionais: continue`).
    Com PLS-DA (obrigatorio) DEPOIS de um opcional no registro, `break`
    pararia o loop antes de chegar nele; `continue` so' pula o opcional e
    segue. Ordem nao deveria importar -- trava essa garantia de verdade."""
    import guaraci.model_registry as mr

    registro_original = mr._REGISTRO
    sem_plsda = [e for e in registro_original if e[0] != "PLS-DA"]
    plsda = next(e for e in registro_original if e[0] == "PLS-DA")
    # PLS-DA (obrigatorio) deslocado para DEPOIS de um opcional -- com `break`
    # no lugar de `continue`, o loop pararia no opcional anterior e PLS-DA
    # JAMAIS seria construido.
    mr._REGISTRO = sem_plsda[:1] + [plsda] + sem_plsda[1:]
    try:
        cfg = pq.Config()
        lista = mr.build_benchmark_list(n_opt=5, cfg=cfg, incluir_opcionais=False)
        assert "PLS-DA" in dict(lista)
    finally:
        mr._REGISTRO = registro_original


def test_nenhum_modulo_importa_xgboost_no_topo():
    """(c) do checklist: import de pacote opcional no TOPO do arquivo derrubaria
    a importacao do MODULO inteiro (nao so' a tecnica) se o pacote estiver
    ausente/bloqueado -- varredura por AST confirma que todo import de xgboost
    em src/guaraci/ esta DENTRO de uma funcao (tardio), nunca no nivel de
    modulo."""
    import ast
    import glob
    import os

    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    achados = []
    # numba (via shap/glasbey) tem DLL nativa que o Controle de Aplicativo do
    # Windows ja' bloqueou nesta maquina (2026-10-02) -- mesmo padrao do xgboost.
    opcionais_nativos = ("xgboost", "numba", "shap", "glasbey", "colorcet")
    for caminho in glob.glob(os.path.join(raiz, "src", "guaraci", "**", "*.py"),
                             recursive=True):
        arvore = ast.parse(open(caminho, encoding="utf-8").read())
        for no in arvore.body:  # so' o nivel de MODULO (topo), nao recursivo
            alvos = []
            if isinstance(no, ast.Import):
                alvos = [a.name for a in no.names]
            elif isinstance(no, ast.ImportFrom) and no.module:
                alvos = [no.module]
            if any(a.split(".")[0] in opcionais_nativos for a in alvos):
                achados.append(f"{os.path.basename(caminho)}:{no.lineno}")
    assert not achados, f"import de pacote opcional nativo no TOPO do modulo: {achados}"


# -- Parte D (2026-10-02): so' falha de CARREGAMENTO vira "indisponivel" ----

def _registro_com_xgboost_falhando(mr, erro):
    def _falha(n_opt, cfg):
        raise erro
    return [(nome, (_falha if nome == "XGBoost" else construtor), obrig)
            for nome, construtor, obrig in mr._REGISTRO]


def test_valueerror_de_outra_origem_propaga_e_nao_vira_indisponivel(pq, monkeypatch):
    """`ValueError` qualquer (ex.: parametro invalido) e' bug real: tem que
    PROPAGAR, nao ser mascarado como "XGBoost indisponivel"."""
    import guaraci.model_registry as mr
    monkeypatch.setattr(mr, "_REGISTRO",
                        _registro_com_xgboost_falhando(mr, ValueError("outro erro")))
    with pytest.raises(ValueError, match="outro erro"):
        mr.build_benchmark_list(n_opt=5, cfg=pq.Config(), incluir_opcionais=True)
    with pytest.raises(ValueError, match="outro erro"):
        mr.tecnicas_indisponiveis(n_opt=5, cfg=pq.Config())


def test_so_o_nome_exato_XGBoostError_e_tratado():
    """Nome PARECIDO nao basta (mutante real: `==` -> `>=` sobrevivia,
    porque "ValueError" < "XGBoostError" na ordem de strings)."""
    import guaraci.model_registry as mr
    parecido = type("XGBoostErrorDeOutraCoisa", (ValueError,), {})("x")
    exato = type("XGBoostError", (ValueError,), {})("x")
    assert mr._e_dependencia_indisponivel(exato)
    assert not mr._e_dependencia_indisponivel(parecido)
    assert not mr._e_dependencia_indisponivel(ValueError("x"))


@pytest.mark.parametrize("erro, trecho_causa, trecho_acao", [
    (type("XGBoostError", (ValueError,), {})(
        "Failed to load libxgboost: [WinError 4551] politica bloqueou a DLL"),
     "controle de aplicativo", "libere o arquivo"),
    (OSError(4551, "[WinError 4551] Controle de Aplicativo bloqueou xgboost.dll"),
     "controle de aplicativo", "libere o arquivo"),
    (OSError("libxgboost.so: cannot open shared object file"),
     "falha ao carregar o pacote (oserror)", "as demais tecnicas continuam disponiveis"),
    (ImportError("No module named 'xgboost'"),
     "nao esta instalado", "pip install -e '.[benchmark]'"),
])
def test_falha_de_carregamento_e_tratada_com_causa_e_acao_em_pt(
        pq, monkeypatch, erro, trecho_causa, trecho_acao):
    import guaraci.model_registry as mr
    monkeypatch.setattr(mr, "_REGISTRO", _registro_com_xgboost_falhando(mr, erro))
    nomes = [n for n, _ in mr.build_benchmark_list(n_opt=5, cfg=pq.Config())]
    assert "XGBoost" not in nomes and "PLS-DA" in nomes
    motivo = mr.tecnicas_indisponiveis(n_opt=5, cfg=pq.Config())["XGBoost"].lower()
    causa_e_acao = motivo.split("detalhe tecnico:")[0]
    assert trecho_causa in causa_e_acao
    assert trecho_acao.lower() in causa_e_acao


def test_truncamento_do_detalhe_nao_corta_a_acao_sugerida():
    """O `[:200]` so' corta o DETALHE tecnico, que vem depois da causa e da
    acao: uma mensagem de erro enorme nao pode engolir a acao."""
    import guaraci.model_registry as mr
    erro = OSError("[WinError 4551] " + "x" * 5000)
    motivo = mr._motivo_indisponivel("XGBoost", erro)
    causa_e_acao, detalhe = motivo.split("Detalhe tecnico: ")
    assert "libere o arquivo" in causa_e_acao
    assert len(detalhe) <= 200
