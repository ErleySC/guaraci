"""pytest configuration — root conftest.py.

Adds ./src to sys.path so the `guaraci` package is importable without an
editable install, and provides a session-scoped fixture that loads the
pipeline module once (avoids reloading the large module per test).
"""
import sys
import os

import pytest

# Torna o pacote `guaraci` (em ./src) importável nos testes sem `pip install -e .`.
_SRC = os.path.join(os.path.dirname(__file__), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# A3-04/Tk (vistoria 2026-10): nenhum teste abre janela. Sem isto o matplotlib escolhe sozinho o backend interativo (TkAgg) e,
# num Python sem Tk utilizavel (instalador sem Tcl/Tk, `init.tcl` ausente), a primeira figura de um teste levanta TclError --
# falha de AMBIENTE, nao do produto. Quem exporta MPLBACKEND por conta propria continua no controle.
os.environ.setdefault("MPLBACKEND", "Agg")

# O modo sintetico gera, por padrao, o cenario realista "nir"
# (`simulador_espectral`). Os limiares da suite foram calibrados no gerador
# antigo; os testes do simulador novo pedem o cenario explicitamente.
os.environ.setdefault("GUARACI_SINTETICO_CENARIO", "classico")

# Profile do Hypothesis (Bloco 13d, Frente 2): busca aleatoria SOZINHA (so'
# `max_examples`, sem `@example` fixo) mostrou-se pouco confiavel para achar
# bug conhecido (achado real, Passo 85 -- 80 exemplos passaram raso por cima
# de tres bugs reais de config.yaml sem achar nenhum). A correcao estrutural
# e' dupla: (1) SEMPRE fixar `@example` para o caso adversarial conhecido
# (nao depender so' do profile), (2) rodar com orcamento MAIOR na CI do que
# localmente -- mais exemplos aleatorios aumentam a chance de achar um caso
# novo que ninguem fixou ainda, sem deixar o ciclo local lento.
# `CI=true` e' setado automaticamente pelo GitHub Actions (nenhuma mudanca
# necessaria em test.yml); localmente cai no profile "dev" a nao ser que
# HYPOTHESIS_PROFILE seja setado explicitamente.
try:
    from hypothesis import settings as _hypothesis_settings

    _hypothesis_settings.register_profile("dev", max_examples=50, deadline=None)
    _hypothesis_settings.register_profile("ci", max_examples=300, deadline=None)
    _perfil_padrao = "ci" if os.environ.get("CI") else "dev"
    _hypothesis_settings.load_profile(
        os.environ.get("HYPOTHESIS_PROFILE", _perfil_padrao))
except ImportError:
    pass   # hypothesis e' dependencia de DEV (extra [dev]) -- ausente e' valido


#: Codigos de TESTE (arbitrarios; o produto nao traz nenhum): o mapa de classes comeca vazio e cada teste ve estes, neutros.
CODIGOS_DE_TESTE = {
    "ESA": "Classe A", "ESH": "Classe B", "ESG": "Classe C", "ESB": "Classe D", "ESC": "Classe E", "ESF": "Classe F", "ESE": "Classe G",
    "ESI": "Classe H", "ESD": "Classe I", "ESJ": "Classe J", "ESK": "Classe K", "EK": "Classe K", "ESM": "Classe L", "ESN": "Classe M",
    "ESO": "Classe N",
}


@pytest.fixture(autouse=True)
def _codigos_de_teste(monkeypatch):
    import guaraci.dados_io as _d
    for cod, nome in CODIGOS_DE_TESTE.items():
        monkeypatch.setitem(_d.CODIGO_ESPECIE, cod, nome)


@pytest.fixture(autouse=True)
def _ram_folgada_na_execucao(monkeypatch):
    """O auto-ajuste de extras por RAM (`auto_adjust_hardware_config`) desliga benchmark, Monte Carlo e SHAP quando a RAM LIVRE da maquina e'
    pequena: o resultado (e a cobertura) de qualquer teste que roda `pipeline.executar` passaria a depender de quanta memoria a maquina tem
    livre naquele instante. A leitura de memoria devolve, nos testes, uma maquina folgada (64 GB, 48 GB livres); os testes que precisam de
    outro valor trocam `psutil.virtual_memory` por conta propria (a troca deles vale por cima desta)."""
    try:
        import psutil
    except ImportError:
        return
    real = psutil.virtual_memory
    gb = 1024 ** 3
    monkeypatch.setattr(psutil, "virtual_memory", lambda *a, **k: real(*a, **k)._replace(total=64 * gb, available=48 * gb))


@pytest.fixture(autouse=True)
def _estado_global_de_figuras_isolado(monkeypatch):
    """Estado de MODULO que uma funcao muda e nenhum teste deveria herdar: a paleta ativa e a opacidade fixa (`paleta_cores`), o aviso
    de uma vez so' das figuras e os `rcParams` do matplotlib. Sob xdist a ordem dos testes muda a cada rodada; um teste que ativava a
    paleta do catalogo sem restaura-la fazia `test_cor_dentro_da_paleta_base` falhar so' quando caia depois dele no mesmo processo.
    Cada teste comeca no padrao e o que ele mudar volta ao fim."""
    import matplotlib
    import guaraci.figuras as _fg
    import guaraci.paleta_cores as _pc
    monkeypatch.setattr(_pc, "_ATIVA", None)
    monkeypatch.setattr(_pc, "_ALPHA_PONTOS", None)
    monkeypatch.setattr(_fg, "_AVISO_MOSTRAR_GRAFICOS_EMITIDO", False)
    monkeypatch.setattr(_fg, "ORGANIZAR_POR_GRUPO", False)   # `main()` da CLI liga; teste que chama main nao vaza para o proximo
    import guaraci.grupos_figuras as _gfg
    monkeypatch.setattr(_gfg, "ORGANIZAR", False)
    with matplotlib.rc_context():
        yield


@pytest.fixture(autouse=True)
def _cli_em_portugues(monkeypatch):
    """O CLI abre em ingles para quem nunca escolheu idioma; os testes de tela foram escritos sobre os textos em portugues.
    Cada teste comeca em PT (o idioma muda com `I` e volta ao fim); quem testa o ingles troca por conta propria."""
    import sys
    g = sys.modules.get("guaraci.guaraci")
    if g is not None:
        monkeypatch.setitem(g._STATE, "lang", "PT")
    # Relatorios, figuras e subcomandos seguem GUARACI_IDIOMA (senao a escolha gravada no menu, senao
    # ingles). Os testes foram escritos sobre os textos em portugues: fixa "pt" (o monkeypatch desfaz
    # ate' o que main()/_set_lang gravarem direto em os.environ). Quem testa o padrao apaga a variavel.
    monkeypatch.setenv("GUARACI_IDIOMA", "pt")


@pytest.fixture(autouse=True)
def _tecnica_selecionada_isolada():
    """A tela [2.4] (`_menu_technique`) grava a tecnica escolhida em `guaraci._TECNICA_SELECIONADA`, estado global do
    modulo que o cabecalho e o resumo leem. Teste que dirige a tela de verdade nao pode deixar a escolha para o teste
    seguinte (a faixa de identidade passava a dizer "UV-VIS" em vez de "FT-NIR"): cada teste volta ao que havia."""
    yield
    import sys
    g = sys.modules.get("guaraci.guaraci")
    if g is not None:
        g._TECNICA_SELECIONADA.clear()
        g._TECNICA_SELECIONADA.update({"key": "ft-nir", "nome": "FT-NIR"})       # o padrao do modulo


@pytest.fixture(autouse=True)
def _chave_assinatura_isolada(tmp_path_factory, monkeypatch):
    """Nenhum teste toca a chave de assinatura de modelos do usuario real
    (~/.guaraci/model_signing.key): cada teste usa uma chave temporaria."""
    monkeypatch.setenv(
        "GUARACI_MODEL_KEY_FILE",
        str(tmp_path_factory.mktemp("chave") / "model_signing.key"))


class _Roteiro:
    """Digitacao scriptada para as telas de terminal: cada `input()` consome uma resposta; sem respostas, vira EOF (a tela volta
    sozinha, como no uso real). Tudo que a tela escreve no console fica em `saida()`, sem codigos de cor."""

    def __init__(self, mod, fila_inicial):
        import io
        self.mod = mod
        self.fila = list(fila_inicial)
        self.prompts = []
        self.eofs = 0
        self.buffer = io.StringIO()

    def dizer(self, *respostas):
        self.fila.extend(respostas)
        return self

    def _input(self, prompt=""):
        self.prompts.append(str(prompt))
        if self.fila:
            return self.fila.pop(0)
        self.eofs += 1
        if self.eofs > 60:
            raise RuntimeError("a tela nao sai com EOF: roteiro esgotado")
        raise EOFError

    def saida(self):
        import re
        return re.sub(r"\x1b\[[0-9;]*m", "", self.buffer.getvalue())


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Roteiro de digitacao para `guaraci.guaraci`: estado do usuario em pasta temporaria, idioma PT, console capturado."""
    import guaraci.guaraci as g
    casa = tmp_path / "casa_cli"
    casa.mkdir()
    for nome, rel in (("_USER_DIR", ""), ("_CFG_PATH", "config.yaml"), ("_PERFIS_DIR", "perfis"), ("_LANG_FLAG", ".cli_wizard_done"),
                      ("_CODIGOS_PATH", "codigos_usuario.json"), ("_VISUAL_PATH", "visual_config.json"), ("_MODO_FLAG", ".cli_modo_usuario")):
        if hasattr(g, nome):
            monkeypatch.setattr(g, nome, casa / rel if rel else casa)
    monkeypatch.setitem(g._STATE, "lang", "PT")
    monkeypatch.setitem(g._STATE, "modo_usuario", "iniciante")
    r_chamadas = []
    monkeypatch.setattr(os, "system", lambda comando: r_chamadas.append(comando) or 0)    # `_cls` nao limpa o terminal de quem testa
    r = _Roteiro(g, [])
    r.comandos = r_chamadas
    monkeypatch.setattr("builtins.input", r._input)
    monkeypatch.setattr(g.console, "_file", r.buffer)
    return r


@pytest.fixture(scope="session")
def pq():
    """Session-scoped pipeline module — loaded once, reused across all tests."""
    import guaraci.pipeline as mod
    return mod


def achar_pastas_run(pasta_saida_raiz):
    """Localiza as pastas de EXECUCAO (folha) sob pasta_saida_raiz.

    Desde a auditoria jul/2026 (item 4), generate_output_name aninha a saida em
    pasta_saida_raiz/<Amostra>/<Modo>/<execucao>/ em vez de uma pasta direta
    em pasta_saida_raiz/<execucao>/. Esta funcao encontra as pastas-folha
    (identificadas pelo prefixo "PLSDA_OE_") em qualquer profundidade, para
    que os testes nao precisem hardcodar o numero de niveis.
    """
    import os as _os
    achadas = []
    for raiz, dirs, _arqs in _os.walk(str(pasta_saida_raiz)):
        for d in dirs:
            if d.startswith("PLSDA_OE_"):
                achadas.append(_os.path.join(raiz, d))
    return achadas


@pytest.fixture(scope="session")
def modelo_e_dados(pq, tmp_path_factory):
    """Roda executar() sintetico UMA vez e devolve (pkg, X_raw_algumas, wavenumbers): o pacote de modelo .joblib REAL que o
    pipeline grava em producao, mais espectros "novos" no mesmo eixo. Compartilhada por test_predicao e test_lacunas_predicao."""
    import joblib
    import numpy as np
    base = tmp_path_factory.mktemp("predicao")
    cfg = pq.Config(
        input_folder=str(base / "dados"),
        output_root_folder=str(base / "saida"),
        mode="sintetico", n_per_class=10, n_synthetic_points=60,
        wn_min=400.0, wn_max=4001.0,
        n_splits_cv=2, n_repeats_cv=1, n_permutations=5,
        n_permutations_wold=5, n_bootstrap_vip=3, n_bootstrap_bca=20,
        n_monte_carlo=3, max_lvs=5,
    )
    os.makedirs(cfg.input_folder, exist_ok=True)
    pq.executar(cfg)

    runs = achar_pastas_run(cfg.output_root_folder)
    assert runs, "executar() nao criou pasta de saida"
    cam_modelo = os.path.join(runs[0], pq.NOME_MODELOS, "modelo_plsda.joblib")
    assert os.path.isfile(cam_modelo), "modelo_plsda.joblib nao foi salvo"
    pkg = joblib.load(cam_modelo)

    # espectros "novos" sinteticos no MESMO eixo do treino (dados de teste, nao precisam ser fisicamente reais)
    wn = np.asarray(pkg["wavenumbers"], dtype=float)
    X_novos = np.random.default_rng(123).normal(loc=0.5, scale=0.05, size=(6, len(wn)))
    return pkg, X_novos, wn


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
    # Idioma padrao do produto e' o do menu (ingles, sem escolha gravada); a suite foi escrita sobre os
    # textos em portugues. Fixado aqui tambem (nao so' no autouse) porque fixtures de modulo/sessao
    # rodam antes dele.
    os.environ.setdefault("GUARACI_IDIOMA", "pt")
    # A3-05 (vistoria 2026-10): no Windows, basetemp longo (perfil do usuario, pasta funda) faz a guarda de MAX_PATH do pipeline
    # reprovar dezenas de testes sem defeito no produto. `tryfirst`: roda antes do plugin tmpdir ler `--basetemp`. So' no processo
    # principal: com -n, o xdist entrega a cada trabalhador `<basetemp>/popen-gwN`, ja' curto.
    if not hasattr(config, "workerinput"):
        from tests._basetemp_curto import criar_basetemp_curto, precisa_trocar, raiz_curta
        if precisa_trocar(config.option.basetemp):
            config.option.basetemp = criar_basetemp_curto(raiz_curta())


# m-26 (parecer de 2026-09-25): estes arquivos dependem de `git ls-files`/historico
# e falhavam em 34 testes fora de um checkout git (zip do GitHub, sdist). Fora de
# um checkout eles sao PULADOS com motivo explicito, nao reprovados. Entrada
# `arquivo.py` pula o arquivo inteiro; `arquivo.py::teste` pula so' esse teste
# (A3-03, vistoria 2026-10: os demais testes do arquivo nao leem git e seguem rodando).
_TESTES_QUE_EXIGEM_CHECKOUT_GIT = {
    "test_isolamento_datasets.py", "test_sem_identificador_real.py",
    "test_verificar_citacoes.py", "test_links_docs.py",
    "test_verificacao_publicada.py",
    "test_medir_vault.py::test_nota_gerada_em_commit_anterior_ao_relevante_e_defasada",
    "test_medir_vault.py::test_defasagem_por_carimbo_de_commit",
    "test_guarda_dados_de_terceiros.py::test_documentos_publicos_nao_citam_conjunto_de_terceiros",
    # leem `git ls-files`: numa arvore sem `.git` a lista vem vazia e a guarda passaria sem olhar nada (rodada de fechamento)
    "test_identidade_e_contato.py::test_nenhum_arquivo_versionado_tem_email_nem_padrao_generico",
    "test_identidade_e_contato.py::test_nenhum_arquivo_versionado_tem_termo_da_lista_privada",
    "test_identidade_e_contato.py::test_ferramental_de_reescrita_do_historico_nao_e_versionado",
    "test_identidade_e_contato.py::test_email_em_qualquer_arquivo_rastreado_de_texto_reprova",
    "test_identidade_e_contato.py::test_nenhum_arquivo_versionado_aponta_curriculo_em_bases_de_vinculo",
    "test_identidade_e_contato.py::test_url_do_repositorio_so_aparece_nos_lugares_listados",
    "test_identidade_e_contato.py::test_nenhum_arquivo_versionado_nomeia_o_servico_de_sincronizacao_fora_das_definicoes_do_auditor",
    "test_identidade_e_contato.py::test_nenhum_arquivo_versionado_cita_a_sigla_do_trabalho_academico",
    "test_identidade_e_contato.py::test_identificador_do_deposito_retirado_so_aparece_na_lapide_do_registro_de_referencias",
}


def exige_checkout_git(arquivo: str, nome_do_teste: str) -> bool:
    return arquivo in _TESTES_QUE_EXIGEM_CHECKOUT_GIT or f"{arquivo}::{nome_do_teste}" in _TESTES_QUE_EXIGEM_CHECKOUT_GIT


def pytest_collection_modifyitems(config, items):
    import os as _os
    import pytest as _pytest
    raiz = _os.path.dirname(_os.path.abspath(__file__))
    if _os.path.exists(_os.path.join(raiz, ".git")):
        return
    marcar_fora_do_git(items, _pytest.mark.skip(reason="requer um checkout git (sem .git aqui: zip/sdist)"))


def marcar_fora_do_git(items, pular) -> None:
    for item in items:
        if exige_checkout_git(os.path.basename(str(item.fspath)), getattr(item, "originalname", item.name)):
            item.add_marker(pular)


def _xgboost_carregavel() -> "tuple[bool, str]":
    """Tenta de fato IMPORTAR xgboost (nao so' checar se o pacote esta
    instalado) -- um pacote presente mas com biblioteca nativa bloqueada pelo
    sistema (achado real: `xgboost.dll` barrada pelo Controle de Aplicativo
    do Windows, WinError 4551) levanta `xgboost.core.XGBoostError`
    (subclasse de ValueError, NAO de ImportError), que `pytest.importorskip`
    nao sabe capturar (seu `exc_type` so' aceita ImportError/subclasses --
    ver `_pytest.outcomes.importorskip`). Retorna (ok, motivo_se_nao)."""
    try:
        import xgboost  # noqa: F401
        return True, ""
    except (ImportError, OSError, ValueError) as e:  # ValueError = XGBoostError
        if isinstance(e, ValueError) and type(e).__name__ != "XGBoostError":
            raise
        return False, f"{type(e).__name__}: {str(e).strip().splitlines()[0][:160]}"


_XGBOOST_OK, _XGBOOST_MOTIVO = _xgboost_carregavel()

#: Marcador para testes que exigem xgboost CARREGAVEL de verdade (nao so'
#: `pip show xgboost`). Uso: `@requer_xgboost` acima da funcao de teste.
requer_xgboost = pytest.mark.skipif(
    not _XGBOOST_OK,
    reason=("xgboost instalado mas nao carregavel neste ambiente "
            f"({_XGBOOST_MOTIVO}); pode ser o pacote ausente ou bloqueado "
            "pelo sistema (ex.: Controle de Aplicativo do Windows)."))


# Guarda de pulados (tests/_pulados.py + tests/pulados_esperados.json): todo
# skip precisa casar um motivo previsto e ficar dentro do teto; senao a sessao
# reprova. O resumo por motivo sai sempre no fim da suite.
_MOTIVOS_PULADOS: "list[str]" = []


def pytest_runtest_logreport(report):
    if report.skipped and not hasattr(report, "wasxfail"):
        lr = report.longrepr
        motivo = lr[2] if isinstance(lr, tuple) and len(lr) == 3 else str(lr or "")
        _MOTIVOS_PULADOS.append(motivo.removeprefix("Skipped: "))


def _e_trabalhador_xdist(config) -> bool:
    return hasattr(config, "workerinput")


def pytest_sessionstart(session):
    # Guarda do repositorio real (tests/_guarda_git.py): so' LEITURA; so' no
    # processo principal (com -n, os trabalhadores nao fotografam).
    if _e_trabalhador_xdist(session.config):
        return
    from tests._guarda_git import fotografar
    session.config._foto_git = fotografar(os.path.dirname(os.path.abspath(__file__)))


def pytest_sessionfinish(session, exitstatus):
    from tests._pulados import avaliar, carregar_esperados, escolher_ambiente
    raiz = os.path.dirname(os.path.abspath(__file__))
    if not _e_trabalhador_xdist(session.config):
        from tests._guarda_git import comparar, fotografar
        antes = getattr(session.config, "_foto_git", None)
        problemas_git = comparar(antes, fotografar(raiz) if antes else None)
        session.config._guarda_git = problemas_git
        if problemas_git and session.exitstatus == 0:
            session.exitstatus = 1
    if os.path.exists(os.path.join(raiz, ".git")):
        ambiente = escolher_ambiente(os.environ)
    else:
        ambiente = "sem-checkout-git"   # zip/sdist (m-26): ver pytest_collection_modifyitems
    contagem, problemas = avaliar(_MOTIVOS_PULADOS, carregar_esperados(ambiente))
    session.config._guarda_pulados = (ambiente, contagem, problemas)
    if problemas and session.exitstatus == 0:
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    ambiente, contagem, problemas = getattr(config, "_guarda_pulados", ("?", {}, []))
    tr = terminalreporter
    tr.section(f"pulados por motivo (ambiente: {ambiente})")
    for nome, n in sorted(contagem.items(), key=lambda kv: -kv[1]):
        tr.write_line(f"{n:4d}  {nome}")
    if not contagem:
        tr.write_line("   0  (nenhum teste pulado)")
    for p in problemas:
        tr.write_line(f"GUARDA DE PULADOS: {p}", red=True)
    for p in getattr(config, "_guarda_git", []):
        tr.write_line(f"GUARDA DO REPOSITORIO: {p}", red=True)
