# -*- coding: utf-8 -*-
"""web_seguranca.py -- Politica de seguranca da interface web (Streamlit).

Logica PURA (sem `streamlit`), testavel isoladamente. Origem: parecer de
revisao por pares de 2026-09-25 (M-01, m-35, m-37): o app web era inseguro
para deploy em rede porque (i) carregava `.joblib` (pickle = execucao de
codigo) por padrao, (ii) aceitava caminhos livres do servidor em varios
campos, (iii) compartilhava um `config.yaml` entre visitantes e (iv) nunca
apagava os uploads.

MODELO DE AMEACA
-----------------
Dois modos, escolhidos por variavel de ambiente:

* **Modo servidor (PADRAO)** -- qualquer visitante e' nao confiavel.
  Nenhum campo de caminho do servidor; so' upload. Todo arquivo (upload e
  resultados) vive numa pasta de SESSAO isolada (`raiz_sessoes()/<id>`),
  apagada ao fim da sessao e varrida por idade. Carga de modelo so' em JSON
  portatil (`model_export`), que nao executa codigo. Nao le nem grava
  `config.yaml` compartilhado: a configuracao e' estado da sessao.
* **Modo local** -- `GUARACI_LOCAL_MODE=1`. Uso single-user na maquina do
  pesquisador (o que `guaraci web` liga): caminhos
  livres, `.joblib` (com manifesto assinado, ver `predicao.load_model`) e
  `config.yaml` na pasta do app, como antes.

Nada aqui e' "seguranca por obscuridade": o visitante nunca ve o caminho
absoluto da sessao nem tracebacks -- so' uma mensagem amigavel e um id de
erro que casa com a linha do log do servidor (`registrar_erro`).
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
import traceback
import uuid
import weakref
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger("guaraci.web")

VAR_MODO_LOCAL = "GUARACI_LOCAL_MODE"
VAR_BLOQUEIO_LEGADO = "GUARACI_DISABLE_MODEL_UPLOAD"
VAR_RAIZ_SESSOES = "GUARACI_SESSION_DIR"
_VERDADEIRO = ("1", "true", "yes", "on", "sim")
_RE_SESSAO = re.compile(r"^[0-9a-f]{16}$")
IDADE_MAXIMA_HORAS = 6.0


def _flag(nome: str) -> bool:
    return os.environ.get(nome, "").strip().lower() in _VERDADEIRO


def modo_local() -> bool:
    """True so' com `GUARACI_LOCAL_MODE` explicito (uso single-user)."""
    return _flag(VAR_MODO_LOCAL)


def modo_servidor() -> bool:
    return not modo_local()


def pickle_permitido() -> bool:
    """Carga de `.joblib` (pickle) na web: so' em modo local, e nunca se o
    operador ainda mantiver a flag legada `GUARACI_DISABLE_MODEL_UPLOAD`."""
    return modo_local() and not _flag(VAR_BLOQUEIO_LEGADO)


# ── Pastas de sessao ─────────────────────────────────────────────────────
def raiz_sessoes() -> Path:
    env = os.environ.get(VAR_RAIZ_SESSOES, "").strip()
    return Path(env) if env else Path(tempfile.gettempdir()) / "guaraci_sessoes"


def nova_sessao() -> str:
    """Id aleatorio, imprevisivel, curto (caminhos curtos ajudam no MAX_PATH
    do Windows). Nunca derivado de dado do visitante."""
    return uuid.uuid4().hex[:16]


def _validar_id(sid: str) -> str:
    if not isinstance(sid, str) or not _RE_SESSAO.match(sid):
        raise ValueError("id de sessao invalido")
    return sid


def pasta_sessao(sid: str, sub: Optional[str] = None, *, criar: bool = True) -> Path:
    """`raiz_sessoes()/<sid>[/<sub>]`. `sub` e' um nome fixo do codigo
    (uploads, resultados), nunca dado do visitante."""
    p = raiz_sessoes() / _validar_id(sid)
    if sub:
        p = p / sub
    if criar:
        p.mkdir(parents=True, exist_ok=True)
    return p


def dentro_da_sessao(caminho: str, sid: str) -> Path:
    """Resolve `caminho` e garante que ele esta DENTRO da pasta da sessao
    (bloqueia `..`, caminho absoluto de fora e links simbolicos que
    escapem). Levanta `ValueError` caso contrario."""
    raiz = pasta_sessao(sid).resolve()
    alvo = Path(caminho).resolve()
    try:
        alvo.relative_to(raiz)
    except ValueError:
        raise ValueError("caminho fora da pasta da sessao") from None
    return alvo


class UploadRecusado(ValueError):
    """Upload recusado por nome ou tamanho. A mensagem e' segura para o
    visitante: nunca traz caminho do servidor."""


#: Extensoes aceitas nos uploaders de tabela do app (csv/txt/tsv).
EXTENSOES_UPLOAD = (".csv", ".txt", ".tsv")
#: Comprimento maximo do nome gravado (sem a extensao). Nomes longos geravam
#: erro do sistema de arquivos com o caminho absoluto da sessao (A1-03).
MAX_NOME_UPLOAD = 80
_NOMES_RESERVADOS_WINDOWS = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)})
_RE_CARACTER_INSEGURO = re.compile(r"[^A-Za-z0-9._\-]+")

# Tetos de recursos do MODO SERVIDOR (A1-04): um visitante anonimo nao pode
# saturar a instancia publica. Nao valem para a CLI nem para o modo local.
TETO_PERMUTACOES_SERVIDOR = 1000
TETO_BOOTSTRAP_SERVIDOR = 2000
MAX_LINHAS_TABELA_SERVIDOR = 100_000
MAX_COLUNAS_TABELA_SERVIDOR = 50_000
MAX_CELULAS_TABELA_SERVIDOR = 20_000_000


def nome_seguro_upload(nome_original: str) -> str:
    """Nome GERADO pelo codigo a partir do nome do visitante: so' o basename,
    caracteres fora de `[A-Za-z0-9._-]` viram `_`, comprimento limitado e
    extensao obrigatoriamente em `EXTENSOES_UPLOAD`. O radical do nome e'
    mantido porque nomeia o conjunto de dados nos resultados
    (`pipeline._nome_dataset`). Recusa vazio, `.`, `..` e nomes reservados
    do Windows com `UploadRecusado` (mensagem sem caminho)."""
    base = Path(str(nome_original).replace("\\", "/")).name.strip()
    if base in ("", ".", ".."):
        raise UploadRecusado("nome de arquivo invalido")
    radical, ext = os.path.splitext(base)
    ext = ext.lower()
    if ext not in EXTENSOES_UPLOAD:
        raise UploadRecusado(
            "extensao nao aceita (use " + ", ".join(EXTENSOES_UPLOAD) + ")")
    radical = _RE_CARACTER_INSEGURO.sub("_", radical).strip("._")[:MAX_NOME_UPLOAD]
    if not radical:
        raise UploadRecusado("nome de arquivo invalido")
    if radical.split(".")[0].lower() in _NOMES_RESERVADOS_WINDOWS:
        raise UploadRecusado("nome de arquivo reservado pelo sistema")
    return radical + ext


def conferir_dimensoes_tabela(dados: bytes) -> Tuple[int, int]:
    """Conta linhas e colunas de uma tabela de texto SEM fazer o parse
    (barato: conta quebras de linha e separadores da primeira linha) e
    recusa o que passar dos tetos do modo servidor. Retorna (linhas, colunas)."""
    linhas = dados.count(b"\n") + (0 if dados.endswith(b"\n") or not dados else 1)
    primeira = dados.split(b"\n", 1)[0]
    colunas = 1 + max(primeira.count(s) for s in (b",", b";", b"\t"))
    if (linhas > MAX_LINHAS_TABELA_SERVIDOR or colunas > MAX_COLUNAS_TABELA_SERVIDOR
            or linhas * colunas > MAX_CELULAS_TABELA_SERVIDOR):
        raise UploadRecusado(
            f"tabela grande demais para o servidor publico ({linhas} linhas x "
            f"{colunas} colunas; maximo {MAX_LINHAS_TABELA_SERVIDOR} linhas, "
            f"{MAX_COLUNAS_TABELA_SERVIDOR} colunas, "
            f"{MAX_CELULAS_TABELA_SERVIDOR} celulas). Rode localmente com "
            "`guaraci web` para tabelas maiores")
    return linhas, colunas


def salvar_upload(sid: str, nome_original: str, dados: bytes,
                  sub: str = "uploads") -> Path:
    """Grava um upload na pasta da sessao com nome gerado por
    `nome_seguro_upload` (sem path traversal, sem nome reservado, sem
    comprimento que estoure o sistema de arquivos). No modo servidor a tabela
    tambem passa pelos tetos de `conferir_dimensoes_tabela` antes de gravar."""
    nome = nome_seguro_upload(nome_original)
    if modo_servidor():
        conferir_dimensoes_tabela(dados)
    destino = pasta_sessao(sid, sub) / nome
    destino.write_bytes(dados)
    return destino


def apagar_sessao(sid: str) -> None:
    try:
        shutil.rmtree(pasta_sessao(sid, criar=False), ignore_errors=True)
    except ValueError:
        pass


def _ultima_modificacao(pasta: Path) -> float:
    """Maior mtime em toda a arvore da sessao (o mtime da raiz sozinho nao
    muda quando so' arquivos de subpastas sao escritos)."""
    ultimo = pasta.stat().st_mtime
    for raiz, dirs, arqs in os.walk(pasta):
        for nome in dirs + arqs:
            try:
                ultimo = max(ultimo, os.stat(os.path.join(raiz, nome)).st_mtime)
            except OSError:
                continue
    return ultimo


def limpar_sessoes_antigas(max_horas: float = IDADE_MAXIMA_HORAS,
                           agora: Optional[float] = None) -> int:
    """Apaga pastas de sessao sem modificacao ha mais de `max_horas` (rede de
    seguranca para sessoes que morreram sem chamar `apagar_sessao`, ex.:
    reinicio do servidor). Retorna quantas removeu."""
    raiz = raiz_sessoes()
    if not raiz.is_dir():
        return 0
    limite = (agora if agora is not None else time.time()) - max_horas * 3600.0
    n = 0
    for p in raiz.iterdir():
        try:
            if (p.is_dir() and _RE_SESSAO.match(p.name)
                    and _ultima_modificacao(p) < limite):
                shutil.rmtree(p, ignore_errors=True)
                n += 1
        except OSError:
            continue
    return n


class GuardaSessao:
    """Objeto guardado em `st.session_state`: quando a sessao do Streamlit
    termina, o estado e' descartado, o objeto e' coletado e a pasta da
    sessao e' apagada (`weakref.finalize`). Cumpre o "uploads apagados ao
    fim da sessao" sem depender de hook do servidor."""

    def __init__(self, sid: str):
        self.sid = sid
        self._fin = weakref.finalize(self, apagar_sessao, sid)

    def encerrar(self) -> None:
        self._fin()


# ── Erros sem vazar traceback nem caminho ────────────────────────────────
def registrar_erro(exc_ou_texto, contexto: str = "") -> str:
    """Grava o traceback completo no LOG DO SERVIDOR e devolve um id curto
    (8 hex) que o visitante pode citar. O texto exibido ao visitante nunca
    contem traceback nem caminho absoluto."""
    eid = uuid.uuid4().hex[:8]
    if isinstance(exc_ou_texto, BaseException):
        detalhe = "".join(traceback.format_exception(
            type(exc_ou_texto), exc_ou_texto, exc_ou_texto.__traceback__))
    else:
        detalhe = str(exc_ou_texto)
    log.error("[erro %s] %s\n%s", eid, contexto, detalhe)
    return eid


_RE_CAMINHO_WIN = re.compile(r"[A-Za-z]:[\\/][^\s'\"<>|:*?]*")
_RE_CAMINHO_POSIX = re.compile(r"(?<![\w.])/(?:[\w.\-]+/)+[\w.\-]*")


def ocultar_caminhos(texto: str) -> str:
    """Troca caminhos absolutos (Windows/POSIX) por `<caminho>`."""
    t = _RE_CAMINHO_WIN.sub("<caminho>", str(texto))
    return _RE_CAMINHO_POSIX.sub("<caminho>", t)


def mensagem_visitante(exc: BaseException, contexto: str,
                       pt: bool = True) -> Tuple[str, str]:
    """(mensagem amigavel, id de erro). No modo servidor e' o que a tela
    mostra; o detalhe tecnico fica so' no log. A mensagem inclui o motivo
    curto (`str(exc)`) com caminhos ocultados, porque erros de entrada do
    usuario (arquivo sem coluna X, poucas amostras...) sao acionaveis."""
    eid = registrar_erro(exc, contexto)
    motivo = ocultar_caminhos(str(exc)).strip().splitlines()[0][:300] if str(exc).strip() else ""
    if pt:
        msg = (f"Nao foi possivel concluir: {contexto}." +
               (f" Motivo: {motivo}" if motivo else "") +
               f" (codigo do erro: {eid})")
    else:
        msg = (f"Could not complete: {contexto}." +
               (f" Reason: {motivo}" if motivo else "") +
               f" (error id: {eid})")
    return msg, eid


def escapar_html(valor) -> str:
    """Escapa dado do usuario (rotulos de classe, nomes) antes de qualquer
    `unsafe_allow_html`."""
    import html
    return html.escape(str(valor), quote=True)


# ── Politica aplicada a uma Config ───────────────────────────────────────
# multielementar (tabela ICP) tambem chega por upload de CSV e passa pela
# MESMA checagem de arquivo dentro da sessao que o csv.
MODOS_ENTRADA_SERVIDOR = ("sintetico", "csv", "multielementar")


def impor_politica_servidor(cfg, sid: str):
    """Forca uma `Config` a respeitar o modo servidor (defesa em profundidade,
    aplicada no boot E imediatamente antes de rodar, mesmo que a UI ja' esconda
    os campos): saida sempre na pasta de resultados da sessao; entrada so'
    sintetica ou CSV que esteja DENTRO da pasta da sessao. Retorna a mesma
    `cfg` (mutada)."""
    cfg.output_root_folder = str(pasta_sessao(sid, "resultados"))
    if getattr(cfg, "mode", "sintetico") not in MODOS_ENTRADA_SERVIDOR:
        cfg.mode = "sintetico"
    cam = getattr(cfg, "csv_file", "") or ""
    if cfg.mode in ("csv", "multielementar"):
        try:
            dentro_da_sessao(cam, sid)
        except (ValueError, OSError):
            cfg.mode = "sintetico"
    # Campos de caminho de outros modos nunca chegam a ser lidos, mas nao
    # carregam valor do visitante.
    gf = getattr(cfg, "group_file", None)
    if gf:
        try:
            dentro_da_sessao(gf, sid)
        except (ValueError, OSError):
            cfg.group_file = None
    cfg.input_folder = ""
    if hasattr(cfg, "hsi_dataset_folder"):
        cfg.hsi_dataset_folder = ""
    # Tetos de recursos (A1-04): so' reduzem, nunca aumentam o que o
    # visitante pediu. A CLI e o modo local nao passam por aqui.
    for attr, _antes, teto in tetos_excedidos(cfg):
        setattr(cfg, attr, teto)
    return cfg


_TETOS_SERVIDOR = (("n_permutations", TETO_PERMUTACOES_SERVIDOR),
                   ("n_permutations_wold", TETO_PERMUTACOES_SERVIDOR),
                   ("n_bootstrap_bca", TETO_BOOTSTRAP_SERVIDOR),
                   ("n_bootstrap_vip", TETO_BOOTSTRAP_SERVIDOR))


def tetos_excedidos(cfg) -> "list[Tuple[str, int, int]]":
    """[(atributo, valor pedido, teto)] dos campos da `cfg` acima dos tetos
    do modo servidor -- a tela usa isto para AVISAR o visitante que o valor
    foi reduzido (nada de reduzir em silencio)."""
    fora = []
    for attr, teto in _TETOS_SERVIDOR:
        valor = getattr(cfg, attr, None)
        if isinstance(valor, int) and valor > teto:
            fora.append((attr, valor, teto))
    return fora


def msg_segura(exc_ou_texto) -> str:
    """Texto de erro para exibir na tela: no modo servidor, sem caminho
    absoluto (o detalhe fica so' no log); no modo local, o texto integral."""
    texto = str(exc_ou_texto)
    return ocultar_caminhos(texto) if modo_servidor() else texto


def rotulo_pasta(pasta) -> str:
    """Como mostrar uma pasta de resultados ao usuario: caminho absoluto no
    modo local; um rotulo neutro no modo servidor (o caminho da sessao no
    servidor nao e' informacao do visitante)."""
    if modo_servidor():
        return "(pasta isolada desta sessão)"
    return os.path.abspath(str(pasta))
