"""
guaraci.py v1.0.0 — Interface profissional GUARACI para o pipeline quimiometrico
☀  GUARACI — Plataforma quimiometrica com validacao anti-vazamento por padrao
Quimiometria • Machine Learning • Espectroscopia multitecnica

Uso:
    python guaraci.py

Requer: pipeline.py e cli_assistente.py no mesmo diretorio.
Rich 15.0+ necessario (pip install rich).
"""

from __future__ import annotations

import contextlib
import logging
import json
import copy
import os
import re as _re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

# ---------------------------------------------------------------------------
# UTF-8 no Windows antes de qualquer import rich
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            # getattr (nao `.reconfigure` direto): o tipo de sys.stdout e'
            # TextIO, que nao declara `reconfigure` -- evita um `type: ignore`
            # cuja necessidade varia com a plataforma/versao do mypy.
            getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")
            getattr(sys.stderr, "reconfigure")(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass   # stdout/stderr redirecionado p/ algo sem reconfigure util
    try:
        try:
            # Mesmo efeito do `chcp 65001`, sem abrir um cmd.exe (centenas de ms no
            # Windows com antivirus) a cada partida.
            import ctypes
            _k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            _k32.SetConsoleOutputCP(65001)
            _k32.SetConsoleCP(65001)
        except (AttributeError, OSError, ValueError):
            import subprocess
            # shell=True e' necessario aqui: `chcp` e' builtin do cmd.exe, nao um
            # executavel. Os argumentos sao constantes -- nada vindo do usuario
            # chega nesta linha, entao nao ha superficie de injecao.
            subprocess.run("chcp 65001", capture_output=True, shell=True)  # noqa: S602
    except OSError:
        pass   # chcp indisponivel neste shell
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ---------------------------------------------------------------------------
# Rich
# ---------------------------------------------------------------------------
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from rich.text import Text
from rich.rule import Rule
from rich.live import Live
from rich.markup import escape
from rich import box as rbox

# ---------------------------------------------------------------------------
# Pipeline — ZERO modificacoes analiticas
# ---------------------------------------------------------------------------
import yaml
from guaraci import config_io as _cio
from guaraci.config import (AUTOR_GITHUB_URL, AUTOR_NOME, AUTOR_ORCID, COPYRIGHT_LINHA, DISCUSSIONS_URL, LICENCA_SPDX,
                            REPO_URL, Config)
from guaraci.app_logic import (
    LogThreadSafe as _LogThreadSafe,
    figures_completed as _figures_completed,
    log_warnings as _avisos_do_log,
    log_progress as _progresso_do_log,
    fmt_time as _fmt_time,
)



class _PipelinePreguicoso:
    """O motor (`guaraci.pipeline`: sklearn, scipy, pandas, matplotlib) so' e' importado
    no primeiro uso. Menu, `--help`, ajuda e assistentes abrem sem pagar esse custo
    (medido: menu em ~5 s com o import no topo)."""

    # Nomes que moram em modulos leves: resolvidos sem importar o motor.
    _LEVES = {
        "guaraci.config": ("__version__", "NIVEL_NOME", "NOME_RELATORIOS", "NOME_GRAFICOS", "Config",
                           "pasta_execucao"),
        "guaraci.config_io": ("validar_fonte_de_dados", "save_config", "load_config"),
        "guaraci.modos_analise": ("OBJETIVO_ROTULO", "resolve_objective", "should_generate", "describe_plan",
                                  "CLASSIFICACAO", "QUANTIFICACAO", "EXPLORATORIO"),
    }

    def __getattr__(self, nome: str) -> Any:
        import importlib
        for modulo, nomes in self._LEVES.items():
            if nome in nomes:
                return getattr(importlib.import_module(modulo), nome)
        import guaraci.pipeline as _p
        return getattr(_p, nome)


pq = _PipelinePreguicoso()


def _nome_pasta(chave: str) -> str:
    """Nome de pasta de saida no idioma da sessao (config.PASTAS_EXECUCAO)."""
    from guaraci.config import nome_pasta
    return nome_pasta(chave)


def _executar(cfg: Config) -> Any:
    res = pq.executar(cfg)
    _figuras_pos_execucao(cfg)
    return res


def _figuras_pos_execucao(cfg: Config) -> None:
    """Graficos que dependem do modelo JA salvo (coeficientes b e Williams de
    cada regressao por especie; auditoria de figuras, secao D). Desenha a
    partir de Modelos/*.joblib da corrida, sem reajustar nada; falha aqui
    nunca muda o resultado da corrida."""
    if str(getattr(cfg, "mode", "")) == "hsi":
        return
    try:
        from guaraci import figuras_extras
        figuras_extras.figuras_regressao_do_modelo(
            str(getattr(cfg, "output_folder", "") or ""), cfg)
    except Exception as e:  # noqa: BLE001 -- figura complementar
        print(f"  [AVISO] graficos de coeficientes/Williams nao gerados: {e}")


# ---------------------------------------------------------------------------
# ASSISTENTE NO FLUXO DE EXECUCAO (docs: assistente/IMPLEMENTACAO.md, "O que
# fica para as outras frentes"): pre-voo antes de rodar (assistente_regras),
# ficha anti-vazamento depois (assistente_vazamento) e codigo GUA-E no painel
# de falha (assistente_erros). Nada aqui altera a configuracao nem o resultado;
# uma falha do assistente so' avisa, nunca derruba a execucao.
# ---------------------------------------------------------------------------
#: Ate este tamanho de entrada o pre-voo carrega os dados para as regras de
#: dados. A execucao os carrega de novo; com dados maiores so' as regras de
#: configuracao rodam antes (`guaraci checar CONFIG --dados` faz o resto).
_PREFLIGHT_MAX_BYTES = 50 * 1024 * 1024
#: `--sem-preflight`: `guaraci run` roda mesmo com achado critico do pre-voo.
_SEM_PREFLIGHT: List[bool] = [False]
_NOME_FICHA = "ficha_vazamento.md"


def _bytes_da_entrada(cfg: Config) -> Optional[int]:
    """Tamanho dos dados de entrada, para decidir se o pre-voo pode carrega-los.
    None = nao carregar (caminho ausente, imagem/HSI ou acima do limite)."""
    modo = str(getattr(cfg, "mode", "") or "")
    if modo == "sintetico":
        return 0
    if modo in ("hsi", "imagem"):
        return None
    alvo = str((getattr(cfg, "csv_file", "") if modo in ("csv", "multielementar")
                else getattr(cfg, "input_folder", "")) or "")
    if not alvo or not os.path.exists(alvo):
        return None
    if os.path.isfile(alvo):
        total = os.path.getsize(alvo)
        return total if total <= _PREFLIGHT_MAX_BYTES else None
    total = 0
    for raiz, _subs, arqs in os.walk(alvo):
        for a in arqs:
            try:
                total += os.path.getsize(os.path.join(raiz, a))
            except OSError:
                continue
            if total > _PREFLIGHT_MAX_BYTES:
                return None
    return total


def _resumo_dados_assistente(cfg: Config, forcar: bool = False) -> Any:
    """`assistente_regras.ResumoDados` dos dados de `cfg` (o mesmo `load_data` da
    execucao, numa copia do cfg e com a saida dos leitores recolhida), ou None se
    os dados passam do limite (sem `forcar`) ou nao carregam -- a execucao mostra
    esse erro com o codigo GUA-E."""
    if not forcar and _bytes_da_entrada(cfg) is None:
        return None
    import io
    from guaraci import assistente_regras as _ar
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            eixo, X, rotulos, _conc, mae_id, _meta = pq.load_data(copy.copy(cfg))
            return _ar.resumir_dados(X, eixo, rotulos, mae_id)
    except Exception as e:  # noqa: BLE001 -- dado externo falha de muitas formas; o pre-voo nao decide por isso
        logging.getLogger(__name__).debug("pre-voo sem resumo dos dados: %s", e)
        return None


def _preflight(cfg: Config, lang: str = "PT", forcar_dados: bool = False) -> Tuple[List[Any], Any]:
    """Regras do assistente antes de rodar: configuracao sempre; dados quando
    cabem no limite (ou com `forcar_dados`). Devolve (achados, resumo_dos_dados)."""
    try:
        from guaraci import assistente_regras as _ar
        dados = _resumo_dados_assistente(cfg, forcar=forcar_dados)
        return _ar.avaliar(cfg, dados, None, lang), dados
    except Exception as e:  # noqa: BLE001 -- o assistente nunca impede a execucao por falha propria
        logging.getLogger(__name__).debug("pre-voo indisponivel: %s", e)
        return [], None


def _texto_preflight(achados: List[Any], dados: Any, lang: str = "PT") -> str:
    """Texto do pre-voo para o terminal (mesmo formato de `guaraci checar`)."""
    from guaraci import assistente_regras as _ar
    en = str(lang).upper() == "EN"
    linhas = ["Pre-flight (assistant rules):" if en else "Pre-voo (regras do assistente):",
              _ar.formatar_achados(achados, lang, refs=False)]
    if dados is None:
        linhas.append("(data not loaded before the run: configuration rules only; `guaraci checar CONFIG --dados` "
                      "applies the data rules)" if en else
                      "(dados nao carregados antes da execucao: so' regras de configuracao; `guaraci checar CONFIG --dados` "
                      "aplica as regras de dados)")
    return "\n".join(linhas)


def _criticos(achados: List[Any]) -> List[Any]:
    return [a for a in achados if getattr(a, "severidade", "") == "critico"]


def _diagnostico_erro(exc: Any, lang: str = "PT") -> str:
    """Mensagem original da falha + codigo GUA-E, causa e acao (assistente_erros).
    Se o diagnostico falhar, a mensagem original continua aparecendo."""
    try:
        from guaraci.assistente_erros import diagnosticar_excecao
        return diagnosticar_excecao(exc, "EN" if str(lang).upper() == "EN" else "PT")
    except Exception:  # noqa: BLE001 -- o painel de erro nunca pode falhar por causa do diagnostico
        return str(exc)


def _assistente_pos_execucao(cfg: Config, dados: Any = None, lang: str = "PT") -> Optional[Tuple[str, List[Any]]]:
    """Depois de rodar: grava <execucao>/Relatorios/ficha_vazamento.md (ao lado do
    resumo_modelo.txt) e devolve (caminho da ficha, achados de resultado). Le o
    resumo_modelo.txt da execucao; sem pasta de execucao devolve None. Falha aqui
    so' avisa e nunca muda o resultado da corrida (como _figuras_pos_execucao)."""
    pasta = str(getattr(cfg, "output_folder", "") or "")
    if not pasta or not os.path.isdir(pasta):
        return None
    en = str(lang).upper() == "EN"
    try:
        from guaraci import assistente_regras as _ar
        from guaraci import assistente_vazamento as _av
        from guaraci.config import NOME_RELATORIOS, pasta_execucao
        rel = pasta_execucao(pasta, NOME_RELATORIOS)
        arq_resumo = os.path.join(rel, "resumo_modelo.txt")
        resultado = None
        if os.path.isfile(arq_resumo):
            from guaraci.resumo_parse import ler_resumo_canonico   # chaves fixas, nao o .txt traduzido
            resultado = _ar.resultado_de_resumo(ler_resumo_canonico(arq_resumo) or "")
        os.makedirs(rel, exist_ok=True)
        destino = os.path.join(rel, _NOME_FICHA)
        with open(destino, "w", encoding="utf-8") as fh:
            fh.write(_av.ficha(cfg, dados, resultado, "EN" if en else "PT"))
        achados = (_ar.avaliar(cfg, dados, resultado, "EN" if en else "PT", momentos=["resultado"])
                   if resultado is not None else [])
        return destino, achados
    except Exception as e:  # noqa: BLE001 -- complemento; a corrida ja terminou e gravou tudo
        print(f"  [{'WARNING' if en else 'AVISO'}] "
              + (f"leakage sheet not written: {e}" if en else f"ficha anti-vazamento nao gravada: {e}"))
        return None


def _mostrar_achados(achados: List[Any], lang: str = "PT", titulo: str = "", dados: Any = "-",
                     vazio: bool = True) -> None:
    """Achados do assistente no terminal (cor e simbolo por severidade). `dados=None`
    avisa que as regras de dados nao rodaram; `vazio=False` nao escreve nada sem achado."""
    en = str(lang).upper() == "EN"
    if not achados and not vazio:
        return
    if titulo:
        console.print(f"  [{PA}]{escape(titulo)}[/{PA}]")
    estilo = {"critico": (PR, "✖", "CRITICAL" if en else "CRITICO"), "aviso": (PA, "⚑", "WARNING" if en else "AVISO"),
              "info": (PS, "ℹ", "INFO")}
    if not achados:
        console.print(f"  [{PG}]✓ {'No problem found by the assistant rules.' if en else 'Nenhum problema encontrado pelas regras do assistente.'}[/{PG}]")
    for a in achados:
        cor, simb, rot = estilo.get(a.severidade, (PW, "-", a.severidade.upper()))
        console.print(f"  [{cor}]{simb} [{rot}][/{cor}] {escape(a.mensagem)}")
        console.print(f"     [{PM}]{'What to do' if en else 'O que fazer'}: {escape(a.fazer)}[/{PM}]")
        if a.campo:
            valor = str(a.valor).lower() if isinstance(a.valor, bool) else a.valor
            console.print(f"     [{PM}]{'Accept' if en else 'Aceitar'}: {escape(f'{a.campo}: {valor}')}[/{PM}]")
    if dados is None:
        console.print(f"  [{PM}]" + escape("(data not loaded: configuration rules only; screen [6.5] or `guaraci checar CONFIG --dados` "
                                           "applies the data rules)" if en else
                                           "(dados nao carregados: so' regras de configuracao; a tela [6.5] ou `guaraci checar CONFIG --dados` "
                                           "aplica as regras de dados)") + f"[/{PM}]")


def _texto_pos_execucao(pos: Optional[Tuple[str, List[Any]]], lang: str = "PT") -> str:
    """Resumo do assistente para o fim da execucao: achados de resultado e onde ficou a ficha."""
    if not pos:
        return ""
    from guaraci import assistente_regras as _ar
    en = str(lang).upper() == "EN"
    destino, achados = pos
    linhas = ["Result check (assistant rules):" if en else "Leitura do resultado (regras do assistente):",
              _ar.formatar_achados(achados, lang, refs=False),
              (f"Leakage sheet: {destino}" if en else f"Ficha anti-vazamento: {destino}")]
    return "\n".join(linhas)


def _figura_tecnica(nome_funcao: str, *args, **kwargs) -> None:
    """Figura das tecnicas especiais [3.5]; falha so' avisa (o CSV ja saiu)."""
    try:
        from guaraci import figuras_extras
        getattr(figuras_extras, nome_funcao)(*args, **kwargs)   # save() ja' imprime o caminho
    except Exception as e:  # noqa: BLE001 -- figura complementar
        console.print(f"  [{PR}]figura {nome_funcao}: {escape(str(e))}[/{PR}]")


_save_config = _cio.save_config
_load_config = _cio.load_config

# Dicionarios de i18n/perfis do cli_assistente. Import de pacote normal (mesmo
# pacote): importar o modulo NAO dispara main() (guardado por __main__), entao
# nao ha efeito colateral — o antigo carregamento por caminho (spec_from_file)
# so existia porque os modulos eram scripts soltos na raiz.
import guaraci.cli_assistente as _cli
from guaraci import catalogo as _cat
from guaraci import arte as _arte
from guaraci import cli_comandos as _cmd
from guaraci import contexto as _contexto
from guaraci import extras_opcionais as _extras
import guaraci.preferencias_visuais as _prefs_visuais

def _try(name):
    """Le um simbolo de `cli_assistente`. FALHA ALTO se ele nao existir: o
    antigo fallback silencioso ({}) escondia nomes inexistentes (m-21 do
    parecer de 2026-09-25 achou 2, `save_config`/`load_config`, que nunca
    estiveram la')."""
    return getattr(_cli, name)

_FIELD_NAMES          = _try("FIELD_NAMES")
_HELP_DB              = _try("HELP_DB")
_RISK_CLASS           = _try("RISK_CLASS")
_PALETAS_COR          = _try("PALETAS_COR")
_aplicar_paleta       = _cli.apply_palette
_FONT_PRESETS         = _try("FONT_PRESETS")
_TECNICAS             = _try("TECNICAS")
_REFERENCIAS_GUARACI  = _try("REFERENCIAS_GUARACI")
_CONFIG_SPEC         = _try("_CONFIG_SPEC")
_SPEC_BY_KEY         = _try("_SPEC_BY_KEY")
_DDSIMCA_DISPLAY     = _try("_DDSIMCA_DISPLAY")
_DDSIMCA_INPUT       = _try("_DDSIMCA_INPUT")
_coagir_valor        = _try("_coagir_valor")
_attr_para_yaml      = _try("_attr_para_yaml")
_fmt_yaml            = _try("_fmt_yaml")

# __all__ so' cobre a superficie realmente consumida de fora deste arquivo
# (confirmado por grep + suite de testes): `main` e' o ponto de entrada da
# CLI; `Config` e' usado extensivamente pela suite via `guaraci_mod.Config`;
# O resto (menu_*, cls, I18N, FIELD_NAMES/HELP_DB/RISK_CLASS/
# PALETAS_COR/FONT_PRESETS/TECNICAS/REFERENCIAS_GUARACI, os aliases locais
# _executar/_save_config/_load_config) e' wiring interno da CLI -- nunca
# consumido fora deste arquivo, renomeado para _privado nesta auditoria.
__all__ = [
    "main",
    "Config",
    "aplicar_perfil",
]

# Persistencia de estado do CLI.
#
# CORRIGIDO em 2026-08-16 (varredura de bugs): estas tres funcoes eram
# wrappers que procuravam implementacoes em `cli_assistente` -- que NUNCA
# existiram la'. `getattr(..., None)` devolvia None, entao `_carregar_*`
# retornava sempre {} e `_salvar_visual_cfg` era um no-op SILENCIOSO. O
# comentario do proprio codigo (secao do Modo Iniciante/Avancado) ja
# registrava "esse esta quebrado ... fora do escopo desta feature
# consertar isso" desde 2026-07-13.
#
# Consequencias reais, ambas do tipo que o projeto mais combate (o software
# mente sem travar):
#   1. As 4 opcoes do menu Visualizacao (Paleta/Fonte/Grid/Alpha) gravavam
#      no dicionario, chamavam _salvar_visual_cfg(), imprimiam "OK Paleta:
#      X" e NAO persistiam nada -- a confirmacao era falsa. Na proxima
#      abertura o valor voltava ao default. O mesmo valia para o DPI
#      (`_sincronizar_dpi`) e para toda a aplicacao de estilo em
#      `_rodar_pipeline` (paleta/fonte/grid/alpha nos rcParams do
#      matplotlib), que le de `_carregar_visual_cfg()`.
#   2. Codigos de especie cadastrados pelo usuario eram gravados
#      corretamente pelo menu (`_salvar_cod`, que tem implementacao propria
#      e funciona), apareciam listados no proprio menu (`_cod_usr`, idem),
#      mas NAO eram aplicados a analise: a unica linha que injeta os
#      codigos no pipeline (`pq.CODIGO_ESPECIE.update(cod_u)`) usava o
#      wrapper quebrado e recebia {} sempre.
#
# Agora as tres leem/gravam direto em _USER_DIR, no mesmo padrao ja usado
# por `_cod_usr`/`_salvar_cod` (que sempre funcionaram) e pelos demais
# arquivos de estado (_CFG_PATH, _LANG_FLAG, _MODO_FLAG).
#
# Leitura/escrita movidas para `guaraci.preferencias_visuais` (2026-09-08):
# a aba Modelo do app web precisa do MESMO arquivo (paleta escolhida na CLI
# aparece selecionada na web e vice-versa) e nao pode importar este modulo
# (5 mil linhas + rich). Aqui ficam so' os wrappers com a apresentacao de
# erro da CLI.
def _carregar_visual_cfg() -> dict:
    """Config visual (paleta/fonte/grid/alpha/dpi) de _VISUAL_PATH."""
    return _prefs_visuais.load_visual_config(_VISUAL_PATH)

def _salvar_visual_cfg(d: dict) -> None:
    """Grava a config visual. Falha de escrita AVISA em vez de sumir em
    silencio -- mesma licao de `_salvar_cod`: o usuario nao pode ver um
    'salvo' que nao aconteceu."""
    try:
        _prefs_visuais.save_visual_config(d, _VISUAL_PATH)
    except OSError as e:
        console.print(f"[err]✗ Falha ao salvar config visual: {e}[/err]")

def _carregar_codigos_usuario() -> dict:
    """Codigos de especie cadastrados pelo usuario, de _CODIGOS_PATH.

    Mesma leitura de `_cod_usr()` (menu de codificacao) -- e' de proposito
    que as duas leiam o MESMO arquivo: o que o menu lista tem de ser o que
    a analise aplica.
    """
    try:
        p = _CODIGOS_PATH
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
# _BASE_DIR: diretorio de INSTALACAO do pacote -- so' para recursos
# somente-leitura que o pacote ja traz consigo (ex.: CITATION.cff).
#
# _USER_DIR: onde o CLI grava ESTADO do usuario (config.yaml, perfis
# salvos, flags de idioma/mode, codigos customizados). CORRIGIDO em
# 2026-08-07 (achado do "checkup geral" de interface -- ver
# ): ate' entao esses arquivos eram gravados dentro de
# _BASE_DIR, ou seja, DENTRO do diretorio de instalacao do pacote. Isso
# quebra em qualquer instalacao read-only (pip de sistema, imagem Docker,
# alguns `pip install --user`) -- `save_config()` logo antes de rodar o
# pipeline (ver `_rodar_pipeline`) nao tinha nenhuma guarda contra isso e
# derrubava o CLI com um PermissionError bem na hora de rodar a analise.
# Home do usuario e' gravavel em praticamente qualquer instalacao.
_BASE_DIR    = Path(os.path.dirname(os.path.abspath(__file__)))
_USER_DIR    = Path.home() / ".guaraci"
_CFG_PATH    = _USER_DIR / "config.yaml"
_PERFIS_DIR  = _USER_DIR / "perfis"
_LANG_FLAG   = _USER_DIR / ".cli_wizard_done"
_CODIGOS_PATH= _USER_DIR / "codigos_usuario.json"
_VISUAL_PATH = _USER_DIR / "visual_config.json"


def _migrar_estado_legado() -> None:
    """Copia, uma vez, o estado gravado pela versao anterior (dentro de
    _BASE_DIR) para o novo local (_USER_DIR), se o novo local ainda nao
    tiver esse arquivo. NUNCA sobrescreve nem apaga o arquivo antigo --
    so' copia o que falta, best-effort (falha de permissao aqui nao pode
    impedir o CLI de abrir). Chamada uma vez no inicio de `main()`, nao na
    importacao do modulo (importar `guaraci.guaraci` -- em testes, por
    exemplo -- nao deve escrever no HOME de quem esta rodando os testes).
    """
    try:
        _USER_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    for nome, alvo in (
        ("config.yaml", _CFG_PATH),
        (".cli_wizard_done", _LANG_FLAG),
        ("codigos_usuario.json", _CODIGOS_PATH),
        (".cli_modo_usuario", _MODO_FLAG),
        ("visual_config.json", _VISUAL_PATH),
    ):
        origem = _BASE_DIR / nome
        if origem.exists() and not alvo.exists():
            try:
                shutil.copy2(origem, alvo)
            except OSError:
                pass
    # `visual_config.json` tinha uma origem legada A MAIS: a versao antiga
    # gravava por caminho RELATIVO, entao o arquivo ficava no diretorio de
    # onde o CLI foi chamado (tipicamente a raiz do repositorio), nao em
    # _BASE_DIR. Recuperado aqui para nao perder a paleta que o usuario ja
    # tinha escolhido -- so' se _USER_DIR ainda nao tiver a sua.
    if not _VISUAL_PATH.exists():
        try:
            origem_cwd = Path.cwd() / "visual_config.json"
            if origem_cwd.is_file():
                shutil.copy2(origem_cwd, _VISUAL_PATH)
        except OSError:
            pass
    origem_perfis = _BASE_DIR / "perfis"
    if origem_perfis.is_dir() and not _PERFIS_DIR.exists():
        try:
            shutil.copytree(origem_perfis, _PERFIS_DIR)
        except OSError:
            pass

# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
# Idioma padrao para quem abre pela primeira vez: ingles (uso principal). A escolha feita com [I]
# fica gravada em _LANG_FLAG e vale nas proximas aberturas.
_STATE: Dict[str, Any] = {"lang": "EN", "modo_usuario": "iniciante"}

def _lang() -> str:
    return _STATE["lang"]

def _set_lang(l: str) -> None:
    _STATE["lang"] = l
    # Relatorios (model_card.md, PDF, Word, Excel, PPTX) saem no idioma da sessao: o pipeline e os
    # geradores leem GUARACI_IDIOMA (guaraci.idioma_relatorio), nao o estado do menu.
    os.environ["GUARACI_IDIOMA"] = "en" if l == "EN" else "pt"
    try:
        _USER_DIR.mkdir(parents=True, exist_ok=True)
        _LANG_FLAG.write_text(l, encoding="utf-8")
    except OSError:
        pass

def _sincronizar_idioma() -> None:
    """Um idioma so' para menu, mensagens e relatorios: GUARACI_IDIOMA definida pelo usuario vence;
    senao vale o idioma gravado com [I], e ele passa a valer para os relatorios. Sem escolha gravada
    a variavel fica como esta' (relatorios e subcomandos seguem o padrao portugues de sempre)."""
    env = os.environ.get("GUARACI_IDIOMA", "").strip().lower()
    if env:
        _STATE["lang"] = "EN" if env.startswith("en") else "PT"
        return
    try:
        salvo = _LANG_FLAG.read_text(encoding="utf-8").strip()
    except OSError:
        return
    if salvo in ("EN", "PT"):
        _STATE["lang"] = salvo
        os.environ["GUARACI_IDIOMA"] = "en" if salvo == "EN" else "pt"


def _toggle_idioma() -> str:
    novo = "EN" if _lang() == "PT" else "PT"
    _set_lang(novo)
    return novo

# Modo Iniciante/Avancado (menos densidade de configuracao para quem so' quer os padroes): alterna
# GLOBALMENTE se os submenus escondem campos avancados por padrao. Usa
# arquivo-flag proprio (mesmo padrao de persistencia do idioma), nao o
# visual_config.json -- que guarda aparencia de FIGURA (paleta/fonte/grid),
# nao estado de navegacao do menu; sao coisas diferentes.
# (Este comentario dizia, ate 2026-08-16, que o mecanismo de
# visual_config.json estava quebrado e que consertar estava "fora do
# escopo". Estava mesmo quebrado desde 2026-07-13 e foi CORRIGIDO na
# varredura de bugs de 2026-08-16 -- ver as funcoes de persistencia no topo
# do modulo.)
_MODO_FLAG = _USER_DIR / ".cli_modo_usuario"

def _modo_usuario() -> str:
    return _STATE["modo_usuario"]

def _set_modo_usuario(m: str) -> None:
    _STATE["modo_usuario"] = m
    try:
        _USER_DIR.mkdir(parents=True, exist_ok=True)
        _MODO_FLAG.write_text(m, encoding="utf-8")
    except OSError:
        pass

def _toggle_modo_usuario() -> str:
    novo = "avancado" if _modo_usuario() == "iniciante" else "iniciante"
    _set_modo_usuario(novo)
    return novo

if _MODO_FLAG.exists():
    try:
        _v = _MODO_FLAG.read_text(encoding="utf-8").strip()
        if _v in ("iniciante", "avancado"):
            _STATE["modo_usuario"] = _v
    except OSError:
        pass

# ---------------------------------------------------------------------------
# PALETA + tema + console: fonte unica em guaraci_theme (compartilhada com o
# cli_assistente para visual identico). Nomes reexportados sem alteracao.
# ---------------------------------------------------------------------------
from guaraci.guaraci_theme import (  # noqa: E402
    PA, PF, PS, PR, PW, PM, PD, PG, console, _W,
)
# Lógica pura extraída da CLI (item 19): testável sem Rich/console. Ver cli_logic.py.
from guaraci.cli_logic import (  # noqa: E402
    trunc as _trunc,
    truncate_desc_by_sentence as _truncar_desc_por_frase,
    validate_ranges as _validar_faixas_puro,
    count_dx as _count_dx,
)

# Estado global da tecnica selecionada (persiste entre menus)
_TECNICA_SELECIONADA: Dict[str, str] = {"key": "ft-nir", "nome": "FT-NIR"}

# ---------------------------------------------------------------------------
# Internacionalizacao — sem repeticao entre idiomas
# ---------------------------------------------------------------------------
_I18N: Dict[str, Dict[str, str]] = {
    "PT": {
        # Titulos de menu
        "t_idioma":     "Idioma",
        # Descricoes de secao (curtas)
        # Faltavam pra' _guaraci_navegar_secoes cobrir as 18 abas reais
        # (achado do Agente 6, docs/projeto/identidade_visual.md): so' existiam d_ das 12
        # abas com campo de _CONFIG_SPEC associado.
        # Grupos do menu principal (docs/projeto/identidade_visual.md secao 4 -- Agente 5.2,
        # aprovado 2026-09-01: substitui os 3 grupos antigos config/analise/
        # sistema por 6 grupos alinhados ao fluxo real de trabalho)
        "grp_preparar": "Preparar",
        "grp_planejar": "Planejar",
        "grp_modelar":  "Modelar",
        "grp_validar":  "Validar",
        "grp_prever":   "Prever",
        "grp_sistema":  "Sistema",
        "grp_execucao": "Execucao",
        # Acoes
        "rodar":        "Rodar Pipeline",
        "salvar":       "Salvar configuracao",
        "carregar":     "Carregar configuracao",
        "nome_saida":   "Nome da Saida",
        "sair":         "Sair",
        # Status
        "status_ok":    "Pronto",
        "status_erro":  "Configurar dados",
        "dados_ok":     "{n} arquivos .dx",
        "dados_err":    "Pasta invalida",
        # Interacao
        "opcao":        "Opcao",
        "voltar":       "Voltar",
        "continuar":    "Enter para continuar",
        "cancelado":    "Cancelado.",
        "invalido":     "Opcao invalida.",
        "mantido":      "Mantido.",
        "novo_valor":   "Novo valor (Enter=manter, ?=ajuda): ",
        "confirmar":    "Confirmar? (s/n): ",
        "conf_anal":    "Alteracao afeta resultados — confirmar? (s/n): ",
        "atualizado":   "Atualizado: {campo} = {valor}",
        "opt_sim":      "Sim",
        "opt_nao":      "Nao",
        "escolha_num":  "Numero ou Enter=manter: ",
        "ajuste_nivel": "Ajustado automaticamente (nao pertinente a este nivel): {campos}",
        # Checklist
        "chk_dados":    "Dados carregados",
        "chk_csv":      "CSV localizado",
        "chk_leak":     "Anti-leakage ativo",
        "chk_leak_high":   "Validação agrupada por amostra física (mae_id) — anti-vazamento ATIVO",
        "chk_leak_medium": "Validação agrupada pelo grupo que VOCÊ declarou (não verificável) — garantia MÉDIA",
        "chk_leak_none":   "SEM validação agrupada: não há chave de grupo (StratifiedKFold). Réplicas podem ir para os dois lados; trate as métricas como exploratórias",
        "chk_leak_verificar": "Anti-vazamento depende da estrutura de pastas (a garantia real é declarada após ler os dados)",
        "aviso_sem_grupo": "AVISO: nenhuma chave de grupo (coluna_grupo/arquivo_grupos): a validação NÃO é agrupada e o resultado NÃO tem proteção contra vazamento de réplicas.",
        "conf_sem_grupo": "Rodar SEM validação agrupada? As métricas serão exploratórias (s/n): ",
        "chk_saida":    "Pasta de saida definida",
        "chk_hw":       "Hardware compativel",
        "chk_preproc":  "Pre-processamento definido",
        "chk_err_dados":"Pasta de dados nao encontrada",
        "chk_err_csv":  "Arquivo CSV nao encontrado",
        "chk_err_leak": "GroupKFold DESATIVADO — risco de leakage",
        "chk_warn_hw":  "RAM baixa com modulos pesados ativos",
        "chk_tempo":    "Tempo estimado",
        "chk_descarte": "{n} espectros serao DESCARTADOS (faixa espectral incompativel)",
        "chk_orfaos":   "{n} amostras sem mae_id — entram SEM protecao anti-leakage",
        "chk_grupos":   "{n} grupos de replica (mae_id)",
        "chk_dica_jobs": "(n_jobs_permutation=1 — subir p/ 4 reduz o tempo sem mudar o resultado)",
        "chk_modo":     "Modo",
        "chk_prescan_erro": "pre-varredura indisponivel ({erro})",
        # Hardware
        "hw_alto":      "Alto Desempenho",
        "hw_medio":     "Desempenho Medio",
        "hw_basico":    "Desempenho Basico",
        "hw_limitado":  "Limitado",
        "hw_rec":       "Perfil recomendado",
        "hw_cpu":       "CPU fisicos",
        "hw_threads":   "Threads logicos",
        "hw_ram_total": "RAM total",
        "hw_ram_livre": "RAM livre",
        "hw_disco":     "Disco livre",
        # Execucao
        "exec_inicio":  "Iniciando analise...",
        "exec_leitura": "Leitura dos espectros",
        "exec_preproc": "Pre-processamento",
        "exec_pca":     "PCA + HCA",
        "exec_plsda":   "PLS-DA",
        "exec_opls":    "OPLS-DA",
        "exec_dds":     "DD-SIMCA",
        "exec_valid":   "Validacao estatistica",
        "exec_relat":   "Relatorios e figuras",
        "exec_bench":   "Benchmark (SVM/RF/XGB)",
        "exec_mc":      "Monte Carlo CV",
        "exec_concluido": "Analise concluida",
        "exec_erro":    "Erro na execucao",
        "exec_saida":   "Resultados salvos em",
        "exec_interrompido": "Interrompido pelo usuario",
        "exec_objetivo": "Objetivo",
        "exec_eta":      "Tempo estimado restante",
        "exec_figuras":  "Figuras geradas",
        "exec_avisos":   "Avisos",
        "exec_sem_avisos": "Nenhum aviso ate agora",
        # Resumo cientifico
        "res_tecnica":  "Tecnica",
        "res_preproc":  "Pre-processamento",
        "res_modelo":   "Modelo principal",
        "res_lvs":      "Max. variaveis latentes",
        "res_valid":    "Validacao",
        "res_perm":     "Permutacoes",
        "res_opls":     "OPLS-DA",
        "res_dds":      "DD-SIMCA",
        "res_bench":    "Benchmark",
        "res_mc":       "Monte Carlo",
        "res_shap":     "SHAP",
        "res_dpi":      "Resolucao (DPI)",
        "res_fmt":      "Formato figura",
        "res_nivel":    "Nivel",
        "res_tag":      "Identificador",
        # Nome de pasta
        "tag_atual":    "Identificador atual",
        "tag_auto":     "(automatico)",
        "tag_novo":     "Novo identificador (Enter=manter, -=limpar): ",
        "tag_limpo":    "Identificador removido — proximo run usa timestamp.",
        # Perfis
        "perf_tempo":   "Tempo",
        "perf_uso":     "Indicado para",
        # Codigos
        "cod_cadastrar":"Cadastrar novo codigo",
        "cod_listar":   "Listar todos os codigos",
        "cod_novo_cod": "Codigo (2-4 letras maiusculas, ex: MAN): ",
        "cod_novo_esp": "Nome da especie para '{cod}': ",
        "cod_salvo":    "Cadastrado: {cod} = {esp}",
        "cod_invalido": "Codigo invalido. Use 2-4 letras maiusculas.",
        # Visualizacao
        "viz_paleta":   "Paleta de Cores",
        "viz_fonte":    "Tamanho de Fonte",
        "viz_grid":     "Grade",
        "viz_alpha":    "Transparencia dos Pontos",
        # Pipeline info
        "pip_sem_dados": "Corrija a pasta de dados antes de rodar.",
        # Guaraci fala
        "g_prefixo":    "Guaraci:",
    },
    "EN": {
        "t_idioma":     "Language",
        "grp_preparar": "Prepare",
        "grp_planejar": "Plan",
        "grp_modelar":  "Model",
        "grp_validar":  "Validate",
        "grp_prever":   "Predict",
        "grp_sistema":  "System",
        "grp_execucao": "Execution",
        "rodar":        "Run Pipeline",
        "salvar":       "Save configuration",
        "carregar":     "Load configuration",
        "nome_saida":   "Output Name",
        "sair":         "Exit",
        "status_ok":    "Ready",
        "status_erro":  "Configure data",
        "dados_ok":     "{n} .dx files",
        "dados_err":    "Invalid folder",
        "opcao":        "Option",
        "voltar":       "Back",
        "continuar":    "Press Enter to continue",
        "cancelado":    "Cancelled.",
        "invalido":     "Invalid option.",
        "mantido":      "Kept.",
        "novo_valor":   "New value (Enter=keep, ?=help): ",
        "confirmar":    "Confirm? (y/n): ",
        "conf_anal":    "This changes results — confirm? (y/n): ",
        "opt_sim":      "Yes",
        "opt_nao":      "No",
        "escolha_num":  "Number or Enter=keep: ",
        "ajuste_nivel": "Automatically adjusted (not relevant to this level): {campos}",
        "atualizado":   "Updated: {campo} = {valor}",
        "chk_dados":    "Data loaded",
        "chk_csv":      "CSV located",
        "chk_leak":     "Anti-leakage active",
        "chk_leak_high":   "Grouped validation by physical sample (mae_id) — anti-leakage ACTIVE",
        "chk_leak_medium": "Grouped validation by the group YOU declared (not verifiable) — MEDIUM guarantee",
        "chk_leak_none":   "NO grouped validation: there is no group key (StratifiedKFold). Replicates may land on both sides; treat metrics as exploratory",
        "chk_leak_verificar": "Anti-leakage depends on the folder structure (the real guarantee is declared after reading the data)",
        "aviso_sem_grupo": "WARNING: no group key (coluna_grupo/arquivo_grupos): validation is NOT grouped and the result has NO protection against replicate leakage.",
        "conf_sem_grupo": "Run WITHOUT grouped validation? Metrics will be exploratory (y/n): ",
        "chk_saida":    "Output folder defined",
        "chk_hw":       "Compatible hardware",
        "chk_preproc":  "Preprocessing defined",
        "chk_err_dados":"Data folder not found",
        "chk_err_csv":  "CSV file not found",
        "chk_err_leak": "GroupKFold DISABLED — leakage risk",
        "chk_warn_hw":  "Low RAM with heavy modules active",
        "chk_tempo":    "Estimated time",
        "chk_descarte": "{n} spectra will be DISCARDED (incompatible spectral range)",
        "chk_orfaos":   "{n} samples without mae_id — enter WITHOUT anti-leakage protection",
        "chk_grupos":   "{n} replicate groups (mae_id)",
        "chk_dica_jobs": "(n_jobs_permutation=1 — raising it to 4 cuts the time without changing results)",
        "chk_modo":     "Mode",
        "chk_prescan_erro": "pre-scan unavailable ({erro})",
        "hw_alto":      "High Performance",
        "hw_medio":     "Medium Performance",
        "hw_basico":    "Basic Performance",
        "hw_limitado":  "Limited",
        "hw_rec":       "Recommended profile",
        "hw_cpu":       "Physical CPUs",
        "hw_threads":   "Logical threads",
        "hw_ram_total": "Total RAM",
        "hw_ram_livre": "Free RAM",
        "hw_disco":     "Free disk",
        "exec_inicio":  "Starting analysis...",
        "exec_leitura": "Reading spectra",
        "exec_preproc": "Preprocessing",
        "exec_pca":     "PCA + HCA",
        "exec_plsda":   "PLS-DA",
        "exec_opls":    "OPLS-DA",
        "exec_dds":     "DD-SIMCA",
        "exec_valid":   "Statistical validation",
        "exec_relat":   "Reports and figures",
        "exec_bench":   "Benchmark (SVM/RF/XGB)",
        "exec_mc":      "Monte Carlo CV",
        "exec_concluido": "Analysis completed",
        "exec_erro":    "Pipeline error",
        "exec_saida":   "Results saved in",
        "exec_interrompido": "Interrupted by user",
        "exec_objetivo": "Objective",
        "exec_eta":      "Estimated time remaining",
        "exec_figuras":  "Figures generated",
        "exec_avisos":   "Warnings",
        "exec_sem_avisos": "No warnings so far",
        "res_tecnica":  "Technique",
        "res_preproc":  "Preprocessing",
        "res_modelo":   "Main model",
        "res_lvs":      "Max. latent variables",
        "res_valid":    "Validation",
        "res_perm":     "Permutations",
        "res_opls":     "OPLS-DA",
        "res_dds":      "DD-SIMCA",
        "res_bench":    "Benchmark",
        "res_mc":       "Monte Carlo",
        "res_shap":     "SHAP",
        "res_dpi":      "Resolution (DPI)",
        "res_fmt":      "Figure format",
        "res_nivel":    "Level",
        "res_tag":      "Run ID",
        "tag_atual":    "Current ID",
        "tag_auto":     "(automatic)",
        "tag_novo":     "New ID (Enter=keep, -=clear): ",
        "tag_limpo":    "ID cleared — next run uses timestamp.",
        "perf_tempo":   "Time",
        "perf_uso":     "Best for",
        "cod_cadastrar":"Register new code",
        "cod_listar":   "List all codes",
        "cod_novo_cod": "Code (2-4 uppercase letters, e.g. MAN): ",
        "cod_novo_esp": "Species name for '{cod}': ",
        "cod_salvo":    "Registered: {cod} = {esp}",
        "cod_invalido": "Invalid code. Use 2-4 uppercase letters.",
        "viz_paleta":   "Color Palette",
        "viz_fonte":    "Font Size",
        "viz_grid":     "Grid",
        "viz_alpha":    "Point Transparency",
        "pip_sem_dados": "Fix data folder before running.",
        "g_prefixo":    "Guaraci:",
    },
}

_CHAVE_TELA = {"projeto": "projeto", "dados": "dados", "preproc": "preproc", "modelagem": "modelo",
               "validacao": "validacao", "avancado": "avancado", "viz": "visualizacao", "tecnica": "tecnica",
               "codigos": "codificacao", "hardware": "hardware", "perfis": "perfis", "predicao": "predicao",
               "hsi": "hsi", "planejamento": "planejamento", "selecao_amostras": "selecao_amostras",
               "auditoria": "auditoria", "tecnicas_avancadas": "tecnicas_avancadas", "ajuda": "ajuda",
               "sobre": "sobre"}


def _t(key: str, **kw) -> str:
    # Titulo ("t_") e descricao ("d_") de tela vem do catalogo (fonte unica).
    if key[:2] in ("t_", "d_") and key[2:] in _CHAVE_TELA:
        t = _cat.tela(_CHAVE_TELA[key[2:]])
        return t["nome" if key[0] == "t" else "faz"][1 if _lang() == "EN" else 0]
    s = _I18N[_lang()].get(key, key)
    if kw:
        try:
            s = s.format(**kw)
        except (KeyError, IndexError):
            pass
    return s

# ---------------------------------------------------------------------------
# Identidade visual — icones e cores por risco
# ---------------------------------------------------------------------------
_RISK_HEX  = {"VISUAL": PF, "ANALITICO": PA, "AVANCADO": PR}
_RISK_ICON = {"VISUAL": "●", "ANALITICO": "◆", "AVANCADO": "▲"}
_RISK_MARK = {"VISUAL": "○", "ANALITICO": "◆", "AVANCADO": "▲"}  # icon inline

def _risco_hex(key: str) -> str:
    return _RISK_HEX.get(_RISK_CLASS.get(key, "ANALITICO"), PA)

def _risco_icon(key: str) -> str:
    return _RISK_ICON.get(_RISK_CLASS.get(key, "ANALITICO"), "◆")

# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------
def modo_linear() -> bool:
    """R5-S1 (revisao cega 2026-09-26): modo para leitor de tela. Com
    `GUARACI_MODO_LINEAR=1` a tela NAO e' limpa a cada acao (o leitor nao
    reanuncia ~50 linhas) e as caixas perdem os caracteres de desenho."""
    return os.environ.get("GUARACI_MODO_LINEAR", "").strip() in ("1", "sim", "true", "yes")


def _ativar_modo_linear() -> None:
    """Troca as bordas de caixa usadas pela CLI por espacos (so' neste processo)."""
    from rich.box import Box
    vazio = Box("    \n    \n    \n    \n    \n    \n    \n    \n")
    for nome in ("ROUNDED", "DOUBLE", "HEAVY", "SIMPLE", "SQUARE", "MINIMAL"):
        setattr(rbox, nome, vazio)


def _cls() -> None:
    # Limpa a tela: string constante, nenhum dado do usuario chega aqui.
    if modo_linear():
        return
    if os.name == "nt":
        console.clear()      # sequencia ANSI pelo rich: `os.system("cls")` abre um cmd.exe a cada tela
        return
    os.system("clear")  # nosec B605

# R5-17 (revisao cega 2026-09-26): em B/K/J/U/T/9, "?" virava caminho ou nome de
# classe. A tela aberta registra aqui QUEM desenha a ajuda dela; `_input`/`_ask`
# mostram o painel de ajuda quando a resposta e' "?" e perguntam de novo.
# Ajuda unificada (2026-11-06): o valor e' uma funcao que imprime o MESMO painel
# de toda tecla "?" (`_painel_ajuda`); um dict {PT, EN} de texto ainda e' aceito.
# `_abrir_tela_uma_vez` restaura o valor anterior ao sair (antes ficava o da
# ultima tela aberta e o "?" do menu de grupo mostrava a ajuda de outra tela).
_AJUDA_TELA: Any = None
_AJUDAS_TELAS: Dict[str, Any] = {
    _t_["tecla"].upper(): (lambda _tid=_tid_: console.print(_painel_tela(_tid)))
    for _tid_, _t_ in _cat.TELAS.items()}


def _foto_cfg(cfg: Config) -> Dict[str, str]:
    """Estado comparavel do Config (R5-22: sair com edicoes nao salvas avisava
    so' DEPOIS de sair)."""
    return {k: repr(v) for k, v in vars(cfg).items()}


def _confirmar_saida(cfg: Config, foto: Dict[str, str]) -> bool:
    """True = pode sair. Com edicoes nao salvas, pergunta antes."""
    mudou = [k for k, v in _foto_cfg(cfg).items() if foto.get(k) != v]
    if not mudou:
        return True
    is_pt = _lang() == "PT"
    console.print(
        f"  [{PA}]{len(mudou)} ajuste(s) de menu ainda nao gravado(s).[/{PA}]" if is_pt else
        f"  [{PA}]{len(mudou)} menu change(s) not saved yet.[/{PA}]")
    r = _input("  [1] gravar em config.yaml e sair  [2] sair sem gravar  [Enter] voltar: "
               if is_pt else
               "  [1] save to config.yaml and quit  [2] quit without saving  [Enter] back: ").strip().upper()
    if r in ("1", "G"):
        try:
            _USER_DIR.mkdir(parents=True, exist_ok=True)
            _save_config(cfg, str(_CFG_PATH))
            console.print(f"  [g]✓ {_CFG_PATH}[/g]")
        except OSError as e:
            console.print(f"  [{PR}]{escape(str(e))}[/{PR}]")
            return False
        return True
    return r in ("2", "S")


def _mostrar_ajuda_tela() -> bool:
    """"?" num prompt: desenha o painel de ajuda da tela atual (sem pausa; o
    prompt volta logo abaixo). False = nenhuma tela registrou ajuda."""
    if not _AJUDA_TELA:
        return False
    if callable(_AJUDA_TELA):
        _AJUDA_TELA()
        return True
    L = 1 if _lang() == "EN" else 0
    console.print(_painel_ajuda("Ajuda" if L == 0 else "Help",
                                [(_ROT_AJUDA["faz"][L], _AJUDA_TELA.get(_lang(), _AJUDA_TELA.get("PT", "")), PW)]))
    return True


def _definir_ajuda(tid: str, opcoes: Any = None) -> None:
    """A tela `tid` passa a responder "?" com o painel dela; `opcoes` (lista de
    (tecla, nome, faz) ou funcao que a devolve) sao as escolhas que ela mostra."""
    global _AJUDA_TELA

    def _mostrar() -> None:
        ops = opcoes() if callable(opcoes) else opcoes
        if not ops:              # tela sem lista (pergunta de valor): o painel da tela direto
            console.print(_painel_tela(tid, opcoes=[]))
            return
        _ajuda_por_escolha(ops, lambda: console.print(_painel_tela(tid, ops)), pausar=False, tid=tid)
    _AJUDA_TELA = _mostrar


_LEITURAS = [0]       # quantas leituras de teclado ja' houve (a tela de passos so' reabre se leu algo)


def _input(msg: str = "", default: str = "", ajuda: bool = True) -> str:
    """Le uma linha. `ajuda=False`: o "?" volta para quem chamou (menus que
    tratam o "?" com a lista numerada deles)."""
    _LEITURAS[0] += 1
    try:
        while True:
            r = input(msg).strip()
            if r == "?" and ajuda and _mostrar_ajuda_tela():
                continue
            return r
    except (EOFError, KeyboardInterrupt):
        _VOLTOU[0] = True     # fim da entrada ou Ctrl+C: a tela de passos nao reabre (com stdin fechado reabria sem fim)
        return default

def _ask(prompt_markup: str = "", default: str = "", ajuda: bool = True) -> str:
    """Le entrada exibindo um prompt COM markup Rich (cores) na mesma linha.

    Use esta funcao sempre que o prompt tiver cores/teclas coloridas — o
    input() puro nao interpreta markup e o exibiria literal (ex.: [#B8963E]).
    """
    _LEITURAS[0] += 1
    try:
        while True:
            if prompt_markup:
                console.print(prompt_markup, end="")
            r = input().strip()
            if r == "?" and ajuda and _mostrar_ajuda_tela():
                continue
            return r
    except (EOFError, KeyboardInterrupt):
        _VOLTOU[0] = True     # fim da entrada ou Ctrl+C: a tela de passos nao reabre (com stdin fechado reabria sem fim)
        return default

def _carregar_modelo_cli(cam_modelo: str, is_pt: bool):
    """Carrega um .joblib ja confirmado pelo operador (confiar=True). Se o
    modelo nao foi assinado por esta instalacao (ex.: veio de outra maquina),
    pergunta se o operador quer assina-lo -- decisao humana explicita; falha
    de INTEGRIDADE (hash divergente) nunca e' oferecida para assinatura."""
    import guaraci.predicao as _p
    try:
        return _p.load_model(cam_modelo, confiar=True)
    except _p.ModeloNaoAssinadoError as e:
        console.print(f"  [{PR}]{escape(str(e))}[/{PR}]")
        pergunta = ("Voce conferiu a origem deste arquivo e quer assina-lo "
                    "com a chave local? (s/n)" if is_pt else
                    "Did you verify where this file came from and want to "
                    "sign it with the local key? (y/n)")
        if _ask(f"  [{PA}]{pergunta}[/{PA}] ").strip().lower() not in (
                "s", "y", "sim", "yes"):
            raise
        _p.assinar_modelo(cam_modelo)
        return _p.load_model(cam_modelo, confiar=True)


def _pause(msg: str = "") -> None:
    lbl = msg or _t("continuar")
    console.print(f"  [{PM}][{escape(lbl)}][/{PM}]", end="")
    _LEITURAS[0] += 1
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        _VOLTOU[0] = True

def _nome_campo(key: str) -> str:
    return _FIELD_NAMES.get(key, {}).get(_lang(), key)

# Traducao dos nomes de nivel APENAS para exibicao. `pq.NIVEL_NOME` (em
# config.py) segue sendo a fonte unica em portugues, porque tambem alimenta o
# `resumo_modelo.txt` — traduzir la' mudaria um ARQUIVO DE SAIDA conforme o
# idioma da interface, o que quebraria a comparabilidade entre execucoes.
# Mesmo padrao ja' usado por _DDSIMCA_DISPLAY.
_NIVEL_NOME_EN = {
    "N1": "Species classification",
    "N2": "Pure vs. adulterated discrimination",
    "N3": "Adulterant content quantification",
}


def _rotulo_opcao(key: str, op: Any) -> str:
    """Nome amigavel de um valor de opcao (so exibicao; o valor gravado no
    config continua o codigo interno, ex.: N1/N2/N3, puros/todos)."""
    if key == "nivel":
        # Lidera com o nome amigavel (P8: aposentar N1/N2/N3 como termo
        # PRIMARIO na UI); o codigo interno fica entre parenteses so' como
        # referencia tecnica, nunca como o rotulo principal exibido ao usuario.
        nome = (_NIVEL_NOME_EN.get(str(op), "") if _lang() == "EN"
                else pq.NIVEL_NOME.get(str(op), ""))
        return f"{nome} ({op})" if nome else str(op)
    if key == "modo_ddsimca":
        return _DDSIMCA_DISPLAY.get(_lang(), {}).get(str(op), str(op))
    if key in ("perfil_matriz", "perfil_tecnica"):
        # Agente 5B: expoe o que hoje e' dado morto (descricao do perfil, e
        # para perfil de tecnica tambem a garantia tipica de agrupamento) --
        # sem isso o usuario via' so' o nome de arquivo (ex. "bancada"), sem
        # saber o que ele significa, tendo que sair da sessao e rodar
        # `guaraci perfis` num terminal separado so' pra ler a descricao.
        if not str(op):
            return "(nao declarado)" if _lang() == "PT" else "(not declared)"
        try:
            from guaraci.perfil_matriz import load_profile
            p = load_profile(str(op))
        except Exception:  # noqa: BLE001 -- rotulo e' so' exibicao; um
            # perfil quebrado/removido nao pode travar a tela de edicao,
            # so' cai pro nome cru (a validacao real acontece ao aplicar).
            return str(op)
        desc = (p.descricao or "").split(" (")[0].strip()
        if _lang() == "EN":
            desc = _DESC_PERFIL_EN.get(str(op), desc)
        # Indicador de cobertura validada (Agente 5B, item pendente):
        # `referencia` ja' e' o dado que distingue perfil validado com dado
        # PUBLICO real (nir_dispersivo_1100_2498nm/nir_transmitancia_8mm tem paper/dataset
        # citado) de perfil so' declarado (visnir_400_1100nm/nir_ft_difuso/bancada/
        # celular/scanner -- referencia vazia, nenhum tem validacao
        # publicada ainda). Nao inventa um campo novo -- so' expoe o que
        # ja' existia sem aparecer em lugar nenhum da UI. "generico" fica
        # de fora do selo -- e' um placeholder neutro, nao uma alegacao.
        selo = ""
        # Perfil embutido (nome, nao caminho) sem referencia e' ponto de
        # partida configuravel, nao alegacao: por regra do projeto ele nao
        # cita conjunto de dados de terceiros (docs/cientifico/numeros_publicos.yaml),
        # entao "nao validado" ali soava como defeito. O ⚠ fica para perfil
        # do usuario que nao cita a sua fonte.
        from pathlib import Path as _P
        import guaraci.perfil_matriz as _pm_mod
        embutido = (_P(_pm_mod.__file__).parent / "perfis_matriz" / f"{op}.yaml").is_file()
        if str(op) != "generico":
            if p.referencia:
                selo = "  ✅"
            elif embutido:
                selo = ("  · perfil de partida" if _lang() == "PT"
                        else "  · starting profile")
            else:
                selo = ("  ⚠ nao validado" if _lang() == "PT"
                        else "  ⚠ not validated")
        if key == "perfil_tecnica" and p.nivel_agrupamento_tipico:
            garantia = ("garantia tipica" if _lang() == "PT" else "typical guarantee")
            return f"{op} — {desc} [{garantia}: {p.nivel_agrupamento_tipico}]{selo}"
        op_txt = _VALOR_EN.get(str(op), str(op)) if (_lang() == "EN" and str(op) == "generico") else str(op)
        return (f"{op_txt} — {desc}{selo}" if desc else f"{op_txt}{selo}")
    return _texto_para_exibir(op)


#: Descricao dos perfis embutidos em ingles (o YAML guarda a descricao em portugues).
_DESC_PERFIL_EN = {
    "bancada": "Digital colorimetry with a benchtop camera (controlled acquisition)",
    "celular": "Digital colorimetry with a phone photo (informal acquisition)",
    "generico": "Unspecified matrix -- no matrix preset applied",
    "nir_dispersivo_1100_2498nm": "Dispersive NIR, 1100-2498 nm",
    "nir_ft_difuso": "Diffuse-reflectance FT-NIR",
    "nir_transmitancia_8mm": "Transmittance NIR, 8 mm optical path",
    "scanner": "Digital colorimetry with a flatbed scanner (documents/flat slides)",
    "visnir_400_1100nm": "Vis-NIR, 400-1100 nm",
}


# Valores de opcao gravados em portugues no config.yaml; so' o que a tela mostra muda com o idioma.
_VALOR_EN = {
    "padrao": "default", "todos": "all", "aleatoria": "random", "excluir": "exclude",
    "substituir": "replace", "classificacao": "classification", "exploratorio": "exploratory",
    "quantificacao": "quantification", "generico": "generic", "bancada": "benchtop",
    "celular": "phone", "Personalizado": "Custom", "puros": "pure only",
    # modos de entrada e cenarios (o codigo gravado continua o mesmo)
    "imagem": "image", "multielementar": "multi-element", "sintetico": "synthetic", "rmn": "nmr",
    "classico": "classic", "sessao": "session", "lista": "list",
    # paleta, transparencia
    "qualitativo": "qualitative", "daltonismo_safe": "colorblind_safe", "cinza": "grayscale",
    "publicacao": "publication", "escuro": "dark", "baixo": "low", "medio": "medium", "alto": "high",
    "sim": "yes", "nao": "no",
}

#: Textos de faixa/padrao do catalogo de opcoes escritos so' em portugues (exibicao em ingles).
_FRASE_EN = {
    "Nome de perfil embutido ou caminho de YAML": "Built-in profile name or YAML path",
    "ver lista de perfis de tecnica": "see the list of technique profiles",
    "nome da coluna": "column name", "caminho .csv": ".csv path",
    "nome de coluna do CSV": "CSV column name", "nome de coluna do CSV ou vazio": "CSV column name or empty",
    "elemento presente no arquivo ou vazio": "element present in the file, or empty",
    "somente puras | todas as amostras  (EN: only pure | all samples)": "only pure | all samples",
    "somente puras (autenticacao)": "only pure (authentication)",
    "Escolha na lista / Choose from list": "Choose from list",
    "lista separada por virgula": "comma-separated list",
    "Texto livre (sem espacos recomendado)": "Free text (no spaces recommended)",
    "Texto livre (sem espacos)": "Free text (no spaces)",
    "Ver menu Codificacao": "See the DX encoding screen",
    "4 valores entre 0 e 1": "4 values between 0 and 1",
    "COD-DD-MM-AAAA_Tn.dx": "COD-DD-MM-YYYY_Tn.dx",
    "sim/nao": "yes/no",
}


#: O caminho inverso: faixas do catalogo escritas em ingles (ou nos dois idiomas) mostradas em
#: portugues na tela PT. "Valid system path" vazava na ajuda em PT (captura do Erley, 2026-11-06).
_FRASE_PT = {
    "Valid system path": "Caminho valido no computador",
    "Valid system path (or empty)": "Caminho valido no computador (ou vazio)",
    "Valid .csv path": "Caminho de um arquivo .csv",
    "Column name or empty": "Nome de coluna ou vazio",
    "Existing column name in CSV": "Nome de uma coluna do CSV",
    "List of class names": "Lista de nomes de classe",
    "Escolha na lista / Choose from list": "Escolha na lista",
    "somente puras | todas as amostras  (EN: only pure | all samples)": "somente puras | todas as amostras",
    "Ver menu Codificacao": "Ver a tela Codificacao DX [2.9]",
    "true | false": "sim | nao", "true/false": "sim | nao", "True": "Sim", "False": "Nao",
}


_TIPO_EN = {"str": "text", "str_opcional": "optional text", "int": "whole number", "float": "number",
            "bool": "yes/no", "choice": "choice", "list": "list", "preproc": "preprocessing", "recorte": "crop"}


def _texto_para_exibir(txt: Any) -> str:
    """Faixa, padrao ou lista de opcoes do catalogo no idioma da interface (so' exibicao)."""
    if isinstance(txt, bool):
        return ("Yes" if txt else "No") if _lang() == "EN" else ("Sim" if txt else "Nao")
    if isinstance(txt, (list, tuple)) and not txt:
        return "(empty)" if _lang() == "EN" else "(vazio)"
    t = str(txt)
    if _lang() != "EN":
        return _FRASE_PT.get(t, t)
    if t in _FRASE_EN:
        return _FRASE_EN[t]
    if t in _VALOR_EN:
        return _VALOR_EN[t]
    if t in ("True", "False"):
        return "Yes" if t == "True" else "No"
    m = _re.fullmatch(r"(-?[\d.,]+) a ([\d.,]+)(, impar)?", t)
    if m:
        return (f"{m.group(1).replace(',', '.')} to {m.group(2).replace(',', '.')}"
                + (", odd" if m.group(3) else ""))
    if t == "0, 1 ou 2":
        return "0, 1 or 2"
    for sep in (" | ", ", "):
        if sep in t:
            return sep.join(_VALOR_EN.get(p, p) for p in t.split(sep))
    return t


#: Mensagens de erro de `config_io._coagir_valor`/`_checar_faixa` (em portugues), so' na exibicao.
_MSG_VALOR_EN = (
    (r"^(.*)=(.*): valor minimo permitido e (.*)$", r"\1=\2: the minimum allowed value is \3"),
    (r"^(.*)=(.*): valor maximo permitido e (.*)$", r"\1=\2: the maximum allowed value is \3"),
    (r"^valor '(.*)' invalido; use (.*)$", r"invalid value '\1'; use \2"),
    (r"^(.*): valor\(es\) (.*) invalido\(s\); use (.*)$", r"\1: invalid value(s) \2; use \3"),
    (r"^(.*): use 4 valores entre 0 e 1 \(x0, y0, x1, y1\) com x0 < x1 e y0 < y1$",
     r"\1: use 4 values between 0 and 1 (x0, y0, x1, y1) with x0 < x1 and y0 < y1"),
    (r"^pre-processamento '(.*)' invalido; use (.*)$", r"invalid preprocessing '\1'; use \2"),
    (r"^Valor invalido: '(.*)'$", r"Invalid value: '\1'"),
    (r"^invalid literal for int\(\) with base 10: (.*)$", r"not a whole number: \1"),
    (r"^could not convert string to float: (.*)$", r"not a number: \1"),
)


def _msg_valor(msg: str) -> str:
    """Erro de valor digitado no idioma da interface."""
    if _lang() == "PT":
        if msg.startswith("invalid literal for int()"):
            return "nao e' um numero inteiro: " + msg.split(":", 1)[1].strip()
        if msg.startswith("could not convert string to float"):
            return "nao e' um numero: " + msg.split(":", 1)[1].strip()
        return msg
    for pad, rep in _MSG_VALOR_EN:
        if _re.match(pad, msg):
            return _re.sub(pad, rep, msg)
    return msg


# Mensagens de `_validar_pasta_dados` (config_io, em portugues: o assistente de erros as reconhece
# pelo texto). Na tela em ingles traduz-se so' na exibicao.
_MSG_VALIDACAO_EN = (
    (r"^pasta nao encontrada: (.*) \(confira o caminho\)$", r"folder not found: \1 (check the path)"),
    (r"^CSV nao encontrado: (.*) \(confira o caminho\)$", r"CSV not found: \1 (check the path)"),
    (r"^arquivo ICP nao encontrado: (.*) \(campo arquivo_csv\)$", r"ICP file not found: \1 (field arquivo_csv)"),
    (r"^informe icp_unidade \(ex\.: mg/kg\) -- obrigatoria no mode multielementar$",
     r"set icp_unidade (e.g. mg/kg) -- required in multielemental mode"),
    (r"^informe coluna_grupo \(a amostra fisica/lote independente\) -- obrigatoria no mode multielementar$",
     r"set coluna_grupo (the independent physical sample/batch) -- required in multielemental mode"),
    (r"^nenhuma imagem em (.*) \(nem nas subpastas\)$", r"no image in \1 (nor in its subfolders)"),
    (r"^nenhum arquivo (.*) em (.*) \(nem nas subpastas\)$", r"no \1 file in \2 (nor in its subfolders)"),
    (r"^pasta (.*) vazia$", r"folder \1 is empty"),
    (r"^nenhum diretorio 'pdata' em (.*) \(esperado por amostra RMN\)$", r"no 'pdata' directory in \1 (expected per NMR sample)"),
    (r"^nenhuma pasta '\.D'/'\.raw' em (.*) \(esperado por amostra HPLC/GC\)$",
     r"no '.D'/'.raw' folder in \1 (expected per HPLC/GC sample)"),
    (r"^nenhum arquivo \.dat em (.*)$", r"no .dat file in \1"),
    (r"^nenhum arquivo \.dx em (.*) \(nem nas subpastas\)$", r"no .dx file in \1 (nor in its subfolders)"),
)


def _msg_validacao(msg: str) -> str:
    """Mensagem de validacao da fonte de dados no idioma da interface."""
    if _lang() != "EN":
        return msg
    for padrao, modelo in _MSG_VALIDACAO_EN:
        if _re.match(padrao, msg):
            return _re.sub(padrao, modelo, msg)
    return msg


def _valor_para_exibir(val: Any) -> str:
    """Texto do valor de um campo na tela, no idioma da interface (o valor gravado nao muda)."""
    txt = str(val)
    return _VALOR_EN.get(txt, txt) if _lang() == "EN" else txt


def _get_val(cfg: Config, key: str) -> Any:
    spec = _SPEC_BY_KEY.get(key)
    if spec is None:
        return getattr(cfg, key, "?")
    raw = _attr_para_yaml(spec, cfg)
    if key == "modo_ddsimca":
        return _DDSIMCA_DISPLAY.get(_lang(), {}).get(str(raw), raw)
    if key == "nivel":
        return _rotulo_opcao(key, raw)
    return raw

def _set_val(cfg: Config, key: str, raw: str) -> None:
    spec = _SPEC_BY_KEY[key]
    if key == "modo_ddsimca":
        interno = _DDSIMCA_INPUT.get(raw.lower().strip())
        if interno is None:
            raise ValueError(f"Valor invalido: '{raw}'")
        raw = interno
    valor = _coagir_valor(spec, raw)
    setattr(cfg, spec["attr"], valor)

def _cfgv(cfg: Config, key: str, default: Any = None) -> Any:
    """Le um valor do Config pela KEY do _CONFIG_SPEC, resolvendo o atributo real.

    Evita o erro comum de usar `getattr(cfg, "benchmark")` quando o atributo
    real e `run_benchmark`. Sempre use esta funcao para ler config por key.
    """
    spec = _SPEC_BY_KEY.get(key)
    attr = spec["attr"] if spec else key
    return getattr(cfg, attr, default)


# _count_dx importada de guaraci.cli_logic (item 19) — ver topo do arquivo.

# ---------------------------------------------------------------------------
# MENSAGENS DINAMICAS — boas-vindas, despedida, pausa
# ---------------------------------------------------------------------------
_BOAS_VINDAS = {
    "PT": [
        "Bem-vindo ao GUARACI. Os seus dados aguardam analise. ☀",
        "GUARACI inicializado. Ciencia comeca agora. ☀",
        "Ola, pesquisador. O GUARACI esta pronto para revelar padroes. ☀",
        "Sistema iniciado. Que a analise seja precisa e os resultados, claros. ☀",
    ],
    "EN": [
        "Welcome to GUARACI. Your data awaits analysis. ☀",
        "GUARACI initialized. Science starts now. ☀",
        "Hello, researcher. GUARACI is ready to reveal patterns. ☀",
        "System started. May the analysis be precise and results clear. ☀",
    ],
}

_DESPEDIDAS = {
    "PT": [
        "Ate logo! Que seus modelos tenham alta acuracia. ☀",
        "Encerrando GUARACI. Boas analises, pesquisador!",
        "Saindo. Resultados de execucoes concluidas ficam na pasta de saida; ajustes de menu nao rodados nao sao salvos. ☀",
        "Ate a proxima sessao. A ciencia continua! ☀",
    ],
    "EN": [
        "Goodbye! May your models have high accuracy. ☀",
        "Closing GUARACI. Good analyses, researcher!",
        "Exiting. Results of finished runs are in the output folder; menu edits you did not run are not saved. ☀",
        "Until next session. Science continues! ☀",
    ],
}

_PAUSAS_CAFE = {
    "PT": [
        "Analise em andamento — otimo momento para uma pausa. ☕",
        "Processando espectros... aproveite para um intervalo. ☕",
        "O pipeline esta trabalhando. Voce merece uma pausa. ☕",
        "Modelos sendo calculados. Hora de esticar as pernas? ☕",
    ],
    "EN": [
        "Analysis in progress — a good moment for a short break. ☕",
        "Processing spectra... enjoy a short break. ☕",
        "Pipeline running. You deserve a pause. ☕",
        "Models being computed. Time to stretch your legs? ☕",
    ],
}

def _exibir_boas_vindas() -> None:
    """Mensagem aleatoria de boas-vindas, numa linha, logo depois do primeiro menu.

    Antes era um painel com pausa de 0,4 s ANTES do menu (puramente estetica: o menu
    redesenhava por cima). A pausa foi tirada: o criterio de tempo ate o menu e'
    <= 1 s, e a mensagem agora fica visivel junto do menu em vez de piscar."""
    import random
    msg = random.choice(_BOAS_VINDAS[_lang()])
    console.print(f"  [{PF}]{escape(msg)}[/{PF}]")


def _exibir_despedida() -> None:
    """Exibe mensagem de despedida ao sair do programa."""
    import random
    import time as _time
    msg = random.choice(_DESPEDIDAS[_lang()])
    console.print()
    console.print(Panel(
        Align.center(Text(f"\n  {msg}\n", style=f"italic {PS}")),
        border_style=PF, box=rbox.ROUNDED, padding=(0, 2), width=_W()
    ))
    _time.sleep(0.6)

def _sugerir_pausa() -> None:
    """Exibe sugestao de pausa durante execucoes longas."""
    import random
    msg = random.choice(_PAUSAS_CAFE[_lang()])
    console.print(f"\n  [{PA}]{msg}[/{PA}]")

# ---------------------------------------------------------------------------
# VALIDACAO DE INTEGRIDADE — faixas e paleta antes de rodar
# ---------------------------------------------------------------------------
def _validate_ranges(cfg: Config) -> list:
    """Wrapper fino: le faixa_min/max do Config e delega a validacao pura
    a `guaraci.cli_logic.validate_ranges` (testada)."""
    f_min = _cfgv(cfg, "faixa_min_cm", 400)
    f_max = _cfgv(cfg, "faixa_max_cm", 4000)
    return _validar_faixas_puro(f_min, f_max)

def _sincronizar_dpi(cfg: Config) -> None:
    """Garante que cfg.dpi reflita o visual_config.json antes de rodar."""
    vcfg = _carregar_visual_cfg()
    dpi_v = vcfg.get("dpi")
    if dpi_v:
        try:
            setattr(cfg, "dpi", int(dpi_v))
        except (TypeError, ValueError):
            pass

# ---------------------------------------------------------------------------
# ASSISTENTE GUARACI — tecla G em qualquer tela
# ---------------------------------------------------------------------------
def _guaraci_revisar_config(cfg: Config) -> None:
    """Revisao da configuracao atual: os valores principais e os achados das
    regras de configuracao do registro unico do assistente (assistente_regras),
    as mesmas do pre-voo e de `guaraci checar`. Nenhuma regra escrita a mao aqui."""
    lang = _lang(); en = lang == "EN"
    linhas = [f"[{PA}]{'Current configuration:' if en else 'Configuracao atual detectada:'}[/{PA}]", ""]

    preproc  = _cfgv(cfg, "pre_processamento",     "msc_sg_mc")
    max_lvs  = _cfgv(cfg, "max_lvs",               40)
    mc       = _cfgv(cfg, "monte_carlo",           False)
    shap     = _cfgv(cfg, "shap_benchmark",        False)
    n_perm   = _cfgv(cfg, "n_permutacoes",         200)
    opls     = _cfgv(cfg, "opls_da",               True)
    dds      = _cfgv(cfg, "ddsimca",               True)
    holdout  = _cfgv(cfg, "holdout_fracao",        0.20)
    bench    = _cfgv(cfg, "benchmark",             False)

    inf = f"[{PS}]ℹ[/{PS}]"
    try:
        holdout_pct = f"{int(round(float(holdout) * 100))}%"
    except (TypeError, ValueError):
        holdout_pct = str(holdout)
    linhas.append(f"  · Pre-proc: {escape(str(preproc))}")
    linhas.append(f"  · max_lvs = {escape(str(max_lvs))}")
    linhas.append(f"  · n_permutacoes = {escape(str(n_perm))}")
    linhas.append(f"  · holdout = {holdout_pct}")
    if mc:
        linhas.append(f"  {inf} " + ("Monte Carlo on: more robust analysis, longer run." if en
                                     else "Monte Carlo ativo: analise mais robusta, tempo maior."))
    if shap:
        linhas.append(f"  {inf} " + ("SHAP on: more interpretability, longer run." if en
                                     else "SHAP ativo: interpretabilidade aumentada, tempo maior."))
    if bench:
        linhas.append(f"  {inf} " + ("Benchmark SVM/RF/XGB on." if en else "Benchmark SVM/RF/XGB ativo."))
    if not opls:
        linhas.append(f"  [{PM}]ℹ {'OPLS-DA off.' if en else 'OPLS-DA desativado.'}[/{PM}]")
    if not dds:
        linhas.append(f"  [{PM}]ℹ {'DD-SIMCA off.' if en else 'DD-SIMCA desativado.'}[/{PM}]")

    # Regras de configuracao do registro (sem carregar dados).
    linhas += ["", f"[{PA}]{'Assistant rules (configuration):' if en else 'Regras do assistente (configuracao):'}[/{PA}]"]
    try:
        from guaraci import assistente_regras as _ar
        achados = _ar.avaliar(cfg, None, None, lang, momentos=["config"])
    except Exception as e:  # noqa: BLE001 -- a tela de revisao nao pode cair por falha do assistente
        achados = []
        linhas.append(f"  [{PR}]{escape(str(e))}[/{PR}]")
    estilo = {"critico": (PR, "✖"), "aviso": (PA, "⚑"), "info": (PS, "ℹ")}
    if not achados:
        linhas.append(f"  [{PG}]✓[/{PG}] " + ("No problem found by the assistant rules." if en
                                               else "Nenhum problema encontrado pelas regras do assistente."))
    for a in achados:
        cor, simb = estilo.get(a.severidade, (PW, "-"))
        linhas.append(f"  [{cor}]{simb}[/{cor}] {escape(a.mensagem)}")
        linhas.append(f"     [{PM}]{'What to do' if en else 'O que fazer'}: {escape(a.fazer)}[/{PM}]")
        if a.campo:
            valor = str(a.valor).lower() if isinstance(a.valor, bool) else a.valor
            linhas.append(f"     [{PM}]{'Accept' if en else 'Aceitar'}: {escape(f'{a.campo}: {valor}')}[/{PM}]")
    linhas.append(f"  [{PM}]" + ("Data rules: screen [6.5] or `guaraci checar CONFIG --dados`." if en
                                 else "Regras de dados: tela [6.5] ou `guaraci checar CONFIG --dados`.") + f"[/{PM}]")

    _titulo_rev = ("Revisao da Configuracao" if _lang() == "PT"
                   else "Configuration Review")
    console.print(Panel(
        "\n".join(linhas),
        title=f"[{PA}]☀ {_titulo_rev}[/{PA}]",
        border_style=PA, box=rbox.ROUNDED, padding=(1, 2), width=_W()
    ))
    _pause()


def _menu_checar(cfg: Config) -> None:
    """Tela [6.5]: o mesmo que `guaraci checar` pelas telas. Pre-voo com a
    configuracao atual e os dados (carregados agora, qualquer tamanho) e, se
    houver uma execucao, a leitura do resultado e a ficha anti-vazamento
    gravada em <execucao>/Relatorios/ficha_vazamento.md."""
    lang = _lang(); en = lang == "EN"
    gate_lbl = "[Enter] Check now" if en else "[Enter] Checar agora"
    while True:
        _cabecalho_tela("checar", cfg)
        _rodape_tela()
        console.print()
        raw_gate = _ask(f"  [{PM}]{gate_lbl}[/{PM}]: ").strip()
        if raw_gate:
            if _tecla_comum(raw_gate, "checar", cfg) == "voltar":
                return
            continue
        break
    console.print()
    with console.status(f"[{PA}]{'Loading data for the pre-flight...' if en else 'Carregando os dados para o pre-voo...'}[/{PA}]"):
        achados, dados = _preflight(cfg, lang, forcar_dados=True)
    _mostrar_achados(achados, lang, titulo=("Pre-flight: configuration and data" if en else "Pre-voo: configuracao e dados"),
                     dados=dados)
    console.print()
    ultima = str(getattr(cfg, "output_folder", "") or "")
    sugestao = ultima if ultima and os.path.isdir(ultima) else ""
    rotulo = ("Run folder to read the result and write the leakage sheet" if en
              else "Pasta de uma execucao para ler o resultado e gravar a ficha anti-vazamento")
    dica = (f" (Enter = {sugestao})" if sugestao else (" (Enter = skip)" if en else " (Enter = pular)"))
    pasta = _input(f"  {rotulo}{dica}: ").strip().strip('"').strip("'") or sugestao
    if pasta:
        if not os.path.isdir(pasta):
            console.print(f"  [{PR}]{'Folder not found' if en else 'Pasta nao encontrada'}: {escape(pasta)}[/{PR}]")
        else:
            cfg_pasta = copy.copy(cfg)
            cfg_pasta.output_folder = pasta
            pos = _assistente_pos_execucao(cfg_pasta, dados, lang)
            if pos:
                _mostrar_achados(pos[1], lang, titulo=("Result check" if en else "Leitura do resultado"))
                console.print(f"  [{PM}]{'Leakage sheet' if en else 'Ficha anti-vazamento'}: {escape(pos[0])}[/{PM}]")
    console.print(f"  [{PM}]" + escape("Same check without the screens: guaraci checar CONFIG [--dados] [--resultado FOLDER] [--ficha]" if en
                                       else "O mesmo sem as telas: guaraci checar CONFIG [--dados] [--resultado PASTA] [--ficha]")
                  + f"[/{PM}]")
    _pause()



def _secoes_do_catalogo() -> List[Tuple[str, str, str]]:
    """(tecla, nome, faz) de cada tela do catalogo, menos o proprio assistente."""
    L = 1 if _lang() == "EN" else 0
    return [(t["tecla"], t["nome"][L], t["faz"][L]) for t in _cat.telas() if t["id"] != "assistente"]


def _guaraci_navegar_secoes(cfg: Config) -> None:
    """Lista as telas do catalogo (tecla, nome, o que faz) e mostra a escolhida.
    Uma tela nova entra aqui sozinha, sem lista paralela."""
    lang = _lang()
    secoes = {k: (nome, faz) for k, nome, faz in _secoes_do_catalogo()}
    console.print()
    t = Table(show_header=False, box=rbox.SIMPLE, padding=(0, 1))
    t.add_column("N", style=PA, width=4)
    t.add_column("Secao", style=PW, width=22)
    t.add_column("Descricao", style=PM)
    for k, (nome, desc) in secoes.items():
        t.add_row(k, nome, desc)
    console.print(t)
    raw = _ask(f"  [{PA}]{'Selecione ([0] voltar)' if _lang() == 'PT' else 'Select ([0] back)'}: [/{PA}]").strip()
    if raw in secoes:
        nome, desc = secoes[raw]
        console.print(Panel(
            f"[{PW}]{desc}[/{PW}]",
            title=f"[{PA}]{nome}[/{PA}]",
            border_style=PF, box=rbox.ROUNDED, padding=(0, 2), width=_W()
        ))
        _pause()


def _guaraci_diagnosticar(cfg: Config) -> None:
    """Diagnostica o dataset carregado -- reaproveita `run_audit`, o MESMO
    motor da aba U (Auditoria de delineamento), so' apresentado dentro do
    assistente. So' roda sob demanda (nao em toda abertura do assistente,
    que seria lento e a maioria das aberturas e' so' pra ajuda pontual).

    Regra dura (Agente 6): nunca inventa numero, nunca esconde ressalva --
    todo achado abaixo vem de uma funcao ja testada (`run_audit`,
    `achievable_alpha`, `n_minimum_for_alpha`), nunca de um calculo novo
    so' pra esta tela.
    """
    lang = _lang(); is_pt = lang == "PT"
    status_msg = ("Carregando dados e diagnosticando..." if is_pt
                  else "Loading data and diagnosing...")
    try:
        with console.status(f"[{PA}]{status_msg}[/{PA}]"):
            wavenumbers, X_raw, rotulos, conc, mae_id, _metadados = pq.load_data(cfg)
            X_raw, wavenumbers, rotulos, conc, mae_id, _relatorio = pq.validate_input(
                X_raw, wavenumbers, rotulos, conc, mae_id)
            from guaraci.auditoria_delineamento import run_audit
            achados = run_audit(X_raw, wavenumbers, rotulos, cfg, conc, mae_id)
    except Exception as e:  # noqa: BLE001 -- dado externo pode falhar de
        # varias formas (parsing, faixa espectral vazia, etc.) -- reportar
        # a mensagem, nunca stack trace cru numa ferramenta interativa
        # (mesmo tratamento de _menu_audit).
        console.print(f"  [{PR}]{'Erro ao carregar dados' if is_pt else 'Error loading data'}: "
                      f"{escape(str(e))}[/{PR}]")
        _pause(); return

    cores = {"ok": PG, "aviso": PA, "critico": PR, "silenciado": PM}
    console.print()
    for a in achados:
        cor = cores.get(a.severidade, PW)
        console.print(f"  [{cor}]{a.severidade.upper():>10}[/{cor}]  "
                      f"[{PW}]{escape(a.nome)}[/{PW}]: {escape(a.mensagem)}")

    n_criticos = sum(1 for a in achados if a.severidade == "critico")
    n_avisos = sum(1 for a in achados if a.severidade == "aviso")
    resumo_lbl = (f"{n_criticos} critico(s), {n_avisos} aviso(s) de {len(achados)} checagem(ns)."
                  if is_pt else
                  f"{n_criticos} critical, {n_avisos} warning(s) out of {len(achados)} check(s).")
    cor_resumo = PR if n_criticos else (PA if n_avisos else PG)
    console.print()
    console.print(Panel(
        Text.from_markup(f"  {resumo_lbl}"),
        border_style=cor_resumo, box=rbox.ROUNDED, padding=(0, 2),
    ))

    # Sugerir+executar (Agente 6, Fase 1 -- so' este 1 caso, como prova do
    # padrao; os demais exemplos do pedido original ficam para uma rodada
    # futura, ver docs/projeto/identidade_visual.md).
    achado_n = next((a for a in achados
                      if a.nome == "n_insuficiente" and a.severidade == "aviso"),
                     None)
    if achado_n is not None and mae_id is not None:
        info = _sugestao_alpha_classe_fraca(rotulos, mae_id)
        if info is not None:
            sugestao = (
                f"💡 Sua classe mais fraca ('{info['classe']}') tem "
                f"{info['n']} sessao(oes) independente(s) → alpha minimo "
                f"alcancavel = {info['alpha_alcancavel']:.2f}. Para "
                f"alpha={info['alpha_ref']} de referencia (gate conformal "
                f"padrao), seriam necessarias {info['n_para_ref']} sessoes "
                "independentes dessa classe."
                if is_pt else
                f"💡 Your weakest class ('{info['classe']}') has "
                f"{info['n']} independent session(s) → minimum achievable "
                f"alpha = {info['alpha_alcancavel']:.2f}. For the "
                f"reference alpha={info['alpha_ref']} (standard conformal "
                f"gate), {info['n_para_ref']} independent sessions of that "
                "class would be needed.")
            console.print()
            console.print(Panel(
                Text.from_markup(f"  [{PA}]{escape(sugestao)}[/{PA}]"),
                border_style=PA, box=rbox.ROUNDED, padding=(0, 2),
            ))
            # Fase 2 (Agente 6): "sugerir" nao para no texto -- oferece a
            # ACAO. Reaproveita plan_from_statistical_target (mesma funcao
            # que _menu_plan chama), nao inventa um calculo novo aqui.
            pergunta = ("Quer ver o plano de coleta para atingir esse alpha? (s/n) "
                        if is_pt else
                        "Want to see the collection plan to reach that alpha? (y/n) ")
            if _ask(f"  [{PA}]{pergunta}[/{PA}]").strip().lower() in ("s", "y", "sim", "yes"):
                from guaraci.plano_coleta import plan_from_statistical_target
                try:
                    classes_unicas = sorted({str(r) for r in rotulos})
                    plano, meta = plan_from_statistical_target(
                        classes_unicas, n_sessoes=2,
                        alpha_conformal=info["alpha_ref"])
                    console.print()
                    console.print(f"  [{PS}]{'n por classe' if is_pt else 'n per class'}: "
                                  f"{meta['n_por_classe']} ({meta['origem']})[/{PS}]")
                    for alerta in plano.alertas:
                        console.print(f"  [{PM}]• {escape(str(alerta))}[/{PM}]")
                except ValueError as e_plano:
                    console.print(f"  [{PR}]{escape(str(e_plano))}[/{PR}]")
    _pause()


def _sugestao_alpha_classe_fraca(rotulos, mae_id) -> Optional[Dict[str, Any]]:
    """Sessoes independentes por classe (mesma funcao `session_from_mae_id`
    que `check_insufficient_n` usa) e o alpha conformal minimo alcancavel
    para a classe mais fraca. Funcao PURA (sem console) para ser testavel
    direto -- usada por `_guaraci_diagnosticar` (Agente 6, sugerir+executar).
    Nao duplica o VEREDITO da auditoria (isso continua em
    `auditoria_delineamento.check_insufficient_n`), so' os numeros
    descritivos que `AuditFinding` nao expoe (so' a mensagem ja formatada).
    """
    import numpy as np
    from guaraci.conformal import achievable_alpha, n_minimum_for_alpha
    from guaraci.dados_io import session_from_mae_id

    rotulos_arr = np.asarray(rotulos, dtype=str)
    sessao = np.array([session_from_mae_id(m) for m in mae_id], dtype=str)
    contagens = {
        classe: len({s for s, r in zip(sessao, rotulos_arr) if r == classe})
        for classe in sorted(set(rotulos_arr))
    }
    if not contagens:
        return None
    classe_fraca, n_fraco = min(contagens.items(), key=lambda kv: kv[1])
    alpha_ref = 0.05
    return {
        "classe": classe_fraca,
        "n": n_fraco,
        "alpha_alcancavel": achievable_alpha(n_fraco),
        "alpha_ref": alpha_ref,
        "n_para_ref": n_minimum_for_alpha(alpha_ref),
    }


def _guaraci_tecnicas() -> None:
    """Lista o catalogo de tecnicas cientificas do GUARACI (Agente 6, item
    d) -- gerado a partir de `technique_registry.REGISTRY`, fonte unica de
    verdade (nunca uma lista escrita a mao so' pra esta tela)."""
    lang = _lang(); is_pt = lang == "PT"
    from guaraci.technique_registry import REGISTRY

    rotulos_categoria = {
        "classificacao_deteccao": ("Classificacao / deteccao", "Classification / detection"),
        "quantificacao": ("Quantificacao", "Quantification"),
        "identificacao_conjunto_aberto": ("Identificacao (conjunto aberto)", "Identification (open set)"),
        "selecao_amostras": ("Selecao de amostras", "Sample selection"),
        "transferencia_calibracao": ("Transferencia de calibracao", "Calibration transfer"),
        "figuras_de_merito": ("Figuras de merito", "Figures of merit"),
        "robustez_linearidade": ("Robustez / linearidade", "Robustness / linearity"),
        "perfis": ("Perfis (matriz / tecnica de aquisicao)", "Profiles (matrix / acquisition technique)"),
        "resolucao_mistura": ("Resolucao de mistura", "Mixture resolution"),
    }
    console.print()
    textos = {t["id"]: t for t in _cat.tecnicas(lang)}     # nome e "use quando" no idioma da interface
    for cat_id, (nome_pt, nome_en) in rotulos_categoria.items():
        entradas = [e for e in REGISTRY if e.categoria == cat_id]
        if not entradas:
            continue
        console.print(f"  [bold {PA}]{nome_pt if is_pt else nome_en}[/bold {PA}]")
        for e in entradas:
            tx_ = textos.get(e.id, {})
            console.print(f"    [{PW}]▸ {escape(tx_.get('nome', e.nome))}[/{PW}]")
            console.print(f"      [{PM}]{escape(tx_.get('use') or e.quando_usar)}[/{PM}]")
        console.print()
    console.print(f"  [{PM}]{len(REGISTRY)} tecnicas cadastradas.[/{PM}]" if is_pt
                  else f"  [{PM}]{len(REGISTRY)} techniques registered.[/{PM}]")
    _pause()


def _faq_metodo_recomendado(cfg: Config, is_pt: bool) -> str:
    """Grounded no cfg REAL da sessao (nivel/objetivo/mode) -- nunca um
    conselho generico solto, regra dura do Agente 6."""
    nivel = str(getattr(cfg, "level", "N1"))
    mode = str(getattr(cfg, "mode", "dx"))
    nome_nivel = pq.NIVEL_NOME.get(nivel, nivel)
    if is_pt:
        base = f"Sua sessao esta configurada para '{nome_nivel}' ({nivel}), modo de entrada '{mode}'."
        if nivel == "N1":
            return (base + " N1 = classificacao multiclasse -> PLS-DA e o "
                    "metodo padrao do pipeline. Se quiser autenticar pureza "
                    "(puro vs. adulterado) em vez de identificar a especie, "
                    "troque pra N2 na aba Modelagem.")
        if nivel == "N2":
            return (base + " N2 = autenticacao one-class -> DD-SIMCA e "
                    "conformal sao os dois metodos disponiveis (aba "
                    "Modelagem, campo 'modo_ddsimca'/similar). DD-SIMCA da' "
                    "um score continuo; conformal da' garantia de cobertura "
                    "explicita, mas exige mais sessoes independentes pra "
                    "alpha baixo (ver [3] Diagnosticar).")
        return (base + " N3 = quantificacao -> PLS-R (pooled ou por "
                "especie, aba Modelagem). Se tambem quiser identificar QUAL "
                "adulterante antes de quantificar, o fluxo cego "
                "(Detectar->Identificar->Quantificar) ja roda automatico na "
                "predicao em lote quando o modelo tem o ensemble de "
                "identificacao treinado.")
    base = f"Your session is set to '{nome_nivel}' ({nivel}), input mode '{mode}'."
    if nivel == "N1":
        return (base + " N1 = multiclass classification -> PLS-DA is the "
                "pipeline's default method. For purity authentication "
                "(pure vs. adulterated) instead of species ID, switch to "
                "N2 in the Model tab.")
    if nivel == "N2":
        return (base + " N2 = one-class authentication -> DD-SIMCA and "
                "conformal are the two available methods (Model tab). "
                "DD-SIMCA gives a continuous score; conformal gives an "
                "explicit coverage guarantee but needs more independent "
                "sessions for a low alpha (see [3] Diagnose).")
    return (base + " N3 = quantification -> PLS-R (pooled or per-species, "
            "Model tab). If you also want to identify WHICH adulterant "
            "before quantifying, the blind flow (Detect->Identify->"
            "Quantify) already runs automatically in batch prediction when "
            "the model has a trained identification ensemble.")


_FAQ: List[Tuple[str, str, Any]] = [
    ("O que o GUARACI sabe fazer?", "What can GUARACI do?",
     lambda cfg, is_pt: (
         "Veja [4] Tecnicas disponiveis no menu do assistente -- lista "
         "gerada do catalogo real do projeto, nunca fica desatualizada."
         if is_pt else
         "See [4] Available techniques in the assistant menu -- generated "
         "from the project's real catalog, never goes stale.")),
    ("Qual metodo devo usar?", "Which method should I use?",
     _faq_metodo_recomendado),
    ("Por que uma quantificacao pode ficar bloqueada?",
     "Why can quantification be blocked?",
     lambda cfg, is_pt: (
         "No fluxo cego (Detectar->Identificar->Quantificar), a "
         "quantificacao SO' roda quando o adulterante foi identificado com "
         "garantia estatistica validada (identificacao_cobertura="
         "'validado', >=2 sessoes de coleta independentes por combinacao "
         "especie x adulterante). Sem essa garantia, a coluna "
         "'quantificacao_motivo_bloqueio' explica o motivo especifico "
         "('identificacao_desconhecida' ou 'identificacao_ambigua') -- "
         "nunca um numero sem base."
         if is_pt else
         "In the blind flow (Detect->Identify->Quantify), quantification "
         "ONLY runs when the adulterant was identified with a validated "
         "statistical guarantee (identificacao_cobertura='validado', >=2 "
         "independent collection sessions per species x adulterant "
         "combination). Without that guarantee, the "
         "'quantificacao_motivo_bloqueio' column explains the specific "
         "reason ('identificacao_desconhecida' or 'identificacao_ambigua') "
         "-- never a number without basis.")),
    ("O que significa um perfil 'nao validado'?",
     "What does an 'not validated' profile mean?",
     lambda cfg, is_pt: (
         "O selo ✅/⚠ no seletor de perfil (aba Dados) vem do campo "
         "'referencia' do perfil: nao-vazio = tem artigo ou fonte "
         "citado (ex.: um perfil seu que cita o artigo de calibracao); vazio = o perfil so' "
         "foi DECLARADO (faixa/vocabulario definidos), sem nenhuma "
         "validacao publicada ainda. Nao impede de usar -- so' nao alegue "
         "que o resultado foi validado com dado real. Os perfis que vem com o "
         "GUARACI aparecem como 'perfil de partida': sao faixa e "
         "pre-processamento iniciais para ajustar ao seu instrumento, e por "
         "regra do projeto nao citam conjunto de dados de terceiros."
         if is_pt else
         "The ✅/⚠ badge on the profile selector (Data tab) comes from the "
         "profile's 'referencia' field: non-empty = cites a paper or "
         "source (e.g. a profile of yours that cites its calibration paper); empty = the "
         "profile was only DECLARED (range/vocabulary set), with no "
         "published validation yet. It doesn't block usage -- just don't "
         "claim the result was validated against real data. Profiles shipped "
         "with GUARACI show as 'starting profile': an initial range and "
         "pre-processing to tune to your instrument; by project rule they "
         "do not cite third-party datasets.")),
]


def _guaraci_faq(cfg: Config) -> None:
    """Perguntas frequentes curadas (Agente 6, Fase 2) -- casadas por
    numero, resposta ancorada no `cfg` real quando aplicavel (regra dura:
    nunca inventa numero). Nao e' um chat de linguagem livre -- o projeto
    nao tem dependencia de LLM/NLP, um FAQ curado e' o que da' pra' fazer
    sem mudar a natureza determinista do software (ver docs/projeto/identidade_visual.md,
    secao do Agente 6, Fase 2)."""
    lang = _lang(); is_pt = lang == "PT"
    console.print()
    t = Table(show_header=False, box=rbox.SIMPLE, padding=(0, 1))
    t.add_column("N", style=PA, width=4)
    t.add_column("Pergunta" if is_pt else "Question", style=PW)
    for i, (q_pt, q_en, _resp) in enumerate(_FAQ, 1):
        t.add_row(f"[{i}]", q_pt if is_pt else q_en)
    console.print(t)
    raw = _ask(f"\n  [1-{len(_FAQ)}] ou Enter=voltar: " if is_pt
               else f"\n  [1-{len(_FAQ)}] or Enter=back: ")
    if raw.isdigit() and 1 <= int(raw) <= len(_FAQ):
        _q_pt, _q_en, resposta_fn = _FAQ[int(raw) - 1]
        resposta = resposta_fn(cfg, is_pt)
        console.print()
        console.print(Panel(
            Text(resposta, style=PW),
            title=f"[bold {PA}]{_q_pt if is_pt else _q_en}[/bold {PA}]",
            border_style=PA, box=rbox.ROUNDED, padding=(0, 2), width=_W(),
        ))
    _pause()


# ---------------------------------------------------------------------------
# FLUXO DE ENTRADA ORIENTADO A DECISAO (Bloco 19) -- "o que voce precisa
# decidir?" em vez de exigir que o usuario ja saiba o nome da tecnica.
# Reaproveita `technique_registry.REGISTRY` (fonte unica) pra sugerir a(s)
# tecnica(s) de cada opcao -- nunca uma lista de nomes escrita a mao aqui
# (o mesmo erro que motivou `technique_registry.py` para [4] Tecnicas
# disponiveis, Agente 6). O modo avancado (acesso direto as abas) continua
# SEMPRE disponivel -- este fluxo e' um atalho A MAIS, nunca substitui o
# menu numerado (nem os outros itens do assistente).
# ---------------------------------------------------------------------------
_FLUXO_DECISAO: List[Dict[str, Any]] = [
    dict(
        id="autenticidade",
        rotulo={"PT": "Autenticidade (e' essa especie mesmo, sem adulteracao?)",
                "EN": "Authenticity (is this really the declared species, unadulterated?)"},
        nivel="N2", preproc_default="msc_sg_mc",
        tecnicas=["ddsimca", "conformal_one_class"],
        validacao={"PT": "LOGO (leave-one-group-out) por mae_id -- "
                          "sensibilidade so' avaliavel com >=2 grupos de "
                          "puros por especie.",
                   "EN": "LOGO (leave-one-group-out) by mae_id -- "
                         "sensitivity only evaluable with >=2 pure-sample "
                         "groups per species."},
        criterio={"PT": "amostra nova cai dentro do limite calibrado "
                        "(DD-SIMCA) ou do conjunto de predicao conformal "
                        "no alpha escolhido.",
                  "EN": "new sample falls within the calibrated boundary "
                        "(DD-SIMCA) or the conformal prediction set at the "
                        "chosen alpha."},
    ),
    dict(
        id="identificar_especie",
        rotulo={"PT": "Identificar especie (qual especie e' essa amostra?)",
                "EN": "Identify species (which species is this sample?)"},
        nivel="N1", preproc_default="msc_sg_mc",
        tecnicas=["pls_da"],
        validacao={"PT": "CV group-aware (StableStratifiedGroupKFold) por mae_id.",
                   "EN": "Group-aware CV (StableStratifiedGroupKFold) by mae_id."},
        criterio={"PT": "balanced accuracy da CV group-aware -- sem limiar "
                        "fixo universal, compare contra o baseline do "
                        "proprio dataset.",
                  "EN": "balanced accuracy from group-aware CV -- no "
                        "universal fixed threshold, compare against the "
                        "dataset's own baseline."},
    ),
    dict(
        id="quantificar_teor",
        rotulo={"PT": "Quantificar teor (qual a porcentagem de adulterante/composto?)",
                "EN": "Quantify content (what's the adulterant/compound percentage?)"},
        nivel="N3", preproc_default="msc_sg_mc",
        tecnicas=["pls_r_pooled", "pls_r_por_especie"],
        validacao={"PT": "Q2 via CV group-aware; RMSEP em holdout externo "
                         "por objeto fisico.",
                   "EN": "Q2 via group-aware CV; RMSEP on external "
                         "per-object holdout."},
        criterio={"PT": "RPD/RER (figuras de merito) -- ver categoria "
                        "'Figuras de merito' no catalogo de tecnicas [4].",
                  "EN": "RPD/RER (figures of merit) -- see 'Figures of "
                        "merit' category in the technique catalog [4]."},
    ),
    dict(
        id="adulterante_desconhecido",
        rotulo={"PT": "Adulterante desconhecido (detectei que e' "
                      "adulterada, qual adulterante?)",
                "EN": "Unknown adulterant (detected adulteration, but "
                      "which adulterant?)"},
        nivel="N2", preproc_default="msc_sg_mc",
        tecnicas=["identificacao_conjunto_aberto"],
        validacao={"PT": "Predicao conforme por combinacao especie x "
                         "adulterante; so' rotula com >=2 sessoes de "
                         "coleta independentes.",
                   "EN": "Conformal prediction per species x adulterant "
                         "combination; only labels with >=2 independent "
                         "collection sessions."},
        criterio={"PT": "identificacao_cobertura='validado' -- senao "
                        "reporta DESCONHECIDO em vez de arriscar palpite.",
                  "EN": "identificacao_cobertura='validado' -- otherwise "
                        "reports UNKNOWN instead of risking an "
                        "unguaranteed guess."},
    ),
    dict(
        id="transferencia_instrumentos",
        rotulo={"PT": "Transferencia entre instrumentos (modelo calibrado "
                      "num equipamento, quero usar noutro)",
                "EN": "Instrument transfer (model calibrated on one "
                      "instrument, want to use it on another)"},
        nivel=None, preproc_default=None,
        tecnicas=["piecewise_direct_standardization", "direct_standardization"],
        validacao={"PT": "amostras medidas nos DOIS instrumentos (amostras "
                         "de transferencia) -- validar erro do modelo "
                         "transferido contra o original.",
                   "EN": "samples measured on BOTH instruments (transfer "
                         "samples) -- validate the transferred model's "
                         "error against the original."},
        criterio={"PT": "erro (RMSE) do modelo transferido comparavel ao "
                        "erro original no instrumento mestre.",
                  "EN": "transferred model's error (RMSE) comparable to "
                        "the original error on the master instrument."},
    ),
    dict(
        id="resolver_mistura",
        rotulo={"PT": "Resolver mistura (quero os espectros puros e "
                      "proporcoes, nao so' classificar/quantificar)",
                "EN": "Resolve mixture (want the pure spectra and "
                      "proportions, not just classify/quantify)"},
        nivel=None, preproc_default=None,
        tecnicas=["mcr_als"],
        aviso={"PT": "MCR-ALS e' ferramenta de interpretacao e resolucao "
                     "de componentes espectrais -- NAO recomendado para "
                     "QUANTIFICAR traco/adulterante minoritario sem "
                     "informacao supervisionada: o componente recuperado pode "
                     "nao correlacionar com o teor real. Se seu objetivo e' um NUMERO de teor com "
                     "garantia, use [3] Quantificar teor (PLS-R) em vez "
                     "desta opcao.",
               "EN": "MCR-ALS is an interpretation/component-resolution "
                     "tool -- NOT recommended to QUANTIFY a minor trace/"
                     "adulterant without supervised information. The "
                     "recovered component may not correlate with the real "
                     "content. If your goal is "
                     "a content NUMBER with a guarantee, use [3] Quantify "
                     "content (PLS-R) instead of this option."},
        validacao={"PT": "compare a proporcao estimada com o teor "
                         "declarado, se disponivel; SEMPRE rode "
                         "avaliar_incerteza_rotacional.",
                   "EN": "compare the estimated proportion against the "
                        "declared content, if available; ALWAYS run "
                        "avaliar_incerteza_rotacional."},
        criterio={"PT": "baixo lack-of-fit (%) E baixa sensibilidade a' "
                        "inicializacao -- nenhum dos dois sozinho basta.",
                  "EN": "low lack-of-fit (%) AND low sensitivity to "
                        "initialization -- neither alone is enough."},
    ),
]


def _guaraci_fluxo_decisao(cfg: Config) -> None:
    """Pergunta 'o que voce precisa decidir?' e sugere tecnica/pre-
    processamento/validacao/criterio de aceitacao a partir de
    `_FLUXO_DECISAO` + `technique_registry.REGISTRY` (fonte unica, ver
    cabecalho da secao). Oferece aplicar nivel/pre-processamento a' sessao
    atual -- SEMPRE opt-in (pergunta antes), nunca muda cfg em silencio."""
    lang = _lang(); is_pt = lang == "PT"
    from guaraci.technique_registry import REGISTRY
    registry_por_id = {e.id: e for e in REGISTRY}

    _cls(); _print_header(cfg)
    console.print()
    console.print(Panel(
        Text("O que voce precisa decidir?" if is_pt else "What do you need to decide?",
             style=f"bold {PA}"),
        border_style=PA, box=rbox.ROUNDED, padding=(1, 2), width=_W()))

    t = Table(show_header=False, box=rbox.SIMPLE, padding=(0, 1))
    t.add_column("Tecla", style=PA, width=4)
    t.add_column("Opcao", style=PW)
    for i, opcao in enumerate(_FLUXO_DECISAO, start=1):
        t.add_row(f"[{i}]", opcao["rotulo"]["PT" if is_pt else "EN"])
    t.add_row("[Q]", "Voltar" if is_pt else "Back")
    console.print(t)

    raw = _ask(f"  [{PA}]{_t('opcao')}: [/{PA}]").strip().upper()
    if raw in ("Q", "") or not raw.isdigit() or not (1 <= int(raw) <= len(_FLUXO_DECISAO)):
        return

    opcao = _FLUXO_DECISAO[int(raw) - 1]
    console.print()
    console.print(f"  [bold {PA}]{opcao['rotulo']['PT' if is_pt else 'EN']}[/bold {PA}]")
    console.print()
    console.print(f"  [{PW}]{'Tecnica(s) sugerida(s)' if is_pt else 'Suggested technique(s)'}:[/{PW}]")
    for tid in opcao["tecnicas"]:
        entrada = registry_por_id.get(tid)
        if entrada is None:
            continue
        console.print(f"    [{PA}]▸ {escape(entrada.nome)}[/{PA}]")
        console.print(f"      [{PM}]{escape(entrada.quando_usar)}[/{PM}]")
    console.print()
    if opcao.get("aviso"):
        console.print(f"  [warn]⚠ {escape(opcao['aviso']['PT' if is_pt else 'EN'])}[/warn]")
        console.print()
    if opcao.get("preproc_default"):
        console.print(
            f"  [{PW}]{'Pre-processamento default' if is_pt else 'Default preprocessing'}:"
            f"[/{PW}] [{PA}]{opcao['preproc_default']}[/{PA}]")
    console.print(f"  [{PW}]{'Validacao' if is_pt else 'Validation'}:[/{PW}] "
                  f"{opcao['validacao']['PT' if is_pt else 'EN']}")
    console.print(f"  [{PW}]{'Criterio de aceitacao' if is_pt else 'Acceptance criterion'}:"
                  f"[/{PW}] {opcao['criterio']['PT' if is_pt else 'EN']}")
    console.print()

    if opcao.get("nivel"):
        pergunta = (
            f"Aplicar nivel={opcao['nivel']}"
            + (f" e pre_processamento={opcao['preproc_default']}"
               if opcao.get("preproc_default") else "")
            + " a' sessao atual? [s/N]: "
        ) if is_pt else (
            f"Apply level={opcao['nivel']}"
            + (f" and preprocessing={opcao['preproc_default']}"
               if opcao.get("preproc_default") else "")
            + " to the current session? [y/N]: "
        )
        resposta = _ask(f"  [{PA}]{pergunta}[/{PA}]").strip().lower()
        if resposta in ("s", "sim", "y", "yes"):
            cfg.level = opcao["nivel"]
            if opcao.get("preproc_default"):
                cfg.default_preprocessing = opcao["preproc_default"]
            console.print(f"  [g]{'Aplicado.' if is_pt else 'Applied.'}[/g]")
    _pause()


def _unicode_ok() -> bool:
    """False pede texto puro (sem simbolos Unicode): GUARACI_ASCII=1, TERM=dumb
    ou modo linear de leitor de tela."""
    return not (os.environ.get("GUARACI_ASCII") == "1" or os.environ.get("TERM") == "dumb" or modo_linear())


def _abrir_confianca(tid: Optional[str] = None, cfg: Optional[Config] = None) -> None:
    """Assistente de Confianca (tecla [c] em qualquer tela): selo, evidencia,
    limites e como reproduzir. Texto gerado do registro (guaraci.confianca)."""
    from guaraci import confianca as _conf
    ascii_only = not _unicode_ok()
    itens = _conf.entradas(tid, lang=_lang())
    escopo = tid
    if not itens:           # tela sem selo proprio: mostra as tecnicas (o quadro todo)
        itens, escopo = _conf.entradas(None, opcoes=False, lang=_lang()), None
    while True:
        titulo = "Assistente de confianca" if _lang() == "PT" else "Trust assistant"
        console.print(Panel(escape(_conf.lista(escopo, _lang(), ascii_only, opcoes=escopo is not None)),
                            title=f"[bold {PA}]{titulo}[/bold {PA}]", border_style=PA, box=rbox.ROUNDED, padding=(0, 1)))
        with _ajuda_local(titulo, "Mostra o selo de cada tecnica ou opcao: a evidencia de validacao, os limites e como reproduzir. "
                          "Digite o numero ou o nome de um item para abrir a pagina dele." if _lang() == "PT" else
                          "Shows the badge of each technique or option: the validation evidence, the limits and how to reproduce it. "
                          "Type an item's number or name to open its page.",
                          [(str(i), str(it[2]), _cat.confianca(it[0], it[1], _lang()).get("rotulo_estado", ""))
                           for i, it in enumerate(itens, 1)]):
            raw = _input("  Numero ou nome (Enter volta): " if _lang() == "PT" else "  Number or name (Enter goes back): ")
        if not raw:
            return
        if raw.isdigit() and 1 <= int(raw) <= len(itens):
            tipo, ident = itens[int(raw) - 1][:2]
        else:
            achado = _conf.buscar_entrada(raw)
            if not achado:
                console.print(f"  [{PM}]{_t('invalido')}[/{PM}]")
                continue
            tipo, ident = achado
        console.print(Panel(escape(_conf.pagina(tipo, ident, _lang(), ascii_only)),
                            title=f"[bold {PA}]{ident}[/bold {PA}]", border_style=PF, box=rbox.ROUNDED,
                            padding=(1, 2), width=_W()))
        _pause()


def _assistente_rota(cfg: Config) -> None:
    """Assistente: objetivo + tipo de dado -> rota recomendada (do catalogo)."""
    from guaraci import assistente as _ass
    from guaraci.catalogo_ajuda import DADOS, OBJETIVOS
    L = 1 if _lang() == "EN" else 0
    for i, o in enumerate(OBJETIVOS, 1):
        console.print(f"  [{PA}][{i}][/{PA}] {escape(o[1 + L])}")
    ro = _ask(f"  [{PA}]{'O que voce quer fazer? (numero)' if L == 0 else 'What do you want to do? (number)'}:[/{PA}] ").strip()
    if not (ro.isdigit() and 1 <= int(ro) <= len(OBJETIVOS)):
        return
    for i, d in enumerate(DADOS, 1):
        console.print(f"  [{PA}][{i}][/{PA}] {escape(d[1 + L])}")
    rd = _ask(f"  [{PA}]{'Que dado voce tem? (numero)' if L == 0 else 'What data do you have? (number)'}:[/{PA}] ").strip()
    if not (rd.isdigit() and 1 <= int(rd) <= len(DADOS)):
        return
    r = _ass.rota(OBJETIVOS[int(ro) - 1][0], DADOS[int(rd) - 1][0])
    if r is None:
        console.print(f"  [{PM}]{'Nao ha rota pronta para essa combinacao: veja guaraci ajuda.' if L == 0 else 'No ready route for this combination: see guaraci ajuda.'}[/{PM}]")
    else:
        console.print(Panel(escape(_ass.formatar_rota(r, _lang())), title=f"[bold {PA}]{'Rota' if L == 0 else 'Route'}[/bold {PA}]",
                            border_style=PA, box=rbox.ROUNDED, padding=(1, 2), width=_W()))
    _pause()


def _assistente_explicar() -> None:
    from guaraci import assistente as _ass
    termo = _ask(f"  [{PA}]{'Opcao, tecnica, grafico ou termo' if _lang() == 'PT' else 'Option, technique, plot or term'}:[/{PA}] ").strip()
    if termo:
        console.print(Panel(escape(_ass.responder(termo, _lang())), title=f"[bold {PA}]{escape(termo)}[/bold {PA}]",
                            border_style=PF, box=rbox.ROUNDED, padding=(1, 2), width=_W()))
        _pause()


def _assistente_sugerir(cfg: Config) -> None:
    """Sugestoes para a configuracao atual. Nada e aplicado sozinho: o usuario escolhe."""
    from guaraci import assistente as _ass
    sug = _ass.sugerir(cfg, _lang())
    for i, x in enumerate(sug, 1):
        marca = f" [{PA}](aplicavel: {x.campo} = {x.valor})[/{PA}]" if x.campo else ""
        console.print(f"  [{PA}][{i}][/{PA}] {escape(x.texto)}{marca}")
    r = _ask(f"  [{PA}]{'Numero para aplicar (Enter volta)' if _lang() == 'PT' else 'Number to apply (Enter goes back)'}:[/{PA}] ").strip()
    if r.isdigit() and 1 <= int(r) <= len(sug) and sug[int(r) - 1].campo:
        x = sug[int(r) - 1]
        try:
            _set_val(cfg, x.campo, str(x.valor).lower() if isinstance(x.valor, bool) else str(x.valor))
            console.print(f"  [g]✓ {escape(x.campo)} = {escape(str(x.valor))}[/g]")
        except ValueError as e:
            console.print(f"  [{PR}]{escape(str(e))}[/{PR}]")
    _pause()


def _assistente_diagnosticar() -> None:
    from guaraci import assistente as _ass
    txt = _ask(f"  [{PA}]{'Cole o aviso ou erro' if _lang() == 'PT' else 'Paste the warning or error'}:[/{PA}] ").strip()
    if txt:
        diag = _ass.diagnosticar(txt, _lang())
        fazer = "O que fazer" if _lang() == "PT" else "What to do"
        corpo = "\n\n".join(f"{d['id']}: {d['significa']}\n{fazer}: {d['fazer']}" for d in diag) or \
            ("Nao reconheci esse aviso. Tente `guaraci buscar` com uma palavra-chave." if _lang() == "PT" else "Warning not recognised. Try `guaraci buscar` with a keyword.")
        console.print(Panel(escape(corpo), border_style=PF, box=rbox.ROUNDED, padding=(1, 2), width=_W()))
        _pause()


def _assistente_buscar() -> None:
    from guaraci import assistente as _ass
    q = _ask(f"  [{PA}]{'Palavra-chave' if _lang() == 'PT' else 'Keyword'}:[/{PA}] ").strip()
    if q:
        achados = _ass.buscar(q, _lang(), 8)
        for a in achados:
            console.print(f"  [{PA}][{escape(a['tipo'])}][/{PA}] {escape(a['titulo'])} ({escape(a['id'])}): {escape(a['trecho'])}")
        if not achados:
            console.print(f"  [{PM}]{'Nada encontrado.' if _lang() == 'PT' else 'Nothing found.'}[/{PM}]")
        _pause()


def _abrir_assistente(contexto: str = "", cfg: Optional[Config] = None, tela: str = "") -> None:
    """Abre o Assistente Guaraci (tecla G em qualquer tela). Com `tela`, abre no
    assunto dela: mostra o que a tela faz, quando usar e o que fazer depois.
    Depois de cada opcao volta para esta lista; so' V/0/Enter sai do assistente."""
    while _assistente_uma_escolha(contexto, cfg, tela):
        pass


#: O que cada opcao do Assistente Guaraci faz (painel do "?" dentro dele).
_FAZ_ASSISTENTE: Dict[str, Tuple[str, str]] = {
    "_": ("Assistente que revisa a configuracao, explica opcoes e termos e sugere o proximo passo, sem rodar nada.",
          "Assistant that reviews the configuration, explains options and terms and suggests the next step, without running anything."),
    "1": ("Confere a configuracao atual e aponta riscos (vazamento, validacao fraca, opcao incompativel).",
          "Checks the current configuration and points out risks (leakage, weak validation, incompatible option)."),
    "2": ("Lista as telas do programa com o que cada uma faz; o numero abre a explicacao.",
          "Lists the program screens with what each one does; the number opens the explanation."),
    "3": ("Carrega os dados apontados em [2.3] e diz se ha classe pequena, grupo faltando ou formato estranho.",
          "Loads the data set in [2.3] and says whether there is a small class, a missing group or an odd format."),
    "4": ("Mostra as tecnicas disponiveis, o que fazem e quando usar.", "Shows the available techniques, what they do and when to use them."),
    "5": ("Respostas curtas para as duvidas mais comuns.", "Short answers to the most common questions."),
    "6": ("Perguntas guiadas para decidir objetivo, validacao e rigor.", "Guided questions to decide goal, validation and rigour."),
    "7": ("Pelo que voce quer fazer e o dado que tem, indica a rota de telas recomendada.",
          "From what you want to do and the data you have, gives the recommended route of screens."),
    "8": ("Explica uma opcao, tecnica, grafico ou termo pelo nome.", "Explains an option, technique, plot or term by name."),
    "9": ("Olha a configuracao e diz qual e' o proximo passo.", "Looks at the configuration and says what the next step is."),
    "10": ("Cole um aviso ou erro e o assistente diz a causa provavel e o que fazer.",
           "Paste a warning or error and the assistant gives the likely cause and what to do."),
    "11": ("Busca uma palavra em todo o catalogo (opcoes, tecnicas, termos, telas).",
           "Searches a word across the whole catalogue (options, techniques, terms, screens)."),
    "V": ("Volta para a tela de onde o assistente foi aberto.", "Goes back to the screen the assistant was opened from."),
}


def _assistente_uma_escolha(contexto: str, cfg: Optional[Config], tela: str) -> bool:
    """Desenha o assistente e trata uma escolha. Devolve False quando a pessoa sai."""
    lang = _lang()
    _cls(); _print_header(cfg)
    if tela in _cat.TELAS:
        _painel_ajuda_tela(tela)
    console.print()

    opcoes = [
        ("1", "Revisar configuracao atual" if lang=="PT" else "Review current configuration"),
        ("2", "Informacoes sobre uma secao" if lang=="PT" else "Information about a section"),
        ("3", "Diagnosticar dados carregados" if lang=="PT" else "Diagnose loaded data"),
        ("4", "Tecnicas disponiveis" if lang=="PT" else "Available techniques"),
        ("5", "Perguntas frequentes" if lang=="PT" else "Frequently asked questions"),
        ("6", "O que voce precisa decidir?" if lang=="PT" else "What do you need to decide?"),
        ("7", "O que quero fazer? (rota recomendada)" if lang=="PT" else "What do I want to do? (recommended route)"),
        ("8", "Explicar uma opcao, tecnica ou termo" if lang=="PT" else "Explain an option, technique or term"),
        ("9", "Sugerir o proximo passo" if lang=="PT" else "Suggest the next step"),
        ("10", "Diagnosticar um aviso ou erro" if lang=="PT" else "Diagnose a warning or error"),
        ("11", "Buscar no catalogo" if lang=="PT" else "Search the catalogue"),
        ("V", "Voltar"                      if lang=="PT" else "Back"),
    ]
    t = Table(show_header=False, box=rbox.SIMPLE, padding=(0, 1))
    t.add_column("Tecla", style=PA, width=6)
    t.add_column("Opcao", style=PW)
    for k, v in opcoes:
        t.add_row(f"[{k}]", v)

    titulo_ctx = ((f"  Chamado de: {contexto}" if lang == "PT" else f"  Opened from: {contexto}") if contexto else "")
    console.print(Panel(
        Group(
            Text(f"\n  {'Ola, pesquisador! Como posso ajudar?' if lang == 'PT' else 'Hello, researcher! How can I help?'}\n{titulo_ctx}\n",
                 style=f"italic {PS}"),
            t,
        ),
        title=f"[bold {PA}]☀ GUARACI — {'Assistente Cientifico' if lang == 'PT' else 'Scientific Assistant'}[/bold {PA}]",
        border_style=PA, box=rbox.ROUNDED, padding=(1, 2), width=_W()
    ))

    L = 1 if lang == "EN" else 0
    with _ajuda_local("Guaraci", _FAZ_ASSISTENTE["_"][L],
                      [(k, v, _FAZ_ASSISTENTE.get(k, ("", ""))[L]) for k, v in opcoes]):
        raw = _ask(f"  [{PA}]{_t('opcao')}: [/{PA}]").strip().upper()
    if raw in ("", "0", "V", "Q"):
        return False
    if raw == "I":
        _toggle_idioma()
        return True
    if raw not in {k for k, _ in opcoes} or (raw == "1" and cfg is None):
        console.print(f"  [{PM}]{_t('invalido')}[/{PM}]")
        _pause()
        return True
    if raw == "1" and cfg is not None:
        _guaraci_revisar_config(cfg)
    elif raw == "2":
        _guaraci_navegar_secoes(cfg or Config())
    elif raw == "3":
        _guaraci_diagnosticar(cfg or Config())
    elif raw == "4":
        _guaraci_tecnicas()
    elif raw == "5":
        _guaraci_faq(cfg or Config())
    elif raw == "6":
        _guaraci_fluxo_decisao(cfg or Config())
    elif raw == "7":
        _assistente_rota(cfg or Config())
    elif raw == "8":
        _assistente_explicar()
    elif raw == "9":
        _assistente_sugerir(cfg or Config())
    elif raw == "10":
        _assistente_diagnosticar()
    elif raw == "11":
        _assistente_buscar()
    return True


def _rotulo_tecnica_efetivo(cfg: Optional[Config]) -> str:
    """Nome de 'tecnica' a exibir nos cabecalhos, ajustado ao `cfg.mode`
    real -- achado do Passo 103 (INSTRUCAO_HSI_ROBUSTEZ_E_VALIDACAO.md):
    `_TECNICA_SELECIONADA` (escolhida em [2.4] Tecnica Analitica) so' faz
    sentido para mode dx/csv/sintetico (espectros vibracionais). Modes
    "imagem" (colorimetria) e "hsi" (imageamento hiperespectral) NAO sao
    "uma tecnica vibracional escolhida" -- mostrar o default global
    'FT-NIR' nessas telas era herdado do template generico, incorreto
    (a tela HSI mostrando "Tecnica: FT-NIR" foi o achado que motivou
    esta correcao). Usado tanto por `_print_header` quanto por
    `_print_status` -- fonte unica, nao duas heuristicas divergentes."""
    modo = getattr(cfg, "mode", None) if cfg is not None else None
    if modo == "hsi":
        return "HSI"
    if modo == "imagem":
        return "Colorimetria digital" if _lang() == "PT" else "Digital colorimetry"
    nome = _TECNICA_SELECIONADA.get("nome", "FT-NIR")
    if _lang() == "EN":
        nome = {"IMAGEM": "IMAGE", "GENERICO": "GENERIC", "FLUORESCENCIA": "FLUORESCENCE"}.get(nome, nome)
    return nome


# ---------------------------------------------------------------------------
# CABECALHO COMPACTO
# ---------------------------------------------------------------------------
def _print_header(cfg: Optional[Config] = None) -> None:
    """Faixa de identidade do GUARACI: a moldura e as cores de antes, numa linha
    so' (3 linhas de terminal). Em janela estreita encurta o texto em vez de
    quebrar a linha, para a tela inteira caber e o foco abrir no topo."""
    is_pt = _lang() == "PT"
    tec_nome = _rotulo_tecnica_efetivo(cfg)
    tec_str = f"Tecnica: {tec_nome}" if is_pt else f"Technique: {tec_nome}"
    sub_longo = ("Plataforma quimiometrica com validacao anti-vazamento" if is_pt
                 else "Chemometrics platform with leakage-safe validation")
    sub_curto = "validacao anti-vazamento" if is_pt else "leakage-safe validation"
    areas = "Quimiometria | Machine Learning" if is_pt else "Chemometrics | Machine Learning"
    util = console.width - 8     # moldura + margem interna
    for partes in ((sub_longo, areas, tec_str), (sub_longo, tec_str), (sub_curto, tec_str), (tec_str,)):
        if len("☀ GUARACI ☀") + sum(len(x) + 5 for x in partes) <= util:
            break
    linha = Text(justify="center")
    linha.append("☀ ", style=PA)
    linha.append("GUARACI", style=f"bold {PA}")
    linha.append(" ☀", style=PA)
    for k, x in enumerate(partes):
        linha.append("  ·  ", style=PD)
        linha.append(x, style=PS if (k == 0 and len(partes) > 1) else PM)
    console.print(Panel(Align.center(linha), border_style=PA, box=rbox.DOUBLE, padding=(0, 1)))

# ---------------------------------------------------------------------------
# BARRA DE STATUS — compacta, 2 linhas
# ---------------------------------------------------------------------------
def _fonte_de_dados_ok(cfg: Config) -> bool:
    """A fonte de dados do modo atual esta configurada? (R5-08: o rodape dizia
    'Configurar dados' com modo csv/sintetico validos, porque so' olhava a pasta
    de arquivos .dx.)"""
    modo = str(getattr(cfg, "mode", "dx"))
    if modo == "sintetico":
        return True
    if modo == "csv":
        csv = str(getattr(cfg, "csv_file", "") or "")
        return bool(csv) and os.path.isfile(csv)
    pasta = _cfgv(cfg, "pasta_dados", "dados")
    return bool(pasta) and os.path.isdir(str(pasta)) and _count_dx(pasta) > 0


def _print_status(cfg: Config) -> None:
    lang = _lang()
    pasta = _cfgv(cfg, "pasta_dados", "dados")
    pasta_ok = bool(pasta) and os.path.isdir(str(pasta))
    n_dx = _count_dx(pasta) if pasta_ok else 0

    _modo = str(getattr(cfg, "mode", "dx"))
    _csv = str(getattr(cfg, "csv_file", "") or "")
    is_pt = lang == "PT"
    if _modo == "sintetico":
        dados_str = f"[g]{'dados sinteticos' if is_pt else 'synthetic data'}[/g]"
    elif _modo == "csv":
        # M-17c: com modo csv valido nao ha' "Pasta invalida" -- a fonte e' o arquivo.
        if _csv and os.path.isfile(_csv):
            dados_str = f"[g]CSV: {escape(os.path.basename(_csv))}[/g]"
        else:
            dados_str = (f"[err]{'CSV nao encontrado' if is_pt else 'CSV not found'}[/err]")
    elif pasta_ok and n_dx > 0:
        dados_str = f"[g]{_t('dados_ok', n=n_dx)}[/g]"
    elif pasta_ok:
        dados_str = f"[warn]{escape(str(pasta))}[/warn]"
    else:
        dados_str = f"[err]{_t('dados_err')}[/err]"

    preproc  = escape(str(_cfgv(cfg, "pre_processamento", "msc_sg_mc")))
    # So' a palavra-chave do nivel (Classificacao/Discriminacao/Quantificacao).
    _niv_raw  = str(_cfgv(cfg, "nivel", "N1"))
    _niv_nome = (_NIVEL_NOME_EN.get(_niv_raw, "") if _lang() == "EN" else pq.NIVEL_NOME.get(_niv_raw, ""))
    nivel    = escape(_niv_nome.split()[0] if _niv_nome else _niv_raw)

    # Guia de design v2: status em uma linha + o proximo passo. Hardware, RAM
    # e modulos pesados saíram da tela inicial (ficam em [H] e [6]); a tela
    # inteira precisa caber na janela para o foco abrir no topo do menu.
    proj_lbl = "Dados" if is_pt else "Data"
    nivel_l = "Nivel" if is_pt else "Level"
    console.print(Text.from_markup(
        f"  [{PM}]Status[/{PM}]  {proj_lbl}: {dados_str}  [{PM}]·[/{PM}]  "
        f"Preproc.: [info]{preproc}[/info]  [{PM}]·[/{PM}]  {nivel_l}: [a]{nivel}[/a]"))
    tecla, texto = _proximo_passo(cfg)
    rot = "Proximo passo" if is_pt else "Next step"
    console.print(Text.from_markup(
        f"  [{PM}]{rot}:[/{PM}] [{PA}][{escape(tecla)}][/{PA}] {escape(texto)}"))


def _proximo_passo(cfg: Config) -> Tuple[str, str]:
    """(tecla, texto) do proximo passo sugerido no menu principal. Sem dados:
    o exemplo guiado (primeiro contato) e a tela Dados; com dados: rodar."""
    is_pt = _lang() == "PT"
    if not _fonte_de_dados_ok(cfg):
        return ("2.3", "Dados: aponte os seus dados (ou [1.1] Exemplo guiado)" if is_pt
                else "Data: point to your data (or [1.1] Guided example)")
    return ("R", "Rodar analise com a configuracao atual" if is_pt
            else "Run the analysis with the current configuration")


# ---------------------------------------------------------------------------
# MENU PRINCIPAL — compacto, numeros adjacentes ao texto
# ---------------------------------------------------------------------------
def _telas_visiveis_do_grupo(num: str, cfg: Optional[Config], *, basico: bool) -> List[Dict[str, Any]]:
    """Telas do grupo `num` que aparecem agora. No Basico as de nivel avancado
    ficam ocultas, menos o HSI quando a tecnica e' HSI (o menu mostra o que a
    pessoa esta usando). O numero de cada tela nao muda entre os modos."""
    hsi_ativo = str(getattr(cfg, "mode", "")) == "hsi"
    return [t for t in _cat.telas_do_grupo_cli(num)
            if not (basico and t["nivel"] == "avancado" and not (hsi_ativo and t["id"] == "hsi"))]


def _n_telas_ocultas(cfg: Optional[Config]) -> int:
    nums = [g["num"] for g in _cat.grupos_cli()]
    return sum(len(_cat.telas_do_grupo_cli(n)) - len(_telas_visiveis_do_grupo(n, cfg, basico=True)) for n in nums)


def _print_main_menu(cfg: Optional[Config] = None) -> None:
    """Tela inicial: so' os 6 grupos numerados (revelacao progressiva). As
    telas ficam dentro deles, com tecla "grupo.item" (ex.: 2.3 Dados, que
    tambem pode ser digitada direto aqui). Letras sao so' acoes fixas."""
    lang = _lang()
    L = 1 if lang == "EN" else 0
    basico = _modo_usuario() == "iniciante"
    t = Table(box=None, show_header=False, padding=(0, 1), expand=False)
    t.add_column("n", no_wrap=True, width=3)
    t.add_column("nome", no_wrap=True)
    t.add_column("resumo", style=PM, overflow="fold")
    for g in _cat.grupos_cli():
        t.add_row(Text.from_markup(f"[{PA}][{g['num']}][/{PA}]"), Text(g["nome"][L], style=f"bold {PW}"),
                  Text(g["resumo"][L], style=PM))
    n_ocultas = _n_telas_ocultas(cfg) if basico else 0
    if lang == "PT":
        modo = ("Modo Basico" + (f": {n_ocultas} telas avancadas ocultas" if n_ocultas else "")) if basico else "Modo Avancado"
        troca = "mostrar avancadas" if basico else "voltar ao Basico"
        dica = "Digite o numero do grupo, ou o caminho direto (ex.: 2.3 = Dados)."
    else:
        modo = ("Basic mode" + (f": {n_ocultas} advanced screens hidden" if n_ocultas else "")) if basico else "Advanced mode"
        troca = "show advanced" if basico else "back to Basic"
        dica = "Type the group number, or the direct path (e.g. 2.3 = Data)."
    corpo = Group(t, Text(""),
                  Text.from_markup(f"  [{PM}]{escape(modo)}  ·  [/{PM}][{PS}][M][/{PS}][{PM}] {troca}[/{PM}]"),
                  Text(f"  {dica}", style=PM))
    tit_menu = "MENU PRINCIPAL" if lang == "PT" else "MAIN MENU"
    console.print(Panel(corpo, title=f"[bold {PA}]  {tit_menu}  [/bold {PA}]",
                        subtitle=f"[{PM}]{escape(_cat.RODAPE_MENU[lang])}[/{PM}]",
                        border_style=PA, box=rbox.ROUNDED, padding=(0, 1)))


_YAML_TOCADO = [False]   # 1.4/1.5 gravaram ou leram o YAML: o menu refaz a foto de "nao salvo"


#: Telas de passos (pedem caminho, rodam e mostram o resultado). Antes, depois de rodar ou de um erro
#: (arquivo nao encontrado), voltavam sozinhas para o grupo; agora a tela reabre e so' o voltar sai dela.
_TELAS_DE_PASSOS = {"selecao_amostras", "predicao", "comparar", "monitorar", "hsi", "relatorios", "hardware"}
_VOLTOU = [False]     # o ultimo prompt de tela recebeu a tecla de voltar


def _abrir_tela(tid: str, cfg: Config) -> None:
    if tid not in _TELAS_DE_PASSOS:
        _abrir_tela_uma_vez(tid, cfg)
        return
    while True:
        _VOLTOU[0] = False
        lidas = _LEITURAS[0]
        _abrir_tela_uma_vez(tid, cfg)
        if _VOLTOU[0] or _LEITURAS[0] == lidas:     # tela que nao pediu nada reabriria sem fim
            return


def _abrir_tela_uma_vez(tid: str, cfg: Config) -> None:
    """Abre a tela `tid` do catalogo pelo handler dela; as poucas que pedem algo
    diferente de handler(cfg) sao tratadas aqui (fonte unica do despacho)."""
    global _AJUDA_TELA
    anterior = _AJUDA_TELA
    _definir_ajuda(tid)          # "?" em qualquer prompt da tela abre o painel de ajuda dela
    if tid in ("salvar_config", "carregar_config"):
        _YAML_TOCADO[0] = True
    try:
        if tid == "assistente":
            _abrir_assistente("menu principal" if _lang() == "PT" else "main menu", cfg)
        elif tid == "confianca":
            _abrir_confianca(None, cfg)
        elif tid == "hardware":
            _cls(); _print_header(cfg); _menu_hardware(cfg)
        else:
            globals()[_cat.TELAS[tid]["handler"]](cfg)
    finally:
        _AJUDA_TELA = anterior   # ao voltar, o "?" e' de novo o de quem abriu esta tela


def _aviso_tecla_antiga(raw: str) -> bool:
    """Tecla do menu de antes do guia v2 (ex.: D, U, X): diz para onde a tela
    foi, em vez de so' 'opcao invalida'. Devolve True se era uma delas."""
    tid = _cat.TECLAS_ANTIGAS.get(raw.strip().upper())
    if not tid:
        return False
    t = _cat.tela(tid)
    L = 1 if _lang() == "EN" else 0
    console.print(f"  [{PM}]{'Esta tecla mudou' if L == 0 else 'This key changed'}: "
                  f"[{PA}][{t['tecla']}][/{PA}] {escape(t['nome'][L])}  "
                  f"({escape(_cat.trilha(tid, _lang()))})[/{PM}]")
    _pause()
    return True


def _menu_grupo(num: str, cfg: Config) -> None:
    """Pasta de um grupo do menu: as telas dele, numeradas pelo item (2.3 -> [3]
    dentro do grupo 2). [+] mostra as avancadas so' nesta visita."""
    g = next(x for x in _cat.grupos_cli() if x["num"] == num)
    mostrar = False
    while True:
        lang = _lang(); L = 1 if lang == "EN" else 0
        basico = _modo_usuario() == "iniciante" and not mostrar
        itens = _telas_visiveis_do_grupo(num, cfg, basico=basico)
        n_oc = len(_cat.telas_do_grupo_cli(num)) - len(itens)
        _cls(); _print_header(cfg)
        console.print(f"  [{PM}]Menu › [{num}] {escape(g['nome'][L])}[/{PM}]")
        t = Table(box=None, show_header=False, padding=(0, 1), expand=True)
        t.add_column("n", no_wrap=True, width=5)
        t.add_column("nome", no_wrap=True, min_width=24)
        t.add_column("faz", style=PM, overflow="fold")
        for x in itens:
            item = x["tecla"].split(".")[1]
            marca = "+" if x["nivel"] == "avancado" else " "
            t.add_row(Text.from_markup(f"[{PA}][{item}][/{PA}]"), Text(f"{marca}{x['nome'][L]}", style=PW),
                      Text(x["curta"][L], style=PM))
        rod = []
        if n_oc:
            rod.append(Text.from_markup(f"  [{PR}][+][/{PR}] "
                                        + (f"Mostrar telas avancadas ({n_oc} ocultas)" if L == 0
                                           else f"Show advanced screens ({n_oc} hidden)")))
        elif mostrar:
            rod.append(Text.from_markup(f"  [{PR}][+][/{PR}] " + ("Ocultar telas avancadas" if L == 0 else "Hide advanced screens")))
        rod.append(Text(f"  {_cat.RODAPE_TELA[lang]}", style=PM))
        console.print(Panel(Group(t, Rule(style=PD), *rod), title=f"[bold {PA}][{num}] {escape(g['nome'][L])}[/bold {PA}]",
                            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)))
        raw = _input(f"\n  {_t('opcao')}: ", ajuda=False).strip()
        por_num = {x["tecla"].split(".")[1]: x["id"] for x in itens}
        if raw == "+" and (n_oc or mostrar):
            mostrar = not mostrar
            continue
        if raw.startswith("?") and raw[1:].strip() in por_num:       # ?N = ajuda da tela N
            _ajuda_da_tela(por_num[raw[1:].strip()])
            continue
        if raw == "?":
            _ajuda_grupo(num, itens, n_oc, mostrar)
            continue
        if raw in por_num:
            _abrir_tela(por_num[raw], cfg)
            continue
        if raw.upper() in ("G", "A"):
            _abrir_assistente(g["nome"][L], cfg)
            continue
        if raw.upper() == "C":
            _abrir_confianca(None, cfg)
            continue
        r = _tecla_comum(raw, "", cfg)
        if r == "voltar":
            return
        if r == "tratada":
            continue
        oculta = next((x["id"] for x in _cat.telas_do_grupo_cli(num) if x["tecla"].split(".")[1] == raw), None)
        if raw.isdigit() and oculta:
            _abrir_tela(oculta, cfg)     # tela avancada oculta: o numero abre direto (antes pedia [+] e outro toque)
            continue
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]")
        _pause()


# ---------------------------------------------------------------------------
# CAIXA DE EXECUCAO — call-to-action destacado para iniciar a analise (R)
# ---------------------------------------------------------------------------
def _print_acao_rodar(cfg: Config) -> None:
    """Uma linha com a acao principal. Substitui a caixa EXECUCAO da tela
    inicial (RAM e modulos pesados continuam em [6.4] e na tela [3.4])."""
    pronto = _fonte_de_dados_ok(cfg)
    is_pt = _lang() == "PT"
    cor = PG if pronto else PM
    if pronto:
        txt = "Rodar analise" if is_pt else "Run analysis"
    else:
        txt = ("Rodar analise (falta apontar os dados em [2.3] Dados)" if is_pt
               else "Run analysis (point to the data in [2.3] Data first)")
    console.print(Text.from_markup(f"  [bold {cor}][R][/bold {cor}] [{cor}]{escape(txt)}[/{cor}]"))


def _print_run_box(cfg: Config) -> None:
    """Caixa de destaque para iniciar a analise.

    Muda de cor conforme a prontidao (verde = pronto, cinza = falta config)
    e exibe info complementar ao Status do Projeto: RAM livre (indicador de
    3 niveis) e os modulos pesados ativos (Benchmark / Monte Carlo / SHAP).
    """
    lang = _lang()
    is_pt = lang == "PT"

    pasta = _cfgv(cfg, "pasta_dados", "dados")
    pasta_ok = bool(pasta) and os.path.isdir(str(pasta))
    n_dx = _count_dx(pasta) if pasta_ok else 0
    pronto = _fonte_de_dados_ok(cfg)

    # RAM livre — indicador visual de 3 niveis
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_livre = vm.available / (1024 ** 3)
        ram_total = vm.total / (1024 ** 3)
        if ram_livre >= 8:
            ram_cor, ram_ico = PG, "●●●"
        elif ram_livre >= 4:
            ram_cor, ram_ico = PA, "●●○"
        else:
            ram_cor, ram_ico = PR, "●○○"
        ram_txt = f"[{ram_cor}]{ram_ico}  RAM {ram_livre:.1f}/{ram_total:.0f} GB[/{ram_cor}]"
    except Exception:  # noqa: BLE001 -- indicador visual best-effort (psutil
        # ausente ou probe falha por variacao de SO); "N/A" e' o fallback
        # documentado, nunca impede a tela de renderizar.
        ram_txt = f"[{PM}]RAM N/A[/{PM}]"

    # Prontidao + cor da chamada
    if pronto:
        cta_cor = PG
        check = (f"[{PG}]✔ Pronto para executar[/{PG}]" if is_pt
                 else f"[{PG}]✔ Ready to run[/{PG}]")
    else:
        cta_cor = PM
        check = (f"[{PR}]✖ {_t('status_erro')}[/{PR}]")

    # Modulos pesados ativos — definem o tempo de execucao e NAO aparecem
    # no Status do Projeto (info complementar, nao redundante).
    extras = []
    if _cfgv(cfg, "benchmark", False):      extras.append("Benchmark")
    if _cfgv(cfg, "monte_carlo", False):    extras.append("Monte Carlo")
    if _cfgv(cfg, "shap_benchmark", False): extras.append("SHAP")
    if extras:
        extras_txt = f"[{PA}]" + " · ".join(extras) + f"[/{PA}]"
    else:
        extras_txt = f"[{PM}]" + ("nenhum" if is_pt else "none") + f"[/{PM}]"
    ext_lbl = "Extras" if is_pt else "Extras"

    big = "RODAR PIPELINE" if is_pt else "RUN PIPELINE"
    sub = ("Pressione  [R]  e Enter para comecar a analise"
           if is_pt else "Press  [R]  then Enter to start the analysis")

    inner = Table(box=None, show_header=False, padding=(0, 1), expand=True)
    inner.add_column("c", justify="center")
    inner.add_row(Text.from_markup(
        f"[bold {cta_cor}]▶   [{cta_cor}]\\[R][/{cta_cor}]   {big}   ◀[/bold {cta_cor}]"))
    inner.add_row(Text.from_markup(f"[{PM}]{sub}[/{PM}]"))
    inner.add_row(Text.from_markup(""))
    inner.add_row(Text.from_markup(
        f"{check}     {ram_txt}     "
        f"[{PM}]{ext_lbl}:[/{PM}] {extras_txt}"))

    tit = "  ☀  EXECUCAO  ☀  " if is_pt else "  ☀  EXECUTION  ☀  "
    console.print(Panel(
        Align.center(inner),
        title=f"[bold {cta_cor}]{tit}[/bold {cta_cor}]",
        border_style=cta_cor,
        box=rbox.HEAVY,
        padding=(0, 1),
    ))

# ---------------------------------------------------------------------------
# SUBMENU COMPACTO — campos com valor na mesma linha
# ---------------------------------------------------------------------------
# _trunc importada de guaraci.cli_logic (item 19) — ver topo do arquivo.


def _desc_curta(key: str, max_c: int = 42) -> str:
    """Retorna descricao resumida do campo (max_c chars) para exibicao inline.

    Resolve idioma/_HELP_DB/_SPEC_BY_KEY (estado desta tela) e delega o
    truncamento a `guaraci.cli_logic.truncate_desc_by_sentence` (funcao pura,
    testada).
    """
    lang = _lang()
    from guaraci.catalogo_curta import CURTA
    if key in CURTA:        # texto curto proprio: completo, nunca cortado com reticencias
        return CURTA[key][1 if lang == "EN" else 0]
    h = _HELP_DB.get(key, {})
    desc = h.get(lang, h.get("PT", {})).get("desc", "")
    if not desc:
        desc = _SPEC_BY_KEY.get(key, {}).get("desc", "")
    return _truncar_desc_por_frase(desc, max_c)


def _print_submenu_compact(
    title: str, desc: str, fields: List[str], cfg: Config,
    extras: Optional[List[Tuple[Any, ...]]] = None,
    campos_avancados: Optional[set] = None,
    mostrar_avancado: bool = True,
) -> List[str]:
    """
    Submenu compacto: [N] ICON Nome  Valor  Descricao-breve
    Exibe o valor atual e uma descricao curta na mesma linha.

    `campos_avancados`: subconjunto de `fields` a ESCONDER quando o mode do
    usuario e' Iniciante e `mostrar_avancado=False` (reduz a densidade de
    configuracao p/ quem so' quer usar os defaults). Quando `None`, nenhum campo e' escondido --
    comportamento identico ao de antes desta feature (compatibilidade com
    chamadores que nao passam esse argumento).

    Retorna a lista de campos REALMENTE exibidos (numerados 1..N na tela),
    que o chamador deve usar para indexar a escolha do usuario -- os
    numeros na tela sempre correspondem a esta lista, nunca ao `fields`
    original quando ha' campos escondidos.
    """
    lang = _lang()
    ocultos = (campos_avancados or set()) if (
        campos_avancados and _modo_usuario() == "iniciante" and not mostrar_avancado
    ) else set()
    fields_visiveis = [f for f in fields if f not in ocultos]

    t = Table(box=None, show_header=False, padding=(0, 0), expand=True)
    t.add_column("N",    no_wrap=True, width=6)
    t.add_column("Ico",  no_wrap=True, width=2)
    # So a descricao pode encolher: numero, icone, nome e valor tem largura fixa
    # (senao o Rich corta o numero da opcao em terminais de 80 colunas).
    desc_w = max(18, _W() - 4 - 6 - 2 - 26 - 14 - 2)
    t.add_column("Nome", overflow="fold", min_width=20, max_width=26)   # nome longo quebra a linha, nunca e' cortado
    t.add_column("Val",  no_wrap=True, min_width=8, max_width=14)
    t.add_column("Desc", style=PM, width=desc_w, overflow="fold")   # texto completo: em janela estreita quebra a linha, nunca corta

    for i, key in enumerate(fields_visiveis, 1):
        nome  = _nome_campo(key)
        val   = _get_val(cfg, key)
        r_hex = _risco_hex(key)
        r_ico = _risco_icon(key)
        breve = _desc_curta(key, desc_w)

        # Formatar valor — sem hifenizacao
        if isinstance(val, bool):
            val_txt = Text("Sim" if lang == "PT" else "Yes", style=PG) if val \
                      else Text("Nao" if lang == "PT" else "No", style=PM)
        elif val is None or str(val) == "":
            val_txt = Text("—", style=PM)
        else:
            val_txt = Text(_trunc(_valor_para_exibir(val), 13), style=PS)

        t.add_row(
            Text.from_markup(f"  [{r_hex}][{i}][/{r_hex}]"),
            Text(r_ico, style=r_hex),
            Text(f"{'+' if key in (campos_avancados or ()) else ' '}{nome}", style=PW),
            val_txt,
            Text(breve, style=PM) if breve else Text(""),
        )

    # Pastas de opcoes avancadas e acoes da tela: numeros depois dos campos, na
    # MESMA tabela (mesmas colunas, alinhadas aos campos). Item (chave, rotulo) ou
    # (chave, nome, valor, descricao); rotulo maior que a coluna Nome continua na Desc.
    if extras:
        t.add_row(Text(""), Text(""), Text(""), Text(""), Text(""))
        for j, item in enumerate(extras, 1):
            ek = len(fields_visiveis) + j
            if len(item) == 4:
                _ek, e_nome, e_val, e_desc = item
            else:
                _ek, rot = item
                e_val = ""
                if 0 < rot.find(" (") <= 25:   # "Nome (detalhe)": nome na coluna Nome, detalhe na Desc
                    corte = rot.find(" (")
                    e_nome, e_desc = rot[:corte], rot[corte + 1:]
                else:                          # rotulo simples: inteiro na coluna Nome (quebra no espaco, na mesma cor)
                    e_nome, e_desc = rot, ""
            t.add_row(Text.from_markup(f"  [{PA}][{ek}][/{PA}]"),
                      Text("›" if str(_ek).startswith("pasta:") else "", style=PA),
                      Text(e_nome if e_nome.startswith("+") else f" {e_nome}", style=PW),
                      Text(e_val, style=PS), Text(e_desc, style=PM))

    # Rodape: legenda de risco + comandos
    n_ocultos = len(fields) - len(fields_visiveis)
    aviso_ocultos = (
        (f"  [{PR}][+][/{PR}] Mostrar opcoes avancadas ({n_ocultos} ocultas)\n"
         if lang == "PT" else
         f"  [{PR}][+][/{PR}] Show advanced options ({n_ocultos} hidden)\n")
        if n_ocultos > 0 else ""
    )
    rodape = Text.from_markup(
        f"{aviso_ocultos}"
        f"  [{PF}]●[/{PF}] Visual  "
        f"[{PA}]◆[/{PA}] {'Analitico' if lang == 'PT' else 'Analytical'}  "
        f"[{PR}]▲[/{PR}] {'Avancado' if lang == 'PT' else 'Advanced'}\n"
        f"  [{PM}]{escape(_cat.RODAPE_TELA[lang])}[/{PM}]"
    )

    console.print(Panel(
        Group(t, Rule(style=PD), rodape),
        title=f"[bold {PA}]{escape(title)}[/bold {PA}]",
        subtitle=f"[{PM}]{escape(desc)}[/{PM}]",
        border_style=PA,
        box=rbox.ROUNDED,
        padding=(0, 1),
    ))
    return fields_visiveis

# Toggles cuja utilidade depende do OBJETIVO cientifico resolvido (nao do
# nivel em si) -- espelha 1:1 modos_analise._FIG_OBJETIVOS: fora do objetivo
# listado, o motor nem computa (gated por should_generate() em pipeline.executar(),
# ou, no caso de teste_wold/teste_cv_anova, pelo guard adicionado no achado
# de 2026-08-06 -- antes rodavam incondicionalmente e escreviam metrica de
# CLASSIFICACAO sem sentido no resumo de um run de Quantificacao).
_TOGGLES_SO_CLASSIFICACAO = (
    "opls_da", "selecao_variaveis_etapa4", "comparar_pre_processamentos",
    "teste_wold", "teste_cv_anova", "teste_martens",
    "benchmark", "monte_carlo", "shap_benchmark",
)
_TOGGLES_SO_QUANTIFICACAO = ("benchmark_regressao",)


def _ajustar_toggles_por_nivel(cfg: Config) -> List[str]:
    """Desliga toggles que ficam INERTES no nivel/objetivo atual de `cfg`.

    Chamado apos o campo "nivel" mudar de valor (pedido do usuario,
    2026-08-06: "quando mudo de mode... continua ativado a dd simca e
    semelhantes... gostaria que ao mudar o N, mudasse as opcoes de modos
    como esse que nao agrega a analise"). Antes disso, o toggle DD-SIMCA
    permanecia visualmente "ligado" no menu mesmo em N1 -- funcionalmente
    inerte (pipeline.executar() ignora com aviso), mas confuso na tela.

    A logica espelha EXATAMENTE o que pipeline.executar()/modos_analise.py
    decidem em runtime -- nao e' uma aproximacao da UI, e' a mesma regra:
      - DD-SIMCA: N1 sempre ignora (forca False); N2 sempre forca ligado
        internamente (forca True, refletindo o que vai acontecer de
        qualquer forma); demais niveis respeitam should_generate (objetivo).
      - Demais toggles (ver `_TOGGLES_SO_CLASSIFICACAO`): forcados False
        fora de objective=Classificacao; `benchmark_regressao` fora de
        Quantificacao.

    Retorna a lista de CHAVES (_CONFIG_SPEC) efetivamente alteradas, para o
    chamador avisar o usuario -- um reset silencioso seria tao confuso
    quanto o problema original.
    """
    mudou: List[str] = []

    def _forcar(key: str, valor: bool) -> None:
        spec = _SPEC_BY_KEY.get(key)
        if spec is None:
            return
        attr = spec["attr"]
        if bool(getattr(cfg, attr, False)) != valor:
            setattr(cfg, attr, valor)
            mudou.append(key)

    if cfg.level == "N1":
        _forcar("ddsimca", False)
    elif cfg.level == "N2":
        _forcar("ddsimca", True)
    else:
        _forcar("ddsimca", pq.should_generate(cfg, "ddsimca"))

    objetivo = pq.resolve_objective(cfg)
    if objetivo != pq.CLASSIFICACAO:
        for key in _TOGGLES_SO_CLASSIFICACAO:
            _forcar(key, False)
    if objetivo != pq.QUANTIFICACAO:
        for key in _TOGGLES_SO_QUANTIFICACAO:
            _forcar(key, False)

    return mudou


# ---------------------------------------------------------------------------
# EDICAO DE CAMPO
# ---------------------------------------------------------------------------
def _editar_campo(cfg: Config, key: str) -> bool:
    lang = _lang()
    spec = _SPEC_BY_KEY.get(key)
    if spec is None:
        _msg = (f"Campo '{key}' nao encontrado." if lang == "PT"
                else f"Field '{key}' not found.")
        console.print(f"  [err]{_msg}[/err]")
        return False

    nome     = _nome_campo(key)
    val_atual = _get_val(cfg, key)
    # Valor interno cru (ex.: "N1") para comparar com as opcoes — val_atual
    # pode ser um rotulo amigavel de exibicao ("Classificacao... (N1)").
    val_cru   = _attr_para_yaml(spec, cfg)
    tipo      = spec.get("tipo", "str")
    opcoes    = spec.get("opcoes")
    risk      = _RISK_CLASS.get(key, "ANALITICO")
    r_hex     = _risco_hex(key)

    # Painel de edicao minimalista
    info = Table(box=None, show_header=False, padding=(0, 1))
    info.add_column("L", style=PM, width=10, no_wrap=True)
    info.add_column("V", no_wrap=False)
    # Campos booleanos SEMPRE viram escolha numerada [1]/[2], nunca texto
    # livre. Antes disso, os 20 campos bool do _CONFIG_SPEC caiam no ramo
    # else (texto livre): _coagir_valor aceitava so' um punhado de palavras
    # magicas ("true"/"sim"/"1"/"yes"/"s"/"v") como True e tratava QUALQUER
    # outra coisa como False SEM AVISO — digitar "y" (comum em outros
    # softwares) virava False silenciosamente, e um erro de digitacao nunca
    # dava mensagem de erro. Unificado com o MESMO padrao numerado que todo
    # campo de escolha ja usa (nivel, modo_ddsimca, etc.) — "transformar tudo
    # em modelo padrao" (pedido do usuario, 2026-08-06).
    eh_bool = (tipo == "bool")
    if eh_bool:
        val_txt_atual = _t("opt_sim") if bool(val_cru) else _t("opt_nao")
    else:
        val_txt_atual = str(val_atual)

    en = lang == "EN"
    info.add_row("Field:" if en else "Campo:", Text(nome, style=f"bold {r_hex}"))
    info.add_row("Current:" if en else "Atual:", Text(_texto_para_exibir(val_txt_atual), style=PS))
    info.add_row("Type:" if en else "Tipo:", Text(_TIPO_EN.get(tipo, tipo) if en else tipo, style=PM))
    if opcoes:
        info.add_row("Options:" if en else "Opcoes:", Text(
            " | ".join(_rotulo_opcao(key, o) for o in opcoes), style=PM))

    console.print(Panel(
        info,
        title=f"[{r_hex}]{'Edit' if en else 'Editar'}: {escape(nome)}[/{r_hex}]",
        border_style=r_hex, box=rbox.ROUNDED, padding=(0, 1)
    ))

    if eh_bool:
        opcoes_bool = (True, False)   # [1]=Sim/Yes, [2]=Nao/No, sempre nesta ordem
        console.print()
        for j, op in enumerate(opcoes_bool, 1):
            mk = f"[{PA}]►[/{PA}]" if bool(op) == bool(val_cru) else " "
            rotulo = _t("opt_sim") if op else _t("opt_nao")
            console.print(f"  {mk} [{PA}][{j}][/{PA}] {rotulo}")
        console.print()
        raw = _input(f"  {_t('escolha_num')}", ajuda=False)
        if not raw:
            console.print(f"  [{PM}]{_t('mantido')}[/{PM}]"); return False
        if raw == "?":
            _mostrar_ajuda(key); return False
        if raw.isdigit() and 1 <= int(raw) <= 2:
            # Sempre "true"/"false" em ingles: e' o vocabulario que
            # _coagir_valor reconhece de forma inequivoca, independente do
            # idioma da interface (mesmo padrao do resto do _CONFIG_SPEC,
            # que grava sempre em ingles/codigo interno no YAML).
            raw = "true" if opcoes_bool[int(raw) - 1] else "false"
        else:
            _msg = ("Digite 1 ou 2." if lang == "PT" else "Enter 1 or 2.")
            console.print(f"  [err]{_msg}[/err]"); return False
    elif opcoes:
        console.print()
        for j, op in enumerate(opcoes, 1):
            mk = f"[{PA}]►[/{PA}]" if str(op) == str(val_cru) else " "
            console.print(f"  {mk} [{PA}][{j}][/{PA}] "
                          f"{escape(_rotulo_opcao(key, op))}")
        console.print()
        raw = _input(f"  [1-{len(opcoes)}] {'ou Enter=manter' if lang == 'PT' else 'or Enter=keep'}: ", ajuda=False)
        if not raw:
            console.print(f"  [{PM}]{_t('mantido')}[/{PM}]"); return False
        if raw == "?":
            _mostrar_ajuda(key); return False
        if raw.isdigit() and 1 <= int(raw) <= len(opcoes):
            raw = str(opcoes[int(raw) - 1])
    else:
        console.print()
        raw = _input(f"  {_t('novo_valor')}", ajuda=False)
        if not raw:
            console.print(f"  [{PM}]{_t('mantido')}[/{PM}]"); return False
        if raw == "?":
            _mostrar_ajuda(key); return False

    # Confirmacao para campos analiticos
    if risk == "ANALITICO":
        conf = _ask(f"  [{PA}]{_t('conf_anal')}[/{PA}] ")
        if conf.lower() not in ("s", "y", "sim", "yes"):
            console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); return False

    try:
        nivel_mudou = (key == "nivel" and raw != str(val_cru))
        _set_val(cfg, key, raw)
        # Bool sempre mostra o rotulo localizado (Sim/Nao ou Yes/No), nunca
        # o "true"/"false" cru gravado internamente.
        valor_msg = (_t("opt_sim") if raw == "true" else _t("opt_nao")) \
                    if eh_bool else raw
        msg = _t("atualizado", campo=nome, valor=valor_msg)
        console.print(f"  [g]✓ {escape(msg)}[/g]")
        if nivel_mudou:
            ajustados = _ajustar_toggles_por_nivel(cfg)
            if ajustados:
                nomes = ", ".join(_nome_campo(k) for k in ajustados)
                console.print(f"  [{PM}]{_t('ajuste_nivel', campos=nomes)}[/{PM}]")
        return True
    except ValueError as e:   # _set_val/_coagir_valor: valor digitado invalido
        console.print(f"  [err]{'Erro' if lang == 'PT' else 'Error'}: {escape(_msg_valor(str(e)))}[/err]")
        return False

# ---------------------------------------------------------------------------
# AJUDA UNIFICADA — toda tecla "?" desenha o MESMO painel
# ---------------------------------------------------------------------------
# Um molde so' (Campo/Tela, Faz, Use quando, Nao use, Impacto, Padrao, Faixa,
# Exemplos e a lista das opcoes) para campo, tela, menu de grupo, menu
# principal e listas de escolha. Antes cada tela tinha o seu: uma linha cinza,
# uma lista solta ou nada (relato do Erley, 2026-11-06).
_ROT_AJUDA: Dict[str, Tuple[str, str]] = {
    "campo": ("Campo:", "Field:"), "tela": ("Tela:", "Screen:"), "menu": ("Menu:", "Menu:"),
    "onde": ("Onde:", "Where:"), "faz": ("Faz:", "Does:"), "use": ("Use quando:", "Use when:"),
    "nao_use": ("Nao use:", "Avoid when:"), "impacto": ("Impacto:", "Impact:"), "padrao": ("Padrao:", "Default:"),
    "faixa": ("Faixa:", "Range:"), "depois": ("Depois:", "Next:"), "precisa": ("Precisa de:", "You need:"),
    "exemplos": ("Exemplos:", "Examples:"), "opcoes": ("Opcoes:", "Options:"), "teclas": ("Teclas:", "Keys:"),
}

def _painel_ajuda(titulo: str, linhas: List[Tuple[str, Any, str]],
                  opcoes: Optional[List[Tuple[str, str, str]]] = None,
                  exemplos: Optional[Dict[str, Any]] = None, cor: str = PA,
                  dica: str = "") -> Panel:
    """O painel unico de ajuda. linhas = (rotulo, texto, estilo) -- texto vazio
    nao aparece; opcoes = (tecla, nome, o que faz) de cada escolha da tela."""
    L = 1 if _lang() == "EN" else 0
    info = Table(box=None, show_header=False, padding=(0, 1))
    info.add_column("L", style=PM, width=12, no_wrap=True)
    info.add_column("V", overflow="fold")
    for rot, val, estilo in linhas:
        if val is not None and str(val).strip():
            info.add_row(rot, Text(str(val), style=estilo))
    if exemplos:
        info.add_row("", Text(""))
        info.add_row(_ROT_AJUDA["exemplos"][L], Text(""))
        for ek, ev in list(exemplos.items())[:4]:
            info.add_row(f"  {_texto_para_exibir(ek)}", Text(str(ev), style=PM))
    if opcoes:
        info.add_row("", Text(""))
        info.add_row(_ROT_AJUDA["opcoes"][L], Text(""))
        for tecla, nome, faz_ in opcoes:
            tx = Text(str(nome), style=f"bold {PW}")
            if faz_:
                tx.append(f": {faz_}", style=PM)
            info.add_row(Text(f"  [{tecla}]" if tecla else "  -", style=PA), tx)
    partes: List[Any] = [info]
    if dica:
        partes += [Text(""), Text(dica, style=PM)]
    return Panel(Group(*partes), title=f"[{cor}] {escape(titulo)} [/{cor}]",
                 border_style=cor, box=rbox.ROUNDED, padding=(0, 1))


def _mostrar_ajuda(key: str) -> None:
    """Ajuda completa de um campo de configuracao."""
    lang   = _lang()
    L      = 1 if lang == "EN" else 0
    h      = _HELP_DB.get(key, {})
    h_lang = h.get(lang) or h.get("PT", {})
    nome   = _nome_campo(key)
    r_hex  = _risco_hex(key)
    spec   = _SPEC_BY_KEY.get(key, {})

    # Fallback: se _HELP_DB nao cobre o campo, usa a descricao do _CONFIG_SPEC
    desc    = h_lang.get("desc") or spec.get("desc") or (
        "Sem descricao detalhada para este campo." if lang == "PT"
        else "No detailed description for this field.")
    default  = h.get("default", spec.get("default", "—"))
    opcoes   = spec.get("opcoes")
    faixa    = h.get("range") or (" | ".join(str(o) for o in opcoes) if opcoes else
                                  spec.get("tipo", "—"))
    R = _ROT_AJUDA
    linhas = [(R["campo"][L], nome, f"bold {r_hex}"), (R["faz"][L], desc, PW),
              (R["use"][L], h_lang.get("use", ""), PW), (R["nao_use"][L], h_lang.get("nao_use", ""), PW),
              (R["impacto"][L], h_lang.get("impacto", "—"), r_hex),
              (R["padrao"][L], _texto_para_exibir(default) if str(default).strip() else ("(vazio)", "(empty)")[L], PS), (R["faixa"][L], _texto_para_exibir(faixa), PM)]
    console.print(_painel_ajuda(nome, linhas, exemplos=h_lang.get("exemplos") or None, cor=r_hex))
    _pause()

# ===========================================================================
# MENUS DE CONFIGURACAO

# ===========================================================================

def _grid_alpha_valido(valor: Any, padrao: float = 0.4) -> float:
    """Transparencia da grade dentro de (0, 1]; qualquer outra coisa (valor
    antigo invalido ja' salvo em ~/.guaraci, texto) volta ao padrao."""
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return padrao
    return v if 0.0 < v <= 1.0 else padrao


@contextlib.contextmanager
def _ajuda_local(titulo: str, faz_: str, opcoes: Optional[List[Tuple[str, str, str]]] = None):
    """Enquanto o bloco roda, "?" num prompt mostra o painel desta escolha
    (uma lista dentro de uma tela: paleta, metodo, tecnica...)."""
    global _AJUDA_TELA
    anterior = _AJUDA_TELA
    L = 1 if _lang() == "EN" else 0

    def _geral() -> None:
        console.print(_painel_ajuda(titulo, [(_ROT_AJUDA["menu"][L], titulo, f"bold {PA}"), (_ROT_AJUDA["faz"][L], faz_, PW)],
                                    opcoes=opcoes))

    def _mostrar() -> None:
        if opcoes:
            _ajuda_por_escolha(opcoes, _geral, pausar=False)
        else:
            _geral()
    _AJUDA_TELA = _mostrar
    try:
        yield
    finally:
        _AJUDA_TELA = anterior


def _opcoes_dos_campos(tid: str, campos: Optional[List[str]] = None, numerar: bool = False) -> List[Tuple[str, str, str]]:
    """Cada campo de uma tela de configuracao como opcao da ajuda: nome e o que faz."""
    L = 1 if _lang() == "EN" else 0
    avancadas = _cat.campos_avancados(tid) if tid in _cat.TELAS else frozenset()
    oculto = " (avancado: ative com [+])" if L == 0 else " (advanced: turn on with [+])"
    lista = campos if campos is not None else (_cat.campos_da_tela(tid, basicos_primeiro=True) if tid in _cat.TELAS else [])
    out = []
    for i, k in enumerate(lista, 1):
        marca = oculto if (campos is None and k in avancadas and _modo_usuario() == "iniciante") else ""
        nome = _cat.opcao(k)["nome"][L] if k in _cat.OPCOES else _nome_campo(k)
        faz_ = _cat.faz(k, _lang()) if k in _cat.OPCOES else (_HELP_DB.get(k, {}).get(_lang(), {}) or {}).get("desc", "")
        out.append((str(i) if numerar else "", nome + marca, faz_))
    return out


def _painel_tela(tid: str, opcoes: Optional[List[Tuple[str, str, str]]] = None, dica: str = "") -> Panel:
    """Painel de ajuda de uma tela: Tela, Onde, Faz, Use quando, Depois, Precisa
    de e a lista das opcoes (as da tela ou, sem lista, os campos dela)."""
    L = 1 if _lang() == "EN" else 0
    t = _cat.tela(tid)
    g = _cat.guia_tela(tid, _lang())
    R = _ROT_AJUDA
    precisa = ", ".join(p[L] for p in t["precisa"]) if t["precisa"] else ""
    linhas = [(R["tela"][L], f"[{t['tecla']}] {t['nome'][L]}", f"bold {PA}"), (R["onde"][L], _cat.trilha(tid, _lang()), PM),
              (R["faz"][L], g["faz"], PW), (R["use"][L], g["use"], PW), (R["depois"][L], g["proximo"], PW),
              (R["precisa"][L], precisa, PM),
              (R["teclas"][L], _cat.RODAPE_TELA[_lang()], PM)]
    if opcoes is None:
        opcoes = _opcoes_dos_campos(tid)
    return _painel_ajuda(t["nome"][L], linhas, opcoes=opcoes or None, dica=dica)


def _painel_ajuda_tela(tid: str) -> None:
    """O que a tela faz, quando usar e o que fazer depois (sem a lista de opcoes)."""
    console.print(_painel_tela(tid, opcoes=[]))


def _ajuda_da_tela(tid: str, opcoes: Optional[List[Tuple[str, str, str]]] = None) -> None:
    """Tecla [?] da tela: o painel de ajuda com a lista das opcoes, e pausa."""
    console.print(_painel_tela(tid, opcoes))
    _pause()


#: O que cada tecla global faz (resposta de "?" seguida de uma letra).
_TECLAS_AJUDA: Dict[str, Tuple[Tuple[str, str], Tuple[str, str]]] = {
    "?": (("Ajuda", "Help"), ("Pergunta qual opcao explicar e mostra so' ela; Enter mostra a tela inteira.",
                              "Asks which option to explain and shows only that one; Enter shows the whole screen.")),
    "G": (("Guaraci (assistente)", "Guaraci (assistant)"), ("Revisa a configuracao, explica opcoes e termos e sugere o proximo passo.",
                                                           "Reviews the configuration, explains options and terms and suggests the next step.")),
    "I": (("Idioma", "Language"), ("Troca a interface entre portugues e ingles.", "Switches the interface between Portuguese and English.")),
    "C": (("Confianca", "Trust"), ("Mostra a evidencia de validacao de cada tecnica e opcao (selo, limites, como reproduzir).",
                                   "Shows the validation evidence of each technique and option (badge, limits, how to reproduce).")),
    "R": (("Rodar", "Run"), ("Roda a analise com a configuracao atual, de qualquer tela.", "Runs the analysis with the current configuration, from any screen.")),
    "V": (("Voltar", "Back"), ("Volta um nivel (V, 0 ou Enter).", "Goes back one level (V, 0 or Enter).")),
    "+": (("Avancadas", "Advanced"), ("Mostra ou esconde as opcoes e telas avancadas nesta visita.",
                                      "Shows or hides the advanced options and screens on this visit.")),
    "M": (("Modo", "Mode"), ("Basico mostra so' o essencial; Avancado mostra tudo.", "Basic shows only the essentials; Advanced shows everything.")),
    "Q": (("Sair", "Quit"), ("No menu principal encerra o programa (pergunta antes se ha ajuste nao gravado); nas telas, volta.",
                             "On the main menu quits the program (asks first if there are unsaved changes); on screens, goes back.")),
}


def _ajuda_por_escolha(opcoes: List[Tuple[str, str, str]], geral: Any, detalhe: Any = None,
                       pausar: bool = True, tid: Optional[str] = None) -> None:
    """Tecla "?": PERGUNTA qual opcao (numero, letra ou parte do nome) e mostra so'
    ela; Enter mostra a tela inteira (`geral`). `detalhe(r)` -> True quando a tela
    tem uma ajuda propria para `r` (campo, tecnica...). Antes o painel inteiro vinha
    primeiro e poluia a tela (pedido do Erley, 2026-11-06)."""
    L = 1 if _lang() == "EN" else 0
    r = _input("  Ajuda de qual opcao? (numero ou letra; Enter = a tela inteira): " if L == 0 else
               "  Help on which option? (number or letter; Enter = the whole screen): ", ajuda=False).strip()
    if not r:
        geral()
        if pausar:
            _pause()
        return
    if detalhe is not None and detalhe(r):
        return
    R_ = _ROT_AJUDA
    op = next((o for o in opcoes if o[0] and o[0].upper() == r.upper()), None)
    if op is None and len(r) > 1:
        op = next((o for o in opcoes if r.lower() in o[1].lower()), None)
    if op is not None:
        linhas = [(("Opcao:", "Option:")[L], f"[{op[0]}] {op[1]}" if op[0] else op[1], f"bold {PA}"), (R_["faz"][L], op[2], PW)]
        if tid in _cat.TELAS:
            linhas.append((R_["tela"][L], f"[{_cat.tela(tid)['tecla']}] {_cat.tela(tid)['nome'][L]}", PM))
        console.print(_painel_ajuda(op[1], linhas))
    elif r.upper() in _TECLAS_AJUDA:
        nome, faz_ = _TECLAS_AJUDA[r.upper()]
        console.print(_painel_ajuda(nome[L], [(("Tecla:", "Key:")[L], f"[{r.upper()}] {nome[L]}", f"bold {PA}"),
                                              (R_["faz"][L], faz_[L], PW)]))
    else:
        console.print(f"  [{PM}]{_t('invalido')}[/{PM}]")
    if pausar:
        _pause()


def _ajuda_de_campo(fields_visiveis: List[str], tid: Optional[str] = None,
                    extras_ajuda: Optional[List[Any]] = None) -> None:
    """Tecla [?] das telas de configuracao: o painel da tela com TODAS as opcoes
    numeradas como na tela (campos, depois pastas e acoes); N ou o nome abre a
    ajuda completa da opcao. Fonte unica para todas as telas de configuracao.
    `extras_ajuda`: (nome, faz, funcao) de cada item depois dos campos (ou so' a funcao)."""
    L = 1 if _lang() == "EN" else 0
    opcoes = _opcoes_dos_campos(tid or "", fields_visiveis, numerar=True)
    extras = list(extras_ajuda or [])
    for j, ex in enumerate(extras, len(fields_visiveis) + 1):
        if isinstance(ex, tuple):
            opcoes.append((str(j), ex[0], ex[1]))
        else:
            opcoes.append((str(j), "", ""))
    if tid in _cat.TELAS:      # campos que a tela esconde agora (modo Basico ou modo de entrada): explicados tambem, sem numero
        oculto = " (oculta: [+] na tela mostra)" if L == 0 else " (hidden: [+] on the screen shows it)"
        opcoes += [("", _cat.opcao(k)["nome"][L] + oculto, _cat.faz(k, _lang()))
                   for k in _cat.campos_da_tela(tid, basicos_primeiro=True) if k not in fields_visiveis]
    extras_n = len(fields_visiveis)

    def _geral() -> None:
        if tid in _cat.TELAS:
            console.print(_painel_tela(tid, opcoes))
        else:
            console.print(_painel_ajuda("Ajuda" if L == 0 else "Help", [], opcoes=opcoes))

    def _detalhe(r2: str) -> bool:
        if r2.isdigit() and 1 <= int(r2) <= extras_n:
            _mostrar_ajuda(fields_visiveis[int(r2) - 1])
            return True
        if r2.isdigit() and extras and 0 <= int(r2) - extras_n - 1 < len(extras):
            ex = extras[int(r2) - extras_n - 1]      # pasta ou acao numerada depois dos campos
            (ex[2] if isinstance(ex, tuple) else ex)()
            return True
        if r2 in _HELP_DB or r2 in _SPEC_BY_KEY:
            _mostrar_ajuda(r2)
            return True
        if len(r2) > 1:
            todos = list(fields_visiveis) + ([k for k in _cat.campos_da_tela(tid, basicos_primeiro=True) if k not in fields_visiveis]
                                             if tid in _cat.TELAS else [])
            found = [k for k in todos if r2.lower() in k.lower() or r2.lower() in _nome_campo(k).lower()]
            if found:
                _mostrar_ajuda(found[0])
                return True
        return False

    _ajuda_por_escolha(opcoes, _geral, _detalhe, tid=tid)


def _ajuda_grupo(num: str, itens: List[Dict[str, Any]], n_ocultas: int = 0, mostrando: bool = False) -> None:
    """[?] num menu de grupo: o painel do grupo com cada tela (o que faz e quando
    usar); N abre o painel completo da tela N."""
    L = 1 if _lang() == "EN" else 0
    g = next(x for x in _cat.grupos_cli() if x["num"] == num)
    R = _ROT_AJUDA
    opcoes = []
    for x in itens:
        guia = _cat.guia_tela(x["id"], _lang())
        marca = (" (avancada)" if L == 0 else " (advanced)") if x["nivel"] == "avancado" else ""
        opcoes.append((x["tecla"].split(".")[1], x["nome"][L] + marca,
                       f"{guia['faz']} {'Use quando' if L == 0 else 'Use when'} {guia['use']}"))
    if n_ocultas or mostrando:
        opcoes.append(("+", "Mostrar/ocultar telas avancadas" if L == 0 else "Show/hide advanced screens",
                       (f"{n_ocultas} oculta(s) no modo Basico; o numero delas tambem abre direto." if L == 0 else
                        f"{n_ocultas} hidden in Basic mode; their number also opens them directly.") if n_ocultas else ""))
    linhas = [(R["menu"][L], f"[{num}] {g['nome'][L]}", f"bold {PA}"), (R["faz"][L], g["resumo"][L], PW),
              (R["teclas"][L], _cat.RODAPE_TELA[_lang()], PM)]
    por_num = {x["tecla"].split(".")[1]: x["id"] for x in itens}

    def _detalhe(r: str) -> bool:
        alvo = por_num.get(r) or (next((x["id"] for x in itens if r.lower() in x["nome"][L].lower()), None) if len(r) > 1 else None)
        if alvo:
            _ajuda_da_tela(alvo, opcoes=[])      # so' a tela pedida: o que faz, quando usar, depois
        return bool(alvo)

    _ajuda_por_escolha(opcoes, lambda: console.print(_painel_ajuda(g["nome"][L], linhas, opcoes=opcoes)), _detalhe)


def _ajuda_menu_principal(cfg: Optional[Config] = None) -> None:
    """[?] no menu principal: o painel do menu (grupos e teclas). N abre o painel
    do grupo N; A abre o indice de todos os campos (a ajuda navegavel)."""
    L = 1 if _lang() == "EN" else 0
    R = _ROT_AJUDA
    opcoes = [(g["num"], g["nome"][L], g["resumo"][L]) for g in _cat.grupos_cli()]
    opcoes += [
        ("R", "Rodar analise" if L == 0 else "Run analysis",
         "Roda o pipeline com a configuracao atual." if L == 0 else "Runs the pipeline with the current configuration."),
        ("G", "Guaraci (assistente)" if L == 0 else "Guaraci (assistant)",
         "Revisa a configuracao, explica termos e sugere o proximo passo." if L == 0
         else "Reviews the configuration, explains terms and suggests the next step."),
        ("C", "Confianca" if L == 0 else "Trust",
         "Mostra a evidencia de validacao de cada tecnica e opcao." if L == 0
         else "Shows the validation evidence of each technique and option."),
        ("I", "Idioma" if L == 0 else "Language", "Troca entre portugues e ingles." if L == 0 else "Switches between Portuguese and English."),
        ("M", "Modo" if L == 0 else "Mode",
         "Basico mostra so' as telas e opcoes essenciais; Avancado mostra todas." if L == 0
         else "Basic shows only the essential screens and options; Advanced shows all."),
        ("N.M", "Atalho" if L == 0 else "Shortcut", "Ex.: 2.3 abre direto a tela Dados." if L == 0 else "E.g. 2.3 opens the Data screen directly."),
        ("A", "Indice de campos" if L == 0 else "Field index",
         "Lista todos os campos de configuracao com a ajuda completa de cada um." if L == 0
         else "Lists every configuration field with its full help."),
        ("Q", "Sair" if L == 0 else "Quit", "Encerra o programa (pergunta antes se ha ajuste nao gravado)." if L == 0
         else "Quits the program (asks first if there are unsaved changes)."),
    ]
    linhas = [(R["menu"][L], "Menu principal" if L == 0 else "Main menu", f"bold {PA}"),
              (R["faz"][L], "Organiza o programa em 6 grupos, na ordem de um estudo: comecar, preparar os dados, modelar, "
                            "comparar, relatar e conferir." if L == 0 else
                            "Organises the program in 6 groups, in the order of a study: start, prepare the data, model, "
                            "compare, report and check.", PW),
              (R["use"][L], "nao souber por onde ir: comece pelo grupo 1 (exemplo guiado ou perfis prontos)." if L == 0
               else "you do not know where to go: start with group 1 (guided example or ready profiles).", PW),
              (R["teclas"][L], _cat.RODAPE_MENU[_lang()], PM)]
    def _detalhe(r: str) -> bool:
        if r in {g["num"] for g in _cat.grupos_cli()}:     # o grupo: as telas dele, uma linha cada
            gr = next(g for g in _cat.grupos_cli() if g["num"] == r)
            ops = [(x["tecla"].split(".")[1], x["nome"][L], x["curta"][L]) for x in _telas_visiveis_do_grupo(r, cfg or Config(), basico=False)]
            console.print(_painel_ajuda(gr["nome"][L], [(R["menu"][L], f"[{r}] {gr['nome'][L]}", f"bold {PA}"),
                                                        (R["faz"][L], gr["resumo"][L], PW)], opcoes=ops))
            _pause()
            return True
        if r.upper() == "A":
            _menu_help(cfg)
            return True
        return False

    _ajuda_por_escolha(opcoes, lambda: console.print(_painel_ajuda("Ajuda" if L == 0 else "Help", linhas, opcoes=opcoes)), _detalhe)


class _SairDoPrograma(Exception):
    """Tecla [s] em qualquer tela: volta ao menu principal e pede para sair."""


def _cabecalho_tela(tid: str, cfg: Optional[Config] = None, *, limpar: bool = True) -> None:
    """Molde unico de tela: banner, trilha, o que a tela faz e o que ela precisa."""
    if limpar:
        _cls()
        _print_header(cfg)
    L = 1 if _lang() == "EN" else 0
    t = _cat.tela(tid)
    console.print(f"  [{PM}]{escape(_cat.trilha(tid, _lang()))}[/{PM}]")
    console.print(f"  [{PW}]{escape(t['faz'][L])}[/{PW}]")
    if t["precisa"]:
        console.print(f"  [{PM}]{'Voce vai precisar de' if L == 0 else 'You will need'}: "
                      f"{escape(', '.join(p[L] for p in t['precisa']))}[/{PM}]")


def _rodape_tela() -> None:
    console.print(f"  [{PM}]{escape(_cat.RODAPE_TELA[_lang()])}[/{PM}]")


def _tecla_valor(raw: str, tid: str) -> Optional[str]:
    """Prompt de VALOR (classe, coluna, caminho): o texto e' o valor. So' Enter
    vazio volta e `?` abre a ajuda da tela; letras como S, Q ou G nao sao
    comandos aqui (uma classe chamada S nao pode encerrar o programa). Um `0`
    sozinho tambem cancela: e' a saida dos assistentes de varios passos e nao e'
    nome razoavel de classe, coluna ou arquivo."""
    r = raw.strip()
    if r in ("", "0"):
        _VOLTOU[0] = True
        return "voltar"
    if r == "?":
        _ajuda_da_tela(tid)
        return "tratada"
    return None


def _tecla_comum(raw: str, tid: str, cfg: Config, campos: Optional[List[str]] = None,
                 extras_ajuda: Optional[List[Any]] = None) -> Optional[str]:
    """Teclas globais de um prompt de MENU, iguais em toda tela: `?` ajuda,
    `G` Guaraci, `I` idioma, `C` confianca, `V` (ou `0`/Enter) volta. Sair do
    programa e' so' `Q` no menu principal; aqui `Q` apenas volta."""
    r = raw.strip().upper()
    if r in ("0", "Q", "", "V"):
        _VOLTOU[0] = True
        return "voltar"      # [+] (mostrar avancadas) e' tratado antes, em _loop_menu
    if r == "I":
        _toggle_idioma()
        return "tratada"
    if r in ("G", "A"):
        _abrir_assistente(_t("t_" + next((k for k, v in _CHAVE_TELA.items() if v == tid), "ajuda")), cfg, tid)
        return "tratada"
    if r == "C":
        _abrir_confianca(tid, cfg)
        return "tratada"
    if r == "R":
        _rodar_pipeline(cfg)         # acao principal, a um toque de qualquer tela
        return "tratada"
    if r == "?":
        _ajuda_de_campo(campos or [], tid, extras_ajuda)
        return "tratada"
    if r.startswith("?") and r[1:].strip().isdigit():     # ?N = ajuda do campo N (ou da pasta/acao N)
        n = int(r[1:].strip())
        nc = len(campos or [])
        if campos and 1 <= n <= nc:
            _mostrar_ajuda(campos[n - 1])
        elif extras_ajuda and 0 <= n - nc - 1 < len(extras_ajuda):
            ex = extras_ajuda[n - nc - 1]          # (nome, faz, funcao) ou so' a funcao
            (ex[2] if isinstance(ex, tuple) else ex)()
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]")
            _pause()
        return "tratada"
    return None


def _titulo_tela(tela: str) -> str:
    return _t("t_" + next(k for k, v in _CHAVE_TELA.items() if v == tela))


def _ajuda_pasta(sec: Dict[str, Any], campos_sec: List[str]) -> None:
    """Ajuda de uma subpasta: o painel com o que cada opcao dentro dela faz e quando usar."""
    L = 1 if _lang() == "EN" else 0
    R = _ROT_AJUDA
    def _faz_k(k: str) -> str:
        return f"{_cat.opcao(k)['faz'][L]} {'Use quando' if L == 0 else 'Use when'} {_cat.opcao(k)['use'][L]}"
    opcoes = [(str(i), _cat.opcao(k)["nome"][L], _faz_k(k)) for i, k in enumerate(campos_sec, 1)]
    oculta = " (oculta: nao se aplica ao modo de entrada atual)" if L == 0 else " (hidden: does not apply to the current input mode)"
    opcoes += [("", _cat.opcao(k)["nome"][L] + oculta, _faz_k(k)) for k in sec.get("campos", []) if k not in campos_sec]
    console.print(_painel_ajuda(sec["nome"][L], [(R["menu"][L], sec["nome"][L], f"bold {PA}"), (R["faz"][L], sec["resumo"][L], PW)],
                                opcoes=opcoes))
    _pause()


def _ajuda_acao(rotulo: str, tela: str, faz_: str = "") -> None:
    """Ajuda de uma acao numerada da tela: o nome dela, o que faz e a tela onde esta."""
    L = 1 if _lang() == "EN" else 0
    R = _ROT_AJUDA
    t = _cat.tela(tela)
    console.print(_painel_ajuda(rotulo, [(R["campo"][L], rotulo, f"bold {PA}"),
                                         (R["faz"][L], faz_ or ("Acao desta tela." if L == 0 else "Action of this screen."), PW),
                                         (R["tela"][L], f"[{t['tecla']}] {t['nome'][L]}", PM)]))
    _pause()


def _loop_secao(tela: str, sec: Dict[str, Any], campos_sec: List[str], cfg: Config) -> None:
    """Subpasta de opcoes avancadas de uma tela densa (ex.: Modelagem › Selecao
    de variaveis): so' os campos dela, numerados de 1; V volta para a tela."""
    while True:
        L = 1 if _lang() == "EN" else 0
        _cabecalho_tela(tela, cfg)
        vis = _print_submenu_compact(f"{_titulo_tela(tela)} › {sec['nome'][L]}", "", campos_sec, cfg, None,
                                     campos_avancados=set(campos_sec), mostrar_avancado=True)
        raw = _input(f"\n  {_t('opcao')}: ", ajuda=False)
        r = _tecla_comum(raw, tela, cfg, vis)
        if r == "voltar":
            return
        if r == "tratada":
            continue
        if raw.strip().isdigit() and 1 <= int(raw.strip()) <= len(vis):
            _editar_campo(cfg, vis[int(raw.strip()) - 1])
            _pause()
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]")
            _pause()


def _loop_menu(tela: str, cfg: Config,
               extras: Optional[List[Tuple[str, str]]] = None,
               on_extra: Optional[Dict[str, Any]] = None,
               antes: Optional[Any] = None,
               faz_extras: Optional[Dict[str, Tuple[str, str]]] = None) -> None:
    """Loop unico das telas de configuracao. Titulo, descricao, campos e quais
    campos sao avancados vem do catalogo (`tela`). `antes()` desenha um painel
    proprio da tela depois do cabecalho. O reveal ("+") e' local a visita: sair
    e voltar a entrar esconde os avancados de novo para o modo Iniciante.

    Telas densas (catalogo.SECOES_TELA: Dados, Modelagem, Validacao) mostram os
    campos basicos na lista e as opcoes avancadas em subpastas numeradas depois
    deles; as acoes da tela vem por ultimo."""
    todos = _cat.campos_da_tela(tela, basicos_primeiro=True)   # numeracao estavel entre os modos
    avancados = set(_cat.campos_avancados(tela))
    secoes = _cat.secoes_da_tela(tela)
    mostrar_avancado = False
    while True:
        L = 1 if _lang() == "EN" else 0
        # Opcoes que nao se aplicam ao modo de entrada atual ficam ocultas ([+] mostra todas).
        campos = todos if mostrar_avancado else _contexto.campos_visiveis(cfg, todos)
        n_ctx = len(todos) - len(campos)
        _cabecalho_tela(tela, cfg)
        if n_ctx:
            console.print(f"  [{PM}]{n_ctx} opcao(oes) oculta(s): nao se aplicam ao modo de entrada "
                          f"'{escape(str(getattr(cfg, 'mode', '')))}'.[/{PM}]" if _lang() == "PT" else
                          f"  [{PM}]{n_ctx} option(s) hidden: they do not apply to input mode "
                          f"'{escape(str(getattr(cfg, 'mode', '')))}'.[/{PM}]")
        if antes:
            antes()
        com_pastas = bool(secoes) and (mostrar_avancado or _modo_usuario() != "iniciante")
        pastas: List[Tuple[Dict[str, Any], List[str]]] = []
        if com_pastas:
            pastas = [(sec, [c for c in sec["campos"] if c in campos]) for sec in secoes]
            pastas = [(sec, cs) for sec, cs in pastas if cs]
            def _n_op(n: int) -> str:
                return f"{n} {('opcao' if n == 1 else 'opcoes') if L == 0 else ('option' if n == 1 else 'options')}"
            itens = [(f"pasta:{sec['id']}", f"+{sec['nome'][L]}", _n_op(len(cs)), sec["resumo"][L])
                     for sec, cs in pastas] + list(extras or [])
            visiveis = _print_submenu_compact(_titulo_tela(tela), "", [c for c in campos if c not in avancados], cfg, itens)
            if mostrar_avancado and _modo_usuario() == "iniciante":
                console.print(f"  [{PR}][+][/{PR}] [{PM}]{'Ocultar as pastas avancadas' if L == 0 else 'Hide the advanced folders'}[/{PM}]")
        else:
            itens = list(extras or [])
            visiveis = _print_submenu_compact(
                _titulo_tela(tela), "", campos, cfg, itens or None,
                campos_avancados=avancados or None, mostrar_avancado=mostrar_avancado)
        raw = _input(f"\n  {_t('opcao')}: ", ajuda=False)
        # ? e ?N tambem explicam as pastas e as acoes numeradas depois dos campos: (nome, faz, abrir ajuda)
        ajudas: List[Any] = []
        for it_ in itens:
            if str(it_[0]).startswith("pasta:"):
                sc, cs = next((a, b) for a, b in pastas if f"pasta:{a['id']}" == it_[0])
                ajudas.append((sc["nome"][L], f"{sc['resumo'][L]} ({list(it_)[2]})",
                               (lambda a, b: lambda: _ajuda_pasta(a, b))(sc, cs)))
            else:
                rot_ = str(it_[1])
                fz_ = (faz_extras or {}).get(str(it_[0]), ("", ""))[L]
                ajudas.append((rot_, fz_, (lambda r_, f_: lambda: _ajuda_acao(r_, tela, f_))(rot_, fz_)))
        if raw.strip() == "+" and (avancados or n_ctx or mostrar_avancado):
            mostrar_avancado = not mostrar_avancado
            continue
        if itens and raw.strip().isdigit():
            j = int(raw.strip()) - len(visiveis) - 1      # pastas e acoes da tela vem depois dos campos
            if 0 <= j < len(itens):
                chave = itens[j][0]
                if chave.startswith("pasta:"):
                    sec, cs = next((sc, c) for sc, c in pastas if f"pasta:{sc['id']}" == chave)
                    _loop_secao(tela, sec, cs, cfg)
                    continue
                if on_extra and chave in on_extra:
                    on_extra[chave]()
                    continue
        r = _tecla_comum(raw, tela, cfg, visiveis, ajudas)
        if r == "voltar":
            break
        if r == "tratada":
            continue
        if raw.isdigit() and 1 <= int(raw) <= len(visiveis):
            _editar_campo(cfg, visiveis[int(raw) - 1])
            _pause()
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]")
            _pause()


def _menu_project(cfg: Config) -> None:
    _loop_menu("projeto", cfg)


_DIR_PERFIS_COMBINADOS = _USER_DIR / "perfis_matriz"


def _salvar_perfil_combinado(cfg: Config) -> None:
    """Funde `perfil_matriz` + `perfil_tecnica` (Agente 5B) e salva como
    YAML de usuario, pronto pra reusar digitando o caminho no campo
    `perfil_matriz` de uma proxima sessao."""
    lang = _lang(); is_pt = lang == "PT"
    nome_matriz = str(getattr(cfg, "matrix_profile", "") or "")
    nome_tecnica = str(getattr(cfg, "acquisition_profile", "") or "")
    if not nome_matriz:
        console.print(f"  [{PR}]{'Defina o Perfil de tecnica antes de combinar.' if is_pt else 'Set the technique profile before combining.'}[/{PR}]")
        _pause(); return
    if not nome_tecnica:
        console.print(f"  [{PR}]{'Defina o Perfil de aquisicao de imagem antes de combinar.' if is_pt else 'Set the image acquisition profile before combining.'}[/{PR}]")
        _pause(); return

    from guaraci.perfil_matriz import (UnknownProfileError, combine_profiles,
                                        load_profile, save_profile)
    try:
        matriz = load_profile(nome_matriz)
        tecnica = load_profile(nome_tecnica)
    except UnknownProfileError as e:
        console.print(f"  [{PR}]{escape(str(e))}[/{PR}]"); _pause(); return

    nome_novo = _ask(
        f"  [{PA}]{'Nome do perfil combinado (ex.: mel_celular)' if is_pt else 'Combined profile name (e.g. honey_phone)'}: [/{PA}]"
    ).strip()
    if not nome_novo:
        console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); _pause(); return

    combinado = combine_profiles(nome_novo, matriz, tecnica)
    caminho = _DIR_PERFIS_COMBINADOS / f"{nome_novo}.yaml"
    try:
        save_profile(combinado, str(caminho))
    except OSError as e:
        console.print(f"  [{PR}]{'Erro ao salvar' if is_pt else 'Error saving'}: {escape(str(e))}[/{PR}]")
        _pause(); return

    msg = (f"✓ Perfil combinado salvo em {caminho}\n"
           f"  Use digitando o caminho no campo 'Perfil de tecnica' numa proxima sessao."
           if is_pt else
           f"✓ Combined profile saved to {caminho}\n"
           f"  Use it by typing the path in the 'Technique profile' field in a future session.")
    console.print(f"  [g]{escape(msg)}[/g]")
    _pause()


def _menu_data(cfg: Config) -> None:
    # Acao da tela, numerada depois dos campos: fundir perfil de matriz e de tecnica.
    extra_lbl = ("Fundir perfis (matriz + tecnica)" if _lang() == "PT"
                 else "Merge profiles (matrix + technique)")
    _loop_menu("dados", cfg, extras=[("F", extra_lbl)],
               on_extra={"F": lambda: _salvar_perfil_combinado(cfg)},
               faz_extras={"F": ("Junta o perfil de tecnica e o de aquisicao num perfil novo, gravado em YAML para reusar "
                                 "no campo Perfil de tecnica de uma proxima sessao.",
                                 "Merges the technique profile and the acquisition profile into a new profile, saved as YAML "
                                 "to reuse in the Technique profile field of a later session.")})


def _menu_preprocessing(cfg: Config) -> None:
    def _show_pipeline():
        preproc = str(_cfgv(cfg, "pre_processamento", "msc_sg_mc"))
        comps = {
            "msc": "MSC — Multiplicative Scatter Correction",
            "snv": "SNV — Standard Normal Variate",
            "sg":  "SG  — Savitzky-Golay",
            "mc":  "MC  — Mean-Centering",
        }
        partes = [c for c in ["msc", "snv", "sg", "mc"] if c in preproc.lower()]
        t = Table(box=None, show_header=False, padding=(0, 1))
        t.add_column("I", style=PG, width=2)
        t.add_column("D", style=PW)
        for p in partes:
            t.add_row("✓", comps.get(p, p))
        lbl = "Pipeline ativo" if _lang() == "PT" else "Active pipeline"
        console.print(Panel(t, title=f"[{PS}]{lbl}: [{PA}]{escape(preproc)}[/{PA}][/{PS}]",
                            border_style=PS, box=rbox.SIMPLE, padding=(0, 1)))

    _loop_menu("preproc", cfg, antes=_show_pipeline)


def _menu_modeling(cfg: Config) -> None:
    _loop_menu("modelo", cfg)


def _menu_validation(cfg: Config) -> None:
    def _alerta_leakage():
        if not _cfgv(cfg, "validacao_group_aware", True):
            console.print(Panel(
                "[err]  GroupKFold DESATIVADO — risco de data leakage![/err]\n"
                f"[err]  Ative o campo [{_cat.campos_da_tela('validacao', basicos_primeiro=True).index('validacao_group_aware') + 1}] imediatamente.[/err]",
                border_style=PR, box=rbox.HEAVY, padding=(0, 1)))

    _loop_menu("validacao", cfg, antes=_alerta_leakage)


def _menu_advanced(cfg: Config) -> None:
    def _aviso_pesado():
        console.print(Panel(
            f"[{PR}]  ▲ " + ("Modulos pesados — verificar hardware em [6.4] antes de ativar." if _lang() == "PT"
                            else "Heavy modules — check the hardware in [6.4] before enabling.") + f"[/{PR}]",
            border_style=PR, box=rbox.SIMPLE, padding=(0, 1)))

    _loop_menu("avancado", cfg, antes=_aviso_pesado)


# ---------------------------------------------------------------------------
# VISUALIZACAO — submenu especial com sub-handlers
# ---------------------------------------------------------------------------
#: O que cada acao da tela Visualizacao faz (ajuda "?" e "?N" da tela).
_FAZ_VISUAL: Dict[str, Tuple[str, str]] = {
    "P": ("Escolhe as cores das figuras: qualitativa, segura para daltonismo, cinza, viridis, publicacao ou escura.",
          "Chooses the figure colours: qualitative, colour-blind safe, grey, viridis, publication or dark."),
    "F": ("Muda o tamanho do texto das figuras, de XS (muito pequeno) a XL (conferencias).",
          "Changes the figure text size, from XS (very small) to XL (conferences)."),
    "D": ("Liga ou desliga a grade principal e a secundaria, o estilo da linha e a transparencia dela.",
          "Turns the major and minor grid on or off, and sets its line style and transparency."),
    "T": ("Ajusta a transparencia dos pontos nos graficos de dispersao (opaco, automatico ou translucido).",
          "Sets the point transparency in scatter plots (opaque, automatic or translucent)."),
}


def _menu_visualization(cfg: Config) -> None:
    # Acoes proprias desta tela, numeradas depois dos campos: paleta, fonte,
    # grade e transparencia (letras so' para teclas globais).
    extras = [
        ("P", _t("viz_paleta")), ("F", _t("viz_fonte")),
        ("D", _t("viz_grid")),   ("T", _t("viz_alpha")),
    ]

    def _pal():
        vcfg = _carregar_visual_cfg()
        t = Table(box=None, show_header=False, padding=(0, 1))
        t.add_column("N", no_wrap=True, width=4)
        t.add_column("Nome", no_wrap=True, width=30)
        t.add_column("Desc", style=PM)
        atual = vcfg.get("paleta", "qualitativo")
        for i, (pk, pd) in enumerate(_PALETAS_COR.items(), 1):
            txt_l = pd.get(_lang()) or pd.get("PT") or {}
            nm = txt_l.get("nome", pk)
            dsc = txt_l.get("desc", "")
            mk = f"[{PA}]►[/{PA}]" if pk == atual else " "
            t.add_row(f"  [{PA}][{i}][/{PA}]", f"{mk} {escape(nm)}", escape(_trunc(dsc, 35)))
        console.print(Panel(t, title=f"[bold {PA}]{_t('viz_paleta')}[/bold {PA}]",
                            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)))
        _ops = [(str(i), (pd.get(_lang()) or pd.get("PT") or {}).get("nome", pk), (pd.get(_lang()) or pd.get("PT") or {}).get("desc", ""))
                for i, (pk, pd) in enumerate(_PALETAS_COR.items(), 1)]
        with _ajuda_local(_t("viz_paleta"), _FAZ_VISUAL["P"][1 if _lang() == "EN" else 0], _ops):
            r = _input(f"  [1-{len(_PALETAS_COR)}] {'ou Enter' if _lang() == 'PT' else 'or Enter'}: ")
        if r.isdigit():
            idx = int(r) - 1
            if 0 <= idx < len(_PALETAS_COR):
                vcfg["paleta"] = list(_PALETAS_COR.keys())[idx]
                _salvar_visual_cfg(vcfg)
                _lbl = "Paleta" if _lang() == "PT" else "Palette"
                _nm_pal = (_PALETAS_COR[vcfg["paleta"]].get(_lang()) or {}).get("nome", vcfg["paleta"])
                console.print(f"  [g]✓ {_lbl}: {escape(_nm_pal)}[/g]")

    def _fonte():
        vcfg = _carregar_visual_cfg()
        presets = [
            ("xs", "XS — Muito pequeno" if _lang()=="PT" else "XS — Very small"),
            ("s",  "S  — Pequeno" if _lang()=="PT" else "S  — Small"),
            ("m",  "M  — Medio (padrao)" if _lang()=="PT" else "M  — Medium (default)"),
            ("l",  "L  — Grande (apresentacoes)" if _lang()=="PT" else "L  — Large (presentations)"),
            ("xl", "XL — Muito grande (conferencias)" if _lang()=="PT" else "XL — Extra large (conferences)"),
        ]
        t = Table(box=None, show_header=False, padding=(0, 1))
        t.add_column("N", no_wrap=True, width=4)
        t.add_column("Desc", style=PW)
        atual = vcfg.get("tamanho_fonte", "m")
        for i, (fk, fd) in enumerate(presets, 1):
            mk = f"[{PA}]►[/{PA}]" if fk == atual else " "
            t.add_row(f"  [{PA}][{i}][/{PA}]", f"{mk} {escape(fd)}")
        console.print(Panel(t, title=f"[bold {PA}]{_t('viz_fonte')}[/bold {PA}]",
                            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)))
        with _ajuda_local(_t("viz_fonte"), _FAZ_VISUAL["F"][1 if _lang() == "EN" else 0],
                          [(str(i), fd, "") for i, (_fk, fd) in enumerate(presets, 1)]):
            r = _input(f"  [1-5] {'ou Enter' if _lang() == 'PT' else 'or Enter'}: ")
        if r.isdigit() and 1 <= int(r) <= 5:
            vcfg["tamanho_fonte"] = presets[int(r)-1][0]
            _salvar_visual_cfg(vcfg)
            _lbl = "Fonte" if _lang() == "PT" else "Font"
            console.print(f"  [g]✓ {_lbl}: {vcfg['tamanho_fonte']}[/g]")

    def _grid():
        vcfg = _carregar_visual_cfg()
        gm = vcfg.get("grid_major", True)
        gmi = vcfg.get("grid_minor", False)
        gs = vcfg.get("grid_style", "dotted")
        ga = vcfg.get("grid_alpha", 0.4)
        t = Table(box=None, show_header=False, padding=(0, 1))
        t.add_column("N", no_wrap=True, width=4)
        t.add_column("Opcao", no_wrap=True, width=26)
        t.add_column("Valor", style=PS)
        pt_ = _lang() == "PT"
        _estilo_txt = ({"solid": "continua", "dotted": "pontilhada", "dashed": "tracejada"} if pt_ else {}).get(gs, gs)
        t.add_row(f"  [{PA}][1][/{PA}]", "Grade principal" if pt_ else "Major grid", "[g]ON[/g]" if gm else "[m]OFF[/m]")
        t.add_row(f"  [{PA}][2][/{PA}]", "Grade secundaria" if pt_ else "Minor grid", "[g]ON[/g]" if gmi else "[m]OFF[/m]")
        t.add_row(f"  [{PA}][3][/{PA}]", "Estilo" if pt_ else "Style", escape(_estilo_txt))
        t.add_row(f"  [{PA}][4][/{PA}]", "Transparencia" if pt_ else "Transparency", str(ga))
        console.print(Panel(t, title=f"[bold {PA}]{_t('viz_grid')}[/bold {PA}]",
                            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)))
        _ops_g = [("1", "Grade principal" if pt_ else "Major grid", "Liga/desliga as linhas nas marcas principais." if pt_ else "Turns the lines at the major ticks on/off."),
                  ("2", "Grade secundaria" if pt_ else "Minor grid", "Liga/desliga as linhas nas marcas menores." if pt_ else "Turns the lines at the minor ticks on/off."),
                  ("3", "Estilo" if pt_ else "Style", "Alterna continua, pontilhada e tracejada." if pt_ else "Cycles solid, dotted and dashed."),
                  ("4", "Transparencia" if pt_ else "Transparency", "Valor de 0.1 (quase invisivel) a 0.9 (forte)." if pt_ else "Value from 0.1 (almost invisible) to 0.9 (strong).")]
        with _ajuda_local(_t("viz_grid"), _FAZ_VISUAL["D"][0 if pt_ else 1], _ops_g):
            r = _input(f"  [1-4] {'ou Enter' if pt_ else 'or Enter'}: ")
        if r == "1": vcfg["grid_major"] = not gm
        elif r == "2": vcfg["grid_minor"] = not gmi
        elif r == "3":
            ests = ["solid","dotted","dashed"]
            vcfg["grid_style"] = ests[(ests.index(gs)+1)%3] if gs in ests else "dotted"
        elif r == "4":
            # Faixa checada ANTES de gravar: um valor fora de 0-1 ficava
            # salvo em ~/.guaraci e toda figura das execucoes seguintes
            # falhava com "alpha outside 0-1 range" (achado B4, 2026-09-24).
            bruto = _input(f"  {'Valor' if pt_ else 'Value'} [0.1-0.9]: ").strip().replace(",", ".")
            try:
                valor = float(bruto)
            except ValueError:
                valor = None
            if valor is not None and 0.0 < valor <= 1.0:
                vcfg["grid_alpha"] = valor
            elif bruto:
                console.print(f"  [err]{'Use um numero entre 0.1 e 0.9 (ex.: 0.4). Valor mantido.' if _lang() == 'PT' else 'Use a number between 0.1 and 0.9 (e.g. 0.4). Value kept.'}[/err]")
        _salvar_visual_cfg(vcfg)

    def _alpha():
        vcfg = _carregar_visual_cfg()
        ops = [
            ("baixo", "0.9 — Opacos" if _lang()=="PT" else "0.9 — Opaque"),
            ("medio", "Automatico — ajusta a densidade de pontos (padrao)" if _lang()=="PT"
                      else "Automatic — adapts to point density (default)"),
            ("alto",  "0.35 — Translucido" if _lang()=="PT" else "0.35 — Translucent"),
        ]
        t = Table(box=None, show_header=False, padding=(0, 1))
        t.add_column("N", no_wrap=True, width=4)
        t.add_column("Desc", style=PW)
        atual = vcfg.get("alpha_pontos", "medio")
        for i, (ak, ad) in enumerate(ops, 1):
            mk = f"[{PA}]►[/{PA}]" if ak == atual else " "
            t.add_row(f"  [{PA}][{i}][/{PA}]", f"{mk} {escape(ad)}")
        console.print(Panel(t, title=f"[bold {PA}]{_t('viz_alpha')}[/bold {PA}]",
                            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)))
        with _ajuda_local(_t("viz_alpha"), _FAZ_VISUAL["T"][1 if _lang() == "EN" else 0],
                          [(str(i), ad, "") for i, (_ak, ad) in enumerate(ops, 1)]):
            r = _input(f"  [1-3] {'ou Enter' if _lang() == 'PT' else 'or Enter'}: ")
        if r in ("1","2","3"):
            vcfg["alpha_pontos"] = ops[int(r)-1][0]
            _salvar_visual_cfg(vcfg)

    def _com_pausa(fn):
        def _chama() -> None:
            fn()
            _pause()
        return _chama

    _loop_menu("visualizacao", cfg, extras=extras,
               on_extra={"P": _com_pausa(_pal), "F": _com_pausa(_fonte),
                         "D": _com_pausa(_grid), "T": _com_pausa(_alpha)},
               faz_extras=_FAZ_VISUAL)


# ---------------------------------------------------------------------------
# TECNICA ANALITICA
# ---------------------------------------------------------------------------
# Agrupamento das tecnicas por categoria (segue o modelo do prompt GUARACI).
# So inclui chaves presentes em _TECNICAS; chaves ausentes sao ignoradas.
_TECNICA_CATEGORIAS = [
    ("Vibracional",          "Vibrational",        ["ft-nir", "nir", "mir", "raman", "uv-vis"]),
    ("Luminescencia",        "Luminescence",       ["fluorescencia", "eem"]),
    ("Cromatografia",        "Chromatography",     ["hplc", "gc-ms"]),
    ("Elementar",            "Elemental",          ["icp"]),
    ("Imagem",               "Imaging",            ["imagem", "hsi"]),
    ("Ressonancia / Outras", "Resonance / Others", ["nmr", "ims", "generico"]),
]


def _tecnica_ordem() -> list:
    """Retorna a lista achatada de chaves de tecnica na ordem das categorias."""
    ordem = []
    vistos = set()
    for _pt, _en, keys in _TECNICA_CATEGORIAS:
        for k in keys:
            if k in _TECNICAS and k not in vistos:
                ordem.append(k); vistos.add(k)
    # Acrescenta quaisquer tecnicas nao categorizadas, ao final
    for k in _TECNICAS:
        if k not in vistos:
            ordem.append(k); vistos.add(k)
    return ordem


#: Texto do eixo mostrado pela tela [2.4] ao aplicar a tecnica (o mesmo rotulo das figuras, sem mathtext).
_ROTULO_UNIDADE_EIXO = {"cm-1": "Wavenumber (cm-1)", "nm": "Wavelength (nm)", "raman_shift": "Raman shift (cm-1)",
                        "min": "Retention time (min)", "ppm": "Chemical shift (ppm)", "ms": "Drift time (ms)",
                        "indice": "Variable index"}


def _formatos_tecnica(tk: str, lang: str) -> List[Tuple[str, str]]:
    """(modo_entrada, rotulo) aceitos pela tecnica; o primeiro e' o padrao."""
    ins = _cat.INSTRUMENTOS.get(tk, {})
    L = 1 if lang == "EN" else 0
    fs = ins.get("formatos") or ((ins.get("mode", "dx"), ("", "")),)
    return [(m, rot[L] or m) for m, rot in fs]


def _onde_dados(mode: str, lang: str) -> str:
    from guaraci.catalogo_tecnicas import ONDE_DADOS, ONDE_DADOS_PADRAO
    return ONDE_DADOS.get(mode, ONDE_DADOS_PADRAO)[1 if lang == "EN" else 0]


def _tecnica_detalhe(tk: str, lang: str) -> None:
    """Painel de ajuda (molde unico) com os detalhes completos de uma tecnica."""
    td = _TECNICAS.get(tk, {})
    tdl = td.get(lang, td.get("PT", {}))
    L = 1 if lang == "EN" else 0
    R = _ROT_AJUDA
    nome = tdl.get("nome", tk)
    linhas = [(("Tecnica:", "Technique:")[L], nome, f"bold {PA}"), (R["faz"][L], tdl.get("desc", ""), PW),
              (R["faixa"][L], str(tdl.get("faixa", "—")), PM),
              (("Pre-proc.:", "Preproc.:")[L], str(tdl.get("preproc_rec", td.get("preproc", "—"))), PS),
              *((((("Exclui:", "Excludes:")[L]), ", ".join(td["faixas_excluir"]) + (" (agua residual, TSP/TMS)" if L == 0 else " (residual water, TSP/TMS)"), PM),)
                if td.get("faixas_excluir") else ()),
              (("Arquivos:", "Files:")[L], " · ".join((rot + (" " + _extras.sufixo_menu(_m, lang) if _extras.sufixo_menu(_m, lang) else ""))
                                                   for _m, rot in _formatos_tecnica(tk, lang)), PW)]
    console.print(_painel_ajuda(nome, linhas))
    _pause()


def _menu_technique(cfg: Config) -> None:
    """Tecnica analitica — agrupada por categoria (modelo GUARACI)."""
    lang = _lang()

    def _aplicar(tk_sel: str) -> None:
        td_sel = _TECNICAS.get(tk_sel, {})
        tdl = td_sel.get(lang, td_sel.get("PT", {}))
        try:
            nm_sel = tdl.get("nome", tk_sel)
            _TECNICA_SELECIONADA["key"]  = tk_sel
            _TECNICA_SELECIONADA["nome"] = tk_sel.upper()
            fmin = td_sel.get("faixa_min"); fmax = td_sel.get("faixa_max")
            prep = td_sel.get("preproc", ""); mode = td_sel.get("mode", "dx")
            # R7: Raman liga a remocao de raios cosmicos; ao sair do Raman volta ao desligado (nao deixa
            # o despike de uma tecnica anterior ligado em NIR/MIR). So' mexe nisso na troca Raman <-> outra.
            _anterior = _TECNICAS.get(str(getattr(cfg, "technique", "") or ""), {})
            if td_sel.get("raios_cosmicos"):
                _set_val(cfg, "remover_raios_cosmicos", "true")
            elif _anterior.get("raios_cosmicos"):
                _set_val(cfg, "remover_raios_cosmicos", "false")
            if fmin is not None: _set_val(cfg, "faixa_min_cm", str(fmin))
            if fmax is not None: _set_val(cfg, "faixa_max_cm", str(fmax))
            if prep: _set_val(cfg, "pre_processamento", prep)
            # Faixas excluidas (agua/TSP em RMN): a tecnica manda, como faixa_min/max; vazio nas demais.
            fex_ant = tuple(getattr(cfg, "exclude_ranges", ()) or ())
            fex = tuple(td_sel.get("faixas_excluir", ()) or ())
            _set_val(cfg, "faixas_excluir", ", ".join(fex))
            fmts = _formatos_tecnica(tk_sel, lang)
            if len(fmts) > 1:   # o formato do arquivo decide o leitor (modo_entrada)
                console.print(f"\n  [{PW}]{'Formato dos arquivos' if lang=='PT' else 'File format'}:[/{PW}]")
                for i, (_m, rot) in enumerate(fmts, 1):
                    _suf_ext = _extras.sufixo_menu(_m, lang)    # "(extra: pip install ...)" so' se o pacote falta
                    console.print(f"   [{PA}][{i}][/{PA}] {escape(rot)}" + (f"  [{PM}]({'padrao' if lang=='PT' else 'default'})[/{PM}]" if i == 1 else "")
                                  + (f"  [warn]{escape(_suf_ext)}[/warn]" if _suf_ext else ""))
                rf = _input(f"  {'Formato (Enter = padrao)' if lang=='PT' else 'Format (Enter = default)'}: ").strip()
                if rf.isdigit() and 1 <= int(rf) <= len(fmts):
                    mode = fmts[int(rf) - 1][0]
            if mode: _set_val(cfg, "modo_entrada", mode)
            # R6: a tela grava a unidade do eixo e a tecnica (rotulo, orientacao e resumo certos);
            # "" limpa a unidade de uma tecnica escolhida antes (nao deixa nm de uma em cm-1 de outra)
            unidade_eixo = td_sel.get("eixo_unidade", "")
            _set_val(cfg, "eixo_unidade", unidade_eixo)
            _set_val(cfg, "tecnica_analitica", tk_sel)
            fa_str = tdl.get("faixa", f"{fmin}-{fmax}")
            console.print(f"  [g]✓ {escape(_trunc(nm_sel, 44))} {'selecionado' if lang=='PT' else 'selected'}.[/g]")
            console.print(f"  [info]  {'Faixa' if lang=='PT' else 'Range'}: {escape(str(fa_str))}[/info]")
            console.print(f"  [info]  Preproc.: {escape(str(prep))}  |  {'Modo' if lang=='PT' else 'Mode'}: {mode if lang=='PT' else _texto_para_exibir(mode)}[/info]")
            if td_sel.get("raios_cosmicos"):
                console.print(f"  [info]  {'Raios cosmicos: remocao ligada (remover_raios_cosmicos)' if lang=='PT' else 'Cosmic rays: removal on (remover_raios_cosmicos)'}[/info]")
            if unidade_eixo:
                console.print(f"  [info]  {'Eixo' if lang=='PT' else 'Axis'}: {escape(_ROTULO_UNIDADE_EIXO.get(unidade_eixo, unidade_eixo))}[/info]")
            if fex:
                console.print(f"  [warn]{'Faixas excluidas (gravadas em faixas_excluir)' if lang=='PT' else 'Excluded ranges (saved in faixas_excluir)'}: "
                              f"{escape(', '.join(cfg.exclude_ranges))} -- "
                              f"{'agua residual e TSP/TMS saem do modelo antes do pre-processamento; mude em [2.3] Dados > Faixas excluidas (vazio = nao excluir).' if lang=='PT' else 'residual water and TSP/TMS leave the model before preprocessing; change in [2.3] Data > Excluded ranges (empty = exclude nothing).'}[/warn]")
            elif fex_ant:
                console.print(f"  [warn]{'faixas_excluir zerada (' + ', '.join(fex_ant) + ') porque esta tecnica nao usa exclusao.' if lang=='PT' else 'faixas_excluir cleared (' + ', '.join(fex_ant) + ') because this technique uses no exclusion.'}[/warn]")
            console.print(f"  [{PM}]{'Proximo' if lang=='PT' else 'Next'}: {escape(_onde_dados(mode, lang))}[/{PM}]")
            # P6: avisa NA HORA que o pacote opcional do formato nao esta instalado (e o recurso opcional da tecnica)
            _aviso_ext = _extras.aviso_de_escolha(mode, lang)
            if _aviso_ext:
                console.print(f"  [warn]! {escape(_aviso_ext)}[/warn]")
            for _e_rec in _extras.extras_de_recurso_da_tecnica(tk_sel):
                console.print(f"  [warn]! {escape(_extras.aviso_de_recurso(_e_rec, lang))}[/warn]")
        except ValueError as e:   # _set_val: faixa/preproc/mode invalido p/ a tecnica
            console.print(f"  [err]{escape(str(e))}[/err]")

    while True:
        _cabecalho_tela("tecnica", cfg)
        lang = _lang()
        ordem = _tecnica_ordem()
        num = {tk: i for i, tk in enumerate(ordem, 1)}  # chave -> numero
        tec_atual = _TECNICA_SELECIONADA.get("key", "ft-nir")

        t = Table(box=None, show_header=True, header_style=PM, padding=(0, 1), expand=True)
        t.add_column("N",      style=PA, width=4, no_wrap=True)
        t.add_column("Tecnica" if lang=="PT" else "Technique", width=28, no_wrap=True)
        t.add_column("Faixa" if lang=="PT" else "Range", style=PS, width=34, no_wrap=True)
        t.add_column("Preproc.", style=PM, no_wrap=True)

        for cat_pt, cat_en, keys in _TECNICA_CATEGORIAS:
            keys_presentes = [k for k in keys if k in _TECNICAS]
            if not keys_presentes:
                continue
            cat = cat_pt if lang == "PT" else cat_en
            t.add_row("", Text.from_markup(f"[{PF}]── {escape(cat)} ──[/{PF}]"), "", "")
            for tk in keys_presentes:
                td = _TECNICAS.get(tk, {})
                tdl = td.get(lang, td.get("PT", {}))
                nm = tdl.get("nome", tk)
                fa = tdl.get("faixa_curta", tdl.get("faixa", "—"))   # rotulo curto: a tabela nao corta
                pr = td.get("preproc", "—")
                mk = f"[{PA}]►[/{PA}] " if tk == tec_atual else "  "
                t.add_row(
                    f"[{PA}][{num[tk]}][/{PA}]",
                    Text.from_markup(f"{mk}{escape(_trunc(nm.split(' (')[0], 25))}"),
                    escape(_trunc(str(fa).split(' (use')[0], 32)),
                    escape(_trunc(str(pr), 12)),
                )

        sub = ("Selecione o numero para aplicar. ?N = detalhes da tecnica N."
               if lang=="PT" else "Select the number to apply. ?N = details of technique N.")
        console.print(Panel(
            t,
            title=f"[bold {PA}]{_t('t_tecnica')}[/bold {PA}]",
            subtitle=f"[{PM}]{sub}[/{PM}]",
            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)
        ))
        _rodape_tela()

        raw = _input(f"\n  {_t('opcao')}: ", ajuda=False).strip().upper()
        if raw.startswith("?") and raw[1:].strip().isdigit() and 1 <= int(raw[1:].strip()) <= len(ordem):
            _tecnica_detalhe(ordem[int(raw[1:].strip()) - 1], lang)      # ?N = detalhes da tecnica N
            continue
        if raw == "?":      # painel da tela com todas as tecnicas; N = detalhes da tecnica N
            ops_t = []
            for tk in ordem:
                tdl_ = _TECNICAS.get(tk, {}).get(lang) or _TECNICAS.get(tk, {}).get("PT", {})
                ops_t.append((str(num[tk]), tdl_.get("nome", tk), tdl_.get("desc", "")))
            def _det_t(r2: str) -> bool:
                if r2.isdigit() and 1 <= int(r2) <= len(ordem):
                    _tecnica_detalhe(ordem[int(r2)-1], lang)
                    return True
                return False
            _ajuda_por_escolha(ops_t, lambda: console.print(_painel_tela("tecnica", ops_t)), _det_t, tid="tecnica")
            continue
        r = _tecla_comum(raw, "tecnica", cfg)
        if r == "voltar": break
        if r == "tratada":
            lang = _lang()
        elif raw.isdigit() and 1 <= int(raw) <= len(ordem):
            _aplicar(ordem[int(raw)-1])
            _pause(); break
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()


# ---------------------------------------------------------------------------
# CODIFICACAO DX
# ---------------------------------------------------------------------------
def _menu_encoding(cfg: Config) -> None:
    """Codificacao DX — explica o conceito e so lista os codigos sob demanda."""
    lang = _lang()
    CODIGOS_BASE = getattr(pq, "CODIGO_ESPECIE", {})

    def _cod_usr() -> dict:
        try:
            p = _CODIGOS_PATH
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except (OSError, json.JSONDecodeError):
            return {}   # arquivo ausente/corrompido -- sem codigos extras do usuario

    def _salvar_cod(d: dict) -> bool:
        try:
            _USER_DIR.mkdir(parents=True, exist_ok=True)
            _CODIGOS_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
            return True
        except OSError as e:
            # Antes engolia o erro: o usuario "salvava" um codigo, nao via aviso,
            # e ele sumia no proximo inicio se a gravacao tivesse falhado.
            console.print(f"[err]✗ Falha ao salvar codigos: {e}[/err]")
            return False

    def _listar() -> None:
        cod_usr = _cod_usr()
        tc = Table(show_header=True, header_style=PM, box=rbox.SIMPLE, padding=(0, 1))
        tc.add_column("COD", style=PS, width=6, no_wrap=True)
        tc.add_column("Especie" if lang=="PT" else "Species", width=26)
        tc.add_column("Origem" if lang=="PT" else "Source", style=PM, width=10, no_wrap=True)
        for cod, esp in CODIGOS_BASE.items():
            tc.add_row(cod, escape(str(esp)), "Pipeline")
        for cod, esp in cod_usr.items():
            tc.add_row(f"[{PA}]{cod}[/{PA}]", escape(str(esp)),
                       f"[{PA}]{'Usuario' if lang=='PT' else 'User'}[/{PA}]")
        n = len(CODIGOS_BASE) + len(cod_usr)
        src_lbl = (f"Codigos cadastrados ({n})" if lang=="PT"
                   else f"Registered codes ({n})")
        console.print(Panel(tc, title=f"[{PS}]{src_lbl}[/{PS}]",
                            border_style=PS, box=rbox.ROUNDED, padding=(0, 1)))
        _pause()

    def _adicionar() -> None:
        cod_usr = _cod_usr()
        console.print()
        cod_n = _input(f"  {_t('cod_novo_cod')}").upper().strip()
        if not cod_n or not _re.match(r'^[A-Z]{2,4}$', cod_n):
            console.print(f"  [{PR}]{_t('cod_invalido')}[/{PR}]")
        else:
            esp_n = _input(f"  {_t('cod_novo_esp', cod=cod_n)}").strip()
            if esp_n:
                cod_usr[cod_n] = esp_n
                _salvar_cod(cod_usr)
                console.print(f"  [g]✓ {_t('cod_salvo', cod=cod_n, esp=esp_n)}[/g]")
        _pause()

    def _importar_csv() -> None:
        import csv as _csv
        console.print()
        prompt = ("  Caminho do CSV (colunas: codigo,especie): " if lang=="PT"
                  else "  CSV path (columns: code,species): ")
        caminho = _input(prompt).strip().strip('"')
        if not caminho:
            return
        if not os.path.isfile(caminho):
            console.print(f"  [{PR}]{'Arquivo nao encontrado.' if lang=='PT' else 'File not found.'}[/{PR}]")
            _pause(); return
        cod_usr = _cod_usr()
        n_add = 0
        try:
            with open(caminho, newline="", encoding="utf-8-sig") as fh:
                # Detecta separador (',' ou ';')
                amostra = fh.read(2048); fh.seek(0)
                sep = ";" if amostra.count(";") > amostra.count(",") else ","
                leitor = _csv.reader(fh, delimiter=sep)
                for linha in leitor:
                    if len(linha) < 2:
                        continue
                    cod = str(linha[0]).strip().upper()
                    esp = str(linha[1]).strip()
                    # Pula cabecalho comum
                    if cod.lower() in ("codigo", "cod", "code") or not cod:
                        continue
                    if not _re.match(r'^[A-Z]{2,4}$', cod) or not esp:
                        continue
                    cod_usr[cod] = esp
                    n_add += 1
            _salvar_cod(cod_usr)
            msg = (f"{n_add} codigo(s) importado(s)." if lang=="PT"
                   else f"{n_add} code(s) imported.")
            console.print(f"  [g]✓ {msg}[/g]")
        except (OSError, UnicodeDecodeError, _csv.Error) as e:
            console.print(f"  [{PR}]{escape(str(e))}[/{PR}]")
        _pause()

    def _exportar_csv() -> None:
        import csv as _csv
        cod_usr = _cod_usr()
        destino = str(_USER_DIR / "codigos_exportados.csv")
        try:
            _USER_DIR.mkdir(parents=True, exist_ok=True)
            with open(destino, "w", newline="", encoding="utf-8-sig") as fh:
                w = _csv.writer(fh)
                w.writerow(["codigo", "especie", "origem"])
                for cod, esp in CODIGOS_BASE.items():
                    w.writerow([cod, esp, "pipeline"])
                for cod, esp in cod_usr.items():
                    w.writerow([cod, esp, "usuario"])
            msg = (f"Exportado para: {destino}" if lang=="PT"
                   else f"Exported to: {destino}")
            console.print(f"  [g]✓ {escape(msg)}[/g]")
        except OSError as e:
            console.print(f"  [{PR}]{escape(str(e))}[/{PR}]")
        _pause()

    while True:
        _cabecalho_tela("codificacao", cfg)

        # Painel explicativo — o que e, padrao de nome, como cadastrar/importar
        if lang == "PT":
            explicacao = (
                "[a]O que e:[/a] cada arquivo .dx comeca com um codigo de 2-4 letras\n"
                "que identifica a classe da amostra. A codificacao mapeia esse\n"
                "codigo para o nome legivel da classe usado nos resultados.\n\n"
                "[a]Padrao de nome dos arquivos:[/a]\n"
                "  COD-DD-MM-AAAA_Tn.dx            (classe pura)\n"
                "  COD-DD-MM-AAAA_AD-X-PP_Tn.dx    (adulterada)\n"
                "  Ex.: CLA-04-01-2099_T1.dx  ->  Classe A pura, triplicata 1\n\n"
                "[a]Como cadastrar:[/a]\n"
                "  [2] um codigo por vez, ou [3] importar um CSV pronto\n"
                "  (CSV com 2 colunas: codigo,especie — separador , ou ;)."
            )
        else:
            explicacao = (
                "[a]What it is:[/a] each .dx file starts with a 2-4 letter code\n"
                "identifying the sample class. The coding maps that code to the\n"
                "readable class name used in the results.\n\n"
                "[a]File name pattern:[/a]\n"
                "  COD-DD-MM-YYYY_Tn.dx            (pure class)\n"
                "  COD-DD-MM-YYYY_AD-X-PP_Tn.dx    (adulterated)\n"
                "  E.g.: CLA-04-01-2099_T1.dx  ->  Class A pure, replicate 1\n\n"
                "[a]How to register:[/a]\n"
                "  [2] one code at a time, or [3] import a ready CSV\n"
                "  (CSV with 2 columns: code,species — separator , or ;)."
            )
        console.print(Panel(
            Text.from_markup(explicacao),
            title=f"[bold {PA}]{_t('t_codigos')}[/bold {PA}]",
            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)
        ))

        # Botoes de acao (a lista so aparece ao pressionar [L])
        n_usr = len(_cod_usr())
        if lang == "PT":
            acoes = (
                f"  [{PA}][1][/{PA}] Listar codigos cadastrados"
                f"   [{PA}][2][/{PA}] Adicionar codigo\n"
                f"  [{PA}][3][/{PA}] Importar de CSV"
                f"            [{PA}][4][/{PA}] Exportar para CSV"
                f"   [{PM}]({n_usr} do usuario)[/{PM}]\n"
                f"  [{PM}]{escape(_cat.RODAPE_TELA['PT'])}[/{PM}]"
            )
        else:
            acoes = (
                f"  [{PA}][1][/{PA}] List registered codes"
                f"   [{PA}][2][/{PA}] Add code\n"
                f"  [{PA}][3][/{PA}] Import from CSV"
                f"           [{PA}][4][/{PA}] Export to CSV"
                f"   [{PM}]({n_usr} user)[/{PM}]\n"
                f"  [{PM}]{escape(_cat.RODAPE_TELA['EN'])}[/{PM}]"
            )
        console.print(Text.from_markup(acoes))

        pt_ = lang == "PT"
        _definir_ajuda("codificacao", [
            ("1", "Listar codigos" if pt_ else "List codes", "Mostra os codigos cadastrados e a classe de cada um." if pt_
             else "Shows the registered codes and the class of each one."),
            ("2", "Adicionar codigo" if pt_ else "Add code", "Cadastra um codigo (2 a 4 letras) e o nome da classe." if pt_
             else "Registers one code (2 to 4 letters) and the class name."),
            ("3", "Importar de CSV" if pt_ else "Import from CSV", "Le um CSV com 2 colunas (codigo,especie; separador , ou ;)." if pt_
             else "Reads a CSV with 2 columns (code,species; separator , or ;)."),
            ("4", "Exportar para CSV" if pt_ else "Export to CSV", "Grava os codigos do usuario num CSV para reusar ou compartilhar." if pt_
             else "Writes the user codes to a CSV to reuse or share.")])
        raw = _input(f"\n  {_t('opcao')}: ").upper()
        r = _tecla_comum(raw, "codificacao", cfg)
        if r == "voltar": break
        if r == "tratada": lang = _lang()
        elif raw == "1": _listar()
        elif raw == "2": _adicionar()
        elif raw == "3": _importar_csv()
        elif raw == "4": _exportar_csv()
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()


# ---------------------------------------------------------------------------
# HARDWARE — dashboard compacto com barras
# ---------------------------------------------------------------------------
def _menu_hardware(cfg: Optional[Config] = None) -> None:
    """Dashboard de hardware com diagnostico e recomendacoes por tier."""
    lang = _lang()
    _cabecalho_tela("hardware", cfg, limpar=False)
    try:
        import psutil
        ram  = psutil.virtual_memory()
        cpu_f = psutil.cpu_count(logical=False) or 1
        cpu_l = psutil.cpu_count(logical=True) or 1
        disk  = psutil.disk_usage(".")
        rt = ram.total / 1024**3
        rl = ram.available / 1024**3
        rp = ram.percent
        df = disk.free / 1024**3
        dp = disk.percent
        ok_psutil = True
    except ImportError:
        rt = rl = rp = cpu_f = cpu_l = df = dp = 0.0
        ok_psutil = False

    # Tiers com recomendacoes especificas por modulo
    if rt >= 16:
        tier, tcor = ("Alto Desempenho" if lang=="PT" else "High Performance"), PG
        tier_perfil = ("Rigor Maximo / Publicacao em Periodicos" if lang=="PT"
                       else "Maximum Rigor / Journal Publication")
        tier_mods = [
            ("Benchmark SVM/RF/XGB", "[g]Ativar[/g]"    if lang=="PT" else "[g]Enable[/g]"),
            ("Monte Carlo CV",        "[g]Ativar[/g]"    if lang=="PT" else "[g]Enable[/g]"),
            ("SHAP TreeExplainer",    "[g]Ativar[/g]"    if lang=="PT" else "[g]Enable[/g]"),
            ("SHAP max. amostras" if lang=="PT" else "SHAP max. samples", "[g]500+[/g]"),
            ("Permutacoes" if lang=="PT" else "Permutations", "[g]500 para publicacao[/g]" if lang=="PT" else "[g]500 for publication[/g]"),
        ]
    elif rt >= 8:
        tier, tcor = ("Desempenho Medio" if lang=="PT" else "Medium Performance"), PA
        tier_perfil = ("Indexacao Cientifica" if lang=="PT" else "Scientific Publication")
        tier_mods = [
            ("Benchmark SVM/RF/XGB", "[g]Ativar[/g]"    if lang=="PT" else "[g]Enable[/g]"),
            ("Monte Carlo CV",        "[warn]Opcional (lento)[/warn]" if lang=="PT" else "[warn]Optional (slow)[/warn]"),
            ("SHAP TreeExplainer",    "[warn]max 300 amostras[/warn]" if lang=="PT" else "[warn]max 300 samples[/warn]"),
            ("Permutacoes" if lang=="PT" else "Permutations", "[g]200 suficiente[/g]" if lang=="PT" else "[g]200 sufficient[/g]"),
        ]
    elif rt >= 4:
        tier, tcor = ("Desempenho Basico" if lang=="PT" else "Basic Performance"), PA
        tier_perfil = ("Pesquisa Exploratoria / Controle de Qualidade" if lang=="PT"
                       else "Exploratory Research / Quality Control")
        tier_mods = [
            ("Benchmark",  "[err]Desativar em [3.4][/err]" if lang=="PT" else "[err]Disable in [3.4][/err]"),
            ("Monte Carlo","[err]Desativar em [3.4][/err]" if lang=="PT" else "[err]Disable in [3.4][/err]"),
            ("SHAP",       "[err]Desativar em [3.4][/err]" if lang=="PT" else "[err]Disable in [3.4][/err]"),
            ("max_lvs",    "[warn]Reduzir para 20[/warn]" if lang=="PT" else "[warn]Reduce to 20[/warn]"),
        ]
    else:
        tier, tcor = ("Limitado" if lang=="PT" else "Limited"), PR
        tier_perfil = ("Apenas Exploracao de Dados" if lang=="PT" else "Data Exploration Only")
        tier_mods = [
            ("Todos em [3.4]", "[err]Desativar tudo[/err]" if lang=="PT" else "[err]Disable all[/err]"),
            ("Modo entrada",  "[warn]Usar modo sintetico[/warn]" if lang=="PT" else "[warn]Use synthetic mode[/warn]"),
            ("max_lvs",       "[warn]Reduzir para 15[/warn]" if lang=="PT" else "[warn]Reduce to 15[/warn]"),
        ]

    def _bar(pct: float, n: int = 22) -> Text:
        filled = int(pct / 100 * n)
        bar = Text()
        col = PG if pct < 70 else PA if pct < 85 else PR
        bar.append("█" * filled, style=col)
        bar.append("░" * max(0, n - filled), style=PD)
        bar.append(f" {pct:.0f}%", style=PM)
        return bar

    # Tabela principal: recursos
    hw = Table(show_header=True, header_style=PM, box=rbox.SIMPLE, padding=(0, 1))
    col_r = "Recurso" if lang=="PT" else "Resource"
    col_v = "Valor"   if lang=="PT" else "Value"
    col_u = "Uso"     if lang=="PT" else "Usage"
    hw.add_column(col_r, style=PM, width=16, no_wrap=True)
    hw.add_column(col_v, width=14, no_wrap=True)
    hw.add_column(col_u, no_wrap=True, min_width=26)

    if ok_psutil:
        hw.add_row(
            "RAM total",
            Text(f"{rt:.1f} GB", style=f"bold {PA}"),
            _bar(rp),
        )
        hw.add_row(
            "RAM disponivel" if lang=="PT" else "Available RAM",
            Text(f"{rl:.1f} GB", style=PG),
            Text(f"({100-rp:.0f}% livre)" if lang=="PT" else f"({100-rp:.0f}% free)", style=PM),
        )
        hw.add_row(
            "CPU fisicos"   if lang=="PT" else "Physical CPUs",
            Text(f"{cpu_f} cores", style=PS),
            Text(f"{cpu_l} threads logicos" if lang=="PT" else f"{cpu_l} logical threads", style=PM),
        )
        hw.add_row(
            "Disco livre"   if lang=="PT" else "Free disk",
            Text(f"{df:.1f} GB", style=PG),
            _bar(dp),
        )
    else:
        hw.add_row("psutil", Text("Nao instalado" if lang=="PT" else "Not installed", style=PR),
                   Text("pip install psutil", style=PM))

    # Tabela de recomendacoes por modulo
    rec = Table(show_header=True, header_style=PM, box=rbox.SIMPLE, padding=(0, 1))
    col_m = "Modulo" if lang=="PT" else "Module"
    col_r2 = "Recomendacao" if lang=="PT" else "Recommendation"
    rec.add_column(col_m, style=PW, width=24, no_wrap=True)
    rec.add_column(col_r2, no_wrap=True)
    for modulo, rec_str in tier_mods:
        rec.add_row(escape(modulo), Text.from_markup(rec_str))

    cap_lbl = "Capacidade" if lang=="PT" else "Capacity"
    cap_txt = Text.from_markup(f"  [{tcor}]{tier}[/{tcor}]  |  [{PA}]{escape(tier_perfil)}[/{PA}]")

    tit_rec  = "Recomendacoes por Modulo" if lang=="PT" else "Per-Module Recommendations"

    console.print(Panel(
        Group(
            Text.from_markup(f"  [{PM}]{cap_lbl}:[/{PM}] {cap_txt.markup}"),
            Text(""),
            hw,
            Rule(style=PD),
            Text.from_markup(f"  [{PM}]{tit_rec}:[/{PM}]"),
            rec,
        ),
        title=f"[bold {PS}]{_t('t_hardware')}[/bold {PS}]",
        border_style=PS, box=rbox.ROUNDED, padding=(0, 1)
    ))
    console.print(f"  [{PA}][1][/{PA}] {'Checar Python e dependencias (gera guaraci_doctor.txt)' if lang=='PT' else 'Check Python and dependencies (writes guaraci_doctor.txt)'}")
    console.print()
    _rodape_tela()
    _definir_ajuda("hardware", [("1", "Checar Python e dependencias" if lang == "PT" else "Check Python and dependencies",
                                 "Confere versao do Python e das bibliotecas e grava guaraci_doctor.txt para anexar num relato."
                                 if lang == "PT" else "Checks the Python and library versions and writes guaraci_doctor.txt to attach to a report.")])
    raw = _ask(f"  [{PA}]{_t('opcao')}:[/{PA}] ").strip().upper()
    if raw == "1":
        _comando_doctor()
        _pause()
        return
    _tecla_comum(raw, "hardware", cfg or Config())


# ---------------------------------------------------------------------------
# PREDICAO EM LOTE — aplica modelo salvo (.joblib) a espectros novos (CSV)
# ---------------------------------------------------------------------------
def _menu_prediction(cfg: Optional[Config] = None) -> None:
    """Predicao em lote via terminal: aplica um modelo .joblib salvo a um
    CSV de espectros novos (colunas=numero de onda, sem coluna de classe).

    Mesma logica cientifica do app web (aba Prediction) via predicao.py --
    zero duplicacao entre as duas interfaces (mesmo padrao da Fase H).
    Abre porta pra integracao com scripts/LIMS sem precisar de navegador.
    """
    lang = _lang()
    is_pt = lang == "PT"

    def _intro() -> None:
        _cabecalho_tela("predicao", cfg)
        _rodape_tela()
        console.print()

    _intro()

    lbl_modelo = "Caminho do modelo (.joblib)" if is_pt else "Model path (.joblib)"
    lbl_csv    = "Caminho do CSV de espectros novos" if is_pt else "New spectra CSV path"
    lbl_saida  = "Caminho de saida do CSV de resultados" if is_pt else "Output results CSV path"
    nao_encontrado = "Arquivo nao encontrado" if is_pt else "File not found"

    while True:
        cam_modelo = _ask(f"  [{PA}]{lbl_modelo}:[/{PA}] ").strip().strip('"')
        r = _tecla_valor(cam_modelo, "predicao")
        if r == "voltar":
            return
        if r == "tratada":
            _intro(); continue
        break
    if not os.path.isfile(cam_modelo):
        console.print(f"  [{PR}]{nao_encontrado}: {escape(cam_modelo)}[/{PR}]")
        _pause(); return

    # Aviso de seguranca (P5): .joblib e' pickle -- executa codigo arbitrario
    # NO CARREGAMENTO, antes de qualquer validacao ser possivel. Confirmacao
    # explicita do operador e' a unica protecao real (ver SECURITY.md).
    aviso_pickle = (
        f"  [{PR}]⚠ '.joblib' executa codigo ao ser carregado (formato "
        f"pickle). So confirme se voce mesmo treinou este modelo ou confia "
        f"plenamente na origem.[/{PR}]" if is_pt else
        f"  [{PR}]⚠ '.joblib' runs code when loaded (pickle format). Only "
        f"confirm if you trained this model yourself or fully trust its "
        f"source.[/{PR}]")
    console.print(aviso_pickle)
    conf_lbl = "Confirma o carregamento? (s/n)" if is_pt else "Confirm loading? (y/n)"
    if _ask(f"  [{PA}]{conf_lbl}[/{PA}] ").strip().lower() not in ("s", "y", "sim", "yes"):
        console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); _pause(); return

    cam_csv = _ask(f"  [{PA}]{lbl_csv}:[/{PA}] ").strip().strip('"')
    if not cam_csv:
        return
    if not os.path.isfile(cam_csv):
        console.print(f"  [{PR}]{nao_encontrado}: {escape(cam_csv)}[/{PR}]")
        _pause(); return

    padrao_saida = str(Path(cam_csv).with_name(Path(cam_csv).stem + "_predicao.csv"))
    cam_saida = _ask(
        f"  [{PA}]{lbl_saida}[/{PA}] [{PM}](Enter = {escape(padrao_saida)})[/{PM}]: "
    ).strip().strip('"')
    if not cam_saida:
        cam_saida = padrao_saida

    # Confirmacao antes de sobrescrever -- mesmo idioma s/n ja usado acima
    # para o aviso de pickle (nao introduz um mecanismo novo).
    if os.path.exists(cam_saida):
        conf_sobre = ("Arquivo ja existe. Sobrescrever? (s/n)" if is_pt
                      else "File already exists. Overwrite? (y/n)")
        if _ask(f"  [{PA}]{conf_sobre}[/{PA}] ").strip().lower() not in ("s", "y", "sim", "yes"):
            console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); _pause(); return

    try:
        import guaraci.predicao as _pred
        status_msg = ("Carregando modelo e aplicando..." if is_pt
                       else "Loading model and applying...")
        # confiar=True: o operador ja confirmou explicitamente acima. Fora
        # do spinner: pode perguntar se assina um modelo de origem externa.
        pkg = _carregar_modelo_cli(cam_modelo, is_pt)
        with console.status(f"[{PA}]{status_msg}[/{PA}]"):
            _pred.validate_model_package(pkg)
            X_new, wn_new, meta_df = _pred.load_prediction_csv(cam_csv)
            # Bloco 9b (D6): estende a predicao existente com o fluxo
            # completo Detectar -> Identificar -> Quantificar quando o
            # pacote traz o ensemble de identificacao (modelos exportados
            # antes do Bloco 9b nao tem essa chave -- cai no caminho
            # anterior, sem quebrar).
            if pkg.get("identification_ensemble"):
                df_res, resultados_cego = _pred.predict_blind(pkg, X_new, wn_new)
                _pred.anexar_colunas_fluxo_cego(df_res, resultados_cego)
            else:
                df_res = _pred.predict_samples(pkg, X_new, wn_new)
            if len(meta_df.columns) > 0 and len(meta_df) == len(df_res):
                df_res = _pred.anexar_metadados(meta_df, df_res)
            df_res.to_csv(cam_saida, index=False, sep=";", decimal=",")
    except Exception as e:  # noqa: BLE001 -- multi-etapa (joblib.load +
        # validacao do pacote + parsing CSV + predicao + escrita); erro
        # exibido ao usuario, tela volta ao menu sem crashar a CLI.
        console.print(f"  [{PR}]{'Erro' if is_pt else 'Error'}: {escape(str(e))}[/{PR}]")
        _pause(); return

    # Resumo
    n_tot = len(df_res)
    n_ac  = int(df_res["aceito"].sum()) if "aceito" in df_res.columns else n_tot
    t_res = Table(show_header=True, header_style=PM, box=rbox.SIMPLE, padding=(0, 1))
    t_res.add_column("Classe" if is_pt else "Class", style=PW)
    t_res.add_column("N")
    if "classe_pred" in df_res.columns:
        for cls_pred, n in df_res["classe_pred"].value_counts().items():
            t_res.add_row(escape(str(cls_pred)), str(n))

    resumo_txt = (
        f"  [{PG}]✔ {n_tot} amostras processadas[/{PG}]  |  "
        f"[{PG}]{n_ac}[/{PG}] amostras dentro do ajuste do modelo PLS-DA "
        f"(distância combinada T2+Q dentro do limite de 95%; T2_ok e Q_ok no "
        f"CSV são só diagnóstico por eixo -- ver 'criterio') / "
        f"[{PR}]{n_tot - n_ac}[/{PR}] fora do ajuste (espectro atipico, "
        f"tratar com cautela)"
        if is_pt else
        f"  [{PG}]✔ {n_tot} samples processed[/{PG}]  |  "
        f"[{PG}]{n_ac}[/{PG}] samples within the PLS-DA model fit "
        f"(combined T2+Q distance within the 95% limit; T2_ok and Q_ok in "
        f"the CSV are per-axis diagnostics only -- see 'criterio') / "
        f"[{PR}]{n_tot - n_ac}[/{PR}] outside the fit (atypical spectrum, "
        f"treat with caution)"
    )
    if "confianca_%" in df_res.columns:
        # A4-04: a coluna nao e' probabilidade; dizer isso no resumo do .joblib.
        resumo_txt += (
            f"\n  [{PM}]'confianca_%' é um escore relativo (PLS-DA "
            f"normalizado), não uma probabilidade calibrada.[/{PM}]"
            if is_pt else
            f"\n  [{PM}]'confianca_%' is a relative score (normalized "
            f"PLS-DA), not a calibrated probability.[/{PM}]")
    # Dominio de Aplicabilidade (PCA exploratorio, Jaworska et al. 2005) --
    # so' aparece se o pacote .joblib foi salvo por uma versao do pipeline
    # que exporta os artefatos leves (retrocompativel com pacotes antigos).
    if "AD_dentro_dominio" in df_res.columns:
        n_ad_dentro = int(df_res["AD_dentro_dominio"].sum())
        ad_txt = (
            f"  [{PG}]🔎 Dominio de aplicabilidade:[/{PG}] "
            f"[{PG}]{n_ad_dentro}[/{PG}] dentro / "
            f"[{PR}]{n_tot - n_ad_dentro}[/{PR}] fora "
            "(espectro parecido/nao-parecido com o que o modelo viu no "
            "treino -- 'fora' NAO significa 'adulterado', so' 'diferente do "
            "que foi calibrado')"
            if is_pt else
            f"  [{PG}]🔎 Applicability domain:[/{PG}] "
            f"[{PG}]{n_ad_dentro}[/{PG}] within / "
            f"[{PR}]{n_tot - n_ad_dentro}[/{PR}] outside "
            "(similar/dissimilar to what the model saw during training -- "
            "'outside' does NOT mean 'adulterated', only 'different from "
            "what was calibrated')"
        )
        resumo_txt += "\n" + ad_txt
        # A4-02: taxa esperada de alarmes em amostras legitimas (rejeicao do
        # dominio em CV da calibracao) + aviso de n pequeno, quando o pacote
        # os traz.
        _cv_n = int(pkg.get("ad_cv_n", 0) or 0)
        if _cv_n > 0:
            _taxa = int(pkg.get("ad_cv_rejeitadas", 0)) / _cv_n
            resumo_txt += (
                f"\n  [{PM}]Taxa esperada de alarmes em amostras legítimas "
                f"(rejeição em validação cruzada da calibração): "
                f"{_taxa:.1%} (nominal 5%)[/{PM}]"
                if is_pt else
                f"\n  [{PM}]Expected alarm rate on legitimate samples "
                f"(cross-validated rejection of the calibration): "
                f"{_taxa:.1%} (nominal 5%)[/{PM}]")
        if pkg.get("ad_aviso_n_pequeno"):
            resumo_txt += f"\n  [{PR}]{escape(str(pkg['ad_aviso_n_pequeno']))}[/{PR}]"

        # Bloco 13b: sentinela de deriva -- persistida ao lado do MODELO
        # (nao do CSV de saida), porque e' o modelo que fica fixo entre
        # varias rodadas de predicao ao longo do tempo -- exatamente o
        # uso continuo que a sentinela existe para acompanhar. Depende de
        # AD_dentro_dominio (nao de identification_ensemble), por isso fica
        # sob este if, nao sob o do fluxo cego abaixo.
        try:
            import guaraci.sentinela_deriva as _sent
            alerta_sent = _sent.hook_apos_predicao(cam_modelo, df_res)
            if alerta_sent is not None:
                cor_sent = PR if alerta_sent.alerta else PM
                sent_txt = (
                    f"  [{cor_sent}]🛰 Sentinela de deriva (n={alerta_sent.n}):"
                    f"[/{cor_sent}] {escape(alerta_sent.mensagem)}"
                )
                resumo_txt += "\n" + sent_txt
        except Exception as _e_sent:  # noqa: BLE001 -- diagnostico
            # opcional; erro impresso, nao afeta a predicao ja gravada.
            resumo_txt += (
                f"\n  [{PM}]{'Sentinela de deriva indisponivel' if is_pt else 'Drift sentinel unavailable'}"
                f": {escape(str(_e_sent))}[/{PM}]")

    # Conjunto de predicao conforme p/ classificacao (Grupo 2, analogo
    # classificatorio do T1 de regressao) -- so' aparece se o pacote foi
    # salvo por uma versao do pipeline que calibrou no holdout.
    if "classes_plausiveis" in df_res.columns:
        _cobertura = df_res["conjunto_cobertura_nominal"].iloc[0] \
            if "conjunto_cobertura_nominal" in df_res.columns else None
        if _cobertura is not None:
            n_ambiguo = int((df_res["classes_plausiveis"].str.contains(
                r"\|", regex=True)).sum())
            n_vazio = int((df_res["classes_plausiveis"] == "").sum())
            conj_txt = (
                f"  [{PG}]🎯 Conjunto de classes plausiveis "
                f"(conforme, {_cobertura:.0%}):[/{PG}] {n_ambiguo} "
                f"amostra(s) com mais de 1 classe plausivel, {n_vazio} "
                "com nenhuma (nenhuma classe atinge a cobertura pedida) "
                "-- ver coluna 'classes_plausiveis' no CSV"
                if is_pt else
                f"  [{PG}]🎯 Plausible class set "
                f"(conformal, {_cobertura:.0%} coverage):[/{PG}] {n_ambiguo} "
                f"sample(s) with more than 1 plausible class, {n_vazio} "
                "with none (no class reaches the requested coverage) -- "
                "see 'classes_plausiveis' column in the CSV"
            )
        else:
            conj_txt = (
                "  [{c}]🎯 Conjunto de classes plausiveis: NAO_VALIDADO "
                "(holdout sem grupos independentes suficientes p/ "
                "alpha=0.05)[/{c}]".format(c=PM) if is_pt else
                "  [{c}]🎯 Plausible class set: NOT VALIDATED "
                "(holdout lacked enough independent groups for "
                "alpha=0.05)[/{c}]".format(c=PM)
            )
        resumo_txt += "\n" + conj_txt

    # Bloco 9b: fluxo cego completo (Detectar -> Identificar -> Quantificar)
    # -- so' presente quando o pacote .joblib traz o ensemble de
    # identificacao. Sem este bloco a saida seria so' uma tabela de colunas
    # tecnicas (identificacao_cobertura, motivo_bloqueio, alpha_total) sem
    # explicacao nenhuma -- o motivo original desta auditoria de veracidade
    # (Agente 3, Passo 92).
    if "classe_identificada" in df_res.columns:
        n_pura = int(sum(1 for r in resultados_cego if r.pureza.aceito is True))
        n_adulterada = int(sum(1 for r in resultados_cego if r.pureza.aceito is False))
        n_pureza_indet = n_tot - n_pura - n_adulterada
        n_identificado = int(sum(
            1 for r in resultados_cego
            if r.identificacao.classe_identificada is not None))
        n_desconhecido = n_tot - n_identificado
        n_quantificado = int(sum(
            1 for r in resultados_cego if r.quantificacao.teor_estimado is not None))
        n_bloqueado = n_tot - n_quantificado
        # Bloco 24: faixa de decisao (LOD/LOQ, Bloco 12) -- so' conta entre
        # as amostras que de fato tiveram um teor estimado (bloqueadas nao
        # tem faixa nenhuma pra contar).
        n_nao_detectavel = int(sum(
            1 for r in resultados_cego
            if r.quantificacao.faixa_decisao == "nao_detectavel"))
        n_zona_cinzenta = int(sum(
            1 for r in resultados_cego
            if r.quantificacao.faixa_decisao == "zona_cinzenta"))
        n_confiavel = int(sum(
            1 for r in resultados_cego
            if r.quantificacao.faixa_decisao == "quantificado_com_confianca"))
        n_sem_lod_loq = n_quantificado - n_nao_detectavel - n_zona_cinzenta - n_confiavel

        if is_pt:
            fluxo_txt = (
                "\n  [bold]Fluxo cego -- Detectar → Identificar → Quantificar"
                ":[/bold]\n"
                f"  🧪 Pureza (DD-SIMCA da espécie prevista): "
                f"[{PG}]{n_pura}[/{PG}] detectada como pura / "
                f"[{PR}]{n_adulterada}[/{PR}] detectada como adulterada"
                + (f" / [{PM}]{n_pureza_indet}[/{PM}] indeterminada "
                   "(sem modelo de pureza calibrado para a espécie prevista)"
                   if n_pureza_indet else "") + "\n"
                f"  🏷 Adulterante: "
                f"[{PG}]{n_identificado}[/{PG}] identificado / "
                f"[{PR}]{n_desconhecido}[/{PR}] DESCONHECIDO (nenhuma "
                "combinação espécie×adulterante teve garantia estatística "
                "suficiente E exclusiva para rotular esta amostra -- por "
                "falta de garantia, ou por 2+ combinações validadas "
                "empatando, o que também bloqueia o rótulo)\n"
                f"  ⚖ Quantificação (teor de adulterante estimado -- mesma "
                "unidade da coluna de referência usada no treino do modelo, "
                f"tipicamente %m/m): [{PG}]{n_quantificado}[/{PG}] com "
                f"número / [{PR}]{n_bloqueado}[/{PR}] BLOQUEADA "
                "(quantificação recusada por não haver identificação "
                "confiável do adulterante -- ver coluna "
                "'quantificacao_motivo_bloqueio' no CSV)\n"
                + (f"  📏 Faixa de decisão (LOD/LOQ) entre as "
                   f"{n_quantificado} quantificadas: "
                   f"[{PR}]{n_nao_detectavel}[/{PR}] não detectável (< LOD) / "
                   f"[{PM}]{n_zona_cinzenta}[/{PM}] zona cinzenta (LOD–LOQ) / "
                   f"[{PG}]{n_confiavel}[/{PG}] quantificado com confiança (≥ LOQ)"
                   + (f" / {n_sem_lod_loq} sem LOD/LOQ calculável" if n_sem_lod_loq else "")
                   + "\n" if n_quantificado else "")
                + f"  [{PM}]⚠ Um rótulo em 'classe_identificada' só existe "
                "quando 'identificacao_cobertura'='validado' (garantia "
                "estatística formal, calibrada com >=2 sessões de coleta "
                "independentes) -- nunca há rótulo 'informativo' sem essa "
                "garantia: nesse caso a amostra é DESCONHECIDA de propósito, "
                "e a coluna 'identificacao_candidatos' traz só o palpite "
                "mais próximo (SEM garantia nenhuma) para referência, nunca "
                "como resultado a usar numa decisão de controle de "
                f"qualidade sem confirmar por método de referência.[/{PM}]"
            )
        else:
            fluxo_txt = (
                "\n  [bold]Blind flow -- Detect → Identify → Quantify"
                ":[/bold]\n"
                f"  🧪 Purity (DD-SIMCA for the predicted species): "
                f"[{PG}]{n_pura}[/{PG}] detected as pure / "
                f"[{PR}]{n_adulterada}[/{PR}] detected as adulterated"
                + (f" / [{PM}]{n_pureza_indet}[/{PM}] undetermined "
                   "(no purity model calibrated for the predicted species)"
                   if n_pureza_indet else "") + "\n"
                f"  🏷 Adulterant: "
                f"[{PG}]{n_identificado}[/{PG}] identified / "
                f"[{PR}]{n_desconhecido}[/{PR}] UNKNOWN (no species x "
                "adulterant combination had a statistical guarantee that "
                "was both sufficient AND exclusive for this sample -- "
                "either no guarantee, or 2+ validated combinations tied, "
                "which also blocks the label)\n"
                f"  ⚖ Quantification (estimated adulterant content -- same "
                "unit as the reference column used to train the model, "
                f"typically %w/w): [{PG}]{n_quantificado}[/{PG}] with a "
                f"number / [{PR}]{n_bloqueado}[/{PR}] BLOCKED "
                "(quantification refused because the adulterant was not "
                "reliably identified -- see the 'quantificacao_motivo_"
                "bloqueio' column in the CSV)\n"
                + (f"  📏 Decision band (LOD/LOQ) among the "
                   f"{n_quantificado} quantified: "
                   f"[{PR}]{n_nao_detectavel}[/{PR}] not detectable (< LOD) / "
                   f"[{PM}]{n_zona_cinzenta}[/{PM}] gray zone (LOD-LOQ) / "
                   f"[{PG}]{n_confiavel}[/{PG}] quantified with confidence (>= LOQ)"
                   + (f" / {n_sem_lod_loq} without computable LOD/LOQ" if n_sem_lod_loq else "")
                   + "\n" if n_quantificado else "")
                + f"  [{PM}]⚠ A 'classe_identificada' label only ever exists "
                "when 'identificacao_cobertura'='validado' (formal "
                "statistical guarantee, calibrated with >=2 independent "
                "collection sessions) -- there is no 'informational' label "
                "without that guarantee: in that case the sample is "
                "UNKNOWN on purpose, and the 'identificacao_candidatos' "
                "column carries only the closest guess (with NO guarantee "
                "at all) for reference, never as a result to act on for a "
                "quality decision without confirming by a reference "
                f"method.[/{PM}]"
            )
        resumo_txt += fluxo_txt
    console.print()
    console.print(Panel(
        Group(Text.from_markup(resumo_txt), Text(""), t_res),
        title=f"[bold {PG}]{'Resultado' if is_pt else 'Result'}[/bold {PG}]",
        border_style=PG, box=rbox.ROUNDED, padding=(1, 2),
    ))
    console.print(f"  [{PM}]{'Salvo em' if is_pt else 'Saved to'}:[/{PM}] "
                  f"{escape(cam_saida)}")
    _pause()


# Aviso de maturidade da tela HSI (Passo 103 da
# INSTRUCAO_HSI_ROBUSTEZ_E_VALIDACAO.md) -- fonte UNICA (nao frase solta
# duplicada em varios lugares, mesmo padrao de _AVISO_PROTOTIPO_TITULO/
# _AVISO_PROTOTIPO_CORPO em reports.py), descrevendo a limitacao REAL e
# especifica, nao um rotulo generico "prototipo". Escolhido em vez do
# carimbo formal "PROTOTYPE OUTPUT" (reports.py) porque aquele carimbo
# tem um criterio objetivo DIFERENTE (ausencia de garantia de
# agrupamento anti-vazamento) que NAO se aplica aqui -- o HSI TEM
# garantia de agrupamento real (group_id por objeto fisico, Passo 97,
# validada por teste de propriedade Hypothesis). A limitacao real do
# HSI e' outra: cobertura de validacao ainda pequena. Atualizar este
# texto sempre que a cobertura de validacao mudar (ver Passo 104).
# Limite de uso, sem relato de validacao (a evidencia fica no assistente de confianca, tecla c).
_AVISO_MATURIDADE_HSI_PT = (
    "Fluxo em validacao: confira o resultado com objetos seus de classe conhecida antes de "
    "decidir; classes desbalanceadas pesam no resultado. Evidencia e limites: tecla c."
)
_AVISO_MATURIDADE_HSI_EN = (
    "Flow under validation: check the result with your own objects of known class before "
    "deciding; imbalanced classes weigh on the result. Evidence and limits: key c."
)


def _menu_hsi(cfg: Optional[Config] = None) -> None:
    """Imageamento hiperespectral (HSI, mode='hsi', prototipo "minimo
    viavel" -- Passos 92-102 da INSTRUCAO_HSI_MINIMO_VIAVEL.md).

    DISTINTO do mode "imagem" (colorimetria de foto comum, tecla [K] da
    lista de fields de _menu_data): HSI opera POR PIXEL de um cubo
    hiperespectral (ENVI .hdr+.bin), com quality gate, segmentacao,
    classificacao por pixel + agregacao por objeto, explicabilidade
    cruzada com banda quimica e validacao externa por dia de medicao --
    fluxo orquestrado por `hsi_pipeline.run_hsi_pipeline`, nao pelo
    `pipeline.executar()` usado pelos outros modes (forma de dado
    diferente, ver docstring de hsi_pipeline.py).
    """
    if cfg is None:
        cfg = Config()
    # "hsi" vale SO' enquanto esta tela esta aberta: faz o cabecalho mostrar
    # "Tecnica: HSI" (Passo 103), mas o modo anterior volta ao sair. Antes
    # ficava gravado -- so' visitar [X] e voltar com [0] fazia o [R] falhar
    # com "Modo de entrada desconhecido: 'hsi'" (nao ha' leitor 'hsi' no
    # io_registry; HSI roda por hsi_pipeline, nao por executar()).
    modo_anterior = cfg.mode
    cfg.mode = "hsi"
    try:
        _menu_hsi_corpo(cfg)
    finally:
        cfg.mode = modo_anterior


def _menu_hsi_corpo(cfg: Config) -> None:
    lang = _lang()
    is_pt = lang == "PT"

    _cabecalho_tela("hsi", cfg)
    console.print(f"  [{PM}]{escape(_AVISO_MATURIDADE_HSI_PT if is_pt else _AVISO_MATURIDADE_HSI_EN)}[/{PM}]")
    _rodape_tela()
    console.print()

    lbl_pasta = ("Pasta com seus cubos hiperespectrais" if is_pt else
                "Folder with your hyperspectral cubes")
    default_pasta = getattr(cfg, "hsi_dataset_folder", "") or ""
    # escape: um caminho Linux entre colchetes ("[/tmp/x]") seria lido como etiqueta de fechamento do rich e derrubaria a tela
    sufixo = " " + escape(f"[{default_pasta}]") if default_pasta else ""
    pasta = _ask(f"  [{PA}]{lbl_pasta}{sufixo}:[/{PA}] ").strip().strip('"')
    if pasta and _tecla_valor(pasta, "hsi") == "voltar":
        return
    if not pasta:
        pasta = default_pasta
    if not pasta or not os.path.isdir(pasta):
        msg = (f"Pasta invalida: {pasta}" if is_pt
              else f"Invalid folder: {pasta}")
        console.print(f"  [{PR}]{escape(msg)}[/{PR}]")
        _pause(); return

    cfg.hsi_dataset_folder = pasta

    from guaraci.hsi_pipeline import run_hsi_pipeline
    console.print(f"  [{PM}]{'Rodando pipeline HSI...' if is_pt else 'Running HSI pipeline...'}[/{PM}]")
    try:
        # cast: run_hsi_pipeline devolve Dict[str, object] (valores
        # heterogeneos por chave); tipos concretos garantidos na construcao.
        resumo = cast(Dict[str, Any], run_hsi_pipeline(cfg))
    except Exception as e:  # noqa: BLE001 -- reporta erro completo, nunca engole
        console.print(f"  [{PR}]{'Erro' if is_pt else 'Error'}: {escape(str(e))}[/{PR}]")
        _pause(); return

    linhas = [
        f"  {'Gravacoes aceitas' if is_pt else 'Accepted recordings'}: "
        f"{resumo['n_gravacoes_aceitas']}/{resumo['n_gravacoes_total']} "
        f"({'rejeitadas pelo quality gate' if is_pt else 'rejected by quality gate'}: "
        f"{resumo['n_gravacoes_rejeitadas']})",
        f"  {'Variaveis latentes (Wold)' if is_pt else 'Latent variables (Wold)'}: "
        f"{resumo['n_components']}",
    ]
    # Passo 111: dataset generico (sem manifest.json) so' tem validacao
    # INTERNA (n_objetos_teste_externo=0, dicts *_externa vazios -- ver
    # hsi_validation.run_internal_validation_group_aware) -- declarado
    # aqui explicitamente, nunca escondido atras de um "0" sem explicacao.
    val = resumo["validacao_externa"]
    tem_externa = val.n_objetos_teste_externo > 0
    if tem_externa:
        linhas.append(
            f"  {'Validacao externa' if is_pt else 'External validation'}: "
            f"n_interno={val.n_objetos_teste_interno}, "
            f"n_externo={val.n_objetos_teste_externo}")
        for classe in val.classes:
            linhas.append(
                f"    {classe}: sens(int/ext)="
                f"{val.sensibilidade_interna[classe]:.2f}/"
                f"{val.sensibilidade_externa[classe]:.2f}")
    else:
        rotulo_val = ("Validacao (so interna -- sem particao externa "
                      "neste dataset)" if is_pt else
                      "Validation (internal only -- no external "
                      "partition for this dataset)")
        if getattr(val, "tipo_validacao", "") == "cv_objetos":   # fase 2: CV por objeto, IC e n de objetos por classe
            rotulo_val = (f"Validacao por objeto (CV, {val.n_dobras} dobras)" if is_pt
                          else f"By-object validation (CV, {val.n_dobras} folds)")
        linhas.append(
            f"  {rotulo_val}: n_interno={val.n_objetos_teste_interno}")
        for classe in val.classes:
            ic = getattr(val, "ic_sensibilidade_interna", {}).get(classe)
            linhas.append(
                f"    {classe}: sens(int)="
                f"{val.sensibilidade_interna[classe]:.2f}"
                + (f" [IC95% {ic[0]:.2f}-{ic[1]:.2f}, n={val.n_objetos_por_classe.get(classe, 0)}]"
                   if ic and ic[0] == ic[0] else ""))

    # Confianca por objeto (Passo 107): heterogeneidade de pixel deixa de
    # ser so' um numero interno -- resumo por faixa + objetos de baixa
    # concordancia listados explicitamente (nunca escondidos).
    conf = resumo.get("confianca_por_objeto", {})
    if conf:
        baixa = [(gid, r) for gid, r in conf.items()
                if r.heterogeneidade > 0.30]
        linhas.append("")
        linhas.append(
            f"  {'Confianca por objeto' if is_pt else 'Per-object confidence'}: "
            f"{len(conf) - len(baixa)}/{len(conf)} "
            f"{'com concordancia alta/moderada' if is_pt else 'high/moderate agreement'}")
        if baixa:
            linhas.append(
                f"    [{PR}]{'baixa concordancia' if is_pt else 'low agreement'} "
                f"({len(baixa)}): " +
                ", ".join(f"{gid} ({r.heterogeneidade:.0%})" for gid, r in baixa[:5]) +
                ("..." if len(baixa) > 5 else "") + f"[/{PR}]")

    console.print(Panel(
        "\n".join(linhas),
        title=f"[bold {PG}]{'Resultado' if is_pt else 'Result'}[/bold {PG}]",
        border_style=PG, box=rbox.ROUNDED, padding=(1, 2),
    ))
    console.print(f"  [{PM}]{'Salvo em' if is_pt else 'Saved to'}:[/{PM}] "
                  f"{escape(cfg.output_folder)}")
    _menu_hsi_extras(pasta, is_pt, cfg)


def _menu_hsi_extras(pasta: str, is_pt: bool, cfg: Any = None) -> None:
    """Extras HSI sobre a mesma pasta (dominio de aplicabilidade,
    reamostragem, multiway) -- logica em `hsi_extras.py`. Enter sai (faz
    o papel do antigo `_pause()` desta tela). So' CLI: o HSI inteiro nao
    tem pagina web (limitacao registrada em docs/cientifico/limitacoes.md)."""
    from guaraci.hsi_extras import (
        preparar_dados_hsi, rodar_hsi_dominio_aplicabilidade,
        rodar_hsi_multiway, rodar_hsi_reamostragem)
    lbl = ("[1] Dominio de aplicabilidade  [2] Reamostragem de classes  "
           "[3] Multiway N-PLS  [Enter] sair" if is_pt else
           "[1] Applicability domain  [2] Class resampling  "
           "[3] Multiway N-PLS  [Enter] exit")
    dados = None
    while True:
        with _ajuda_local("HSI extras", "Analises extras sobre a mesma pasta de cubos." if is_pt else "Extra analyses on the same cube folder.", [
                ("1", "Dominio de aplicabilidade" if is_pt else "Applicability domain",
                 "Diz quais cubos novos estao fora do dominio do modelo." if is_pt else "Says which new cubes fall outside the model domain."),
                ("2", "Reamostragem de classes" if is_pt else "Class resampling",
                 "Equilibra classes desiguais e mostra o efeito no resultado." if is_pt else "Balances unequal classes and shows the effect on the result."),
                ("3", "Multiway N-PLS", "Modelo que usa o cubo inteiro (linhas x colunas x bandas) sem desdobrar." if is_pt
                 else "Model that uses the whole cube (rows x columns x bands) without unfolding.")]):
            escolha = {"1": "D", "2": "R", "3": "M"}.get(_ask(f"  [{PA}]{lbl}:[/{PA}] ").strip(), "")
        if not escolha:
            return
        try:
            if dados is None:
                with console.status(f"[{PA}]{'Carregando cubos...' if is_pt else 'Loading cubes...'}[/{PA}]"):
                    # a mesma calibracao e checagem de eixo do modelo principal (sem cfg: leitura bruta, como antes)
                    kw = {} if cfg is None else dict(
                        calibracao=str(getattr(cfg, "hsi_calibration", "nenhuma") or "nenhuma"),
                        referencia_branco=str(getattr(cfg, "hsi_white_ref", "") or ""),
                        referencia_escuro=str(getattr(cfg, "hsi_dark_ref", "") or ""),
                        eixo_bandas=str(getattr(cfg, "hsi_band_axis", "recusar") or "recusar"))
                    dados = preparar_dados_hsi(pasta, **kw)
            with console.status(f"[{PA}]{'Rodando...' if is_pt else 'Running...'}[/{PA}]"):
                if escolha == "D":
                    r = rodar_hsi_dominio_aplicabilidade(dados)
                elif escolha == "R":
                    r = rodar_hsi_reamostragem(dados)
                else:
                    r = rodar_hsi_multiway(dados)
        except Exception as e:  # noqa: BLE001 -- dado externo; mensagem, nunca stack trace
            console.print(f"  [{PR}]{escape(str(e))}[/{PR}]")
            continue
        if escolha == "D":
            console.print(f"  [{PG}]✔[/{PG}] {'Dentro do dominio (objetos de teste)' if is_pt else 'Inside domain (test objects)'}: "
                          f"{r['fracao_dentro']:.1%}  ({r['n_objetos_treino']} "
                          f"{'objetos de treino' if is_pt else 'train objects'} / {r['n_objetos_teste']} "
                          f"{'de teste' if is_pt else 'test'})")
            for g, f in r["fracao_dentro_por_objeto"].items():
                console.print(f"    {escape(g)}: {f:.1%}")
        elif escolha == "R":
            for c, ev in r["avaliabilidade"].items():
                console.print(f"  {escape(c)}: {r['antes'][c]['pixels']} → {r['depois'][c]['pixels']} "
                              f"pixels ({ev['n_grupos']} {'objetos distintos' if is_pt else 'distinct objects'})"
                              + (f"  [{PA}]{escape(ev['nota'])}[/{PA}]" if ev["nota"] else ""))
            console.print(f"  [{PM}]{'Duplica objetos (mesmo group_id) -- equaliza pixels, nao cria objetos novos; classe abaixo do minimo continua nao avaliavel' if is_pt else 'Duplicates objects (same group_id) -- equalizes pixels, creates no new objects; a class below the minimum stays non-evaluable'}[/{PM}]")
        else:
            console.print(f"  [{PW}]N-PLS bal.acc={r['balanced_accuracy_npls']:.3f}  "
                          f"PLS-DA pixel bal.acc={r['balanced_accuracy_pixelwise']:.3f}[/{PW}]")
            console.print(f"  [{PM}]{'Exploratorio -- poucos objetos por classe tornam esta comparacao instavel' if is_pt else 'Exploratory -- few objects per class make this comparison unstable'}[/{PM}]")


def _menu_plan(cfg: Optional[Config] = None) -> None:
    """Planejamento de coleta (Bloco 10): quantas amostras por classe
    (`plano_amostral.py`) + como distribui-las entre sessoes e em que
    ordem le-las (`plano_coleta.py`), evitando os dois confundimentos ja
    documentados no projeto (classe x sessao, ordem de leitura x teor).
    """
    lang = _lang()
    is_pt = lang == "PT"

    def _intro() -> None:
        _cabecalho_tela("planejamento", cfg)
        _rodape_tela()
        console.print()

    _intro()

    lbl_classes = "Classes (separadas por virgula)" if is_pt else "Classes (comma-separated)"
    while True:
        classes_raw = _ask(f"  [{PA}]{lbl_classes}:[/{PA}] ").strip()
        r = _tecla_valor(classes_raw, "planejamento")
        if r == "voltar":
            return
        if r == "tratada":
            _intro(); continue
        break
    classes = [c.strip() for c in classes_raw.split(",") if c.strip()]
    if not classes:
        return

    lbl_sessoes = "Numero de sessoes de coleta" if is_pt else "Number of collection sessions"
    sessoes_raw = _ask(f"  [{PA}]{lbl_sessoes}:[/{PA}] [{PM}](Enter = 2)[/{PM}] ")
    try:
        n_sessoes = int(sessoes_raw) if sessoes_raw else 2
    except ValueError:
        console.print(f"  [{PR}]{'Numero invalido' if is_pt else 'Invalid number'}[/{PR}]")
        _pause(); return
    if n_sessoes < 1:
        console.print(f"  [{PR}]{'Precisa de pelo menos 1 sessao' if is_pt else 'Needs at least 1 session'}[/{PR}]")
        _pause(); return

    lbl_alvo = (
        "Alvo: (C)onformal (Identificar/agrupado) ou (D)D-SIMCA (pureza por especie)?"
        if is_pt else
        "Target: (C)onformal (Identify/pooled) or (D)D-SIMCA (per-species purity)?"
    )
    alvo = _ask(f"  [{PA}]{lbl_alvo}[/{PA}] ").strip().upper()
    if alvo not in ("C", "D"):
        console.print(f"  [{PR}]{'Opcao invalida' if is_pt else 'Invalid option'}[/{PR}]")
        _pause(); return

    if alvo == "C":
        lbl_valor = "Alpha desejado (ex.: 0.05)" if is_pt else "Desired alpha (e.g., 0.05)"
    else:
        lbl_valor = "Cobertura-alvo (ex.: 0.90)" if is_pt else "Target coverage (e.g., 0.90)"
    valor_raw = _ask(f"  [{PA}]{lbl_valor}:[/{PA}] ")
    try:
        valor = float(valor_raw)
    except ValueError:
        console.print(f"  [{PR}]{'Numero invalido' if is_pt else 'Invalid number'}[/{PR}]")
        _pause(); return

    try:
        import guaraci.plano_coleta as _plano
        if alvo == "C":
            plano, meta = _plano.plan_from_statistical_target(
                classes, n_sessoes, alpha_conformal=valor)
        else:
            plano, meta = _plano.plan_from_statistical_target(
                classes, n_sessoes, cobertura_ddsimca=valor)
    except ValueError as e:
        console.print(f"  [{PR}]{'Erro' if is_pt else 'Error'}: {escape(str(e))}[/{PR}]")
        _pause(); return

    padrao_saida = str(Path.cwd() / "plano_coleta")
    lbl_saida = "Prefixo dos arquivos de saida" if is_pt else "Output file prefix"
    cam_saida = _ask(
        f"  [{PA}]{lbl_saida}[/{PA}] [{PM}](Enter = {escape(padrao_saida)})[/{PM}]: "
    ).strip().strip('"')
    if not cam_saida:
        cam_saida = padrao_saida

    lbl_pdf = ("Gerar tambem PDF (opcional, alem de Markdown+Excel)? (s/N)"
               if is_pt else
               "Also generate PDF (optional, in addition to Markdown+Excel)? (y/N)")
    quer_pdf = _ask(f"  [{PA}]{lbl_pdf}[/{PA}] ").strip().lower() in ("s", "y", "sim", "yes")

    # Confirmacao antes de sobrescrever -- mesmo idioma s/n de _menu_prediction.
    _existentes = [p for p in (cam_saida + ".md", cam_saida + ".xlsx",
                                cam_saida + ".pdf" if quer_pdf else None)
                   if p and os.path.exists(p)]
    if _existentes:
        conf_sobre = ("Arquivo(s) ja existem. Sobrescrever? (s/n)" if is_pt
                      else "File(s) already exist. Overwrite? (y/n)")
        if _ask(f"  [{PA}]{conf_sobre}[/{PA}] ").strip().lower() not in ("s", "y", "sim", "yes"):
            console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); _pause(); return

    try:
        md = _plano.export_markdown(plano)
        cam_md = cam_saida + ".md"
        with open(cam_md, "w", encoding="utf-8") as f:
            f.write(md)
        cam_xlsx = cam_saida + ".xlsx"
        _plano.export_excel(plano, cam_xlsx)
        cam_pdf = None
        if quer_pdf:
            cam_pdf = cam_saida + ".pdf"
            _plano.export_pdf(plano, cam_pdf)
    except OSError as e:
        console.print(f"  [{PR}]{'Erro ao salvar' if is_pt else 'Error saving'}: {escape(str(e))}[/{PR}]")
        _pause(); return

    resumo_txt = (
        f"  [{PG}]✔ n por classe:[/{PG}] [{PG}]{meta['n_por_classe']}[/{PG}]  |  "
        f"[{PG}]origem:[/{PG}] {escape(meta['origem'])}  |  "
        f"[{PG}]total de amostras:[/{PG}] {len(plano.itens)}"
    )
    console.print()
    console.print(Panel(
        Text.from_markup(resumo_txt),
        title=f"[bold {PG}]{'Plano gerado' if is_pt else 'Plan generated'}[/bold {PG}]",
        border_style=PG, box=rbox.ROUNDED, padding=(1, 2),
    ))
    console.print(f"  [{PM}]{'Alertas' if is_pt else 'Alerts'}:[/{PM}]")
    for a in plano.alertas:
        console.print(f"    [{PA}]•[/{PA}] {escape(a)}")
    console.print()
    caminhos_salvos = [cam_md, cam_xlsx] + ([cam_pdf] if cam_pdf else [])
    console.print(f"  [{PM}]{'Salvo em' if is_pt else 'Saved to'}:[/{PM}] "
                  f"{', '.join(escape(c) for c in caminhos_salvos)}")

    # Bloco 25: refinamento por amostragem ativa -- opt-in, so' roda se o
    # usuario tiver um modelo .joblib JA treinado com ensemble de
    # identificacao calibrado (Bloco 9b). Nao substitui o plano acima
    # (que so precisa das classes digitadas), so' reordena a prioridade
    # ENTRE combinacoes especie x adulterante ja presentes no dataset de
    # treino do modelo.
    console.print()
    lbl_refinar = ("Refinar com amostragem ativa (modelo .joblib ja "
                   "treinado)? (s/n)" if is_pt else
                   "Refine with active sampling (already-trained .joblib "
                   "model)? (y/n)")
    if _ask(f"  [{PA}]{lbl_refinar}[/{PA}] ").strip().lower() in ("s", "y", "sim", "yes"):
        _refinar_plano_com_amostragem_ativa(is_pt)
    _pause()


def _refinar_plano_com_amostragem_ativa(is_pt: bool) -> None:
    """Bloco 25: carrega um pacote de modelo .joblib e mostra a lista de
    combinacoes especie x adulterante priorizadas por
    `amostragem_ativa.priorizar_amostragem`, usando o ensemble de
    identificacao ja calibrado no treino (`pkg["identification_
    ensemble"]", Bloco 9b) -- nunca recalibra nada aqui, so' consome."""
    lbl_modelo = "Caminho do modelo (.joblib)" if is_pt else "Model path (.joblib)"
    cam_modelo = _ask(f"  [{PA}]{lbl_modelo}:[/{PA}] ").strip().strip('"')
    if not cam_modelo or not os.path.isfile(cam_modelo):
        console.print(f"  [{PR}]{'Arquivo nao encontrado' if is_pt else 'File not found'}"
                      f"[/{PR}]")
        return

    aviso_pickle = (
        f"  [{PR}]⚠ '.joblib' executa codigo ao ser carregado (formato "
        f"pickle). So confirme se voce mesmo treinou este modelo ou confia "
        f"plenamente na origem.[/{PR}]" if is_pt else
        f"  [{PR}]⚠ '.joblib' runs code when loaded (pickle format). Only "
        f"confirm if you trained this model yourself or fully trust its "
        f"source.[/{PR}]")
    console.print(aviso_pickle)
    conf_lbl = "Confirma o carregamento? (s/n)" if is_pt else "Confirm loading? (y/n)"
    if _ask(f"  [{PA}]{conf_lbl}[/{PA}] ").strip().lower() not in ("s", "y", "sim", "yes"):
        console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); return

    try:
        from guaraci.amostragem_ativa import priorizar_amostragem
        # load_model (nao joblib.load cru): confere o SHA-256 do manifesto
        # ANTES de executar o pickle -- o joblib.load direto que havia
        # aqui contornava essa protecao (achado da auditoria de
        # seguranca de 2026-09-19); confiar=True porque o operador
        # ja confirmou acima.
        pkg = _carregar_modelo_cli(cam_modelo, is_pt)
        ensemble = pkg.get("identification_ensemble")
        if not ensemble:
            console.print(
                f"  [{PM}]{'Modelo sem ensemble de identificacao calibrado -- nada a priorizar.' if is_pt else 'Model has no calibrated identification ensemble -- nothing to prioritize.'}[/{PM}]")
            return
        lista = priorizar_amostragem(ensemble)
    except Exception as e:  # noqa: BLE001 -- carregamento de pacote
        # externo, mostrado ao usuario, nunca crasha o menu.
        console.print(f"  [{PR}]{'Erro' if is_pt else 'Error'}: {escape(str(e))}[/{PR}]")
        return

    t_prior = Table(show_header=True, header_style=PM, box=rbox.SIMPLE, padding=(0, 1))
    t_prior.add_column("Especie" if is_pt else "Species", style=PW)
    t_prior.add_column("Adulterante" if is_pt else "Adulterant", style=PW)
    t_prior.add_column("Sessoes" if is_pt else "Sessions")
    t_prior.add_column("Faltam" if is_pt else "Missing")
    t_prior.add_column("Status")
    for r in lista[:15]:
        cor = PG if r.prioridade == 0.0 else PA
        t_prior.add_row(
            escape(r.especie), escape(r.adulterante), str(r.n_sessoes_atual),
            str(r.sessoes_faltantes),
            f"[{cor}]{r.cobertura_status.value if r.cobertura_status else '-'}[/{cor}]")
    console.print()
    console.print(t_prior)
    console.print(f"  [{PM}]{'Ordenado por prioridade -- topo = maior impacto esperado por sessao investida.' if is_pt else 'Sorted by priority -- top = highest expected impact per invested session.'}[/{PM}]")


# Nomes de coluna reconhecidos como "amostra fisica" na tela [2.8] (sugestao
# padrao do prompt; o usuario pode digitar qualquer outra).
_COLUNAS_GRUPO_CONHECIDAS = {"mae_id", "amostra", "amostra_fisica", "grupo",
                             "group", "group_id", "sample", "sample_id"}


def _ler_csv_detectando_separador(caminho: str):
    """Le CSV com `,` ou `;`. `;` implica virgula decimal -- formato do
    Excel em PT-BR e o mesmo que o proprio GUARACI grava em [B]. Antes a
    tela [2.8] so' lia `,`: um CSV com `;` virava 1 coluna de texto e a
    mensagem era "sem colunas numericas" (achado B11, 2026-09-24)."""
    import pandas as _pd
    with open(caminho, encoding="utf-8-sig", errors="replace") as fh:
        primeira = fh.readline()
    if primeira.count(";") > primeira.count(","):
        return _pd.read_csv(caminho, sep=";", decimal=",", encoding="utf-8-sig")
    if primeira.count("\t") > primeira.count(","):
        return _pd.read_csv(caminho, sep="\t", encoding="utf-8-sig")
    return _pd.read_csv(caminho, encoding="utf-8-sig")


def _menu_selecao_amostras(cfg: Optional[Config] = None) -> None:
    """Selecao de amostras de calibracao/validacao (Bloco 10, Passo 87):
    dado um CSV com espectros JA medidos (e opcionalmente uma coluna de
    referencia/teor), escolhe QUAIS amostras vao para calibracao via
    Kennard-Stone, Duplex ou SPXY (`dados_io.py`) -- mesmo fluxo de
    planejamento experimental de `_menu_plan` (que decide QUANTAS
    coletar), agora atuando sobre dados que ja existem.
    """
    lang = _lang(); is_pt = lang == "PT"

    def _intro() -> None:
        _cabecalho_tela("selecao_amostras", cfg)
        _rodape_tela()
        console.print()

    _intro()

    lbl_csv = ("Caminho do CSV com os espectros (1 amostra por linha)"
               if is_pt else
               "Path to the CSV with spectra (1 sample per row)")
    while True:
        caminho_csv = _ask(f"  [{PA}]{lbl_csv}:[/{PA}] ").strip().strip('"')
        r = _tecla_valor(caminho_csv, "selecao_amostras")
        if r == "voltar":
            return
        if r == "tratada":
            _intro(); continue
        break
    if not caminho_csv or not os.path.isfile(caminho_csv):
        console.print(f"  [{PR}]{'Arquivo nao encontrado' if is_pt else 'File not found'}[/{PR}]")
        _pause(); return

    import numpy as _np
    try:
        df = _ler_csv_detectando_separador(caminho_csv)
    except (OSError, ValueError, UnicodeDecodeError) as e:
        console.print(f"  [{PR}]{'Erro ao ler CSV' if is_pt else 'Error reading CSV'}: {escape(str(e))}[/{PR}]")
        _pause(); return
    console.print(f"  [{PM}]{len(df)} {'linhas' if is_pt else 'rows'}, "
                  f"{df.shape[1]} {'colunas' if is_pt else 'columns'}[/{PM}]")

    lbl_alvo = ("Coluna de referencia/teor (Enter = nenhuma -- so' habilita "
                "Kennard-Stone/Duplex, nao SPXY)"
                if is_pt else
                "Reference/target column (Enter = none -- only enables "
                "Kennard-Stone/Duplex, not SPXY)")
    col_alvo = _ask(f"  [{PA}]{lbl_alvo}:[/{PA}] ").strip()
    if col_alvo and col_alvo not in df.columns:
        console.print(f"  [{PR}]{'Coluna nao encontrada' if is_pt else 'Column not found'}: {escape(col_alvo)}[/{PR}]")
        _pause(); return

    # Coluna de grupo (amostra fisica): sem ela, replicas T1/T2/T3 da MESMA
    # amostra podiam cair uma na calibracao e outra na validacao -- o
    # vazamento que o projeto existe para evitar. Antes esta tela chamava
    # as versoes SEM grupo.
    detectada = next((c for c in df.columns
                      if str(c).strip().lower() in _COLUNAS_GRUPO_CONHECIDAS
                      and c != col_alvo), "")
    lbl_grupo = (f"Coluna que identifica a amostra fisica/replicas "
                 f"(Enter = {detectada or 'nenhuma'})" if is_pt else
                 f"Column identifying the physical sample/replicates "
                 f"(Enter = {detectada or 'none'})")
    col_grupo = _ask(f"  [{PA}]{escape(lbl_grupo)}:[/{PA}] ").strip() or detectada
    if col_grupo and col_grupo not in df.columns:
        console.print(f"  [{PR}]{'Coluna nao encontrada' if is_pt else 'Column not found'}: {escape(col_grupo)}[/{PR}]")
        _pause(); return
    grupos = df[col_grupo].astype(str).to_numpy() if col_grupo else None
    if grupos is None or len(set(grupos)) < 4:
        console.print(
            f"  [{PA}]⚠ {'Sem grupos suficientes (minimo 4): replicas da mesma amostra podem ser separadas entre calibracao e validacao.' if is_pt else 'Not enough groups (minimum 4): replicates of the same sample may be split between calibration and validation.'}[/{PA}]")

    colunas_x = [c for c in df.columns if c not in (col_alvo, col_grupo)]
    df_x = df[colunas_x].select_dtypes(include=[_np.number])
    if df_x.shape[1] == 0 or df_x.shape[0] < 2:
        console.print(f"  [{PR}]{'CSV sem colunas numericas suficientes (pelo menos 2 amostras, 1 variavel)' if is_pt else 'CSV without enough numeric columns (at least 2 samples, 1 variable)'}[/{PR}]")
        _pause(); return
    X = df_x.to_numpy(dtype=float)

    metodos = ["Kennard-Stone", "Duplex"] + (["SPXY"] if col_alvo else [])
    quando_usar = {
        "Kennard-Stone": ("calibracao cobre os extremos do espectro (padrao)" if is_pt
                          else "calibration covers the spectral extremes (default)"),
        "Duplex": ("calibracao e validacao igualmente representativas" if is_pt
                   else "calibration and validation equally representative"),
        "SPXY": ("cobre espectro E faixa do teor ao mesmo tempo" if is_pt
                 else "covers spectrum AND content range at once"),
    }
    lbl_metodo = "Metodo" if is_pt else "Method"
    console.print(f"  [{PA}]{lbl_metodo}:[/{PA}]")
    for i, m in enumerate(metodos, 1):
        console.print(f"    ({i}) {m} [{PM}]— {quando_usar[m]}[/{PM}]")
    with _ajuda_local(lbl_metodo, "Como escolher as amostras de calibracao." if is_pt else "How to choose the calibration samples.",
                      [(str(i), m, quando_usar[m]) for i, m in enumerate(metodos, 1)]):
        escolha_m = _ask("  > ").strip()
    if not escolha_m.isdigit() or not (1 <= int(escolha_m) <= len(metodos)):
        console.print(f"  [{PR}]{'Opcao invalida' if is_pt else 'Invalid option'}[/{PR}]")
        _pause(); return
    metodo = metodos[int(escolha_m) - 1]

    lbl_frac = "Fracao para calibracao (Enter = 0.7)" if is_pt else "Calibration fraction (Enter = 0.7)"
    frac_raw = _ask(f"  [{PA}]{lbl_frac}:[/{PA}] ").strip()
    try:
        frac_cal = float(frac_raw) if frac_raw else 0.7
    except ValueError:
        console.print(f"  [{PR}]{'Numero invalido' if is_pt else 'Invalid number'}[/{PR}]")
        _pause(); return
    if not (0.0 < frac_cal < 1.0):
        console.print(f"  [{PR}]{'Fracao precisa estar entre 0 e 1' if is_pt else 'Fraction must be between 0 and 1'}[/{PR}]")
        _pause(); return

    from guaraci.dados_io import (duplex_split_group_aware,
                                  kennard_stone_split_group_aware,
                                  spxy_split_group_aware)
    # console.status: Kennard-Stone/Duplex/SPXY sao O(n^2) em distancias --
    # sem isso a tela ficava parada sem nenhum sinal para datasets maiores
    # (mesmo padrao ja usado em _menu_prediction para o carregamento do modelo).
    # *_group_aware com grupos=None (ou <4 grupos) cai no metodo por amostra.
    status_msg = "Calculando particao..." if is_pt else "Computing split..."
    with console.status(f"[{PA}]{status_msg}[/{PA}]"):
        if metodo == "Kennard-Stone":
            idx_cal, idx_val = kennard_stone_split_group_aware(X, grupos, frac_cal)
        elif metodo == "Duplex":
            idx_cal, idx_val = duplex_split_group_aware(X, grupos, frac_cal)
        else:
            y = df[col_alvo].to_numpy(dtype=float)
            idx_cal, idx_val = spxy_split_group_aware(X, y, grupos, frac_cal)

    df_saida = df.copy()
    col_conjunto = "conjunto" if is_pt else "set"
    df_saida[col_conjunto] = ""
    df_saida.iloc[idx_cal, df_saida.columns.get_loc(col_conjunto)] = (
        "calibracao" if is_pt else "calibration")
    df_saida.iloc[idx_val, df_saida.columns.get_loc(col_conjunto)] = (
        "validacao" if is_pt else "validation")

    padrao_saida = str(Path.cwd() / "selecao_amostras.csv")
    lbl_saida = "Arquivo de saida" if is_pt else "Output file"
    cam_saida = _ask(
        f"  [{PA}]{lbl_saida}[/{PA}] [{PM}](Enter = {escape(padrao_saida)})[/{PM}]: "
    ).strip().strip('"')
    if not cam_saida:
        cam_saida = padrao_saida

    # Confirmacao antes de sobrescrever -- mesmo idioma s/n de _menu_prediction.
    if os.path.exists(cam_saida):
        conf_sobre = ("Arquivo ja existe. Sobrescrever? (s/n)" if is_pt
                      else "File already exists. Overwrite? (y/n)")
        if _ask(f"  [{PA}]{conf_sobre}[/{PA}] ").strip().lower() not in ("s", "y", "sim", "yes"):
            console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); _pause(); return

    try:
        df_saida.to_csv(cam_saida, index=False)
    except OSError as e:
        console.print(f"  [{PR}]{'Erro ao salvar' if is_pt else 'Error saving'}: {escape(str(e))}[/{PR}]")
        _pause(); return

    resumo_txt = (
        f"  [{PG}]✔ {escape(metodo)}:[/{PG}] {len(idx_cal)} "
        f"{'calibracao' if is_pt else 'calibration'} / {len(idx_val)} "
        f"{'validacao' if is_pt else 'validation'} "
        f"({'de' if is_pt else 'of'} {len(df)})"
    )
    console.print()
    console.print(Panel(
        Text.from_markup(resumo_txt),
        title=f"[bold {PG}]{'Selecao gerada' if is_pt else 'Selection generated'}[/bold {PG}]",
        border_style=PG, box=rbox.ROUNDED, padding=(1, 2),
    ))
    console.print(f"  [{PM}]{'Salvo em' if is_pt else 'Saved to'}:[/{PM}] {escape(cam_saida)}")
    _pause()


# ---------------------------------------------------------------------------
# TECNICAS AVANCADAS — ASCA / EPO-GLSW / MCR-ALS / fusao multibloco
# ---------------------------------------------------------------------------
def _tecavan_carregar_dataset(cfg: Config, is_pt: bool, *, manter_constantes: bool = False):
    """Carrega+valida o dataset configurado em [2.3] Dados -- mesmo caminho
    de `pq.load_data`/`pq.validate_input` usado por `_menu_audit`/
    `_guaraci_diagnosticar`. Devolve None em erro (ja' reportado ao
    usuario). `manter_constantes=True` nao remove variaveis constantes (PARAFAC:
    a regiao mascarada/zerada da EEM e' constante e remove-la quebra a grade)."""
    status_msg = "Carregando dados..." if is_pt else "Loading data..."
    try:
        with console.status(f"[{PA}]{status_msg}[/{PA}]"):
            wn, X, rotulos, conc, mae_id, metadados = pq.load_data(cfg)
            X, wn, rotulos, conc, mae_id, _rel = pq.validate_input(
                X, wn, rotulos, conc, mae_id,
                **({"tol_const": -1.0} if manter_constantes else {}))
    except Exception as e:  # noqa: BLE001 -- dado externo, mesma disciplina
        # de _menu_audit/_guaraci_diagnosticar.
        console.print(f"  [{PR}]{'Erro ao carregar dados' if is_pt else 'Error loading data'}: "
                      f"{escape(str(e))}[/{PR}]")
        _pause(); return None
    return wn, X, rotulos, conc, mae_id, metadados


def _tecavan_escolher_fatores(rotulos, metadados, is_pt: bool, max_fatores: int = 2):
    """Lista fatores candidatos (`tecnicas_avancadas.fatores_categoricos_
    disponiveis`) e deixa o usuario escolher 1 ou 2 por numero."""
    from guaraci.tecnicas_avancadas import fatores_categoricos_disponiveis
    disponiveis = fatores_categoricos_disponiveis(rotulos, metadados)
    nomes = list(disponiveis)
    console.print(f"  [{PA}]{'Fatores disponiveis' if is_pt else 'Available factors'}:[/{PA}]")
    for i, nome in enumerate(nomes, 1):
        n_niveis = len(set(disponiveis[nome].tolist()))
        console.print(f"    ({i}) {nome} ({n_niveis} "
                      f"{'niveis' if is_pt else 'levels'})")
    lbl = (f"Escolha ate {max_fatores} (numeros separados por virgula)"
           if is_pt else f"Choose up to {max_fatores} (comma-separated numbers)")
    escolha = _ask(f"  [{PA}]{lbl}:[/{PA}] ").strip()
    try:
        idxs = [int(s) for s in escolha.split(",") if s.strip()]
    except ValueError:
        idxs = []
    idxs = [i for i in idxs if 1 <= i <= len(nomes)][:max_fatores]
    if not idxs:
        console.print(f"  [{PR}]{'Nenhum fator valido escolhido' if is_pt else 'No valid factor chosen'}[/{PR}]")
        return None
    return {nomes[i - 1]: disponiveis[nomes[i - 1]] for i in idxs}


def _menu_tec_asca(cfg: Config, is_pt: bool) -> None:
    from guaraci.tecnicas_avancadas import rodar_asca
    carregado = _tecavan_carregar_dataset(cfg, is_pt)
    if carregado is None:
        return
    wn, X, rotulos, conc, mae_id, metadados = carregado
    fatores = _tecavan_escolher_fatores(rotulos, metadados, is_pt)
    if fatores is None:
        _pause(); return
    try:
        rel = rodar_asca(X, fatores, mae_id=mae_id)
    except ValueError as e:
        console.print(f"  [{PR}]{escape(str(e))}[/{PR}]"); _pause(); return

    console.print()
    console.print(f"  [{PM}]{'Group-aware (mae_id)' if is_pt else 'Group-aware (mae_id)'}: "
                  f"{'sim' if rel.group_aware else 'NAO'}[/{PM}]")
    for nome, efeito in rel.decomposicao["efeitos"].items():
        p_txt = ""
        if rel.permutacao is not None:
            p = rel.permutacao[nome]["p_value"]
            p_txt = f", p={p:.4f}" + ("  [SIGNIFICATIVO]" if p < 0.05 else "")
        console.print(f"  [{PW}]{nome}[/{PW}]: "
                      f"{efeito['fracao_ss_total']*100:.1f}% "
                      f"{'da variancia total' if is_pt else 'of total variance'}{p_txt}")
    console.print(f"  [{PM}]ss_desbalanco: "
                  f"{rel.decomposicao['ss_desbalanco']:.3g} "
                  f"({'grande = fatores correlacionados, ver limite de escopo do modulo' if is_pt else 'large = correlated factors, see module scope limit'})[/{PM}]")
    _figura_tecnica("fig_asca", rel.decomposicao, fatores, wn, cfg,
                    _tecavan_pasta_saida(cfg, "asca"), permutacao=rel.permutacao)
    _pause()


def _menu_tec_epo_glsw(cfg: Config, is_pt: bool) -> None:
    from guaraci.tecnicas_avancadas import rodar_epo_glsw
    carregado = _tecavan_carregar_dataset(cfg, is_pt)
    if carregado is None:
        return
    wn, X, rotulos, conc, mae_id, metadados = carregado
    console.print(f"  [{PM}]{'grupo_interesse = o que PRESERVAR; fator_incomodo = o que REMOVER' if is_pt else 'grupo_interesse = what to PRESERVE; fator_incomodo = what to REMOVE'}[/{PM}]")
    console.print(f"  [{PA}]1) {'grupo_interesse' if is_pt else 'grupo_interesse'}:[/{PA}]")
    f1 = _tecavan_escolher_fatores(rotulos, metadados, is_pt, max_fatores=1)
    if f1 is None:
        _pause(); return
    console.print(f"  [{PA}]2) {'fator_incomodo' if is_pt else 'fator_incomodo'}:[/{PA}]")
    f2 = _tecavan_escolher_fatores(rotulos, metadados, is_pt, max_fatores=1)
    if f2 is None:
        _pause(); return
    (nome1, vals1), = f1.items()
    (nome2, vals2), = f2.items()
    if nome1 == nome2:
        console.print(f"  [{PR}]{'Os dois fatores precisam ser diferentes' if is_pt else 'The two factors must be different'}[/{PR}]")
        _pause(); return

    metodo = _ask(f"  [{PA}]{'Metodo (1=EPO, 2=GLSW, Enter=EPO)' if is_pt else 'Method (1=EPO, 2=GLSW, Enter=EPO)'}:[/{PA}] ").strip()
    metodo_nome = "GLSW" if metodo == "2" else "EPO"
    try:
        rel = rodar_epo_glsw(X, vals1, vals2, nome1, nome2, metodo=metodo_nome)
    except ValueError as e:
        console.print(f"  [{PR}]{escape(str(e))}[/{PR}]"); _pause(); return

    console.print()
    console.print(f"  [{PG}]✔ {rel.metodo}[/{PG}] ({rel.n_pares_diferenca} "
                  f"{'pares de diferenca' if is_pt else 'difference pairs'})")
    console.print(f"  [{PW}]{'Fracao de variancia total removida' if is_pt else 'Total variance fraction removed'}: "
                  f"{rel.variancia_removida_fracao*100:.1f}%[/{PW}]")
    import pandas as _pd
    caminho = os.path.join(_tecavan_pasta_saida(cfg, rel.metodo.lower()), f"X_corrigido_{rel.metodo.lower()}.csv")
    df = _pd.DataFrame(rel.X_corrigido, columns=[f"{v:g}" for v in wn])
    df.insert(0, "classe", list(rotulos))
    df.to_csv(caminho, index=False, sep=";", decimal=",")
    console.print(f"  [{PM}]{escape(caminho)}[/{PM}]")
    console.print(f"  [{PM}]{'Diagnostico exploratorio -- NAO integrado ao pipeline de treino automatico (ver limite de escopo do modulo: nao se aplica quando o fator de perturbacao e colinear com o alvo cientifico)' if is_pt else 'Exploratory diagnostic -- NOT wired into the automatic training pipeline (see module scope limit: does not apply when the nuisance factor is collinear with the scientific target)'}[/{PM}]")
    _pause()


def _menu_tec_mcr_als(cfg: Config, is_pt: bool) -> None:
    import numpy as np
    from guaraci.tecnicas_avancadas import rodar_mcr_als
    carregado = _tecavan_carregar_dataset(cfg, is_pt)
    if carregado is None:
        return
    wn, X, rotulos, conc, mae_id, metadados = carregado
    if np.any(X < 0):
        console.print(f"  [{PA}]{'Aviso: X tem valores negativos -- nao-negatividade sera desligada (MCR-ALS assume concentracao/espectro >= 0)' if is_pt else 'Warning: X has negative values -- non-negativity will be off (MCR-ALS assumes concentration/spectrum >= 0)'}[/{PA}]")
    n_comp_raw = _ask(f"  [{PA}]{'Numero de componentes (Enter=2)' if is_pt else 'Number of components (Enter=2)'}:[/{PA}] ").strip()
    try:
        n_comp = int(n_comp_raw) if n_comp_raw else 2
    except ValueError:
        console.print(f"  [{PR}]{'Numero invalido' if is_pt else 'Invalid number'}[/{PR}]"); _pause(); return

    usar_restricao = False
    indice_alvo = None
    if conc is not None and np.any(~np.isnan(conc)):
        resp = _ask(f"  [{PA}]{'Usar restricao de correlacao com o teor/concentracao conhecido? (s/n, Enter=n)' if is_pt else 'Use correlation restriction with known content/concentration? (y/n, Enter=n)'}:[/{PA}] ").strip().lower()
        usar_restricao = resp in ("s", "sim", "y", "yes")
        if usar_restricao:
            idx_raw = _ask(f"  [{PA}]{'Indice do componente-alvo (0-indexado, Enter=0)' if is_pt else 'Target component index (0-indexed, Enter=0)'}:[/{PA}] ").strip()
            try:
                indice_alvo = int(idx_raw) if idx_raw else 0
            except ValueError:
                console.print(f"  [{PR}]{'Numero invalido' if is_pt else 'Invalid number'}[/{PR}]"); _pause(); return

    status_msg = "Rodando MCR-ALS..." if is_pt else "Running MCR-ALS..."
    try:
        with console.status(f"[{PA}]{status_msg}[/{PA}]"):
            resultado = rodar_mcr_als(
                np.clip(X, 0, None) if np.any(X < 0) else X, n_comp,
                conc=conc if usar_restricao else None,
                indice_componente_alvo=indice_alvo if usar_restricao else None,
                nao_negativo_c=True, nao_negativo_s=True)
    except ValueError as e:
        console.print(f"  [{PR}]{escape(str(e))}[/{PR}]"); _pause(); return

    console.print()
    console.print(f"  [{PG}]✔ MCR-ALS[/{PG}] ({resultado.n_iter} "
                  f"{'iteracoes' if is_pt else 'iterations'}, "
                  f"{'convergiu' if resultado.convergiu else ('NAO convergiu' if is_pt else 'did NOT converge')})")
    console.print(f"  [{PW}]lack-of-fit: {resultado.lof_percent:.2f}%[/{PW}]")
    from guaraci.mcr_als import MCRALSResultadoSupervisionado
    if isinstance(resultado, MCRALSResultadoSupervisionado):
        console.print(f"  [{PW}]{'Correlacao componente-alvo x referencia (calibracao)' if is_pt else 'Target component x reference correlation (calibration)'}: "
                      f"{resultado.correlacao_calibracao:.3f}[/{PW}]")
    import pandas as _pd
    pasta = _tecavan_pasta_saida(cfg, "mcr_als")
    nomes_c = [f"comp{i + 1}" for i in range(resultado.C.shape[1])]
    dc = _pd.DataFrame(resultado.C, columns=nomes_c)
    dc.insert(0, "classe", list(rotulos))
    dc.to_csv(os.path.join(pasta, "mcr_concentracoes_C.csv"), index=False, sep=";", decimal=",")
    _pd.DataFrame(resultado.S, index=nomes_c, columns=[f"{v:g}" for v in wn]).to_csv(
        os.path.join(pasta, "mcr_espectros_puros_S.csv"), sep=";", decimal=",")
    console.print(f"  [{PM}]{escape(pasta)} (C e S)[/{PM}]")
    _figura_tecnica("fig_mcr_als", resultado.C, resultado.S, wn, rotulos, cfg, pasta,
                    lof_percent=resultado.lof_percent)
    console.print(f"  [{PM}]{'Interpretativo -- NAO substitui PLS-R para quantificacao (ver docstring de mcr_als_com_restricao_correlacao)' if is_pt else 'Interpretive -- does NOT replace PLS-R for quantification (see mcr_als_com_restricao_correlacao docstring)'}[/{PM}]")
    _pause()


def _tecavan_pasta_saida(cfg: Config, nome: str) -> str:
    pasta = os.path.join(str(getattr(cfg, "output_root_folder", "") or "resultados"), "TecnicasEspeciais", nome)
    os.makedirs(pasta, exist_ok=True)
    return pasta


def _menu_tec_parafac(cfg: Config, is_pt: bool) -> None:
    """PARAFAC sobre o dado do modo eem (a grade vem do leitor)."""
    import numpy as np
    from guaraci.tecnicas_avancadas import rodar_parafac_eem
    if str(getattr(cfg, "mode", "")) != "eem":
        console.print(f"  [{PR}]{'PARAFAC exige modo_entrada: eem (em [2.3] Dados).' if is_pt else 'PARAFAC requires modo_entrada: eem (in [2.3] Data).'}[/{PR}]")
        _pause(); return
    carregado = _tecavan_carregar_dataset(cfg, is_pt, manter_constantes=True)
    if carregado is None:
        return
    _wn, X, _rot, _conc, _mae, metadados = carregado
    aviso_esp = metadados.attrs.get("aviso_espalhamento")
    if aviso_esp:
        console.print(f"  [{PA}]{escape(str(aviso_esp))}[/{PA}]")
    raw = _ask(f"  [{PA}]{'Numero de componentes [ex.: 3] (Enter = 3)' if is_pt else 'Number of components [e.g. 3] (Enter = 3)'}:[/{PA}] ").strip()
    nao_neg = bool(getattr(cfg, "parafac_nao_negativo", True))
    try:
        r = rodar_parafac_eem(X, metadados.attrs["grade_excitacao"], metadados.attrs["grade_emissao"], int(raw or 3),
                              nao_negativo=nao_neg, mascara_faltantes=metadados.attrs.get("mascara_eem_faltante"))
    except ImportError:
        console.print(f"  [{PR}]{'PARAFAC precisa do pacote opcional tensorly' if is_pt else 'PARAFAC needs the optional package tensorly'}: "
                      f"pip install '.[multiway]'[/{PR}]"); _pause(); return
    except (ValueError, KeyError) as e:
        console.print(f"  [{PR}]{escape(str(e))}[/{PR}]"); _pause(); return
    pasta = _tecavan_pasta_saida(cfg, "parafac")
    caminho = os.path.join(pasta, "parafac_fator_amostra.csv")
    np.savetxt(caminho, r.fator_amostra, delimiter=";", fmt="%.6g")
    _gravar_resumo_parafac(r, pasta, cfg, metadados)
    _figura_tecnica("fig_parafac_eem", r, metadados.attrs["grade_excitacao"],
                    metadados.attrs["grade_emissao"], _rot, cfg, pasta)
    console.print(f"  [{PG}]PARAFAC[/{PG}] {r.fator_amostra.shape[1]} {'componentes' if is_pt else 'components'}; "
                  f"{'erro relativo de reconstrucao' if is_pt else 'relative reconstruction error'}: {r.erro_reconstrucao_relativo:.3f}; "
                  f"{'consistencia do nucleo (CORCONDIA)' if is_pt else 'core consistency (CORCONDIA)'}: {r.consistencia_nucleo:.0f}%")
    console.print(f"  [{PM}]{'Restricao' if is_pt else 'Constraint'}: {r.restricao} ({r.algoritmo})"
                  + (f"; {r.n_faltantes} {'celulas faltantes (espalhamento removido) ignoradas no ajuste' if is_pt else 'missing cells (removed scatter) ignored in the fit'}" if r.n_faltantes else "")
                  + f"[/{PM}]")
    console.print(f"  [{PM}]{escape(r.nota if is_pt else r.nota_en)}[/{PM}]")
    console.print(f"  [{PM}]{'Compare 2, 3, 4... componentes: fique com o maior numero que mantem a consistencia do nucleo alta (> ~60-80%). Espalhamento Rayleigh/Raman nao removido derruba o ajuste: use eem_espalhamento: nan (ou interp) em [2.3] Dados, ou exporte a EEM sem espalhamento.' if is_pt else 'Compare 2, 3, 4... components: keep the largest number that keeps core consistency high (> ~60-80%). Unremoved Rayleigh/Raman scatter ruins the fit: use eem_espalhamento: nan (or interp) in [2.3] Data, or export the EEM without scatter.'}[/{PM}]")
    console.print(f"  [{PM}]{escape(caminho)}[/{PM}]")
    _pause()


def _gravar_resumo_parafac(r, pasta: str, cfg: Config, metadados) -> None:
    """`parafac_resumo.json` e `parafac_resumo.txt` da pasta do PARAFAC: restricao, algoritmo, CORCONDIA, tratamento do
    espalhamento e a nota (PT e EN) sobre o PARAFAC nao-negativo."""
    import json
    from guaraci.config import __version__
    esp = dict(metadados.attrs.get("eem_espalhamento") or {})
    resumo = {
        "tecnica": "PARAFAC (EEM)",
        "versao_guaraci": __version__,
        "restricao": r.restricao,
        "algoritmo": r.algoritmo,
        "parafac_nao_negativo": r.restricao == "nao-negativo",
        "n_componentes": int(r.fator_amostra.shape[1]),
        "erro_reconstrucao_relativo": round(float(r.erro_reconstrucao_relativo), 6),
        "consistencia_nucleo_percent": round(float(r.consistencia_nucleo), 2),
        "n_celulas_faltantes": int(r.n_faltantes),
        "n_iteracoes_em": int(r.n_iter_em),
        "espalhamento": esp,
        "nota_pt": r.nota,
        "nota_en": r.nota_en,
    }
    with open(os.path.join(pasta, "parafac_resumo.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    linhas = [f"PARAFAC (EEM) - GUARACI {__version__}",
              f"Restricao / Constraint: {r.restricao}",
              f"Algoritmo / Algorithm: {r.algoritmo}",
              f"Componentes / Components: {resumo['n_componentes']}",
              f"Erro relativo de reconstrucao / Relative reconstruction error: {resumo['erro_reconstrucao_relativo']}",
              f"CORCONDIA (%): {resumo['consistencia_nucleo_percent']}",
              f"Celulas faltantes ignoradas / Missing cells ignored: {resumo['n_celulas_faltantes']}",
              "", "PT: " + r.nota, "EN: " + r.nota_en, ""]
    with open(os.path.join(pasta, "parafac_resumo.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))


def _menu_tec_cow(cfg: Config, is_pt: bool) -> None:
    """COW sobre os cromatogramas do modo hplc ou gcms (mesma grade)."""
    from guaraci.tecnicas_avancadas import rodar_cow
    if str(getattr(cfg, "mode", "")) not in ("hplc", "gcms"):
        console.print(f"  [{PR}]{'COW exige modo_entrada: hplc ou gcms (em [2.3] Dados).' if is_pt else 'COW requires modo_entrada: hplc or gcms (in [2.3] Data).'}[/{PR}]")
        _pause(); return
    carregado = _tecavan_carregar_dataset(cfg, is_pt)
    if carregado is None:
        return
    wn, X, rotulos, _conc, _mae, _meta = carregado
    seg = _ask(f"  [{PA}]{'Segmentos [ex.: 30] (Enter = 30)' if is_pt else 'Segments [e.g. 30] (Enter = 30)'}:[/{PA}] ").strip()
    folga = _ask(f"  [{PA}]{'Folga em pontos [ex.: 10] (Enter = 10)' if is_pt else 'Slack in points [e.g. 10] (Enter = 10)'}:[/{PA}] ").strip()
    try:
        with console.status(f"[{PA}]COW...[/{PA}]"):
            r = rodar_cow(X, n_segmentos=int(seg or 30), slack=int(folga or 10))
    except ValueError as e:
        console.print(f"  [{PR}]{escape(str(e))}[/{PR}]"); _pause(); return
    pasta = _tecavan_pasta_saida(cfg, "cow")
    caminho = os.path.join(pasta, "cromatogramas_alinhados.csv")
    import pandas as _pd
    df = _pd.DataFrame(r.X_alinhado, columns=[f"{v:g}" for v in wn])
    df.insert(0, "classe", list(rotulos))
    # grupo e teor seguem junto: o CSV alinhado volta ao modo csv (coluna_grupo/
    # coluna_concentracao) sem perder a validacao por grupo nem a quantificacao.
    if _mae is not None:
        df.insert(1, "grupo", [str(v) for v in _mae])
    if _conc is not None:
        df.insert(1, "conc", list(_conc))
    df.to_csv(caminho, index=False, sep=";", decimal=",")
    console.print(f"  [{PG}]COW[/{PG}] {'correlacao media par a par' if is_pt else 'mean pairwise correlation'}: "
                  f"{r.correlacao_antes:.3f} -> {r.correlacao_depois:.3f} ({'referencia' if is_pt else 'reference'}: "
                  f"{'linha' if is_pt else 'row'} {r.indice_referencia + 1})")
    console.print(f"  [{PM}]{escape(caminho)}[/{PM}]")
    _pause()


def _menu_tec_portao(cfg: Config, is_pt: bool) -> None:
    """Portao de aceite de EMSC ou OSC sobre o dataset de [2] Dados (com x sem, mesmos folds)."""
    from guaraci.tecnicas_avancadas import rodar_portao_correcao
    carregado = _tecavan_carregar_dataset(cfg, is_pt)
    if carregado is None:
        return
    _wn, X, rotulos, conc, mae, _meta = carregado
    metodo = _ask(f"  [{PA}]{'Correcao (emsc/osc) [Enter = emsc]' if is_pt else 'Correction (emsc/osc) [Enter = emsc]'}:[/{PA}] ").strip().lower() or "emsc"
    classif = str(getattr(cfg, "objective", "classificacao")) != "quantificacao"
    try:
        with console.status(f"[{PA}]{metodo.upper()}: {'com x sem, 10 particoes' if is_pt else 'with x without, 10 partitions'}...[/{PA}]"):
            v = rodar_portao_correcao(X, rotulos if classif else conc, mae, metodo, classificacao=classif)
    except ValueError as e:
        console.print(f"  [{PR}]{escape(str(e))}[/{PR}]"); _pause(); return
    console.print(f"  [{PW}]{escape(v.resumo())}[/{PW}]")
    if not v.poder_suficiente:
        console.print(f"  [{PA}]{'Poder insuficiente: o resultado nao permite concluir que a correcao nao ajuda.' if is_pt else 'Insufficient power: the result does not show that the correction does not help.'}[/{PA}]")
    console.print(f"  [{PM}]{'A decisao de usar a correcao e sua; nada foi alterado na configuracao.' if is_pt else 'Whether to use the correction is your decision; the configuration was not changed.'}[/{PM}]")
    _pause()


def _menu_tec_pooled_local(cfg: Config, is_pt: bool) -> None:
    """Modelo unico (todas as especies) x modelo por especie, sobre a concentracao de [2] Dados."""
    import numpy as np
    from guaraci.tecnicas_avancadas import rodar_pooled_local
    carregado = _tecavan_carregar_dataset(cfg, is_pt)
    if carregado is None:
        return
    _wn, X, rotulos, conc, mae, _meta = carregado
    try:
        with console.status(f"[{PA}]{'Comparando por especie (10 particoes)' if is_pt else 'Comparing per species (10 partitions)'}...[/{PA}]"):
            decisoes = rodar_pooled_local(X, conc if conc is not None else np.full(len(rotulos), np.nan), rotulos, mae)
    except ValueError as e:
        console.print(f"  [{PR}]{escape(str(e))}[/{PR}]"); _pause(); return
    for d in decisoes:
        det = d.veredito.resumo() if d.veredito is not None else (
            f"{'dados insuficientes' if is_pt else 'not enough data'} ({d.n_amostras} < {d.n_minimo})")
        console.print(f"  [{PA}]{escape(d.especie)}[/{PA}] -> {d.recomendacao}: {escape(det)}")
    console.print(f"  [{PM}]{'Recomendacao conservadora: so troca para o modelo da especie se o teste aprovar com poder suficiente.' if is_pt else 'Conservative recommendation: switches to the per-species model only if the test approves with enough power.'}[/{PM}]")
    _pause()


def _menu_tec_fusao_multibloco(cfg: Config, is_pt: bool) -> None:
    import numpy as np
    from guaraci.tecnicas_avancadas import rodar_fusao_multibloco
    console.print(f"  [{PM}]{'Precisa de 2 pastas com a MESMA amostra fisica, MESMA ordem de linha' if is_pt else 'Needs 2 folders with the SAME physical sample, SAME row order'}[/{PM}]")
    blocos: Dict[str, np.ndarray] = {}
    y_bloco: Optional[np.ndarray] = None
    for i in (1, 2):
        nome = _ask(f"  [{PA}]{'Nome do bloco' if is_pt else 'Block name'} {i} "
                    f"(ex: NIR):[/{PA}] ").strip() or f"bloco{i}"
        modo = _ask(f"  [{PA}]{'Modo de entrada' if is_pt else 'Input mode'} "
                    f"({nome}) (dx/csv/spc/sp/opus/rmn/hplc/gcms):[/{PA}] ").strip() or "dx"
        pasta = _ask(f"  [{PA}]{'Pasta (ou arquivo CSV)' if is_pt else 'Folder (or CSV file)'} "
                     f"({nome}):[/{PA}] ").strip().strip('"')
        cfg_bloco = Config()
        cfg_bloco.mode = modo
        cfg_bloco.input_folder = pasta
        cfg_bloco.csv_file = pasta
        if modo == "csv":
            cfg_bloco.class_column = _ask(f"  [{PA}]{'Coluna de classe' if is_pt else 'Class column'} ({nome}) "
                                          f"[ex.: Class] (Enter = {cfg_bloco.class_column}):[/{PA}] ").strip() or cfg_bloco.class_column
            cfg_bloco.conc_column = _ask(f"  [{PA}]{'Coluna de teor' if is_pt else 'Content column'} ({nome}) "
                                         f"[ex.: PeroxideValue] (Enter = {cfg_bloco.conc_column or 'nenhuma'}):[/{PA}] ").strip() or cfg_bloco.conc_column
        try:
            with console.status(f"[{PA}]{'Carregando' if is_pt else 'Loading'} {nome}...[/{PA}]"):
                wn_b, X_b, rot_b, conc_b, mae_b, meta_b = pq.load_data(cfg_bloco)
        except Exception as e:  # noqa: BLE001 -- mesma disciplina de _tecavan_carregar_dataset
            _m = str(e) if is_pt else _re.sub(r"Modo de entrada desconhecido: (.*)\. Modos disponiveis: ",
                                                 r"Unknown input mode: \1. Available modes: ", str(e))
            console.print(f"  [{PR}]{'Erro ao carregar' if is_pt else 'Error loading'} {nome}: {escape(_m)}[/{PR}]")
            _pause(); return
        blocos[nome] = X_b
        console.print(f"  [{PG}]✔ {nome}: {X_b.shape[0]} "
                      f"{'amostras' if is_pt else 'samples'}, {X_b.shape[1]} "
                      f"{'variaveis' if is_pt else 'variables'}[/{PG}]")
        if y_bloco is None and conc_b is not None:
            y_bloco = conc_b

    if y_bloco is None:
        console.print(f"  [{PR}]{'Nenhum dos dois blocos tem teor/concentracao (y) -- fusao multibloco aqui so suporta regressao' if is_pt else 'Neither block has content/concentration (y) -- multiblock fusion here only supports regression'}[/{PR}]")
        _pause(); return

    try:
        resultado, fatias = rodar_fusao_multibloco(blocos, y_bloco)
    except ValueError as e:   # BlockMismatchError e' subclasse de ValueError
        console.print(f"  [{PR}]{escape(str(e))}[/{PR}]"); _pause(); return

    console.print()
    console.print(f"  [{PG}]✔ {'Fusao multibloco' if is_pt else 'Multiblock fusion'} "
                  f"({', '.join(fatias)})[/{PG}]")
    console.print(f"  [{PW}]RMSEP={resultado.rmsep:.4g}  R2cal={resultado.r2cal:.3f}  "
                  f"R2val={resultado.r2val:.3f}  n_lv={resultado.n_lv}  "
                  f"({resultado.n_cal} cal / {resultado.n_val} val)[/{PW}]")
    console.print(f"  [{PM}]{'A fusao nem sempre supera o melhor bloco sozinho: compare este numero contra cada bloco treinado isoladamente antes de adotar' if is_pt else 'Fusion does not always beat the best single block: compare this number against each block trained alone before adopting'}[/{PM}]")
    _pause()


def _menu_tec_vip(cfg: Config, is_pt: bool) -> None:
    """[3.5] 9 -- VIP do PLS com estabilidade nas dobras agrupadas (mesma logica de `guaraci vip`)."""
    from guaraci import vip_cli
    tab = _perguntar("tecnicas_avancadas", cfg, 1, 3, "Tabela CSV (amostras x variaveis)" if is_pt else "CSV table (samples x variables)",
                     "tabela.csv", str(cfg.csv_file) if os.path.isfile(str(getattr(cfg, "csv_file", "") or "")) else "")
    if not tab:
        return
    if not os.path.isfile(tab):
        console.print(f"  [{PR}]{'Arquivo nao encontrado' if is_pt else 'File not found'}: {escape(tab)}[/{PR}]"); _pause(); return
    alvo = _perguntar("tecnicas_avancadas", cfg, 2, 3, "Coluna da classe (ou r:COLUNA para uma resposta numerica)" if is_pt
                      else "Class column (or r:COLUMN for a numeric response)", "classe", str(getattr(cfg, "class_column", "") or ""))
    if not alvo:
        return
    grupo = _perguntar("tecnicas_avancadas", cfg, 3, 3, "Coluna da unidade independente (amostra, lote)" if is_pt
                       else "Independent unit column (sample, batch)", "amostra", str(getattr(cfg, "group_column", "") or ""))
    if grupo is None:
        return
    args = [tab] + (["--resposta", alvo[2:]] if alvo.lower().startswith("r:") else ["--classe", alvo])
    args += (["--grupo", grupo] if grupo else []) + ["--saida", os.path.join(os.path.dirname(os.path.abspath(tab)), _nome_pasta("resultados_vip"))]
    extras = _opcoes_extras("tecnicas_avancadas", cfg, [
        ("--componentes=", "variaveis latentes (padrao 2)", "latent variables (default 2)"),
        ("--limiar=", "limiar do VIP (padrao 1.0)", "VIP threshold (default 1.0)"),
        ("--divisoes=", "dobras por repeticao (padrao 5)", "folds per repeat (default 5)"),
        ("--repeticoes=", "repeticoes (padrao 10)", "repeats (default 10)"),
        ("--variaveis=", "colunas a usar, separadas por virgula", "columns to use, comma-separated"),
        ("--semente=", "semente aleatoria", "random seed"),
        ("--sem-figuras", "nao gera figuras", "do not draw figures")])
    if extras is None:
        return
    vip_cli.executar(args + extras + (["--idioma", "en"] if not is_pt else []))
    _pause()


def _menu_tec_transferir(cfg: Config, is_pt: bool) -> None:
    """[3.5] 10 -- transferencia de calibracao DS/PDS (mesma logica de `guaraci transferir`)."""
    from guaraci import transferir_cli
    caminhos = []
    for i, (pt, en, ex) in enumerate((("CSV das amostras de transferencia no instrumento mestre", "CSV of the transfer samples on the master instrument", "mestre.csv"),
                                      ("CSV das MESMAS amostras no instrumento escravo", "CSV of the SAME samples on the slave instrument", "escravo.csv"),
                                      ("CSV dos espectros novos do escravo, a converter", "CSV of the new slave spectra to convert", "novos_escravo.csv")), 1):
        c = _perguntar("tecnicas_avancadas", cfg, i, 4, pt if is_pt else en, ex)
        if not c:
            return
        if not os.path.isfile(c):
            console.print(f"  [{PR}]{'Arquivo nao encontrado' if is_pt else 'File not found'}: {escape(c)}[/{PR}]"); _pause(); return
        caminhos.append(c)
    metodo = _perguntar("tecnicas_avancadas", cfg, 4, 4, "Metodo: pds (padrao) ou ds" if is_pt else "Method: pds (default) or ds", "pds", "pds")
    if metodo is None:
        return
    args = ["--mestre", caminhos[0], "--escravo", caminhos[1], "--novos", caminhos[2], "--metodo", (metodo or "pds").lower(),
            "--saida", os.path.join(os.path.dirname(os.path.abspath(caminhos[2])), _nome_pasta("resultados_transferencia"))]
    extras = _opcoes_extras("tecnicas_avancadas", cfg, [
        ("--janela=", "janela do PDS (padrao 5)", "PDS window (default 5)"),
        ("--alpha=", "regularizacao (padrao: escolhida por validacao)", "regularisation (default: chosen by validation)")])
    if extras is None:
        return
    if transferir_cli.executar(args + extras + (["--idioma", "en"] if not is_pt else [])) == 0:
        _figura_tecnica("figuras_transferencia_de_argv", args + extras, cfg)
    _pause()


#: O que cada item de [3.5] Tecnicas especiais faz (painel do "?").
_FAZ_TEC_AVANCADAS: Dict[str, Tuple[str, str]] = {
    "1": ("ASCA: separa a variacao dos espectros por fator do delineamento (ex.: especie, dia) e testa cada fator por permutacao.",
          "ASCA: splits the spectral variation by design factor (e.g. species, day) and tests each factor by permutation."),
    "2": ("EPO/GLSW: remove dos espectros a variacao de um fator indesejado (temperatura, instrumento) antes do modelo.",
          "EPO/GLSW: removes the variation of an unwanted factor (temperature, instrument) from the spectra before the model."),
    "3": ("MCR-ALS: decompoe os espectros em perfis de concentracao e espectros puros; pode usar o teor conhecido como restricao.",
          "MCR-ALS: decomposes the spectra into concentration profiles and pure spectra; can use the known content as a constraint."),
    "4": ("Fusao multibloco: junta dois blocos de dados (ex.: NIR + Raman) da mesma amostra num modelo so'.",
          "Multiblock fusion: joins two data blocks (e.g. NIR + Raman) of the same sample in a single model."),
    "5": ("PARAFAC: decompoe matrizes de fluorescencia excitacao-emissao em perfis de excitacao, emissao e escores.",
          "PARAFAC: decomposes excitation-emission fluorescence matrices into excitation, emission and score profiles."),
    "6": ("COW: alinha os tempos de retencao dos cromatogramas a uma referencia antes do modelo.",
          "COW: aligns the retention times of the chromatograms to a reference before the model."),
    "7": ("Testa se EMSC ou OSC melhora o PLS nas mesmas particoes por grupo; so' aceita a correcao se ganhar.",
          "Tests whether EMSC or OSC improves PLS on the same group partitions; accepts the correction only if it wins."),
    "8": ("Compara, por especie, o modelo unico de todas as especies com um modelo so' daquela especie.",
          "Compares, per species, the single model of all species with a model of that species alone."),
    "9": ("VIP do PLS recalculado em cada dobra agrupada: mostra quais variaveis sao importantes de forma estavel.",
          "PLS VIP recomputed in each grouped fold: shows which variables are stably important."),
    "10": ("Leva um modelo de um instrumento para outro (DS ou PDS) usando amostras medidas nos dois.",
           "Moves a model from one instrument to another (DS or PDS) using samples measured on both."),
}


def _menu_tecnicas_avancadas(cfg: Optional[Config] = None) -> None:
    """[3.5] Tecnicas especiais -- ASCA, EPO/GLSW, MCR-ALS (com/sem restricao
    de correlacao) e fusao multibloco: existiam implementadas e testadas em
    isolamento, mas sem NENHUM caminho de execucao real em CLI/web (achado
    da auditoria de acessibilidade, preparacao p/ o 1o usuario externo).
    Cada tecnica roda sobre o dataset JA configurado em [2.3] Dados (ASCA/
    EPO-GLSW/MCR-ALS) ou sobre 2 pastas apontadas na hora (fusao
    multibloco, que por definicao precisa de 2 blocos). Logica real em
    `tecnicas_avancadas.py` (testavel sem Rich) -- esta funcao so' coleta
    entrada e formata saida, mesmo padrao de `_menu_audit`/`_menu_selecao_
    amostras`."""
    cfg = cfg or Config()
    while True:     # voltar de uma tecnica volta para esta lista, nao para o grupo
        if _tecnicas_avancadas_uma_escolha(cfg):
            return


def _tecnicas_avancadas_uma_escolha(cfg: Config) -> bool:
    """Desenha a lista de tecnicas especiais e trata uma escolha. Devolve True para sair da tela."""
    lang = _lang(); is_pt = lang == "PT"
    _cabecalho_tela("tecnicas_avancadas", cfg)

    opcoes = [
        ("1", "ASCA", _menu_tec_asca),
        ("2", "EPO/GLSW", _menu_tec_epo_glsw),
        ("3", "MCR-ALS", _menu_tec_mcr_als),
        ("4", "Fusao multibloco" if is_pt else "Multiblock fusion", _menu_tec_fusao_multibloco),
        ("5", "PARAFAC (fluorescencia EEM)" if is_pt else "PARAFAC (EEM fluorescence)", _menu_tec_parafac),
        ("6", "Alinhar cromatogramas (COW)" if is_pt else "Align chromatograms (COW)", _menu_tec_cow),
        ("7", "Testar correcao de sinal (EMSC/OSC)" if is_pt else "Test a signal correction (EMSC/OSC)", _menu_tec_portao),
        ("8", "Modelo unico x modelo por especie" if is_pt else "Pooled x per-species model", _menu_tec_pooled_local),
        ("9", "VIP com estabilidade (sua tabela)" if is_pt else "VIP with stability (your table)", _menu_tec_vip),
        ("10", "Transferir calibracao DS/PDS (seus CSV)" if is_pt else "Transfer calibration DS/PDS (your CSVs)", _menu_tec_transferir),
    ]
    for k, nome, _fn in opcoes:
        console.print(f"  [{PA}][{k}][/{PA}] {nome}")
    _rodape_tela()
    L = 0 if is_pt else 1
    _definir_ajuda("tecnicas_avancadas", [(k, nome, _FAZ_TEC_AVANCADAS[k][L]) for k, nome, _fn in opcoes])
    escolha = _ask(f"  [{PA}]{'Opcao' if is_pt else 'Option'}:[/{PA}] ").strip()
    for k, _nome, fn in opcoes:
        if escolha == k:
            fn(cfg, is_pt)
            return False
    r = _tecla_comum(escolha, "tecnicas_avancadas", cfg)
    if r == "voltar":
        return True
    if r is None:
        console.print(f"  [{PM}]{_t('invalido')}[/{PM}]")
        _pause()
    return False


# ---------------------------------------------------------------------------
# AUDITORIA DE DELINEAMENTO — comando dedicado (Bloco 11)
# ---------------------------------------------------------------------------
def _menu_audit(cfg: Optional[Config] = None) -> None:
    """Auditoria de delineamento (Bloco 11) isolada -- roda
    `auditoria_delineamento.run_audit` sobre o dataset configurado (as
    MESMAS checagens que ja rodam automaticamente em toda execucao e
    aparecem no model card), sem exigir rodar classificacao/quantificacao
    inteira. Reaproveita `load_data`/`validate_input` -- mesmo caminho de
    dados que `pipeline.executar()` usa antes de chamar `run_audit`, nao
    duplica logica."""
    cfg = cfg or Config()
    lang = _lang()
    is_pt = lang == "PT"

    from guaraci.config_io import _validar_pasta_dados

    # Gate antes de carregar dados + rodar a auditoria (pode demorar em
    # datasets grandes): [G] Guaraci e [0] Voltar sem rodar. Depois do [G]
    # a tela e' redesenhada e pergunta de novo -- antes saia da tela
    # (achado B9, 2026-09-24; mesmo padrao de _menu_prediction/_menu_plan).
    gate_lbl = "[Enter] Rodar auditoria" if is_pt else "[Enter] Run audit"
    while True:
        _cabecalho_tela("auditoria", cfg)
        _rodape_tela()
        console.print()
        ok, msg = _validar_pasta_dados(cfg)
        if not ok:
            console.print(f"  [{PR}]{escape(_msg_validacao(msg))}[/{PR}]")
            console.print(f"  [{PM}]{'Configure a fonte de dados em' if is_pt else 'Configure the data source in'} "
                          f"[{PA}][2.3] {_t('t_dados')}[/{PA}].[/{PM}]")
            _pause(); return
        console.print(f"  [{PM}]{msg}[/{PM}]")
        raw_gate = _ask(f"  [{PM}]{gate_lbl}[/{PM}]: ").strip()
        if raw_gate:
            r = _tecla_comum(raw_gate, "auditoria", cfg)
            if r == "voltar":
                return
            continue
        break

    status_msg = "Carregando dados e auditando..." if is_pt else "Loading data and auditing..."
    try:
        with console.status(f"[{PA}]{status_msg}[/{PA}]"):
            wavenumbers, X_raw, rotulos, conc, mae_id, _metadados = pq.load_data(cfg)
            X_raw, wavenumbers, rotulos, conc, mae_id, _relatorio = pq.validate_input(
                X_raw, wavenumbers, rotulos, conc, mae_id)
            from guaraci.auditoria_delineamento import run_audit
            achados = run_audit(X_raw, wavenumbers, rotulos, cfg, conc, mae_id)
    except Exception as e:  # noqa: BLE001 -- dado externo pode falhar de
        # varias formas (parsing, faixa espectral vazia, etc.) -- reportar
        # a mensagem, nunca stack trace cru numa ferramenta interativa.
        console.print(f"  [{PR}]{'Erro ao carregar dados' if is_pt else 'Error loading data'}: "
                      f"{escape(str(e))}[/{PR}]")
        _pause(); return

    cores = {"ok": PG, "aviso": PA, "critico": PR, "silenciado": PM}
    console.print()
    for a in achados:
        cor = cores.get(a.severidade, PW)
        console.print(f"  [{cor}]{a.severidade.upper():>10}[/{cor}]  "
                      f"[{PW}]{escape(a.nome)}[/{PW}]: {escape(a.mensagem)}")

    from guaraci.plano_coleta import lacunas_de_sessoes, texto_lacunas
    lac = lacunas_de_sessoes(rotulos, mae_id)
    if lac["classes"]:
        tab = Table(box=rbox.SIMPLE_HEAD, border_style=PM, header_style=f"bold {PA}")
        tab.add_column("Classe" if is_pt else "Class")
        tab.add_column("Sessoes" if is_pt else "Sessions", justify="right")
        tab.add_column(f"Faltam p/ {lac['minimo_honesto']}" if is_pt else f"Missing for {lac['minimo_honesto']}", justify="right")
        tab.add_column(f"Faltam p/ {lac['minimo_conformal']}" if is_pt else f"Missing for {lac['minimo_conformal']}", justify="right")
        for x in lac["classes"]:
            tab.add_row(escape(x["classe"]), str(x["sessoes"]), str(x["faltam_honesto"]), str(x["faltam_conformal"]))
        console.print()
        console.print(Panel(Group(tab, *[Text(f"• {ln}") for ln in texto_lacunas(lac, lang)]),
                            title=f"[bold {PA}]{'Plano de coleta a partir dos seus dados' if is_pt else 'Collection plan from your data'}[/bold {PA}]",
                            border_style=PA, box=rbox.ROUNDED, padding=(0, 2)))

    n_criticos = sum(1 for a in achados if a.severidade == "critico")
    n_avisos = sum(1 for a in achados if a.severidade == "aviso")
    resumo_lbl = (f"{n_criticos} critico(s), {n_avisos} aviso(s) de {len(achados)} checagem(ns)."
                  if is_pt else
                  f"{n_criticos} critical, {n_avisos} warning(s) out of {len(achados)} check(s).")
    cor_resumo = PR if n_criticos else (PA if n_avisos else PG)
    console.print()
    console.print(Panel(
        Text.from_markup(f"  {resumo_lbl}"),
        border_style=cor_resumo, box=rbox.ROUNDED, padding=(0, 2),
    ))
    _pause()


def _menu_monitorar(cfg: Optional[Config] = None) -> None:
    """[4.2] Monitorar lotes: coleta os 3 campos (Passo n de 3) e chama
    `monitorar_cli.executar` -- a mesma logica de `guaraci monitorar`."""
    cfg = cfg or Config()
    is_pt = _lang() == "PT"
    from guaraci import monitorar_cli as _mon

    def _campo(passo: int, rotulo: str, exemplo: str, padrao: str = "") -> Optional[str]:
        while True:
            _cabecalho_tela("monitorar", cfg)
            _rodape_tela()
            console.print(f"\n  [{PM}]{'Passo' if is_pt else 'Step'} {passo} {'de' if is_pt else 'of'} 3[/{PM}]")
            if passo == 1:
                console.print(f"  [{PM}]{'x = usar o exemplo sintetico (gera e roda)' if is_pt else 'x = use the synthetic example (generates and runs)'}[/{PM}]")
            dica = f" [ex.: {exemplo}]" + (f" (Enter = {padrao})" if padrao else "")
            raw = _ask(f"  [{PA}]{escape(rotulo + dica)}:[/{PA}] ").strip().strip('"')
            if raw.lower() == "x" and passo == 1:
                return "x"
            r = _tecla_valor(raw, "monitorar") if raw else None
            if r == "voltar":
                return None
            if r == "tratada":
                continue
            return raw or padrao

    ref = _campo(1, "CSV de referencia (lotes produzidos em controle)" if is_pt else "Reference CSV (batches produced in control)",
                 "referencia.csv")
    if ref is None:
        return
    if ref == "x":
        from guaraci import exemplos as _ex
        from guaraci.cli_comandos import _rodar_monitorar_exemplo
        pasta = os.path.abspath(os.path.join("GUARACI_Exemplo", "monitorar"))
        _ex.gerar("monitorar", pasta)
        _rodar_monitorar_exemplo(pasta)
        _pause()
        return
    if not os.path.isfile(ref):
        console.print(f"  [{PR}]{'Arquivo nao encontrado' if is_pt else 'File not found'}: {escape(ref)}[/{PR}]")
        _pause(); return
    novos = _campo(2, "CSV com os lotes novos (mesmas colunas numericas)" if is_pt else "CSV with the new batches (same numeric columns)",
                   "novos.csv")
    if not novos:
        return
    if not os.path.isfile(novos):
        console.print(f"  [{PR}]{'Arquivo nao encontrado' if is_pt else 'File not found'}: {escape(novos)}[/{PR}]")
        _pause(); return
    try:
        cab = _ler_csv_detectando_separador(ref).columns
    except (OSError, ValueError, UnicodeDecodeError) as e:
        console.print(f"  [{PR}]{'Erro ao ler CSV' if is_pt else 'Error reading CSV'}: {escape(str(e))}[/{PR}]")
        _pause(); return
    detectada = next((str(c) for c in cab if str(c).strip().lower() in ("lote", "batch", "sessao", "session")), "")
    lote = _campo(3, "Coluna de lote ou sessao" if is_pt else "Batch or session column", "lote", detectada)
    if lote is None:
        return
    saida = os.path.join(os.path.dirname(os.path.abspath(novos)), _nome_pasta("resultados_monitoramento"))
    args = ["--referencia", ref, "--novos", novos, "--saida", saida] + (["--lote", lote] if lote else [])
    extras = _opcoes_extras("monitorar", cfg, [
        ("--id=", "coluna de identificacao da amostra", "sample id column"),
        ("--pcs=", "numero de componentes (padrao auto)", "number of components (default auto)"),
        ("--alpha=", "nivel do limite de controle (padrao 0.05)", "control-limit level (default 0.05)"),
        ("--min-lotes=", "lotes minimos de referencia", "minimum reference batches"),
        ("--autoescala", "autoescala as variaveis", "autoscale the variables"),
        ("--mais-graficos", "graficos extras de contribuicao", "extra contribution plots")])
    if extras is None:
        return
    _mon.executar(args + extras)
    _pause()


def _perguntar(tela: str, cfg: Config, passo: int, total: int, rotulo: str, exemplo: str,
               padrao: str = "") -> Optional[str]:
    """Um campo de tela-assistente no molde: Passo n de N, exemplo e padrao visiveis.
    Devolve None se o usuario voltou."""
    is_pt = _lang() == "PT"
    while True:
        _cabecalho_tela(tela, cfg)
        _rodape_tela()
        console.print(f"\n  [{PM}]{'Passo' if is_pt else 'Step'} {passo} {'de' if is_pt else 'of'} {total}[/{PM}]")
        dica = f" [ex.: {exemplo}]" + (f" (Enter = {padrao})" if padrao else "")
        raw = _ask(f"  [{PA}]{escape(rotulo + dica)}:[/{PA}] ").strip().strip('"')
        r = _tecla_valor(raw, tela) if raw else None
        if r == "voltar":
            return None
        if r == "tratada":
            continue
        return raw or padrao


def _opcoes_extras(tela: str, cfg: Config, opcoes: List[Tuple[str, str, str]]) -> Optional[List[str]]:
    """Ultimo passo opcional das telas-assistente: as opcoes finas do subcomando
    equivalente (as mesmas de `guaraci <comando> --ajuda`). opcoes = (flag, PT, EN);
    flag terminada em '=' pede valor. Devolve a lista de argumentos ou None (voltou)."""
    is_pt = _lang() == "PT"
    while True:
        _cabecalho_tela(tela, cfg)
        _rodape_tela()
        console.print(f"\n  [{PW}]{'Opcoes extras (opcional)' if is_pt else 'Extra options (optional)'}[/{PW}]")
        for flag, pt, en in opcoes:
            uso = flag[:-1] + " VALOR" if flag.endswith("=") else flag
            console.print(f"   [{PA}]{escape(uso):<24}[/{PA}] [{PM}]{escape(pt if is_pt else en)}[/{PM}]")
        raw = _ask(f"  [{PA}]{'Digite as opcoes (Enter = nenhuma, V = voltar)' if is_pt else 'Type the options (Enter = none, V = back)'}:[/{PA}] ").strip()
        if not raw:
            return []
        if raw.upper() in ("0", "V"):
            return None
        if raw == "?":
            _ajuda_da_tela(tela); continue
        import shlex
        try:
            toks = shlex.split(raw, posix=os.name != "nt")
        except ValueError as e:
            console.print(f"  [{PR}]{escape(str(e))}[/{PR}]"); _pause(); continue
        validas = {f.rstrip("=") for f, _p, _e in opcoes}
        ruins = [t for t in toks if t.startswith("--") and t not in validas]
        if ruins:
            console.print(f"  [{PR}]{'Opcao desconhecida' if is_pt else 'Unknown option'}: {escape(' '.join(ruins))}[/{PR}]")
            _pause(); continue
        return [t.strip('"') for t in toks]


def _menu_comparar(cfg: Optional[Config] = None) -> None:
    """[4.1] Comparar grupos: ANOVA sobre a unidade independente (mesma logica de `guaraci anova`)."""
    cfg = cfg or Config()
    is_pt = _lang() == "PT"
    from guaraci import anova_cli
    tab = _perguntar("comparar", cfg, 1, 5, "Tabela CSV (x = usar o exemplo)" if is_pt else "CSV table (x = use the example)",
                     "tabela.csv")
    if tab is None:
        return
    if tab.lower() == "x":
        from guaraci import exemplos as _ex
        from guaraci.cli_comandos import _rodar_anova_exemplo
        pasta = os.path.abspath(os.path.join("GUARACI_Exemplo", "comparar"))
        _ex.gerar("comparar", pasta)
        _rodar_anova_exemplo(pasta)
        _pause()
        return
    if not os.path.isfile(tab):
        console.print(f"  [{PR}]{'Arquivo nao encontrado' if is_pt else 'File not found'}: {escape(tab)}[/{PR}]")
        _pause(); return
    resp = _perguntar("comparar", cfg, 2, 5, "Coluna(s) da resposta, separadas por virgula" if is_pt
                      else "Response column(s), comma-separated", "teor")
    grupo = _perguntar("comparar", cfg, 3, 5, "Coluna do grupo" if is_pt else "Group column", "origem") if resp else None
    unidade = _perguntar("comparar", cfg, 4, 5, "Coluna da unidade independente (amostra fisica, lote ou sessao)" if is_pt
                         else "Independent unit column (physical sample, batch or session)", "amostra") if grupo else None
    if not (resp and grupo):
        return
    metodo = _perguntar("comparar", cfg, 5, 5, "Metodo: games-howell (padrao), auto, tukey, kruskal-dunn ou recomendar" if is_pt
                        else "Method: games-howell (default), auto, tukey, kruskal-dunn or recomendar", "games-howell") or "games-howell"
    args = [tab] + sum((["--resposta", r.strip()] for r in resp.split(",") if r.strip()), []) + ["--grupo", grupo]
    args += (["--unidade", unidade] if unidade else []) + ["--metodo", metodo.strip().lower()]
    args += (["--idioma", "en"] if not is_pt else [])
    args += ["--saida", os.path.join(os.path.dirname(os.path.abspath(tab)), _nome_pasta("resultados_anova"))]
    extras = _opcoes_extras("comparar", cfg, [
        ("--bloco=", "coluna de bloco (ex.: dia de analise)", "block column (e.g. analysis day)"),
        ("--ajuste=", "correcao multipla: holm (padrao) ou bh", "multiple-testing correction: holm (default) or bh"),
        ("--alpha=", "nivel de significancia (padrao 0.05)", "significance level (default 0.05)"),
        ("--sem-bootstrap", "nao calcula intervalos bootstrap", "skip bootstrap intervals"),
        ("--sem-figuras", "nao gera figuras", "do not draw figures")])
    if extras is None:
        return
    anova_cli.executar(args + extras)
    _pause()


def _menu_relatorios(cfg: Optional[Config] = None) -> None:
    """[5.2] Relatorios de uma execucao (mesmos geradores de `guaraci relatorio`)."""
    cfg = cfg or Config()
    is_pt = _lang() == "PT"
    from guaraci import relatorio_cli
    ultima = str(getattr(cfg, "output_folder", "") or "")
    pasta = _perguntar("relatorios", cfg, 1, 2, "Pasta da execucao" if is_pt else "Run folder",
                       "resultados/.../PLSDA_OE_...", ultima if os.path.isdir(ultima) else "")
    if not pasta:
        return
    fmt = _perguntar("relatorios", cfg, 2, 2, "Formatos (pdf, docx, xlsx, tex, pptx ou todos)" if is_pt
                     else "Formats (pdf, docx, xlsx, tex, pptx or all)", "pdf,xlsx", "pdf")
    if fmt is None:
        return
    extras = _opcoes_extras("relatorios", cfg, [
        ("--nome=", "titulo do relatorio", "report title"),
        ("--autor=", "autor", "author"),
        ("--instituicao=", "instituicao", "institution"),
        ("--tipo=", "tipo do documento (ex.: laudo, artigo)", "document type (e.g. report, paper)")])
    if extras is None:
        return
    relatorio_cli.executar([pasta, "--formato", fmt] + extras)
    _pause()


def _menu_fluxo_planejamento(cfg: Optional[Config] = None) -> None:
    """[2.1] Planejar -> selecionar -> auditar: as tres telas em sequencia, com a opcao de parar
    entre elas (a coleta acontece entre o passo 1 e o 2)."""
    cfg = cfg or Config()
    is_pt = _lang() == "PT"
    passos = [("planejamento", _menu_plan, "Planejar a coleta" if is_pt else "Plan the collection"),
              ("selecao_amostras", _menu_selecao_amostras, "Escolher as amostras de calibracao" if is_pt
               else "Choose the calibration samples"),
              ("auditoria", _menu_audit, "Auditar o delineamento" if is_pt else "Audit the design")]
    for i, (tid, fn, nome) in enumerate(passos, 1):
        _cabecalho_tela("fluxo_planejamento", cfg)
        _rodape_tela()
        for j, (_t2, _f2, n2) in enumerate(passos, 1):
            marca = "->" if j == i else ("ok" if j < i else "  ")
            console.print(f"  [{PA if j == i else PM}]{marca} {j}. {escape(n2)} [{_cat.tela(_t2)['tecla']}][/{PA if j == i else PM}]")
        _definir_ajuda("fluxo_planejamento", [(str(j), n2, _cat.tela(_t2)["faz"][0 if is_pt else 1]) for j, (_t2, _f2, n2) in enumerate(passos, 1)])
        r = _ask(f"\n  [{PA}]{'[1] ou Enter fazer este passo  [2] pular  [V] voltar' if is_pt else '[1] or Enter do this step  [2] skip  [V] back'}:[/{PA}] ").strip().lower()
        if r == "2":
            continue
        if r and r != "1" and _tecla_comum(r, "fluxo_planejamento", cfg) == "voltar":
            return
        _definir_ajuda(tid)      # "?" dentro do passo explica a tela do passo
        fn(cfg)


# ---------------------------------------------------------------------------
# PERFIS — cartoes compactos (2 por linha)
# ---------------------------------------------------------------------------
def _menu_exemplo(cfg: Config) -> None:
    """[1.1] Exemplo guiado: gera dados sinteticos e a configuracao de um cenario
    (mesmo motor de `guaraci exemplo`) e, se quiser, carrega na sessao."""
    from guaraci import exemplos as _ex
    L = 1 if _lang() == "EN" else 0
    ids = _ex.listar()
    while True:
        _cabecalho_tela("exemplo", cfg)
        t = Table(box=None, show_header=False, padding=(0, 1))
        t.add_column("N", style=PA, width=4, no_wrap=True)
        t.add_column("Cenario", style=PW, no_wrap=True, width=40)
        t.add_column("O que faz", style=PM)
        for i, k in enumerate(ids, 1):
            t.add_row(f"[{i}]", escape(_ex.EXEMPLOS[k]["nome"][L]), escape(_trunc(_ex.EXEMPLOS[k]["faz"][L], 60)))
        console.print(Panel(t, title=f"[bold {PA}]{_cat.tela('exemplo')['nome'][L]}[/bold {PA}]",
                            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)))
        _rodape_tela()
        _definir_ajuda("exemplo", [(str(i), _ex.EXEMPLOS[k]["nome"][L], _ex.EXEMPLOS[k]["faz"][L]) for i, k in enumerate(ids, 1)])
        raw = _input(f"\n  [1-{len(ids)} {'ou o nome' if L == 0 else 'or the name'}]: ").strip()
        r = _tecla_comum(raw, "exemplo", cfg) if raw not in ids else None
        if r == "voltar":
            return
        if r == "tratada":
            continue
        if raw in ids:
            cen = raw
        elif raw.isdigit() and 1 <= int(raw) <= len(ids):
            cen = ids[int(raw) - 1]
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]")
            _pause()
            continue
        pasta_padrao = os.path.join("GUARACI_Exemplo", cen)
        pasta = _input(f"  {'Pasta de destino (V volta)' if L == 0 else 'Destination folder (V goes back)'} [{pasta_padrao}]: ").strip() or pasta_padrao
        if pasta.upper() in ("V", "0"):      # volta para a lista de cenarios, nao cria pasta "V"
            continue
        res = _ex.gerar(cen, pasta)
        console.print(f"  [g]✓ {escape(os.path.abspath(res['pasta']))}[/g]")
        for a in res["arquivos"]:
            console.print(f"    {escape(a)}")
        from guaraci import cli_comandos as _cc
        if cen in _cc.EXECUTORES_SEM_CONFIG:
            console.print(f"  [{PM}]cd {escape(res['pasta'])} && {escape(res['comando'])}[/{PM}]")
            if _input(f"  {'Rodar agora (r) ou nao (n)?' if L == 0 else 'Run now (r) or no (n)?'} ").strip().lower() == "r":
                _cc.EXECUTORES_SEM_CONFIG[cen](res["pasta"])
        else:
            cam = os.path.join(res["pasta"], "config.yaml")
            console.print(f"  [{PM}]guaraci run {escape(cam)}[/{PM}]")
            carregar = _input(f"  {'Carregar na sessao (s), carregar e rodar agora (r) ou nao (n)?' if L == 0 else 'Load into the session (y), load and run now (r) or no (n)?'} ").strip().lower()
            if carregar in ("s", "y", "sim", "yes", "r"):
                try:
                    novo = _load_config(cam)
                    for k, v in vars(novo).items():
                        try:
                            setattr(cfg, k, v)
                        except (AttributeError, TypeError):
                            pass
                    console.print(f"  [g]✓ {'Carregado. Use [R] para rodar.' if L == 0 else 'Loaded. Use [R] to run.'}[/g]")
                    if carregar == "r":
                        _rodar_pipeline(cfg)
                        return
                except (RuntimeError, FileNotFoundError, ValueError) as e:
                    console.print(f"  [{PR}]{escape(str(e))}[/{PR}]")
        _pause()


def aplicar_perfil(cfg: Config, objetivo: str, rigor: str) -> List[Tuple[str, Any, Any]]:
    """Aplica o perfil (objetivo, depois rigor) do catalogo em `cfg` e devolve
    [(chave, antes, depois)] do que mudou -- o mesmo resumo que a tela mostra."""
    mudancas: List[Tuple[str, Any, Any]] = []
    chaves = dict(_cat.PERFIS_OBJETIVO[objetivo]["cfg"])
    chaves.update(_cat.PERFIS_RIGOR[rigor]["cfg"])
    for k, v in chaves.items():
        sp = _SPEC_BY_KEY[k]
        antes = getattr(cfg, sp["attr"])
        novo = _coagir_valor(sp, v)
        if antes != novo:
            setattr(cfg, sp["attr"], novo)
            mudancas.append((k, antes, novo))
    return mudancas


def _menu_profiles(cfg: Config) -> None:
    """[1.2] Perfis: Passo 1 objetivo, Passo 2 rigor, resumo, aplicar; rodar so' se pedir."""
    L = 1 if _lang() == "EN" else 0
    is_pt = L == 0
    obj_ids, rig_ids = list(_cat.PERFIS_OBJETIVO), list(_cat.PERFIS_RIGOR)

    def _escolher(passo: int, titulo: str, itens: List[Tuple[str, str, str]]) -> Optional[str]:
        while True:
            _cabecalho_tela("perfis", cfg)
            t = Table(box=None, show_header=False, padding=(0, 1))
            t.add_column("N", style=PA, width=4, no_wrap=True)
            t.add_column("Nome", width=40, no_wrap=True)
            t.add_column("Faz", style=PM)
            for n, (_k, nome, faz) in enumerate(itens, 1):
                t.add_row(f"[{n}]", escape(nome), escape(_trunc(faz, 70)))
            console.print(Panel(t, title=f"[bold {PA}]{'Passo' if is_pt else 'Step'} {passo} {'de' if is_pt else 'of'} 2: "
                                         f"{escape(titulo)}[/bold {PA}]", border_style=PA, box=rbox.ROUNDED, padding=(0, 1)))
            n_salvar, n_carregar = len(itens) + 1, len(itens) + 2     # acoes da tela: numeros, depois dos itens
            console.print(Text.from_markup(f"  [{PA}][{n_salvar}][/{PA}] {'gravar config atual' if is_pt else 'save current config'}   "
                                           f"[{PA}][{n_carregar}][/{PA}] {'carregar salva' if is_pt else 'load saved'}"))
            _rodape_tela()
            _definir_ajuda("perfis", [(str(n), nome, faz) for n, (_k, nome, faz) in enumerate(itens, 1)] + [
                (str(n_salvar), "Gravar config atual" if is_pt else "Save current config",
                 "Grava a configuracao desta sessao num YAML para reusar depois." if is_pt
                 else "Saves this session's configuration to a YAML file to reuse later."),
                (str(n_carregar), "Carregar salva" if is_pt else "Load saved",
                 "Le uma configuracao gravada antes e aplica nesta sessao." if is_pt
                 else "Reads a previously saved configuration and applies it to this session.")])
            raw = _input(f"\n  [1-{n_carregar}]: ").strip()
            if raw == str(n_salvar):
                _salvar_yaml(cfg); continue
            if raw == str(n_carregar):
                _carregar_yaml(cfg); continue
            r = _tecla_comum(raw, "perfis", cfg)
            if r == "voltar":
                return None
            if r == "tratada":
                continue
            if raw.isdigit() and 1 <= int(raw) <= len(itens):
                return itens[int(raw) - 1][0]
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()

    # V volta UM passo: no Passo 2 volta ao Passo 1; so' no Passo 1 sai da tela.
    while True:
        obj = _escolher(1, "o que voce quer responder" if is_pt else "what you want to answer",
                        [(k, _cat.PERFIS_OBJETIVO[k]["nome"][L], _cat.PERFIS_OBJETIVO[k]["faz"][L]) for k in obj_ids])
        if obj is None:
            return
        po = _cat.PERFIS_OBJETIVO[obj]
        if po.get("comando"):                      # transferir: e' um comando, nao uma configuracao
            console.print(Panel(f"{escape(po['faz'][L])}\n\n  {escape(po['comando'])}\n  "
                                f"{'(exemplo: guaraci exemplo transferir --rodar)' if is_pt else '(example: guaraci exemplo transferir --rodar)'}",
                                border_style=PA, box=rbox.ROUNDED, padding=(1, 2), width=_W()))
            _pause()
            continue
        rig = _escolher(2, "quanto rigor" if is_pt else "how much rigour",
                        [(k, _cat.PERFIS_RIGOR[k]["nome"][L], _cat.PERFIS_RIGOR[k]["faz"][L]) for k in rig_ids])
        if rig is not None:
            break
    previa = copy.deepcopy(cfg)
    mud = aplicar_perfil(previa, obj, rig)
    _cabecalho_tela("perfis", cfg)
    console.print(f"  [{PW}]{escape(po['nome'][L])}  +  {escape(_cat.PERFIS_RIGOR[rig]['nome'][L])}[/{PW}]")
    seg = _cat.TEMPO_PERFIL.get((obj, rig))
    if seg is not None:
        console.print(f"  [{PM}]{'Tempo medido no exemplo sintetico' if is_pt else 'Time measured on the synthetic example'} "
                      f"'{po['exemplo']}': {seg:.0f} s ({escape(_cat.TEMPO_PERFIL_MEDIDO_EM)}). "
                      f"{'Com seus dados o tempo cresce com o numero de amostras e de variaveis.' if is_pt else 'With your data the time grows with the number of samples and variables.'}[/{PM}]")
    for k, antes, depois in mud:
        console.print(f"    {escape(_nome_campo(k))}: {escape(_fmt_yaml(antes))} -> [{PA}]{escape(_fmt_yaml(depois))}[/{PA}]")
    if not mud:
        console.print(f"  [{PM}]{'Nada muda: a configuracao atual ja e esta.' if is_pt else 'Nothing changes: the current configuration is already this one.'}[/{PM}]")
    if _input(f"  {'Aplicar? [S/n]' if is_pt else 'Apply? [Y/n]'} ").strip().lower() in ("n", "nao", "no"):
        return
    aplicar_perfil(cfg, obj, rig)
    ajustados = _ajustar_toggles_por_nivel(cfg)
    console.print(f"  [g]✓ {'Perfil aplicado' if is_pt else 'Profile applied'} ({len(mud)} {'campos' if is_pt else 'fields'})[/g]")
    if ajustados:
        console.print(f"  [{PM}]{_t('ajuste_nivel', campos=', '.join(_nome_campo(k) for k in ajustados))}[/{PM}]")
    # rodar so' com confirmacao explicita: um Enter distraido nao dispara uma execucao longa
    if _input("  Rodar agora com essa configuracao? [s/N]: " if is_pt else "  Run now with this configuration? [y/N]: "
              ).strip().lower() in ("s", "y", "sim", "yes"):
        _rodar_pipeline(cfg)
    else:
        _pause()


# ---------------------------------------------------------------------------
# SOBRE — identidade, citacao e referencias (para publicacao)
# ---------------------------------------------------------------------------
def _referencias_tecnicas(lang: str) -> None:
    """Sobre > Referencias: todas as tecnicas, por tema (guaraci.catalogo_referencias).

    Numero abre um tema; T mostra todos; Enter ou V volta."""
    from guaraci import catalogo_referencias as _cr

    def _tema(tid: str, nome: str, itens) -> None:
        t = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        t.add_column("Metodo", style=f"bold {PA}", width=22, overflow="fold")
        t.add_column("Ref", style=PW, overflow="fold")
        for it in itens:
            linhas = Text()
            for n, k in enumerate(it.refs):
                if n:
                    linhas.append("\n")
                linhas.append(_cr.citacao(k), style=PW)
            if it.nota:
                if it.refs:
                    linhas.append("\n")
                linhas.append(it.nota, style=PM if (it.refs or it.proprio) else PR)
            t.add_row(it.metodo, linhas)
        console.print(Panel(t, title=f"[bold {PA}]{escape(nome)}[/bold {PA}]",
                            border_style=PF, box=rbox.ROUNDED, padding=(0, 1), width=_W()))

    temas = _cr.TEMAS
    while True:
        tit = "Referencias por tecnica" if lang == "PT" else "References by technique"
        t = Table(show_header=False, box=None, padding=(0, 1))
        t.add_column("N", style=PA, width=4, justify="right", no_wrap=True)
        t.add_column("Tema", style=PW)
        t.add_column("Qtd", style=PM, no_wrap=True)
        for i, (_tid, npt, nen, itens) in enumerate(temas, 1):
            n_ref = len({k for it in itens for k in it.refs})
            qtd = f"{n_ref} ref."
            t.add_row(str(i), npt if lang == "PT" else nen, qtd)
        total = len(_cr.REFS)
        sub = (f"{total} referencias, todas no registro conferido (docs/cientifico)."
               if lang == "PT" else
               f"{total} references, all in the verified registry (docs/cientifico).")
        console.print(Panel(t, title=f"[bold {PA}]{tit}[/bold {PA}]",
                            subtitle=f"[{PM}]{sub}[/{PM}]",
                            border_style=PF, box=rbox.ROUNDED, padding=(0, 1), width=_W()))
        dica = ("numero abre o tema · T todos · Enter volta" if lang == "PT"
                else "number opens the topic · T all · Enter goes back")
        raw = _input(f"  [{dica}]: ").strip().upper()
        if raw in ("", "V", "Q"):
            return
        if raw == "T":
            for _tid, npt, nen, itens in temas:
                _tema(_tid, npt if lang == "PT" else nen, itens)
            _pause()
        elif raw.isdigit() and 1 <= int(raw) <= len(temas):
            _tid, npt, nen, itens = temas[int(raw) - 1]
            _tema(_tid, npt if lang == "PT" else nen, itens)
            _pause()
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]")


def _menu_about(cfg: Optional[Config] = None) -> None:
    """Secao Sobre — proposito, autor, citacao em multiplos formatos e referencias."""
    # Dados fixos do projeto e do autor
    _AUTOR_NOME    = AUTOR_NOME
    _AUTOR_ORCID   = AUTOR_ORCID
    _AUTOR_TAG     = "Pesquisador / Desenvolvedor"
    _AUTOR_GITHUB  = AUTOR_GITHUB_URL
    _REPO          = REPO_URL
    _VERSAO        = pq.__version__
    _ANO           = "2026"
    _TITULO_CURTO  = "GUARACI"
    _LIC           = LICENCA_SPDX

    def _titulo(lang: str) -> str:
        return ("Plataforma quimiometrica com validacao anti-vazamento por padrao"
                if lang == "PT" else
                "Chemometrics platform with leakage-safe validation by default")

    def _painel_identidade(lang: str) -> None:
        """Painel principal: nome, proposito e links rapidos."""
        t = Table(box=None, show_header=False, padding=(0, 1), expand=True)
        t.add_column("L", style=PA, width=14, no_wrap=True)
        t.add_column("V", style=PW, overflow="fold")

        # Titulo
        t.add_row(
            "",
            Text.from_markup(f"[bold {PA}]{_TITULO_CURTO}[/bold {PA}]"
                             f"[{PW}] — {escape(_titulo(lang))}[/{PW}]"),
        )
        t.add_row("", Text(""))

        # Proposito
        if lang == "PT":
            p1 = ("Democratizar o acesso a analises quimiometricas de alta qualidade"
                  " para pesquisadores que nao dominam programacao.")
            p2 = ("Oferece um ambiente confiavel, reproducivel e bilingue (PT/EN)"
                  " para classificacao, autenticacao e exploracao de matrizes complexas"
                  " — do FT-NIR ao GC-MS, sem escrever uma linha de codigo.")
            p3 = ("Metodologia geral, aplicavel a"
                  " qualquer tecnica analitica com dados multivariados.")
        else:
            p1 = ("Democratize access to high-quality chemometric analyses"
                  " for researchers without a programming background.")
            p2 = ("Provides a reliable, reproducible and bilingual (PT/EN)"
                  " environment for classification, authentication and exploration"
                  " of complex matrices — from FT-NIR to GC-MS, without writing code.")
            p3 = ("General methodology, applicable to any"
                  " analytical technique with multivariate data.")
        prop_lbl = "Proposito" if lang == "PT" else "Purpose"
        t.add_row(f"{prop_lbl}:", Text(p1, style=PW))
        t.add_row("", Text(p2, style=PM))
        t.add_row("", Text(p3, style=PM))
        t.add_row("", Text(""))

        # Tecnicas suportadas
        tec_lbl = "Tecnicas" if lang == "PT" else "Techniques"
        # lido do catalogo: tecnica nova aparece aqui sem editar esta tela
        _i = 0 if lang == "PT" else 1
        tec_val = " · ".join(
            d["nome"][_i].split(" (")[0] for d in _cat.INSTRUMENTOS.values())
        t.add_row(f"{tec_lbl}:", Text(tec_val, style=PW))
        t.add_row("", Text(""))

        # Metadados
        ver_lbl = "Versao" if lang == "PT" else "Version"
        lic_lbl = "Licenca" if lang == "PT" else "License"
        t.add_row(f"{ver_lbl}:", Text(f"{_VERSAO}  ({_ANO})", style=PS))
        t.add_row(f"{lic_lbl}:", Text(_LIC, style=PS))
        t.add_row("Repo:", Text(_REPO, style=PS))

        titulo_p = "Sobre" if lang == "PT" else "About"
        corpo_id = t
        # Arte do sol ao lado do texto, so' com painel largo (o texto precisa
        # de ~58 colunas alem da arte) e nunca no modo linear.
        if (_arte.pode_mostrar(console.width, modo_linear())
                and _W() >= _arte.LARGURA + 58):
            g = Table.grid(padding=(0, 2), expand=True)
            g.add_column(width=_arte.LARGURA, no_wrap=True, vertical="middle")
            g.add_column(ratio=1)
            g.add_row(Text(_arte.SOL_BRAILLE, style=PA), t)
            corpo_id = g
        console.print(Panel(corpo_id,
            title=f"[bold {PA}]{titulo_p}[/bold {PA}]",
            border_style=PA, box=rbox.ROUNDED, padding=(1, 2), width=_W()))

    def _painel_autor(lang: str) -> None:
        """Painel do autor com ORCID, GitHub, repositorio e Discussions."""
        t = Table(box=None, show_header=False, padding=(0, 1), expand=True)
        t.add_column("L", style=PA, width=14, no_wrap=True)
        t.add_column("V", style=PW, overflow="fold")

        nome_lbl  = "Nome" if lang == "PT" else "Name"
        cargo_lbl = "Cargo" if lang == "PT" else "Role"
        t.add_row(f"{nome_lbl}:",  Text(_AUTOR_NOME,   style=f"bold {PW}"))
        t.add_row(f"{cargo_lbl}:", Text(_AUTOR_TAG if lang == "PT" else "Researcher / Developer", style=PW))
        t.add_row("", Text(""))
        t.add_row("GitHub:",  Text(_AUTOR_GITHUB, style=PS))
        t.add_row("ORCID:",   Text(f"https://orcid.org/{_AUTOR_ORCID}", style=PS))
        t.add_row("", Text(""))
        t.add_row("Projeto:" if lang == "PT" else "Project:", Text(_REPO, style=PS))
        t.add_row("Discussions:", Text(DISCUSSIONS_URL, style=PS))

        titulo_a = "Autor" if lang == "PT" else "Author"
        console.print(Panel(t,
            title=f"[bold {PA}]{titulo_a}[/bold {PA}]",
            border_style=PF, box=rbox.ROUNDED, padding=(1, 2), width=_W()))
        _pause()

    def _painel_citar(lang: str) -> None:
        """Formatos de citacao: APA, ABNT, BibTeX."""
        tit_full = f"{_TITULO_CURTO}: {_titulo(lang)}"
        autor_abnt = "COSTA, Erley Silva da"

        # APA
        apa = (f"Costa, Erley Silva da. ({_ANO}). {tit_full} (v{_VERSAO})"
               f" [Software]. {_REPO}")
        # ABNT (NBR 6023:2018 — software)
        abnt = (f"{autor_abnt}. {_TITULO_CURTO}: {_titulo('PT')}."
                f" Versao {_VERSAO}. {_ANO}."
                f" Disponivel em: <{_REPO}>.")
        # BibTeX
        bibtex = (
            f"@software{{guaraci_{_ANO},\n"
            f"  author    = {{Costa, Erley Silva da}},\n"
            f"  title     = {{{{{_TITULO_CURTO}: {_titulo(lang)}}}}},\n"
            f"  version   = {{{_VERSAO}}},\n"
            f"  year      = {{{_ANO}}},\n"
            f"  url       = {{{_REPO}}},\n"
            f"  license   = {{{_LIC}}}\n"
            f"}}"
        )

        corpo = Text()
        corpo.append("APA\n", style=f"bold {PA}")
        corpo.append(apa + "\n\n", style=PW)
        corpo.append("ABNT (NBR 6023:2018)\n", style=f"bold {PA}")
        corpo.append(abnt + "\n\n", style=PW)
        corpo.append("BibTeX\n", style=f"bold {PA}")
        corpo.append(bibtex + "\n\n", style=PS)
        nota = ("Detalhes completos em CITATION.cff (raiz do projeto)."
                if lang == "PT" else
                "Full details in CITATION.cff (project root).")
        corpo.append(nota, style=PM)

        titulo_c = "Como Citar" if lang == "PT" else "How to Cite"
        console.print(Panel(corpo,
            title=f"[bold {PA}]{titulo_c}[/bold {PA}]",
            border_style=PF, box=rbox.ROUNDED, padding=(1, 2), width=_W()))
        _pause()

    def _painel_diferenciais(lang: str) -> None:
        """Por que o GUARACI: o nome, o metodo e o comparativo honesto."""
        pt = lang == "PT"
        # -- pagina 1: o nome e o metodo
        corpo = Text()
        corpo.append(("O nome\n" if pt else "The name\n"), style=f"bold {PA}")
        corpo.append((
            "Guaraci vem do tupi kûarasy, \"sol\". A espectroscopia mede o que\n"
            "a luz revela sobre a materia: cada espectro e' a resposta da amostra\n"
            "a uma faixa de luz. O GUARACI e' a luz sobre os dados -- e e' um\n"
            "nome brasileiro para um software feito no Brasil, em portugues primeiro.\n\n"
            if pt else
            "Guaraci comes from Tupi kûarasy, \"sun\". Spectroscopy measures what\n"
            "light reveals about matter: each spectrum is the sample's answer to a\n"
            "band of light. GUARACI is light shed on the data -- and a Brazilian\n"
            "name for software made in Brazil, Portuguese first.\n\n"), style=PW)
        corpo.append(("O metodo\n" if pt else "The method\n"), style=f"bold {PA}")
        passos = ([
            ("1 Auditar", "antes de modelar, procura duplicatas, lote confundido com classe e grupos pequenos demais."),
            ("2 Preparar", "pre-processamento escolhido por tecnica (SNV, derivadas, MSC...) e ajustado so' no treino."),
            ("3 Validar", "validacao cruzada por grupo (amostra, lote, replica): replicas nunca ficam dos dois lados."),
            ("4 Testar", "permutacao, CV-ANOVA e IC bootstrap BCa dizem se o modelo e' melhor que o acaso."),
            ("5 Explicar", "VIP, cargas e selos de confianca mostram o que pesou e o quanto o metodo foi conferido."),
            ("6 Registrar", "config.yaml, semente, versao e relatorio permitem refazer a analise identica."),
        ] if pt else [
            ("1 Audit", "before modelling, looks for duplicates, batch confounded with class, groups too small."),
            ("2 Prepare", "preprocessing chosen per technique (SNV, derivatives, MSC...) fitted on training data only."),
            ("3 Validate", "group-aware cross-validation (sample, batch, replicate): replicates never split."),
            ("4 Test", "permutation, CV-ANOVA and BCa bootstrap intervals show whether it beats chance."),
            ("5 Explain", "VIP, loadings and trust badges show what mattered and how far the method was checked."),
            ("6 Record", "config.yaml, seed, version and report let anyone rerun the identical analysis."),
        ])
        tp = Table(box=None, show_header=False, padding=(0, 1), expand=True)
        tp.add_column("P", style=PA, width=12, no_wrap=True)
        tp.add_column("D", style=PW, overflow="fold")
        for a, b in passos:
            tp.add_row(a, b)
        tit_d = ("Por que o GUARACI?" if pt else "Why GUARACI?")
        console.print(Panel(Group(corpo, tp),
            title=f"[bold {PA}]{tit_d}[/bold {PA}]",
            subtitle=f"[{PM}]1/2[/{PM}]",
            border_style=PF, box=rbox.ROUNDED, padding=(1, 2), width=_W()))
        _pause()

        # -- pagina 2: comparativo
        t = Table(show_header=True, header_style=f"bold {PA}", box=rbox.SIMPLE,
                  padding=(0, 1), expand=True)
        t.add_column("Criterio" if pt else "Criterion", style=PW, ratio=3)
        t.add_column("GUARACI", style=PG, justify="center", ratio=1)
        t.add_column("Pagos*" if pt else "Paid*", style=PM, justify="center", ratio=1)
        t.add_column("Livres**" if pt else "Free**", style=PM, justify="center", ratio=1)
        if pt:
            linhas = [
                ("Custo", "Gratuito", "Licenca", "Gratuito"),
                ("Codigo aberto e auditavel", "Sim (GPL)", "Nao", "Sim"),
                ("Usar sem programar", "Sim", "Sim", "Nao"),
                ("Validacao por grupo/lote", "Padrao", "Opcional", "Programar"),
                ("Auditoria do delineamento", "Automatica", "Manual", "Programar"),
                ("Testes contra o acaso", "Padrao", "Sim", "Programar"),
                ("Autenticacao conformal", "Sim", "Raro", "Programar"),
                ("Selo de confianca por metodo", "Sim", "Nao", "Nao"),
                ("Referencia por metodo na tela", "Sim", "Manual", "Manual"),
                ("Reproduzir a analise identica", "Sim", "Parcial", "Depende"),
                ("Portugues e ingles", "Sim", "Raro", "Raro"),
                ("Tecnicas / leitores", "14 / 12", "Muitos", "Poucos"),
            ]
            nota = ("* MATLAB/PLS_Toolbox, The Unscrambler, SIMCA e similares.\n"
                    "** Pacotes em R/Python (mdatools, ropls, scikit-learn...).\n"
                    "Onde os pagos ainda vencem: suporte comercial, conformidade\n"
                    "regulatoria (21 CFR Part 11) e anos de uso em industria.")
            intro = ("O que pesa: o GUARACI recusa por padrao os erros que mais inflam\n"
                     "resultados em quimiometria -- replica vazando entre treino e\n"
                     "teste, lote confundido com classe e modelo sem teste contra o acaso.")
        else:
            linhas = [
                ("Cost", "Free", "License", "Free"),
                ("Open, auditable source", "Yes (GPL)", "No", "Yes"),
                ("Usable without coding", "Yes", "Yes", "No"),
                ("Group/batch-aware validation", "Default", "Optional", "Code it"),
                ("Experimental design audit", "Automatic", "Manual", "Code it"),
                ("Tests against chance", "Default", "Yes", "Code it"),
                ("Conformal authentication", "Yes", "Rare", "Code it"),
                ("Trust badge per method", "Yes", "No", "No"),
                ("Reference per method on screen", "Yes", "Manual", "Manual"),
                ("Rerun the identical analysis", "Yes", "Partial", "Depends"),
                ("Portuguese and English", "Yes", "Rare", "Rare"),
                ("Techniques / readers", "14 / 12", "Many", "Few"),
            ]
            nota = ("* MATLAB/PLS_Toolbox, The Unscrambler, SIMCA and similar.\n"
                    "** R/Python packages (mdatools, ropls, scikit-learn...).\n"
                    "Where paid tools still win: commercial support, regulatory\n"
                    "compliance (21 CFR Part 11) and years of industrial use.")
            intro = ("What matters: by default GUARACI refuses the mistakes that most\n"
                     "inflate chemometric results -- replicates leaking between train\n"
                     "and test, batch confounded with class, no test against chance.")
        for linha in linhas:
            t.add_row(*linha)
        console.print(Panel(
            Group(Text(intro, style=PW), Text(""), t, Text(""), Text(nota, style=PM)),
            title=f"[bold {PA}]{tit_d}[/bold {PA}]",
            subtitle=f"[{PM}]2/2[/{PM}]",
            border_style=PF, box=rbox.ROUNDED, padding=(1, 2), width=_W()))
        _pause()

    def _painel_referencias(lang: str) -> None:
        _referencias_tecnicas(lang)

    # Loop principal da secao Sobre
    while True:
        lang = _lang()
        _cabecalho_tela("sobre", cfg)
        _painel_identidade(lang)

        if lang == "PT":
            console.print(
                f"  [{PA}][1][/{PA}] Por que o GUARACI?  "
                f"  [{PA}][2][/{PA}] Autor / Contato\n"
                f"  [{PA}][3][/{PA}] Como Citar          "
                f"  [{PA}][4][/{PA}] Referencias\n"
                f"  [{PM}]{escape(_cat.RODAPE_TELA['PT'])}[/{PM}]"
            )
        else:
            console.print(
                f"  [{PA}][1][/{PA}] Why GUARACI?        "
                f"  [{PA}][2][/{PA}] Author / Contact\n"
                f"  [{PA}][3][/{PA}] How to Cite         "
                f"  [{PA}][4][/{PA}] References\n"
                f"  [{PM}]{escape(_cat.RODAPE_TELA['EN'])}[/{PM}]"
            )

        pt_ = lang == "PT"
        _definir_ajuda("sobre", [
            ("1", "Diferenciais" if pt_ else "Differentiators", "O que o GUARACI faz de diferente (validacao por grupo, anti-vazamento, selos)." if pt_
             else "What GUARACI does differently (group validation, anti-leakage, badges)."),
            ("2", "Autor / Contato" if pt_ else "Author / Contact", "Quem mantem o projeto e como entrar em contato." if pt_
             else "Who maintains the project and how to reach them."),
            ("3", "Como citar" if pt_ else "How to cite", "Texto e BibTeX para citar o GUARACI num trabalho." if pt_
             else "Text and BibTeX to cite GUARACI in a paper."),
            ("4", "Referencias" if pt_ else "References", "Artigos das tecnicas usadas, com DOI." if pt_
             else "Papers of the techniques used, with DOI.")])
        raw = _input(f"\n  {_t('opcao')}: ").strip().upper()
        r = _tecla_comum(raw, "sobre", cfg)
        if r == "voltar": break
        if r == "tratada": pass
        elif raw == "1": _painel_diferenciais(lang)
        elif raw == "2": _painel_autor(lang)
        elif raw == "3": _painel_citar(lang)
        elif raw == "4": _painel_referencias(lang)
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()


# ---------------------------------------------------------------------------
# AJUDA INTERATIVA
# ---------------------------------------------------------------------------
def _manual_interativo(cfg: Optional[Config] = None) -> None:
    """[6.2] 0 -- o manual empacotado (mesmo texto de `guaraci manual`): lista os topicos e abre um."""
    from guaraci import docs_pacote as dp
    is_pt = _lang() == "PT"
    while True:
        _cabecalho_tela("ajuda", cfg or Config())
        console.print(Text(dp.lista(_lang()), style=PW))
        _rodape_tela()
        alvo = _ask(f"  [{PA}]{'Topico ou secao (Enter = voltar)' if is_pt else 'Topic or section (Enter = back)'}:[/{PA}] ").strip()
        if not alvo or _tecla_comum(alvo, "ajuda", cfg or Config()) == "voltar":
            return
        a = alvo.lower()
        texto = dp.ler(a) if a in dp.TOPICOS else dp.secao_do_manual(a)
        if texto is None:
            console.print(f"  [{PM}]{'Nao encontrado.' if is_pt else 'Not found.'}[/{PM}]")
        else:
            console.print(Text(texto, style=PW))
        _pause()


def _menu_help(cfg: Optional[Config] = None) -> None:
    """Ajuda navegavel — lista todos os campos de cara; numero abre a ajuda."""
    lang = _lang()
    # Lista unificada a partir do _CONFIG_SPEC (todos os campos editaveis).
    keys = [s["key"] for s in _CONFIG_SPEC] if _CONFIG_SPEC else list(_HELP_DB.keys())
    # Remove duplicatas mantendo ordem
    keys = list(dict.fromkeys(keys))

    while True:
        _cabecalho_tela("ajuda", cfg)

        t = Table(show_header=True, header_style=PM, box=rbox.SIMPLE, padding=(0, 1))
        t.add_column("N", style=PA, width=4, no_wrap=True)
        t.add_column("Campo" if lang=="PT" else "Field", width=24, no_wrap=True)
        t.add_column("Tipo" if lang=="PT" else "Risk", width=11, no_wrap=True)
        t.add_column("Descricao" if lang=="PT" else "Description", style=PM)
        for i, key in enumerate(keys, 1):
            r_hex = _risco_hex(key)
            t.add_row(
                str(i),
                escape(_nome_campo(key)),
                Text(_risco_icon(key) + " " + (_RISK_CLASS.get(key, "—") if _lang() == "PT" else
                     {"ANALITICO": "ANALYTICAL", "AVANCADO": "ADVANCED"}.get(_RISK_CLASS.get(key, "—"), _RISK_CLASS.get(key, "—"))),
                     style=r_hex),
                escape(_desc_curta(key, 40)),
            )

        sub = ("Numero do campo = ajuda completa · 0 = manual do GUARACI · V volta · ou busque por nome."
               if lang=="PT" else
               "Field number = full help · 0 = GUARACI manual · V goes back · or search by name.")
        console.print(Panel(
            t,
            title=f"[bold {PA}]{_t('t_ajuda')}[/bold {PA}]",
            subtitle=f"[{PM}]{sub}[/{PM}]",
            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)
        ))
        _rodape_tela()

        _definir_ajuda("ajuda", [
            ("N", "Numero do campo" if lang == "PT" else "Field number",
             "Abre a ajuda completa do campo (faz, quando usar, impacto, padrao, faixa e exemplos)." if lang == "PT"
             else "Opens the field's full help (does, when to use, impact, default, range and examples)."),
            ("0", "Manual" if lang == "PT" else "Manual", "Abre o manual do GUARACI por topico ou secao." if lang == "PT"
             else "Opens the GUARACI manual by topic or section."),
            ("txt", "Busca por nome" if lang == "PT" else "Search by name", "Parte do nome do campo abre o primeiro que combinar." if lang == "PT"
             else "Part of a field name opens the first match.")])
        raw = _input(f"\n  {_t('opcao')}: ").strip()
        if raw == "0":     # antes de _tecla_comum, onde 0 = voltar
            _manual_interativo(cfg)
            continue
        r = _tecla_comum(raw, "ajuda", cfg)
        if r == "voltar":
            break
        if r == "tratada":
            lang = _lang()
        elif raw.lower().startswith("help "):
            campo = raw[5:].strip()
            found = [k for k in keys if campo.lower() in k.lower() or campo.lower() in _nome_campo(k).lower()]
            if found:
                _mostrar_ajuda(found[0])
            else:
                console.print(f"  [{PM}]{'Nao encontrado.' if lang=='PT' else 'Not found.'}[/{PM}]")
                _pause()
        elif raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(keys):
                _mostrar_ajuda(keys[idx])
            else:
                console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()
        elif raw in keys:
            _mostrar_ajuda(raw)
        else:
            found = [k for k in keys if raw.lower() in k.lower() or raw.lower() in _nome_campo(k).lower()]
            if found:
                _mostrar_ajuda(found[0])
            else:
                console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()


# ---------------------------------------------------------------------------
# CHECKLIST PRE-EXECUCAO
# ---------------------------------------------------------------------------
def _estimar_tempo(cfg: Config, n_amostras: int) -> Optional[str]:
    """Estimativa de ORDEM DE GRANDEZA do tempo de execucao, em texto.

    Existe porque o usuario apertava [R] sem nenhuma nocao de estar
    comprometendo 5 minutos ou 3 horas -- e a diferenca entre esses dois
    casos e' so' um campo de configuracao (n_jobs_permutacao).

    Calibracao (ordem de grandeza numa maquina de desenvolvimento com
    CPython 3.12 e 8 nucleos fisicos, alguns milhares de espectros x 2000
    variaveis):
        - 1 ajuste PLS com 40 LVs .......... ~3,5 s
        - carga de alguns milhares de .dx ... ~20 s
    O custo de um ajuste escala aproximadamente com (n_amostras x n_LVs),
    o que da a constante `_S_POR_AMOSTRA_LV` abaixo.

    E' ESTIMATIVA, nao promessa: ignora I/O de figuras, variacao de hardware
    e o custo dos modulos opcionais mais pesados (SHAP em especial). Por isso
    o texto devolvido usa faixa ("~15-25 min"), nunca um numero exato.
    Devolve None quando nao ha' base para estimar (n_amostras desconhecido).
    """
    if not n_amostras or n_amostras <= 0:
        return None

    # Segundos por amostra por LV, da calibracao acima.
    _S_POR_AMOSTRA_LV = 5.2e-5

    max_lvs = int(_cfgv(cfg, "max_lvs", 40) or 40)
    n_splits = int(_cfgv(cfg, "n_splits_cv", 5) or 5)
    frac_treino = 1.0 - 1.0 / max(n_splits, 2)

    # Selecao de LVs: para cada n=1..max_lvs, um ajuste por fold. O custo
    # cresce com n, entao a soma equivale a max_lvs*(max_lvs+1)/2 "LVs-ajuste".
    lvs_ajuste = max_lvs * (max_lvs + 1) / 2.0
    t_cv = lvs_ajuste * n_splits * n_amostras * frac_treino * _S_POR_AMOSTRA_LV

    # Permutacao/Wold: n_perm repeticoes de uma CV completa no LV escolhido
    # (aproximado por max_lvs/2, o valor tipico do criterio de parcimonia).
    t_perm = 0.0
    n_perm = int(_cfgv(cfg, "n_permutacoes", 0) or 0)
    if _cfgv(cfg, "teste_wold", False):
        n_perm += int(_cfgv(cfg, "n_permutations_wold", 0) or 0)
    if n_perm:
        n_jobs = max(1, int(_cfgv(cfg, "n_jobs_permutacao", 1) or 1))
        t_1perm = (max_lvs / 2.0) * n_splits * n_amostras * frac_treino * _S_POR_AMOSTRA_LV
        t_perm = n_perm * t_1perm / n_jobs

    # Modulos opcionais pesados, como multiplicadores grosseiros do custo de CV.
    t_extra = 0.0
    if _cfgv(cfg, "benchmark", False):      t_extra += 3.0 * t_cv
    if _cfgv(cfg, "monte_carlo", False):
        t_extra += int(_cfgv(cfg, "n_monte_carlo", 60) or 60) / max(n_splits, 2) * (t_cv / max_lvs)
    if _cfgv(cfg, "shap_benchmark", False): t_extra += 2.0 * t_cv
    if _cfgv(cfg, "selecao_variaveis_etapa4", False): t_extra += 2.0 * t_cv
    if _cfgv(cfg, "selecao_ag", False):     t_extra += 10.0 * t_cv
    if _cfgv(cfg, "selecao_spa", False):    t_extra += 4.0 * t_cv
    if _cfgv(cfg, "selecao_cars", False):   t_extra += 8.0 * t_cv
    if _cfgv(cfg, "selecao_uve", False):    t_extra += 2.0 * t_cv

    total_s = t_cv + t_perm + t_extra
    if total_s < 90:
        return "< 2 min"
    minutos = total_s / 60.0
    # Faixa de +-40%: a incerteza real e' dessa ordem (hardware, figuras, I/O).
    lo, hi = max(1, int(minutos * 0.6)), int(minutos * 1.4) + 1
    if hi < 60:
        return f"~{lo}-{hi} min"
    return f"~{lo / 60.0:.1f}-{hi / 60.0:.1f} h"


def _checklist(cfg: Config) -> Tuple[bool, List[str], List[Tuple[Optional[bool], str]]]:
    lang = _lang()
    checks = []; erros = []

    pasta = _cfgv(cfg, "pasta_dados", "dados")
    modo_chk = _cfgv(cfg, "modo_entrada", "dx")
    pasta_ok = bool(pasta) and os.path.isdir(str(pasta))
    n_dx = _count_dx(pasta) if (pasta_ok and modo_chk == "dx") else 0

    n_para_estimar = n_dx
    if modo_chk == "hsi":
        # executar() desvia o modo hsi para o fluxo da tela [3.6] (desde 2026-10-03); o [R]
        # so' precisa da pasta de cubos ENVI (antes bloqueava sempre -- achado B1, 2026-09-24).
        pasta_hsi = str(_cfgv(cfg, "hsi_pasta_dataset", "") or "")
        if os.path.isdir(pasta_hsi):
            checks.append((True, f"Cubos HSI: {pasta_hsi}" if lang == "PT" else f"HSI cubes: {pasta_hsi}"))
        else:
            checks.append((False, "Pasta de cubos HSI nao encontrada (campo hsi_pasta_dataset ou tela [3.6])"
                           if lang == "PT" else "HSI cube folder not found (hsi_pasta_dataset field or [3.6] screen)"))
            erros.append("hsi_pasta_dataset")
    elif modo_chk in ("csv", "sintetico"):
        # Achado do roteiro de teste externo (2026-09-23): este bloco so'
        # conhecia a pasta de arquivos .dx -- em modo csv um usuario novo
        # (sem a pasta padrao `dados`) via "Pasta de dados nao encontrada" e
        # a execucao era BLOQUEADA por um requisito que o modo csv nem usa.
        # csv valida o arquivo mais abaixo; sintetico nao usa pasta_dados.
        if modo_chk == "csv":
            arq_csv = str(_cfgv(cfg, "arquivo_csv", "") or "")
            if arq_csv and os.path.isfile(arq_csv):
                try:
                    with open(arq_csv, "rb") as _f:
                        n_para_estimar = max(0, sum(1 for _ in _f) - 1)
                except OSError:
                    n_para_estimar = 0
    elif modo_chk != "dx":
        # Demais modos que leem uma pasta (imagem, opus, spc, sp, rmn, hplc,
        # gcms, eem): mesma validacao por-modo do app web/`_menu_audit`.
        ok_p, msg_p = pq.validar_fonte_de_dados(cfg)
        checks.append((True if ok_p else False, msg_p))
        if not ok_p:
            erros.append("pasta_dados")
    elif pasta_ok and n_dx > 0:
        checks.append((True,  _t("chk_dados") + f" ({n_dx} .dx)"))
        # Varredura barata dos cabecalhos (decimos de segundo): antecipa
        # os dois efeitos que MUDAM O N da analise e que antes so' apareciam
        # no meio do log, depois de o usuario ja' ter iniciado a rodada.
        # (neste ramo `modo_chk` ja e "dx": `n_dx` so conta arquivos nesse modo)
        try:
            from guaraci.dados_io import prescan_dx
            pre = prescan_dx(str(pasta))
        except (OSError, ValueError) as _e_pre:  # diagnostico e' best-effort
            checks.append((None, _t("chk_prescan_erro", erro=_e_pre)))
        else:
            n_para_estimar = pre["n_apos_descarte"] or n_dx
            if pre["n_fora_da_faixa"]:
                detalhe = ", ".join(
                    f"{esp} {n}" for esp, n in
                    sorted(pre["fora_por_especie"].items(), key=lambda kv: -kv[1]))
                checks.append((None, _t("chk_descarte", n=pre["n_fora_da_faixa"])
                                      + f" — {detalhe}"))
            if pre["n_sem_mae_id"]:
                checks.append((None, _t("chk_orfaos", n=pre["n_sem_mae_id"])))
            if pre["n_grupos"]:
                checks.append((True, _t("chk_grupos", n=pre["n_grupos"])))
    elif pasta_ok:
        checks.append((None,  _t("chk_dados") + " (0 .dx)"))
        erros.append("pasta_dados vazia")
    else:
        checks.append((False, _t("chk_err_dados")))
        erros.append("pasta_dados")

    mode = _cfgv(cfg, "modo_entrada", "dx")
    if mode == "csv":
        arq = _cfgv(cfg, "arquivo_csv", "")
        if arq and os.path.isfile(str(arq)):
            checks.append((True, _t("chk_csv")))
        else:
            checks.append((False, _t("chk_err_csv")))
            erros.append("arquivo_csv")
    else:
        checks.append((True, f"{_t('chk_modo')}: {mode}"))

    ga = _cfgv(cfg, "validacao_group_aware", True)
    if not ga:
        checks.append((False, _t("chk_err_leak")))
        erros.append("validacao_group_aware")
    else:
        # A flag e' intencao; a GARANTIA e' o que os dados permitem (B4b do
        # parecer de 2026-09-25): nunca "anti-leakage ativo" sem chave de grupo.
        from guaraci.dados_io import garantia_agrupamento_prevista
        nivel_g = garantia_agrupamento_prevista(cfg)
        if nivel_g == "high":
            checks.append((True, _t("chk_leak_high")))
        elif nivel_g == "medium":
            checks.append((None, _t("chk_leak_medium")))
        elif nivel_g == "none":
            checks.append((False, _t("chk_leak_none")))
        else:
            checks.append((None, _t("chk_leak_verificar")))

    ps = _cfgv(cfg, "pasta_saida", "")
    checks.append((True if ps else None, f"{_t('chk_saida')}: {ps or '(padrao)'}"))

    try:
        import psutil
        rt = psutil.virtual_memory().total / 1024**3
        bench = _cfgv(cfg, "benchmark", False)
        mc = _cfgv(cfg, "monte_carlo", False)
        shap = _cfgv(cfg, "shap_benchmark", False)
        if rt < 4 and (bench or mc or shap):
            checks.append((None, _t("chk_warn_hw") + f" ({rt:.1f}GB)"))
        else:
            checks.append((True, _t("chk_hw") + f" ({rt:.1f}GB)"))
    except ImportError:
        checks.append((None, "psutil N/A"))

    pp = _cfgv(cfg, "pre_processamento", "")
    checks.append((True if pp else None, f"{_t('chk_preproc')}: {pp or '—'}"))
    _fex = _cfgv(cfg, "faixas_excluir", ()) or ()
    if _fex:
        checks.append((True, ("Faixas excluidas (nao entram no modelo): " if _lang() == "PT" else "Excluded ranges (not in the model): ")
                       + ", ".join(map(str, _fex))))

    est = _estimar_tempo(cfg, n_para_estimar)
    if est:
        # Dica acionavel junto da estimativa: o unico campo que muda a ordem
        # de grandeza sem mudar nenhum resultado e' o paralelismo do teste de
        # permutacao (resultado identico, so' o tempo muda).
        dica = ""
        n_jobs = int(_cfgv(cfg, "n_jobs_permutacao", 1) or 1)
        if n_jobs == 1 and int(_cfgv(cfg, "n_permutacoes", 0) or 0) >= 50:
            dica = "  " + _t("chk_dica_jobs")
        checks.append((True, f"{_t('chk_tempo')}: {est}{dica}"))

    return (len(erros) == 0), erros, checks


def _print_checklist(cfg: Config) -> bool:
    ok, erros, checks = _checklist(cfg)
    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column("S", width=3, no_wrap=True)
    t.add_column("Item", style=PW)
    for estado, msg in checks:
        if estado is True:
            t.add_row(Text("✓", style=PG), escape(msg))
        elif estado is False:
            t.add_row(Text("✖", style=PR), escape(msg))
        else:
            t.add_row(Text("—", style=PM), escape(msg))
    b_hex = PG if ok else PR
    lbl = "Checklist Pre-Execucao" if _lang()=="PT" else "Pre-Execution Checklist"
    console.print(Panel(t, title=f"[{PA}]{lbl}[/{PA}]",
                        border_style=b_hex, box=rbox.ROUNDED, padding=(0, 1)))
    return ok


# ---------------------------------------------------------------------------
# RESUMO CIENTIFICO
# ---------------------------------------------------------------------------
def _print_resumo(cfg: Config) -> None:
    lang = _lang()
    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column("P", style=PM, width=22, no_wrap=True)
    t.add_column("V", style=PS, no_wrap=True)

    def row(lbl, val):
        if isinstance(val, bool):
            # Text.from_markup, NAO Text(): o construtor trata a string como
            # LITERAL e imprimia "[g]Yes[/g]" cru na tela (em PT e EN).
            if val:
                v_txt = Text.from_markup("[g]Sim[/g]" if lang == "PT" else "[g]Yes[/g]")
            else:
                v_txt = Text.from_markup("[m]Nao[/m]" if lang == "PT" else "[m]No[/m]")
            t.add_row(lbl, v_txt)
        else:
            # Mesmo motivo: alguns valores (ex.: "[err]KFold[/err]") carregam
            # markup e precisam ser interpretados, nao exibidos literalmente.
            texto = str(val)
            t.add_row(lbl, Text.from_markup(texto) if "[" in texto
                            else Text(texto, style=PS))

    row(_t("res_tecnica"),  _TECNICA_SELECIONADA.get("nome", "FT-NIR"))
    row(_t("res_preproc"),  _cfgv(cfg, "pre_processamento", "—"))
    _fex = _cfgv(cfg, "faixas_excluir", ()) or ()
    if _fex:
        row("Faixas excluidas" if _lang() == "PT" else "Excluded ranges", ", ".join(map(str, _fex)))
    _norm = _cfgv(cfg, "normalizacao", "nenhuma")
    if _norm and _norm != "nenhuma":
        row("Normalizacao" if _lang() == "PT" else "Normalisation", str(_norm))
    row(_t("res_modelo"),   "PLS-DA")
    row(_t("res_lvs"),      _cfgv(cfg, "max_lvs", "—"))
    from guaraci.dados_io import garantia_agrupamento_prevista as _gap
    _ng = _gap(cfg)
    row(_t("res_valid"),
        {"high": "GroupKFold (mae_id)",
         "medium": "GroupKFold (grupo declarado)",
         "verificar": "GroupKFold (se houver estrutura de grupo)"}.get(
            _ng, "[err]StratifiedKFold — SEM agrupamento[/err]"))
    row(_t("res_perm"),     _cfgv(cfg, "n_permutacoes", "—"))
    row(_t("res_opls"),     _cfgv(cfg, "opls_da", True))
    row(_t("res_dds"),      _cfgv(cfg, "ddsimca", True))
    row(_t("res_bench"),    _cfgv(cfg, "benchmark", False))
    row(_t("res_mc"),       _cfgv(cfg, "monte_carlo", False))
    row(_t("res_shap"),     _cfgv(cfg, "shap_benchmark", False))
    row(_t("res_dpi"),      _cfgv(cfg, "dpi", 300))
    row(_t("res_fmt"),      _cfgv(cfg, "formato_figura", "png"))
    row(_t("res_nivel"),    _rotulo_opcao("nivel", _cfgv(cfg, "nivel", "N1")))
    tag = getattr(cfg, "tag", "") or ""
    if tag: row(_t("res_tag"), tag)

    lbl = "Configuracao Cientifica" if lang=="PT" else "Scientific Configuration"
    console.print(Panel(t, title=f"[{PA}]{lbl}[/{PA}]",
                        border_style=PA, box=rbox.ROUNDED, padding=(0, 1)))


# ---------------------------------------------------------------------------
# EXECUCAO DO PIPELINE
# ---------------------------------------------------------------------------
def _montar_painel_execucao(texto_log: str, elapsed: float,
                             objetivo_rotulo: str,
                             plano_figuras: List[str]) -> Panel:
    """Monta o painel de acompanhamento ao vivo (auditoria jul/2026, item 5):
    objetivo cientifico, barra de progresso + rotulo da analise em
    andamento (via app_logic.log_progress), figuras ja concluidas
    (app_logic.figures_completed) contra o plano do mode (modos_analise.
    describe_plan), tempo decorrido/estimado restante e avisos nao-fatais
    (app_logic.log_warnings). Extraida de _rodar_pipeline como funcao de
    modulo para ser testavel isoladamente (ver test_guaraci_cli.py) sem
    precisar rodar o pipeline de verdade nem simular entrada interativa."""
    frac, label = _progresso_do_log(texto_log, len(plano_figuras) or None)
    figs = _figures_completed(texto_log)
    avisos = _avisos_do_log(texto_log)

    # LIMITE DE ALTURA (bug real, 2026-08-07: "tela preta").
    # `figs` e `avisos` crescem sem teto durante a execucao. Numa corrida
    # completa (26 figuras + varios avisos distintos) o painel passava de 35
    # linhas num terminal de 24. O `Live` do Rich reposiciona o cursor para
    # redesenhar; quando o bloco nao cabe na janela ele nao consegue e a tela
    # fica preta com so' o cursor piscando -- exatamente o sintoma relatado.
    # O calculo seguia rodando por baixo, so' o painel morria.
    # Solucao: manter o painel com altura LIMITADA, mostrando os itens mais
    # recentes (que e' o que interessa acompanhar) + um contador do resto.
    MAX_AVISOS  = 4
    MAX_FIG_LIN = 3      # linhas gastas com a lista de figuras
    n_ocultos = max(0, len(avisos) - MAX_AVISOS)
    avisos_vis = avisos[-MAX_AVISOS:]

    bar_w = 32
    preenchido = int(bar_w * frac)
    barra = "█" * preenchido + "░" * (bar_w - preenchido)
    eta_txt = "…"
    if frac > 0.05:
        eta_txt = _fmt_time(max(0.0, elapsed / frac - elapsed))

    partes: List[Any] = [
        Text.assemble(
            (f"{_t('exec_objetivo')}: ", PM), (objetivo_rotulo, f"bold {PA}")),
        Text(f"[{barra}] {frac * 100:5.1f}%  {label}", style=PA),
        Text(f"{_t('exec_eta')}: {eta_txt}   |   "
             f"{_fmt_time(elapsed)}", style=PW),
        Rule(style=PD),
        Text(f"{_t('exec_figuras')} ({len(figs)}/{len(plano_figuras)} "
             f"{'planejadas' if _lang() == 'PT' else 'planned'}):", style=f"bold {PM}"),
        # overflow="ellipsis" + no_wrap corta na largura; as figuras mais
        # recentes ficam visiveis e a linha nunca cresce em altura.
        Text("  ".join(f"✓ {f}" for f in figs[-MAX_FIG_LIN * 3:])
             if figs else "…",
             style=PW, overflow="ellipsis", no_wrap=False),
    ]
    if avisos:
        cab = f"{_t('exec_avisos')} ({len(avisos)})"
        if n_ocultos:
            cab += (f" — mostrando os {MAX_AVISOS} ultimos, +{n_ocultos} antes" if _lang() == "PT"
                    else f" — showing the last {MAX_AVISOS}, +{n_ocultos} earlier")
        partes += [
            Rule(style=PD),
            Text(f"{cab}:", style=f"bold {PR}"),
            Text("\n".join(f"⚠ {a[:110]}" for a in avisos_vis), style=PR,
                 overflow="ellipsis"),
        ]
    return Panel(Group(*partes), border_style=PA, box=rbox.ROUNDED,
                 padding=(0, 1), title=f"  {_t('exec_inicio')}  ")


@contextlib.contextmanager
def _sem_quickedit():
    """No console do Windows, um clique na janela entra no modo de selecao (QuickEdit) e
    PAUSA toda escrita do programa ate' a pessoa apertar Enter/Esc: o painel e o
    relogio param e parece travamento. Desliga o QuickEdit durante a analise e
    devolve o modo original no fim. Fora do Windows (ou sem console) nao faz nada."""
    if os.name != "nt":
        yield
        return
    modo_orig = None
    try:
        import ctypes
        k32 = ctypes.windll.kernel32           # type: ignore[attr-defined]
        h = k32.GetStdHandle(-10)              # STD_INPUT_HANDLE
        modo = ctypes.c_uint32()
        if k32.GetConsoleMode(h, ctypes.byref(modo)):
            modo_orig = modo.value
            # tira ENABLE_QUICK_EDIT_MODE (0x40); ENABLE_EXTENDED_FLAGS (0x80) e' exigido para valer
            k32.SetConsoleMode(h, (modo_orig & ~0x40) | 0x80)
    except Exception:  # noqa: BLE001 -- conforto de interface; nunca impede a analise
        modo_orig = None
    try:
        yield
    finally:
        if modo_orig is not None:
            try:
                k32.SetConsoleMode(h, modo_orig)
            except Exception:  # noqa: BLE001
                pass


#: Segundos sem nenhuma linha nova no log da analise antes de gravar o diagnostico de travamento.
_SEM_NOVIDADE_S = 120.0


def _gravar_diagnostico_travamento() -> Optional[str]:
    """Grava a pilha de todas as threads (onde cada uma esta' parada) num arquivo em ~/.guaraci.
    Serve para descobrir a etapa que trava numa maquina que nao conseguimos reproduzir."""
    import faulthandler
    try:
        _USER_DIR.mkdir(parents=True, exist_ok=True)
        destino = _USER_DIR / f"travamento_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(destino, "w", encoding="utf-8") as fh:
            fh.write(f"GUARACI {pq.__version__}: sem linha nova no log por {_SEM_NOVIDADE_S:.0f} s\n\n")
            fh.flush()
            faulthandler.dump_traceback(file=fh, all_threads=True)
        return str(destino)
    except Exception:  # noqa: BLE001 -- diagnostico e' extra; falhar aqui nao pode derrubar a analise
        return None


def _rodar_pipeline(cfg: Config, retomar: bool = False) -> None:
    """Menu [R]. `retomar=True`: a pessoa acabou de aceitar retirar classes depois de uma falha de validacao;
    roda de novo direto, sem repetir as perguntas de nome, confirmacao e pre-voo."""
    lang = _lang()
    if retomar:
        _dados_pv = None
        return _rodar_pipeline_execucao(cfg, lang, _dados_pv)
    _cls(); _print_header(cfg)

    pode = _print_checklist(cfg)
    if not pode:
        console.print(f"  [{PR}]{_t('pip_sem_dados')}[/{PR}]")
        _pause(); return

    console.print(); _print_resumo(cfg)

    # Nome da execucao
    console.print()
    tag_atual = getattr(cfg, "tag", "") or ""
    console.print(f"  [{PM}]{_t('tag_atual')}:[/{PM}] [{PA}]{escape(tag_atual) or _t('tag_auto')}[/{PA}]")
    novo = _input(f"  {_t('tag_novo')}")
    if novo == "-":
        cfg.tag = ""; console.print(f"  [{PM}]{_t('tag_limpo')}[/{PM}]")
    elif novo:
        san = _re.sub(r"[^\w\-_]", "_", novo)
        cfg.tag = san; console.print(f"  [g]✓ ID: {escape(san)}[/g]")

    console.print()
    iniciar = _ask(f"  [{PA}]► {_t('rodar')}?[/{PA}] {'(s/n)' if lang == 'PT' else '(y/n)'} ")
    if iniciar.lower() not in ("s","y","sim","yes"):
        console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); _pause(); return
    from guaraci.dados_io import garantia_agrupamento_prevista as _gap2
    if _gap2(cfg) == "none" and _cfgv(cfg, "validacao_group_aware", True):
        console.print(f"  [{PR}]{escape(_t('aviso_sem_grupo'))}[/{PR}]")
        if _ask(f"  [{PA}]{_t('conf_sem_grupo')}[/{PA}]").strip().lower() not in ("s","y","sim","yes"):
            console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); _pause(); return

    # Pre-voo do assistente: regras de configuracao (e de dados, se cabem no
    # limite). Achado critico pergunta antes de rodar; o padrao e' NAO rodar.
    with console.status(f"[{PA}]{'Pre-voo do assistente...' if lang == 'PT' else 'Assistant pre-flight...'}[/{PA}]"):
        _achados_pv, _dados_pv = _preflight(cfg, lang)
    _mostrar_achados(_achados_pv, lang, titulo=("Pre-voo (regras do assistente)" if lang == "PT"
                                                 else "Pre-flight (assistant rules)"), dados=_dados_pv)
    # Classe com uma so' unidade independente: a validacao cruzada nao roda (o pipeline para com erro logo
    # depois de carregar os dados). Em vez de "rodar mesmo assim" para falhar, oferece retirar essas classes.
    # No objetivo exploratorio nao ha' validacao: retirar so' tiraria classes dos graficos.
    from guaraci.modos_analise import EXPLORATORIO, resolve_objective
    _uma = [] if resolve_objective(cfg) == EXPLORATORIO else sorted(
        k for k, v in (getattr(_dados_pv, "unidades_por_classe", None) or {}).items() if v < 2)
    if _uma:
        _lista_uma = ", ".join(_uma)
        _resp = _ask(f"  [{PR}]{'Retirar desta analise as classes com uma so unidade' if lang == 'PT' else 'Remove the single-unit classes from this analysis'}"
                     f" ({escape(_lista_uma)}) {'e rodar?' if lang == 'PT' else 'and run?'}[/{PR}] "
                     f"{'(s/N)' if lang == 'PT' else '(y/N)'} ").strip().lower()
        if _resp not in ("s", "y", "sim", "yes"):
            console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); _pause(); return
        cfg.exclude_classes = tuple(sorted(set(cfg.exclude_classes or ()) | set(_uma)))
        console.print(f"  [{PM}]{'Retiradas' if lang == 'PT' else 'Removed'}: {escape(_lista_uma)} "
                      f"({'campo excluir_classes; apague-o em Dados para trazer de volta' if lang == 'PT' else 'field excluir_classes; clear it in Data to bring them back'})[/{PM}]")
        _achados_pv = [a for a in _achados_pv if getattr(a, "id", "") != "classe_com_uma_unidade"]
    if _criticos(_achados_pv):
        _resp = _ask(f"  [{PR}]{'Ha achado critico. Rodar mesmo assim?' if lang == 'PT' else 'There is a critical finding. Run anyway?'}"
                     f"[/{PR}] {'(s/N)' if lang == 'PT' else '(y/N)'} ").strip().lower()
        if _resp not in ("s", "y", "sim", "yes"):
            console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); _pause(); return
    _rodar_pipeline_execucao(cfg, lang, _dados_pv)


def _rodar_pipeline_execucao(cfg: Config, lang: str, _dados_pv: Any) -> None:
    """Parte do [R] depois das perguntas: prepara, roda com o painel e mostra o resultado."""
    # Mesclar codigos usuario
    try:
        cod_u = _carregar_codigos_usuario()
        if cod_u: pq.CODIGO_ESPECIE.update(cod_u)
    except (OSError, json.JSONDecodeError) as _e_cod:
        logging.getLogger(__name__).debug(
            "codigos de usuario nao mesclados: %s", _e_cod)

    # Aplicar configuracoes visuais
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        vcfg = _carregar_visual_cfg()
        # Implementacao unica em cli_assistente.apply_palette (compartilhada
        # com a aba Modelo do app web) -- e' ela que tambem ativa a paleta em
        # `paleta_cores`, sem o que a escolha nao chegava as figuras.
        _aplicar_paleta(vcfg.get("paleta", "qualitativo"))
        fp = _FONT_PRESETS.get(vcfg.get("tamanho_fonte","m"), {})
        for k, v in fp.items(): plt.rcParams[k] = v
        if vcfg.get("grid_major", True):
            plt.rcParams["axes.grid"] = True
            plt.rcParams["grid.linestyle"] = vcfg.get("grid_style","dotted")
            plt.rcParams["grid.alpha"] = _grid_alpha_valido(vcfg.get("grid_alpha", 0.4))
        else:
            plt.rcParams["axes.grid"] = False
        # `alpha_pontos`: opacidade dos pontos por classe. Antes gravava
        # `plt.rcParams["lines.alpha"]` (chave inexistente -- a preferencia
        # nunca chegou a figura nenhuma, achado B5 de 2026-09-24). Agora
        # passa por `figuras.adaptive_scatter_parameters`, o ponto unico
        # dos graficos de pontos por classe; "medio" = automatico.
        from guaraci.paleta_cores import ALPHA_PONTOS_PRESETS, set_point_alpha
        set_point_alpha(ALPHA_PONTOS_PRESETS.get(
            str(vcfg.get("alpha_pontos", "medio")), None))
    except Exception as _e_vis:  # noqa: BLE001 -- configuracao visual
        # cosmetica (paleta/fonte/grid); um erro aqui nunca deve impedir a
        # corrida de acontecer, so' os defaults do matplotlib ficam em uso.
        logging.getLogger(__name__).debug(
            "configuracao visual nao aplicada: %s", _e_vis)

    # Sincronizar DPI do visual_config antes de salvar
    _sincronizar_dpi(cfg)
    try:
        _USER_DIR.mkdir(parents=True, exist_ok=True)
        _save_config(cfg, str(_CFG_PATH))
    except OSError as _e_cfg_save:
        # Achado do "checkup geral" de interface (2026-08-07): esta chamada
        # nao tinha NENHUMA guarda -- um PermissionError aqui (HOME
        # read-only, disco cheio) derrubava o CLI com traceback bem na hora
        # de rodar a analise, no meio de uma sessao interativa. Config nao
        # persistida so' significa que as escolhas desta sessao nao vao
        # sobreviver ao proximo start -- nao pode impedir a corrida atual.
        _msg = (f"[AVISO] config.yaml nao pode ser salvo ({_e_cfg_save}); "
               f"as preferencias desta sessao nao serao lembradas."
               if lang == "PT" else
               f"[WARNING] config.yaml could not be saved ({_e_cfg_save}); "
               f"this session's preferences will not be remembered.")
        console.print(f"  [{PM}]{escape(_msg)}[/{PM}]")

    # Sugestao de pausa em execucoes longas
    if (_cfgv(cfg, "monte_carlo", False)
            or _cfgv(cfg, "shap_benchmark", False)
            or _cfgv(cfg, "benchmark", False)
            or _cfgv(cfg, "comparar_pre_processamentos", False)):
        _sugerir_pausa()

    # Painel de acompanhamento (auditoria jul/2026, item 5): o terminal deve
    # mostrar quais figuras serao calculadas, quais ja foram concluidas,
    # analise em andamento, % de progresso, tempo estimado e avisos — em vez
    # da simulacao por tempo decorrido usada antes (etapas fixas avancando a
    # cada 15s, sem relacao real com o que o pipeline estava fazendo).
    objetivo_rotulo = pq.OBJETIVO_ROTULO.get(
        pq.resolve_objective(cfg), cfg.level)
    plano_figuras = pq.describe_plan(cfg)

    _done: Dict[str, object] = {"ok": False, "error": None}
    _logger = _LogThreadSafe()

    def _run():
        try:
            with contextlib.redirect_stdout(_logger), \
                 contextlib.redirect_stderr(_logger):
                _executar(cfg)
        except KeyboardInterrupt:
            _done["error"] = _t("exec_interrompido")
        except Exception as e:  # noqa: BLE001 -- boundary de topo da thread de
            # execucao: QUALQUER excecao do pipeline (1269 linhas de
            # executar()) precisa ser capturada aqui para nao matar a
            # thread silenciosamente -- e' reportada em _done["error"] e
            # exibida ao usuario, nunca perdida.
            _done["error"] = str(e)
            _done["exc"] = e      # o painel de falha mostra o codigo GUA-E (assistente_erros)
        finally:
            _done["ok"] = True

    def _render_painel(elapsed: float) -> Panel:
        return _montar_painel_execucao(
            texto_log=_logger.text(), elapsed=elapsed,
            objetivo_rotulo=objetivo_rotulo, plano_figuras=plano_figuras)

    console.print()
    t_ini = time.time()

    # CAUSA RAIZ do bug da "tela preta" (2026-08-07/08): `console` (definido
    # em guaraci_theme.py) e' construido SEM `file=`, entao `Console.file` e'
    # uma property que resolve `sys.stdout` DINAMICAMENTE a cada escrita
    # (rich/console.py: `self._file or sys.stdout`). `contextlib.redirect_
    # stdout` troca `sys.stdout` GLOBALMENTE no processo -- nao por thread.
    # Enquanto `_run()` (rodando em background) segura esse redirect durante
    # TODA a execucao do pipeline, o `Live` deste thread principal tambem
    # passa a escrever no MESMO buffer (`_logger`), nao no terminal de
    # verdade. Medido isolado: 0 bytes chegavam ao "terminal", 100% ia pro
    # buffer engolido. O painel nao travava nem estourava altura -- ele
    # simplesmente escrevia no lugar errado o tempo todo, daí a tela ficar
    # preta com so' o cursor. A correcao anterior (limitar altura do painel,
    # vertical_overflow="crop") ficou valida mas nao atacava esta causa.
    #
    # Fix: capturar a referencia REAL de stdout/stderr ANTES do redirect
    # comecar, e fixar `console._file` nela pela duracao do Live -- assim
    # o Console para de resolver `sys.stdout` dinamicamente e continua
    # escrevendo no terminal de verdade mesmo com o redirect global ativo
    # na outra thread.
    _stdout_real, _stderr_real = sys.stdout, sys.stderr
    _file_original = console._file
    console._file = _stdout_real

    thr = threading.Thread(target=_run, daemon=True)
    thr.start()

    def _aviso_parado(parado_s: float, arquivo: Optional[str]) -> Text:
        if lang == "PT":
            msg = (f"Sem linha nova ha {_fmt_time(parado_s)}: a etapa atual pode ser longa, a analise continua. "
                   + (f"Diagnostico salvo em {arquivo}" if arquivo else ""))
        else:
            msg = (f"No new output for {_fmt_time(parado_s)}: the current step may be long, the analysis keeps running. "
                   + (f"Diagnostics saved to {arquivo}" if arquivo else ""))
        return Text(msg, style=PR)

    try:
        # vertical_overflow="crop": rede de seguranca complementar. Mesmo
        # que o painel volte a crescer alem da janela, o Rich corta o
        # excesso em vez de perder o controle do cursor.
        # redirect_stdout/stderr=False: o Live do Rich, por padrao, troca sys.stdout por um proxy dele e, ao
        # sair, devolve o que encontrou na entrada. A thread da analise ja' tinha trocado sys.stdout pelo
        # buffer do log; quando ela entrava primeiro (o comum), o log escapava para a tela (acima do painel,
        # que ficava em "0.0% Starting...") e, no fim, o Live devolvia o BUFFER como sys.stdout: o painel de
        # erro/conclusao e o "Enter para continuar" iam para o buffer e a tela parecia travada para sempre.
        with _sem_quickedit(), Live(console=console, refresh_per_second=3, vertical_overflow="crop",
                                     redirect_stdout=False, redirect_stderr=False) as live:
            tam_log, t_novidade, diag = -1, time.time(), None
            while not _done["ok"]:
                agora = time.time()
                n_log = len(_logger.text())
                if n_log != tam_log:
                    tam_log, t_novidade = n_log, agora
                parado = agora - t_novidade
                painel: Any = _render_painel(agora - t_ini)
                if parado >= _SEM_NOVIDADE_S:
                    if diag is None:
                        diag = _gravar_diagnostico_travamento() or ""
                    painel = Group(painel, _aviso_parado(parado, diag))
                live.update(painel)
                time.sleep(0.3)
            live.update(_render_painel(time.time() - t_ini))
    finally:
        # Restaura a resolucao dinamica de sys.stdout assim que o Live
        # termina -- nao deixar o pin permanente afetaria qualquer outro
        # uso de `console` depois desta tela (ex.: redirecionamento em
        # outro comando da mesma sessao do CLI).
        console._file = _file_original

    thr.join()
    sys.stdout, sys.stderr = _stdout_real, _stderr_real    # nada do que vem depois pode cair no buffer do log
    console.print()

    if _done.get("error"):
        # Mensagem original + codigo GUA-E, causa provavel e o que fazer; a
        # interrupcao pelo usuario (Ctrl+C) nao e' erro a diagnosticar.
        _corpo_erro = (_diagnostico_erro(_done["exc"], lang) if _done.get("exc") is not None
                       else str(_done["error"]))
        console.print(Panel(
            Text(f"✖ {_t('exec_erro')}\n{_corpo_erro}", style=PR),
            border_style=PR, box=rbox.ROUNDED, padding=(0, 1)
        ))
        # Validacao impossivel por causa de classes pequenas (contadas DEPOIS do holdout, entao o pre-voo
        # pode nao te-las visto): oferece retira-las e roda de novo, sem a pessoa editar a configuracao.
        _sem_validacao = [c for c in getattr(_done.get("exc"), "classes", None) or []
                          if c not in set(cfg.exclude_classes or ())]
        if _sem_validacao:
            _lista = ", ".join(_sem_validacao)
            _resp = _ask(f"  [{PA}]{'Retirar tambem' if lang == 'PT' else 'Also remove'} {escape(_lista)} "
                         f"{'e rodar de novo?' if lang == 'PT' else 'and run again?'}[/{PA}] "
                         f"{'(s/N)' if lang == 'PT' else '(y/N)'} ").strip().lower()
            if _resp in ("s", "y", "sim", "yes"):
                cfg.exclude_classes = tuple(sorted(set(cfg.exclude_classes or ()) | set(_sem_validacao)))
                console.print(f"  [{PM}]{'Retiradas' if lang == 'PT' else 'Removed'}: {escape(_lista)} "
                              f"(excluir_classes)[/{PM}]")
                _rodar_pipeline(cfg, retomar=True)
                return
    else:
        pasta_s = _cfgv(cfg, "pasta_saida", "resultados")
        tag     = getattr(cfg, "tag", "") or ""
        destino = f"{pasta_s}/{tag}" if tag else pasta_s
        lbl_concluido = _t("exec_concluido").upper()
        lbl_saida     = _t("exec_saida")
        console.print(Panel(
            Align.center(Group(
                Text(f"\n  ✓ {lbl_concluido}\n", style=f"bold {PG}"),
                Text(f"  {lbl_saida}:\n  {destino}/\n", style=PW),
            )),
            border_style=PG, box=rbox.ROUNDED, padding=(0, 2)
        ))
        # Assistente depois de rodar: ficha anti-vazamento ao lado do
        # resumo_modelo.txt e os achados de resultado (falha so' avisa).
        _pos = _assistente_pos_execucao(cfg, _dados_pv, lang)
        if _pos:
            _mostrar_achados(_pos[1], lang, titulo=("Leitura do resultado (regras do assistente)" if lang == "PT"
                                                    else "Result check (assistant rules)"))
            console.print(f"  [{PM}]{'Ficha anti-vazamento' if lang == 'PT' else 'Leakage sheet'}: {escape(_pos[0])}[/{PM}]")

    _pause()


# ---------------------------------------------------------------------------
# SALVAR / CARREGAR PERFIL
# ---------------------------------------------------------------------------
def _salvar_yaml(cfg: Config) -> None:
    lang = _lang()
    console.print()
    lbl = "Nome do perfil: " if lang=="PT" else "Profile name: "
    nome = _input(f"  {lbl}").strip()
    if not nome:
        console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); _pause(); return
    san = _re.sub(r"[^\w\-_]", "_", nome)
    _PERFIS_DIR.mkdir(parents=True, exist_ok=True)
    path = _PERFIS_DIR / f"{san}.yaml"
    try:
        _save_config(cfg, str(path))
        _lbl = "Salvo" if _lang() == "PT" else "Saved"
        console.print(f"  [g]✓ {_lbl}: {escape(str(path))}[/g]")
    except OSError as e:
        console.print(f"  [err]{escape(str(e))}[/err]")
    _pause()


def _carregar_yaml(cfg: Config) -> None:
    _PERFIS_DIR.mkdir(parents=True, exist_ok=True)
    arquivos = sorted(_PERFIS_DIR.glob("*.yaml"))
    if not arquivos:
        lbl = "Nenhum perfil salvo." if _lang()=="PT" else "No saved profiles."
        console.print(f"  [{PM}]{lbl}[/{PM}]"); _pause(); return

    t = Table(show_header=False, box=rbox.SIMPLE, padding=(0, 1))
    t.add_column("N", style=PA, width=4)
    t.add_column("Nome", style=PW)
    for i, f in enumerate(arquivos, 1):
        t.add_row(str(i), escape(f.stem))
    titulo = "Perfis Salvos" if _lang()=="PT" else "Saved Profiles"
    console.print(Panel(t, title=f"[bold {PA}]{titulo}[/bold {PA}]",
                        border_style=PA, box=rbox.ROUNDED, padding=(0,1)))
    raw = _input("  N: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(arquivos):
        path = arquivos[int(raw) - 1]
        try:
            cfg2 = _load_config(str(path))
            for k, v in vars(cfg2).items():
                try: setattr(cfg, k, v)
                except (AttributeError, TypeError) as _e_attr:
                    # Campo do perfil salvo desalinhado com o Config atual
                    # (ex.: perfil antigo de uma versao com schema diferente).
                    logging.getLogger(__name__).debug(
                        "perfil '%s': campo '%s' nao aplicado: %s",
                        path.stem, k, _e_attr)
            _lbl = "Carregado" if _lang() == "PT" else "Loaded"
            console.print(f"  [g]✓ {_lbl}: {escape(path.stem)}[/g]")
        except (RuntimeError, FileNotFoundError, ValueError) as e:
            console.print(f"  [err]{escape(str(e))}[/err]")
    else:
        console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]")
    _pause()


# ===========================================================================
# COMANDOS DE LINHA DE COMANDO (--version, demo, doctor)
# ===========================================================================
def _texto_versao() -> str:
    """Texto de `--version`: versao, autor, licenca, aviso de garantia e a pasta de instalacao."""
    pasta = Path(__file__).resolve().parent
    linhas = [
        f"GUARACI v{pq.__version__}",
        f"Autor: {AUTOR_NOME} (ORCID {AUTOR_ORCID})",
        COPYRIGHT_LINHA,
        f"Repositorio: {REPO_URL}",
        f"Duvidas e suporte: {DISCUSSIONS_URL}",
        f"Licenca: {LICENCA_SPDX} (software livre). SEM GARANTIA: nenhuma, nem implicita, de "
        "comercializacao ou de adequacao a um fim; veja o arquivo LICENSE.",
        f"Instalado em: {pasta}",
    ]
    return chr(10).join(linhas)


def _comando_versao() -> None:
    print(_texto_versao())


# Modulos cujo nome de import difere do nome da distribuicao no PyPI. Sem este
# mapa, buscar os metadados por `<nome do modulo>` falha e a versao sai como "?".
_DIST_DE_MODULO = {"sklearn": "scikit-learn", "yaml": "PyYAML", "PIL": "pillow",
                   "skimage": "scikit-image", "fpdf": "fpdf2",
                   "docx": "python-docx", "pptx": "python-pptx"}


def _versao_por_metadados(modulo: str) -> str:
    """Versao pelos metadados da distribuicao, para pacotes sem __version__."""
    from importlib import metadata
    try:
        return metadata.version(_DIST_DE_MODULO.get(modulo, modulo))
    except metadata.PackageNotFoundError:
        return "?"


def _caminho_app_web() -> Path:
    """Arquivo do app web dentro do PACOTE (`guaraci/app_web.py`), achado por
    `importlib.resources`: funciona instalado por pip, em modo editavel e numa
    copia do repositorio, sem depender da raiz do repositorio."""
    from importlib import resources
    return Path(str(resources.files("guaraci").joinpath("app_web.py")))


def _opcoes_streamlit_do_modo_local() -> List[str]:
    """Opcoes do Streamlit passadas na linha de comando: o `.streamlit/config.toml`
    do repositorio so' vale com a pasta atual na raiz dele, e o pacote instalado
    nao o leva. Tema vem dos design tokens (mesma fonte que o app usa)."""
    from guaraci.design_tokens import tokens
    claro, escuro = tokens("light"), tokens("dark")
    return [
        "--server.address=127.0.0.1", "--server.maxUploadSize=500",
        "--server.headless=true", "--server.enableXsrfProtection=true",
        "--browser.gatherUsageStats=false",
        # o modo servidor esconde o erro; no modo local o pesquisador continua
        # vendo o detalhe completo, como antes
        "--client.showErrorDetails=full",
        "--theme.base=light", f"--theme.primaryColor={claro['primary']}",
        f"--theme.backgroundColor={claro['bg']}",
        f"--theme.secondaryBackgroundColor={claro['surface']}",
        f"--theme.textColor={claro['text']}",
        "--theme.font=sans serif", "--theme.baseRadius=0.6rem",
        f"--theme.dark.primaryColor={escuro['primary']}",
        f"--theme.dark.backgroundColor={escuro['bg']}",
        f"--theme.dark.secondaryBackgroundColor={escuro['surface']}",
        f"--theme.dark.textColor={escuro['text']}",
    ]


def _comando_web() -> None:
    """`guaraci web`: interface web em MODO LOCAL (GUARACI_LOCAL_MODE=1),
    presa a 127.0.0.1. Sem esta flag o app web roda em modo servidor (so'
    upload, sessao isolada, sem pickle) -- ver `guaraci.web_seguranca`."""
    import subprocess
    app = _caminho_app_web()
    if not app.is_file():
        print("Erro: o app web (guaraci/app_web.py) nao foi encontrado nesta "
              "instalacao. Instale com `pip install '.[web]'`.",
              file=sys.stderr)
        raise SystemExit(1)
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Erro: falta o Streamlit. Instale com "
              "`pip install '.[web]'`.", file=sys.stderr)
        raise SystemExit(1)
    env = dict(os.environ)
    env["GUARACI_LOCAL_MODE"] = "1"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app),
           *_opcoes_streamlit_do_modo_local()]
    print("Abrindo a interface web local (o endereco aparece abaixo; "
          "Ctrl+C para encerrar)...")
    raise SystemExit(subprocess.call(cmd, env=env))  # nosec B603 -- lista fixa, sem shell


def _comando_doctor() -> None:
    """Diagnostico de ambiente: Python, dependencias, RAM/CPU/disco.

    Nao lanca excecao — cada checagem e best-effort, pensada para rodar
    em qualquer maquina antes do usuario abrir um issue de instalacao.
    """
    import importlib.util
    import platform

    pt = _lang() == "PT"

    def t(a: str, b: str) -> str:
        return a if pt else b

    linhas: List[str] = []
    linhas.append(f"GUARACI v{pq.__version__} — " + t("diagnostico de ambiente", "environment check"))
    linhas.append(t("Data: ", "Date: ") + f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
    linhas.append("")
    linhas.append(f"Python:   {platform.python_version()} ({platform.python_implementation()})")
    linhas.append(t("SO:       ", "OS:       ") + f"{platform.system()} {platform.release()} ({platform.machine()})")

    hw = pq.hardware_probe()
    if hw.get("ram_total_gb") is None or hw.get("ram_livre_gb") is None:
        linhas.append(t("RAM:      nao medida (nenhuma fonte disponivel neste sistema)",
                         "RAM:      not measured (no source available on this system)"))
    else:
        linhas.append(f"RAM:      {hw['ram_livre_gb']:.1f} GB " + t("livre", "free") + f" / {hw['ram_total_gb']:.1f} GB total")
    linhas.append(f"CPU:      {hw['cpu_fisicos']} " + t("fisicos", "physical") + f" / {hw['cpu_logicos']} " + t("logicos", "logical"))
    linhas.append(t("Disco:    ", "Disk:     ") + f"{hw['disco_livre_gb']:.1f} GB " + t("livre (pasta atual)", "free (current folder)"))
    if not hw["psutil_ok"]:
        linhas.append(t("          (psutil ausente — RAM lida do sistema, quando possivel; CPU estimada)",
                         "          (psutil missing — RAM read from the system when possible; CPU estimated)"))

    linhas.append("")
    linhas.append(t("Dependencias obrigatorias:", "Required dependencies:"))
    obrigatorias = ["numpy", "pandas", "scipy", "sklearn", "matplotlib",
                     "joblib", "threadpoolctl", "yaml", "rich", "PIL"]
    faltando_obrig = []
    for mod in obrigatorias:
        spec = importlib.util.find_spec(mod)
        if spec is None:
            linhas.append(t("  [FALTA] ", "  [MISSING] ") + mod)
            faltando_obrig.append(mod)
        else:
            try:
                m = importlib.import_module(mod)
                ver = getattr(m, "__version__", None)
                if ver is None:
                    # Nem todo pacote expoe __version__ (rich, por exemplo) --
                    # os metadados da distribuicao sempre tem. Sem este fallback
                    # o doctor imprimia "rich ?" com o rich instalado e sao.
                    ver = _versao_por_metadados(mod)
            except ImportError as _e_imp:
                ver = t("erro ao importar: ", "import error: ") + str(_e_imp)
            linhas.append(f"  [ok]    {mod} {ver}")

    linhas.append("")
    linhas.append(t("Extras opcionais", "Optional extras") + " ([web], [reports], [benchmark], [imagem]):")
    # CHAVE = nome do MODULO a importar; valor = (rotulo exibido, extra).
    # Nao confundir com o nome do PACOTE no pip: `pip install fpdf2` instala o
    # modulo `fpdf`, e `find_spec("fpdf2")` sempre devolve None -- o doctor
    # reportava "ausente" com o pacote instalado (bug achado em 2026-08-05
    # rodando o doctor de verdade num ambiente completo).
    opcionais = {
        "streamlit": ("streamlit", "web"),
        "psutil":    ("psutil", "web"),
        "fpdf":      ("fpdf2", "reports"),
        "docx":      ("python-docx", "reports"),
        "openpyxl":  ("openpyxl", "reports"),
        "pptx":      ("python-pptx", "reports"),
        "xgboost":   ("xgboost", "benchmark"),
        "shap":      ("shap", "benchmark"),
        "skimage":   ("scikit-image", "imagem"),
    }
    for mod, (rotulo, extra) in opcionais.items():
        spec = importlib.util.find_spec(mod)
        status = "ok" if spec is not None else t("ausente", "missing")
        linhas.append(f"  [{status:7s}] {rotulo:13s} ({extra})")

    # P6: extras de FORMATO e de recurso (OPUS, SPC, RMN, HPLC/GC, PARAFAC, textura, PCV): cada um com o que deixa
    # de funcionar e como instalar. Fonte unica: extras_opcionais.EXTRAS (a mesma da tela [2.4] e do `guaraci checar`).
    linhas.append("")
    linhas.append(t("Extras de formatos e recursos (o que cada um habilita):", "Format and feature extras (what each one enables):"))
    for e in _extras.EXTRAS:
        ok = _extras.instalado(e)
        status = "ok" if ok else t("ausente", "missing")
        linhas.append(f"  [{status:7s}] {e.pacote:13s} ({e.extra}): " + (e.afeta[0] if pt else e.afeta[1]))
        if not ok:
            linhas.append("            " + t("tecnicas afetadas: ", "affected techniques: ") + (", ".join(e.tecnicas) or "-")
                          + " | " + t("instalar: ", "install: ") + _extras.texto_instalar(e, "PT" if pt else "EN"))
    _faltam_ext = _extras.ausentes()
    if _faltam_ext:
        linhas.append("  " + t("Para instalar todos de uma vez: ", "To install all of them at once: ") + _extras.COMANDO_TUDO)
    if _faltam_ext or any(importlib.util.find_spec(m) is None for ms in _extras.EXTRAS_DE_CONFORTO.values() for m in ms):
        linhas.append("  " + t("Ou deixe o GUARACI instalar o que falta (pede confirmacao): ",
                                "Or let GUARACI install what is missing (asks for confirmation): ") + "guaraci doctor --instalar")

    console.print()
    for l in linhas:
        console.print(f"  {escape(l)}")
    console.print()

    if faltando_obrig:
        console.print("  [err]✗ " + t("Dependencias obrigatorias faltando: ", "Missing required dependencies: ")
                       + ", ".join(faltando_obrig) + t(". Rode, na pasta do repositorio: ", ". Run, in the repository folder: ")
                       + "pip install -e .[/err]")
    else:
        console.print("  [g]✓ " + t("Todas as dependencias obrigatorias estao instaladas.",
                                  "All required dependencies are installed.") + "[/g]")

    destino = Path.cwd() / "guaraci_doctor.txt"
    try:
        destino.write_text("\n".join(linhas), encoding="utf-8")
        console.print(f"  [{PM}]" + t("Relatorio salvo em: ", "Report saved to: ") + f"{escape(str(destino))}[/{PM}]")
    except OSError as e:
        console.print("  [err]" + t("Nao foi possivel salvar o relatorio: ", "Could not save the report: ") + f"{escape(str(e))}[/err]")


def _doctor_instalar(sim: bool = False) -> None:
    """`guaraci doctor --instalar`: instala, com o pip DESTE Python, o que falta para o GUARACI funcionar por inteiro
    (navegador, relatorios e os leitores/recursos opcionais disponiveis para esta versao do Python). Mostra o que vai
    instalar e pede confirmacao; `--sim` pula a pergunta. Sai com 1 se o pip falhar. `benchmark` (xgboost, shap) nao entra:
    e' grande e so' serve a comparacao com outros algoritmos."""
    import subprocess

    pt = _lang() == "PT"

    def t(a: str, b: str) -> str:
        return a if pt else b

    plano = _extras.plano_instalacao()
    if not plano:
        console.print("  [g]✓ " + t("Nada a instalar: tudo o que o GUARACI usa ja esta neste Python.",
                                    "Nothing to install: everything GUARACI uses is already in this Python.") + "[/g]")
        return
    reqs = sorted({r for lista in plano.values() for r in lista})
    console.print("  " + t("Falta instalar:", "Missing:"))
    for extra, lista in plano.items():
        console.print(f"    [{PA}]{extra}[/{PA}]: " + (", ".join(lista) if lista else t("(sem requisito legivel)", "(no readable requirement)")))
    if not reqs:
        # GUARACI rodando da pasta de fontes (sem metadados do pacote): o pip precisa do pyproject.toml da pasta.
        extras = ",".join(plano)
        console.print("  [warn]" + t("O GUARACI nao esta instalado como pacote, entao nao sei as versoes. Na pasta do projeto rode: ",
                                      "GUARACI is not installed as a package, so I do not know the versions. In the project folder run: ")
                      + escape(f'pip install -e ".[{extras}]"') + "[/warn]")
        raise SystemExit(1)
    comando = [sys.executable, "-m", "pip", "install", *reqs]
    console.print("  " + t("Comando: ", "Command: ") + escape(" ".join(f'"{c}"' if any(ch in c for ch in "<>") else c for c in comando)))
    if not sim:
        resp = _ask("  " + t("Instalar agora? [s/N] ", "Install now? [y/N] ")).strip().lower()
        if resp not in ("s", "sim", "y", "yes"):
            console.print("  " + t("Nada foi instalado.", "Nothing was installed."))
            return
    rc = subprocess.run(comando, check=False).returncode
    if rc != 0:
        console.print("  [err]✗ " + t("O pip falhou (codigo ", "pip failed (code ") + f"{rc}). "
                      + t("Confira a conexao e rode de novo; 'guaraci doctor' mostra o que continua faltando.",
                          "Check the connection and run again; 'guaraci doctor' shows what is still missing.") + "[/err]")
        raise SystemExit(1)
    restante = _extras.plano_instalacao()
    if restante:
        console.print("  [warn]" + t("Instalado, mas ainda falta: ", "Installed, but still missing: ") + ", ".join(restante) + "[/warn]")
        raise SystemExit(1)
    console.print("  [g]✓ " + t("Tudo instalado. Rode 'guaraci doctor' para conferir.", "All installed. Run 'guaraci doctor' to check.") + "[/g]")


def _comando_run(caminho_config: Optional[str]) -> None:
    """Executa o pipeline uma vez a partir de um `config.yaml`, sem abrir
    o assistente interativo, e sai (0 sucesso, 1 erro de execução, 2 uso
    incorreto -- mesmo contrato documentado em `_TEXTO_AJUDA`).

    Proposta P1 da rodada multiagente de 2026-09-10 (lacuna de produto
    frente a concorrentes com linha de comando não interativa, ex.
    PLS_Toolbox/MATLAB, SIMCA-Q): antes, a única forma de rodar sem o
    menu era a API Python (`guaraci.pipeline.executar(cfg)`) -- isso
    continua valendo para uso programático mais rico; `run` cobre o caso
    comum de automação/CI (`guaraci run config.yaml`).

    NÃO abre o explorador de arquivos ao final (diferente de `demo`) --
    uso não-interativo/em script não deve depender de sessão gráfica.
    `caminho_config` omitido usa `./config.yaml` (diretório de trabalho
    atual), o mesmo arquivo que o comentário de cabeçalho de
    `config.yaml` já descreve como o jeito de rodar."""
    from guaraci import erros_entrada as _erros
    lang = "en" if (_lang() != "PT" or _erros.idioma() == "en") else "pt"
    rotulo_erro = "Erro" if lang == "pt" else "Error"
    caminho = Path(caminho_config) if caminho_config else Path.cwd() / "config.yaml"
    if not caminho.is_file():
        print(f"{rotulo_erro}: " + (
            f"arquivo de config nao encontrado: {caminho}. Rode `guaraci exemplo classificar PASTA` para gerar um config.yaml pronto, "
            "ou copie config.example.yaml e preencha." if lang == "pt" else
            f"config file not found: {caminho}. Run `guaraci exemplo classificar FOLDER` to build a ready config.yaml, "
            "or copy config.example.yaml and fill it in.") + "\n", file=sys.stderr)
        print(_TEXTO_AJUDA, file=sys.stderr)
        raise SystemExit(2)

    try:
        cfg = _load_config(str(caminho))
    except yaml.YAMLError as e:
        marca = getattr(e, "problem_mark", None)
        onde = f" (linha {marca.line + 1})" if marca is not None else ""
        print(f"{rotulo_erro}: " + (
            f"{caminho} nao e' um YAML valido{onde}. Confira a indentacao, os dois-pontos e os colchetes, ou recomece de config.example.yaml."
            if lang == "pt" else
            f"{caminho} is not valid YAML{' (line ' + str(marca.line + 1) + ')' if marca is not None else ''}. "
            "Check the indentation, the colons and the brackets, or start again from config.example.yaml.") + "\n", file=sys.stderr)
        raise SystemExit(2) from e
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"{rotulo_erro}: " + (f"config invalido em {caminho}: {e}" if lang == "pt" else f"invalid config in {caminho}: {e}") + "\n",
              file=sys.stderr)
        raise SystemExit(2) from e
    if cfg.mode == "dx" and cfg.input_folder and not os.path.exists(cfg.input_folder):
        print(f"{rotulo_erro}: " + _erros.pasta_de_dados_ausente(str(cfg.input_folder), lang) + "\n", file=sys.stderr)
        raise SystemExit(2)

    if _SEM_AJUSTE[0]:
        cfg.auto_adjust = False
    if _MAIS_GRAFICOS[0]:
        cfg.figure_selection = "todos"
    print(f"[GUARACI] Executando com {caminho} ...")
    from guaraci.dados_io import garantia_agrupamento_prevista
    if (garantia_agrupamento_prevista(cfg) == "none"
            and getattr(cfg, "group_by_mae_id", True)):
        # Nao bloqueia (run e' nao interativo), mas o aviso e' visivel ANTES
        # da execucao e repetido no fim (pipeline) e no model card.
        print(f"[GUARACI] {_t('aviso_sem_grupo')}", file=sys.stderr)
    # Pre-voo do assistente (assistente_regras): o mesmo criterio de `guaraci
    # checar`, que sai com 1 havendo achado critico -- `run` tambem para, antes
    # de comecar, a menos que venha --sem-preflight (ai so' avisa).
    lang_ass = "EN" if lang == "en" else "PT"
    achados_pv, dados_pv = _preflight(cfg, lang_ass)
    print("[GUARACI] " + _texto_preflight(achados_pv, dados_pv, lang_ass))
    criticos = _criticos(achados_pv)
    if criticos and not _SEM_PREFLIGHT[0]:
        sys.stdout.flush()
        print(f"{rotulo_erro}: " + (
            f"o pre-voo encontrou {len(criticos)} achado(s) critico(s); a execucao nao comecou. Corrija a configuracao "
            "ou rode com --sem-preflight para executar mesmo assim." if lang == "pt" else
            f"the pre-flight found {len(criticos)} critical finding(s); the run did not start. Fix the configuration "
            "or run with --sem-preflight to run anyway."), file=sys.stderr)
        raise SystemExit(1)
    if criticos:
        print("[GUARACI] " + ("--sem-preflight: executando apesar do(s) achado(s) critico(s) acima." if lang == "pt" else
                              "--sem-preflight: running despite the critical finding(s) above."), file=sys.stderr)
    try:
        pq.executar(cfg)
        _figuras_pos_execucao(cfg)
    except Exception as e:  # noqa: BLE001 -- `run` e' o caminho nao-interativo/
        # script: precisa devolver uma mensagem legivel + codigo de saida
        # 1, nunca um stack trace cru estourando pro chamador (mesmo
        # padrao de _comando_demo acima).
        sys.stdout.flush()   # R2-14: o erro saia no TOPO de um log redirecionado
        print(("Erro: a execucao falhou." if lang == "pt" else "Error: the run failed.") + "\n"
              + _diagnostico_erro(e, lang_ass), file=sys.stderr)
        raise SystemExit(1) from e

    texto_pos = _texto_pos_execucao(_assistente_pos_execucao(cfg, dados_pv, lang_ass), lang_ass)
    if texto_pos:
        print("[GUARACI] " + texto_pos)
    print(f"[GUARACI] Concluido. Saida em: {cfg.output_folder}")


#: Quadro que fecha o `demo`: explica por que aparecem 2 avisos CRITICOS e por que a
#: acuracia 1,000 nao e um resultado de desempenho (o dado do demo e sintetico e
#: confundido de proposito simples; um dado sem confundimento fica para a v1.1).
_QUADRO_FINAL_DEMO: Tuple[str, ...] = (
    "",
    "  O que os 2 avisos [AUDITORIA][CRITICO] acima significam",
    "  - Neste dado sintetico cada especie foi 'medida' em UMA unica sessao de coleta.",
    "    Assim o modelo pode estar distinguindo a sessao e nao a especie (confundimento",
    "    entre classe e sessao), e as replicas de uma mesma sessao nao sao amostras independentes.",
    "  - Por isso a acuracia 1,000 NAO prova desempenho: e consequencia desse desenho,",
    "    nao uma qualidade do modelo. O demo serve para mostrar o fluxo e as saidas.",
    "  - No seu estudo, o mesmo aviso pede que cada classe apareca em varias sessoes",
    "    (por exemplo, coletar cada especie em dias diferentes) antes de confiar no numero.",
)


def _comando_demo() -> None:
    """Roda o pipeline completo com dados sinteticos, sem exigir dado do
    usuario. Fluxo dos 5 minutos: pip install -> guaraci demo -> figuras."""
    console.print()
    console.print(f"  [{PA}]GUARACI demo — gerando espectros sinteticos e rodando o pipeline...[/{PA}]")
    console.print(f"  [{PM}](nenhum dado seu e usado; tudo aqui e gerado artificialmente)[/{PM}]")
    console.print()

    saida = Path.cwd() / "GUARACI_Demo"
    cfg = Config(
        input_folder=str(saida / "dados_dummy"),  # mode sintetico ignora isto
        output_root_folder=str(saida),
        mode="sintetico",
        tag="demo",
        level="N2",       # DD-SIMCA (sensibilidade LOGO) — diferencial central do projeto
        objective="auto",
        n_per_class=15,
        n_synthetic_points=300,
        wn_min=400.0,
        wn_max=4001.0,
        synthetic_adulterants=("S", "M", "A"),
        max_lvs=15,
        n_splits_cv=3,
        n_repeats_cv=1,
        n_permutations=50,
        n_permutations_wold=50,
        n_bootstrap_vip=10,
        n_bootstrap_bca=100,
        n_monte_carlo=20,
        run_benchmark=False,
        run_monte_carlo=False,
        run_shap=False,
        run_wold=False,
        run_cv_anova=False,
        run_opls=False,
        executar_etapa4=False,
        comparar_pipelines=False,
        compare_hca_pipelines=False,
    )
    os.makedirs(cfg.input_folder, exist_ok=True)

    try:
        pq.executar(cfg)
    except Exception as e:  # noqa: BLE001 -- demo deve reportar erro legivel, nao stack trace cru
        console.print(f"  [err]✗ A demo falhou: {escape(str(e))}[/err]")
        console.print(f"  [{PM}]Rode 'guaraci doctor' para checar o ambiente.[/{PM}]")
        raise SystemExit(1) from e

    runs: List[str] = []
    # `saida` existe sempre: `os.makedirs(cfg.input_folder)` acima a cria (a pasta de dados fica dentro dela).
    for raiz, dirs, _arqs in os.walk(str(saida)):
        for d in dirs:
            if d.startswith("PLSDA_OE_"):
                runs.append(os.path.join(raiz, d))
    if not runs:
        console.print(f"  [err]✗ Pipeline rodou mas nenhuma pasta de saida foi encontrada em {escape(str(saida))}[/err]")
        raise SystemExit(1)

    pasta_run = Path(sorted(runs)[-1])
    console.print()
    console.print(f"  [g]✓ Demo concluida.[/g] Resultados em: [{PA}]{escape(str(pasta_run))}[/{PA}]")
    _graf = os.path.basename(pq.pasta_execucao(str(pasta_run), pq.NOME_GRAFICOS))
    _rel = os.path.basename(pq.pasta_execucao(str(pasta_run), pq.NOME_RELATORIOS))
    console.print(f"  [{PM}]Veja {_graf}/ para as figuras e {_rel}/resumo_modelo.txt "
                  f"para o resumo numerico.[/{PM}]")
    for _ln in _QUADRO_FINAL_DEMO:
        console.print(_ln)

    try:
        if sys.platform == "win32":
            os.startfile(str(pasta_run))  # noqa: S606 -- abre o explorador, caminho e nosso proprio output
        else:
            # subprocess com lista de argumentos (achado de auditoria de
            # seguranca, 2026-08-07): a versao anterior interpolava
            # pasta_run direto numa string de shell (os.system(f'open
            # "{pasta_run}"')) -- pasta_run e' sempre gerado internamente
            # neste caminho (guaraci demo), entao nao era explora'vel HOJE,
            # mas e' o mesmo PADRAO que seria uma injecao de comando real
            # se algum dia alimentado por um caminho influenciado pelo
            # usuario. Lista de argumentos nunca passa por um shell --
            # elimina a classe de vulnerabilidade por completo, nao so'
            # o caso de uso atual.
            import subprocess
            cmd = ["open"] if sys.platform == "darwin" else ["xdg-open"]
            subprocess.run(cmd + [str(pasta_run)], check=False)
    except OSError as _e_open:
        logging.getLogger(__name__).debug("nao foi possivel abrir a pasta de saida: %s", _e_open)


# ===========================================================================
# MAIN LOOP
# ===========================================================================
#: Texto de `--help`. Fonte unica: `--help`, `help` e a mensagem de
#: argumento desconhecido leem daqui, para nunca divergirem entre si.
_TEXTO_AJUDA = _cmd.texto_ajuda("PT")   # derivado do catalogo (comandos e grupos)
#: comandos que nao tem `--help` proprio: com `--help` ou `-h` mostram a ajuda geral e saem com 0, sem executar nada.
_COMANDOS_SEM_AJUDA_PROPRIA = ("run", "demo", "web", "doctor", "confianca", "manual")

#: Opcoes de linha de comando aplicadas ao Config depois de carrega-lo.
_OPCOES_CLI: Dict[str, str] = {}
#: `--mais-graficos`: grava todos os graficos que se aplicam (graficos: todos) nesta execucao.
_MAIS_GRAFICOS: List[bool] = [False]
#: `--sem-ajuste-automatico`: desliga o ajuste por contexto nesta execucao.
_SEM_AJUSTE: List[bool] = [False]


def _extrair_opcoes(argv: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Separa `--chave=valor` dos argumentos posicionais.

    Aceita as opcoes em qualquer posicao (`guaraci --mode=controle demo` e
    `guaraci demo --mode=controle` sao equivalentes) -- comportamento
    previsivel importa mais aqui do que economizar codigo.
    """
    opcoes: Dict[str, str] = {}
    restantes: List[str] = []
    for arg in argv:
        if arg.startswith("--") and "=" in arg:
            chave, valor = arg[2:].split("=", 1)
            opcoes[chave.strip().lower()] = valor.strip()
        else:
            restantes.append(arg)
    return opcoes, restantes


def _validar_opcoes(opcoes: Dict[str, str]) -> None:
    """Valida ANTES de processar qualquer dado. Sai com codigo 2 (uso
    incorreto), nunca 1 (erro de execucao) -- sao coisas diferentes para
    quem chama o programa de um script."""
    # M-07 do parecer de 2026-09-25: `--modo=controle` (README/paper) era
    # ignorado sem aviso e qualquer opcao desconhecida rodava com rc=0 (um
    # `--perfl=nir_dispersivo_1100_2498nm` usava o perfil generico). `modo` e' alias de `mode`;
    # o que nao for conhecido e' ERRO de uso (codigo 2).
    if "modo" in opcoes:
        opcoes.setdefault("mode", opcoes.pop("modo"))
    desconhecidas = sorted(set(opcoes) - {"perfil", "mode"})
    if desconhecidas:
        print("Erro: opcao desconhecida: "
              + ", ".join(f"--{c}" for c in desconhecidas)
              + ". Opcoes validas: --perfil=NOME, --mode=cego|controle "
              "(--modo e' aceito como sinonimo).", file=sys.stderr)
        raise SystemExit(2)
    for chave in ("perfil", "mode"):
        if chave in opcoes and not opcoes[chave]:
            print(f"Erro: --{chave} exige um valor (ex.: --mode=cego).",
                  file=sys.stderr)
            raise SystemExit(2)
    mode = opcoes.get("mode")
    if mode is not None and mode not in ("cego", "controle"):
        print(f"Erro: --mode={mode} invalido. Use 'cego' (padrao, usa a "
              f"classe predita) ou 'controle' (usa a classe verdadeira, "
              f"so' para diagnostico interno).", file=sys.stderr)
        raise SystemExit(2)
    if "perfil" in opcoes:
        from guaraci.perfil_matriz import (UnknownProfileError,
                                           load_profile)
        try:
            load_profile(opcoes["perfil"])
        except UnknownProfileError as e:
            print(f"Erro: {e}", file=sys.stderr)
            raise SystemExit(2) from None


def _listar_perfis() -> None:
    from guaraci.perfil_matriz import DIR_PERFIS, load_profile
    print("Perfis de tecnica disponiveis:")
    for arquivo in sorted(DIR_PERFIS.glob("*.yaml")):
        perfil = load_profile(arquivo.stem)
        print(f"  {perfil.nome:<14} {perfil.descricao}")
    print("\nUse: guaraci --perfil=NOME  (ou o caminho de um YAML proprio)")


def main(argv: Optional[List[str]] = None) -> None:
    """Ponto de entrada GUARACI (versao unica em _VERSAO).

    `argv` sao os argumentos SEM o nome do programa. O default lê de
    `sys.argv`, que e' o caminho do entry point instalado. Chamadores
    programaticos (testes, embutimento em outro app) devem passar a lista
    explicitamente -- caso contrario herdam o `sys.argv` de quem os
    hospeda, e desde que argumento desconhecido virou erro de uso isso
    significaria sair com codigo 2 por causa das flags do pytest.
    """
    # Saida linha a linha: com stdout redirecionado/em pipe o Python poe
    # buffer de blocos e o usuario nao ve o progresso de uma execucao longa.
    try:
        getattr(sys.stdout, "reconfigure")(line_buffering=True)
    except (AttributeError, OSError, ValueError):
        pass   # stdout sem reconfigure (objeto de teste, redirecionamento)
    if modo_linear():
        _ativar_modo_linear()
    _sincronizar_idioma()
    _argv = list(sys.argv[1:]) if argv is None else list(argv)
    _SEM_AJUSTE[0] = "--sem-ajuste-automatico" in _argv
    _SEM_PREFLIGHT[0] = "--sem-preflight" in _argv
    _MAIS_GRAFICOS[0] = "--mais-graficos" in _argv
    # Pela CLI as figuras saem em subpastas por grupo de analise (DD-SIMCA,
    # PLS-DA, validacao...): quem le sabe quais graficos se corroboram. A API
    # (`import guaraci`) continua com Graficos/ plano.
    from guaraci import grupos_figuras as _grupos_figuras   # leve: nao carrega o matplotlib
    _grupos_figuras.ORGANIZAR = True
    _argv = [a for a in _argv if a not in ("--sem-ajuste-automatico", "--sem-preflight")]
    if (len(_argv) == 2 and _argv[0].strip().lower() in _COMANDOS_SEM_AJUDA_PROPRIA
            and _argv[1].strip().lower() in ("--help", "-h")):
        # `guaraci demo --help` rodava a demonstracao e `guaraci web --help` subia o servidor: pedir ajuda nunca executa nada.
        print(_TEXTO_AJUDA)
        return
    if not (_argv and _argv[0].strip().lower() in _cmd.SUBCOMANDOS):
        _argv = [a for a in _argv if a != "--mais-graficos"]     # subcomandos (monitorar) tratam o proprio
    if _argv and _argv[0].strip().lower() in _cmd.SUBCOMANDOS:
        # Subcomandos com argumentos proprios (anova, exemplo, confianca).
        codigo = _cmd.subcomando(_argv[0].strip().lower(), _argv[1:])
        if codigo:
            raise SystemExit(codigo)
        return
    opcoes, restantes = _extrair_opcoes(_argv)
    _validar_opcoes(opcoes)
    _OPCOES_CLI.clear()
    _OPCOES_CLI.update(opcoes)
    sys.argv = [sys.argv[0]] + restantes

    if restantes:
        comando = restantes[0].strip().lower().lstrip("-")
        if comando in ("version", "v"):
            _comando_versao(); return
        if comando == "demo":
            _comando_demo(); return
        if comando == "run":
            _comando_run(restantes[1] if len(restantes) > 1 else None); return
        if comando == "doctor":
            _comando_doctor()
            _pedidos = {a.strip().lower() for a in restantes[1:]}
            if _pedidos & {"--instalar", "--install"}:
                _doctor_instalar(sim=bool(_pedidos & {"--sim", "--yes", "-y"}))
            return
        if comando == "perfis":
            _listar_perfis(); return
        if comando == "web":
            _comando_web(); return
        if comando in ("help", "h"):
            print(_TEXTO_AJUDA)
            return
        print(f"Erro: comando desconhecido '{sys.argv[1]}'.\n",
              file=sys.stderr)
        print(_TEXTO_AJUDA, file=sys.stderr)
        raise SystemExit(2)

    # Migra estado gravado pela versao anterior (dentro do pacote instalado)
    # para _USER_DIR, se aplicavel -- ver docstring de _migrar_estado_legado.
    _migrar_estado_legado()

    # Carregar config
    cfg = Config()
    if _CFG_PATH.exists():
        try:
            cfg = _load_config(str(_CFG_PATH))
            # M-17a: dizer QUAL config.yaml foi lido (ha' mais de um lugar).
            print(f"[GUARACI] Configuracao: {_CFG_PATH}")
        except (RuntimeError, FileNotFoundError, ValueError) as _e_cfg:
            logging.getLogger(__name__).debug(
                "config.yaml nao carregado no boot, usando defaults: %s", _e_cfg)

    if _SEM_AJUSTE[0]:
        cfg.auto_adjust = False

    # Opcoes de linha de comando vencem o config.yaml: quem digitou a flag
    # agora quer ela agora. Ja' validadas em _validar_opcoes (saida 2).
    if "perfil" in _OPCOES_CLI:
        cfg.matrix_profile = _OPCOES_CLI["perfil"]
    if "mode" in _OPCOES_CLI:
        cfg.label_mode = _OPCOES_CLI["mode"]

    # Idioma salvo: ja' recuperado por _sincronizar_idioma() no inicio de main().

    # Boas-vindas uma vez por sessao (impressa depois do primeiro menu, sem pausa)
    _boas_vindas_pendente = True

    _foto_menu = _foto_cfg(cfg)   # R5-22: base para detectar edicao nao salva
    while True:
        _cls()
        _print_header(cfg)
        _print_status(cfg)
        _print_main_menu(cfg)
        _print_acao_rodar(cfg)
        if _boas_vindas_pendente:
            _exibir_boas_vindas()
            _boas_vindas_pendente = False

        try:
            # input() direto (nao _input()): _input() engole EOFError/
            # KeyboardInterrupt internamente e devolve "" -- com isso o
            # except abaixo NUNCA disparava em EOF real (stdin fechado/
            # redirecionado de arquivo vazio/pipe encerrado). "" nao bate
            # com nenhuma opcao do menu, cai no ramo "invalida" + _pause()
            # (tambem EOF-safe), e o loop volta a chamar _cls() (spawna
            # subprocesso via os.system) e ler de novo -- SEMPRE "" de novo
            # em EOF permanente -- girando para sempre, sem sair, gastando
            # CPU (achado 2026-08-07, "checkup geral" de interface:
            # reproduzido com stdin vazio, >350 redesenhos em 8s sem
            # terminar). input() aqui deixa o EOFError propagar ate o
            # except que ja existe para tratar exatamente este caso.
            raw = input(f"  {_t('opcao')}: ").strip()
            escolha = "?" if raw == "?" else raw.upper().strip()
        except (EOFError, KeyboardInterrupt):
            _exibir_despedida()
            break

        global _AJUDA_TELA
        try:
            _AJUDA_TELA = None
            # Guia v2: numero = grupo (pasta); "grupo.item" = tela direto
            # (2.3 Dados); letras so' para as acoes fixas abaixo.
            _tid = _cat.tela_por_tecla().get(escolha)
            if escolha in {g["num"] for g in _cat.grupos_cli()}:
                _menu_grupo(escolha, cfg)
            elif _tid and escolha != "R":
                _abrir_tela(_tid, cfg)
            elif escolha in ("G", "A"):
                _abrir_assistente("menu principal" if _lang() == "PT" else "main menu", cfg)
            elif escolha == "M":
                novo_modo = _toggle_modo_usuario()
                _lbl = ("Basico" if novo_modo == "iniciante" else "Avancado") \
                    if _lang() == "PT" else \
                    ("Basic" if novo_modo == "iniciante" else "Advanced")
                console.print(f"  [g]✓ Modo: {_lbl}[/g]" if _lang() == "PT"
                             else f"  [g]✓ Mode: {_lbl}[/g]")
                _pause()
            elif escolha == "I":
                _toggle_idioma()
            elif escolha == "R":
                _rodar_pipeline(cfg)
                _foto_menu = _foto_cfg(cfg)   # rodar grava o config.yaml
            elif escolha == "C":
                _abrir_confianca(None, cfg)
            elif escolha == "?":
                _ajuda_menu_principal(cfg)
            elif escolha == "Q":
                if not _confirmar_saida(cfg, _foto_menu):
                    continue
                _exibir_despedida()
                break
            elif not _aviso_tecla_antiga(escolha):
                console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()
            if _YAML_TOCADO[0]:
                _YAML_TOCADO[0] = False
                _foto_menu = _foto_cfg(cfg)
        except _SairDoPrograma:
            if _confirmar_saida(cfg, _foto_menu):
                _exibir_despedida()
                break


if __name__ == "__main__":
    main()
