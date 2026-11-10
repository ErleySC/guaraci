"""
dados_io.py — Parsing de JCAMP-DX/ASDF e carregamento de dados (.dx, CSV,
sintético) para o pipeline quimiométrico.

Extraído de pipeline.py como parte da modularização (Fase H). Depende de
Config só para type hints (import guardado por TYPE_CHECKING, para não criar
import circular com pipeline.py, que importa este módulo). pipeline.py
reexporta estes nomes, então `pipeline.load_data(...)`,
`pipeline.parse_title(...)` etc. continuam funcionando sem alteração.
"""
from __future__ import annotations

import glob
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from guaraci.io_registry import register_reader, get_reader

if TYPE_CHECKING:
    from guaraci.pipeline import Config

__all__ = [
    "adulterant_from_mae_id",
    "session_from_mae_id",
    "parse_title",
    "extract_dx_title",
    "kennard_stone",
    "kennard_stone_split",
    "kennard_stone_split_group_aware",
    "duplex_split",
    "duplex_split_group_aware",
    "spxy_split",
    "spxy_split_group_aware",
    "generate_synthetic_data",
    "load_csv",
    "load_csv_com_grupos",
    "carregar_mapa_grupos",
    "atribuir_grupos",
    "parse_dx",
    "parse_spectrum",
    "sanitizar_metadados",
    "prescan_dx",
    "load_dx",
    "load_data",
    "definir_correcoes_locais",
    "resumo_correcoes_locais",
    "garantia_agrupamento_prevista",
    "chave_unidade_validacao",
    "CODIGO_ESPECIE",
    "ADULTERANTE_NOME",
]

# =========================================================================
#  parse_title v3 — metadata extraction from ##TITLE= JCAMP-DX
#  Expected format (lab naming convention, JCAMP-DX titles):
#      PURE:        {COD}-{DD-MM-YYYY}_T{N}
#      ADULTERATED: {COD}-{DD-MM-YYYY}-AD-{A|M|S}-{N,NN}%_T{N}
# =========================================================================

#: Codigo do titulo/arquivo -> nome legivel da classe. VAZIO por padrao: a codificacao e' do laboratorio de quem usa o programa, nao do
#: produto. Cadastre os seus codigos no menu "Codificacao" (um por vez ou importando um CSV `codigo,classe`); codigo nao cadastrado
#: vira o proprio codigo como rotulo e e' reportado como "nao mapeado".
CODIGO_ESPECIE: Dict[str, str] = {}
#: Letra do token de adulterante (A, M ou S, como a regex do titulo aceita) -> nome legivel. Neutro por padrao; o usuario ou o perfil
#: (`MatrixProfile.codigos_adulterante`) declara o seu.
ADULTERANTE_NOME: Dict[str, str] = {"A": "adulterante A", "M": "adulterante M", "S": "adulterante S"}


def adulterant_from_mae_id(mae_id: Optional[str],
                            mapa_adulterante: Optional[Dict[str, str]] = None,
                            ) -> Optional[str]:
    """Nome do adulterante a partir do mae_id, ou None.

    O mae_id de uma amostra ADULTERADA termina no token '{letra}{teor}':
        real       'ESB-03-01-2099-A1.03' -> 'A' -> 'adulterante A' (mapa default)
        sintetico  'ESA-S05.00'           -> 'S' -> 'adulterante S' (mapa default)
    Amostras PURAS ('ESB-03-01-2099') e orfaos ('orfao_...') nao tem esse
    token e retornam None. Deriva o adulterante do mae_id (que sobrevive
    alinhado a validate_input), evitando desalinhamento com metadados_df.

    `mapa_adulterante` (Passo 122, INSTRUCAO_HIPOTESE_D_IDENTIFICACAO_
    GENERICA.md): letra -> nome legivel, default `ADULTERANTE_NOME`
    (comportamento historico, retrocompativel -- nenhum `.joblib`
    persistido muda de forma). Uma matriz nova pode declarar seu PROPRIO
    mapa via `MatrixProfile.codigos_adulterante` em vez de editar este
    modulo -- so' o MAPA letra->nome e' parametrizavel, a ESTRUTURA do
    token (1 letra + digitos no ultimo segmento) continua fixa (e' a
    convencao de `mae_id` deste projeto, nao da matriz)."""
    if mae_id is None:
        return None
    mapa = mapa_adulterante if mapa_adulterante is not None else ADULTERANTE_NOME
    ultimo = str(mae_id).split("-")[-1]
    m = re.match(r"^([A-Za-z])[0-9]", ultimo)
    if not m:
        return None
    letra = m.group(1).upper()
    # So' letras REGISTRADAS no mapa sao adulterantes reais; qualquer
    # outra letra+digito (ex.: o ponto puro sintetico 'ESA-P00') NAO e
    # adulterante -> None.
    if letra not in mapa:
        return None
    return mapa[letra]


def session_from_mae_id(mae_id: Optional[str]) -> Optional[str]:
    """Identificador de SESSAO DE COLETA (especie+data), removendo o token
    de teor/adulterante do final do mae_id.

    ACHADO (Bloco 9b, medido contra o dataset real em 2026-08-25): o
    `mae_id` de uma amostra ADULTERADA e' UM POR NIVEL DE TEOR, nao um por
    sessao de coleta -- ex., os 15 mae_id de 'classe A x adulterante A'
    ('ESA-04-01-2099-A1.05', 'ESA-04-01-2099-A2.11', ...) sao 15 DILUICOES
    de UMA UNICA sessao (mesma data '04-01-2099'), nao 15 preparos
    independentes. Contar `mae_id` bruto como "grupo" para calibracao
    conformal (ver `identificacao.py`) infla `n` de 1 sessao real para ate'
    15 -- pseudo-replicacao que violaria a garantia de cobertura. Removendo
    o token final (o mesmo que `adulterant_from_mae_id` identifica), sobra
    a sessao real: reproduz exatamente os 36 casos de 1 sessao + 2 casos de
    2 sessoes (classe A/adulterante S, classe K/adulterante A) ja documentados em
    MANUAL.md.

    Amostras PURAS (sem token de teor) devolvem o proprio `mae_id` --
    nao ha' nada a remover, e cada amostra pura ja e' sua propria sessao
    do ponto de vista deste agrupamento.
    """
    if mae_id is None:
        return None
    partes = str(mae_id).split("-")
    ultimo = partes[-1]
    if re.match(r"^[A-Za-z][0-9]", ultimo) and len(partes) > 1:
        return "-".join(partes[:-1])
    return str(mae_id)

# Regex robust to deviations found in the real reference dataset:
#   - surrounding whitespace                   "## TITLE= ESD-..."
#   - separator after COD/DATE: "-" or "_"     "XXX_DD-MM-YYYY_AD-S-..."
#   - optional separator before T              "...%T_3"  (no '-' or '_')
#   - Triplicate: "T1" or "T_1"
#   - optional decimal content                 "11%"  /  "1,1%"  /  "10,52%"
#   - "%%" sign (typo)                         "...4,13%%_T1"
_RE_TITLE = re.compile(
    r"^\s*"
    r"(?P<cod>[A-Z]{2,4})"
    r"[-_](?P<data>\d{2}-\d{2}-\d{4})"
    r"(?P<adulteracao>(?:[-_]AD-[AMS]-\d+(?:[.,]\d+)?%%?)?)"
    r"[-_]?T_?(?P<trip>[123])"
    r"\s*$"
)
# Accepts decimal with comma OR period (11,11% and 11.11% coexist in the dataset)
_RE_ADULT = re.compile(r"[-_]AD-([AMS])-(\d+(?:[.,]\d+)?)%%?")

# =========================================================================
#  Correcoes de metadado especificas do dataset -- CARREGADAS DE ARQUIVO
#  EXTERNO, fora da arvore versionada.
#
#  POR QUE FORA DO REPOSITORIO (decisao de 2026-08-17): as tabelas abaixo
#  sao METADADO DE AMOSTRA de um dataset de terceiro nao publicado --
#  codigo de especie, data de coleta, adulterante e teor de amostras
#  especificas. Este repositorio e' PUBLICO, e a titularidade/distribuicao
#  desses dados e' assunto em aberto com a detentora do dataset. Publicar
#  identificadores de amostra seria distribuir metadado do dataset pela
#  porta dos fundos, mesmo sem nenhum espectro junto.
#
#  O CODIGO (a logica de correcao) fica versionado; os DADOS (quais
#  amostras, com que rotulo) vivem na pasta local de correcoes (`_DIR_LOCAL`,
#  no diretorio do usuario, fora do repositorio). Quem nao tiver
#  o arquivo simplesmente nao aplica correcao nenhuma -- o parser continua
#  funcionando, so' sem os casos particulares deste dataset.
#
#  Formato (CSV, ';' como separador, UTF-8) -- ver
#  o cabecalho de cada CSV para o registro da decisao:
#
#    correcoes_titulo.csv    title;cod;data;adulterante;teor;trip
#    pureza_indeterminada.csv title;motivo
#    alias_mae_id.csv        mae_id_origem;mae_id_destino
# =========================================================================

_DIR_LOCAL = Path(os.path.expanduser("~")) / ".guaraci_local"


def _ler_csv_local(nome: str, colunas: int) -> List[List[str]]:
    """Le um CSV de correcao da pasta local `_DIR_LOCAL`, ou devolve [].

    Arquivo AUSENTE e' silencioso e legitimo (outra maquina, outro
    dataset). Arquivo PRESENTE mas malformado levanta erro claro -- nunca
    aplica correcao pela metade nem ignora em silencio, porque o efeito
    seria um rotulo de pureza errado entrando na analise.
    """
    caminho = _DIR_LOCAL / nome
    if not caminho.is_file():
        return []
    linhas: List[List[str]] = []
    try:
        texto = caminho.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(
            f"Correcoes de metadado ilegiveis em {caminho}: {e}. "
            f"Corrija o arquivo ou remova-o para rodar sem correcoes."
        ) from e
    for i, linha in enumerate(texto.splitlines(), 1):
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        partes = linha.split(";")
        if len(partes) != colunas:
            raise RuntimeError(
                f"{caminho}, linha {i}: esperadas {colunas} colunas "
                f"separadas por ';', encontradas {len(partes)}. "
                f"Uma correcao de rotulo aplicada pela metade produz "
                f"amostra com pureza errada -- abortando em vez de "
                f"adivinhar.")
        linhas.append([p.strip() for p in partes])
    return linhas


def _carregar_correcoes_titulo() -> Dict[str, Dict[str, Any]]:
    """title -> dict(cod, data, adulterante, teor, trip).

    TITLEs com erro de digitacao que fazem `_RE_TITLE` nao casar. Sem a
    correcao o parser cai no fallback por nome de arquivo, que tambem
    falha em extrair o teor, e o espectro entra como PURO (conc=0.0)
    quando e' ADULTERADO -- contaminando o treino one-class do DD-SIMCA
    daquela especie (achado A2-2).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for t, cod, data, adult, teor, trip in _ler_csv_local(
            "correcoes_titulo.csv", 6):
        out[t] = dict(cod=cod, data=data, adulterante=adult,
                      teor=float(teor), trip=int(trip))
    return out


def _carregar_pureza_indeterminada() -> Dict[str, str]:
    """title -> motivo. Espectros cuja PUREZA nao e' determinavel por
    nenhum caminho; excluidos do carregamento com aviso nominal, para nao
    entrarem como falso 'puro' (achado A2-2, 2a parte)."""
    return {t: motivo for t, motivo in _ler_csv_local(
        "pureza_indeterminada.csv", 2)}


def _carregar_alias_mae_id() -> Dict[str, str]:
    """mae_id_origem -> mae_id_destino.

    `mae_id = cod + data` separa em grupos distintos replicas da mesma
    amostra fisica lidas em datas diferentes -- o GroupKFold entao as
    trata como independentes, que e' o vazamento que o projeto existe
    para impedir (achado A2-1). Cada alias e' decisao metodologica
    DOCUMENTADA, nao fato verificado: ver o motivo registrado no proprio
    CSV, que fica fora da arvore versionada.
    """
    return {o: d for o, d in _ler_csv_local("alias_mae_id.csv", 2)}


# Estes tres dicionarios comecam VAZIOS. Antes eram preenchidos na
# IMPORTACAO a partir da pasta local `_DIR_LOCAL`, reclassificando rotulos e mae_id
# sem nada no log, no manifesto ou no model card -- o mesmo dado bruto dava
# rotulos diferentes em outra maquina. Agora `definir_correcoes_locais(True)`
# (opcao explicita: `apply_local_corrections` / GUARACI_APLICAR_CORRECOES_
# LOCAIS=1) os preenche E registra o que foi aplicado.
_CORRECOES_TITLE_CONHECIDAS: Dict[str, Dict[str, Any]] = {}
_TITLES_PUREZA_INDETERMINADA: Dict[str, str] = {}
_ALIAS_MAE_ID: Dict[str, str] = {}
_REGISTRO_CORRECOES: Dict[str, Any] = {"ativo": False, "arquivos": []}
_CONTAGEM_CORRECOES: Dict[str, int] = {
    "titulos_corrigidos": 0, "alias_aplicados": 0, "pureza_excluidos": 0}
_ARQUIVOS_CORRECOES = ("correcoes_titulo.csv", "pureza_indeterminada.csv",
                       "alias_mae_id.csv")


def _sha256_arquivo(caminho: Path) -> str:
    import hashlib
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def definir_correcoes_locais(ativar: bool) -> None:
    """Liga/desliga as correcoes de metadado da pasta local `_DIR_LOCAL` e ZERA a
    contagem de amostras alteradas. Ao ligar, registra cada arquivo lido
    (nome, sha256, n de linhas) -- ver `resumo_correcoes_locais`."""
    _CORRECOES_TITLE_CONHECIDAS.clear()
    _TITLES_PUREZA_INDETERMINADA.clear()
    _ALIAS_MAE_ID.clear()
    for k in _CONTAGEM_CORRECOES:
        _CONTAGEM_CORRECOES[k] = 0
    _REGISTRO_CORRECOES["ativo"] = bool(ativar)
    _REGISTRO_CORRECOES["arquivos"] = []
    if not ativar:
        return
    _CORRECOES_TITLE_CONHECIDAS.update(_carregar_correcoes_titulo())
    _TITLES_PUREZA_INDETERMINADA.update(_carregar_pureza_indeterminada())
    _ALIAS_MAE_ID.update(_carregar_alias_mae_id())
    _dicts: List[Dict[Any, Any]] = [
        _CORRECOES_TITLE_CONHECIDAS, _TITLES_PUREZA_INDETERMINADA, _ALIAS_MAE_ID]
    for nome, dic in zip(_ARQUIVOS_CORRECOES, _dicts):
        caminho = _DIR_LOCAL / nome
        if caminho.is_file():
            _REGISTRO_CORRECOES["arquivos"].append({
                "arquivo": nome, "sha256": _sha256_arquivo(caminho),
                "n_linhas": len(dic)})


def resumo_correcoes_locais() -> Dict[str, Any]:
    """O que as correcoes locais fizeram NESTA carga: {ativo, arquivos:
    [{arquivo, sha256, n_linhas}], titulos_corrigidos, alias_aplicados,
    pureza_excluidos}. Sem `ativo`, tudo zero."""
    return {"ativo": bool(_REGISTRO_CORRECOES["ativo"]),
            "arquivos": [dict(a) for a in _REGISTRO_CORRECOES["arquivos"]],
            **{k: int(v) for k, v in _CONTAGEM_CORRECOES.items()}}


def _ativar_correcoes_por_ambiente() -> bool:
    """Opt-in por ambiente (scripts do autor sem Config): GUARACI_APLICAR_CORRECOES_LOCAIS=1 liga as correcoes locais."""
    if os.environ.get("GUARACI_APLICAR_CORRECOES_LOCAIS", "").strip().lower() in ("1", "true", "yes", "on", "sim"):
        definir_correcoes_locais(True)
        return True
    return False


_ativar_correcoes_por_ambiente()        # na importacao


def parse_title(title: str) -> Optional[Dict[str, Any]]:
    """JCAMP-DX TITLE parser. Returns complete dict or None if invalid.

    mae_id field: uniquely identifies the physical sampling point.
    Triplicates T1/T2/T3 of the same point share mae_id, enabling
    GroupKFold/GroupShuffleSplit to prevent replica leakage.

    mae_id format:
        Pure:        'ESB-03-01-2099'
        Adulterated: 'ESB-03-01-2099-A1.03'  (content always 2 decimal places)

    Correcoes conhecidas (achado A2-2, ver `_CORRECOES_TITLE_CONHECIDAS`):
    5 TITLEs com erro de digitacao verificado sao reconhecidos por
    correspondencia EXATA antes do regex geral, para nao entrarem no
    dataset como amostra pura por engano.
    """
    titulo_limpo = title.strip()
    if titulo_limpo in _CORRECOES_TITLE_CONHECIDAS:
        _CONTAGEM_CORRECOES["titulos_corrigidos"] += 1
        c = _CORRECOES_TITLE_CONHECIDAS[titulo_limpo]
        cod, data, trip = c["cod"], c["data"], c["trip"]
        adulterante, teor = c["adulterante"], c["teor"]
        adulterante_nome = ADULTERANTE_NOME.get(adulterante, adulterante)
        mae_id = f"{cod}-{data}-{adulterante}{teor:.2f}"
        if mae_id in _ALIAS_MAE_ID:
            _CONTAGEM_CORRECOES["alias_aplicados"] += 1
        mae_id = _ALIAS_MAE_ID.get(mae_id, mae_id)
        return {
            "title_original":   titulo_limpo,
            "cod":              cod,
            "especie":          CODIGO_ESPECIE.get(cod, cod),
            "cod_conhecido":    cod in CODIGO_ESPECIE,
            "data":             data,
            "puro":             False,
            "adulterante":      adulterante,
            "adulterante_nome": adulterante_nome,
            "teor":             teor,
            "triplicata":       trip,
            "mae_id":           mae_id,
        }

    m = _RE_TITLE.match(titulo_limpo)
    if not m:
        return None
    cod  = m.group("cod").upper()
    data = m.group("data")
    trip = int(m.group("trip"))
    am   = _RE_ADULT.match(m.group("adulteracao"))
    # sem anotacao: os 3 nomes ja foram tipados no ramo das correcoes
    # conhecidas acima (mypy no-redef); aqui so' recebem None/valor.
    adulterante = None
    teor = None
    adulterante_nome = None
    if am:
        adulterante = str(am.group(1))
        teor = float(str(am.group(2)).replace(",", "."))
        adulterante_nome = ADULTERANTE_NOME.get(adulterante, adulterante)
        if teor <= 0:
            return None
    mae_id = (f"{cod}-{data}-{adulterante}{teor:.2f}"
              if adulterante is not None and teor is not None
              else f"{cod}-{data}")
    # Unificacao verificada de replicas separadas por data (achado A2-1) --
    # ver `_ALIAS_MAE_ID` para o caso e a ressalva.
    if mae_id in _ALIAS_MAE_ID:
        _CONTAGEM_CORRECOES["alias_aplicados"] += 1
    mae_id = _ALIAS_MAE_ID.get(mae_id, mae_id)
    return {
        "title_original":   title.strip(),
        "cod":              cod,
        "especie":          CODIGO_ESPECIE.get(cod, cod),
        "cod_conhecido":    cod in CODIGO_ESPECIE,
        "data":             data,
        "puro":             adulterante is None,
        "adulterante":      adulterante,
        "adulterante_nome": adulterante_nome,
        "teor":             teor,
        "triplicata":       trip,
        "mae_id":           mae_id,
    }


def extract_dx_title(caminho: str) -> Optional[str]:
    """Extracts the ##TITLE= line without loading all 8192 spectral points."""
    try:
        with open(caminho, "r", encoding="latin-1", errors="replace") as f:
            for linha in f:
                linha = linha.strip()
                if linha.startswith("##TITLE="):
                    return linha[len("##TITLE="):]
                if linha.startswith("##XYDATA") or linha.startswith("##XYPOINTS"):
                    break
    except OSError as _e_title:   # arquivo ilegivel/ausente/permissao
        logging.getLogger(__name__).debug(
            "extract_dx_title('%s'): %s", caminho, _e_title)
    return None


# =========================================================================
#  Data loading
# =========================================================================

def kennard_stone(X: np.ndarray, n_selecionar: int) -> np.ndarray:
    """Selecao de amostras de Kennard & Stone (1969), Technometrics 11:137-148.

    Escolhe `n_selecionar` amostras que cobrem o espaco espectral de forma
    MAXIMAMENTE representativa (uniforme), em vez de aleatoria/estratificada:
    comeca pelas amostras mais extremas e vai sempre adicionando a amostra
    mais distante (em distancia euclidiana) do conjunto ja escolhido. E o
    padrao para dividir calibracao/validacao em calibracao multivariada — a
    validacao herda amostras dentro do span do treino (nao extrapola) e o
    treino cobre bem as bordas do espaco.

    Semente: para evitar a matriz de distancias n×n (custosa em memoria para
    milhares de espectros), o par inicial e aproximado pelo ponto mais distante
    do centroide e, em seguida, o mais distante desse — variante eficiente e
    comum de KS. As iteracoes seguintes sao exatas (distancia minima ao
    conjunto selecionado, atualizada incrementalmente).

    Retorna os indices selecionados, na ORDEM de selecao (os primeiros sao os
    mais extremos), o que permite usar um prefixo como subconjunto menor.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    n_sel = int(min(max(n_selecionar, 1), n))

    centroide = X.mean(axis=0)
    i1 = int(np.argmax(np.sum((X - centroide) ** 2, axis=1)))
    selecionados = [i1]
    min_dist = np.sqrt(np.sum((X - X[i1]) ** 2, axis=1))

    if n_sel >= 2:
        i2 = int(np.argmax(min_dist))
        selecionados.append(i2)
        min_dist = np.minimum(min_dist,
                              np.sqrt(np.sum((X - X[i2]) ** 2, axis=1)))

    while len(selecionados) < n_sel:
        candidato_dist = min_dist.copy()
        candidato_dist[selecionados] = -np.inf   # nunca reescolher
        prox = int(np.argmax(candidato_dist))
        selecionados.append(prox)
        min_dist = np.minimum(min_dist,
                              np.sqrt(np.sum((X - X[prox]) ** 2, axis=1)))

    return np.array(selecionados, dtype=int)


def kennard_stone_split(X: np.ndarray, frac_treino: float = 0.7
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Divisao calibracao/validacao por Kennard-Stone: o treino recebe as
    `frac_treino` amostras mais representativas (bordas + cobertura uniforme);
    o resto vai para validacao. Retorna (idx_treino, idx_val), ambos ordenados.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    n_treino = int(round(max(0.0, min(1.0, frac_treino)) * n))
    n_treino = int(min(max(n_treino, 1), n)) if n > 0 else 0
    ordem = kennard_stone(X, n)          # ranking completo por KS
    idx_treino = np.sort(ordem[:n_treino])
    idx_val = np.sort(ordem[n_treino:])
    return idx_treino, idx_val


def _avisar_poucos_grupos(nome: str, mae_subset) -> None:
    """m-03 do parecer de 2026-09-25: os splits "group-aware" (KS/Duplex/SPXY)
    so' agrupam com >=4 grupos; com menos, caem no split por AMOSTRA e podem
    separar replicas da mesma amostra fisica entre calibracao e validacao --
    o vazamento que o projeto existe para impedir. Antes isso era silencioso."""
    if mae_subset is None:
        return
    n = len(np.unique(mae_subset))
    if 0 < n < 4:
        msg = (f"[AVISO] {nome}: so' {n} grupo(s) (mae_id) -- abaixo de 4 o "
               f"split cai em nivel de AMOSTRA e pode separar replicas da mesma "
               f"amostra fisica entre calibracao e validacao (vazamento).")
        print(msg)
        logging.getLogger(__name__).warning(msg)


def kennard_stone_split_group_aware(
        X: np.ndarray, mae_subset: Optional[np.ndarray], frac_cal: float
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Split calibracao/validacao por Kennard-Stone (1969), group-aware.

    Kennard-Stone escolhe as amostras que cobrem o espaco espectral de
    forma maximamente representativa (bordas + cobertura uniforme), em vez
    de aleatoria -- o padrao classico p/ dividir cal/val em calibracao
    multivariada. Com mae_id disponivel (>=4 grupos), colapsa cada grupo de
    replicas fisicas (T1/T2/T3) num espectro MEDIO antes de rodar KS (no
    nivel de GRUPO, nao de amostra individual) e depois expande de volta
    para os indices de amostra -- preserva o invariante do projeto de nunca
    separar replicas fisicas entre calibracao e validacao. Sem mae_id
    suficiente, roda KS diretamente por amostra.

    Usada por `pipeline.pls_regression_by_species`/bloco de regressao
    pooled E por `avaliacao_modelos.benchmark_regression_by_species` (mesmo
    split reproduzido deterministicamente nos dois lugares -- comparacao
    apples-to-apples entre PLS-R e os modelos de benchmark).
    """
    _avisar_poucos_grupos("kennard_stone_split_group_aware", mae_subset)
    if mae_subset is not None and len(np.unique(mae_subset)) >= 4:
        grupos_unicos = np.unique(mae_subset)
        X_grupo = np.array([X[mae_subset == g].mean(axis=0)
                             for g in grupos_unicos])
        idx_treino_g, _idx_val_g = kennard_stone_split(
            X_grupo, frac_treino=frac_cal)
        grupos_treino = set(grupos_unicos[idx_treino_g].tolist())
        mask_treino = np.isin(mae_subset, list(grupos_treino))
        return np.where(mask_treino)[0], np.where(~mask_treino)[0]
    return kennard_stone_split(X, frac_treino=frac_cal)


def _split_cal_val_por_especie(
        X_c: np.ndarray, Y_c: np.ndarray, mae_c: Optional[np.ndarray],
        cfg: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Split calibracao/validacao (group-aware) de UMA especie -- FONTE
    UNICA da decisao, usada por `pipeline.pls_regression_by_species` e por
    `avaliacao_modelos.benchmark_regression_by_species` (as duas copias
    inline, ~36 linhas identicas, foram achadas pela varredura de
    duplicacao de 2026-09-19; a divergencia entre elas quebraria a
    comparacao apples-to-apples PLS-R x benchmark sem nenhum aviso).

    Regra: Kennard-Stone group-aware (`cfg.cal_val_split ==
    "kennard_stone"`); senao GroupShuffleSplit por `mae_id` (>=4 grupos);
    senao permutacao simples com `cfg.seed`. Devolve `None` quando a
    especie nao sustenta o split (ValueError/IndexError do splitter, ou
    menos de 4 amostras de calibracao / 2 de validacao) -- o chamador
    pula a especie."""
    from sklearn.model_selection import GroupShuffleSplit
    try:
        if cfg.cal_val_split == "kennard_stone":
            ic, iv = kennard_stone_split_group_aware(X_c, mae_c, cfg.frac_cal)
        elif mae_c is not None and len(np.unique(mae_c)) >= 4:
            gss = GroupShuffleSplit(n_splits=1, train_size=cfg.frac_cal,
                                    random_state=cfg.seed)
            ic, iv = next(gss.split(X_c, Y_c, groups=mae_c))
        else:
            rng = np.random.default_rng(cfg.seed)
            perm = rng.permutation(len(Y_c))
            ncal = max(2, int(cfg.frac_cal * len(Y_c)))
            ic, iv = perm[:ncal], perm[ncal:]
    except (ValueError, IndexError):
        return None   # especie com amostras/grupos insuficientes p/ o split
    if len(ic) < 4 or len(iv) < 2:
        return None
    return ic, iv


def duplex_split(X: np.ndarray, frac_treino: float = 0.5
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Split calibracao/validacao por DUPLEX (Snee, 1977, "Validation of
    Regression Models: Methods and Examples", Technometrics 19(4):415-428,
    DOI 10.1080/00401706.1977.10489581 -- verificado no Crossref em
    2026-08-27).

    Diferenca p/ Kennard-Stone: KS enche PRIMEIRO o treino inteiro, so'
    depois o que sobra vai p/ validacao -- a validacao herda qualquer
    vies que sobrar. DUPLEX cresce os DOIS conjuntos EM PARALELO,
    alternando (cada novo ponto vai pro conjunto que estiver mais atras da
    proporcao alvo `frac_treino`, dentre os que sobraram, o mais distante
    do PROPRIO conjunto que esta recebendo) -- os dois ficam
    representativos do espaco, nao so' o treino.

    Semente aproximada (mesma tecnica de `kennard_stone`, evita matriz de
    distancias n×n): o par semente do treino e' o ponto mais distante do
    centroide + o mais distante desse; o par semente da validacao e' a
    MESMA logica aplicada so' aos pontos restantes.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int), np.array([], dtype=int)

    def _semente(idx_pool: np.ndarray) -> List[int]:
        Xp = X[idx_pool]
        centro = Xp.mean(axis=0)
        p1 = int(idx_pool[np.argmax(np.sum((Xp - centro) ** 2, axis=1))])
        dist_p1 = np.sqrt(np.sum((Xp - X[p1]) ** 2, axis=1))
        p2 = int(idx_pool[np.argmax(dist_p1)])
        return [p1, p2] if p2 != p1 else [p1]

    treino = _semente(np.arange(n))
    restantes = set(range(n)) - set(treino)
    val = _semente(np.array(sorted(restantes))) if restantes else []
    restantes -= set(val)

    def _min_dist_a(conjunto: List[int]) -> np.ndarray:
        if not conjunto:
            return np.full(n, np.inf)
        d = np.sqrt(np.sum((X - X[conjunto[0]]) ** 2, axis=1))
        for k in conjunto[1:]:
            d = np.minimum(d, np.sqrt(np.sum((X - X[k]) ** 2, axis=1)))
        return d

    min_dist_treino = _min_dist_a(treino)
    min_dist_val = _min_dist_a(val)

    while restantes:
        prop_treino = len(treino) / (len(treino) + len(val))
        candidatos = np.array(sorted(restantes))
        if prop_treino < frac_treino:
            alvo, min_dist = treino, min_dist_treino
        else:
            alvo, min_dist = val, min_dist_val
        escolhido = int(candidatos[np.argmax(min_dist[candidatos])])
        alvo.append(escolhido)
        restantes.discard(escolhido)
        d_escolhido = np.sqrt(np.sum((X - X[escolhido]) ** 2, axis=1))
        min_dist_treino = np.minimum(min_dist_treino, d_escolhido)
        min_dist_val = np.minimum(min_dist_val, d_escolhido)

    return np.array(sorted(treino), dtype=int), np.array(sorted(val), dtype=int)


def duplex_split_group_aware(
        X: np.ndarray, mae_subset: Optional[np.ndarray], frac_cal: float
        ) -> Tuple[np.ndarray, np.ndarray]:
    """DUPLEX group-aware -- mesma logica de colapso por grupo/expansao de
    `kennard_stone_split_group_aware`, aplicada a `duplex_split`."""
    _avisar_poucos_grupos("duplex_split_group_aware", mae_subset)
    if mae_subset is not None and len(np.unique(mae_subset)) >= 4:
        grupos_unicos = np.unique(mae_subset)
        X_grupo = np.array([X[mae_subset == g].mean(axis=0)
                             for g in grupos_unicos])
        idx_treino_g, _idx_val_g = duplex_split(X_grupo, frac_treino=frac_cal)
        grupos_treino = set(grupos_unicos[idx_treino_g].tolist())
        mask_treino = np.isin(mae_subset, list(grupos_treino))
        return np.where(mask_treino)[0], np.where(~mask_treino)[0]
    return duplex_split(X, frac_treino=frac_cal)


def spxy_split(X: np.ndarray, y: np.ndarray, frac_treino: float = 0.7
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Split calibracao/validacao por SPXY -- Sample set Partitioning based
    on joint X-Y distances (Galvao et al., 2005, "A method for calibration
    and validation subset partitioning", Talanta 67(4):736-740, DOI
    10.1016/j.talanta.2005.03.025 -- verificado no Crossref em 2026-08-27).

    Mesmo algoritmo guloso de `kennard_stone`, mas a distancia usada para
    escolher o proximo ponto e' `d_x/max(d_x) + d_y/max(d_y)` -- KS puro
    cobre bem o espaco ESPECTRAL mas pode deixar o extremo do TEOR de fora
    do treino (ex.: a amostra com o maior teor do dataset cai por acaso na
    validacao); SPXY faz o treino cobrir X e y ao mesmo tempo. `y` pode ser
    1D (um alvo) ou 2D (varios alvos empilhados por coluna).

    Usa a matriz de distancias n×n completa (ao contrario de
    `kennard_stone`, que aproxima a semente para evitar isso) porque a
    normalizacao pelo MAXIMO de `d_x`/`d_y` e' parte da definicao do
    metodo -- precisa do maximo real, nao de uma aproximacao. Aceitavel
    para os tamanhos de dataset deste projeto (dezenas a poucas centenas
    de amostras); para datasets muito maiores isto viraria o gargalo.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int), np.array([], dtype=int)
    n_sel = int(min(max(round(frac_treino * n), 1), n))

    Dx = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    Dy = np.sqrt(((y[:, None, :] - y[None, :, :]) ** 2).sum(axis=2))
    max_dx = Dx.max()
    max_dy = Dy.max()
    D = Dx / max_dx if max_dx > 0 else Dx
    D = D + (Dy / max_dy if max_dy > 0 else Dy)
    np.fill_diagonal(D, -np.inf)   # nunca parear um ponto com ele mesmo

    i1, i2 = np.unravel_index(np.argmax(D), D.shape)
    selecionados = [int(i1), int(i2)]
    min_dist = np.minimum(D[i1], D[i2])

    while len(selecionados) < n_sel:
        candidato_dist = min_dist.copy()
        candidato_dist[selecionados] = -np.inf
        prox = int(np.argmax(candidato_dist))
        selecionados.append(prox)
        min_dist = np.minimum(min_dist, D[prox])

    idx_treino = np.sort(np.array(selecionados[:n_sel], dtype=int))
    idx_val = np.array(sorted(set(range(n)) - set(idx_treino.tolist())),
                        dtype=int)
    return idx_treino, idx_val


def spxy_split_group_aware(
        X: np.ndarray, y: np.ndarray, mae_subset: Optional[np.ndarray],
        frac_cal: float) -> Tuple[np.ndarray, np.ndarray]:
    """SPXY group-aware -- mesma logica de colapso por grupo/expansao de
    `kennard_stone_split_group_aware`, aplicada a `spxy_split` (`y` tambem
    colapsa por MEDIA do grupo, igual `X`)."""
    _avisar_poucos_grupos("spxy_split_group_aware", mae_subset)
    if mae_subset is not None and len(np.unique(mae_subset)) >= 4:
        grupos_unicos = np.unique(mae_subset)
        y_arr = np.asarray(y, dtype=float)
        X_grupo = np.array([X[mae_subset == g].mean(axis=0)
                             for g in grupos_unicos])
        y_grupo = np.array([y_arr[mae_subset == g].mean(axis=0)
                             for g in grupos_unicos])
        idx_treino_g, _idx_val_g = spxy_split(X_grupo, y_grupo,
                                              frac_treino=frac_cal)
        grupos_treino = set(grupos_unicos[idx_treino_g].tolist())
        mask_treino = np.isin(mae_subset, list(grupos_treino))
        return np.where(mask_treino)[0], np.where(~mask_treino)[0]
    return spxy_split(X, y, frac_treino=frac_cal)


def generate_synthetic_data(cfg: "Config"):
    """Gera espectros sinteticos de teste, incluindo REPLICAS FISICAS
    (n_synthetic_replicates por ponto amostral, como T1/T2/T3 do mesmo ponto real) e
    mae_id — sem isso, DD-SIMCA (N2) e as figuras de merito de regressao (N3)
    nunca tinham dados suficientes para treinar/estimar ruido em mode
    sintetico (so 1 amostra "pura" por especie, sem nocao de replica).

    mae_id usa um codigo de 3 letras POR ESPECIE (ESA/ESB/ESC) como prefixo —
    o mesmo formato de 3 letras maiusculas do dataset de referencia,
    do qual `executar()` deriva o numero de especies via prefixo."""
    cenario_nome = str(getattr(cfg, "synthetic_scenario", "classico") or "classico")
    if cenario_nome != "classico":
        return _gerar_sintetico_realista(cfg, cenario_nome)
    print("[INFO] Synthetic MODE — generating test spectra.")
    rng = np.random.default_rng(cfg.seed)
    # As bandas sao desenhadas num eixo-modelo (4000 -> 400); no retorno o eixo
    # e' remapeado LINEARMENTE para a faixa configurada [wn_min, wn_max], de modo
    # que o dado sintetico sempre cabe na faixa (B1 do parecer de 2026-09-25: a
    # faixa padrao 4000-10000 deixava 1 variavel do eixo 4000->400 e o primeiro
    # uso falhava). Com wn_min=400/wn_max=4000 o remapeamento e' a identidade.
    wavenumbers = np.linspace(4000, 400, cfg.n_synthetic_points)
    _wn_lo, _wn_hi = float(cfg.wn_min), float(cfg.wn_max)
    conc_base = np.linspace(0, 40, cfg.n_per_class)
    n_replicas = max(1, int(cfg.n_synthetic_replicates))

    def esp(c, p1, p2, ruido=0.015):
        frac = c / 100
        return ((1 - frac) * np.exp(-((wavenumbers - p1) ** 2) / (2 * 50 ** 2))
                + frac     * np.exp(-((wavenumbers - (p1 + 20)) ** 2) / (2 * 45 ** 2))
                + 0.6      * np.exp(-((wavenumbers - p2) ** 2) / (2 * 30 ** 2))
                + rng.normal(0, ruido, cfg.n_synthetic_points))

    params  = [(2900, 1740), (2850, 1650), (2960, 1710)]
    classes = ["Esp_A", "Esp_B", "Esp_C"]
    codigos = {"Esp_A": "ESA", "Esp_B": "ESB", "Esp_C": "ESC"}
    X_list, rot_list, conc_list, mae_list = [], [], [], []
    adults = [a.upper() for a in (getattr(cfg, "synthetic_adulterants", None) or ())]
    for (p1, p2), cls in zip(params, classes):
        cod = codigos[cls]
        if adults:
            # Puros (teor=0): 3 pontos amostrais independentes por especie
            # (mae_id 'ESA-P00'... sem token de adulterante -> lidos como puros).
            for i in range(3):
                mae_ponto = f"{cod}-P{i:02d}"
                for _ in range(n_replicas):
                    X_list.append(esp(0.0, p1, p2)); rot_list.append(cls)
                    conc_list.append(0.0); mae_list.append(mae_ponto)
            # Adulterados: cada adulterante = uma banda marcadora limpa,
            # proporcional ao teor (PLS quantifica), em posicao propria. A
            # FORCA do marcador difere por adulterante -> alguns combos ficam
            # quantificaveis (R2cv alto) e outros afogam no ruido (R2cv baixo),
            # gerando um heatmap com a mistura realista de aprovados/reprovados.
            desloc_por_adult = {"A": 0.0, "M": 8.0, "S": -8.0}
            forca_marcador   = {"S": 0.055, "M": 0.030, "A": 0.008}
            n_niveis = max(3, cfg.n_per_class // 2)
            for a in adults:
                dp = desloc_por_adult.get(a, 4.0)
                fm = forca_marcador.get(a, 0.02)
                banda = np.exp(-((wavenumbers - (p1 + dp)) ** 2) / (2 * 25 ** 2))
                for c in np.linspace(5, 40, n_niveis):
                    mae_ponto = f"{cod}-{a}{c:.2f}"   # 'ESA-S05.00'
                    for _ in range(n_replicas):
                        X_list.append(esp(0.0, p1, p2) + fm * float(c) * banda)
                        rot_list.append(cls); conc_list.append(float(c))
                        mae_list.append(mae_ponto)
        else:
            for i, c in enumerate(conc_base):
                mae_ponto = f"{cod}-{i:02d}"
                for _ in range(n_replicas):
                    X_list.append(esp(c, p1, p2))
                    rot_list.append(cls)
                    conc_list.append(c)
                    mae_list.append(mae_ponto)

    if _wn_hi > _wn_lo:
        wavenumbers = _wn_hi - (4000.0 - wavenumbers) / 3600.0 * (_wn_hi - _wn_lo)
    return (wavenumbers,
            np.array(X_list, dtype=float),
            np.array(rot_list, dtype=str),
            np.array(conc_list, dtype=float),
            np.array(mae_list, dtype=str))


def _gerar_sintetico_realista(cfg: "Config", nome: str):
    """Cenario realista de `simulador_espectral` (nir/raman/rmn), com o
    mesmo contrato de retorno do gerador classico."""
    from guaraci.simulador_espectral import cenario, decompor_variancia
    cen = cenario(nome)
    adults = [a.upper() for a in (getattr(cfg, "synthetic_adulterants", None) or ())]
    eixo, X, rot, conc, mae, lote = gerar_cenario_sintetico(
        cen, n_pontos=cfg.n_synthetic_points, n_por_classe=cfg.n_per_class,
        n_replicas=cfg.n_synthetic_replicates, adulterantes=adults,
        seed=cfg.seed)
    dv = decompor_variancia(X, mae)
    print(f"[INFO] Synthetic MODE — cenario '{nome}': {X.shape[0]} espectros, "
          f"{len(np.unique(mae))} amostras fisicas, {cen.n_lotes} lotes; "
          f"variancia dentro das amostras (replicas) = "
          f"{100 * dv['frac_dentro']:.1f}% do total.")
    _wn_lo, _wn_hi = float(cfg.wn_min), float(cfg.wn_max)
    if _wn_hi > _wn_lo:
        eixo = _wn_hi - (4000.0 - eixo) / 3600.0 * (_wn_hi - _wn_lo)
    return eixo, X, rot, conc, mae


def gerar_cenario_sintetico(*args, **kwargs):
    from guaraci.simulador_espectral import gerar_cenario
    return gerar_cenario(*args, **kwargs)


def _ler_tabela_csv(caminho) -> pd.DataFrame:
    """Le um CSV detectando o separador (`,` `;` tab) e, se os valores
    espectrais nao forem numericos, tentando decimal `,` (padrao pt-BR/Excel)
    -- o mesmo tratamento do CSV de predicao (`predicao.load_prediction_csv`),
    para que o CSV de TREINO aceite o proprio formato que o GUARACI grava. Erros viram `ValueError` acionaveis."""
    import csv as _csv_mod
    if not os.path.isfile(str(caminho)):
        raise FileNotFoundError(
            f"CSV nao encontrado: '{caminho}'. Confira o caminho em "
            f"arquivo_csv (modo csv).")

    from guaraci.csv_robusto import ler_csv

    def _ler(**kw):
        return ler_csv(caminho, **kw)

    try:
        df = _ler()
    except (pd.errors.EmptyDataError, _csv_mod.Error) as e:
        raise ValueError(
            f"O CSV '{os.path.basename(str(caminho))}' esta vazio (sem "
            f"cabecalho nem linhas). Confirme que o caminho aponta para o "
            f"arquivo certo.") from e
    except pd.errors.ParserError as e:
        raise ValueError(
            f"Nao foi possivel interpretar '{os.path.basename(str(caminho))}' "
            f"como CSV ({e}). Confirme que e' texto (nao um Excel renomeado) "
            f"e que todas as linhas tem o mesmo numero de colunas.") from e
    if df.empty or len(df.columns) == 0:
        raise ValueError(
            f"O CSV '{os.path.basename(str(caminho))}' tem so' cabecalho ou "
            f"nenhuma linha de dados.")
    # Valores com virgula decimal: as colunas "numericas" vieram como texto.
    obj_num = [c for c in df.columns
               if not pd.api.types.is_numeric_dtype(df[c]) and _nome_numerico(c)]
    if obj_num:
        try:
            df = _ler(decimal=",")
        except (pd.errors.ParserError, _csv_mod.Error, ValueError):
            pass
    return df


def _localizar_texto(bloco: pd.DataFrame, n: int = 3) -> str:
    """R2-11: diz ONDE esta o texto (linha do arquivo e coluna), nao so' o valor."""
    achados = []
    conv = bloco.apply(pd.to_numeric, errors="coerce")
    ruins = conv.isna() & bloco.notna()
    for i, c in zip(*np.where(ruins.values)):
        achados.append(f"linha {int(i) + 2}, coluna '{bloco.columns[c]}' = "
                       f"'{bloco.iat[int(i), int(c)]}'")
        if len(achados) >= n:
            break
    total = int(ruins.values.sum())
    return "; ".join(achados) + (f" (+{total - n} outras)" if total > n else "")


def _nome_numerico(nome) -> bool:
    try:
        float(str(nome).strip().replace(",", "."))
        return True
    except ValueError:
        return False


def _lista_colunas(df: pd.DataFrame, n: int = 12) -> str:
    cols = [str(c) for c in df.columns]
    resto = f", ... (+{len(cols) - n})" if len(cols) > n else ""
    return ", ".join(cols[:n]) + resto


def carregar_mapa_grupos(caminho: str) -> Tuple[str, Dict[str, str]]:
    """Le o arquivo de associacao amostra -> grupo (B4a do parecer de
    2026-09-25). CSV/TSV com CABECALHO e >=2 colunas: a PRIMEIRA nomeia a
    identificacao da amostra (nas fontes por arquivo: o nome do arquivo sem
    extensao / da pasta; no CSV de dados: o nome de uma coluna do proprio CSV)
    e a SEGUNDA e' o grupo (amostra fisica independente). Retorna
    (nome_da_coluna_id, {id: grupo}). Erra se houver id duplicado com grupos
    diferentes ou grupo vazio -- nunca "adivinha"."""
    df = _ler_tabela_csv(caminho)
    if df.shape[1] < 2:
        raise ValueError(
            f"O arquivo de grupos '{os.path.basename(str(caminho))}' precisa "
            f"de cabecalho e >=2 colunas (id da amostra, grupo); encontrei: "
            f"{_lista_colunas(df)}.")
    col_id, col_g = str(df.columns[0]), str(df.columns[1])
    ids = df.iloc[:, 0].astype(str).str.strip()
    grupos = df.iloc[:, 1]
    if grupos.isna().any() or (grupos.astype(str).str.strip() == "").any():
        raise ValueError(
            f"O arquivo de grupos tem linhas sem grupo (coluna '{col_g}'). "
            f"Toda amostra precisa de um grupo.")
    mapa: Dict[str, str] = {}
    for i, g in zip(ids, grupos.astype(str).str.strip()):
        if i in mapa and mapa[i] != g:
            raise ValueError(
                f"A amostra '{i}' aparece no arquivo de grupos com grupos "
                f"diferentes ('{mapa[i]}' e '{g}').")
        mapa[i] = g
    return col_id, mapa


def atribuir_grupos(nomes, mapa: Dict[str, str], origem: str = "") -> np.ndarray:
    """Grupo de cada amostra (`nomes`) segundo `mapa`; erra listando as que
    ficaram sem grupo (o pipeline nunca completa grupos em silencio)."""
    nomes = [str(n).strip() for n in nomes]
    faltam = [n for n in nomes if n not in mapa]
    if faltam:
        exemplo = ", ".join(faltam[:5]) + (" ..." if len(faltam) > 5 else "")
        raise ValueError(
            f"{len(faltam)} de {len(nomes)} amostras nao tem grupo no arquivo "
            f"de grupos{(' (' + origem + ')') if origem else ''}: {exemplo}. "
            f"Cada amostra precisa aparecer no arquivo (1a coluna = id).")
    return np.array([mapa[n] for n in nomes], dtype=str)


def _carregar_csv_completo(caminho, col_classe, col_conc, col_grupo: str = "",
                           arquivo_grupos: str = ""):
    """`load_csv` + fonte de GRUPO (B4a). Retorna
    (wavenumbers, X, rotulos, conc, mae_id_ou_None, nomes_ou_None); `nomes` so' existe
    quando os cabecalhos sao NOMES de variavel (eixo = posicao 1..k).

    Fontes de grupo (mutuamente exclusivas): `col_grupo` = coluna do proprio
    CSV; `arquivo_grupos` = CSV/TSV (id, grupo) cuja 1a coluna nomeia uma
    coluna do CSV de dados. Sem nenhuma das duas, `mae_id=None` e o pipeline
    declara a AUSENCIA de garantia (nunca finge protecao)."""
    print(f"[INFO] Loading CSV: {caminho}")
    if col_grupo and arquivo_grupos:
        raise ValueError(
            "Use so' UMA fonte de grupo: coluna_grupo (coluna do CSV) OU "
            "arquivo_grupos (arquivo id->grupo), nao as duas.")
    df = _ler_tabela_csv(caminho)
    if col_classe not in df.columns:
        raise ValueError(
            f"A coluna de classe '{col_classe}' nao existe no CSV. Colunas "
            f"encontradas: {_lista_colunas(df)}. Ajuste coluna_classe.")
    if col_conc and col_conc not in df.columns:
        raise ValueError(
            f"A coluna de concentracao '{col_conc}' nao existe no CSV. "
            f"Colunas encontradas: {_lista_colunas(df)}. Ajuste "
            f"coluna_concentracao (ou deixe vazio se nao ha' teor).")
    mae_id = None
    excluir = [c for c in (col_classe, col_conc) if c]
    if col_grupo:
        if col_grupo not in df.columns:
            raise ValueError(
                f"A coluna de grupo '{col_grupo}' nao existe no CSV. Colunas "
                f"encontradas: {_lista_colunas(df)}. Ajuste coluna_grupo.")
        if df[col_grupo].isna().any():
            raise ValueError(
                f"A coluna de grupo '{col_grupo}' tem valores vazios; toda "
                f"amostra precisa de um grupo.")
        mae_id = np.asarray(df[col_grupo].astype(str).str.strip().values, dtype=str)
        excluir.append(col_grupo)
    elif arquivo_grupos:
        col_id, mapa = carregar_mapa_grupos(arquivo_grupos)
        if col_id not in df.columns:
            raise ValueError(
                f"A 1a coluna do arquivo de grupos e' '{col_id}', mas o CSV de "
                f"dados nao tem essa coluna. Colunas do CSV: "
                f"{_lista_colunas(df)}.")
        mae_id = atribuir_grupos(df[col_id], mapa, origem=col_id)
        excluir.append(col_id)
    # R2-05: classe vazia/NaN virava a classe "nan" e "A " != "A" criava classes
    # fantasmas. Espacos nas pontas saem; linha sem classe e' erro, nao classe.
    _cls = df[col_classe]
    _vazias = _cls.isna() | (_cls.astype(str).str.strip() == "")
    if _vazias.any():
        _linhas = ", ".join(str(i + 2) for i in np.where(_vazias.values)[0][:8])
        raise ValueError(
            f"{int(_vazias.sum())} linha(s) do CSV sem valor na coluna de classe "
            f"'{col_classe}' (linhas {_linhas}{'...' if _vazias.sum() > 8 else ''} "
            f"do arquivo). Preencha o rotulo ou remova essas linhas.")
    rotulos = np.asarray(_cls.astype(str).str.strip().values, dtype=str)
    try:
        conc = np.asarray(df[col_conc].values, dtype=float) if col_conc else None
    except ValueError as e:
        raise ValueError(
            f"A coluna de concentracao '{col_conc}' tem valores nao "
            f"numericos ({e}).") from e
    cols_spec, nomes, ignoradas = _escolher_variaveis_csv(df, excluir)
    if ignoradas:
        print(f"[AVISO] Colunas nao numericas ignoradas (nao sao espectro): "
              f"{', '.join(ignoradas[:6])}{' ...' if len(ignoradas) > 6 else ''}")
    if len(cols_spec) < 2:
        raise ValueError(
            f"O CSV tem menos de 2 colunas numericas de variaveis (achadas: "
            f"{len(cols_spec)}). Cabecalhos numericos (numeros de onda, tempos) "
            f"viram o eixo; cabecalhos com NOMES (pH, composto_a...) viram "
            f"variaveis nomeadas. Colunas encontradas: {_lista_colunas(df)}.")
    try:
        X_raw = np.asarray(df[cols_spec].values, dtype=float)
    except ValueError as e:
        raise ValueError(
            "Ha' valores espectrais nao numericos no CSV (celulas com texto): "
            f"{_localizar_texto(df[cols_spec])}.") from e
    if nomes is None:
        wavenumbers = np.array([float(str(c).strip().replace(",", ".")) for c in cols_spec])
    else:
        wavenumbers = np.arange(1.0, len(cols_spec) + 1.0)
    return wavenumbers, X_raw, rotulos, conc, mae_id, nomes


#: colunas numericas de METADADO (teor, lote, replica...): nunca sao variaveis nomeadas
_META_NUMERICO = frozenset({"conc", "concentracao", "teor", "lote", "grupo", "rep", "replica",
                            "injecao", "ordem", "dia", "sessao", "id", "amostra", "y"})


def _eixo_monotono(nomes) -> bool:
    v = np.array([float(str(c).strip().replace(",", ".")) for c in nomes])
    d = np.diff(v)
    return len(v) >= 2 and (bool(np.all(d > 0)) or bool(np.all(d < 0)))


def _escolher_variaveis_csv(df, excluir):
    """Decide quais colunas do CSV sao as VARIAVEIS e se o eixo e' o proprio cabecalho.

    Devolve `(colunas, nomes, ignoradas)`; `nomes is None` = eixo real (cabecalhos
    numericos: numero de onda, tempo de retencao...); caso contrario `nomes` sao os
    cabecalhos como texto e o eixo e' a POSICAO 1..k (tabela de compostos, fisico-quimica,
    qualquer dado multivariado nao espectral).

    - so' cabecalhos numericos: espectro (eixo = cabecalho); colunas de nome (texto,
      observacao...) ficam de fora com aviso, como sempre.
    - so' cabecalhos nomeados: todas as colunas numericas (fora metadado) sao variaveis.
    - MISTO: se os cabecalhos numericos formam um bloco espectral (monotonico e ao menos
      4x mais numerosos que as colunas nomeadas), as nomeadas sao extras ignoradas com aviso;
      senao e' uma tabela com nomes e numeros misturados: TODAS as colunas numericas viram
      variaveis nomeadas, na ordem do arquivo, e o eixo e' a posicao (os numeros do
      cabecalho nunca viram eixo, que ficaria fora de ordem/corrompido)."""
    cand = [c for c in df.columns if c not in excluir]
    num_h = [c for c in cand if _nome_numerico(c)]
    nomeadas = [c for c in cand if not _nome_numerico(c)
                and str(c).strip().lower() not in _META_NUMERICO
                and pd.api.types.is_numeric_dtype(df[c])]
    ignoradas = [str(c) for c in cand if not _nome_numerico(c)]
    if num_h and not nomeadas:
        return num_h, None, ignoradas
    if num_h:
        espectral = (len(num_h) >= 4 * len(nomeadas) and len(num_h) >= 3
                     and _eixo_monotono(num_h))
        if espectral:
            return num_h, None, ignoradas
        tabela = [c for c in cand if (c in nomeadas)
                  or (_nome_numerico(c) and pd.api.types.is_numeric_dtype(df[c]))]
    else:
        tabela = nomeadas
    ignoradas = [str(c) for c in cand if c not in tabela]
    if len(tabela) >= 2:
        nomes = [str(c).strip() for c in tabela]
        print(f"[AVISO] Cabecalhos com nomes: {len(tabela)} colunas numericas lidas como "
              f"variaveis nomeadas (tabela de compostos/areas ou dado multivariado); eixo = "
              f"posicao da coluna: "
              + ", ".join(f"{i}={c}" for i, c in enumerate(nomes[:8], 1))
              + (" ..." if len(nomes) > 8 else ""))
        return tabela, nomes, ignoradas
    return tabela, None, ignoradas


def load_csv_com_grupos(caminho, col_classe, col_conc, col_grupo: str = "",
                        arquivo_grupos: str = ""):
    """Ver `_carregar_csv_completo`; devolve so' (eixo, X, rotulos, conc, mae_id)
    (assinatura publica estavel)."""
    return _carregar_csv_completo(caminho, col_classe, col_conc, col_grupo,
                                  arquivo_grupos)[:5]


def load_csv(caminho, col_classe, col_conc):
    """Compat: (wavenumbers, X, rotulos, conc). Ver `load_csv_com_grupos`."""
    return load_csv_com_grupos(caminho, col_classe, col_conc)[:4]


def _flush_asdf(y_raw, sign, digits, is_dif):
    if sign == 0:
        return
    num = int("".join(str(d) for d in digits)) if digits else 0
    val = sign * num
    if is_dif:
        val = (y_raw[-1] if y_raw else 0) + val
    y_raw.append(val)


def _decodificar_linha_asdf(s: str, SQZ: dict, DIF: dict, DUP: dict
                             ) -> Tuple[Optional[float], List[float]]:
    """Decodes an ASDF line '(X++(Y..Y))': returns (x_check, [y_raw]).

    x_check is the abscissa value (WITHOUT xfactor yet) that prefixes the line;
    used to anchor the block position on the global grid.
    Linhas AFFN/PAC (sem compressao) vao para `_decodificar_linha_affn`; a escolha e' por arquivo, em parse_dx."""
    i, n = 0, len(s)
    x_str = ""
    while i < n and (s[i].isdigit() or s[i] in "+-."):
        x_str += s[i]; i += 1
    try:
        x_check = float(x_str)
    except ValueError:
        return None, []

    y_raw: List[float] = []
    sign, is_dif = 0, False
    digits: List[int] = []
    while i < n:
        ch = s[i]; i += 1
        if ch in SQZ:
            _flush_asdf(y_raw, sign, digits, is_dif)
            v = SQZ[ch]; sign = 1 if v >= 0 else -1
            digits = [abs(v)]; is_dif = False
        elif ch in DIF:
            _flush_asdf(y_raw, sign, digits, is_dif)
            v = DIF[ch]; sign = 1 if v >= 0 else -1
            digits = [abs(v)]; is_dif = True
        elif ch in DUP:
            # DUP (JCAMP-DX 4.24) = numero TOTAL de ocorrencias do item anterior, contando ele mesmo; em forma DIF o
            # item repetido e' a DIFERENCA, nao o valor. A contagem pode ter mais digitos ('S2' = 12).
            delta = (sign * int("".join(str(d) for d in digits)) if digits else 0) if (is_dif and sign != 0) else 0
            _flush_asdf(y_raw, sign, digits, is_dif)
            rep = str(DUP[ch])
            while i < n and s[i].isdigit():
                rep += s[i]; i += 1
            for _ in range(int(rep) - 1):
                y_raw.append((y_raw[-1] if y_raw else 0) + delta)
            sign = 0
        elif ch.isdigit() and sign != 0:
            digits.append(int(ch))
    _flush_asdf(y_raw, sign, digits, is_dif)
    return x_check, y_raw


#: caracteres que so' existem na forma comprimida ASDF (SQZ/DIF/DUP). 'E'/'e' ficam de fora de proposito: tambem
#: aparecem no expoente de um numero AFFN ("1.2E-03").
_CHARS_ASDF = frozenset("@ABCDFGHIabcdfghi%JKLMNOPQRjklmnopqrSTUVWXYZs")
_REGEX_AFFN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _decodificar_linha_affn(s: str) -> Tuple[Optional[float], List[float]]:
    """Linha AFFN/PAC '(X++(Y..Y))' -- numeros em texto separados por espaco, virgula ou sinal (JCAMP-DX 4.24,
    McDonald & Wilks 1988, Appl. Spectrosc. 42:151): o 1o e' a abscissa, os demais sao Y."""
    tokens = _REGEX_AFFN.findall(s)
    if not tokens:
        return None, []
    return float(tokens[0]), [float(t) for t in tokens[1:]]




def _rotulo_ldr(linha: str) -> Tuple[str, str]:
    """("##NPOINTS = 8192") -> ("NPOINTS", "8192"). O rotulo de um LDR do JCAMP-DX e' comparado sem caixa e sem
    espacos, hifens, barras e sublinhados (norma 4.24, secao 4.1): "##NPOINTS = 8192", "##Npoints=8192" e
    "##N-POINTS=8192" sao o mesmo campo. Fora de um LDR ("##" ausente) devolve ("", "")."""
    if not linha.startswith("##"):
        return "", ""
    if "=" not in linha:                       # "##END" sem o sinal
        return re.sub(r"[\s\-/_]", "", linha[2:]).upper(), ""
    rotulo, _, valor = linha[2:].partition("=")
    return re.sub(r"[\s\-/_]", "", rotulo).upper(), valor.split("$$", 1)[0].strip()   # "$$ ..." e' comentario


def parse_dx(filepath):
    """JCAMP-DX parser for compressed '(X++(Y..Y))' format (ASDF).

    Robust axis reconstruction strategy:
      - The X axis is reconstructed as np.linspace(FIRSTX, LASTX, NPOINTS)
        using the header (authoritative), NEVER the encoded X values.
      - Each data line is ANCHORED by its check abscissa (x_check) to the
        correct index on the global grid. This is self-correcting: lines
        that over/under-decode are re-anchored by the next line (fixes the
        blind concatenation bug that used to lose points).
      - Residual gaps (NaN) are linearly interpolated.

    Fallback: if FIRSTX/LASTX/NPOINTS are missing, uses simple concatenation
    with encoded X (legacy mode, less reliable)."""
    SQZ, DIF, DUP = {"@": 0}, {"%": 0}, {}
    for i, c in enumerate("ABCDEFGHI", 1): SQZ[c] =  i
    for i, c in enumerate("abcdefghi", 1): SQZ[c] = -i
    for i, c in enumerate("JKLMNOPQR", 1): DIF[c] =  i
    for i, c in enumerate("jklmnopqr", 1): DIF[c] = -i
    for i, c in enumerate("STUVWXYZs", 1): DUP[c] =  i

    xfactor, yfactor, lendo_dados = 1.0, 1.0, False
    pares_xy = False                     # ##XYPOINTS=(XY..XY): pares x,y em texto
    firstx = lastx = None
    npoints: Optional[int] = None
    linhas_dados: List[str] = []

    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()

    vistos: Dict[str, int] = {}
    for line in lines:
        s = line.strip()
        rotulo, valor = _rotulo_ldr(s)
        if rotulo:
            vistos[rotulo] = vistos.get(rotulo, 0) + 1
        if   rotulo == "XFACTOR": xfactor = float(valor)
        elif rotulo == "YFACTOR": yfactor = float(valor)
        elif rotulo == "FIRSTX":  firstx  = float(valor)
        elif rotulo == "LASTX":   lastx   = float(valor)
        elif rotulo == "NPOINTS":
            try: npoints = int(float(valor))
            except ValueError: npoints = None
        elif rotulo in ("XYDATA", "XYPOINTS"):
            pares_xy = "(XY..XY)" in valor.replace(" ", "").upper()
            lendo_dados = True; continue
        elif rotulo == "END":      lendo_dados = False   # nao para: um 2o bloco apos o ##END e' contado (arquivo composto)
        elif s.startswith("##") and lendo_dados: lendo_dados = False
        elif lendo_dados and s:
            # "$$" abre um comentario ate o fim da linha (norma 4.24); sem tirar, "$$No of bytes of data = 23200" no
            # fim dos dados (o INTTODX escreve isso) tem letras de SQZ/DIF e fazia o arquivo AFFN ser lido como ASDF.
            s = s.split("$$", 1)[0].strip()
            if s:
                linhas_dados.append(s)

    nome_arq_dx = os.path.basename(filepath)
    if vistos.get("XYDATA", 0) + vistos.get("XYPOINTS", 0) > 1 or vistos.get("BLOCKS"):
        raise ValueError(
            f"{nome_arq_dx}: arquivo JCAMP-DX composto (varios blocos ##XYDATA=/##TITLE=): ler tudo como um espectro "
            f"misturaria os blocos. Exporte um espectro por arquivo.")
    if not linhas_dados and (vistos.get("PEAKTABLE") or vistos.get("NTUPLES")):
        raise ValueError(
            f"{nome_arq_dx}: o arquivo traz "
            f"{'uma tabela de picos (##PEAK TABLE=)' if vistos.get('PEAKTABLE') else 'dados NTUPLES (##NTUPLES=)'}, nao um "
            f"espectro continuo (##XYDATA=). O GUARACI le ##XYDATA=(X++(Y..Y)) e ##XYPOINTS=(XY..XY); exporte o espectro "
            f"nesse formato.")

    # Forma dos dados: ASDF comprimido (SQZ/DIF/DUP, ex. export do OPUS) ou AFFN/PAC em texto (OMNIC, Spectrum,
    # WiRE, LabSpec...). Antes so' o ASDF era lido: um .dx AFFN virava um espectro todo NaN, sem erro no parser.
    asdf = any(ch in _CHARS_ASDF for s in linhas_dados for ch in s)
    if pares_xy and not asdf:
        nums = [float(t) for s in linhas_dados for t in _REGEX_AFFN.findall(s)]
        if len(nums) < 4 or len(nums) % 2:
            raise ValueError(f"{os.path.basename(filepath)}: ##XYPOINTS=(XY..XY) sem pares x,y completos")
        return (np.asarray(nums[0::2], dtype=float) * xfactor, np.asarray(nums[1::2], dtype=float) * yfactor)

    def _decodificar(s: str) -> Tuple[Optional[float], List[float]]:
        return _decodificar_linha_asdf(s, SQZ, DIF, DUP) if asdf else _decodificar_linha_affn(s)

    # --- Robust reconstruction anchored by X-check ----------------------
    if (firstx is not None and lastx is not None and npoints
            and npoints > 1 and lastx != firstx):
        dx = (lastx - firstx) / (npoints - 1)
        Y = np.full(npoints, np.nan, dtype=float)
        decodificadas = [_decodificar(s) for s in linhas_dados]
        validas = [(xc, yr) for xc, yr in decodificadas if xc is not None and yr]
        # Se os valores lidos, na ordem do arquivo, somam exatamente NPOINTS, cada valor e' um ponto distinto (nao ha'
        # valor de checagem repetido, como na forma DIF) e a ORDEM e' a verdade: os rotulos de abscissa das linhas
        # servem so' de conferencia. Os arquivos de gas do NIST (AFFN, 6 valores por linha) trazem os rotulos de todas
        # as linhas a partir da 2a deslocados em um ponto; ancorar por eles perdia ate' um ponto por linha e
        # distorcia o espectro. Com a contagem diferente de NPOINTS, o ancoramento abaixo continua valendo.
        if validas and sum(len(yr) for _xc, yr in validas) == npoints:
            Y[:] = np.fromiter((v for _xc, yr in validas for v in yr), dtype=float, count=npoints) * yfactor
        else:
            for x_check, y_raw in validas:
                x_real = x_check * xfactor
                idx0 = int(round((x_real - firstx) / dx))
                for j, yv in enumerate(y_raw):
                    pos = idx0 + j
                    if 0 <= pos < npoints:
                        Y[pos] = yv * yfactor
        X = np.linspace(firstx, lastx, npoints)
        nan_mask = np.isnan(Y)
        n_nan = int(nan_mask.sum())
        if n_nan == npoints:
            raise ValueError(
                f"{os.path.basename(filepath)}: nenhum ponto de ##XYDATA= foi decodificado "
                f"(forma de compressao JCAMP-DX nao reconhecida).")
        if n_nan > 0 and n_nan < npoints:
            # np.interp exige `xp` CRESCENTE e nao ordena sozinho. Em JCAMP-DX
            # firstx pode ser MAIOR que lastx (convencao comum em FTIR), o que
            # deixa X decrescente -- nesse caso a interpolacao devolveria
            # valores errados SEM erro, corrompendo o espectro em silencio.
            # Fica latente com equipamento que grava crescente, mas nao com
            # os que gravam decrescente. Ver tambem predicao.py/spectra_preview.py.
            xp, fp = X[~nan_mask], Y[~nan_mask]
            if xp.size > 1 and xp[0] > xp[-1]:
                xp, fp = xp[::-1], fp[::-1]
            Y[nan_mask] = np.interp(X[nan_mask], xp, fp)
        return X, Y

    # --- Legacy fallback (concatenation with encoded X) -----------------
    x_all: List[float] = []
    y_all: List[float] = []
    for s in linhas_dados:
        x_check, y_raw = _decodificar(s)
        if x_check is None:
            continue
        x_first = x_check * xfactor
        for yr in y_raw:
            x_all.append(x_first); y_all.append(yr * yfactor)
    if y_all and firstx is not None and lastx is not None:
        x_arr = np.linspace(firstx, lastx, len(y_all))
    else:
        x_arr = np.array(x_all, dtype=float)
    return np.asarray(x_arr, dtype=float), np.array(y_all, dtype=float)


_REGEX_FLOAT = re.compile(r"[-+]?\d+[.,]?\d*(?:[eE][-+]?\d+)?")
_REGEX_CONC  = re.compile(r"(\d+[.,]?\d*)\s*%")


def parse_spectrum(filepath):
    """Generic ASCII parser for .spectrum/.txt/.csv files with x,y per line.
    Tolerates headers, variable separators (space/tab/comma/;) and
    decimal comma. Returns (x, y) as ndarrays.

    Detects known binary formats (Bomem MB, PerkinElmer, Bruker OPUS)
    and emits a message with instructions to re-export as JCAMP-DX."""
    with open(filepath, "rb") as f:
        head = f.read(256)

    # Detect Bomem format (ABB Horizon MB) - UTF-16-LE "Bomem File"
    # (decode com errors="ignore" nao levanta UnicodeDecodeError)
    head_text = head.decode("utf-16-le", errors="ignore")
    if "Bomem" in head_text or "Horizon" in head_text:
        raise ValueError(
            f"Detected format: ABB Bomem Horizon MB (.spectrum binary).\n"
            f"  File: {os.path.basename(filepath)}\n"
            f"  THIS PARSER DOES NOT READ PROPRIETARY BINARIES for\n"
            f"  scientific integrity. To use with this pipeline:\n"
            f"    1. Open the spectra in Bomem Horizon software\n"
            f"    2. File -> Export -> JCAMP-DX (.dx) or ASCII (.txt)\n"
            f"    3. Point cfg.input_folder to the newly exported folder\n"
            f"  The parser already supports .dx and .txt automatically.")

    # Detect PerkinElmer .sp / Bruker OPUS
    if head[:4] == b"PEPE" or head[:4] == b"\x00\x00\x00\x00" and b"OPUS" in head:
        raise ValueError(
            f"Proprietary binary format detected: {os.path.basename(filepath)}.\n"
            f"  -> export as JCAMP-DX (.dx) or ASCII (.txt) from the original software.")

    n_bad = sum(1 for b in head[:64] if b < 9 or (13 < b < 32) or b > 126)
    if n_bad > 64 * 0.2:
        raise ValueError(
            f"Unrecognized binary file: {os.path.basename(filepath)}\n"
            f"  -> export as JCAMP-DX (.dx) or ASCII (.txt) from the\n"
            f"     instrument software (Bomem/PerkinElmer/Bruker/etc).")

    x_list: List[float] = []
    y_list: List[float] = []
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            tokens = _REGEX_FLOAT.findall(line)
            if len(tokens) < 2:
                continue
            # _REGEX_FLOAT so' casa texto que float() aceita (ex.: "5,e3" -> "5.e3")
            x_list.append(float(tokens[0].replace(",", ".")))
            y_list.append(float(tokens[1].replace(",", ".")))
    return np.array(x_list, dtype=float), np.array(y_list, dtype=float)


def _extrair_conc_filename(nome: str) -> Optional[float]:
    """Extracts the first occurrence of N,NN% from the filename."""
    m = _REGEX_CONC.search(nome)
    if not m:
        return None
    return float(m.group(1).replace(",", "."))      # _REGEX_CONC so' casa digitos[.,]digitos


def _listar_arquivos_espectro(pasta: str
                                ) -> Tuple[List[str], Optional[str]]:
    """Searches for spectral files by extension in the folder. Returns (list, ext)."""
    extensoes = [".dx", ".spectrum", ".txt", ".csv"]
    for ext in extensoes:
        candidatos = sorted(glob.glob(os.path.join(pasta, f"*{ext}")))
        if candidatos:
            return candidatos, ext
    return [], None


def _detectar_subpastas_classe(raiz: str) -> List[str]:
    """Returns subfolders that contain >=1 .dx/.spectrum/.txt/.csv file."""
    if not os.path.isdir(raiz):
        return []
    subpastas = []
    for nome in sorted(os.listdir(raiz)):
        caminho = os.path.join(raiz, nome)
        if os.path.isdir(caminho):
            arqs, _ = _listar_arquivos_espectro(caminho)
            if arqs:
                subpastas.append(caminho)
    return subpastas


#: Colunas de metadado que IDENTIFICAM a amostra de origem, e que nenhum
#: artefato gravado em disco precisa. `title_original`/`arquivo` carregam o
#: identificador inteiro (codigo de especie, data de coleta, adulterante e
#: teor); `cod`/`data`/`mae_id` carregam as partes; `subpasta` carrega a
#: organizacao das pastas de quem cedeu os dados.
_COLUNAS_IDENTIFICADORAS = ("title_original", "arquivo", "cod", "data",
                            "mae_id", "subpasta")


def sanitizar_metadados(df: "pd.DataFrame") -> "pd.DataFrame":
    """Remove identificacao da amostra de origem, preservando o que a
    analise usa.

    POR QUE. O GUARACI le arquivos de uma colecao que pode nao ser do usuario
    do software -- e o cabecalho JCAMP carrega mais do que o espectro. O
    parser ja' descarta `##AUDIT TRAIL` (operador e local) por nunca le-lo
    (ver `parse_dx`), mas `##TITLE` ele PRECISA ler: e' de onde saem classe,
    teor e o agrupamento de replicas. O identificador entao existe em
    memoria de forma legitima -- o que nao pode e' ser GRAVADO em disco,
    porque a partir dai ele viaja junto com resultados.

    O que sai: `title_original`, `arquivo`, `cod`, `data`, `mae_id`,
    `subpasta`.
    O que fica: `especie`, `teor`, `puro`, `adulterante`, `triplicata` --
    tudo que descreve a AMOSTRA sem dizer QUAL amostra da colecao ela e'.

    `grupo_replica` substitui `mae_id`: um rotulo sequencial (`G000`,
    `G001`, ...) que agrupa exatamente como o `mae_id` agrupava. Sem ele o
    arquivo perderia a informacao que sustenta a validacao group-aware --
    o diferencial do projeto -- e ninguem conseguiria auditar de fora se as
    replicas foram mesmo mantidas juntas.

    Idempotente: aplicar duas vezes da' o mesmo resultado.
    """
    limpo = df.copy()
    if "mae_id" in limpo.columns and "grupo_replica" not in limpo.columns:
        # Ordem de primeira aparicao (nao alfabetica): o rotulo nao deve
        # permitir reordenar/reidentificar os grupos pelo nome original.
        vistos: Dict[Any, str] = {}
        rotulos: List[str] = []
        for valor in limpo["mae_id"]:
            if pd.isna(valor):
                rotulos.append("")
                continue
            if valor not in vistos:
                vistos[valor] = f"G{len(vistos):03d}"
            rotulos.append(vistos[valor])
        limpo["grupo_replica"] = rotulos
    presentes = [c for c in _COLUNAS_IDENTIFICADORAS if c in limpo.columns]
    return limpo.drop(columns=presentes)


def prescan_dx(pasta: str) -> Dict[str, Any]:
    """Varredura BARATA dos cabecalhos .dx, para avisar ANTES de executar.

    Le apenas o cabecalho JCAMP-DX (`##TITLE=` e `##LASTX=`), parando no
    inicio dos dados espectrais -- nao carrega os 8192 pontos de cada
    arquivo. Com alguns milhares de arquivos leva **decimos de segundo**,
    contra dezenas de segundos da carga completa, o que a torna viavel
    num checklist interativo.

    Existe porque dois efeitos que MUDAM O N DA ANALISE so' apareciam no meio
    do log da execucao, quando o usuario ja tinha se comprometido com uma
    rodada longa:

    1. **Descarte por faixa espectral.** Datasets reais misturam janelas de
       aquisicao (ex.: NIR completo [0, 15797] vs faixa estreita [300, 4000]).
       Misturar e' invalido, entao `load_dx` mantem so' a faixa dominante
       e descarta o resto. Os descartados podem se concentrar em poucas
       classes, que perdem N sem que isso fosse
       visivel antes de rodar.
    2. **Amostras sem `mae_id`.** Sem grupo, a amostra entra na analise SEM
       protecao contra vazamento de replica -- exatamente o que a validacao
       group-aware existe para impedir.

    Nao levanta excecao: e' diagnostico best-effort. Arquivo ilegivel ou sem
    cabecalho reconhecivel entra em `n_sem_cabecalho` e segue.

    Returns
    -------
    dict com:
        n_arquivos       : total de .dx encontrados
        faixa_dominante  : xmax (cm-1) da janela majoritaria, ou None
        n_fora_da_faixa  : quantos serao DESCARTADOS na carga
        fora_por_especie : {especie: n} dos descartados (pasta = especie)
        n_sem_mae_id     : amostras sem grupo de replica identificavel
        n_grupos         : grupos `mae_id` distintos
        n_sem_cabecalho  : arquivos ilegiveis/sem ##LASTX
    """
    import glob as _glob
    from collections import Counter

    arquivos = _glob.glob(os.path.join(str(pasta), "**", "*.dx"), recursive=True)
    # (caminho, lastx, mae_id_ou_None) por arquivo legivel.
    lidos: List[Tuple[str, float, Optional[str]]] = []
    n_sem_cabecalho = 0

    for caminho in arquivos:
        titulo = lastx = None
        try:
            with open(caminho, "r", encoding="latin-1", errors="replace") as f:
                for linha in f:
                    if linha.startswith("##TITLE="):
                        titulo = linha[len("##TITLE="):].strip()
                    elif linha.startswith("##LASTX="):
                        lastx = linha[len("##LASTX="):].strip()
                    elif linha.startswith("##XYDATA") or linha.startswith("##XYPOINTS"):
                        break
        except OSError:
            n_sem_cabecalho += 1
            continue

        try:
            valor_lastx = float(str(lastx).replace(",", "."))
        except (TypeError, ValueError):
            n_sem_cabecalho += 1
            continue

        info = parse_title(titulo) if titulo else None
        mae = str(info["mae_id"]) if info and info.get("mae_id") else None
        lidos.append((caminho, valor_lastx, mae))

    faixa_dominante: Optional[float] = None
    n_fora = 0
    fora_por_especie: Dict[str, int] = {}
    sobreviventes = lidos
    if lidos:
        # Mesma regra de `load_dx`: moda do xmax arredondado a 100 cm-1,
        # tolerancia de 50 cm-1. Manter identico e' o que faz o aviso PREVER
        # o descarte real em vez de dar um numero proximo porem diferente.
        maxes = np.array([v for _c, v, _m in lidos])
        chave = np.round(maxes / 100.0) * 100.0
        valores, contagens = np.unique(chave, return_counts=True)
        faixa_dominante = float(valores[int(np.argmax(contagens))])
        incompat = np.abs(chave - faixa_dominante) >= 50.0
        n_fora = int(incompat.sum())
        if n_fora:
            # A especie vem da PASTA (o layout esperado e' uma subpasta por
            # especie); e' o mesmo agrupamento que o relatorio da carga usa.
            fora_por_especie = dict(Counter(
                os.path.basename(os.path.dirname(lidos[i][0]))
                for i in range(len(lidos)) if incompat[i]))
        sobreviventes = [lidos[i] for i in range(len(lidos)) if not incompat[i]]

    # mae_id contado SO' entre os sobreviventes: e' o numero que o usuario vera'
    # no log da carga. Contar sobre o total daria um valor maior e sem
    # correspondencia com a analise (os sem mae_id podem se concentrar numa
    # classe cujo padrao de titulo e faixa espectral sao incompativeis, e
    # essa ja' sai pelo
    # descarte).
    n_sem_mae_id = sum(1 for _c, _v, m in sobreviventes if m is None)
    grupos = {m for _c, _v, m in sobreviventes if m is not None}
    # `load_dx` transforma cada orfa num grupo de 1 (`orfao_<arquivo>`),
    # para isolar a amostra sem desligar o GroupKFold do dataset inteiro.
    # Somamos aqui pela MESMA razao de sempre nesta funcao: o numero exibido
    # tem de ser o que o usuario vera' no log da carga (grupos reais mais
    # os orfaos).
    n_grupos_efetivos = len(grupos) + n_sem_mae_id

    return {
        "n_arquivos":       len(arquivos),
        "n_apos_descarte":  len(sobreviventes),
        "faixa_dominante":  faixa_dominante,
        "n_fora_da_faixa":  n_fora,
        "fora_por_especie": fora_por_especie,
        "n_sem_mae_id":     n_sem_mae_id,
        "n_grupos":         n_grupos_efetivos,
        "n_grupos_reais":   len(grupos),
        "n_sem_cabecalho":  n_sem_cabecalho,
    }


#: teto padrao de pontos do eixo comum dos ESPECTROS (comportamento historico); 0 = sem limite
MAX_PONTOS_ESPECTRO = 2000


def _media_por_bin(xs: np.ndarray, ys: np.ndarray, grade: np.ndarray) -> np.ndarray:
    """Media do sinal em cada intervalo da `grade` (bucketing anti-alias). `xs` crescente.
    Intervalo sem nenhum ponto nativo: interpolacao linear (so' ocorre com eixo muito irregular)."""
    meio = (grade[1:] + grade[:-1]) / 2.0
    bordas = np.concatenate(([grade[0] - (meio[0] - grade[0])], meio, [grade[-1] + (grade[-1] - meio[-1])]))
    soma = np.concatenate(([0.0], np.cumsum(ys)))
    k = np.searchsorted(xs, bordas)
    n = np.diff(k)
    media = np.diff(soma[k]) / np.maximum(n, 1)
    return np.where(n > 0, media, np.interp(grade, xs, ys))


def reamostrar_espectros(espectros, max_pontos: int = MAX_PONTOS_ESPECTRO, nome: str = "espectros",
                         integrar: bool = False, avisar: bool = True):
    """Leva `[(x, y), ...]` a UMA grade comum (faixa de sobreposicao dos eixos). Devolve `(grade, X, info)`.

    - Numero de pontos da grade = `min(n nativo)`, limitado por `max_pontos` (`0` = sem limite).
    - Quando uma amostra tem >= 2x mais pontos que a grade, cada ponto da grade e' a MEDIA do sinal no seu
      intervalo (binning anti-alias), nunca uma amostra pontual: um pico mais fino que o passo da grade
      mudaria de altura conforme a fase da amostragem (ate' ~8,5 % entre injecoes identicas). Abaixo de 2:1
      (ou sem reducao) usa-se interpolacao linear. `integrar=True` (RMN) usa a media por bin sempre que ha'
      reducao.
    - `info` e' None quando nenhuma amostra perdeu pontos; senao um dict com `n_original_min/max`, `n_final`,
      `largura_bin` (unidades do eixo), `metodo` e `teto`; a mensagem de AVISO ja' foi impressa."""
    xmin = max(float(np.min(x)) for x, _ in espectros)
    xmax = min(float(np.max(x)) for x, _ in espectros)
    if xmin >= xmax:
        raise ValueError(
            "Amostras sem sobreposicao de eixo utilizavel -- verifique se "
            "todos os arquivos/pastas sao do mesmo instrumento/faixa.")
    n_nat = [len(x) for x, _ in espectros]
    teto = int(max_pontos or 0)
    n_pts = min(n_nat) if teto <= 0 else min(teto, min(n_nat))
    grade = np.linspace(xmin, xmax, n_pts)
    linhas = []
    usou_media = False
    for x, y in espectros:
        idx = np.argsort(x)
        xs, ys = np.asarray(x)[idx], np.asarray(y)[idx]
        if n_pts > 1 and ((integrar and xs.size > n_pts) or xs.size >= 2 * n_pts):
            linhas.append(_media_por_bin(xs, ys, grade))
            usou_media = True
        else:
            linhas.append(np.interp(grade, xs, ys))
    X = np.array(linhas, dtype=float)
    if max(n_nat) <= n_pts:
        return grade, X, None
    larg = float((xmax - xmin) / max(n_pts - 1, 1))
    metodo = "media por bin" if usou_media else "interpolacao linear"
    info = {"formato": nome, "n_original_min": int(min(n_nat)), "n_original_max": int(max(n_nat)),
            "n_final": int(n_pts), "largura_bin": larg, "metodo": metodo, "teto": teto}
    if avisar:
        avisar_reamostragem(info)
    return grade, X, info


def avisar_reamostragem(info: dict, sufixo: str = "") -> None:
    """Imprime o AVISO de reducao do eixo (n original -> n final) descrito por `info`."""
    ori = (f"{info['n_original_min']}" if info["n_original_min"] == info["n_original_max"]
           else f"{info['n_original_min']}-{info['n_original_max']}")
    nome, n_pts, metodo, larg, teto = (info["formato"], info["n_final"], info["metodo"],
                                       info["largura_bin"], info["teto"])
    print(f"[AVISO] Reamostragem do eixo ({nome}): {ori} -> {n_pts} pontos, {metodo} (passo da grade "
          f"{larg:.6g} unidades do eixo; teto {'sem limite' if teto <= 0 else teto}){sufixo}. "
          f"Picos mais finos que o passo perdem altura; aumente o teto (0 = sem limite) para preservar "
          f"a resolucao nativa.")


def load_dx(pasta: str, class_part: int = 0,
                 extrair_conc: bool = False,
                 use_parse_title: bool = True,
                 max_pontos: int = MAX_PONTOS_ESPECTRO
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                              Optional[np.ndarray], Optional[np.ndarray],
                              Optional[pd.DataFrame]]:
    """Loads spectra with auto-detection of folder structure:

        (A) root folder with subfolders (each subfolder = 1 species/class)
            -> recursive via _detectar_subpastas_classe
        (B) single folder with .dx/.spectrum/.txt/.csv files
            -> legacy mode (class_part)

    When use_parse_title=True and the file is .dx, extracts ##TITLE= and
    uses parse_title() for rich metadata (species, adulterant, content,
    replicate, mae_id). Otherwise, uses the filename (fallback).

    Returns: (wavenumbers, X, rotulos, conc, mae_id, metadados_df).
        - mae_id      : ndarray of strings or None if not available
        - metadados_df: pd.DataFrame with all parsed fields
    """
    subpastas = _detectar_subpastas_classe(pasta)
    if subpastas:
        print(f"[INFO] Multi-folder structure detected: {len(subpastas)} "
              f"subfolders in {pasta}")
        arquivos: List[Tuple[str, str]] = []   # (caminho, nome_subpasta)
        for sp in subpastas:
            arqs, _ = _listar_arquivos_espectro(sp)
            arquivos.extend((a, os.path.basename(sp)) for a in arqs)
        ext_usada = os.path.splitext(arquivos[0][0])[1] if arquivos else None
    else:
        if not os.path.isdir(pasta):
            raise FileNotFoundError(
                f"Path does NOT exist: {pasta}\n"
                f"  -> check cfg.input_folder or use cfg.mode='sintetico'.")
        arqs, ext_usada = _listar_arquivos_espectro(pasta)
        if not arqs:
            raise FileNotFoundError(
                f"Folder exists but contains no known spectral files.\n"
                f"  Folder: {pasta}\n"
                f"  Contents (up to 10 items): {os.listdir(pasta)[:10]}")
        arquivos = [(a, "") for a in arqs]

    parser = parse_dx if ext_usada == ".dx" else parse_spectrum
    print(f"[INFO] {len(arquivos)} {ext_usada} files found "
          f"(parser={parser.__name__})")

    pode_parse_title = use_parse_title and ext_usada == ".dx"

    espectros: List[Tuple[np.ndarray, np.ndarray]] = []
    rotulos: List[str] = []
    concs:   List[Optional[float]] = []
    mae_ids: List[Optional[str]]   = []
    meta_rows: List[Dict[str, Any]] = []
    n_falhos = 0
    n_title_falhos = 0
    cods_desconhecidos: set = set()

    for arq, subpasta_nome in arquivos:
        try:
            x, y = parser(arq)
        except Exception as e:  # noqa: BLE001 -- parsing defensivo de arquivo
            # externo (JCAMP-DX/CSV de instrumento, formato variavel entre
            # fabricantes/versoes); erro impresso COM NOME DO ARQUIVO e
            # contabilizado em n_falhos (reportado ao usuario abaixo), nunca
            # silencioso -- 1 arquivo malformado nao derruba o dataset inteiro.
            n_falhos += 1
            print(f"  [ERROR] {os.path.basename(arq)}: {e}")
            continue
        if len(x) == 0:
            print(f"  [WARNING] {os.path.basename(arq)} has no data — skipped")
            continue

        nome_arq = os.path.splitext(os.path.basename(arq))[0]
        title_parsed: Optional[Dict[str, Any]] = None
        if pode_parse_title:
            title = extract_dx_title(arq)
            # Pureza indeterminada (achado A2-2): sem TITLE parseavel E sem
            # teor recuperavel do nome, a amostra entraria como "pura" e
            # contaminaria o treino one-class. Excluida com aviso nominal,
            # nunca em silencio. Ver `_TITLES_PUREZA_INDETERMINADA`.
            if title and title.strip() in _TITLES_PUREZA_INDETERMINADA:
                _CONTAGEM_CORRECOES["pureza_excluidos"] += 1
                print(f"  [WARNING] {os.path.basename(arq)} EXCLUIDO — "
                      f"{_TITLES_PUREZA_INDETERMINADA[title.strip()]}")
                continue
            if title:
                title_parsed = parse_title(title)
            if title_parsed is None:
                n_title_falhos += 1

        if title_parsed is not None:
            classe = title_parsed["especie"]
            conc_i = title_parsed["teor"]
            mae_i  = title_parsed["mae_id"]
            if not title_parsed["cod_conhecido"]:
                cods_desconhecidos.add(title_parsed["cod"])
            meta = dict(title_parsed)
            meta["arquivo"]   = os.path.basename(arq)
            meta["subpasta"]  = subpasta_nome
        else:
            # Fallback: try to extract COD from filename prefix and map
            # to canonical name (avoids duplicate class due to accent:
            # subfolder 'Classe' vs CODIGO_ESPECIE['COD']='Classe').
            m_cod = re.match(r"^([A-Z]{2,4})[-_]", nome_arq)
            cod_fb = m_cod.group(1).upper() if m_cod else None
            if cod_fb and cod_fb in CODIGO_ESPECIE:
                classe = CODIGO_ESPECIE[cod_fb]
            elif subpasta_nome:
                classe = subpasta_nome
            else:
                partes = nome_arq.replace("_", "-").split("-")
                classe = partes[class_part] if abs(class_part) < len(partes) \
                          else nome_arq
            conc_i = _extrair_conc_filename(nome_arq) if extrair_conc else None
            mae_i  = None
            meta = {
                "arquivo":     os.path.basename(arq),
                "subpasta":    subpasta_nome,
                "especie":     classe,
                "teor":        conc_i,
                "puro":        conc_i is None,
                "mae_id":      None,
                "cod":         cod_fb,
                "cod_conhecido": bool(cod_fb and cod_fb in CODIGO_ESPECIE),
            }

        espectros.append((x, y))
        rotulos.append(classe)
        concs.append(conc_i)
        mae_ids.append(mae_i)
        meta_rows.append(meta)

    if not espectros:
        raise ValueError(
            f"No valid spectra loaded. ({n_falhos} files with errors)")
    if n_falhos > 0:
        print(f"[WARNING] {n_falhos} files with parsing errors — skipped.")
    if pode_parse_title:
        if n_title_falhos > 0 and n_title_falhos >= len(espectros):
            # Nenhum arquivo segue a convencao OPCIONAL de ##TITLE (COD-DD-MM-AAAA[_AD-x-N%]_Tn): e' o caso normal de quem
            # tem os proprios dados; nao e' defeito do arquivo. Classe e grupo vem da subpasta/nome do arquivo.
            print("[INFO] No ##TITLE= follows the optional GUARACI title pattern (this is normal for your own data): "
                  "class comes from the subfolder or file name. To protect replicas from leaking between training and "
                  "test, give a groups file (arquivo_grupos: CSV file,group).")
        elif n_title_falhos > 0:
            print(f"[WARNING] {n_title_falhos} of {len(espectros)} files do not follow the ##TITLE= pattern that the "
                  f"others follow — using fallback (name/subfolder) for them.")
        if cods_desconhecidos:
            print(f"[CAUTION] CODs not mapped in CODIGO_ESPECIE: "
                  f"{sorted(cods_desconhecidos)}")

    # --- Detection of dominant acquisition range ------------------------
    # Real datasets may mix spectral ranges (e.g. full NIR
    # [0,15797] 8192pts vs narrow range [300,4000]). Mixing is
    # scientifically invalid. Detects the dominant range (mode of xmax
    # rounded to 100 cm-1) and DISCARDS incompatible files with a report.
    maxes = np.array([float(e[0].max()) for e in espectros])
    chave_faixa = np.round(maxes / 100.0) * 100.0
    valores, contagens = np.unique(chave_faixa, return_counts=True)
    faixa_dominante = float(valores[int(np.argmax(contagens))])
    compat = np.abs(chave_faixa - faixa_dominante) < 50.0
    n_drop = int((~compat).sum())

    if n_drop > 0:
        print(f"[CAUTION] Heterogeneous spectral ranges detected. "
              f"Dominant range: xmax~{faixa_dominante:.0f} cm-1 "
              f"({int(compat.sum())} files).")
        print(f"          DISCARDING {n_drop} files with "
              f"incompatible range (not comparable in the same spectral window):")
        # Report by species for discarded files
        from collections import Counter
        drop_especies = Counter(
            meta_rows[i].get("especie", "?")
            for i in range(len(espectros)) if not compat[i])
        for esp, n in sorted(drop_especies.items(), key=lambda kv: -kv[1]):
            exemplos = [meta_rows[i]["arquivo"]
                         for i in range(len(espectros))
                         if not compat[i]
                         and meta_rows[i].get("especie") == esp][:1]
            print(f"            {esp}: {n} arquivos (ex: {exemplos[0] if exemplos else '?'})")
        # Filter all parallel lists
        keep = [i for i in range(len(espectros)) if compat[i]]
        espectros = [espectros[i] for i in keep]
        rotulos   = [rotulos[i]   for i in keep]
        concs     = [concs[i]     for i in keep]
        mae_ids   = [mae_ids[i]   for i in keep]
        meta_rows = [meta_rows[i] for i in keep]
        print(f"          {len(espectros)} files remain in the "
              f"dominant range.")

    # (a faixa dominante e' a moda: ao menos um espectro sempre sobrevive ao filtro)

    # Common grid by interpolation (now within the dominant range)
    grade, X_raw, _info_reamostragem = reamostrar_espectros(espectros, max_pontos, "dx")

    # Concentrations (pure samples = 0% by convention)
    if any(c is not None for c in concs):
        conc_arr: Optional[np.ndarray] = np.array(
            [c if c is not None else 0.0 for c in concs], dtype=float)
        n_com_conc = sum(1 for c in concs if c is not None)
        print(f"[INFO] Concentrations extracted: {n_com_conc}/{len(concs)} "
              f"(pure samples treated as 0%).")
    else:
        conc_arr = None

    # mae_id array. For files without valid parse, assigns unique ID
    # (filename) — becomes a "group of 1", isolating them without
    # disabling the entire dataset's GroupKFold.
    n_com_mae = sum(1 for m in mae_ids if m is not None)
    if n_com_mae == 0:
        mae_arr: Optional[np.ndarray] = None
    else:
        n_orfaos = 0
        mae_final: List[str] = []
        # strict=True: `mae_ids` e `meta_rows` sao preenchidos em lockstep no
        # laco de carregamento, entao ter comprimentos diferentes e' um bug
        # de programacao -- e um `zip()` normal o esconderia TRUNCANDO em
        # silencio, gerando um `mae_arr` mais curto que X. Como mae_id e' o
        # que impede vazamento de replica entre treino e teste, um
        # desalinhamento aqui deslocaria os grupos de todas as amostras
        # seguintes sem nenhum sinal. Melhor estourar do que mentir.
        for m, row in zip(mae_ids, meta_rows, strict=True):
            if m is not None:
                mae_final.append(m)
            else:
                mae_final.append(f"orfao_{row['arquivo']}")
                n_orfaos += 1
        mae_arr = np.array(mae_final, dtype=str)
        n_grupos = int(len(np.unique(mae_arr)))
        msg_orfaos = f", {n_orfaos} isolated orphans" if n_orfaos > 0 else ""
        print(f"[INFO] mae_id: {n_com_mae}/{len(mae_ids)} samples parsed, "
              f"{n_grupos} unique groups{msg_orfaos}.")

    metadados_df = pd.DataFrame(meta_rows)
    metadados_df.attrs["reamostragem"] = _info_reamostragem

    return (grade,
            np.array(X_raw, dtype=float),
            np.array(rotulos, dtype=str),
            conc_arr, mae_arr, metadados_df)


def _leitor_sintetico(cfg: "Config"):
    wn, X, rot, conc, mae = generate_synthetic_data(cfg)
    return wn, X, rot, conc, mae, None


def _leitor_csv(cfg: "Config"):
    wn, X, rot, conc, mae_id, nomes = _carregar_csv_completo(
        cfg.csv_file, cfg.class_column, cfg.conc_column,
        col_grupo=getattr(cfg, "group_column", "") or "",
        arquivo_grupos=getattr(cfg, "group_file", "") or "")
    # nomes das variaveis (cabecalho nao numerico): figuras e relatorios os mostram no lugar da posicao
    setattr(cfg, "nomes_variaveis", list(nomes) if nomes else None)   # estado de execucao: fora do dataclass
    return wn, X, rot, conc, mae_id, None


def _leitor_dx(cfg: "Config"):
    wn, X, rot, conc, mae_id, meta = load_dx(cfg.input_folder, cfg.class_part,
                                             cfg.extract_conc_from_filename, cfg.use_parse_title,
                                             max_pontos=int(getattr(cfg, "espectro_max_pontos", MAX_PONTOS_ESPECTRO)))
    setattr(cfg, "reamostragem_aplicada", meta.attrs.get("reamostragem") if meta is not None else None)
    # arquivo_grupos (documentado "ex.: arquivos .dx") era ignorado neste modo: sem ##TITLE= no
    # padrao do projeto, as replicas caiam em StratifiedKFold (vazamento). Mesma chave dos
    # leitores de instrumento: o nome do arquivo sem extensao.
    arq_grupos = getattr(cfg, "group_file", "") or ""
    if arq_grupos and meta is not None and "arquivo" in meta.columns:
        _col, mapa = carregar_mapa_grupos(arq_grupos)
        nomes = [os.path.splitext(str(a))[0] for a in meta["arquivo"]]
        mae_id = atribuir_grupos(nomes, mapa, origem="dx")
    return wn, X, rot, conc, mae_id, meta


def _leitor_imagem(cfg: "Config"):
    from guaraci.dados_imagem import load_images
    resultado = load_images(cfg.input_folder, cfg.image_crop,
                             cfg.include_image_texture,
                             referencia=tuple(getattr(cfg, "image_reference", ()) or ()) or None,
                             segmentacao=str(getattr(cfg, "image_segmentation", "nenhuma") or "nenhuma"))
    metadados_df = resultado[5]
    if metadados_df is not None:
        cfg.grouping_guarantee = metadados_df.attrs.get(
            "grouping_guarantee", cfg.grouping_guarantee)
    return resultado


# Leitores built-in (item 20 da auditoria: registry em vez de if/elif fixo —
# ver io_registry.py para o contrato e como registrar um novo mode).
register_reader("sintetico", _leitor_sintetico)
register_reader("csv", _leitor_csv)
register_reader("dx", _leitor_dx)
register_reader("imagem", _leitor_imagem)


def chave_unidade_validacao(mae_id, unidade: str = "mae_id"):
    """Chave da UNIDADE INDEPENDENTE da validacao agrupada (B4c do parecer de
    2026-09-25). `unidade="mae_id"` (padrao): cada `mae_id` e' um grupo.
    `unidade="sessao"`: grupo = sessao de coleta (`session_from_mae_id`, p.ex.
    as diluicoes da mesma amostra-base contam como UMA unidade) -- necessario
    quando o `mae_id` e' um nivel de teor dentro de uma amostra fisica: um
    teste nulo com rotulo sorteado por sessao deu bal.acc 0,90-0,98 (acaso =
    0,25) agrupando por `mae_id` e 0,22-0,35 agrupando por sessao. Devolve
    `mae_id` intacto para `None`/`"mae_id"`."""
    if mae_id is None or unidade != "sessao":
        return mae_id
    return np.array([session_from_mae_id(m) for m in mae_id], dtype=str)


def garantia_agrupamento_prevista(cfg: "Config") -> str:
    """Nivel de garantia de agrupamento que esta configuracao PRODUZIRA, antes
    de ler qualquer dado ("high" | "medium" | "none" | "verificar"). Existe
    para que a interface (checklist da CLI, tela Modelo) nunca afirme
    "anti-vazamento ativo" so' porque a FLAG de validacao agrupada esta
    ligada: a flag e' intencao, a garantia e' o que os dados permitem.

    - dx / sintetico: "high" (o `mae_id` sai do ##TITLE= / do gerador; a
      contagem de amostras sem `mae_id` so' aparece apos ler -- o checklist a
      antecipa por `prescan_dx`).
    - csv e leitores de instrumento: "medium" se ha' `coluna_grupo`/
      `arquivo_grupos` (grupo declarado, nao verificavel), senao "none".
    - imagem / hsi: "verificar" (depende da estrutura de pastas).
    """
    if not getattr(cfg, "group_by_mae_id", True):
        return "none"
    mode = getattr(cfg, "mode", "dx")
    if mode == "dx" and getattr(cfg, "group_file", None):
        return "medium"   # grupo declarado pelo arquivo_grupos, nao pelo ##TITLE=
    if mode in ("dx", "sintetico"):
        return "high"
    if mode in ("imagem", "hsi"):
        return "verificar"
    declarado = bool(getattr(cfg, "group_column", None)
                     or getattr(cfg, "group_file", None))
    return "medium" if declarado else "none"


def load_data(cfg: "Config"
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                Optional[np.ndarray], Optional[np.ndarray],
                                Optional[pd.DataFrame]]:
    """Unified data loader. Despacha para o leitor registrado em `cfg.mode`
    (ver io_registry.py). Returns 6-tuple:
        (wavenumbers, X, rotulos, conc, mae_id, metadados_df)
    metadados_df is always None in 'sintetico'/'csv' mode; mae_id is None
    only in 'csv'/'imagem' mode (sem replicas fisicas conhecidas)."""
    # A garantia e' RECALCULADA a cada carga (a Config pode ser reaproveitada
    # entre execucoes -- p.ex. o app web --, e um "none" de uma execucao CSV
    # nao pode vazar para uma execucao .dx seguinte). Base "high"; os leitores
    # que sabem menos (imagem/hsi) a rebaixam por conta propria.
    cfg.grouping_guarantee = "high"
    setattr(cfg, "canais_dad_lidos", None)
    setattr(cfg, "reamostragem_aplicada", None)     # os leitores que reduzem o eixo (dx, opus, spc, hplc...) a preenchem
    _env = os.environ.get("GUARACI_APLICAR_CORRECOES_LOCAIS", "").strip().lower() in (
        "1", "true", "yes", "on", "sim")
    definir_correcoes_locais(bool(getattr(cfg, "apply_local_corrections", False)) or _env)
    # m-20: os leitores avancados (opus/spc/sp/rmn/hplc/gcms/eem) se registram ao
    # importar `leitores_avancados`; antes so' `import guaraci.pipeline` fazia isso.
    import guaraci.leitores_avancados  # noqa: F401
    import guaraci.icp_multielementar  # noqa: F401
    resultado = get_reader(cfg.mode)(cfg)
    cfg.correcoes_locais_aplicadas = resumo_correcoes_locais()
    if cfg.correcoes_locais_aplicadas["ativo"]:
        _rc = cfg.correcoes_locais_aplicadas
        print("[AVISO] Correcoes de metadado LOCAIS aplicadas: "
              f"{_rc['titulos_corrigidos']} titulo(s) corrigido(s), "
              f"{_rc['alias_aplicados']} alias de mae_id, "
              f"{_rc['pureza_excluidos']} espectro(s) excluido(s) por pureza "
              f"indeterminada; arquivos: "
              f"{', '.join(a['arquivo'] + ' sha256=' + a['sha256'][:12] for a in _rc['arquivos']) or 'nenhum encontrado'}. "
              f"O mesmo dado bruto SEM esses arquivos daria rotulos diferentes.")
    # Achado do roteiro de teste externo (2026-09-23): csv (e qualquer leitor
    # sem mae_id) deixava `grouping_guarantee` no default "high", e a
    # auditoria/resumo afirmavam "mae_id confiavel para toda amostra" numa
    # execucao SEM nenhum mae_id -- alegacao falsa sobre o diferencial
    # central do projeto. Sem mae_id nao ha' garantia de agrupamento.
    # R2-06: UM unico valor de grupo nao separa nada (a CV cai em StratifiedKFold):
    # nao ha' protecao, entao a garantia e' "none", nao "medium".
    if resultado[4] is None or len(np.unique(np.asarray(resultado[4]))) < 2:
        cfg.grouping_guarantee = "none"
    elif ((cfg.mode not in ("dx", "imagem", "hsi", "sintetico")
           and (getattr(cfg, "group_column", None)
                or getattr(cfg, "group_file", None)))
          or (cfg.mode == "dx" and getattr(cfg, "group_file", None))):
        # Grupo DECLARADO pelo usuario (coluna/arquivo): a validacao e'
        # agrupada, mas o software nao tem como verificar que os grupos sao
        # de fato independentes -- "medium", nunca "high" (B4a/B4b).
        cfg.grouping_guarantee = "medium"
    return resultado
