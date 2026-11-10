r"""Troca caminhos absolutos de maquina por caminhos com o usuario sentinela
`alguem` nos arquivos de saida de reproducao (docs/revisao_pares_*/antes|depois):
o teste `test_sem_identificador_real` proibe usuario real em arquivo versionado.

    python scripts/sanitizar_saidas.py <pasta> ...

Cobre o usuario em `<letra>:\Users\...` e `<letra>:\home\...` (separador simples
ou repetido), no home do Unix (`/home/alguem`, `/Users/alguem`) e na forma
achatada `<letra>--Users-...`.

Padroes pessoais adicionais (ex.: nome de pasta institucional) ficam FORA do
repositorio, em `~/.guaraci_local/padroes_sanitizar.txt`: uma expressao regular
por linha, `#` comenta. Cada ocorrencia vira `<pessoal>`. Arquivo ausente: nada
muda. Arquivo nao UTF-8 ou regex invalida: erro com o numero da linha.
"""
import re
import sys
from pathlib import Path
from typing import List, Optional

SENTINELA = "alguem"
ARQUIVO_PADROES_LOCAIS = Path.home() / ".guaraci_local" / "padroes_sanitizar.txt"

# Qualquer usuario (nao uma lista de nomes conhecidos -- a lista escrita
# aqui era ela mesma um nome de usuario versionado; achado de 2026-10-02).
_NAO_SENTINELA = r"(?!" + SENTINELA + r"(?![^\\/\s]))"
_RX_WIN = re.compile(r"([A-Za-z]:[\\/]+(?:Users|home)[\\/]+)" + _NAO_SENTINELA + r"[^\\/\s\"'`)]+", re.I)
_RX_POSIX = re.compile(r"(/+(?:home|Users)/+)" + _NAO_SENTINELA + r"[^/\s\"'`)]+")
# Forma achatada que ferramentas usam para nomear pasta temporaria a partir do
# caminho do projeto (letra, '--Users-', usuario, resto): nao ha' como isolar o
# usuario, entao o token inteiro vira um marcador neutro.
_RX_ACHATADA = re.compile(r"[A-Za-z]--(?:Users|home)-[^\s\\/\"'`)|]+")


def carregar_padroes_locais(caminho: Optional[Path] = None) -> List["re.Pattern[str]"]:
    """Regexes pessoais do arquivo local; [] se ausente."""
    caminho = caminho or ARQUIVO_PADROES_LOCAIS
    if not caminho.exists():
        return []
    try:
        linhas = caminho.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as e:
        raise ValueError(f"{caminho.name}: precisa ser UTF-8 ({e})") from e
    padroes = []
    for n, linha in enumerate(linhas, start=1):
        s = linha.strip()
        if not s or s.startswith("#"):
            continue
        try:
            padroes.append(re.compile(s, re.I))
        except re.error as e:
            raise ValueError(f"{caminho.name}, linha {n}: expressao regular invalida ({e})") from e
    return padroes


def sanitizar_texto(t: str, padroes_locais: Optional[List["re.Pattern[str]"]] = None) -> str:
    t = _RX_ACHATADA.sub("<pasta-temporaria-do-projeto>", t)
    t = _RX_WIN.sub(lambda m: m.group(1) + SENTINELA, t)
    t = _RX_POSIX.sub(lambda m: m.group(1) + SENTINELA, t)
    for rx in padroes_locais or ():
        t = rx.sub("<pessoal>", t)
    return t


def main(pastas, padroes_locais: Optional[List["re.Pattern[str]"]] = None):
    if padroes_locais is None:
        padroes_locais = carregar_padroes_locais()
    n = 0
    for pasta in pastas:
        for arq in Path(pasta).rglob("*"):
            if not arq.is_file() or arq.suffix.lower() not in (".txt", ".json", ".log", ".md"):
                continue
            try:
                t = arq.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            novo = sanitizar_texto(t, padroes_locais)
            if novo != t:
                arq.write_text(novo, encoding="utf-8")
                n += 1
    print(f"{n} arquivo(s) sanitizado(s)")


if __name__ == "__main__":
    main(sys.argv[1:])
