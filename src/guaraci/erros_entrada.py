"""erros_entrada.py -- mensagens de entrada ruim dos comandos de linha de comando, em PT e EN, sem rastro tecnico e com a acao sugerida.

Tres entradas ruins por comando principal (arquivo vazio, coluna ausente, formato errado) caem aqui em vez de virar traceback ou
nome de excecao do pandas. O texto diz (1) o que aconteceu em palavras do usuario, (2) o que fazer agora.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Tuple

__all__ = ["idioma", "arquivo_nao_encontrado", "leitura_csv", "coluna_ausente", "pasta_de_dados_ausente"]

def _exemplo(cenario: str = "comparar") -> Tuple[str, str]:
    """Frase que aponta o exemplo do PROPRIO comando (comparar, vip, monitorar, transferir)."""
    return (f"`guaraci exemplo {cenario}` gera uma tabela de modelo", f"`guaraci exemplo {cenario}` builds a model table")


def idioma(args_l: Optional[Sequence[str]] = None) -> str:
    """'en' ou 'pt': a flag `--idioma` vence; senao o idioma comum (guaraci.idioma_relatorio.idioma)."""
    args_l = list(args_l or [])
    for i, a in enumerate(args_l):
        if a == "--idioma" and i + 1 < len(args_l):
            return "en" if args_l[i + 1].lower().startswith("en") else "pt"
        if a.startswith("--idioma="):
            return "en" if a.split("=", 1)[1].lower().startswith("en") else "pt"
    from guaraci.idioma_relatorio import idioma as _idioma_padrao   # GUARACI_IDIOMA, escolha do menu, ingles
    return _idioma_padrao()


def arquivo_nao_encontrado(caminho: Any, lang: str, exemplo: str = "comparar") -> str:
    frase = _exemplo(exemplo)
    c = caminho if caminho else "(nenhum)" if lang == "pt" else "(none)"
    if lang == "en":
        return f"file not found: {c}. Check the path (relative to the current folder) and the spelling; {frase[1]}."
    return f"arquivo nao encontrado: {c}. Confira o caminho (relativo a pasta atual) e a grafia; {frase[0]}."


def _causa(exc: BaseException, caminho: str) -> str:
    """'vazio' | 'binario' | 'formato'."""
    try:
        if os.path.getsize(caminho) == 0:
            return "vazio"
    except OSError:
        pass
    nome = type(exc).__name__
    msg = str(exc)
    if nome == "EmptyDataError" or "Could not determine delimiter" in msg or "No columns to parse" in msg:
        return "vazio"
    if nome == "UnicodeDecodeError" or isinstance(exc, UnicodeError):
        return "binario"
    return "formato"


_CAUSA = {
    "vazio": ("o arquivo esta vazio ou sem linhas de dados", "the file is empty or has no data rows"),
    "binario": ("o arquivo nao e' texto (parece binario, imagem ou outra codificacao)", "the file is not text (it looks binary, an image or another encoding)"),
    "formato": ("o conteudo nao tem formato de tabela reconhecivel", "the content does not look like a table"),
}
_ACAO = ("Abra o arquivo, confira que ha cabecalho e linhas e salve como CSV (UTF-8); ", "Open the file, check it has a header and rows, and save it as CSV (UTF-8); ")


def leitura_csv(caminho: str, exc: BaseException, lang: str, exemplo: str = "comparar") -> str:
    frase = _exemplo(exemplo)
    c = _causa(exc, caminho)
    if lang == "en":
        return f"could not read {caminho} as CSV: {_CAUSA[c][1]}. {_ACAO[1]}{frase[1]}."
    return f"nao consegui ler {caminho} como CSV: {_CAUSA[c][0]}. {_ACAO[0]}{frase[0]}."


def coluna_ausente(papel: str, nome: str, colunas: Sequence[Any], lang: str, arquivo: str = "") -> str:
    cols = ", ".join(map(str, list(colunas)[:12])) + (" ..." if len(colunas) > 12 else "")
    onde = f" em {arquivo}" if arquivo and lang == "pt" else f" in {arquivo}" if arquivo else ""
    if lang == "en":
        return (f"{papel} column '{nome}' does not exist{onde}; columns found: {cols}. "
                f"Pick one of these names (the match is exact, including case) or fix the header of the file.")
    return (f"coluna {papel} '{nome}' nao existe{onde}; colunas encontradas: {cols}. "
            f"Escolha um destes nomes (a comparacao e' exata, inclusive maiusculas) ou corrija o cabecalho do arquivo.")


def pasta_de_dados_ausente(pasta: str, lang: str) -> str:
    if lang == "en":
        return (f"the data folder '{pasta}' does not exist. Set `pasta_dados` in config.yaml to the folder that holds one subfolder "
                "per class, or run `guaraci exemplo classificar` to build a ready example.")
    return (f"a pasta de dados '{pasta}' nao existe. Aponte `pasta_dados` do config.yaml para a pasta que tem uma subpasta por classe, "
            "ou rode `guaraci exemplo classificar` para montar um exemplo pronto.")
