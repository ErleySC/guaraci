"""Documentos que o programa cita em texto mostrado ao usuario, entregues DENTRO do pacote.

O programa instalado (wheel) nao tem `docs/` nem `MANUAL.md` ao lado: tudo que a CLI, o assistente e a
ajuda apontam precisa existir aqui. Os arquivos de `dados_docs/` sao COPIAS geradas por
`scripts/gerar_docs_pacote.py` (nunca editar a mao; `--verificar` falha se estiverem defasados).

Uso pela CLI: sem argumento lista os topicos; com topico ou numero de secao mostra o texto.
"""
from __future__ import annotations

import re
from importlib import resources
from typing import Dict, Optional

__all__ = ["TOPICOS", "ler", "secao_do_manual", "lista"]

#: topico -> (arquivo no pacote, descricao curta)
TOPICOS: Dict[str, tuple] = {
    "manual": ("manual.md", "Manual completo do programa"),
    "seguranca": ("seguranca.md", "Politica de seguranca e carregamento de modelos"),
    "limitacoes": ("limitacoes.md", "Limitacoes conhecidas, com a evidencia de cada uma"),
    "suporte": ("suporte.md", "Onde perguntar e como reportar erro"),
    "privacidade": ("privacidade.md", "Aviso de privacidade do demonstrativo hospedado (rascunho)"),
    "termos": ("termos_de_uso.md", "Termos de uso do demonstrativo hospedado (rascunho)"),
}

_TITULO = re.compile(r"^(#{2,4})\s+(\S+)\s")


def ler(topico: str) -> str:
    """Texto integral do documento `topico`. KeyError se o topico nao existe; FileNotFoundError se o pacote esta sem o arquivo."""
    arquivo = TOPICOS[topico][0]
    return (resources.files("guaraci") / "dados_docs" / arquivo).read_text(encoding="utf-8")


def secao_do_manual(numero: str) -> Optional[str]:
    """Texto da secao `numero` (ex.: "5.8", "9") do manual, do titulo ate o proximo titulo de nivel igual ou maior. None se nao existe."""
    linhas = ler("manual").splitlines()
    for i, ln in enumerate(linhas):
        m = _TITULO.match(ln)
        if m and m.group(2) == numero:
            nivel = len(m.group(1))
            fim = len(linhas)
            for j in range(i + 1, len(linhas)):
                k = _TITULO.match(linhas[j]) or re.match(r"^(#{1,4})\s", linhas[j])
                if k and len(k.group(1)) <= nivel:
                    fim = j
                    break
            return "\n".join(linhas[i:fim]).rstrip() + "\n"
    return None


#: descricao curta em ingles (os documentos em si estao escritos em portugues)
_DESC_EN: Dict[str, str] = {
    "manual": "Full program manual",
    "seguranca": "Security policy and model loading",
    "limitacoes": "Known limitations, with the evidence for each",
    "suporte": "Where to ask and how to report a bug",
    "privacidade": "Privacy notice of the hosted demo (draft)",
    "termos": "Terms of use of the hosted demo (draft)",
}


def lista(idioma: str = "PT") -> str:
    """Texto de `guaraci manual` sem argumento (`idioma` "EN" para a interface em ingles)."""
    if str(idioma).upper() == "EN":
        L = ["Documents shipped with the program (use: guaraci manual <topic>). They are written in Portuguese.", ""]
        for nome in TOPICOS:
            L.append(f"  {nome:<11} {_DESC_EN.get(nome, TOPICOS[nome][1])}")
        L += ["", "One section of the manual: guaraci manual 5.8   (section number, as in 'section 5.8')"]
        return "\n".join(L)
    L = ["Documentos que acompanham o programa (use: guaraci manual <topico>):", ""]
    for nome, (_arq, desc) in TOPICOS.items():
        L.append(f"  {nome:<11} {desc}")
    L += ["", "Uma secao do manual: guaraci manual 5.8   (numero da secao, como em 'secao 5.8')"]
    return "\n".join(L)
