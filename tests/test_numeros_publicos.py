"""Numeros publicados: (1) a unica contagem citada nos READMEs vem de `docs/cientifico/numeros_publicos.yaml` e bate com a
arvore; (2) nenhum numero derivado de conjunto de dados de terceiros aparece em documento publico.

Para (2) o teste guarda so' o SHA-256 das strings proibidas, normalizadas (minusculas, virgula -> ponto, espacos colapsados),
e procura no texto normalizado janelas do mesmo tamanho com o mesmo hash. Assim o proprio teste (que vai no sdist) nao
publica os numeros que diz esconder. Para acrescentar uma string: `_hash_normalizado("...")` e copiar (tamanho, hash).
"""
import hashlib
import re
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]
FONTE = yaml.safe_load((RAIZ / "docs" / "cientifico" / "numeros_publicos.yaml").read_text(encoding="utf-8"))

#: (tamanho, SHA-256) das strings proibidas ja' normalizadas.
_PROIBIDOS = frozenset({
    (3, "1fcb84cfc5a4bcdbfc194d75510cce0756916c808c0ad69cbd454c7cb1b5f835"),
    (5, "1400c462cd6fa9c2570825632783918e59ca957562e9d22a4465c4396506895e"),
    (5, "358af707dd386adfbc8c84f393883e33bfabbb0da01852c632b6bc84e36d82c2"),
    (5, "a8a9d3a5a4d7f0e21d15a6339499b2d276ba7bcd078dc545bd883fee461573d6"),
    (6, "e61eed607a4d4acb3656cb376a734ff51165271bbbdf89ac5b2908be9558e0a3"),
    (7, "68bad2b01b4ae0fed6e5b7df7ee354f501d4774025ff1958245abc00d0f4e83d"),
    (8, "67ee5fbeaefed0b903b6b00fa1c26b135fe30d9f49e085f1603c8624164a5806"),
    (9, "fe7b0d5e262e3547de701cd5fb1f1ef46f3b7fad3c40b025c49c8a75d87e8f05"),
    (11, "e5431f20520473be05de63ccb87108bfd629da5fd0458f90493b81a9621171e5"),
})

#: Documentos publicos ou distribuidos varridos (inclui este proprio arquivo: o teste nao pode carregar os numeros em claro).
_DOCUMENTOS = ("paper/paper.md", "MANUAL.md", "src/guaraci/dados_docs/manual.md", "docs/cientifico/limitacoes.md",
               "src/guaraci/dados_docs/limitacoes.md", "src/guaraci/catalogo_validacao.py", "README.md", "README.pt-BR.md",
               "CHANGELOG.md", "docs/cientifico/numeros_publicos.yaml", "tests/test_numeros_publicos.py")


def _normalizar(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace(",", "."))


def _hash_normalizado(s: str) -> tuple:
    n = _normalizar(s).strip()
    return len(n), hashlib.sha256(n.encode("utf-8")).hexdigest()


def _achados(texto: str, proibidos=_PROIBIDOS) -> list:
    """Posicoes (no texto normalizado) onde comeca uma string proibida."""
    t = _normalizar(texto)
    tamanhos = sorted({n for n, _h in proibidos})
    achados = []
    for i in range(len(t)):
        for n in tamanhos:
            janela = t[i:i + n]
            if len(janela) == n and (n, hashlib.sha256(janela.encode("utf-8")).hexdigest()) in proibidos:
                achados.append(i)
    return achados


def _le(rel):
    return (RAIZ / rel).read_text(encoding="utf-8")


def test_o_detector_acha_string_proibida_por_hash_e_ignora_o_resto():
    """O mecanismo funciona sem expor nenhum numero real: string de exemplo -> (tamanho, hash) -> achada no texto."""
    proibidos = {_hash_normalizado("Erro 9,87 ±")}
    assert _achados("antes  erro 9.87 ± depois", proibidos) == [6]     # espacos colapsados antes de procurar
    assert _achados("erro 9.88 ±", proibidos) == []
    assert _hash_normalizado("1,5") == _hash_normalizado("1.5")


def test_lista_de_proibidos_e_valida():
    assert _PROIBIDOS and all(isinstance(n, int) and n > 0 and re.fullmatch(r"[0-9a-f]{64}", h) for n, h in _PROIBIDOS)


def test_nenhum_numero_derivado_de_conjunto_de_terceiros_em_documento_publico():
    for rel in _DOCUMENTOS:
        assert not _achados(_le(rel)), f"{rel}: numero derivado de conjunto de dados de terceiros em documento publico"


def test_fonte_cobre_so_as_contagens_proprias():
    """O yaml nao volta a guardar numero de conjunto de dados: so' a contagem que os READMEs citam."""
    assert set(FONTE) == {"perfis_de_matriz_distribuidos"}


def test_paper_tem_as_secoes_do_joss_e_respeita_o_limite_de_palavras():
    """B7 do parecer de 2026-09-25 (JOSS exige estas secoes; corpo <= 1750 palavras)."""
    txt = _le("paper/paper.md")
    for secao in ("Summary", "Statement of need", "State of the field", "Software design",
                  "Research impact statement", "AI usage disclosure"):
        assert re.search(rf"^# {secao}\s*$", txt, re.M), f"paper sem a secao '{secao}'"
    corpo = txt.split("# Summary", 1)[1].split("# References", 1)[0]
    corpo = re.sub(r"<!--.*?-->", "", corpo, flags=re.S)
    assert len(corpo.split()) <= 1750, len(corpo.split())


def test_todo_citekey_do_paper_existe_no_bib():
    txt = _le("paper/paper.md")
    bib = _le("paper/paper.bib")
    chaves = set(re.findall(r"@([A-Za-z0-9_]+)", " ".join(re.findall(r"\[[^\]]*@[^\]]*\]", txt))))
    definidas = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
    assert chaves <= definidas, chaves - definidas


def test_contagens_do_readme_batem_com_o_repositorio():
    """m-09: '8 perfis' sai de UMA fonte; os READMEs nao carregam numero de validacao."""
    n = FONTE["perfis_de_matriz_distribuidos"]
    assert len(list((RAIZ / "src/guaraci/perfis_matriz").glob("*.yaml"))) == n
    assert f"{n} shipped" in _le("README.md") and f"{n} distribuídos" in _le("README.pt-BR.md")
    for rel in ("README.md", "README.pt-BR.md"):
        assert not re.search(r"RMSEP\s*\d", _le(rel)), f"{rel}: numero de validacao no README"
