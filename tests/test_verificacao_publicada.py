"""A verificacao final publicada e' do hash que se publica: do hash verificado ate o HEAD so' mudam arquivos GERADOS de docs/qualidade/."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List

RAIZ = Path(__file__).resolve().parents[1]
GERADOS = ("docs/qualidade/",)
EXT_GERADA = (".md", ".json")


def fora_do_gerado(arquivos: List[str]) -> List[str]:
    """Arquivos de um diff que NAO sao saida gerada de docs/qualidade/."""
    return [a for a in arquivos if not (a.startswith(GERADOS) and a.endswith(EXT_GERADA))]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True, encoding="utf-8")


def test_so_arquivo_gerado_de_qualidade_pode_mudar_depois_da_verificacao():
    assert fora_do_gerado(["docs/qualidade/estado_do_projeto.md", "docs/qualidade/resultado_abc.json"]) == []
    assert fora_do_gerado(["src/guaraci/x.py"]) == ["src/guaraci/x.py"]
    assert fora_do_gerado(["docs/qualidade/script.py"]) == ["docs/qualidade/script.py"]
    assert fora_do_gerado(["docs/cientifico/limitacoes.md"]) == ["docs/cientifico/limitacoes.md"]
    assert fora_do_gerado(["docs/qualidade_falsa/x.md"]) == ["docs/qualidade_falsa/x.md"]
    assert fora_do_gerado([]) == []


def test_resultado_publicado_e_do_hash_a_publicar():
    """Sem `resultado_<hash>.json` ainda nao ha o que conferir (a verificacao final ainda nao rodou). Com ele: 0 falhas, HEAD estavel,
    o hash existe no repositorio e nada fora de docs/qualidade/ mudou depois dele."""
    resultados = sorted((RAIZ / "docs" / "qualidade").glob("resultado_*.json"))
    problemas = []
    for arq in resultados:
        r = json.loads(arq.read_text(encoding="utf-8"))
        h = r["hash"]
        if r.get("falhas") != 0 or r.get("head_estavel") is not True:
            problemas.append(f"{arq.name}: falhas={r.get('falhas')} head_estavel={r.get('head_estavel')}")
            continue
        if _git("cat-file", "-e", f"{h}^{{commit}}").returncode != 0:
            problemas.append(f"{arq.name}: o hash {h[:12]} nao existe neste repositorio")
            continue
        mudou = _git("diff", "--name-only", f"{h}..HEAD").stdout.split()
        fora = fora_do_gerado(mudou)
        if fora:
            problemas.append(f"{arq.name}: depois de {h[:12]} mudaram arquivos que nao sao gerados: {fora[:5]}")
    assert not problemas, problemas
