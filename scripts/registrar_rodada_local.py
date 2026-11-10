#!/usr/bin/env python3
"""registrar_rodada_local.py -- roda a conformidade dos leitores com ARQUIVOS REAIS de fabricante (so' no checkout do mantenedor) e
registra o RESULTADO, sem nenhum arquivo, nome de arquivo ou caminho.

    python scripts/registrar_rodada_local.py            # roda validacao/tests/test_formatos_reais_locais.py e grava o registro
    python scripts/registrar_rodada_local.py --verificar # sai 1 se o registro nao cobre o codigo atual dos leitores

O registro (`docs/qualidade/rodada_local_formatos_reais.json`) guarda: contagens (passaram/falharam/pulados), o resultado de cada teste (nome
do teste e desfecho, nunca o arquivo lido), o hash SHA-256 desse resultado e o hash do CODIGO dos leitores que foi testado. Se os leitores
mudarem depois, `--verificar` acusa que a rodada local ficou velha. Sem os arquivos reais (pasta `validacao/fixtures_locais` ausente),
todos os testes pulam e o registro diz isso: nao e' aprovacao.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

RAIZ = Path(__file__).resolve().parents[1]
TESTE = "validacao/tests/test_formatos_reais_locais.py"
REGISTRO = RAIZ / "docs" / "qualidade" / "rodada_local_formatos_reais.json"
LEITORES = ["src/guaraci/importadores_proprietarios.py", "src/guaraci/leitores_avancados.py", "src/guaraci/eem_io.py",
            "src/guaraci/gcms_io.py", TESTE]


def hash_do_codigo(raiz: Path = RAIZ) -> str:
    """SHA-256 do conteudo (fim de linha normalizado) dos leitores e do proprio teste de conformidade."""
    h = hashlib.sha256()
    for rel in LEITORES:
        h.update(rel.encode("utf-8"))
        h.update((raiz / rel).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def resultados_do_junit(caminho: Path) -> List[Dict[str, str]]:
    """[{teste, desfecho}] ordenado por nome; desfecho = passou | falhou | pulado. O motivo do pulo nao entra (pode citar caminho)."""
    out = []
    for caso in ET.parse(caminho).getroot().iter("testcase"):
        if caso.find("failure") is not None or caso.find("error") is not None:
            desfecho = "falhou"
        elif caso.find("skipped") is not None:
            desfecho = "pulado"
        else:
            desfecho = "passou"
        out.append({"teste": caso.get("name", ""), "desfecho": desfecho})
    return sorted(out, key=lambda r: r["teste"])


def montar_registro(resultados: List[Dict[str, str]], codigo: str) -> Dict[str, object]:
    contagem = {d: sum(1 for r in resultados if r["desfecho"] == d) for d in ("passou", "falhou", "pulado")}
    canonico = json.dumps(resultados, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"hash_do_codigo_testado": codigo, "total": len(resultados), "passaram": contagem["passou"],
            "falharam": contagem["falhou"], "pulados": contagem["pulado"],
            "hash_do_resultado": hashlib.sha256(canonico.encode("utf-8")).hexdigest(),
            "resultados": resultados,
            "aprovada": bool(resultados) and contagem["falhou"] == 0 and contagem["pulado"] == 0 and contagem["passou"] > 0,
            "observacao": "arquivos reais de fabricante ficam fora do repositorio; este registro guarda so' o desfecho de cada teste"}


def registro_atual_cobre_o_codigo(raiz: Path = RAIZ) -> Optional[str]:
    """None se o registro existe, foi aprovado e e' do codigo atual dos leitores; senao, o motivo."""
    arq = raiz / "docs" / "qualidade" / "rodada_local_formatos_reais.json"
    if not arq.is_file():
        return "nao ha registro da rodada local (rode scripts/registrar_rodada_local.py no checkout com os arquivos reais)"
    r = json.loads(arq.read_text(encoding="utf-8"))
    if not r.get("aprovada"):
        return "a rodada local registrada nao foi aprovada (falhas ou testes pulados)"
    if r.get("hash_do_codigo_testado") != hash_do_codigo(raiz):
        return "os leitores mudaram depois da rodada local registrada"
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verificar", action="store_true")
    a = ap.parse_args(argv)
    if a.verificar:
        motivo = registro_atual_cobre_o_codigo()
        print(motivo or "rodada local em dia")
        return 1 if motivo else 0
    with tempfile.TemporaryDirectory() as d:
        junit = Path(d, "r.xml")
        subprocess.run([sys.executable, "-m", "pytest", TESTE, "-q", "-p", "no:cacheprovider", "-o", "addopts=", f"--junitxml={junit}"],
                       cwd=str(RAIZ), capture_output=True, text=True)
        if not junit.is_file():
            print("a rodada nao produziu resultado")
            return 2
        reg = montar_registro(resultados_do_junit(junit), hash_do_codigo())
    REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    REGISTRO.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"rodada local: {reg['passaram']} passaram, {reg['falharam']} falharam, {reg['pulados']} pulados; "
          f"hash do resultado {str(reg['hash_do_resultado'])[:12]}")
    return 0 if reg["aprovada"] else 1


if __name__ == "__main__":
    sys.exit(main())
