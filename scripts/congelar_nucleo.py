#!/usr/bin/env python3
"""congelar_nucleo.py -- manifesto do NUCLEO: o que produz ou valida um numero, com o SHA-256 de cada arquivo.

    python scripts/congelar_nucleo.py              # grava docs/qualidade/nucleo_congelado.json
    python scripts/congelar_nucleo.py --verificar  # sai 1 e diz quais arquivos mudaram, sumiram ou entraram

Regenerar o manifesto e' o gesto de dizer "repeti a verificacao" (`scripts/verificar_hash_final.py`): o teste
`tests/test_nucleo_congelado.py` falha se um arquivo do nucleo mudar sem o manifesto ser regenerado. O hash e' do conteudo com fim de
linha normalizado (LF), para nao variar entre Windows e Linux. A lista sai de `PADROES` abaixo (docs/projeto/FASE_DESIGN.md, secao 2).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

RAIZ = Path(__file__).resolve().parents[1]
MANIFESTO = RAIZ / "docs" / "qualidade" / "nucleo_congelado.json"
PACOTE = RAIZ / "src" / "guaraci"

#: Arquivos (ou padroes) de `src/guaraci/` que produzem ou validam um numero. Fora daqui: apresentacao (catalogo, tema, menus, ajuda).
PADROES = [
    "anova_grupos.py", "anova_cli.py", "vip_estabilidade.py", "vip_cli.py",
    "preprocessamento.py", "chemometric_stats.py", "classificadores.py", "dual_spls.py", "pipeline.py", "config.py", "config_io.py",
    "validacao_estatistica.py", "agrupamento_pastas.py", "dados_io.py", "csv_robusto.py",
    "selecao_variaveis.py", "selecao_amostras.py",
    "importadores_proprietarios.py", "leitores_avancados.py", "eem_io.py", "gcms_io.py", "icp_multielementar.py", "io_registry.py",
    "hsi_*.py", "transferencia_calibracao.py", "transferir_cli.py", "sentinela_deriva.py", "monitorar_cli.py", "conformal.py",
    "identificacao.py", "predicao.py", "model_export.py", "mcr_als.py", "asca.py", "epo_glsw.py", "eem_multiway.py", "fusao_multibloco.py",
    "alinhamento_retencao.py", "deriva_qc.py", "linearity.py", "robustness.py", "portao_correcao_sinal.py", "politica_pooled_local.py",
    "plano_amostral.py", "plano_coleta.py", "amostragem_ativa.py", "auditoria_delineamento.py", "avaliacao_modelos.py", "aumento_dados.py",
    "dados_imagem.py", "eixo_espectral.py", "perfil_matriz.py",
]


def sha256_normalizado(caminho: Path) -> str:
    """SHA-256 do conteudo com CRLF -> LF."""
    return hashlib.sha256(caminho.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def arquivos_do_nucleo(pacote: Path = PACOTE) -> List[str]:
    presentes = sorted(p.name for p in pacote.glob("*.py"))
    escolhidos = sorted({n for n in presentes if any(fnmatch.fnmatch(n, pad) for pad in PADROES)})
    return [f"src/guaraci/{n}" for n in escolhidos]


def construir(raiz: Path = RAIZ) -> Dict[str, object]:
    arqs = arquivos_do_nucleo(raiz / "src" / "guaraci")
    return {"_doc": "Nucleo que produz ou valida um numero (docs/projeto/FASE_DESIGN.md, secao 2). Gerado por scripts/congelar_nucleo.py; "
                    "nao editar a mao.",
            "n_arquivos": len(arqs), "arquivos": {a: sha256_normalizado(raiz / a) for a in arqs}}


def comparar(esperado: Dict[str, object], atual: Dict[str, object]) -> Dict[str, List[str]]:
    e, a = dict(esperado.get("arquivos", {})), dict(atual.get("arquivos", {})) # type: ignore[call-overload]
    return {"mudaram": sorted(k for k in e if k in a and e[k] != a[k]), "sumiram": sorted(k for k in e if k not in a),
            "entraram": sorted(k for k in a if k not in e)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verificar", action="store_true")
    a = ap.parse_args(argv)
    atual = construir()
    if a.verificar:
        if not MANIFESTO.exists():
            print("manifesto ausente")
            return 1
        d = comparar(json.loads(MANIFESTO.read_text(encoding="utf-8")), atual)
        for k, v in d.items():
            for f in v:
                print(f"{k}: {f}")
        return 1 if any(d.values()) else 0
    MANIFESTO.parent.mkdir(parents=True, exist_ok=True)
    MANIFESTO.write_text(json.dumps(atual, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"gravado: {MANIFESTO.relative_to(RAIZ).as_posix()} ({atual['n_arquivos']} arquivos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
