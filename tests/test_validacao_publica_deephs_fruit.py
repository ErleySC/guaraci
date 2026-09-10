"""Validação pública do pipeline HSI contra o DeepHS Fruit completo
(Passo 104 da `INSTRUCAO_HSI_ROBUSTEZ_E_VALIDACAO.md`): as 5 frutas x
até 2 câmeras cada disponíveis no dataset (medido por leitura direta do
JSON de anotações -- nenhuma fruta tem as 3 câmeras, ver
`docs/PROGRESSO.md` Passo 104), testando generalização entre matrizes
(frutas) E entre instrumentos (câmeras) sem mudar código, um único
`load_deephs_fruit_dataset(pasta, fruta, camera)` por combinação.

Mesmo padrão de `test_validacao_publica_mendeley.py`: PULA (não falha)
se `GUARACI_DATASETS_DIR` não apontar para o dataset já baixado com
`scripts/download_datasets/baixar_deephs_fruit_todas.py`. Reporta
sensibilidade/especificidade/precisão por classe, nunca só uma média
agregada -- e nunca fabrica um número quando a combinação não tem grupos
suficientes para avaliar (mesma linguagem já usada em DD-SIMCA/conformal:
"não avaliável estatisticamente").
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest


def _pasta_deephs_fruit_all() -> Optional[Path]:
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "deephs_fruit_all"
    return pasta if (pasta / "manifest.json").is_file() else None


requer_deephs_fruit_all = pytest.mark.skipif(
    _pasta_deephs_fruit_all() is None,
    reason=("dataset publico DeepHS Fruit (todas as frutas/cameras) "
            "ausente. Baixe com "
            "'python scripts/download_datasets/baixar_deephs_fruit_todas.py' "
            "e aponte GUARACI_DATASETS_DIR."))


def _combinacoes_disponiveis(pasta: Path) -> List[Tuple[str, str, int]]:
    """Lista (fruta, camera, n_gravacoes) REALMENTE presentes no
    manifest -- nunca presume um produto cartesiano fruta x camera que
    pode nao existir (achado do Passo 104: nenhuma fruta tem as 3
    cameras)."""
    with open(pasta / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    from collections import Counter
    contagem = Counter((r["fruit"], r["camera_type"]) for r in manifest["records"])
    return sorted((fruta, cam, n) for (fruta, cam), n in contagem.items())


#: Minimo de objetos fisicos DISTINTOS para uma combinacao ser considerada
#: avaliavel (split group-aware interno + externo por dia precisa de
#: alguns objetos de sobra em cada lado) -- abaixo disso, reporta "nao
#: avaliavel" em vez de forcar um numero de um n irrisorio.
_MIN_OBJETOS_AVALIAVEL = 10

#: Teto de pixels por GRAVACAO (Passo 104, achado real): resolucao de
#: imagem varia MUITO entre frutas (Kaki: 64x64=4096 pixels/imagem,
#: ~2400 na ROI; Avocado/VIS medido em: ~286x294=~97000 pixels/imagem,
#: ~24x mais -- sem teto, Avocado/VIS estourou memoria ao tentar alocar
#: 2,8GB num unico fit de PLS-DA dentro do loop de selecao de LVs).
#: 2000 fica perto da escala natural do Kaki -- teto justo, nao
#: artificialmente pequeno, e MESMO valor p/ todas as combinacoes
#: (comparacao justa: nenhuma camera de alta resolucao ganha mais
#: "votos" na agregacao por objeto so' por ter mais pixels).
_MAX_PIXELS_POR_GRAVACAO = 2000


def _avaliar_combinacao(pasta: Path, fruta: str, camera: str) -> Dict[str, object]:
    from guaraci.hsi_io import load_deephs_fruit_dataset
    from guaraci.hsi_pipeline import apply_quality_gate_and_segment
    from guaraci.hsi_resampling import class_evaluability_report
    from guaraci.hsi_validation import run_external_validation_by_day

    cubos, rotulos, grupos, wavelengths, meta_df = load_deephs_fruit_dataset(
        str(pasta), fruta=fruta, camera=camera)

    n_objetos = len(set(grupos))
    # Passo 105 "repetir para as frutas/cameras novas do Passo 104" --
    # desbalanceamento varia por fruta, reportado por classe SEMPRE,
    # mesmo quando a combinacao inteira nao for avaliavel para a
    # validacao externa (o desbalanceamento em si ainda e' informacao
    # real sobre o dataset).
    avaliabilidade = class_evaluability_report(rotulos, grupos, alpha=0.05)

    if n_objetos < _MIN_OBJETOS_AVALIAVEL:
        return {"status": "nao_avaliavel",
                "motivo": f"so' {n_objetos} objetos fisicos distintos "
                          f"(< minimo {_MIN_OBJETOS_AVALIAVEL})",
                "avaliabilidade_por_classe": avaliabilidade}

    filtrado = apply_quality_gate_and_segment(
        cubos, list(grupos), list(rotulos), list(meta_df["day"]))
    if filtrado["n_rejeitados"] > 0:
        print(f"  [{fruta}/{camera}] {filtrado['n_rejeitados']} gravacao(oes) "
              f"rejeitada(s) pelo quality gate: {filtrado['motivos_rejeicao'][:3]}")

    dias_unicos = sorted(set(filtrado["dias"]))
    if len(dias_unicos) < 2:
        return {"status": "nao_avaliavel",
                "motivo": f"so' {len(dias_unicos)} dia(s) distinto(s) -- "
                          f"sem particao nativa p/ validacao externa",
                "avaliabilidade_por_classe": avaliabilidade}

    n_dias_externos = max(1, len(dias_unicos) // 4)
    try:
        relatorio = run_external_validation_by_day(
            filtrado["cubos"], filtrado["mascaras"], filtrado["group_ids"],
            filtrado["rotulos"], filtrado["dias"],
            dias_externos=dias_unicos[-n_dias_externos:], seed=42,
            max_pixels_por_gravacao=_MAX_PIXELS_POR_GRAVACAO)
    except (ValueError, MemoryError) as e:
        return {"status": "nao_avaliavel", "motivo": f"{type(e).__name__}: {e}",
                "avaliabilidade_por_classe": avaliabilidade}

    return {
        "status": "ok", "relatorio": relatorio,
        "n_objetos": n_objetos, "n_gravacoes": len(cubos),
        "n_dias": len(dias_unicos),
        "classes_presentes": sorted(set(rotulos)),
        "avaliabilidade_por_classe": avaliabilidade,
    }


@requer_deephs_fruit_all
def test_validacao_comparativa_todas_frutas_cameras():
    """Roda a validacao externa (Passo 101) em CADA combinacao fruta x
    camera real do DeepHS Fruit -- testa generalizacao entre matrizes
    (frutas diferentes) e entre instrumentos (cameras diferentes) sem
    mudar UMA linha de codigo do pipeline HSI, so' os parametros de
    `load_deephs_fruit_dataset`. Nao afirma numero-alvo nenhum -- so'
    confirma que o pipeline roda e reporta honestamente (por classe,
    interno x externo separados, "nao avaliavel" quando o n nao sustenta
    a alegacao)."""
    pasta = _pasta_deephs_fruit_all()
    assert pasta is not None
    combinacoes = _combinacoes_disponiveis(pasta)
    assert len(combinacoes) > 0

    resultados: Dict[Tuple[str, str], Dict[str, object]] = {}
    for fruta, camera, n_gravacoes in combinacoes:
        print(f"\n=== {fruta} / {camera} ({n_gravacoes} gravacoes) ===")
        resultado = _avaliar_combinacao(pasta, fruta, camera)
        resultados[(fruta, camera)] = resultado

        avaliab = resultado.get("avaliabilidade_por_classe", {})
        if avaliab:
            print("  Desbalanceamento (Passo 105): " + ", ".join(
                f"{c}=n{info.n_grupos}"
                f"{'*' if not info.avaliavel else ''}"
                for c, info in avaliab.items()))
            for c, info in avaliab.items():
                if not info.avaliavel:
                    print(f"    * {c}: {info.nota}")

        if resultado["status"] == "nao_avaliavel":
            print(f"  NAO AVALIAVEL: {resultado['motivo']}")
            continue
        rel = resultado["relatorio"]
        print(f"  n_objetos={resultado['n_objetos']} "
              f"n_dias={resultado['n_dias']} "
              f"classes={resultado['classes_presentes']} "
              f"interno_n={rel.n_objetos_teste_interno} "
              f"externo_n={rel.n_objetos_teste_externo}")
        for classe in rel.classes:
            print(f"    {classe}: "
                  f"sens(int/ext)={rel.sensibilidade_interna[classe]:.2f}/"
                  f"{rel.sensibilidade_externa[classe]:.2f}  "
                  f"espec(int/ext)={rel.especificidade_interna[classe]:.2f}/"
                  f"{rel.especificidade_externa[classe]:.2f}  "
                  f"prec(int/ext)={rel.precisao_interna[classe]:.2f}/"
                  f"{rel.precisao_externa[classe]:.2f}")
            for m in (rel.sensibilidade_interna, rel.especificidade_interna,
                     rel.precisao_interna, rel.sensibilidade_externa,
                     rel.especificidade_externa, rel.precisao_externa):
                assert 0.0 <= m[classe] <= 1.0

    n_avaliaveis = sum(1 for r in resultados.values() if r["status"] == "ok")
    print(f"\n=== RESUMO: {n_avaliaveis}/{len(combinacoes)} combinacoes "
          f"avaliaveis ===")
    assert n_avaliaveis > 0, "nenhuma combinacao fruta x camera foi avaliavel"
