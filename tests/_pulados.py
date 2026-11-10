"""Guarda de testes pulados (2026-10-02).

Achado que motivou: entre duas rodadas da suite, 19 testes de validacao
publica passaram a PULAR em silencio (a variavel GUARACI_DATASETS_DIR saiu do
ambiente) e o resumo continuou "0 falhas". Pulado e' cobertura que nao rodou;
so' e' aceitavel com motivo declarado, previsto PARA AQUELE AMBIENTE e dentro
de um teto.

Funcao pura (testavel e mutavel sem pytest rodando): recebe os motivos dos
testes pulados e os motivos esperados do ambiente, devolve (contagem por
motivo, problemas). O hook em `conftest.py` so' coleta e reporta.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Mapping, Optional

ARQUIVO_ESPERADOS = Path(__file__).with_name("pulados_esperados.json")
#: Validacao local do mantenedor: fica FORA do repositorio; se for posta em `validacao/` (ignorada pelo git), os seus
#: perfis tem precedencia sobre os da suite padrao para o mesmo nome.
ARQUIVO_ESPERADOS_VALIDACAO = Path(__file__).resolve().parents[1] / "validacao" / "pulados_esperados.json"
SEM_MOTIVO = "(sem motivo declarado)"


def escolher_ambiente(env: Mapping[str, str]) -> str:
    """GUARACI_PULADOS_AMBIENTE explicito; senao 'ci' quando a variavel CI existe (setada pelo GitHub Actions); senao
    'local-completo' quando GUARACI_DATASETS_DIR existe (a validacao do mantenedor tem os dados); senao 'local-sem-dados'."""
    explicito = env.get("GUARACI_PULADOS_AMBIENTE", "").strip()
    if explicito:
        return explicito
    if env.get("CI"):
        return "ci"
    return "local-completo" if env.get("GUARACI_DATASETS_DIR") else "local-sem-dados"


def carregar_esperados(ambiente: str, caminho: Optional[Path] = None) -> dict:
    """Motivos esperados do perfil. Sem `caminho`, procura o perfil na validacao local do mantenedor (se presente) e depois
    na suite padrao."""
    arquivos = [caminho] if caminho else [ARQUIVO_ESPERADOS_VALIDACAO, ARQUIVO_ESPERADOS]
    conhecidos = []
    for arq in arquivos:
        if not Path(arq).exists():
            continue
        ambientes = json.loads(Path(arq).read_text(encoding="utf-8"))["ambientes"]
        if ambiente in ambientes:
            return ambientes[ambiente]["motivos"]
        conhecidos += list(ambientes)
    raise KeyError(f"ambiente de pulados desconhecido: {ambiente!r} "
                   f"(conhecidos: {sorted(conhecidos)})")


def classificar(motivo: str, esperados: dict) -> str:
    """Nome do motivo esperado cujo padrao casa, ou o proprio texto (motivo
    NOVO). Motivo vazio nunca casa nada."""
    texto = (motivo or "").strip()
    if not texto:
        return SEM_MOTIVO
    for nome, regra in esperados.items():
        if re.search(regra["padrao"], texto):
            return nome
    return texto


def avaliar(motivos: list[str], esperados: dict) -> tuple[Counter, list[str]]:
    contagem: Counter = Counter(classificar(m, esperados) for m in motivos)
    problemas = []
    for nome, n in sorted(contagem.items()):
        if nome not in esperados:
            problemas.append(f"motivo de pulo NAO previsto ({n}x): {nome}")
        elif n > esperados[nome]["max"]:
            problemas.append(f"{nome}: {n} pulados, teto {esperados[nome]['max']}")
    return contagem, problemas
