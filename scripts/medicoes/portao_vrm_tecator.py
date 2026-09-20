#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""portao_vrm_tecator.py -- roda o aumento de dados VRM
(`guaraci.aumento_dados.AumentoVRM`) atraves do portao de aceite (Bloco 20)
contra o dataset PUBLICO Tecator (215 amostras reais de carne, 100 canais
NIR 850-1050nm, teor de gordura por analise quimica de referencia).

FONTE DO DADO -- achado desta rodada (registrar tambem no relatorio de
consolidacao, e' relevante alem desta tarefa): a fonte original do Tecator
(StatLib, `lib.stat.cmu.edu`) esta' bloqueada pela politica de saida de
rede desta sessao, mesma situacao do Corn (`eigenvector.com`) usado em
`scripts/medicoes/portao_emsc_osc.py`. O pacote PyPI **sktime** (licenca
BSD-3-Clause, compativel com GPL-3.0-or-later) empacota o MESMO dataset
Tecator real como arquivo `.ts` de texto simples dentro do wheel -- sem
precisar instalar `sktime` nem seus dependentes (baixado so' com
`pip download --no-deps`, os `.ts` extraidos via `zipfile` da biblioteca
padrao, e o `.whl` pode ser descartado depois). `sktime` NAO e' e nunca
deve virar dependencia do projeto -- e' usado aqui so' como container de
distribuicao de um arquivo de dado publico, com checksum pinado abaixo
para reprodutibilidade e para detectar se o pacote mudar.

LIMITACAO HONESTA (leia antes de interpretar o veredito) -- Tecator (172
treino + 43 teste = 215 amostras) e' a melhor alternativa REAL e
verificavel disponivel neste ambiente (nem o Corn nem o acervo privado
`GUARACI_DADOS_REAIS` do usuario, motivador original da proposta VRM,
estao acessiveis aqui). Isto **NAO** e' o cenario de "n pequeno" do
acervo privado (dezenas de amostras por classe) que motivou a proposta --
e' uma quantidade de dado real, mas MAIOR e sem a estrutura de replica
por `mae_id` do caso de uso original. O veredito abaixo e' evidencia REAL
(nao sintetica), mas nao substitui a validacao contra o acervo privado,
que so' pode ser rodada localmente pelo usuario fora desta sessao (mesmo
padrao ja' usado para EMSC/OSC, Passo 134,
`scripts/medicoes/portao_emsc_osc.py`).

Tambem NAO e' uma replicacao do artigo-fonte da proposta VRM (Tumoine et
al. 2026) -- aquele usa manga (Vis-NIR) e um modelo CONVOLUCIONAL; aqui
mede-se o efeito do aumento de dados implementado (inspirado, nao
identico, ver docstring de `guaraci.aumento_dados`) num PLS-R classico
sobre Tecator, que e' o modelo/dataset disponivel e verificavel neste
ambiente.

Sem grupo/replica documentado no Tecator original -- `grupos` e' trivial
(1 amostra = 1 grupo), EXCETO para as copias que o proprio VRM cria
dentro do fold de treino, que herdam o grupo do original (condicao de
seguranca nao-negociavel de `AumentoVRM`, ver
`tests/test_aumento_dados_hypothesis.py`).

Uso:
    python scripts/medicoes/portao_vrm_tecator.py

Nao requer variavel de ambiente (o dado e' baixado do PyPI, publico) --
usa `GUARACI_TECATOR_CACHE_DIR` opcionalmente para escolher onde cachear
o `.whl`/`.ts` baixados (default: pasta temporaria do sistema), para nao
rebaixar a cada execucao.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Tuple

_RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ / "src"))

import numpy as np  # noqa: E402
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from guaraci.aumento_dados import AumentoVRM  # noqa: E402
from guaraci.portao_correcao_sinal import avaliar_correcao_sinal  # noqa: E402
from guaraci.validacao_estatistica import StableStratifiedGroupKFold  # noqa: E402

SKTIME_VERSION = "1.1.0"
# Checksums pinados -- confirmados por WebSearch/download nesta sessao
# (2026-09-20). Se o pacote no PyPI mudar, o download e' REJEITADO (nunca
# usa dado nao verificado silenciosamente).
SHA256_WHEEL = "6af5430777aa56aa85b2a5c1c5363e7f4a468f666737aaf178ae3f941c3dd6c4"
SHA256_TRAIN = "e38de03007d2d6f29181b07a972fb2048c19ea67fe92df8cf43818f1dcb45f95"
SHA256_TEST = "3d06319c07274e6236d4b4d39e2d2a31e172a8a86510ab72ff50724463a22698"

_CAMINHO_NO_WHEEL = {
    "TRAIN": "sktime/datasets/data/Tecator/Tecator_TRAIN.ts",
    "TEST": "sktime/datasets/data/Tecator/Tecator_TEST.ts",
}


def _sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _cache_dir() -> Path:
    base = os.environ.get("GUARACI_TECATOR_CACHE_DIR") \
        or str(Path(tempfile.gettempdir()) / "guaraci_tecator_cache")
    caminho = Path(base)
    caminho.mkdir(parents=True, exist_ok=True)
    return caminho


def _baixar_wheel_sktime(cache: Path) -> Path:
    caminho_whl = cache / f"sktime-{SKTIME_VERSION}-py3-none-any.whl"
    if not caminho_whl.exists() or _sha256(caminho_whl) != SHA256_WHEEL:
        print(f"Baixando sktime=={SKTIME_VERSION} (--no-deps, so' como fonte "
              "do arquivo de dado Tecator -- NAO e' dependencia do projeto)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "download", "--no-deps",
             "-d", str(cache), f"sktime=={SKTIME_VERSION}"],
            check=True)
    sha_obtido = _sha256(caminho_whl)
    if sha_obtido != SHA256_WHEEL:
        raise SystemExit(
            f"Checksum do wheel sktime nao bate (esperado {SHA256_WHEEL}, "
            f"obtido {sha_obtido}) -- pacote alterado ou download corrompido; "
            "abortando em vez de usar dado nao verificado.")
    return caminho_whl


def _extrair_e_verificar(caminho_whl: Path, cache: Path) -> Dict[str, Path]:
    saida: Dict[str, Path] = {}
    with zipfile.ZipFile(caminho_whl) as z:
        for nome, caminho_zip in _CAMINHO_NO_WHEEL.items():
            destino = cache / f"Tecator_{nome}.ts"
            if not destino.exists():
                with z.open(caminho_zip) as origem, open(destino, "wb") as f:
                    f.write(origem.read())
            saida[nome] = destino
    checagens = {"TRAIN": SHA256_TRAIN, "TEST": SHA256_TEST}
    for nome, sha_esperado in checagens.items():
        sha_obtido = _sha256(saida[nome])
        if sha_obtido != sha_esperado:
            raise SystemExit(
                f"Checksum de Tecator_{nome}.ts nao bate (esperado "
                f"{sha_esperado}, obtido {sha_obtido}) -- abortando.")
    return saida


def _parse_ts(caminho: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parser minimo do formato `.ts` (sktime/aeon): linhas `#` (comentario)
    e `@...` (metadado) sao ignoradas; cada linha de dado e'
    `v1,v2,...,v100:alvo`."""
    X_linhas = []
    y_linhas = []
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith("#") or linha.startswith("@"):
                continue
            valores_str, alvo_str = linha.rsplit(":", 1)
            X_linhas.append([float(v) for v in valores_str.split(",")])
            y_linhas.append(float(alvo_str))
    return np.array(X_linhas, dtype=float), np.array(y_linhas, dtype=float)


def carregar_tecator_real() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baixa (ou reusa cache com checksum verificado) o Tecator real via o
    pacote PyPI sktime -- ver docstring do modulo para a fonte alternativa
    ao StatLib bloqueado e os checksums. Devolve `(X, y, grupos)` com as
    215 amostras (172 TRAIN + 43 TEST) combinadas -- a divisao treino/teste
    original do sktime NAO e' usada aqui (o portao faz sua PROPRIA CV
    group-aware sobre o conjunto completo)."""
    cache = _cache_dir()
    caminho_whl = _baixar_wheel_sktime(cache)
    arquivos = _extrair_e_verificar(caminho_whl, cache)
    X_tr, y_tr = _parse_ts(arquivos["TRAIN"])
    X_te, y_te = _parse_ts(arquivos["TEST"])
    X = np.vstack([X_tr, X_te])
    y = np.concatenate([y_tr, y_te])
    grupos = np.array([f"tecator_{i}" for i in range(len(X))])
    return X, y, grupos


def _rodar(
        X: np.ndarray, y: np.ndarray, grupos: np.ndarray, seed: int, *,
        usar_vrm: bool, n_splits: int = 5, n_componentes: int = 10,
        vrm_kwargs: dict | None = None,
        ) -> float:
    """Roda 1 particao (`seed`) do pipeline INTEIRO (split group-aware +
    [VRM opcional so' no treino] + PLS-R + predict + RMSEP), como exige o
    contrato de `avaliar_correcao_sinal`. NAO usa
    `avaliar_correcao_sinal_pls` (o atalho existente): aquele assume um
    transformer que PRESERVA o numero de linhas (`fit_transform(X_tr)`
    seguido de `transform(X_va)`), o que nao serve para um aumento de
    dados que MUDA o numero de linhas do treino -- verificado lendo
    `portao_correcao_sinal.py` antes de decidir por este `_rodar` proprio."""
    splitter = StableStratifiedGroupKFold(n_splits=n_splits, seed=seed)
    folds = list(splitter.split(np.zeros(len(X)), np.zeros(len(X)), groups=grupos))

    y_hat = np.zeros_like(y, dtype=float)
    contador = np.zeros(len(y), dtype=int)
    for idx_tr, idx_va in folds:
        X_tr, y_tr, g_tr = X[idx_tr].copy(), y[idx_tr].copy(), grupos[idx_tr].copy()
        X_va = X[idx_va].copy()

        if usar_vrm:
            av = AumentoVRM(seed=seed, **(vrm_kwargs or {}))
            X_tr, y_tr, _g_tr = av.aumentar(X_tr, y_tr, g_tr)
            # `_g_tr` (grupo herdado) nao e' usado aqui porque a validacao
            # `idx_va` ja' foi fixada ANTES do aumento, sobre o dado
            # original -- e' exatamente o padrao "aumenta so' o fold de
            # treino, prediz no MESMO fold de validacao original" exigido
            # pelo portao. A heranca de grupo em si e' garantida e testada
            # em `tests/test_aumento_dados_hypothesis.py`.

        mc = StandardScaler(with_std=False)
        X_tr_c = mc.fit_transform(X_tr)
        X_va_c = mc.transform(X_va)
        n_comp_eff = int(max(1, min(n_componentes, X_tr_c.shape[1], len(X_tr_c) - 1)))
        pls = PLSRegression(n_components=n_comp_eff, scale=False)
        pls.fit(X_tr_c, y_tr.reshape(-1, 1))
        y_hat[idx_va] += np.asarray(pls.predict(X_va_c), dtype=float).ravel()
        contador[idx_va] += 1

    contador[contador == 0] = 1
    y_hat = y_hat / contador
    erro = y - y_hat
    return float(np.sqrt(np.mean(erro ** 2)))


def main() -> None:
    print("Carregando Tecator real (via sktime, checksum verificado)...")
    X, y, grupos = carregar_tecator_real()
    print(f"  {X.shape[0]} amostras, {X.shape[1]} canais NIR, "
          f"gordura {y.min():.1f}-{y.max():.1f}%")

    vrm_kwargs = dict(
        n_aumentos=2,
        amplitude_multiplicativa=0.01,
        amplitude_baseline=0.01,
        amplitude_ruido_estruturado=0.0,
        ordem_polinomial_baseline=2,
    )
    print(f"\nParametros VRM testados: {vrm_kwargs}")

    v = avaliar_correcao_sinal(
        "VRM_tecator",
        avaliar_sem_fn=lambda seed: _rodar(X, y, grupos, seed, usar_vrm=False),
        avaliar_com_fn=lambda seed: _rodar(X, y, grupos, seed, usar_vrm=True,
                                            vrm_kwargs=vrm_kwargs),
        metrica="RMSEP", n_seeds=20)
    print("\n" + v.resumo())
    print(f"  efeito absoluto: {v.tamanho_efeito:.4g} RMSEP "
          f"({100 * v.tamanho_efeito / v.valor_sem:.3f}% relativo)")
    print(f"  efeito padronizado (~Cohen's d pareado): {v.tamanho_efeito_padronizado:.3f}")
    print("\nVer docstring do modulo e de guaraci.aumento_dados.AumentoVRM "
          "para o veredito completo, incluindo a sensibilidade a "
          "hiperparametros (amplitude_ruido_estruturado ligado PIORA nesta "
          "config -- por isso fica desligado por default).")


if __name__ == "__main__":
    main()
