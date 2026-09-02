"""agrupamento_pastas.py — Hierarquia de 3 niveis de garantia de
agrupamento por amostra fisica (Bloco 8, 2026-08-25), extraida de
`dados_imagem.py` no Passo 111 (INSTRUCAO_HSI_DADO_PROPRIO.md) para ser
reaproveitada TAMBEM pelo modo `hsi` (`hsi_io.load_hsi_folder_dataset`)
sem duplicar a logica -- a unica coisa que muda entre "fotos" e "cubos
hiperespectrais" e' a extensao de arquivo que conta como 1 gravacao,
por isso toda funcao aqui recebe `extensoes` como parametro em vez de
assumir `.jpg/.png/...`.

Convencao (identica em ambos os modos que usam este modulo):
  - "high"   — subpasta por amostra fisica: cada subpasta de classe
    contem SO' subpastas (nunca arquivo solto), uma por amostra fisica;
    cada gravacao dentro dela e' uma replica do mesmo grupo.
  - "medium" — CSV de associacao manual (`amostras.csv` por padrao) na
    RAIZ da pasta de dados, colunas `arquivo,id_amostra`. TODO arquivo
    carregado precisa aparecer no CSV -- cobertura parcial e' erro.
  - "none"   — nem subpasta por amostra nem CSV: aceita processar mesmo
    assim, mas quem chama deve declarar a limitacao explicitamente (log,
    model card/relatorio, manifesto) -- nunca silenciosamente.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence

import pandas as pd

__all__ = [
    "GROUPING_HIGH",
    "GROUPING_MEDIUM",
    "GROUPING_NONE",
    "NOME_CSV_AMOSTRAS",
    "listar_arquivos_por_extensao",
    "tem_arquivo_direto_ou_em_subpasta",
    "detectar_subpastas_por_extensao",
    "subpasta_e_grupo_de_amostras",
    "detectar_nivel_high",
    "detectar_nivel_medium",
]

GROUPING_HIGH = "high"
GROUPING_MEDIUM = "medium"
GROUPING_NONE = "none"

#: Nome do CSV de associacao manual (nivel "medium"), procurado na raiz da
#: pasta de dados. Nao configuravel -- mesma convencao em todo o projeto.
NOME_CSV_AMOSTRAS = "amostras.csv"


def listar_arquivos_por_extensao(pasta: str, extensoes: Sequence[str]) -> List[str]:
    """Busca arquivos por extensao. Usa um set p/ deduplicar: em sistemas
    de arquivo case-insensitive (Windows, macOS default), buscar "*.ext"
    e "*.EXT" separadamente devolve o MESMO arquivo duas vezes."""
    encontrados: set = set()
    for ext in extensoes:
        encontrados.update(glob.glob(os.path.join(pasta, f"*{ext}")))
        encontrados.update(glob.glob(os.path.join(pasta, f"*{ext.upper()}")))
    return sorted(encontrados)


def tem_arquivo_direto_ou_em_subpasta(caminho: str, extensoes: Sequence[str]) -> bool:
    """True se `caminho` tem arquivo solto OU (nivel "high") subpastas de
    amostra que por sua vez tem arquivo -- sem isso, uma classe organizada
    em subpasta-por-amostra seria invisivel para `detectar_subpastas_por_
    extensao` (a classe pareceria vazia, ja que so' checava arquivo DIRETO)."""
    if listar_arquivos_por_extensao(caminho, extensoes):
        return True
    return any(os.path.isdir(os.path.join(caminho, n))
               and listar_arquivos_por_extensao(os.path.join(caminho, n), extensoes)
               for n in os.listdir(caminho))


def detectar_subpastas_por_extensao(raiz: str, extensoes: Sequence[str]) -> List[str]:
    """Subpastas (1 por classe) que contem >=1 arquivo da(s) extensao(oes)
    dada(s), direto ou dentro de subpasta de amostra (nivel "high")."""
    if not os.path.isdir(raiz):
        return []
    subpastas = []
    for nome in sorted(os.listdir(raiz)):
        caminho = os.path.join(raiz, nome)
        if os.path.isdir(caminho) and tem_arquivo_direto_ou_em_subpasta(caminho, extensoes):
            subpastas.append(caminho)
    return subpastas


def subpasta_e_grupo_de_amostras(caminho_classe: str, extensoes: Sequence[str]) -> bool:
    """True se `caminho_classe` contem SO' subpastas (cada uma = 1 amostra
    fisica), nunca arquivo solto. False se tiver ao menos 1 arquivo solto
    (mistura de niveis nao e' suportada -- ambigua)."""
    entradas = [os.path.join(caminho_classe, n)
                for n in os.listdir(caminho_classe)]
    exts_lower = tuple(e.lower() for e in extensoes)
    arquivos_soltos = [e for e in entradas if os.path.isfile(e)
                       and e.lower().endswith(exts_lower)]
    subpastas_amostra = [e for e in entradas if os.path.isdir(e)
                         and listar_arquivos_por_extensao(e, extensoes)]
    return not arquivos_soltos and bool(subpastas_amostra)


def detectar_nivel_high(subpastas_classe: List[str], extensoes: Sequence[str],
                         ) -> Optional[Dict[str, str]]:
    """Nivel "high": cada subpasta de CLASSE contem so' subpastas de
    AMOSTRA FISICA (nunca arquivo solto). Se TODA subpasta de classe
    satisfizer isso, devolve {caminho_arquivo: grupo_id}; senao None (cai
    p/ nivel "medium"). Grupo_id e' qualificado por classe
    ("Classe/Amostra") p/ nunca colidir entre classes com o mesmo nome de
    amostra."""
    if not subpastas_classe:
        return None
    if not all(subpasta_e_grupo_de_amostras(sp, extensoes) for sp in subpastas_classe):
        return None
    grupos: Dict[str, str] = {}
    for sp in subpastas_classe:
        classe = os.path.basename(sp)
        for nome_amostra in sorted(os.listdir(sp)):
            caminho_amostra = os.path.join(sp, nome_amostra)
            if not (os.path.isdir(caminho_amostra)
                    and listar_arquivos_por_extensao(caminho_amostra, extensoes)):
                continue
            grupo_id = f"{classe}/{nome_amostra}"
            for arq in listar_arquivos_por_extensao(caminho_amostra, extensoes):
                grupos[arq] = grupo_id
    return grupos


def detectar_nivel_medium(pasta_raiz: str, arquivos: List[str],
                           nome_csv: str = NOME_CSV_AMOSTRAS,
                           ) -> Optional[Dict[str, str]]:
    """Nivel "medium": CSV (default `amostras.csv`) na raiz da pasta de
    dados, colunas `arquivo,id_amostra`. `arquivo` e' o caminho RELATIVO a
    `pasta_raiz` (separador "/", como grava `os.path.relpath` normalizado).
    Cobertura parcial e' erro explicito -- nunca processamento parcial."""
    caminho_csv = os.path.join(pasta_raiz, nome_csv)
    if not os.path.isfile(caminho_csv):
        return None
    df_csv = pd.read_csv(caminho_csv)
    colunas_faltando = {"arquivo", "id_amostra"} - set(df_csv.columns)
    if colunas_faltando:
        raise ValueError(
            f"{caminho_csv}: faltam as colunas {sorted(colunas_faltando)}. "
            f"Esperado: 'arquivo,id_amostra' (uma linha por gravacao).")
    mapa_csv = {str(r["arquivo"]).replace("\\", "/"): str(r["id_amostra"])
                for _, r in df_csv.iterrows()}
    rel = {arq: os.path.relpath(arq, pasta_raiz).replace("\\", "/")
           for arq in arquivos}
    sem_cobertura = [rel[a] for a in arquivos if rel[a] not in mapa_csv]
    if sem_cobertura:
        raise ValueError(
            f"{caminho_csv} existe mas nao cobre {len(sem_cobertura)} "
            f"gravacao(oes) do dataset -- nenhuma foi processada. Arquivos "
            f"sem entrada no CSV: {', '.join(sorted(sem_cobertura))}")
    return {arq: mapa_csv[rel[arq]] for arq in arquivos}
