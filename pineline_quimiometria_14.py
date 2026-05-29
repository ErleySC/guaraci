"""Pipeline quimiometrico publication-quality: PCA, HCA, PLS-DA, DD-SIMCA, PLS reg.

# ========================= CHANGELOG =========================
# v08 — base: Sprints 1-3, parse_title, GroupKFold mae_id, truncamento espectral
# v09 — autoscaling default, ddsimca_n_components=5, PALETA 14 cores
# v10 — 2026-05-28 — max_lvs=30; ddsimca_n_components=7;
#                    C2: comparar_pipelines usa max_lv=cfg.max_lvs (era min(8,..))
# v11 — 2026-05-28 — C3: HCA dendrograma (Ward); C4: DD-SIMCA one-class
#                    (treina so em puros, sens/esp); C5: N3 PLS reg GroupKFold
#                    por mae_id; C6: outliers T2 por classe no resumo
# v12 — 2026-05-28 — M1: marcadores puros(*)/adulterados(o) nos score plots;
#                    M2: sens/esp no titulo dos acceptance plots DD-SIMCA
# v13 — 2026-05-28 — M3: anotacao quimica de bandas VIP; M4: accuracy por
#                    classe no resumo_modelo.txt
# v14 — 2026-05-28 — DESCOBERTA: MSC->SG+MC = 0.923 bal.acc na base completa
#                    (1807) vs autoscaling 0.472 (vantagem do AUTO era
#                    artefato do subset 80%). Mudancas:
#                    (1) preset "msc_sg_mc" em construir_preprocessador;
#                    (2) preprocessamento_padrao default = "msc_sg_mc";
#                    (3) frac_holdout default = 0.20;
#                    (4) gerar_nome_saida case "msc_sg_mc" -> "MSC-SGd-MC";
#                    (5) M1: estrelas -> circulo borda preta (sem poluir 1807pts);
#                    (6) DD-SIMCA volta a treinar em TODOS (3 puros/classe
#                        inviabiliza one-class; requer >=15 puros/classe)
# v15 — 2026-05-28 — (1) holdout_preserva_puros=True: puros sempre no treino
#                        (resolve "puros=0" em 4 classes apos holdout);
#                    (2) aviso automatico "LVs no teto" (console + resumo);
#                    (3) DD-SIMCA acceptance plot em escala LOG-LOG
#                        (resolve dados esmagados no canto; padrao Pomerantsev)
# v16 — 2026-05-28 — Organizacao/visualizacao:
#                    (1) salvar() aceita subpasta; (2) fig3 Hotelling T2 em
#                    escala log (Y) e T2vsQ em log-log (centraliza nuvem);
#                    (3) score_contribution dividido em 2 figs (espectro +
#                    top-discriminante alta/legivel c/ legenda lateral);
#                    (4) DD-SIMCA: 14 plots individuais em subpasta ddsimca/
# v17 — 2026-05-28 — Sistema de cores de MAXIMA DISTINCAO:
#                    (1) PALETA Trubetskoy/Glasbey 20 cores (deltaE_min 27.4
#                        vs ~15 antes; elimina 3 azuis/2 verdes parecidos);
#                    (2) deteccao opcional de libs glasbey/colorcet;
#                    (3) atribuicao SEQUENCIAL deterministica (contraste
#                        adjacente) substitui hash; (4) canal secundario de
#                        FORMA (mapear_marcadores_classes, 14 shapes) p/
#                        daltonismo/P&B; (5) edge_para_cor por luminancia
# v18 — 2026-05-28 — Legibilidade de eixos: _ticks_x_inteiros() aplica
#                    MaxNLocator(integer, nbins=10) quando ha >15 ticks
#                    (selecao de LVs e PLS reg com 30-50 LVs nao sobrepoem
#                    mais os numeros); <=15 mostra todos os valores.
# v19 — 2026-05-28 — V3 HCA/VIP:
#                    (1) HCA sobre centroides em PCA(hca_n_pcs=65) — reduz
#                        ruido; (2) dendrograma eixos invertidos
#                        (orientation=top: especies no X inferior coloridas
#                        e rotacionadas, distancia no Y esquerdo);
#                    (3) fig_hca_comparacao_pipelines: painel de dendrogramas
#                        (bruto/SNV/MSC/SG1/SG2/SNV+SG1/MSC+SG1/norm);
#                    (4) interpretacao automatica de clusters (k=2);
#                    (5) VIP: y-lim no range real + caixa de estatisticas
#                        (min/max/media/dp/n>=1) — verifica dispersao real
#                    + flags Config: mostrar_marcadores_classe/elipses_grupo
# v20 — 2026-05-28 — Organizacao Q1: pasta PLSDA_OE_{nivel}_{preproc}_
#                    {YYYYMMDD_HHMMSS} com subpastas dados/ figuras/
#                    modelos/ logs/. Figuras->figuras/; metadados,
#                    identificadores, comparacao->dados/; resumo->logs/;
#                    modelo final (joblib: preproc+PLS+LB+wavenumbers)
#                    ->modelos/. Auditoria Sprint1 (A1,A2,A3,A5,A6,A11):
#                    confirmado JA implementado nas versoes anteriores.
# v22 — 2026-05-29 — Fase 0 (correcoes de rigor):
#                    B1: validar_entrada sincroniza mae_id na MESMA mascara
#                        de remocao de NaN/Inf (antes, 1 NaN desligava
#                        silenciosamente o GroupKFold = vazamento de replicas);
#                    B4: DD-SIMCA modo 'todos' nao reporta mais "esp"
#                        in-sample enganosa (esp=n/a; rotulo de modo no
#                        resumo deixa claro que sens/esp != autenticacao);
#                    B7: Q-residual no resumo com notacao adaptativa (:.4g
#                        quando <1e-3) em vez de exibir 0.0000.
# v21 — 2026-05-28 — ETAPA 4 (selecao de variaveis) + exclusao de classes:
#                    (1) Config.excluir_classes (ex: Copaiba lote anomalo);
#                    (2) iPLS (intervalos), selecao por VIP>=limiar, por SR
#                        (top fracao), sPLS-DA (NIPALS soft-selection);
#                    (3) avaliador unico _avaliar_subset_cv (CV group-aware,
#                        MC re-ajustado por fold = sem leakage);
#                    (4) figuras fig_etapa4_ipls_intervalos +
#                        fig_etapa4_comparacao_metodos; CSVs em dados/;
#                    (5) escolha do metodo mais PARCIMONIOSO (bal.acc dentro
#                        de 1% do max, menos variaveis) no resumo.
# v24 — 2026-05-29 — Sprint v24: Figuras de Publicacao:
#                    (1) fig_loadings_pca: Loading Plot PC1/PC2 (barras
#                        coloridas por sinal, eixo X invertido NIR);
#                    (2) fig_roc_auc: Curvas ROC multiclasse OvR (scores
#                        CV group-aware; AUC macro no titulo e resumo);
#                    (3) fig_splot_opls: S-Plot OPLS-DA (covariancia x
#                        correlacao com t_pred; top-N anotados; colormap
#                        RdBu_r; ref. Bylesjo 2006);
#                    (4) fig_cooman_ddsimca: Cooman's Plot DD-SIMCA (pares
#                        A x B; escala sqrt(dQ); grade de subplots;
#                        ref. Pomerantsev 2020).
#                    Integracao: aucs_roc adicionado ao resumo_modelo.txt.
# v23 — 2026-05-29 — CAMADA ACESSIVEL (sem editar codigo):
#                    (1) _CONFIG_SPEC: fonte unica que mapeia nomes amigaveis
#                        (em portugues) <-> atributos do Config, com tipo,
#                        descricao e opcoes para validacao;
#                    (2) salvar_config/carregar_config: YAML comentado em
#                        linguagem simples; defaults preservados p/ chaves
#                        ausentes; chaves desconhecidas ignoradas;
#                    (3) menu_interativo: assistente de terminal (estilo CMD)
#                        p/ editar campos, salvar/carregar e rodar sem abrir
#                        o codigo no editor;
#                    (4) __main__ novo: --rodar (usa config.yaml), --codigo
#                        (CFG legado), ou menu interativo quando em terminal;
#                    (5) config.yaml template gerado (exclui Copaiba lote
#                        anomalo, max_lvs=40). Logica do pipeline INTACTA.
# =============================================================

Inclui:
  - Pre-processamento (SNV -> SG -> MC), ordem segundo Rinnan et al. (2009)
  - Validacao cruzada estratificada repetida (sem vazamento de dados)
  - Teste de permutacao (Y-randomization)
  - Deteccao de outliers via Hotelling T2 e Q-residuals
  - VIP scores, metricas por classe, ROC multiclasse
  - Figuras separadas em padrao publication-quality
"""

import os
import re
import glob
import hashlib
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Callable, cast, Any

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from scipy.signal import savgol_filter
from scipy.stats import f as f_dist, chi2
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_val_predict, RepeatedStratifiedKFold,
    StratifiedShuffleSplit, GroupKFold, GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    balanced_accuracy_score, cohen_kappa_score, f1_score,
    precision_score, recall_score, r2_score,
)
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import norm as _norm_dist

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                CONFIGURACOES — edite APENAS aqui                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class Config:
    modo: str = "dx"                          # "dx" | "csv" | "sintetico"

    # ---- ENTRADA ----
    # Pasta-raiz com subpastas por especie (cada subpasta = 1 classe) OU
    # pasta unica com arquivos .dx (modo legado). Auto-detectado.
    # NAO versione caminhos pessoais: o caminho real vai no config.yaml
    # (gitignored). Este default e so um exemplo portavel.
    pasta_entrada: str = r"dados"
    parte_classe: int = 0                     # fallback se sem subpastas
    extrair_conc_filename: bool = True        # so usado no fallback antigo
    arquivo_csv: str = "seus_espectros.csv"
    coluna_classe: str = "classe"
    coluna_conc: Optional[str] = None
    usar_parse_title: bool = True             # usa ##TITLE= do JCAMP-DX

    # ---- SAIDA (auto-gerada pelo gerar_nome_saida) ----
    pasta_saida_raiz: str = "resultados_tcc"
    nivel: str = "N1"                         # "N1" | "N2" | "N3"
    tag:   str = ""                            # rotulo livre ex: "puros", "soja"
    pasta_saida: str = ""                     # NAO editar manualmente
    formato_saida: str = "png"                # png | pdf | svg
    dpi_salvar: int = 600
    mostrar_graficos: bool = False

    # ---- Estilo dos score plots (PCA / PLS-DA / OPLS) ----
    # Desligue para scatter "limpo" (so cor por classe, sem formas/contornos).
    mostrar_marcadores_classe: bool = False   # False -> todos circulo 'o'
    mostrar_elipses_grupo:     bool = False   # False -> sem elipse T2/convex hull

    # ---- Validacao group-aware (diferencial Q1 do TCC) ----
    # Quando True e mae_id estiver disponivel, usa GroupKFold/GroupShuffleSplit
    # para que T1/T2/T3 do mesmo ponto fiquem no mesmo fold/holdout.
    agrupar_por_mae_id: bool = True

    # ---- Truncamento espectral (FT-NIR util: 4000-10000 cm-1) -------------
    # Remove ruido de borda da FFT (0/8/16/24 cm-1 aparecem como falsos top
    # VIP quando SG derivativo amplifica essa regiao). Aplicado ANTES de
    # qualquer pre-processamento.
    wn_min: float = 4000.0
    wn_max: float = 10000.0

    # ---- Pre-processamento -----------------------------------------------
    # Preset rapido. Quando != 'custom', sobrescreve as flags individuais.
    #   'msc_sg_mc'   : MSC -> SG -> mean-centering (MELHOR: 0.923 bal.acc full)
    #   'snv_sg_mc'   : SNV -> SG -> mean-centering (Rinnan et al.)
    #   'autoscaling' : apenas StandardScaler (bom em subset, ruim no full: 0.472)
    #   'mc'          : apenas mean-centering
    #   'custom'      : honra aplicar_snv / aplicar_sg / aplicar_mc abaixo
    preprocessamento_padrao: str = "msc_sg_mc"   # v14: melhor na base completa (1807)

    aplicar_snv: bool = True
    aplicar_sg: bool = True
    sg_window: int = 25
    sg_polyorder: int = 2
    sg_deriv: int = 1
    aplicar_mc: bool = True

    max_lvs: int = 40
    n_pcs_pca: int = 10
    # HCA: dendrograma usa scores de PCA com hca_n_pcs componentes
    # (reduz ruido espectral antes do clustering). comparar_hca_pipelines
    # gera um painel de dendrogramas por pre-processamento.
    hca_n_pcs: int = 65
    comparar_hca_pipelines: bool = True
    n_splits_cv: int = 5
    n_repeats_cv: int = 3

    frac_cal: float = 0.70

    n_permutacoes: int = 200
    n_permutacoes_wold: int = 50         # Wold e diagnostico — 50 ja basta
    n_bootstrap_vip: int = 30
    n_bootstrap_bca: int = 500
    comparar_pipelines: bool = True
    executar_wold: bool = True
    executar_cv_anova: bool = True

    frac_holdout: float = 0.20         # v14: holdout externo por padrao
    # v15: puros (conc==0) ficam SEMPRE no treino — sao escassos (3/classe)
    # e preciosos para DD-SIMCA/interpretacao. So adulterados vao ao holdout.
    holdout_preserva_puros: bool = True
    seed_holdout: int = 42

    n_por_classe: int = 20
    n_pontos_sint: int = 1000

    seed: int = 42

    # Sprint 3
    executar_ddsimca: bool = True
    ddsimca_n_components: int = 7       # LVs PCA por modelo DD-SIMCA (3 era insuficiente — UCL mal calibrado)
    ddsimca_ucl_method: str = "empirical"  # 'empirical' | 'theoretical' | 'chi2'
    # v14: 'todos' treina cada modelo com TODAS as amostras da classe
    # (exploratorio; funciona com poucos puros). 'puros' = one-class N2
    # verdadeiro, mas requer >=15 puros/classe (dados atuais: 3/classe).
    ddsimca_treinar_em: str = "todos"   # 'todos' | 'puros'
    executar_opls: bool = True
    n_ortho_opls: int = 1               # componentes ortogonais OPLS-DA
    executar_benchmark: bool = False    # v27: SVM / RF / XGBoost vs PLS-DA (mesma CV)
    executar_monte_carlo: bool = False      # v28: MC CV (N × GroupShuffleSplit) para IC95%
    n_monte_carlo: int = 100               # numero de repeticoes MC
    monte_carlo_test_size: float = 0.25    # fracao de teste por repeticao MC
    monte_carlo_incluir_todos: bool = False # v28: rodar todos os modelos do benchmark no MC CV
    executar_shap: bool = False            # v28: SHAP values (TreeExplainer) para ensemble
    shap_max_amostras: int = 500        # limite de amostras para SHAP (memoria)

    # ---- Exclusao de classes (ex: Copaiba com lote anomalo) ----
    excluir_classes: Tuple[str, ...] = ()

    # ---- ETAPA 4: Selecao de variaveis ----
    executar_etapa4: bool = True
    ipls_n_intervalos: int = 20         # iPLS: numero de intervalos
    vip_threshold_sel: float = 1.0      # selecao por VIP >= limiar
    sr_top_frac: float = 0.20           # selecao por SR: top fracao
    splsda_keep_por_comp: int = 50      # sPLS-DA: variaveis nao-nulas/componente


CFG = Config()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                FIM DAS CONFIGURACOES                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝


# =========================================================================
#  parse_title v3 — extracao de metadados a partir do ##TITLE= JCAMP-DX
#  Formato esperado (oleos amazonicos, ABB MB3600 — GEAAp/UFPA):
#      PURO:       {COD}-{DD-MM-YYYY}_T{N}
#      ADULTERADO: {COD}-{DD-MM-YYYY}-AD-{A|M|S}-{N,NN}%_T{N}
# =========================================================================

CODIGO_ESPECIE: Dict[str, str] = {
    "ACA": "Açaí",       "AND": "Andiroba",        "BAB": "Babaçu",
    "BCB": "Bacaba",     "BUR": "Buriti",          "CAP": "Castanha do Pará",
    "COC": "Coco",       "COP": "Copaíba",         "GOI": "Goiaba",
    "GRA": "Graviola",   "MAR": "Maracujá",
    "AR":  "Maracujá",   # codificacao encontrada no dataset GEAAp/UFPA
    "PAL": "Palmiste",   "PAT": "Patauá",          "PRA": "Pracaxi",
}
ADULTERANTE_NOME: Dict[str, str] = {"A": "algodão", "M": "milho", "S": "soja"}

# Regex robusto a desvios encontrados no dataset real GEAAp/UFPA:
#   - espaco circundante                       "## TITLE= GOI-..."
#   - separador apos COD/DATE: "-" ou "_"      "AND_10-06-2020_AD-S-..."
#   - separador antes de T opcional             "...%T_3"  (sem '-' nem '_')
#   - Triplicata: "T1" ou "T_1"
#   - Teor decimal opcional                     "11%"  /  "1,1%"  /  "10,52%"
#   - Sinal "%%" (typo)                         "...4,13%%_T1"
_RE_TITLE = re.compile(
    r"^\s*"
    r"(?P<cod>[A-Z]{2,4})"
    r"[-_](?P<data>\d{2}-\d{2}-\d{4})"
    r"(?P<adulteracao>(?:[-_]AD-[AMS]-\d+(?:[.,]\d+)?%%?)?)"
    r"[-_]?T_?(?P<trip>[123])"
    r"\s*$"
)
# Aceita decimal com virgula OU ponto (11,11% e 11.11% coexistem no dataset)
_RE_ADULT = re.compile(r"[-_]AD-([AMS])-(\d+(?:[.,]\d+)?)%%?")


def parse_title(title: str) -> Optional[Dict[str, Any]]:
    """Parser do TITLE JCAMP-DX. Retorna dict completo ou None se invalido.

    Campo mae_id: identifica univocamente o ponto fisico amostrado.
    Triplicatas T1/T2/T3 do mesmo ponto compartilham mae_id, permitindo
    GroupKFold/GroupShuffleSplit para evitar vazamento de replicas.

    Formato de mae_id:
        Puro:        'CAP-04-11-2020'
        Adulterado:  'CAP-04-11-2020-A1.03'  (teor sempre 2 casas decimais)
    """
    m = _RE_TITLE.match(title.strip())
    if not m:
        return None
    cod  = m.group("cod").upper()
    data = m.group("data")
    trip = int(m.group("trip"))
    am   = _RE_ADULT.match(m.group("adulteracao"))
    adulterante: Optional[str]      = None
    teor:        Optional[float]    = None
    adulterante_nome: Optional[str] = None
    if am:
        adulterante = str(am.group(1))
        teor = float(str(am.group(2)).replace(",", "."))
        adulterante_nome = ADULTERANTE_NOME.get(adulterante, adulterante)
        if teor <= 0:
            return None
    mae_id = (f"{cod}-{data}-{adulterante}{teor:.2f}"
              if adulterante is not None and teor is not None
              else f"{cod}-{data}")
    return {
        "title_original":   title.strip(),
        "cod":              cod,
        "especie":          CODIGO_ESPECIE.get(cod, cod),
        "cod_conhecido":    cod in CODIGO_ESPECIE,
        "data":             data,
        "puro":             adulterante is None,
        "adulterante":      adulterante,
        "adulterante_nome": adulterante_nome,
        "teor":             teor,
        "triplicata":       trip,
        "mae_id":           mae_id,
    }


def extrair_title_do_dx(caminho: str) -> Optional[str]:
    """Extrai linha ##TITLE= sem carregar os 8192 pontos do espectro."""
    try:
        with open(caminho, "r", encoding="latin-1", errors="replace") as f:
            for linha in f:
                linha = linha.strip()
                if linha.startswith("##TITLE="):
                    return linha[len("##TITLE="):]
                if linha.startswith("##XYDATA") or linha.startswith("##XYPOINTS"):
                    break
    except Exception:
        pass
    return None


def gerar_nome_saida(cfg: Config, n_classes: int, n_amostras: int) -> str:
    """Caminho de saida padrao (v20): prefixo do projeto + tipo de analise
    (nivel + pre-processamento) + data/hora compacta.
        {raiz}/PLSDA_OE_{nivel}_{preproc}_{YYYYMMDD_HHMMSS}
    Exemplo: resultados_tcc/PLSDA_OE_N1_MSC-SG1-MC_20260528_191500
    Subpastas (criadas em executar): dados/ figuras/ modelos/ logs/
    """
    preset = (cfg.preprocessamento_padrao or "custom").lower()
    if preset == "autoscaling":
        preproc = ["AUTO"]
    elif preset == "mc":
        preproc = ["MC"]
    elif preset == "snv_sg_mc":
        preproc = ["SNV", f"SG{cfg.sg_deriv}", "MC"]
    elif preset == "msc_sg_mc":
        preproc = ["MSC", f"SG{cfg.sg_deriv}", "MC"]
    else:  # custom
        preproc = []
        if cfg.aplicar_snv: preproc.append("SNV")
        if cfg.aplicar_sg:  preproc.append(f"SG{cfg.sg_deriv}")
        if cfg.aplicar_mc:  preproc.append("MC")
        if not preproc:     preproc.append("bruto")
    partes = ["PLSDA_OE", cfg.nivel]
    if cfg.tag.strip():
        partes.append(cfg.tag.strip().replace(" ", "_"))
    partes.append("-".join(preproc))
    partes.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    return os.path.join(cfg.pasta_saida_raiz, "_".join(partes))


# Paleta de MAXIMA DISTINCAO perceptual (base: Trubetskoy "20 distinct
# colors" + Glasbey). Ordenada por contraste em fundo branco: cores fortes/
# saturadas primeiro, claras por ultimo. SEM azuis/verdes quase-iguais.
# Para impressao/daltonismo o canal de FORMA (MARCADORES) complementa a cor.
PALETA = [
    "#E6194B",  # vermelho
    "#4363D8",  # azul
    "#3CB44B",  # verde
    "#F58231",  # laranja
    "#911EB4",  # roxo
    "#42D4F4",  # ciano
    "#F032E6",  # magenta
    "#9A6324",  # marrom
    "#469990",  # teal
    "#800000",  # bordo
    "#808000",  # oliva
    "#000075",  # marinho
    "#E6A000",  # ambar/ouro
    "#BFEF45",  # lima
    "#FABED4",  # rosa-claro
    "#DCBEFF",  # lavanda
    "#AAFFC3",  # menta
    "#FFD8B1",  # pessego
    "#A9A9A9",  # cinza
    "#FFE119",  # amarelo
]

# Canal secundario: formas de marcador (identidade por SHAPE, robusto a
# daltonismo/impressao P&B). 14 formas distintas antes de repetir.
MARCADORES = ["o", "s", "^", "D", "v", "P", "X", "*",
              "<", ">", "h", "p", "8", "d"]


def _paleta_externa(n: int) -> Optional[List[str]]:
    """Tenta gerar paleta de max-distincao via libs opcionais (glasbey,
    colorcet). Retorna lista de hex ou None se nenhuma disponivel."""
    try:
        import glasbey as _gb  # type: ignore
        return list(_gb.create_palette(palette_size=n))
    except Exception:
        pass
    try:
        import colorcet as _cc  # type: ignore
        base = _cc.glasbey_category10
        return [base[i % len(base)] for i in range(n)]
    except Exception:
        pass
    return None


def _luminancia(hex_cor: str) -> float:
    """Luminancia relativa (0=escuro, 1=claro) p/ decidir cor da borda."""
    r, g, b = mcolors.to_rgb(hex_cor)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def edge_para_cor(hex_cor: str) -> str:
    """Borda inteligente: cinza-escuro para preenchimentos claros (visiveis
    em fundo branco), branco para preenchimentos escuros."""
    return "0.25" if _luminancia(hex_cor) > 0.65 else "white"


def cor(i: int) -> str:
    """Cor i da paleta de maxima distincao. Acima do tamanho, usa lib externa
    (se houver) ou cicla com leve variacao de luminancia via HSV."""
    if i < len(PALETA):
        return PALETA[i]
    ext = _paleta_externa(i + 1)
    if ext is not None and i < len(ext):
        return mcolors.to_hex(ext[i])
    # fallback: tab20 deslocado
    cmap = plt.get_cmap("tab20")
    return mcolors.to_hex(cmap(((i - len(PALETA)) % 20) / 20))


def mapear_cores_classes(classes) -> Dict[str, str]:
    """Atribui cor por ordem alfabetica a partir da paleta de maxima
    distincao. Atribuicao SEQUENCIAL (nao-hash) garante que classes
    adjacentes recebam cores bem separadas — a paleta ja e ordenada para
    maximizar contraste entre indices vizinhos. Deterministico."""
    classes_sorted = sorted({str(c) for c in classes})
    n = len(classes_sorted)
    externa = _paleta_externa(n) if n > len(PALETA) else None
    mapa: Dict[str, str] = {}
    for idx, cls in enumerate(classes_sorted):
        if externa is not None:
            mapa[cls] = mcolors.to_hex(externa[idx])
        else:
            mapa[cls] = cor(idx)
    return mapa


def mapear_marcadores_classes(classes) -> Dict[str, str]:
    """Atribui forma de marcador por classe (canal secundario). Combinado
    com a cor, garante distincao mesmo em P&B/daltonismo e alta densidade."""
    classes_sorted = sorted({str(c) for c in classes})
    return {cls: MARCADORES[i % len(MARCADORES)]
            for i, cls in enumerate(classes_sorted)}


def setup_matplotlib(cfg: Config) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlepad": 8,
        "axes.labelsize": 9,
        "axes.labelpad": 4,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "figure.dpi": 110,
        "savefig.dpi": cfg.dpi_salvar,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "lines.linewidth": 1.2,
        "lines.markersize": 4.5,
        "mathtext.default": "regular",
    })


def salvar(fig, nome: str, pasta: str, cfg: Config,
           subpasta: str = "") -> None:
    """Salva figura sempre sob pasta/figuras/[subpasta]/ (estrutura v20).
    subpasta agrupa figuras detalhadas (ex: 'ddsimca')."""
    base = os.path.join(pasta, "figuras")
    destino = os.path.join(base, subpasta) if subpasta else base
    os.makedirs(destino, exist_ok=True)
    caminho = os.path.join(destino, f"{nome}.{cfg.formato_saida}")
    try:
        fig.savefig(caminho)
        print(f"  -> {caminho}")
    except Exception as e:
        print(f"  [ERRO] {caminho}: {e}")
    if cfg.mostrar_graficos:
        plt.show()
    else:
        plt.close(fig)


# =========================================================================
#  Transformadores compatíveis com sklearn (necessários para CV sem leakage)
# =========================================================================

class SNV(BaseEstimator, TransformerMixin):
    """Standard Normal Variate: z-score por amostra (correcao de scatter)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        return (X - mu) / sd


class SavGol(BaseEstimator, TransformerMixin):
    """Filtro Savitzky-Golay (suavizacao ou derivada)."""

    def __init__(self, window_length: int = 25, polyorder: int = 2, deriv: int = 1):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return savgol_filter(np.asarray(X, dtype=float),
                             window_length=self.window_length,
                             polyorder=self.polyorder,
                             deriv=self.deriv, axis=1)


class MSC(BaseEstimator, TransformerMixin):
    """Multiplicative Scatter Correction. Usa espectro medio do treino como
    referencia; para cada amostra estima (a, b) tal que X_i ~ a + b * ref e
    retorna (X_i - a) / b. Stateful: precisa ficar dentro do Pipeline+CV."""

    def fit(self, X, y=None):
        self.ref_ = np.asarray(X, dtype=float).mean(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        A = np.column_stack([np.ones_like(self.ref_), self.ref_])
        out = np.zeros_like(X)
        for i in range(X.shape[0]):
            sol, *_ = np.linalg.lstsq(A, X[i], rcond=None)
            a, b = float(sol[0]), float(sol[1])
            out[i] = (X[i] - a) / b if abs(b) > 1e-12 else X[i] - a
        return out


def construir_preprocessador(cfg: Config) -> Pipeline:
    """Constroi pre-processador segundo cfg.preprocessamento_padrao.

    Presets:
        'snv_sg_mc'   : SNV -> SG -> mean-centering (Rinnan et al. 2009,
                        recomendado para FTIR/NIR com scatter)
        'autoscaling' : StandardScaler (mean + unit variance)
                        — recomendado quando SG derivativo destroi sinal
                        ou para NIR sem scatter pronunciado
        'mc'          : apenas mean-centering
        'custom'      : honra aplicar_snv / aplicar_sg / aplicar_mc

    Mean-centering / autoscaling ficam DENTRO do Pipeline para que
    cross_val_predict nao vaze estatisticas entre folds.
    """
    preset = (cfg.preprocessamento_padrao or "custom").lower()

    if preset == "autoscaling":
        return Pipeline([("auto", StandardScaler(with_mean=True, with_std=True))])
    if preset == "mc":
        return Pipeline([("mc", StandardScaler(with_std=False))])
    if preset == "snv_sg_mc":
        return Pipeline([
            ("snv", SNV()),
            ("sg",  SavGol(cfg.sg_window, cfg.sg_polyorder, cfg.sg_deriv)),
            ("mc",  StandardScaler(with_std=False)),
        ])
    if preset == "msc_sg_mc":
        # MSC->SG+MC: melhor pipeline na base completa (0.923 bal.acc).
        # MSC e stateful (referencia = media do treino) -> dentro do
        # Pipeline para nao vazar entre folds da CV.
        return Pipeline([
            ("msc", MSC()),
            ("sg",  SavGol(cfg.sg_window, cfg.sg_polyorder, cfg.sg_deriv)),
            ("mc",  StandardScaler(with_std=False)),
        ])
    # custom — usa flags individuais
    etapas: List[Tuple[str, BaseEstimator]] = []
    if cfg.aplicar_snv:
        etapas.append(("snv", SNV()))
    if cfg.aplicar_sg:
        etapas.append(("sg", SavGol(cfg.sg_window, cfg.sg_polyorder, cfg.sg_deriv)))
    if cfg.aplicar_mc:
        etapas.append(("mc", StandardScaler(with_std=False)))
    if not etapas:
        etapas.append(("mc", StandardScaler(with_std=False)))
    return Pipeline(etapas)


# =========================================================================
#  Diagnósticos quimiométricos
# =========================================================================

def vip_scores(modelo: PLSRegression) -> np.ndarray:
    """VIP scores per Chong & Jun (2005), Chemom. Intell. Lab. Syst. 78:103-112."""
    T = np.asarray(modelo.x_scores_, dtype=float)
    W = np.asarray(modelo.x_weights_, dtype=float)
    Q = np.asarray(modelo.y_loadings_, dtype=float)
    p, _ = W.shape
    ss = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    normas = np.linalg.norm(W, axis=0); normas[normas == 0] = 1.0
    W_norm = W / normas
    return np.sqrt(p * np.sum(ss * W_norm ** 2, axis=1) / (np.sum(ss) + 1e-12))


def calcular_selectivity_ratio(modelo: PLSRegression,
                                X: np.ndarray) -> np.ndarray:
    """Selectivity Ratio (SR) per Rajalahti et al. (2009),
    Chemom. Intell. Lab. Syst. 95:20-28.

    Para cada variavel j, decompoe X_j em parte explicada pela projecao
    alvo (primeiro peso preditivo PLS) e residuo:
        t_tp  = X @ w1 / ||w1||   (target projection scores)
        p_tp_j = (t_tp^T * X_j) / (t_tp^T * t_tp)
        SR_j  = Var(t_tp * p_tp_j) / Var(X_j - t_tp * p_tp_j)

    Complementa o VIP: SR e mais sensivel a variaveis com correlacao
    direcional com Y no 1o componente; VIP integra todos os LVs.
    Concordancia entre VIP >= 1 e SR alto reforca a relevancia.
    """
    X = np.asarray(X, dtype=float)
    W = np.asarray(modelo.x_weights_, dtype=float)   # (p, n_lv)
    w1 = W[:, 0]
    norm_w = float(np.linalg.norm(w1))
    if norm_w < 1e-12:
        return np.zeros(X.shape[1])
    w1_unit = w1 / norm_w

    t_tp = X @ w1_unit                  # (n,)
    tt = float(t_tp @ t_tp)
    if tt < 1e-12:
        return np.zeros(X.shape[1])

    p_tp   = (t_tp @ X) / tt            # (p,) — target projection loadings
    X_tp   = np.outer(t_tp, p_tp)       # (n, p) — target-projected X
    X_res  = X - X_tp                   # (n, p) — residual

    var_tp  = X_tp.var(axis=0, ddof=1)
    var_res = X_res.var(axis=0, ddof=1)
    var_res[var_res < 1e-12] = 1e-12
    return var_tp / var_res


def hotelling_t2(T: np.ndarray) -> np.ndarray:
    var_t = T.var(axis=0, ddof=1)
    var_t[var_t == 0] = 1.0
    return np.sum((T ** 2) / var_t, axis=1)


def hotelling_t2_limite(n: int, k: int, alpha: float = 0.05) -> float:
    """Limite superior de Hotelling T2 (Tracy-Young-Mason 1992).

    Formula correta para small-sample, valida tanto para observacao
    dentro do conjunto de calibracao quanto para nova observacao:

        T2_UCL = k * (n - 1) * (n + 1) / (n * (n - k)) * F_(alpha, k, n - k)

    Substitui a aproximacao (k(n-1)/(n-k))*F que subestimava o limite
    em ~5-10% para n<30 (causa falsos outliers em datasets pequenos).
    """
    if n - k <= 0:
        print(f"[AVISO] Hotelling T2: n={n} muito pequeno para k={k} LVs.")
        return float("inf")
    if n < 3 * k:
        print(f"[AVISO] Hotelling T2: n={n} < 3k={3*k}. Limite pode ser "
              f"impreciso (intervalo de confianca largo).")
    return float(((k * (n - 1) * (n + 1)) / (n * (n - k)))
                  * f_dist.ppf(1 - alpha, k, n - k))


def q_residuos(X: np.ndarray, T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.sum((X - T @ P) ** 2, axis=1)


def q_residuos_limite(q: np.ndarray, alpha: float = 0.05) -> float:
    media = float(q.mean()); var = float(q.var())
    if var <= 0 or media <= 0:
        return float(np.percentile(q, (1 - alpha) * 100)) if q.size else 0.0
    g = var / (2 * media); h = 2 * (media ** 2) / var
    return float(g * chi2.ppf(1 - alpha, h))


def variancia_explicada(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Variancia explicada (%) de X por cada coluna de T."""
    var_X_total = float(np.var(X, axis=0).sum())
    if var_X_total <= 0:
        return np.zeros(T.shape[1])
    return np.var(T, axis=0) / var_X_total * 100


# =========================================================================
#  Sprint 3 — Classificadores avancados
# =========================================================================

class DDSimca:
    """Data-Driven SIMCA: classificador one-class por classe via PCA.

    Para cada classe treina um modelo PCA independente e define
    limites de aceitacao (UCL) para T2 e Q-residuos:
        T2_UCL  — calculado por ucl_method:
                    'empirical'  : percentil (1-alpha) do T2 de treino
                    'theoretical': Tracy-Young-Mason (F-distribution)
                    'chi2'       : chi2(1-alpha, n_components)
                  'empirical' e o unico que VARIA POR CLASSE de fato
                  (theoretical e chi2 dependem so de n,k); recomendado.
        Q_UCL   — aproximacao chi2 (Jackson & Mudholkar) via mean/var de
                  Q-residuos de treino — naturalmente data-driven.

    Uma nova amostra e 'aceita' pela classe se T2 <= UCL **e** Q <= UCL.

    Referencias:
        Rodionova O.Y. & Pomerantsev A.L. (2020). Chemom. Intell. Lab.
        Syst. 200:103958.
    """

    def __init__(self, n_components: int = 3, alpha: float = 0.05,
                 ucl_method: str = "empirical"):
        self.n_components = n_components
        self.alpha = alpha
        self.ucl_method = ucl_method
        self._modelos: Dict[str, Dict[str, Any]] = {}
        self._classes: np.ndarray = np.array([], dtype=str)

    def _compute_t2_ucl(self, T2_train: np.ndarray, n: int, k: int) -> float:
        method = (self.ucl_method or "empirical").lower()
        if method == "empirical":
            if T2_train.size == 0:
                return float("inf")
            return float(np.percentile(T2_train, 100 * (1 - self.alpha)))
        if method == "theoretical":
            return hotelling_t2_limite(n, k, self.alpha)
        if method == "chi2":
            return float(chi2.ppf(1 - self.alpha, k))
        # fallback
        return float(np.percentile(T2_train, 100 * (1 - self.alpha)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DDSimca":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=str)
        self._classes = np.unique(y)
        self._modelos = {}
        for cls in self._classes:
            Xc = X[y == cls]
            nc = len(Xc)
            n_comp = min(self.n_components, nc - 1, Xc.shape[1])
            if n_comp < 1:
                print(f"[DDSimca] Classe '{cls}': amostras insuficientes "
                      f"(n={nc}) — modelo omitido.")
                continue
            pca = PCA(n_components=n_comp)
            T = pca.fit_transform(Xc)
            var_t = T.var(axis=0, ddof=1)
            var_t[var_t == 0] = 1.0
            X_rec = pca.inverse_transform(T)
            Q_train = np.sum((Xc - X_rec) ** 2, axis=1)
            T2_train = np.sum((T ** 2) / var_t, axis=1)
            self._modelos[cls] = {
                "pca":      pca,
                "var_t":    var_t,
                "T2_ucl":   self._compute_t2_ucl(T2_train, nc, n_comp),
                "Q_ucl":    q_residuos_limite(Q_train, self.alpha),
                "T_train":  T,
                "T2_train": T2_train,
                "Q_train":  Q_train,
                "n_train":  nc,
                "n_comp":   n_comp,
            }
        return self

    def _t2_q(self, X: np.ndarray, cls: str
              ) -> Tuple[np.ndarray, np.ndarray]:
        m = self._modelos[cls]
        pca = m["pca"]
        T = pca.transform(X)
        X_rec = pca.inverse_transform(T)
        Q  = np.sum((X - X_rec) ** 2, axis=1)
        T2 = np.sum((T ** 2) / m["var_t"], axis=1)
        return T2, Q

    def score_matrix(self, X: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """T2, Q e versoes normalizadas (T2/UCL, Q/UCL) por classe."""
        X = np.asarray(X, dtype=float)
        res: Dict[str, Dict[str, Any]] = {}
        for cls in self._classes:
            if cls not in self._modelos:
                continue
            m = self._modelos[cls]
            T2, Q = self._t2_q(X, cls)
            res[cls] = {
                "T2":       T2,
                "Q":        Q,
                "T2_ucl":   m["T2_ucl"],
                "Q_ucl":    m["Q_ucl"],
                "T2_norm":  T2 / max(m["T2_ucl"], 1e-12),
                "Q_norm":   Q  / max(m["Q_ucl"],  1e-12),
                "T_train":  m["T_train"],
                "Q_train":  m["Q_train"],
                "n_train":  m["n_train"],
            }
        return res

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna: nome da classe | 'Ambiguo' | 'Desconhecido'."""
        X = np.asarray(X, dtype=float)
        preds = []
        for i in range(len(X)):
            xi = X[i:i+1]
            aceitas = []
            for cls in self._classes:
                if cls not in self._modelos:
                    continue
                m = self._modelos[cls]
                T2, Q = self._t2_q(xi, cls)
                if T2[0] <= m["T2_ucl"] and Q[0] <= m["Q_ucl"]:
                    aceitas.append(cls)
            if   len(aceitas) == 1: preds.append(aceitas[0])
            elif len(aceitas) >  1: preds.append("Ambiguo")
            else:                   preds.append("Desconhecido")
        return np.array(preds)


class OPLSDAWrapper(BaseEstimator):
    """OPLS-DA: deflacao ortogonal + 1 componente preditivo.

    Extrai n_ortho componentes de X ortogonais a Y via NIPALS, depois
    ajusta 1 LV preditivo no X deflacionado. Saidas:
        t_pred  — score preditivo (separa classes)
        t_orth  — score(s) ortogonal(is) (variacao estruturada de X
                  nao correlacionada com Y, e.g., baseline, scatter)

    O plot tp x to1 decompoe: separacao real (tp) vs variacao instrumental
    sistematica (to). Para FTIR de oleos, to captura tipicamente variacao
    de espessura de caminho optico e scatter multiplicativo.

    Referencias:
        Trygg J. & Wold S. (2002) J. Chemometrics 16:119-128.
        Bylesjo M. et al. (2006) J. Chemometrics 20:341-351.
    """

    def __init__(self, n_ortho: int = 1):
        self.n_ortho = n_ortho
        self.W_orth_: List[np.ndarray] = []
        self.P_orth_: List[np.ndarray] = []

    @staticmethod
    def _nipals_pls1(X: np.ndarray, y: np.ndarray,
                     max_iter: int = 500, tol: float = 1e-10
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """NIPALS para 1 componente PLS com y unidimensional.
        Retorna (w, t, p) normalizados."""
        u = y.astype(float).copy()
        t_old = np.zeros(X.shape[0])
        w = np.zeros(X.shape[1])
        t = t_old.copy()
        for _ in range(max_iter):
            w = X.T @ u
            nw = float(np.linalg.norm(w))
            if nw < 1e-12:
                break
            w /= nw
            t = X @ w
            nt = float(t @ t)
            if nt < 1e-12:
                break
            c = float(y @ t) / nt
            u_new = y * c
            if float(np.linalg.norm(t - t_old)) / (float(np.linalg.norm(t)) + 1e-12) < tol:
                break
            t_old = t.copy()
            u = u_new
        nt = float(t @ t)
        p = X.T @ t / nt if nt > 1e-12 else np.zeros(X.shape[1])
        return w, t, p

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "OPLSDAWrapper":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        # Usa primeira coluna de Y como resposta binaria
        y = Y[:, 0] if Y.ndim == 2 else Y.copy()
        y = y - float(y.mean())

        n = X.shape[0]
        Xr = X.copy()
        self.W_orth_ = []
        self.P_orth_ = []
        T_orth_train: List[np.ndarray] = []

        for _ in range(self.n_ortho):
            w, t, p = self._nipals_pls1(Xr, y)
            # Componente ortogonal: direcao de p ortogonalizada contra w
            proj = float(p @ w)
            w_orth = p - proj * w
            no = float(np.linalg.norm(w_orth))
            if no < 1e-10:
                break
            w_orth /= no
            t_orth = Xr @ w_orth
            nto = float(t_orth @ t_orth)
            if nto < 1e-12:
                break
            p_orth = Xr.T @ t_orth / nto
            self.W_orth_.append(w_orth.copy())
            self.P_orth_.append(p_orth.copy())
            T_orth_train.append(t_orth.copy())
            Xr = Xr - np.outer(t_orth, p_orth)   # deflate

        # 1 componente preditivo no X deflacionado
        self._pls_pred = PLSRegression(n_components=1, scale=False)
        self._pls_pred.fit(Xr, Y)

        # Scores de treino (sklearn >= 1.x retorna ndarray diretamente)
        _t_arr = self._pls_pred.transform(Xr)
        t_pred_tr = _t_arr if isinstance(_t_arr, np.ndarray) else _t_arr[0]
        self.t_pred_train_ = t_pred_tr[:, 0]
        self.t_orth_train_ = (np.column_stack(T_orth_train)
                               if T_orth_train else np.zeros((n, 1)))
        self.n_ortho_fitted_ = len(self.W_orth_)
        return self

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna (t_pred ndarray, t_orth matrix (n, n_ortho))."""
        X = np.asarray(X, dtype=float)
        Xr = X.copy()
        T_orth: List[np.ndarray] = []
        for w_o, p_o in zip(self.W_orth_, self.P_orth_):
            t_o = Xr @ w_o
            T_orth.append(t_o)
            Xr = Xr - np.outer(t_o, p_o)
        _t_new = self._pls_pred.transform(Xr)
        t_pred_arr = _t_new if isinstance(_t_new, np.ndarray) else _t_new[0]
        t_orth = (np.column_stack(T_orth)
                  if T_orth else np.zeros((len(X), 1)))
        return t_pred_arr[:, 0], t_orth


def metricas_modelo_pls(modelo: PLSRegression, X: np.ndarray, Y: np.ndarray,
                         Y_cv: np.ndarray) -> Tuple[float, float, float]:
    """R2X via 1 - SS(X - T @ P^T) / SS(X_centrado); R2Y via 1 - SS(res)/SS(Y);
    Q2 idem usando predicoes CV. Formula rigorosa de reconstrucao."""
    X = np.asarray(X, dtype=float)
    T = np.asarray(modelo.x_scores_,   dtype=float)
    P = np.asarray(modelo.x_loadings_, dtype=float)

    X_c = X - X.mean(axis=0)
    ss_x_total = float(np.sum(X_c ** 2))
    ss_x_res   = float(np.sum((X_c - T @ P.T) ** 2))
    r2x = float(1.0 - ss_x_res / ss_x_total) if ss_x_total > 0 else 0.0

    Y = np.asarray(Y, dtype=float)
    Y_cv = np.asarray(Y_cv, dtype=float)
    Y_hat = np.asarray(modelo.predict(X), dtype=float)
    ss_total = float(np.sum((Y - Y.mean(axis=0)) ** 2))
    if ss_total <= 0:
        return r2x, 0.0, 0.0
    r2y = float(1.0 - float(np.sum((Y - Y_hat) ** 2)) / ss_total)
    q2  = float(1.0 - float(np.sum((Y - Y_cv)  ** 2)) / ss_total)
    return r2x, r2y, q2


def salvar_identificadores(rotulos: np.ndarray, pred_lab: np.ndarray,
                            scores_pls: np.ndarray, T2: np.ndarray,
                            Q: np.ndarray, t2_lim: float, q_lim: float,
                            pasta: str) -> None:
    """Tabela com IDs, classes reais/preditas e diagnosticos por amostra."""
    n = len(rotulos)
    n_lvs_save = min(3, scores_pls.shape[1])
    dados = {
        "ID":             [f"S{i:03d}" for i in range(n)],
        "indice":         np.arange(n),
        "classe_real":    rotulos,
        "classe_predita": pred_lab,
        "correto":        rotulos == pred_lab,
    }
    for k in range(n_lvs_save):
        dados[f"LV{k+1}"] = scores_pls[:, k]
    dados["T2"]          = T2
    dados["T2_outlier"]  = T2 > t2_lim
    dados["Q"]           = Q
    dados["Q_outlier"]   = Q > q_lim
    pd.DataFrame(dados).to_csv(
        os.path.join(pasta, "amostras_identificadores.csv"),
        index=False, sep=";", decimal=",")


def salvar_resumo_modelo(pasta: str, info: Dict[str, object]) -> None:
    caminho = os.path.join(pasta, "resumo_modelo.txt")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  Resumo do Modelo PLS-DA\n")
        f.write("=" * 60 + "\n\n")
        largura = max(len(k) for k in info.keys()) + 2
        for k, v in info.items():
            if isinstance(v, float):
                f.write(f"  {k:<{largura}s}: {v:.4f}\n")
            else:
                f.write(f"  {k:<{largura}s}: {v}\n")


def validar_entrada(X: np.ndarray, wavenumbers: np.ndarray,
                     rotulos: np.ndarray, conc: Optional[np.ndarray] = None,
                     mae_id: Optional[np.ndarray] = None,
                     tol_const: float = 1e-12,
                     limiar_correlacao: float = 0.99995,
                     max_n_para_corr: int = 500
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                  Optional[np.ndarray], Optional[np.ndarray],
                                  Dict[str, object]]:
    """Validacao robusta. Remove NaN/Inf, variaveis constantes, e detecta
    duplicatas exatas/aproximadas. Retorna dados limpos + relatorio.

    mae_id (B1): sincronizado com a MESMA mascara de remocao de NaN/Inf, para
    que a validacao group-aware sobreviva a remocao de amostras (antes, uma
    unica amostra com NaN desligava silenciosamente o mae_id e o GroupKFold).

    Retorna (X, wavenumbers, rotulos, conc, mae_id, relatorio).
    """
    X = np.asarray(X, dtype=float)
    rotulos = np.asarray(rotulos, dtype=str)
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    if conc is not None:
        conc = np.asarray(conc, dtype=float)
    if mae_id is not None:
        mae_id = np.asarray(mae_id, dtype=str)

    warnings_list: List[str] = []
    relatorio: Dict[str, object] = {
        "n_inicial":               int(len(X)),
        "n_variaveis_inicial":     int(X.shape[1]),
        "n_nan_amostras":          0,
        "n_inf_amostras":          0,
        "n_amostras_removidas":    0,
        "n_constantes_removidas":  0,
        "n_duplicatas_exatas":     0,
        "n_duplicatas_aproximadas": 0,
    }

    # --- NaN / Inf por amostra (linha inteira eliminada) -----------------
    nan_mask = np.any(np.isnan(X), axis=1)
    inf_mask = np.any(np.isinf(X), axis=1)
    bad_mask = nan_mask | inf_mask
    n_nan = int(nan_mask.sum())
    n_inf = int(inf_mask.sum())
    relatorio["n_nan_amostras"] = n_nan
    relatorio["n_inf_amostras"] = n_inf
    if bad_mask.any():
        n_rem = int(bad_mask.sum())
        relatorio["n_amostras_removidas"] = n_rem
        warnings_list.append(
            f"{n_rem} amostras removidas por NaN ({n_nan}) ou Inf ({n_inf})")
        print(f"[AVISO] Removidas {n_rem} amostras com NaN/Inf.")
        keep = ~bad_mask
        X = X[keep]; rotulos = rotulos[keep]
        if conc is not None:
            conc = conc[keep]
        if mae_id is not None:
            mae_id = mae_id[keep]   # B1: mantem group-aware sincronizado

    # --- Colunas constantes ---------------------------------------------
    var_cols = np.var(X, axis=0)
    mask_var = var_cols > tol_const
    n_const = int((~mask_var).sum())
    relatorio["n_constantes_removidas"] = n_const
    if n_const > 0:
        warnings_list.append(
            f"{n_const} variaveis com variancia ~= 0 removidas")
        print(f"[AVISO] Removidas {n_const} variaveis constantes.")
        X = X[:, mask_var]
        wavenumbers = wavenumbers[mask_var]

    # --- Duplicatas exatas ----------------------------------------------
    _, idx_unique = np.unique(X, axis=0, return_index=True)
    n_dup_exatas = int(len(X) - len(idx_unique))
    relatorio["n_duplicatas_exatas"] = n_dup_exatas
    if n_dup_exatas > 0:
        msg = (f"ATENCAO: {n_dup_exatas} amostras duplicadas EXATAS "
                f"detectadas. Possivel vazamento treino/validacao se "
                f"copias caem em folds diferentes.")
        warnings_list.append(msg)
        print(f"[ATENCAO] {msg}")

    # --- Duplicatas aproximadas (correlacao alta) -----------------------
    if len(X) <= max_n_para_corr and len(X) >= 2:
        Xc = X - X.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(Xc, axis=1)
        norms[norms == 0] = 1.0
        Xn = Xc / norms[:, None]
        corr = Xn @ Xn.T
        np.fill_diagonal(corr, 0.0)
        pares_altos = np.argwhere(corr > limiar_correlacao)
        if len(pares_altos) > 0:
            envolvidas = np.unique(pares_altos.flatten())
            n_aprox = int(len(envolvidas))
            relatorio["n_duplicatas_aproximadas"] = n_aprox
            if n_aprox > n_dup_exatas:
                warnings_list.append(
                    f"ATENCAO: {n_aprox} amostras com correlacao > "
                    f"{limiar_correlacao} (possiveis replicas tecnicas). "
                    f"Considere GroupKFold para evitar vazamento.")
                print(f"[ATENCAO] {n_aprox} amostras com corr > "
                      f"{limiar_correlacao:.5f}.")

    relatorio["n_final"] = int(len(X))
    relatorio["n_variaveis_final"] = int(X.shape[1])
    relatorio["warnings"] = warnings_list
    return X, wavenumbers, rotulos, conc, mae_id, relatorio


def verificar_balanceamento(rotulos: np.ndarray, ratio_alvo: float = 5.0
                              ) -> Dict[str, object]:
    """Detecta desbalanceamento severo entre classes."""
    cls_unicas, counts = np.unique(rotulos, return_counts=True)
    n_max = int(counts.max())
    n_min = int(counts.min())
    ratio = n_max / max(n_min, 1)
    rel: Dict[str, object] = {
        "imbalance_ratio": float(ratio),
        "n_max":           n_max,
        "n_min":           n_min,
        "distribuicao":    {str(c): int(n) for c, n in zip(cls_unicas, counts)},
        "desbalanceado":   ratio > ratio_alvo,
    }
    if ratio > ratio_alvo:
        print(f"[AVISO] Desbalanceamento severo: razao max/min = "
              f"{ratio:.2f} (max={n_max}, min={n_min}).")
        print(f"        Sugestoes: priorizar balanced_accuracy/F1-macro "
              f"sobre accuracy; considerar class_weight ou subamostragem.")
    return rel


def especificidade_por_classe(cm: np.ndarray) -> np.ndarray:
    """Specificity = TN / (TN + FP), por classe (one-vs-rest)."""
    n = cm.shape[0]
    total = cm.sum()
    spec = np.zeros(n)
    for i in range(n):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        spec[i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return spec


def metricas_classificacao(y_true, y_pred, classes) -> Dict[str, float]:
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa":       float(cohen_kappa_score(y_true, y_pred)),
        "f1_macro":          float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "precision_macro":   float(precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "recall_macro":      float(recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
    }


def _cv_predict_manual(pipeline_factory, X, Y_bin, cv_indices):
    """cross_val_predict manual, compativel com Y multilabel + cv estratificada."""
    y_hat = np.zeros_like(Y_bin, dtype=float)
    contador = np.zeros(len(Y_bin), dtype=int)
    for tr, va in cv_indices:
        pipe = pipeline_factory()
        pipe.fit(X[tr], Y_bin[tr])
        y_hat[va] += pipe.predict(X[va])
        contador[va] += 1
    contador[contador == 0] = 1
    return y_hat / contador[:, None]


def comparar_pipelines(cfg: Config, X_raw: np.ndarray, Y_bin: np.ndarray,
                        y_int: np.ndarray, cv_indices: list,
                        max_lv: int = 8) -> Dict[str, Dict[str, float]]:
    """Avalia varios pipelines de pre-processamento via CV. Para cada
    pipeline encontra o melhor n_lv por RMSECV e reporta accuracy,
    balanced accuracy e Q2."""

    def _etapas_sg():
        return ("sg", SavGol(cfg.sg_window, cfg.sg_polyorder, cfg.sg_deriv))
    def _mc():
        return ("mc", StandardScaler(with_std=False))

    presets: Dict[str, Callable[[], List]] = {
        "Apenas MC":              lambda: [_mc()],
        "Autoscaling":            lambda: [("auto", StandardScaler())],
        "SNV + MC":               lambda: [("snv", SNV()), _mc()],
        "MSC + MC":               lambda: [("msc", MSC()), _mc()],
        "SG + MC":                lambda: [_etapas_sg(), _mc()],
        "SG -> SNV + MC":         lambda: [_etapas_sg(), ("snv", SNV()), _mc()],
        "SNV -> SG + MC (atual)": lambda: [("snv", SNV()), _etapas_sg(), _mc()],
        "MSC -> SG + MC":         lambda: [("msc", MSC()), _etapas_sg(), _mc()],
    }

    ss_total = float(np.sum((Y_bin - Y_bin.mean(axis=0)) ** 2))
    resultados: Dict[str, Dict[str, float]] = {}

    for nome, build_etapas in presets.items():
        melhor = {"q2": -np.inf, "accuracy": 0.0, "balanced_acc": 0.0, "n_lv": 1}
        for n_lv in range(1, max_lv + 1):
            def factory(_etapas=build_etapas, _n=n_lv):
                return Pipeline(_etapas() + [
                    ("pls", PLSRegression(n_components=_n, scale=False))])
            try:
                y_cv = _cv_predict_manual(factory, X_raw, Y_bin, cv_indices)
            except Exception:
                continue
            ss_res = float(np.sum((Y_bin - y_cv) ** 2))
            q2 = 1.0 - ss_res / ss_total if ss_total > 0 else 0.0
            y_int_hat = np.argmax(y_cv, axis=1)
            acc = float(accuracy_score(y_int, y_int_hat))
            bal = float(balanced_accuracy_score(y_int, y_int_hat))
            if q2 > melhor["q2"]:
                melhor = {"q2": q2, "accuracy": acc,
                          "balanced_acc": bal, "n_lv": n_lv}
        resultados[nome] = melhor
        print(f"  {nome:<26s} -> LVs={melhor['n_lv']:2d}  "
              f"Acc={melhor['accuracy']:.3f}  "
              f"BalAcc={melhor['balanced_acc']:.3f}  "
              f"Q2={melhor['q2']:.3f}")

    return resultados


def bootstrap_vip_estratificado(X_processed: np.ndarray, Y_bin: np.ndarray,
                                  y_int: np.ndarray, n_opt: int, n_boot: int,
                                  seed: int, vip_threshold: float = 1.0
                                  ) -> Dict[str, object]:
    """Bootstrap ESTRATIFICADO do VIP. Reamostra com reposicao DENTRO de
    cada classe, garantindo presenca de todas as classes em toda iteracao.

    Retorna dict com:
        mean, std, ci95_low, ci95_high  - estatisticas pontuais por variavel
        selection_frequency             - fracao de bootstraps em que
                                          VIP >= vip_threshold
        n_validos, n_falhos             - contagem de iteracoes
    """
    rng = np.random.default_rng(seed)
    n_var = X_processed.shape[1]
    classes = np.unique(y_int)
    indices_por_classe = {int(c): np.where(y_int == c)[0] for c in classes}

    vips_arr: List[np.ndarray] = []
    n_validos = 0
    n_falhos = 0

    for _ in range(n_boot):
        partes = []
        for c in classes:
            idx_c = indices_por_classe[int(c)]
            partes.append(rng.choice(idx_c, size=len(idx_c), replace=True))
        idx = np.concatenate(partes)
        try:
            pls = PLSRegression(n_components=n_opt, scale=False)
            pls.fit(X_processed[idx], Y_bin[idx])
            v = vip_scores(pls)
            if np.all(np.isfinite(v)):
                vips_arr.append(v)
                n_validos += 1
            else:
                n_falhos += 1
        except Exception:
            n_falhos += 1

    if not vips_arr:
        zeros = np.zeros(n_var)
        return {
            "mean": zeros, "std": zeros,
            "ci95_low": zeros, "ci95_high": zeros,
            "selection_frequency": zeros,
            "n_validos": 0, "n_falhos": n_falhos,
        }

    arr = np.asarray(vips_arr)
    return {
        "mean":      arr.mean(axis=0),
        "std":       arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros(n_var),
        "ci95_low":  np.percentile(arr, 2.5,  axis=0),
        "ci95_high": np.percentile(arr, 97.5, axis=0),
        "selection_frequency": (arr >= vip_threshold).mean(axis=0),
        "n_validos": n_validos,
        "n_falhos":  n_falhos,
    }


def bootstrap_vip(X_processed, Y_bin, n_opt, n_boot, seed):
    """DEPRECATED: usar bootstrap_vip_estratificado. Mantido para
    compatibilidade. NAO usa estratificacao por classe."""
    print("[AVISO] bootstrap_vip nao-estratificado e DEPRECATED. "
          "Use bootstrap_vip_estratificado.")
    rng = np.random.default_rng(seed)
    n = len(X_processed)
    vips = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            pls = PLSRegression(n_components=n_opt, scale=False)
            pls.fit(X_processed[idx], Y_bin[idx])
            vips.append(vip_scores(pls))
        except Exception:
            continue
    if not vips:
        p = X_processed.shape[1]
        return np.zeros(p), np.zeros(p)
    vips = np.asarray(vips)
    return vips.mean(axis=0), vips.std(axis=0)


def bootstrap_bca_ci(y_true: np.ndarray, y_pred: np.ndarray,
                      metric_fn: Callable, n_boot: int = 500,
                      alpha: float = 0.05, seed: int = 42
                      ) -> Tuple[float, float, float]:
    """Intervalo de confianca BCa (bias-corrected & accelerated, Efron 1987)
    para uma metrica de classificacao via bootstrap estratificado.

    Retorna (low, high, valor_observado).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    observed = float(metric_fn(y_true, y_pred))

    classes = np.unique(y_true)
    idx_por_classe = {c: np.where(y_true == c)[0] for c in classes}

    boot_stats = []
    for _ in range(n_boot):
        partes = []
        for c in classes:
            ic = idx_por_classe[c]
            partes.append(rng.choice(ic, size=len(ic), replace=True))
        idx = np.concatenate(partes)
        try:
            boot_stats.append(float(metric_fn(y_true[idx], y_pred[idx])))
        except Exception:
            continue
    boot_stats = np.asarray(boot_stats)

    if len(boot_stats) < 20:
        return float("nan"), float("nan"), observed

    # Bias correction z0
    prop_less = float(np.mean(boot_stats < observed))
    if prop_less <= 0 or prop_less >= 1:
        return (float(np.percentile(boot_stats, 100 * alpha / 2)),
                float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
                observed)
    z0 = _norm_dist.ppf(prop_less)

    # Aceleracao via jackknife
    jack = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        try:
            jack[i] = float(metric_fn(y_true[mask], y_pred[mask]))
        except Exception:
            jack[i] = observed
    mean_jack = jack.mean()
    diffs = mean_jack - jack
    num = float(np.sum(diffs ** 3))
    den = 6.0 * (float(np.sum(diffs ** 2))) ** 1.5
    a = num / den if den > 0 else 0.0

    z_a_lo = _norm_dist.ppf(alpha / 2)
    z_a_hi = _norm_dist.ppf(1 - alpha / 2)
    denom_lo = 1 - a * (z0 + z_a_lo)
    denom_hi = 1 - a * (z0 + z_a_hi)
    if denom_lo == 0 or denom_hi == 0:
        return (float(np.percentile(boot_stats, 100 * alpha / 2)),
                float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
                observed)
    alpha_lo = _norm_dist.cdf(z0 + (z0 + z_a_lo) / denom_lo)
    alpha_hi = _norm_dist.cdf(z0 + (z0 + z_a_hi) / denom_hi)

    low  = float(np.percentile(boot_stats, 100 * alpha_lo))
    high = float(np.percentile(boot_stats, 100 * alpha_hi))
    return low, high, observed


def cv_anova_eriksson(Y: np.ndarray, Y_cv: np.ndarray, n_components: int
                       ) -> Dict[str, float]:
    """CV-ANOVA de Eriksson, Trygg & Wold (J. Chemometrics 22:594-600, 2008).

    Testa se PRESS (residuo CV) e significativamente menor que SS_total
    (variancia em torno da media de Y). H0: modelo nao melhora previsao.

    F = ((SS_total - PRESS) / df_model) / (PRESS / df_residual)
    """
    Y = np.asarray(Y, dtype=float)
    Y_cv = np.asarray(Y_cv, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
        Y_cv = Y_cv.reshape(-1, 1)
    n, m = Y.shape

    ss_total = float(np.sum((Y - Y.mean(axis=0)) ** 2))
    press    = float(np.sum((Y - Y_cv) ** 2))

    if ss_total <= 0:
        return {"F": float("nan"), "p_value": 1.0, "Q2": 0.0,
                "df_model": 0, "df_resid": 0}
    if press >= ss_total:
        return {"F": 0.0, "p_value": 1.0,
                "Q2": 1.0 - press / ss_total,
                "df_model": n_components * m, "df_resid": max(n * m - n_components * m, 1)}

    df_model = n_components * m
    df_resid = n * m - df_model
    if df_resid <= 0:
        return {"F": float("nan"), "p_value": 1.0,
                "Q2": 1.0 - press / ss_total,
                "df_model": df_model, "df_resid": df_resid}

    F = ((ss_total - press) / df_model) / (press / df_resid)
    p = float(1.0 - f_dist.cdf(F, df_model, df_resid))
    return {"F":        float(F),
             "p_value":  p,
             "Q2":       1.0 - press / ss_total,
             "df_model": int(df_model),
             "df_resid": int(df_resid)}


def teste_wold(pipeline_factory: Callable[[], Pipeline],
                X: np.ndarray, Y_bin: np.ndarray, y_int: np.ndarray,
                cv, n_perm: int, seed: int,
                groups: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Permutation test estilo Wold/Westerhuis (J. Chemometrics 22:578-585):
    rastreia R2Y e Q2Y para cada permutacao em funcao da similaridade
    do Y permutado com o original. Ajusta reta e reporta interceptos.

    Criterio classico para validade do modelo (one-hot Y):
        intercepto R2Y < 0.4
        intercepto Q2Y < 0.05
    """
    rng = np.random.default_rng(seed)
    cv_indices = list(cv.split(X, y_int, groups=groups))

    # --- Observado --------------------------------------------------------
    pipe = pipeline_factory(); pipe.fit(X, Y_bin)
    Y_train_obs = pipe.predict(X)
    Y_cv_obs    = _cv_predict_manual(pipeline_factory, X, Y_bin, cv_indices)
    ss_total    = float(np.sum((Y_bin - Y_bin.mean(axis=0)) ** 2))
    r2_obs = 1.0 - float(np.sum((Y_bin - Y_train_obs) ** 2)) / ss_total \
                if ss_total > 0 else 0.0
    q2_obs = 1.0 - float(np.sum((Y_bin - Y_cv_obs)    ** 2)) / ss_total \
                if ss_total > 0 else 0.0

    sims: List[float] = []
    r2s:  List[float] = []
    q2s:  List[float] = []
    n_validos = 0
    n_falhos  = 0

    import time as _time
    t0 = _time.time()
    progress_every = max(1, n_perm // 20)   # ~20 updates

    for i in range(n_perm):
        idx = rng.permutation(len(Y_bin))
        Y_perm = Y_bin[idx]
        y_perm_int = np.argmax(Y_perm, axis=1)
        sim = float(np.mean(y_perm_int == y_int))
        try:
            cv_perm_idx = list(cv.split(X, y_perm_int, groups=groups))
            pipe = pipeline_factory(); pipe.fit(X, Y_perm)
            Y_tr_p = pipe.predict(X)
            Y_cv_p = _cv_predict_manual(pipeline_factory, X, Y_perm,
                                         cv_perm_idx)
            ss_tot_p = float(np.sum((Y_perm - Y_perm.mean(axis=0)) ** 2))
            if ss_tot_p <= 0:
                continue
            r2 = 1.0 - float(np.sum((Y_perm - Y_tr_p) ** 2)) / ss_tot_p
            q2 = 1.0 - float(np.sum((Y_perm - Y_cv_p) ** 2)) / ss_tot_p
            sims.append(sim); r2s.append(r2); q2s.append(q2)
            n_validos += 1
        except Exception:
            n_falhos += 1

        # Progresso com ETA
        if (i + 1) % progress_every == 0 or (i + 1) == n_perm:
            elapsed = _time.time() - t0
            taxa    = (i + 1) / max(elapsed, 1e-6)
            eta_s   = (n_perm - i - 1) / max(taxa, 1e-6)
            pct = (i + 1) / n_perm * 100
            print(f"    Wold {i+1:4d}/{n_perm}  ({pct:5.1f}%)  "
                  f"valid={n_validos} falhos={n_falhos}  "
                  f"elapsed={elapsed:5.1f}s  ETA={eta_s:5.1f}s",
                  flush=True)

    sims_arr = np.asarray(sims); r2s_arr = np.asarray(r2s); q2s_arr = np.asarray(q2s)
    # Adiciona ponto observado (sim=1)
    sims_all = np.append(sims_arr, 1.0)
    r2_all   = np.append(r2s_arr, r2_obs)
    q2_all   = np.append(q2s_arr, q2_obs)

    if len(sims_all) >= 2 and np.ptp(sims_all) > 0:
        slope_r2, int_r2 = np.polyfit(sims_all, r2_all, 1)
        slope_q2, int_q2 = np.polyfit(sims_all, q2_all, 1)
    else:
        slope_r2 = int_r2 = slope_q2 = int_q2 = float("nan")

    return {
        "sims":         sims_arr,
        "r2s":          r2s_arr,
        "q2s":          q2s_arr,
        "r2_obs":       float(r2_obs),
        "q2_obs":       float(q2_obs),
        "intercept_r2": float(int_r2),
        "intercept_q2": float(int_q2),
        "slope_r2":     float(slope_r2),
        "slope_q2":     float(slope_q2),
        "valid_r2":     bool(int_r2 < 0.40) if np.isfinite(int_r2) else False,
        "valid_q2":     bool(int_q2 < 0.05) if np.isfinite(int_q2) else False,
        "n_validos":    n_validos,
        "n_falhos":     n_falhos,
    }


def teste_permutacao(pipeline_factory: Callable[[], Pipeline],
                      X: np.ndarray, Y_bin: np.ndarray, y_int: np.ndarray,
                      cv, n_perm: int, seed: int,
                      groups: Optional[np.ndarray] = None
                      ) -> Dict[str, object]:
    """Y-randomization robusta. Iteracoes que falham (e.g. estratificacao
    impossivel apos shuffle) sao registradas e ignoradas no p-value.

    Retorna dict com:
        acc_observada      - acuracia com Y verdadeiro
        accs_permutadas    - array de acuracias H0 (apenas iteracoes validas)
        p_value            - (sum(accs >= obs) + 1) / (n_validos + 1)
        n_validos          - iteracoes que completaram com sucesso
        n_falhos           - iteracoes abortadas por erro
        failure_rate       - n_falhos / n_perm
    """
    rng = np.random.default_rng(seed)

    cv_indices = list(cv.split(X, y_int, groups=groups))
    y_hat = _cv_predict_manual(pipeline_factory, X, Y_bin, cv_indices)
    acc_obs = accuracy_score(y_int, np.argmax(y_hat, axis=1))

    accs: List[float] = []
    n_falhos = 0
    import time as _time
    t0 = _time.time()
    progress_every = max(1, n_perm // 20)

    for i in range(n_perm):
        idx = rng.permutation(len(Y_bin))
        Y_perm = Y_bin[idx]
        y_perm_int = np.argmax(Y_perm, axis=1)
        try:
            cv_perm_idx = list(cv.split(X, y_perm_int, groups=groups))
            y_hat = _cv_predict_manual(pipeline_factory, X, Y_perm, cv_perm_idx)
            accs.append(float(accuracy_score(y_perm_int,
                                              np.argmax(y_hat, axis=1))))
        except Exception as e:
            n_falhos += 1
            continue

        # Progresso com ETA
        if (i + 1) % progress_every == 0 or (i + 1) == n_perm:
            elapsed = _time.time() - t0
            taxa    = (i + 1) / max(elapsed, 1e-6)
            eta_s   = (n_perm - i - 1) / max(taxa, 1e-6)
            pct = (i + 1) / n_perm * 100
            print(f"    Perm {i+1:4d}/{n_perm}  ({pct:5.1f}%)  "
                  f"valid={len(accs)} falhos={n_falhos}  "
                  f"elapsed={elapsed:5.1f}s  ETA={eta_s:5.1f}s",
                  flush=True)

    n_validos = len(accs)
    failure_rate = n_falhos / n_perm if n_perm > 0 else 0.0
    accs_arr = np.asarray(accs, dtype=float)

    if failure_rate > 0.30:
        print(f"[AVISO] Permutation test: taxa de falha = "
              f"{failure_rate:.1%} ({n_falhos}/{n_perm}). "
              f"Resultado pode ser nao-confiavel (classes muito "
              f"desbalanceadas para CV estratificada apos shuffle).")

    if n_validos == 0:
        print("[ERRO] Permutation test: 0 iteracoes validas. "
              "p_value retornado como 1.0 (nao informativo).")
        p_val = 1.0
    else:
        p_val = float((np.sum(accs_arr >= acc_obs) + 1) / (n_validos + 1))

    return {
        "acc_observada":   float(acc_obs),
        "accs_permutadas": accs_arr,
        "p_value":         p_val,
        "n_validos":       n_validos,
        "n_falhos":        n_falhos,
        "failure_rate":    failure_rate,
    }


# =========================================================================
#  Carregamento de dados
# =========================================================================

def gerar_dados_sinteticos(cfg: Config):
    print("[INFO] MODO sintetico — gerando espectros de teste.")
    rng = np.random.default_rng(cfg.seed)
    wavenumbers = np.linspace(4000, 400, cfg.n_pontos_sint)
    conc_base = np.linspace(0, 40, cfg.n_por_classe)

    def esp(c, p1, p2, ruido=0.015):
        frac = c / 100
        return ((1 - frac) * np.exp(-((wavenumbers - p1) ** 2) / (2 * 50 ** 2))
                + frac     * np.exp(-((wavenumbers - (p1 + 20)) ** 2) / (2 * 45 ** 2))
                + 0.6      * np.exp(-((wavenumbers - p2) ** 2) / (2 * 30 ** 2))
                + rng.normal(0, ruido, cfg.n_pontos_sint))

    params  = [(2900, 1740), (2850, 1650), (2960, 1710)]
    classes = ["Esp_A", "Esp_B", "Esp_C"]
    X_list, rot_list, conc_list = [], [], []
    for (p1, p2), cls in zip(params, classes):
        for c in conc_base:
            X_list.append(esp(c, p1, p2)); rot_list.append(cls); conc_list.append(c)

    return (wavenumbers,
            np.array(X_list, dtype=float),
            np.array(rot_list, dtype=str),
            np.array(conc_list, dtype=float))


def carregar_csv(caminho, col_classe, col_conc):
    print(f"[INFO] Carregando CSV: {caminho}")
    df          = pd.read_csv(caminho)
    rotulos     = np.asarray(df[col_classe].values, dtype=str)
    conc        = np.asarray(df[col_conc].values, dtype=float) if col_conc else None
    excluir     = [c for c in [col_classe, col_conc] if c]
    cols_spec   = [c for c in df.columns if c not in excluir]
    X_raw       = np.asarray(df[cols_spec].values, dtype=float)
    wavenumbers = np.array([float(c) for c in cols_spec])
    return wavenumbers, X_raw, rotulos, conc


def _flush_asdf(y_raw, sign, digits, is_dif):
    if sign == 0:
        return
    num = int("".join(str(d) for d in digits)) if digits else 0
    val = sign * num
    if is_dif:
        val = (y_raw[-1] if y_raw else 0) + val
    y_raw.append(val)


def _decodificar_linha_asdf(s: str, SQZ: dict, DIF: dict, DUP: dict
                             ) -> Tuple[Optional[float], List[float]]:
    """Decodifica uma linha ASDF '(X++(Y..Y))': retorna (x_check, [y_raw]).

    x_check e o valor de abscissa (ainda SEM xfactor) que prefixa a linha;
    serve para ancorar a posicao do bloco na grade global."""
    i, n = 0, len(s)
    x_str = ""
    while i < n and (s[i].isdigit() or s[i] in "+-"):
        x_str += s[i]; i += 1
    if not x_str:
        return None, []
    x_check = float(x_str)

    y_raw: List[float] = []
    sign, digits, is_dif = 0, [], False
    while i < n:
        ch = s[i]; i += 1
        if ch in SQZ:
            _flush_asdf(y_raw, sign, digits, is_dif)
            v = SQZ[ch]; sign = 1 if v >= 0 else -1
            digits = [abs(v)]; is_dif = False
        elif ch in DIF:
            _flush_asdf(y_raw, sign, digits, is_dif)
            v = DIF[ch]; sign = 1 if v >= 0 else -1
            digits = [abs(v)]; is_dif = True
        elif ch in DUP:
            _flush_asdf(y_raw, sign, digits, is_dif)
            sign = 0; last = y_raw[-1] if y_raw else 0
            for _ in range(DUP[ch]): y_raw.append(last)
        elif ch.isdigit() and sign != 0:
            digits.append(int(ch))
    _flush_asdf(y_raw, sign, digits, is_dif)
    return x_check, y_raw


def parse_dx(filepath):
    """Parser JCAMP-DX para formato comprimido '(X++(Y..Y))' (ASDF).

    Estrategia robusta de reconstrucao do eixo:
      - O eixo X e reconstruido como np.linspace(FIRSTX, LASTX, NPOINTS)
        usando o cabecalho (autoritativo), NUNCA os X encodados.
      - Cada linha de dados e ANCORADA pela sua abscissa de checagem
        (x_check) no indice correto da grade. Isso e auto-corretivo:
        linhas que sobre/sub-decodificam sao re-ancoradas pela linha
        seguinte (resolve o bug de concatenacao cega que perdia pontos).
      - Lacunas residuais (NaN) sao interpoladas linearmente.

    Fallback: se faltar FIRSTX/LASTX/NPOINTS, usa concatenacao simples
    com o X encodado (modo legado, menos confiavel)."""
    SQZ, DIF, DUP = {"@": 0}, {"%": 0}, {}
    for i, c in enumerate("ABCDEFGHI", 1): SQZ[c] =  i
    for i, c in enumerate("abcdefghi", 1): SQZ[c] = -i
    for i, c in enumerate("JKLMNOPQR", 1): DIF[c] =  i
    for i, c in enumerate("jklmnopqr", 1): DIF[c] = -i
    for i, c in enumerate("STUVWXYZs", 1): DUP[c] =  i

    xfactor, yfactor, lendo_dados = 1.0, 1.0, False
    firstx = lastx = None
    npoints: Optional[int] = None
    linhas_dados: List[str] = []

    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        s = line.strip()
        if   s.startswith("##XFACTOR="): xfactor = float(s.split("=", 1)[1])
        elif s.startswith("##YFACTOR="): yfactor = float(s.split("=", 1)[1])
        elif s.startswith("##FIRSTX="):  firstx  = float(s.split("=", 1)[1])
        elif s.startswith("##LASTX="):   lastx   = float(s.split("=", 1)[1])
        elif s.startswith("##NPOINTS="):
            try: npoints = int(float(s.split("=", 1)[1]))
            except ValueError: npoints = None
        elif s.startswith("##XYDATA=") or s.startswith("##XYPOINTS="):
            lendo_dados = True; continue
        elif s.startswith("##END"):      break
        elif s.startswith("##") and lendo_dados: lendo_dados = False
        elif lendo_dados and s:
            linhas_dados.append(s)

    # --- Reconstrucao robusta ancorada por X-check ----------------------
    if (firstx is not None and lastx is not None and npoints
            and npoints > 1 and lastx != firstx):
        dx = (lastx - firstx) / (npoints - 1)
        Y = np.full(npoints, np.nan, dtype=float)
        for s in linhas_dados:
            x_check, y_raw = _decodificar_linha_asdf(s, SQZ, DIF, DUP)
            if x_check is None or not y_raw:
                continue
            x_real = x_check * xfactor
            idx0 = int(round((x_real - firstx) / dx))
            for j, yv in enumerate(y_raw):
                pos = idx0 + j
                if 0 <= pos < npoints:
                    Y[pos] = yv * yfactor
        X = np.linspace(firstx, lastx, npoints)
        nan_mask = np.isnan(Y)
        n_nan = int(nan_mask.sum())
        if n_nan > 0 and n_nan < npoints:
            Y[nan_mask] = np.interp(X[nan_mask], X[~nan_mask], Y[~nan_mask])
        return X, Y

    # --- Fallback legado (concatenacao com X encodado) ------------------
    x_all: List[float] = []
    y_all: List[float] = []
    for s in linhas_dados:
        x_check, y_raw = _decodificar_linha_asdf(s, SQZ, DIF, DUP)
        if x_check is None:
            continue
        x_first = x_check * xfactor
        for yr in y_raw:
            x_all.append(x_first); y_all.append(yr * yfactor)
    if y_all and firstx is not None and lastx is not None:
        x_arr = np.linspace(firstx, lastx, len(y_all))
    else:
        x_arr = np.array(x_all, dtype=float)
    return np.asarray(x_arr, dtype=float), np.array(y_all, dtype=float)


_REGEX_FLOAT = re.compile(r"[-+]?\d+[.,]?\d*(?:[eE][-+]?\d+)?")
_REGEX_CONC  = re.compile(r"(\d+[.,]?\d*)\s*%")


def parse_spectrum(filepath):
    """Parser ASCII generico para arquivos .spectrum/.txt/.csv com x,y por
    linha. Tolera cabecalho, separadores variaveis (espaco/tab/virgula/;)
    e decimal-virgula. Retorna (x, y) como ndarrays.

    Detecta arquivos binarios conhecidos (Bomem MB, PerkinElmer, Bruker OPUS)
    e emite mensagem com instrucao de re-exportar como JCAMP-DX."""
    with open(filepath, "rb") as f:
        head = f.read(256)

    # Detecta formato Bomem (ABB Horizon MB) - UTF-16-LE "Bomem File"
    try:
        head_text = head.decode("utf-16-le", errors="ignore")
        if "Bomem" in head_text or "Horizon" in head_text:
            raise ValueError(
                f"Formato detectado: ABB Bomem Horizon MB (.spectrum binario).\n"
                f"  Arquivo: {os.path.basename(filepath)}\n"
                f"  ESTE PARSER NAO LE BINARIOS PROPRIETARIOS por seguranca\n"
                f"  cientifica. Para usar com este pipeline:\n"
                f"    1. Abra os espectros no Bomem Horizon software\n"
                f"    2. File -> Export -> JCAMP-DX (.dx) ou ASCII (.txt)\n"
                f"    3. Aponte cfg.pasta_entrada para a nova pasta exportada\n"
                f"  O parser ja suporta .dx e .txt automaticamente.")
    except UnicodeDecodeError:
        pass

    # Detecta PerkinElmer .sp / Bruker OPUS
    if head[:4] == b"PEPE" or head[:4] == b"\x00\x00\x00\x00" and b"OPUS" in head:
        raise ValueError(
            f"Formato binario proprietario detectado: {os.path.basename(filepath)}.\n"
            f"  -> exporte como JCAMP-DX (.dx) ou ASCII (.txt) pelo software original.")

    n_bad = sum(1 for b in head[:64] if b < 9 or (13 < b < 32) or b > 126)
    if n_bad > 64 * 0.2:
        raise ValueError(
            f"Arquivo binario nao reconhecido: {os.path.basename(filepath)}\n"
            f"  -> exporte como JCAMP-DX (.dx) ou ASCII (.txt) pelo software\n"
            f"     do instrumento (Bomem/PerkinElmer/Bruker/etc).")

    x_list: List[float] = []
    y_list: List[float] = []
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            tokens = _REGEX_FLOAT.findall(line)
            if len(tokens) < 2:
                continue
            try:
                x = float(tokens[0].replace(",", "."))
                y = float(tokens[1].replace(",", "."))
            except ValueError:
                continue
            x_list.append(x); y_list.append(y)
    return np.array(x_list, dtype=float), np.array(y_list, dtype=float)


def _extrair_conc_filename(nome: str) -> Optional[float]:
    """Extrai a primeira ocorrencia de N,NN% no nome do arquivo."""
    m = _REGEX_CONC.search(nome)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except ValueError:
        return None


def _listar_arquivos_espectro(pasta: str
                                ) -> Tuple[List[str], Optional[str]]:
    """Procura arquivos espectrais por extensao na pasta. Retorna (lista, ext)."""
    extensoes = [".dx", ".spectrum", ".txt", ".csv"]
    for ext in extensoes:
        candidatos = sorted(glob.glob(os.path.join(pasta, f"*{ext}")))
        if candidatos:
            return candidatos, ext
    return [], None


def _detectar_subpastas_classe(raiz: str) -> List[str]:
    """Retorna subpastas que contem >=1 arquivo .dx/.spectrum/.txt/.csv."""
    if not os.path.isdir(raiz):
        return []
    out: List[str] = []
    for nome in sorted(os.listdir(raiz)):
        cand = os.path.join(raiz, nome)
        if os.path.isdir(cand):
            arqs, _ = _listar_arquivos_espectro(cand)
            if arqs:
                out.append(cand)
    return out


def carregar_dx(pasta: str, parte_classe: int = 0,
                 extrair_conc: bool = False,
                 usar_parse_title: bool = True
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                              Optional[np.ndarray], Optional[np.ndarray],
                              Optional[pd.DataFrame]]:
    """Carrega espectros, com auto-deteccao de estrutura:

        (A) pasta-raiz com subpastas (cada subpasta = 1 especie/classe)
            -> recursivo via _detectar_subpastas_classe
        (B) pasta unica com arquivos .dx/.spectrum/.txt/.csv
            -> modo legado (parte_classe)

    Quando usar_parse_title=True e o arquivo for .dx, extrai ##TITLE= e
    usa parse_title() para metadados ricos (especie, adulterante, teor,
    triplicata, mae_id). Caso contrario, usa o nome do arquivo (fallback).

    Retorna: (wavenumbers, X, rotulos, conc, mae_id, metadados_df).
        - mae_id   : ndarray de strings ou None se nao disponivel
        - metadados_df: pd.DataFrame com todos os campos parseados
    """
    subpastas = _detectar_subpastas_classe(pasta)
    if subpastas:
        print(f"[INFO] Estrutura multi-pasta detectada: {len(subpastas)} "
              f"subpastas em {pasta}")
        arquivos: List[Tuple[str, str]] = []   # (caminho, nome_subpasta)
        for sp in subpastas:
            arqs, _ = _listar_arquivos_espectro(sp)
            arquivos.extend((a, os.path.basename(sp)) for a in arqs)
        ext_usada = os.path.splitext(arquivos[0][0])[1] if arquivos else None
    else:
        if not os.path.isdir(pasta):
            raise FileNotFoundError(
                f"Caminho NAO existe: {pasta}\n"
                f"  -> verifique cfg.pasta_entrada ou use cfg.modo='sintetico'.")
        arqs, ext_usada = _listar_arquivos_espectro(pasta)
        if not arqs:
            raise FileNotFoundError(
                f"Pasta existe mas nao contem espectros conhecidos.\n"
                f"  Pasta: {pasta}\n"
                f"  Conteudo (ate 10 itens): {os.listdir(pasta)[:10]}")
        arquivos = [(a, "") for a in arqs]

    parser = parse_dx if ext_usada == ".dx" else parse_spectrum
    print(f"[INFO] {len(arquivos)} arquivos {ext_usada} encontrados "
          f"(parser={parser.__name__})")

    pode_parse_title = usar_parse_title and ext_usada == ".dx"

    espectros: List[Tuple[np.ndarray, np.ndarray]] = []
    rotulos: List[str] = []
    concs:   List[Optional[float]] = []
    mae_ids: List[Optional[str]]   = []
    meta_rows: List[Dict[str, Any]] = []
    n_falhos = 0
    n_title_falhos = 0
    cods_desconhecidos: set = set()

    for arq, subpasta_nome in arquivos:
        try:
            x, y = parser(arq)
        except Exception as e:
            n_falhos += 1
            print(f"  [ERRO] {os.path.basename(arq)}: {e}")
            continue
        if len(x) == 0:
            print(f"  [AVISO] {os.path.basename(arq)} sem dados — ignorado")
            continue

        nome_arq = os.path.splitext(os.path.basename(arq))[0]
        title_parsed: Optional[Dict[str, Any]] = None
        if pode_parse_title:
            title = extrair_title_do_dx(arq)
            if title:
                title_parsed = parse_title(title)
            if title_parsed is None:
                n_title_falhos += 1

        if title_parsed is not None:
            classe = title_parsed["especie"]
            conc_i = title_parsed["teor"]
            mae_i  = title_parsed["mae_id"]
            if not title_parsed["cod_conhecido"]:
                cods_desconhecidos.add(title_parsed["cod"])
            meta = dict(title_parsed)
            meta["arquivo"]   = os.path.basename(arq)
            meta["subpasta"]  = subpasta_nome
        else:
            # Fallback: tenta extrair COD do prefixo do filename e mapear
            # para o nome canonico (evita classe duplicada por acento:
            # subpasta 'Copaiba' vs CODIGO_ESPECIE['COP']='Copaíba').
            m_cod = re.match(r"^([A-Z]{2,4})[-_]", nome_arq)
            cod_fb = m_cod.group(1).upper() if m_cod else None
            if cod_fb and cod_fb in CODIGO_ESPECIE:
                classe = CODIGO_ESPECIE[cod_fb]
            elif subpasta_nome:
                classe = subpasta_nome
            else:
                partes = nome_arq.replace("_", "-").split("-")
                classe = partes[parte_classe] if abs(parte_classe) < len(partes) \
                          else nome_arq
            conc_i = _extrair_conc_filename(nome_arq) if extrair_conc else None
            mae_i  = None
            meta = {
                "arquivo":     os.path.basename(arq),
                "subpasta":    subpasta_nome,
                "especie":     classe,
                "teor":        conc_i,
                "puro":        conc_i is None,
                "mae_id":      None,
                "cod":         cod_fb,
                "cod_conhecido": bool(cod_fb and cod_fb in CODIGO_ESPECIE),
            }

        espectros.append((x, y))
        rotulos.append(classe)
        concs.append(conc_i)
        mae_ids.append(mae_i)
        meta_rows.append(meta)

    if not espectros:
        raise ValueError(
            f"Nenhum espectro valido carregado. ({n_falhos} arquivos com erro)")
    if n_falhos > 0:
        print(f"[AVISO] {n_falhos} arquivos com erro de parsing — ignorados.")
    if pode_parse_title:
        if n_title_falhos > 0:
            print(f"[AVISO] {n_title_falhos} arquivos com ##TITLE= "
                  f"nao-conforme — usando fallback (nome/subpasta).")
        if cods_desconhecidos:
            print(f"[ATENCAO] CODs nao mapeados em CODIGO_ESPECIE: "
                  f"{sorted(cods_desconhecidos)}")

    # --- Deteccao de faixa de aquisicao dominante -----------------------
    # Datasets reais podem misturar faixas espectrais (e.g. NIR completo
    # [0,15797] 8192pts vs faixa estreita [300,4000]). Misturar e
    # cientificamente invalido. Detecta a faixa dominante (moda do xmax
    # arredondado a 100 cm-1) e DESCARTA os incompativeis com relatorio.
    maxes = np.array([float(e[0].max()) for e in espectros])
    chave_faixa = np.round(maxes / 100.0) * 100.0
    valores, contagens = np.unique(chave_faixa, return_counts=True)
    faixa_dominante = float(valores[int(np.argmax(contagens))])
    compat = np.abs(chave_faixa - faixa_dominante) < 50.0
    n_drop = int((~compat).sum())

    if n_drop > 0:
        print(f"[ATENCAO] Faixas espectrais heterogeneas detectadas. "
              f"Faixa dominante: xmax~{faixa_dominante:.0f} cm-1 "
              f"({int(compat.sum())} arquivos).")
        print(f"          DESCARTANDO {n_drop} arquivos de faixa "
              f"incompativel (nao comparaveis na mesma janela espectral):")
        # Relatorio por especie dos descartados
        from collections import Counter
        drop_especies = Counter(
            meta_rows[i].get("especie", "?")
            for i in range(len(espectros)) if not compat[i])
        for esp, n in sorted(drop_especies.items(), key=lambda kv: -kv[1]):
            exemplos = [meta_rows[i]["arquivo"]
                         for i in range(len(espectros))
                         if not compat[i]
                         and meta_rows[i].get("especie") == esp][:1]
            print(f"            {esp}: {n} arquivos (ex: {exemplos[0] if exemplos else '?'})")
        # Filtra todas as listas paralelas
        keep = [i for i in range(len(espectros)) if compat[i]]
        espectros = [espectros[i] for i in keep]
        rotulos   = [rotulos[i]   for i in keep]
        concs     = [concs[i]     for i in keep]
        mae_ids   = [mae_ids[i]   for i in keep]
        meta_rows = [meta_rows[i] for i in keep]
        print(f"          Restam {len(espectros)} arquivos na faixa "
              f"dominante.")

    if not espectros:
        raise ValueError("Nenhum espectro restante apos filtro de faixa.")

    # Grade comum por interpolacao (agora dentro da faixa dominante)
    xmin  = max(e[0].min() for e in espectros)
    xmax  = min(e[0].max() for e in espectros)
    n_pts = min(2000, min(len(e[0]) for e in espectros))
    grade = np.linspace(xmin, xmax, n_pts)
    X_raw = []
    for x, y in espectros:
        idx = np.argsort(x)
        X_raw.append(np.interp(grade, x[idx], y[idx]))

    # Concentracoes (puros = 0% por convencao)
    if any(c is not None for c in concs):
        conc_arr: Optional[np.ndarray] = np.array(
            [c if c is not None else 0.0 for c in concs], dtype=float)
        n_com_conc = sum(1 for c in concs if c is not None)
        print(f"[INFO] Teores extraidos: {n_com_conc}/{len(concs)} "
              f"(puros tratados como 0%).")
    else:
        conc_arr = None

    # mae_id array. Para arquivos sem parse valido, atribui ID unico
    # (filename) — vira "grupo de 1", isolando-os sem desativar todo o
    # GroupKFold do dataset.
    n_com_mae = sum(1 for m in mae_ids if m is not None)
    if n_com_mae == 0:
        mae_arr: Optional[np.ndarray] = None
    else:
        n_orfaos = 0
        mae_final: List[str] = []
        for m, row in zip(mae_ids, meta_rows):
            if m is not None:
                mae_final.append(m)
            else:
                mae_final.append(f"orfao_{row['arquivo']}")
                n_orfaos += 1
        mae_arr = np.array(mae_final, dtype=str)
        n_grupos = int(len(np.unique(mae_arr)))
        msg_orfaos = f", {n_orfaos} orfaos isolados" if n_orfaos > 0 else ""
        print(f"[INFO] mae_id: {n_com_mae}/{len(mae_ids)} amostras parseadas, "
              f"{n_grupos} grupos unicos{msg_orfaos}.")

    metadados_df = pd.DataFrame(meta_rows)

    return (grade,
            np.array(X_raw, dtype=float),
            np.array(rotulos, dtype=str),
            conc_arr, mae_arr, metadados_df)


def carregar_dados(cfg: Config
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                Optional[np.ndarray], Optional[np.ndarray],
                                Optional[pd.DataFrame]]:
    """Loader unificado. Retorna 6-tupla:
        (wavenumbers, X, rotulos, conc, mae_id, metadados_df)
    mae_id e metadados_df podem ser None em modo 'sintetico'/'csv'."""
    if cfg.modo == "sintetico":
        wn, X, rot, conc = gerar_dados_sinteticos(cfg)
        return wn, X, rot, conc, None, None
    if cfg.modo == "csv":
        wn, X, rot, conc = carregar_csv(
            cfg.arquivo_csv, cfg.coluna_classe, cfg.coluna_conc)
        return wn, X, rot, conc, None, None
    if cfg.modo == "dx":
        return carregar_dx(cfg.pasta_entrada, cfg.parte_classe,
                            cfg.extrair_conc_filename,
                            cfg.usar_parse_title)
    raise ValueError(f"MODO desconhecido: '{cfg.modo}'.")


# =========================================================================
#  Plot helpers
# =========================================================================

def elipse_t2(ax, x, y, color, lw=1.4, alpha=0.85,
              max_excentricidade: float = 50.0,
              limite_dispersao: Optional[Tuple[float, float]] = None):
    """Hotelling T2 95% (chi2_{2,0.95}). Sem preenchimento. Retorna False
    em caso degenerado: <4 pontos, autovalor minimo nulo, excentricidade
    excessiva ou elipse maior que o proprio range dos dados."""
    if len(x) < 4:
        return False
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if vals[1] <= 1e-12 or vals[0] / max(vals[1], 1e-12) > max_excentricidade:
        return False
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    escala = np.sqrt(chi2.ppf(0.95, df=2))
    w = 2 * escala * np.sqrt(vals[0])
    h = 2 * escala * np.sqrt(vals[1])
    if limite_dispersao is not None:
        max_w, max_h = limite_dispersao
        if w > max_w * 2.5 or h > max_h * 2.5:
            return False
    ell = Ellipse(xy=(x.mean(), y.mean()), width=w, height=h, angle=angle,
                  edgecolor=color, facecolor="none", lw=lw,
                  alpha=alpha, zorder=2)
    ax.add_patch(ell)
    return True


def convex_hull_contorno(ax, x, y, color, lw=1.4, alpha=0.85):
    """Convex hull (fallback quando n_per_class < 4 para elipse)."""
    pts = np.column_stack([x, y])
    if len(pts) < 3:
        return False
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        verts = pts[hull.vertices]
        verts = np.vstack([verts, verts[0]])
        ax.plot(verts[:, 0], verts[:, 1], color=color, lw=lw, alpha=alpha,
                zorder=2)
        return True
    except Exception:
        return False


def parametros_scatter_adaptativos(n_total: int, n_classes: int
                                    ) -> Tuple[float, float, float]:
    """Tamanho do marker, alpha e largura da borda em funcao da densidade."""
    n_pc = n_total / max(n_classes, 1)
    if n_pc < 12:
        return 60.0, 0.92, 0.6
    if n_pc < 40:
        return 44.0, 0.82, 0.5
    if n_pc < 100:
        return 30.0, 0.65, 0.4
    if n_pc < 250:
        return 20.0, 0.50, 0.35
    return 14.0, 0.40, 0.30


def _ticks_x_inteiros(ax, valores, limiar: int = 15, nbins: int = 10):
    """Evita ticks sobrepostos no eixo X. Se ha mais de `limiar` valores,
    usa MaxNLocator (passos inteiros, ~nbins divisoes). Caso contrario
    mostra todos os valores. Usado em graficos de selecao de LVs."""
    from matplotlib.ticker import MaxNLocator
    valores = np.asarray(valores)
    if len(valores) > limiar:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=nbins))
    else:
        ax.set_xticks(valores)


def plot_scores_panel(ax, scores, rotulos, mapa_cores, var_exp,
                      titulo, xlabel, ylabel, puros_mask=None,
                      mapa_marcadores=None, desenhar_elipses=True):
    """Plot scores SEM legenda interna. A legenda deve ser desenhada
    externamente via _legenda_lateral usando os handles do scatter.

    Canais de distincao: COR (classe) + FORMA (mapa_marcadores, opcional)
    + borda inteligente (clara/escura por luminancia). Se puros_mask for
    dado, puros recebem borda preta (destaque) mantendo a forma da classe."""
    rotulos = np.asarray(rotulos, dtype=str)
    scores  = np.asarray(scores,  dtype=float)

    classes_unicas = np.unique(rotulos)
    s, alpha, lw_e = parametros_scatter_adaptativos(len(rotulos),
                                                      len(classes_unicas))

    xrange = float(scores[:, 0].max() - scores[:, 0].min())
    yrange = float(scores[:, 1].max() - scores[:, 1].min())
    limite = (xrange, yrange)

    pmask = (np.asarray(puros_mask, dtype=bool)
             if puros_mask is not None else None)

    for cls in classes_unicas:
        idx = rotulos == cls
        c = mapa_cores[cls]
        mk = mapa_marcadores.get(cls, "o") if mapa_marcadores else "o"
        edge = edge_para_cor(c)
        n_cls = int(idx.sum())
        if pmask is not None:
            idx_puro  = idx & pmask
            idx_adult = idx & (~pmask)
            if idx_adult.any():
                ax.scatter(scores[idx_adult, 0], scores[idx_adult, 1],
                           color=c, s=s, marker=mk, edgecolors=edge,
                           linewidths=lw_e, zorder=3, alpha=alpha,
                           label=f"{cls} (n={n_cls})")
            # Puros: mesma forma/cor, borda preta espessa = destaque
            if idx_puro.any():
                ax.scatter(scores[idx_puro, 0], scores[idx_puro, 1],
                           color=c, s=s * 1.15, marker=mk,
                           edgecolors="black", linewidths=1.2, zorder=5,
                           label=(f"{cls} (n={n_cls})"
                                  if not idx_adult.any() else None))
        else:
            ax.scatter(scores[idx, 0], scores[idx, 1],
                       color=c, s=s, marker=mk, edgecolors=edge,
                       linewidths=lw_e, zorder=3, alpha=alpha,
                       label=f"{cls} (n={n_cls})")
        if desenhar_elipses:
            ok = elipse_t2(ax, scores[idx, 0], scores[idx, 1], c,
                            lw=1.4, limite_dispersao=limite)
            if not ok:
                convex_hull_contorno(ax, scores[idx, 0], scores[idx, 1],
                                       c, lw=1.3)

    ax.axhline(0, color="0.82", lw=0.5, ls="-", zorder=1)
    ax.axvline(0, color="0.82", lw=0.5, ls="-", zorder=1)
    ax.set_xlabel(f"{xlabel} ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"{ylabel} ({var_exp[1]:.1f}%)")
    ax.set_title(titulo, loc="left")

    xmin, xmax = float(scores[:, 0].min()), float(scores[:, 0].max())
    ymin, ymax = float(scores[:, 1].min()), float(scores[:, 1].max())
    px = max((xmax - xmin) * 0.08, 1e-3)
    py = max((ymax - ymin) * 0.08, 1e-3)
    ax.set_xlim(xmin - px, xmax + px)
    ax.set_ylim(ymin - py, ymax + py)
    ax.grid(color="0.94", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def _legenda_lateral(ax_leg, ax_dados, titulo: str = "Classe",
                      max_col_alta: int = 18):
    """Desenha legenda externa num axes dedicado, sem ocupar area de dados."""
    ax_leg.axis("off")
    handles, labels = ax_dados.get_legend_handles_labels()
    ncol = 1 if len(labels) <= max_col_alta else 2
    ax_leg.legend(handles, labels, loc="center left", frameon=False,
                   title=titulo, title_fontsize=9.5, fontsize=8.5,
                   ncol=ncol, borderaxespad=0, handletextpad=0.6,
                   labelspacing=0.6)


# =========================================================================
#  FIGURAS
# =========================================================================

def fig1_selecao_lvs(erros_rmsecv, metricas_por_lv, n_opt, cfg, pasta):
    """Selecao de LVs + metricas de CV por LV."""
    n_max = len(erros_rmsecv)
    lvs = np.arange(1, n_max + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2),
                              constrained_layout=True)

    ax = axes[0]
    ax.plot(lvs, erros_rmsecv, "o-", color=cor(0), ms=5.5, lw=1.6)
    ax.axvline(n_opt, color=cor(3), ls="--", lw=1.3,
               label=f"Otimo: {n_opt} LVs")
    ax.set_xlabel("Numero de variaveis latentes")
    ax.set_ylabel("RMSECV")
    ax.set_title("(a) Selecao de LVs", loc="left")
    _ticks_x_inteiros(ax, lvs)
    ax.grid(axis="y", color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(lvs, [m["accuracy"]          for m in metricas_por_lv],
            "o-", color=cor(0), ms=4.5, lw=1.3, label="Accuracy")
    ax.plot(lvs, [m["balanced_accuracy"] for m in metricas_por_lv],
            "s-", color=cor(2), ms=4.5, lw=1.3, label="Balanced acc.")
    ax.plot(lvs, [m["f1_macro"]          for m in metricas_por_lv],
            "^-", color=cor(1), ms=4.5, lw=1.3, label="F1 (macro)")
    ax.plot(lvs, [m["cohen_kappa"]       for m in metricas_por_lv],
            "d-", color=cor(3), ms=4.5, lw=1.3, label="Cohen's $\\kappa$")
    ax.axvline(n_opt, color="0.55", ls=":", lw=1)
    ax.set_xlabel("Numero de variaveis latentes")
    ax.set_ylabel("Metrica (CV)")
    ax.set_ylim(-0.05, 1.05)
    _ticks_x_inteiros(ax, lvs)
    ax.set_title("(b) Metricas de validacao cruzada", loc="left")
    ax.grid(axis="y", color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower right", ncol=2, frameon=False)

    salvar(fig, "figS1_selecao_lvs", pasta, cfg)


def _centroides_pca(X, rotulos, n_pcs):
    """Centroide por classe no espaco de PCA(n_pcs). Reduz ruido espectral
    antes do clustering. Retorna (M_centroides, classes)."""
    rotulos = np.asarray(rotulos, dtype=str)
    classes = np.unique(rotulos)
    n_comp = int(min(n_pcs, X.shape[1], X.shape[0]))
    scores = PCA(n_components=n_comp).fit_transform(X)
    M = np.vstack([scores[rotulos == c].mean(axis=0) for c in classes])
    return M, classes


def fig_hca_dendrograma(X_processed, rotulos, mapa_cores, cfg, pasta,
                         metodo="ward"):
    """HCA dendrograma (Ward, euclidiana) sobre CENTROIDES por especie no
    espaco de PCA(hca_n_pcs componentes) — N1 obrigatorio.

    Eixos (V3): orientacao TOP — especies no eixo X (inferior, rotulos
    rotacionados e coloridos por especie); distancia no eixo Y (esquerda).

    HCA complementa PCA: revela a hierarquia de similaridade espectral
    entre especies (potenciais confusoes no PLS-DA).
    """
    rotulos = np.asarray(rotulos, dtype=str)
    M, classes = _centroides_pca(X_processed, rotulos, cfg.hca_n_pcs)
    if len(M) < 2:
        print("[AVISO] HCA: <2 grupos, dendrograma pulado.")
        return

    Z = linkage(pdist(M, metric="euclidean"), method=metodo)

    largura = max(8.0, 0.55 * len(classes) + 2.0)
    fig, ax = plt.subplots(figsize=(largura, 5.2), constrained_layout=True)
    dendrogram(
        Z, labels=list(classes), orientation="top", ax=ax,
        color_threshold=0, above_threshold_color="0.55",
        leaf_font_size=10,
    )
    # Rotulos das especies no eixo X (inferior), coloridos e rotacionados
    mapa_lbl_cor = {str(c): mapa_cores.get(c, "0.4") for c in classes}
    for lbl in ax.get_xmajorticklabels():
        lbl.set_color(mapa_lbl_cor.get(lbl.get_text(), "0.2"))
        lbl.set_fontweight("bold")
        lbl.set_rotation(40)
        lbl.set_horizontalalignment("right")
    ax.set_ylabel(f"Distancia ({metodo}, euclidiana)")
    ax.set_xlabel("Especie", labelpad=6)
    ax.set_title(f"HCA — dendrograma (centroides, PCA {M.shape[1]} comp.)",
                  loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    salvar(fig, "fig_hca_dendrograma", pasta, cfg)

    # Interpretacao automatica dos agrupamentos (2 clusters principais)
    try:
        from scipy.cluster.hierarchy import fcluster
        grupos = fcluster(Z, t=2, criterion="maxclust")
        comp = {}
        for g in np.unique(grupos):
            comp[int(g)] = [str(classes[i]) for i in range(len(classes))
                            if grupos[i] == g]
        print("  [HCA] Agrupamentos principais (k=2):")
        for g, membros in comp.items():
            print(f"    Cluster {g}: {', '.join(membros)}")
    except Exception:
        pass


def fig_hca_comparacao_pipelines(X_raw, rotulos, mapa_cores, cfg, pasta,
                                  metodo="ward"):
    """Compara o HCA (centroides em PCA hca_n_pcs) sob varios pre-
    processamentos espectrais, num painel de dendrogramas. Avalia
    estabilidade dos agrupamentos e impacto do pre-processamento."""
    rotulos = np.asarray(rotulos, dtype=str)

    def _snv(X):
        mu = X.mean(axis=1, keepdims=True); sd = X.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0; return (X - mu) / sd

    def _msc(X):
        ref = X.mean(axis=0); A = np.column_stack([np.ones_like(ref), ref])
        out = np.zeros_like(X)
        for i in range(X.shape[0]):
            sol, *_ = np.linalg.lstsq(A, X[i], rcond=None)
            a, b = float(sol[0]), float(sol[1])
            out[i] = (X[i] - a) / b if abs(b) > 1e-12 else X[i] - a
        return out

    def _sg(X, d):
        return savgol_filter(X, cfg.sg_window, cfg.sg_polyorder, deriv=d, axis=1)

    presets = {
        "Bruto":         lambda X: X,
        "SNV":           lambda X: _snv(X),
        "MSC":           lambda X: _msc(X),
        "SG 1a deriv":   lambda X: _sg(X, 1),
        "SG 2a deriv":   lambda X: _sg(X, 2),
        "SNV+SG1":       lambda X: _sg(_snv(X), 1),
        "MSC+SG1":       lambda X: _sg(_msc(X), 1),
        "Normalizacao":  lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12),
    }

    n = len(presets); ncols = 4; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows),
                              constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    mapa_lbl_cor = {str(c): mapa_cores.get(c, "0.4")
                    for c in np.unique(rotulos)}
    for k, (nome, fn) in enumerate(presets.items()):
        ax = axes[k]
        try:
            Xp = np.asarray(fn(X_raw), dtype=float)
            M, classes = _centroides_pca(Xp, rotulos, cfg.hca_n_pcs)
            Z = linkage(pdist(M, metric="euclidean"), method=metodo)
            dendrogram(Z, labels=list(classes), orientation="top", ax=ax,
                       color_threshold=0, above_threshold_color="0.6",
                       leaf_font_size=6.5)
            for lbl in ax.get_xmajorticklabels():
                lbl.set_color(mapa_lbl_cor.get(lbl.get_text(), "0.2"))
                lbl.set_rotation(80); lbl.set_horizontalalignment("right")
                lbl.set_fontsize(6.5)
            ax.set_title(nome, loc="left", fontsize=9, fontweight="bold")
            ax.set_yticks([])
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        except Exception as e:
            ax.text(0.5, 0.5, f"falhou\n{e}", ha="center", va="center",
                    fontsize=7, transform=ax.transAxes); ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("HCA por pre-processamento — estabilidade dos agrupamentos",
                  fontsize=11, fontweight="bold", x=0.01, ha="left")
    salvar(fig, "fig_hca_comparacao_pipelines", pasta, cfg)


def fig1_pca_scores(scores_pca, var_pca, rotulos, mapa_cores, cfg, pasta,
                     puros_mask=None, mapa_marcadores=None):
    """Figura 1: scores de PCA. Legenda externa (fora da area dos dados).
    Cor + forma por classe; puros com borda preta (se puros_mask)."""
    fig = plt.figure(figsize=(8.5, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[5.0, 1.1])
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    plot_scores_panel(ax, scores_pca[:, :2], rotulos, mapa_cores,
                       var_pca[:2],
                       titulo="PCA — exploratorio (nao supervisionado)",
                       xlabel="PC1", ylabel="PC2", puros_mask=puros_mask,
                       mapa_marcadores=mapa_marcadores,
                       desenhar_elipses=cfg.mostrar_elipses_grupo)
    _legenda_lateral(ax_leg, ax)
    salvar(fig, "fig1_pca_scores", pasta, cfg)


def fig2_plsda_scores(T_pls, var_lv_pls, rotulos, mapa_cores, cfg, pasta,
                       puros_mask=None, mapa_marcadores=None):
    """Figura 2: PLS-DA. Mostra LV1xLV2 e LV2xLV3 (se >=3 LVs).
    Cor + forma por classe; puros com borda preta (se puros_mask)."""
    n_lv_avail = T_pls.shape[1]
    if n_lv_avail < 2:
        print("[AVISO] PLS-DA com <2 LVs: scores plot pulado.")
        return

    if n_lv_avail >= 3:
        fig = plt.figure(figsize=(13.5, 5.6), constrained_layout=True)
        gs = fig.add_gridspec(1, 3, width_ratios=[5.0, 5.0, 1.1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax_leg = fig.add_subplot(gs[2])
        plot_scores_panel(ax1, T_pls[:, [0, 1]], rotulos, mapa_cores,
                           [var_lv_pls[0], var_lv_pls[1]],
                           titulo="(a) PLS-DA — LV1 × LV2",
                           xlabel="LV1", ylabel="LV2", puros_mask=puros_mask,
                           mapa_marcadores=mapa_marcadores,
                           desenhar_elipses=cfg.mostrar_elipses_grupo)
        plot_scores_panel(ax2, T_pls[:, [1, 2]], rotulos, mapa_cores,
                           [var_lv_pls[1], var_lv_pls[2]],
                           titulo="(b) PLS-DA — LV2 × LV3",
                           xlabel="LV2", ylabel="LV3", puros_mask=puros_mask,
                           mapa_marcadores=mapa_marcadores,
                           desenhar_elipses=cfg.mostrar_elipses_grupo)
        _legenda_lateral(ax_leg, ax1)
    else:
        fig = plt.figure(figsize=(8.5, 5.8), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[5.0, 1.1])
        ax = fig.add_subplot(gs[0])
        ax_leg = fig.add_subplot(gs[1])
        plot_scores_panel(ax, T_pls[:, :2], rotulos, mapa_cores,
                           var_lv_pls[:2],
                           titulo="PLS-DA — LV1 × LV2",
                           xlabel="LV1", ylabel="LV2", puros_mask=puros_mask,
                           mapa_marcadores=mapa_marcadores,
                           desenhar_elipses=cfg.mostrar_elipses_grupo)
        _legenda_lateral(ax_leg, ax)
    salvar(fig, "fig2_plsda_scores", pasta, cfg)


def fig3_outliers(T_scores, P_loadings, X_processed, rotulos, mapa_cores,
                   n_lv, cfg, pasta):
    n = X_processed.shape[0]
    T2 = hotelling_t2(T_scores[:, :n_lv])
    Q  = q_residuos(X_processed, T_scores[:, :n_lv], P_loadings[:n_lv])
    t2_lim = hotelling_t2_limite(n, n_lv)
    q_lim  = q_residuos_limite(Q)

    rotulos = np.asarray(rotulos, dtype=str)
    classes_unicas = np.unique(rotulos)
    s_pt, alpha_pt, lw_pt = parametros_scatter_adaptativos(
        len(rotulos), len(classes_unicas))

    fig = plt.figure(figsize=(13.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[5.0, 5.0, 1.1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax_leg = fig.add_subplot(gs[2])

    # Escala log em Y: poucas amostras com T2 ~1000 esmagam o resto na
    # base. Log distribui e mantem o limite 95% visivel.
    for cls in classes_unicas:
        idx = rotulos == cls
        ax1.scatter(np.where(idx)[0], np.clip(T2[idx], 1e-2, None),
                    color=mapa_cores[cls],
                    s=s_pt, edgecolors="white", linewidths=lw_pt,
                    label=str(cls), zorder=3, alpha=alpha_pt)
    ax1.axhline(t2_lim, color="0.30", ls="--", lw=1.0,
                label=f"limite 95% ({t2_lim:.1f})")
    ax1.set_yscale("log")
    ax1.set_xlabel("Indice da amostra")
    ax1.set_ylabel("Hotelling T$^2$ (log)")
    ax1.set_title(f"(a) Hotelling T$^2$ ({n_lv} LVs) — limite 95%",
                   loc="left")
    ax1.grid(axis="y", color="0.94", lw=0.5); ax1.set_axisbelow(True)

    # Painel (b): log-log centraliza a nuvem (estava no canto inferior-esq).
    for cls in classes_unicas:
        idx = rotulos == cls
        ax2.scatter(np.clip(T2[idx], 1e-2, None), np.clip(Q[idx], 1e-12, None),
                    color=mapa_cores[cls],
                    s=s_pt, edgecolors="white", linewidths=lw_pt,
                    zorder=3, alpha=alpha_pt)
    ax2.axvline(t2_lim, color="0.30", ls="--", lw=1.0)
    ax2.axhline(q_lim,  color="0.30", ls="--", lw=1.0)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Hotelling T$^2$ (log)")
    ax2.set_ylabel("Q-residuals / SPE (log)")
    ax2.set_title("(b) T$^2$ vs Q — deteccao de outliers", loc="left")
    ax2.grid(color="0.94", lw=0.5); ax2.set_axisbelow(True)

    _legenda_lateral(ax_leg, ax1)

    salvar(fig, "fig3_outliers_T2_Q", pasta, cfg)

    outliers_t2 = np.where(T2 > t2_lim)[0]
    outliers_q  = np.where(Q  > q_lim)[0]
    return T2, Q, t2_lim, q_lim, outliers_t2, outliers_q


def fig4_confusao(cm_mat, classes, y_true, y_pred, cfg, pasta):
    """Matriz de confusao + metricas por classe (precision, sensitivity,
    specificity, F1). Tamanho escala com numero de classes."""
    n_cls = len(classes)
    largura = max(10.5, 5.5 + 0.55 * n_cls)
    altura  = max(4.8, 4.2 + 0.10 * n_cls)
    fig = plt.figure(figsize=(largura, altura), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15])

    # ---- (a) Matriz -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    cm_norm = cm_mat.astype(float) / np.maximum(
        cm_mat.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    thresh = 0.55

    fs_anot = 10 if n_cls <= 6 else (8.5 if n_cls <= 10 else 7)
    for i in range(n_cls):
        for j in range(n_cls):
            ax.text(j, i, f"{cm_mat[i, j]}\n{cm_norm[i, j]*100:.1f}%",
                    ha="center", va="center",
                    fontsize=fs_anot, fontweight="bold",
                    color="white" if cm_norm[i, j] > thresh else "0.15")

    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    fs_tick = 9 if n_cls <= 8 else 8
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=fs_tick)
    ax.set_yticklabels(classes, fontsize=fs_tick)
    ax.set_xlabel("Predito", labelpad=8)
    ax.set_ylabel("Real",    labelpad=8)
    ax.set_title("(a) Matriz de confusao (CV)", loc="left")

    ax.set_xticks(np.arange(-0.5, n_cls, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_cls, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proporcao por classe real", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    # ---- (b) Metricas por classe ---------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    prec = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    rec  = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    spec = especificidade_por_classe(cm_mat)
    f1   = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

    nomes_m   = ["Precision", "Sensitivity", "Specificity", "F1-score"]
    valores_m = [prec, rec, spec, f1]
    cores_m   = [cor(0), cor(2), cor(1), cor(3)]

    x = np.arange(n_cls)
    width = 0.20
    for k, (nome, val, cm_) in enumerate(zip(nomes_m, valores_m, cores_m)):
        ax.bar(x + (k - 1.5) * width, val, width, label=nome,
                color=cm_, edgecolor="white", lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=fs_tick)
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1.10)
    ax.set_title("(b) Metricas por classe (one-vs-rest)", loc="left")
    ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", ncol=2, fontsize=8, frameon=False)

    salvar(fig, "fig4_confusao_e_metricas_por_classe", pasta, cfg)


def fig4b_metricas_globais(metricas, perm_obs, perm_dist, perm_p, cfg, pasta):
    """Metricas globais consolidadas + resultado do teste de permutacao."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)

    nomes  = ["Accuracy", "Balanced accuracy", "F1 (macro)",
              "Precision (macro)", "Recall (macro)", "Cohen's $\\kappa$"]
    chaves = ["accuracy", "balanced_accuracy", "f1_macro",
              "precision_macro", "recall_macro", "cohen_kappa"]
    valores = [metricas[k] for k in chaves]
    cores_b = [cor(0), cor(2), cor(1), cor(4), cor(5), cor(3)]
    pos = np.arange(len(nomes))

    bars = ax.barh(pos, valores, color=cores_b, edgecolor="white",
                    lw=0.6, height=0.72)
    for b, v in zip(bars, valores):
        ax.text(min(v + 0.015, 1.0), b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=9)

    ax.set_yticks(pos); ax.set_yticklabels(nomes, fontsize=9.5)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Valor")
    ax.set_title("Metricas globais — validacao cruzada", loc="left")
    ax.invert_yaxis()
    ax.grid(axis="x", color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.05,
            f"Permutation test (Y-randomization)\n"
            f"Acc obs. = {perm_obs:.3f}   p = {perm_p:.4f}   N = {len(perm_dist)}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="0.25",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.82", lw=0.6))

    salvar(fig, "fig4b_metricas_globais", pasta, cfg)


# Atribuicao quimica de bandas FT-NIR para oleos vegetais (M3).
# Referencias: Workman & Weyer (2012) Practical Guide to Interpretive
# Near-IR Spectroscopy; Cen & He (2007) Trends Food Sci Technol 18:72.
BANDAS_NIR: List[Tuple[float, str]] = [
    (4255, "C-H comb.\n(ac. graxos)"),
    (4325, "O-H comb."),
    (4665, "C-H/C=C"),
    (5180, "O-H (agua)"),
    (5800, "C=O 1o overt."),
    (5900, "C-H 1o overt."),
    (7050, "O-H 1o overt."),
    (8500, "C-H 2o overt."),
]


def _anotar_bandas_vip(ax, wavenumbers, vip, limiar=2.0,
                        janela=120.0):
    """M3: anota bandas quimicas conhecidas nos picos de VIP > limiar.

    Para cada banda de referencia, se existe um ponto com VIP > limiar
    dentro de +-janela cm-1, desenha uma anotacao apontando para o pico
    local. Evita poluir: so anota bandas efetivamente relevantes no modelo.
    """
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    vip = np.asarray(vip, dtype=float)
    ymax = float(np.nanmax(vip)) if vip.size else 1.0
    for centro, rotulo in BANDAS_NIR:
        viz = np.abs(wavenumbers - centro) <= janela
        if not viz.any():
            continue
        idx_viz = np.where(viz)[0]
        i_pico = idx_viz[int(np.argmax(vip[idx_viz]))]
        if vip[i_pico] < limiar:
            continue
        ax.annotate(
            rotulo,
            xy=(wavenumbers[i_pico], vip[i_pico]),
            xytext=(wavenumbers[i_pico], min(vip[i_pico] + 0.18 * ymax,
                                              ymax * 1.12)),
            ha="center", va="bottom", fontsize=6.8, color="0.20",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7),
            zorder=6,
        )


def fig5_vip(vip, wavenumbers, top_n, cfg, pasta):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3),
                              constrained_layout=True,
                              gridspec_kw={"width_ratios": [1.65, 1]})

    ax = axes[0]
    mask_hi = vip >= 1.0
    ax.plot(wavenumbers, vip, color="0.45", lw=1.0, alpha=0.85, zorder=2)
    ax.scatter(wavenumbers[mask_hi], vip[mask_hi],
               color=cor(1), s=12, zorder=3, label="VIP $\\geq$ 1")
    ax.axhline(1.0, color=cor(3), ls="--", lw=1.1, label="VIP = 1")
    _anotar_bandas_vip(ax, wavenumbers, vip, limiar=2.0)
    ax.set_xlabel("Numero de onda (cm$^{-1}$)")
    ax.set_ylabel("VIP score")
    ax.set_title("(a) VIP scores ao longo do espectro", loc="left")
    ax.invert_xaxis()
    # V3: y-lim no range REAL (sem comprimir a variacao perto de 1.0) +
    # caixa de estatisticas para verificar dispersao verdadeira dos VIPs.
    vmin, vmax = float(np.min(vip)), float(np.max(vip))
    ax.set_ylim(max(0.0, vmin - 0.05 * (vmax - vmin)), vmax * 1.08)
    ax.text(0.02, 0.97,
            f"min={vmin:.2f}  max={vmax:.2f}\n"
            f"media={float(np.mean(vip)):.2f}  dp={float(np.std(vip)):.2f}\n"
            f"n(VIP$\\geq$1)={int(mask_hi.sum())}/{len(vip)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
            color="0.25", bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                     ec="0.82", lw=0.6))
    ax.grid(axis="y", color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    ord_vip = np.argsort(vip)[::-1]
    top_n = min(top_n, len(wavenumbers))
    idx_top = ord_vip[:top_n][::-1]
    valores = vip[idx_top]
    cores_b = [cor(1) if v >= 1.0 else "0.7" for v in valores]
    pos = np.arange(top_n)
    ax.barh(pos, valores, color=cores_b, edgecolor="white", lw=0.5, height=0.78)
    ax.axvline(1.0, color=cor(3), ls="--", lw=1.0)
    ax.set_yticks(pos)
    ax.set_yticklabels([f"{wavenumbers[i]:.0f}" for i in idx_top], fontsize=8)
    ax.set_xlabel("VIP score")
    ax.set_ylabel("Numero de onda (cm$^{-1}$)")
    ax.set_title(f"(b) Top {top_n} variaveis", loc="left")
    ax.grid(axis="x", color="0.93", lw=0.5); ax.set_axisbelow(True)

    salvar(fig, "fig5_vip", pasta, cfg)


def fig6_preprocessamento(wavenumbers, X_raw, X_processed, rotulos,
                           mapa_cores, cfg, pasta):
    rotulos = np.asarray(rotulos, dtype=str)
    fig = plt.figure(figsize=(11.5, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[6.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_leg = fig.add_subplot(gs[:, 1])

    for cls in np.unique(rotulos):
        idx = rotulos == cls
        m_raw = X_raw[idx].mean(axis=0)
        s_raw = X_raw[idx].std(axis=0)
        m_pp  = X_processed[idx].mean(axis=0)
        s_pp  = X_processed[idx].std(axis=0)
        c = mapa_cores[cls]
        ax_a.plot(wavenumbers, m_raw, color=c, lw=1.2, label=str(cls))
        ax_a.fill_between(wavenumbers, m_raw - s_raw, m_raw + s_raw,
                           color=c, alpha=0.15, lw=0)
        ax_b.plot(wavenumbers, m_pp, color=c, lw=1.2)
        ax_b.fill_between(wavenumbers, m_pp - s_pp, m_pp + s_pp,
                           color=c, alpha=0.15, lw=0)

    ax_a.set_ylabel("Absorbancia")
    ax_a.set_title("(a) Espectros brutos (media $\\pm$ DP por classe)",
                    loc="left")
    ax_a.grid(axis="y", color="0.94", lw=0.5); ax_a.set_axisbelow(True)

    descricao = []
    if cfg.aplicar_snv: descricao.append("SNV")
    if cfg.aplicar_sg:
        descricao.append(f"SG(w={cfg.sg_window},p={cfg.sg_polyorder},d={cfg.sg_deriv})")
    if cfg.aplicar_mc:  descricao.append("mean-centering")
    ax_b.set_ylabel("Sinal pre-processado")
    ax_b.set_xlabel("Numero de onda (cm$^{-1}$)")
    ax_b.set_title(f"(b) Apos {' → '.join(descricao)}", loc="left")
    ax_b.invert_xaxis()
    ax_b.axhline(0, color="0.75", lw=0.5, ls=":")
    ax_b.grid(axis="y", color="0.94", lw=0.5); ax_b.set_axisbelow(True)

    _legenda_lateral(ax_leg, ax_a)

    salvar(fig, "fig6_preprocessamento", pasta, cfg)


def fig_extra_wold(wold: Dict[str, object], cfg, pasta):
    """Wold-style permutation plot: R2Y e Q2Y vs similaridade do Y permutado."""
    sims = np.asarray(cast(Any, wold["sims"]))
    r2s  = np.asarray(cast(Any, wold["r2s"]))
    q2s  = np.asarray(cast(Any, wold["q2s"]))
    r2_obs   = cast(float, wold["r2_obs"])
    q2_obs   = cast(float, wold["q2_obs"])
    int_r2   = cast(float, wold["intercept_r2"])
    int_q2   = cast(float, wold["intercept_q2"])
    slope_r2 = cast(float, wold["slope_r2"])
    slope_q2 = cast(float, wold["slope_q2"])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                              constrained_layout=True)

    x_line = np.linspace(0, 1, 50)

    ax = axes[0]
    ax.scatter(sims, r2s, color=cor(0), s=28, alpha=0.55,
                edgecolors="white", linewidths=0.4, label="Permutacoes")
    ax.scatter([1.0], [r2_obs], color=cor(3), s=90, marker="D",
                edgecolors="black", linewidths=0.8, zorder=5,
                label="Observado")
    if np.isfinite(slope_r2):
        ax.plot(x_line, slope_r2 * x_line + int_r2, color="0.35", lw=1.2,
                 ls="--", label=f"Reta (intercepto = {int_r2:.3f})")
    ax.axhline(0.40, color=cor(1), lw=0.9, ls=":",
                label="Limiar R2Y = 0.40")
    cor_status = cor(2) if int_r2 < 0.40 else cor(3)
    status = "VALIDO" if int_r2 < 0.40 else "FALHA"
    ax.text(0.02, 0.97, f"R2Y intercept: {status}",
             transform=ax.transAxes, ha="left", va="top",
             fontsize=9, fontweight="bold", color=cor_status,
             bbox=dict(boxstyle="round,pad=0.35", fc="white",
                        ec=cor_status, lw=0.8))
    ax.set_xlabel("Similaridade (Y permutado, Y original)")
    ax.set_ylabel("R$^2$Y (ajuste training)")
    ax.set_xlim(-0.05, 1.08); ax.set_ylim(-0.5, 1.08)
    ax.set_title("(a) Wold — R$^2$Y vs permutacao", loc="left")
    ax.grid(color="0.94", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    ax = axes[1]
    ax.scatter(sims, q2s, color=cor(2), s=28, alpha=0.55,
                edgecolors="white", linewidths=0.4, label="Permutacoes")
    ax.scatter([1.0], [q2_obs], color=cor(3), s=90, marker="D",
                edgecolors="black", linewidths=0.8, zorder=5,
                label="Observado")
    if np.isfinite(slope_q2):
        ax.plot(x_line, slope_q2 * x_line + int_q2, color="0.35", lw=1.2,
                 ls="--", label=f"Reta (intercepto = {int_q2:.3f})")
    ax.axhline(0.05, color=cor(1), lw=0.9, ls=":",
                label="Limiar Q2Y = 0.05")
    cor_status = cor(2) if int_q2 < 0.05 else cor(3)
    status = "VALIDO" if int_q2 < 0.05 else "FALHA"
    ax.text(0.02, 0.97, f"Q2Y intercept: {status}",
             transform=ax.transAxes, ha="left", va="top",
             fontsize=9, fontweight="bold", color=cor_status,
             bbox=dict(boxstyle="round,pad=0.35", fc="white",
                        ec=cor_status, lw=0.8))
    ax.set_xlabel("Similaridade (Y permutado, Y original)")
    ax.set_ylabel("Q$^2$Y (CV)")
    ax.set_xlim(-0.05, 1.08); ax.set_ylim(-0.6, 1.08)
    ax.set_title("(b) Wold — Q$^2$Y vs permutacao", loc="left")
    ax.grid(color="0.94", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    salvar(fig, "fig_extra_wold_permutacao", pasta, cfg)


def fig_extra_holdout(metricas_cv: Dict[str, float],
                       metricas_holdout: Dict[str, float],
                       cm_holdout: np.ndarray, classes: np.ndarray,
                       n_holdout: int, cfg, pasta):
    """Confusao no holdout + comparacao CV vs holdout (deteccao de overfit)."""
    n_cls = len(classes)
    fig = plt.figure(figsize=(13.0, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1])

    # (a) Matriz de confusao no holdout
    ax = fig.add_subplot(gs[0])
    cm_norm = cm_holdout.astype(float) / np.maximum(
        cm_holdout.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    fs_an = 10 if n_cls <= 6 else (8.5 if n_cls <= 10 else 7)
    for i in range(n_cls):
        for j in range(n_cls):
            ax.text(j, i, f"{cm_holdout[i, j]}\n{cm_norm[i, j]*100:.1f}%",
                    ha="center", va="center", fontsize=fs_an,
                    fontweight="bold",
                    color="white" if cm_norm[i, j] > 0.55 else "0.15")
    ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predito", labelpad=8); ax.set_ylabel("Real", labelpad=8)
    ax.set_title(f"(a) Matriz de confusao — holdout ({n_holdout} amostras)",
                  loc="left")
    ax.set_xticks(np.arange(-0.5, n_cls, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_cls, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proporcao por classe real", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    # (b) CV vs Holdout
    ax = fig.add_subplot(gs[1])
    nomes  = ["Accuracy", "Balanced acc.", "F1 (macro)", "Cohen's $\\kappa$"]
    chaves = ["accuracy", "balanced_accuracy", "f1_macro", "cohen_kappa"]
    cv_vals = [metricas_cv[k]      for k in chaves]
    ho_vals = [metricas_holdout[k] for k in chaves]
    x = np.arange(len(nomes)); w = 0.36
    ax.bar(x - w/2, cv_vals, w, color=cor(0), label="CV (training)",
            edgecolor="white", lw=0.5)
    ax.bar(x + w/2, ho_vals, w, color=cor(1), label="Holdout (test)",
            edgecolor="white", lw=0.5)
    for k, (cv_v, ho_v) in enumerate(zip(cv_vals, ho_vals)):
        ax.text(k - w/2, cv_v + 0.015, f"{cv_v:.3f}", ha="center",
                va="bottom", fontsize=8)
        ax.text(k + w/2, ho_v + 0.015, f"{ho_v:.3f}", ha="center",
                va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(nomes, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valor")
    ax.set_title("(b) Validacao cruzada vs holdout (overfitting check)",
                  loc="left")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(axis="y", color="0.93", lw=0.5); ax.set_axisbelow(True)

    salvar(fig, "fig_extra_holdout", pasta, cfg)


def fig_extra_comparacao_pipelines(resultados, cfg, pasta):
    """Comparacao entre pipelines de pre-processamento (Q2, Acc, BalAcc)."""
    nomes = list(resultados.keys())
    accs = [resultados[n]["accuracy"]     for n in nomes]
    bals = [resultados[n]["balanced_acc"] for n in nomes]
    q2s  = [max(resultados[n]["q2"], -0.2) for n in nomes]
    n_lvs = [resultados[n]["n_lv"]        for n in nomes]

    fig, ax = plt.subplots(figsize=(11.0, 0.55 * len(nomes) + 2.2),
                            constrained_layout=True)
    pos = np.arange(len(nomes))
    h = 0.26
    ax.barh(pos - h, accs, h, color=cor(0), label="Accuracy",
             edgecolor="white", lw=0.5)
    ax.barh(pos,     bals, h, color=cor(2), label="Balanced acc.",
             edgecolor="white", lw=0.5)
    ax.barh(pos + h, q2s,  h, color=cor(1), label="Q$^2$",
             edgecolor="white", lw=0.5)

    for k, n_lv in enumerate(n_lvs):
        ax.text(1.005, k, f" {n_lv} LVs", va="center", ha="left",
                 fontsize=8.5, color="0.30",
                 transform=ax.get_yaxis_transform())

    ax.set_yticks(pos)
    ax.set_yticklabels(nomes, fontsize=9.5)
    ax.set_xlim(-0.05, 1.12)
    ax.set_xlabel("Valor (validacao cruzada)")
    ax.set_title("Comparacao de pipelines de pre-processamento",
                  loc="left")
    ax.axvline(0, color="0.7", lw=0.5)
    ax.invert_yaxis()
    ax.grid(axis="x", color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, ncol=3)

    salvar(fig, "fig_extra_comparacao_pipelines", pasta, cfg)


def fig5b_vip_estabilidade(boot: Dict[str, object], wavenumbers,
                            top_n, cfg, pasta):
    """Bootstrap estratificado do VIP: media, CI95 e frequencia de selecao."""
    vip_mean = np.asarray(cast(Any, boot["mean"]))
    ci_lo    = np.asarray(cast(Any, boot["ci95_low"]))
    ci_hi    = np.asarray(cast(Any, boot["ci95_high"]))
    sel_freq = np.asarray(cast(Any, boot["selection_frequency"]))

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3),
                              constrained_layout=True,
                              gridspec_kw={"width_ratios": [1.7, 1.0, 1.0]})

    ax = axes[0]
    ax.fill_between(wavenumbers, ci_lo, ci_hi,
                     color=cor(0), alpha=0.22, lw=0, zorder=2,
                     label="IC 95% (bootstrap)")
    ax.plot(wavenumbers, vip_mean, color="0.25", lw=1.0, alpha=0.95, zorder=3)
    ax.axhline(1.0, color=cor(3), ls="--", lw=1.0, label="VIP = 1")
    _anotar_bandas_vip(ax, wavenumbers, vip_mean, limiar=2.0)
    ax.set_xlabel("Numero de onda (cm$^{-1}$)")
    ax.set_ylabel("VIP (media bootstrap estratificado)")
    ax.set_title("(a) VIP medio com IC 95%", loc="left")
    ax.invert_xaxis()
    ax.grid(axis="y", color="0.94", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    ax = axes[1]
    ord_vip = np.argsort(vip_mean)[::-1]
    top_n = min(top_n, len(wavenumbers))
    idx_top = ord_vip[:top_n][::-1]
    valores = vip_mean[idx_top]
    erro_inf = valores - ci_lo[idx_top]
    erro_sup = ci_hi[idx_top] - valores
    pos = np.arange(top_n)
    cores_b = [cor(1) if v >= 1.0 else "0.7" for v in valores]
    ax.barh(pos, valores, color=cores_b, edgecolor="white", lw=0.5,
             height=0.78,
             xerr=np.vstack([erro_inf, erro_sup]),
             error_kw=dict(ecolor="0.35", lw=0.8, capsize=2))
    ax.axvline(1.0, color=cor(3), ls="--", lw=1.0)
    ax.set_yticks(pos)
    ax.set_yticklabels([f"{wavenumbers[i]:.0f}" for i in idx_top], fontsize=8)
    ax.set_xlabel("VIP score (com IC 95%)")
    ax.set_ylabel("Numero de onda (cm$^{-1}$)")
    ax.set_title(f"(b) Top {top_n} variaveis", loc="left")
    ax.grid(axis="x", color="0.94", lw=0.5); ax.set_axisbelow(True)

    ax = axes[2]
    freq_top = sel_freq[idx_top]
    ax.barh(pos, freq_top, color=cor(2), edgecolor="white", lw=0.5,
             height=0.78)
    ax.axvline(0.5, color="0.55", ls=":", lw=0.9)
    ax.set_yticks(pos)
    ax.set_yticklabels([f"{wavenumbers[i]:.0f}" for i in idx_top], fontsize=8)
    ax.set_xlabel("Frequencia de selecao (VIP $\\geq$ 1)")
    ax.set_xlim(0, 1.05)
    ax.set_title("(c) Estabilidade da selecao", loc="left")
    ax.grid(axis="x", color="0.94", lw=0.5); ax.set_axisbelow(True)

    salvar(fig, "fig5b_vip_bootstrap", pasta, cfg)


def fig7_pls_regressao(Yc, Yc_hat, Yv, Yv_hat, erros_reg, n_opt_reg,
                        r2c, r2v, rmsec, rmsecv, rmsep, bias_v, cfg, pasta):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2),
                              constrained_layout=True)

    n_max = len(erros_reg)
    lvs = np.arange(1, n_max + 1)
    ax = axes[0]
    ax.plot(lvs, erros_reg, "o-", color=cor(0), ms=5, lw=1.4)
    ax.axvline(n_opt_reg, color=cor(1), ls="--", lw=1.2,
               label=f"Otimo: {n_opt_reg} LVs")
    ax.set_xlabel("Numero de variaveis latentes")
    ax.set_ylabel("RMSECV")
    _ticks_x_inteiros(ax, lvs)
    ax.set_title("(a) Selecao de LVs", loc="left")
    ax.legend()

    Yc_f = np.asarray(Yc).flatten(); Yc_h = np.asarray(Yc_hat).flatten()
    Yv_f = np.asarray(Yv).flatten(); Yv_h = np.asarray(Yv_hat).flatten()
    todos = np.concatenate([Yc_f, Yc_h, Yv_f, Yv_h])
    lim = [todos.min() - 1, todos.max() + 1]

    ax = axes[1]
    ax.scatter(Yc_f, Yc_h, color=cor(0), s=36, edgecolors="white",
               linewidths=0.5, label="Calibracao", zorder=3, alpha=0.9)
    ax.scatter(Yv_f, Yv_h, color=cor(1), s=44, marker="^",
               edgecolors="white", linewidths=0.5,
               label="Validacao", zorder=3, alpha=0.9)
    ax.plot(lim, lim, "k--", lw=0.8, label="y = x")
    ax.set_xlabel("Valor de referencia")
    ax.set_ylabel("Valor predito")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(
        f"(b) Predito vs Real\n"
        f"$R^2_{{cal}}$={r2c:.3f}  $R^2_{{val}}$={r2v:.3f}",
        loc="left")
    ax.legend(loc="best")

    res = Yv_f - Yv_h
    ax = axes[2]
    ax.scatter(Yv_h, res, color=cor(2), s=44, edgecolors="white",
               linewidths=0.5, zorder=3, alpha=0.9)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axhline( rmsep, color=cor(3), lw=0.8, ls=":",
                label=f"$\\pm$RMSEP ({rmsep:.2f})")
    ax.axhline(-rmsep, color=cor(3), lw=0.8, ls=":")
    ax.set_xlabel("Valor predito")
    ax.set_ylabel("Residuo")
    ax.set_title(f"(c) Residuos — Validacao\nBias = {bias_v:.3f}", loc="left")
    ax.legend(loc="best")

    salvar(fig, "figS2_pls_regressao", pasta, cfg)


# =========================================================================
#  Figuras Sprint 3
# =========================================================================

def fig_sprint3_sr_vip(vip: np.ndarray, sr: np.ndarray,
                        wavenumbers: np.ndarray, top_n: int,
                        cfg: Config, pasta: str) -> None:
    """VIP (todos LVs) x SR (1o componente) lado a lado.

    Concordancia entre VIP >= 1 e SR alto reforca que a variavel e
    relevante tanto global (VIP) quanto no componente discriminante
    principal (SR). Discordancias revelam variaveis importantes apenas
    em LVs secundarios.

    Referencia: Rajalahti et al. (2009), Chemom. Intell. Lab. Syst.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4),
                              constrained_layout=True)

    # (a) Espectro: VIP + SR em eixos duplos
    ax = axes[0]
    ax2 = ax.twinx()
    c_vip, c_sr = cor(0), cor(1)

    l1, = ax.plot(wavenumbers, vip, color=c_vip, lw=1.1,
                   alpha=0.9, label="VIP")
    ax.axhline(1.0, color=c_vip, ls="--", lw=0.8, alpha=0.5)
    mask_vip = vip >= 1.0
    ax.scatter(wavenumbers[mask_vip], vip[mask_vip],
               color=c_vip, s=9, zorder=3, alpha=0.8)

    l2, = ax2.plot(wavenumbers, sr, color=c_sr, lw=1.1,
                    alpha=0.8, label="SR")
    sr_thr = float(np.percentile(sr, 75))   # quartil 75 como referencia
    ax2.axhline(sr_thr, color=c_sr, ls="--", lw=0.8, alpha=0.5,
                label=f"SR Q75 = {sr_thr:.2f}")

    ax.set_xlabel("Numero de onda (cm$^{-1}$)")
    ax.set_ylabel("VIP score", color=c_vip)
    ax2.set_ylabel("Selectivity Ratio", color=c_sr)
    ax.tick_params(axis="y", labelcolor=c_vip)
    ax2.tick_params(axis="y", labelcolor=c_sr)
    ax.invert_xaxis()
    ax.set_title("(a) VIP e SR ao longo do espectro", loc="left")
    ax.grid(axis="y", color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.legend(handles=[l1, l2], loc="upper left", frameon=False,
               fontsize=8.5)

    # (b) Top N por SR (barras duplas VIP + SR normalizados)
    ax = axes[1]
    top_n = min(top_n, len(wavenumbers))
    idx_sr = np.argsort(sr)[::-1][:top_n][::-1]
    pos     = np.arange(top_n)
    sr_max  = max(float(sr.max()),  1e-12)
    vip_max = max(float(vip.max()), 1e-12)
    sr_top  = sr[idx_sr]  / sr_max
    vip_top = vip[idx_sr] / vip_max
    w_b = 0.38
    ax.barh(pos - w_b / 2, sr_top,  w_b, color=c_sr,
             edgecolor="white", lw=0.5, label="SR (norm.)")
    ax.barh(pos + w_b / 2, vip_top, w_b, color=c_vip,
             edgecolor="white", lw=0.5, label="VIP (norm.)")
    ax.axvline(1.0 / vip_max, color=c_vip, ls=":", lw=0.8, alpha=0.6)
    ax.set_yticks(pos)
    ax.set_yticklabels([f"{wavenumbers[i]:.0f}" for i in idx_sr],
                        fontsize=8)
    ax.set_xlabel("Valor normalizado (max global = 1)")
    ax.set_ylabel("Numero de onda (cm$^{-1}$)")
    ax.set_title(f"(b) Top {top_n} por SR vs VIP", loc="left")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.grid(axis="x", color="0.93", lw=0.5); ax.set_axisbelow(True)

    salvar(fig, "fig_sprint3_sr_vip", pasta, cfg)


def fig_sprint3_score_contribution(pls_model: PLSRegression,
                                    X_processed: np.ndarray,
                                    rotulos: np.ndarray,
                                    wavenumbers: np.ndarray,
                                    mapa_cores: Dict[str, str],
                                    top_n: int,
                                    cfg: Config, pasta: str) -> None:
    """Contribuicao por variavel ao score LV1 por classe (B8).

    Contribuicao da variavel j para o score da amostra i no LV1:
        c_ij = x_ij * w*_j   (peso PLS normalizado)

    Media por classe revela quais regioes espectrais 'empurram' cada
    classe para seu lado no espaco de scores — informacao complementar
    ao VIP (que e global) e ao SR (que e escalar).

    Paineis:
        (a) Espectro de contribuicao medio ± SE por classe
        (b) Top N variaveis com maior poder discriminante entre classes
    """
    rotulos = np.asarray(rotulos, dtype=str)
    classes = np.unique(rotulos)
    W = np.asarray(pls_model.x_weights_, dtype=float)   # (p, n_lv)
    if W.shape[1] < 1:
        return

    w1 = W[:, 0] / (float(np.linalg.norm(W[:, 0])) + 1e-12)
    contrib = X_processed * w1[np.newaxis, :]   # (n, p)

    means: Dict[str, np.ndarray] = {}
    sems:  Dict[str, np.ndarray] = {}
    for cls in classes:
        idx = rotulos == cls
        c = contrib[idx]
        means[cls] = c.mean(axis=0)
        sems[cls]  = c.std(axis=0, ddof=1) / max(float(np.sqrt(idx.sum())), 1.0)

    n_cls = len(classes)

    # ===== FIGURA 1: espectro de contribuicao media (legenda lateral) =====
    fig = plt.figure(figsize=(10.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[6.0, 1.1])
    ax = fig.add_subplot(gs[0]); ax_leg = fig.add_subplot(gs[1])
    for cls in classes:
        c = mapa_cores.get(cls, "0.5")
        m = means[cls]; s = sems[cls]
        ax.plot(wavenumbers, m, color=c, lw=1.2, label=str(cls))
        ax.fill_between(wavenumbers, m - s, m + s, color=c, alpha=0.18, lw=0)
    ax.axhline(0, color="0.70", lw=0.7, ls=":")
    ax.invert_xaxis()
    ax.set_xlabel("Numero de onda (cm$^{-1}$)")
    ax.set_ylabel("$c_j = x_j \\cdot w^*_j$  (LV1)")
    ax.set_title("Contribuicao espectral media por classe — LV1", loc="left")
    ax.grid(axis="y", color="0.94", lw=0.5); ax.set_axisbelow(True)
    _legenda_lateral(ax_leg, ax)
    salvar(fig, "fig_score_contribution_espectro", pasta, cfg)

    # ===== FIGURA 2: Top N poder discriminante (separada, alta, legivel) ===
    mean_stack = np.vstack([means[c] for c in classes])   # (n_cls, p)
    disc_power = mean_stack.var(axis=0)                    # (p,)
    top_idx = np.argsort(disc_power)[::-1][:top_n][::-1]
    pos = np.arange(top_n)
    w_bar = 0.84 / max(n_cls, 1)

    altura = max(6.0, 0.42 * top_n + 1.6)   # escala com top_n -> legivel
    fig2 = plt.figure(figsize=(11.0, altura), constrained_layout=True)
    gs2 = fig2.add_gridspec(1, 2, width_ratios=[6.0, 1.2])
    ax = fig2.add_subplot(gs2[0]); ax_leg2 = fig2.add_subplot(gs2[1])
    for k, cls in enumerate(classes):
        offset = (k - (n_cls - 1) / 2.0) * w_bar
        ax.barh(pos + offset, means[cls][top_idx], w_bar,
                color=mapa_cores.get(cls, "0.5"),
                edgecolor="white", lw=0.4, label=str(cls),
                xerr=sems[cls][top_idx],
                error_kw=dict(ecolor="0.45", lw=0.6, capsize=1.2))
    ax.axvline(0, color="0.40", lw=0.8, ls="--")
    ax.set_yticks(pos)
    ax.set_yticklabels([f"{wavenumbers[i]:.0f}" for i in top_idx], fontsize=8.5)
    ax.set_ylim(-0.5, top_n - 0.5)
    ax.set_xlabel("Contribuicao media $\\pm$ SE por classe")
    ax.set_ylabel("Numero de onda (cm$^{-1}$)")
    ax.set_title(f"Top {top_n} variaveis — poder discriminante entre classes",
                  loc="left")
    ax.grid(axis="x", color="0.93", lw=0.5); ax.set_axisbelow(True)
    _legenda_lateral(ax_leg2, ax)
    salvar(fig2, "fig_score_contribution_top_discriminante", pasta, cfg)


def fig_sprint3_ddsimca_acceptance(scores: Dict[str, Dict[str, Any]],
                                    rotulos: np.ndarray,
                                    mapa_cores: Dict[str, str],
                                    cfg: Config, pasta: str,
                                    sens_esp: Optional[Dict[str, Any]] = None
                                    ) -> None:
    """DD-SIMCA acceptance plot: T2/UCL vs Q/UCL por modelo de classe.

    Cada painel = 1 modelo one-class. Quadrado unitario = regiao de
    aceitacao. Pontos coloridos pela classe real.

    Se sens_esp fornecido {classe: (sens, esp, n_puro, n_adult)}, o titulo
    mostra 'sens.=XX% | esp.=YY%' (M2): sens = puros aceitos,
    esp = adulterados rejeitados.
    """
    classes = [c for c in scores.keys()]
    n_cls = len(classes)
    if n_cls == 0:
        return

    ncols = min(3, n_cls)
    nrows = int(np.ceil(n_cls / ncols))
    has_leg = n_cls > 1
    fig_w = 4.4 * ncols + (1.2 if has_leg else 0)
    fig_h = max(4.2 * nrows, 4.2)

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    total_cols = ncols + (1 if has_leg else 0)
    gs = fig.add_gridspec(nrows, total_cols,
                           width_ratios=[4.2] * ncols + ([1.1] if has_leg else []))

    rotulos = np.asarray(rotulos, dtype=str)
    all_classes = np.unique(rotulos)
    s_pt, alpha_pt, lw_pt = parametros_scatter_adaptativos(
        len(rotulos), len(all_classes))

    ax_leg_ref = None
    for k, cls in enumerate(classes):
        row, col = divmod(k, ncols)
        ax = fig.add_subplot(gs[row, col])
        m = scores[cls]
        t2n = np.asarray(m["T2_norm"])
        qn  = np.asarray(m["Q_norm"])

        # Escala log-log (Pomerantsev): clamp em piso pequeno para evitar
        # log(0) e tornar visivel a regiao de aceitacao (canto inferior-esq).
        piso = 1e-2
        t2p = np.clip(t2n, piso, None)
        qp  = np.clip(qn,  piso, None)
        for true_cls in all_classes:
            idx = rotulos == true_cls
            ax.scatter(t2p[idx], qp[idx],
                       color=mapa_cores.get(true_cls, "0.5"),
                       s=s_pt, alpha=alpha_pt,
                       edgecolors="white", linewidths=lw_pt,
                       label=str(true_cls), zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")

        # Fronteira de aceitacao em (1,1): quadrante inferior-esquerdo aceito
        ax.axvline(1.0, color="0.20", ls="--", lw=1.2, zorder=4)
        ax.axhline(1.0, color="0.20", ls="--", lw=1.2, zorder=4)
        ax.axvspan(piso, 1.0, ymin=0, ymax=1, color=mapa_cores.get(cls, cor(0)),
                   alpha=0.0)  # placeholder p/ manter cor no titulo

        # Titulo: usa sens/esp do one-class se disponivel (M2); senao
        # fallback para fracao da propria classe aceita.
        if sens_esp is not None and cls in sens_esp:
            sens_c, esp_c = sens_esp[cls][0], sens_esp[cls][1]
            s_txt = f"{sens_c*100:.0f}%" if sens_c == sens_c else "n/a"
            e_txt = f"{esp_c*100:.0f}%"  if esp_c == esp_c else "n/a"
            titulo_painel = f"Modelo: {cls}  sens.={s_txt} | esp.={e_txt}"
        else:
            idx_cls   = rotulos == cls
            n_cls_tot = int(idx_cls.sum())
            n_aceitos = int(np.sum((t2n[idx_cls] <= 1.0) & (qn[idx_cls] <= 1.0)))
            titulo_painel = f"Modelo: {cls}  sens.={n_aceitos/max(n_cls_tot,1):.0%}"

        lim_hi = max(float(np.percentile(np.concatenate([t2p, qp]), 99)) * 1.5,
                     3.0)
        ax.set_xlim(piso * 0.8, lim_hi)
        ax.set_ylim(piso * 0.8, lim_hi)
        ax.set_xlabel(r"$T^2$ / UCL($T^2$)  (log)", fontsize=8.5)
        ax.set_ylabel("$Q$ / UCL($Q$)  (log)", fontsize=8.5)
        ax.set_title(titulo_painel, loc="left",
                      fontsize=8.5, fontweight="bold")
        ax.text(0.98, 0.98,
                f"UCL($T^2$)={float(m['T2_ucl']):.1f}\n"
                f"UCL($Q$)={float(m['Q_ucl']):.2g}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color="0.35",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                           ec="0.82", lw=0.5))
        ax.grid(color="0.94", lw=0.5); ax.set_axisbelow(True)
        if ax_leg_ref is None:
            ax_leg_ref = ax

    if has_leg and ax_leg_ref is not None:
        ax_leg = fig.add_subplot(gs[:, ncols])
        _legenda_lateral(ax_leg, ax_leg_ref)

    salvar(fig, "fig_sprint3_ddsimca_acceptance", pasta, cfg)


def fig_ddsimca_individuais(scores: Dict[str, Dict[str, Any]],
                             rotulos: np.ndarray,
                             mapa_cores: Dict[str, str],
                             cfg: Config, pasta: str,
                             sens_esp: Optional[Dict[str, Any]] = None
                             ) -> None:
    """Um acceptance plot DD-SIMCA por classe, salvo em pasta/ddsimca/.
    Escala log-log, legenda lateral, anotacao UCL — versao individual
    legivel (o grid 5x3 fica pequeno para inspecao detalhada)."""
    rotulos = np.asarray(rotulos, dtype=str)
    all_classes = np.unique(rotulos)
    s_pt, alpha_pt, lw_pt = parametros_scatter_adaptativos(
        len(rotulos), len(all_classes))
    piso = 1e-2
    for cls in scores.keys():
        m = scores[cls]
        t2p = np.clip(np.asarray(m["T2_norm"]), piso, None)
        qp  = np.clip(np.asarray(m["Q_norm"]),  piso, None)
        fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[5.0, 1.2])
        ax = fig.add_subplot(gs[0]); ax_leg = fig.add_subplot(gs[1])
        for true_cls in all_classes:
            idx = rotulos == true_cls
            ax.scatter(t2p[idx], qp[idx], color=mapa_cores.get(true_cls, "0.5"),
                       s=s_pt, alpha=alpha_pt, edgecolors="white",
                       linewidths=lw_pt, label=str(true_cls), zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.axvline(1.0, color="0.20", ls="--", lw=1.2, zorder=4)
        ax.axhline(1.0, color="0.20", ls="--", lw=1.2, zorder=4)
        lim_hi = max(float(np.percentile(np.concatenate([t2p, qp]), 99)) * 1.5,
                     3.0)
        ax.set_xlim(piso * 0.8, lim_hi); ax.set_ylim(piso * 0.8, lim_hi)
        if sens_esp is not None and cls in sens_esp:
            sc, ec = sens_esp[cls][0], sens_esp[cls][1]
            st = f"{sc*100:.0f}%" if sc == sc else "n/a"
            et = f"{ec*100:.0f}%" if ec == ec else "n/a"
            tt = f"DD-SIMCA: {cls}  sens.={st} | esp.={et}"
        else:
            tt = f"DD-SIMCA: {cls}"
        ax.set_xlabel(r"$T^2$ / UCL($T^2$)  (log)")
        ax.set_ylabel("$Q$ / UCL($Q$)  (log)")
        ax.set_title(tt, loc="left", fontsize=9.5, fontweight="bold")
        ax.text(0.98, 0.98, f"UCL($T^2$)={float(m['T2_ucl']):.1f}\n"
                f"UCL($Q$)={float(m['Q_ucl']):.2g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color="0.35", bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                         ec="0.82", lw=0.5))
        ax.grid(color="0.94", lw=0.5); ax.set_axisbelow(True)
        _legenda_lateral(ax_leg, ax)
        nome_seguro = str(cls).replace(" ", "_").replace("/", "-")
        salvar(fig, f"ddsimca_{nome_seguro}", pasta, cfg, subpasta="ddsimca")


def fig_sprint3_opls_scores(t_pred: np.ndarray, t_orth: np.ndarray,
                             rotulos: np.ndarray,
                             mapa_cores: Dict[str, str],
                             n_ortho: int,
                             cfg: Config, pasta: str) -> None:
    """OPLS-DA scores: tp (preditivo) vs to1 (1o ortogonal).

    Eixo tp: separacao real entre classes.
    Eixo to: variacao sistematica de X nao correlacionada com Y
    (baseline, scatter multiplicativo, espessura de caminho optico).

    O plot decompoe a variacao espectral total em 'util para discriminacao'
    (tp) vs 'interferencia estruturada' (to), auxiliando na interpretacao
    de quao pura e a separacao e quanta variacao pode ser atribuida a
    artefatos instrumentais.
    """
    rotulos = np.asarray(rotulos, dtype=str)
    t_orth1 = t_orth[:, 0] if t_orth.ndim == 2 and t_orth.shape[1] > 0 \
              else np.asarray(t_orth).flatten()

    var_p = float(np.var(t_pred,  ddof=1)) if len(t_pred)  > 1 else 1.0
    var_o = float(np.var(t_orth1, ddof=1)) if len(t_orth1) > 1 else 1.0
    total = var_p + var_o + 1e-12
    pct_p = var_p / total * 100
    pct_o = var_o / total * 100

    fig = plt.figure(figsize=(8.5, 5.8), constrained_layout=True)
    gs  = fig.add_gridspec(1, 2, width_ratios=[5.0, 1.1])
    ax     = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])

    scores2d = np.column_stack([t_pred, t_orth1])
    plot_scores_panel(
        ax, scores2d, rotulos, mapa_cores,
        var_exp=[pct_p, pct_o],
        titulo=f"OPLS-DA — preditivo × ortogonal ({n_ortho} comp. orth.)",
        xlabel="$t_p$ (preditivo)",
        ylabel="$t_o$ (ortogonal 1)",
        desenhar_elipses=cfg.mostrar_elipses_grupo,
    )
    _legenda_lateral(ax_leg, ax)
    salvar(fig, "fig_sprint3_opls_scores", pasta, cfg)


# =========================================================================
#  Sprint v24 — Figuras de Publicacao
#  (Loading Plot PCA, ROC/AUC, S-Plot OPLS-DA, Cooman's Plot DD-SIMCA)
# =========================================================================

def fig_loadings_pca(pca, wavenumbers: np.ndarray, cfg: "Config",
                      pasta: str, n_pcs: int = 2) -> None:
    """Loading Plot PCA: contribuicao espectral de cada variavel por componente.

    Barras em vermelho (positivo) e azul (negativo). Eixo X invertido por
    convencao NIR/FTIR (numero de onda decrescente da esq. para dir.).

    Ref: Bro & Smilde (2014) Anal. Methods 6:2812-2831.
    """
    n_comp = min(n_pcs, int(pca.n_components_))
    fig, axes = plt.subplots(n_comp, 1,
                              figsize=(12.0, 3.4 * n_comp),
                              constrained_layout=True)
    if n_comp == 1:
        axes = [axes]

    dx = float(abs(wavenumbers[1] - wavenumbers[0])) if len(wavenumbers) > 1 else 1.0

    for i, ax in enumerate(axes):
        loadings = pca.components_[i]
        var_exp  = float(pca.explained_variance_ratio_[i]) * 100
        cores_b  = [cor(0) if v >= 0 else cor(1) for v in loadings]
        ax.bar(wavenumbers, loadings, width=dx * 0.9,
               color=cores_b, alpha=0.80, edgecolor="none")
        ax.axhline(0, color="0.45", lw=0.7, ls="--")
        ax.set_xlabel("Número de onda (cm$^{-1}$)")
        ax.set_ylabel(f"Loading PC{i + 1}")
        ax.set_title(f"PC{i + 1} — {var_exp:.1f}% da variância explicada  "
                     "| vermelho = positivo · azul = negativo",
                     loc="left", fontsize=9)
        # Convencao NIR: wavenumbers crescentes no array -> inverte eixo X
        if len(wavenumbers) > 1 and wavenumbers[0] < wavenumbers[-1]:
            ax.invert_xaxis()

    fig.suptitle("Loading Plot PCA — contribuição espectral por componente",
                  fontsize=10, fontweight="bold")
    salvar(fig, "fig_loadings_pca", pasta, cfg)


def fig_roc_auc(Y_bin: np.ndarray, Y_cv: np.ndarray,
                classes: np.ndarray, cfg: "Config",
                pasta: str) -> Dict[str, float]:
    """Curvas ROC multiclasse One-vs-Rest usando predicoes CV group-aware.

    Os scores continuos do PLS-DA (Y_cv) sao usados como funcao discriminante
    — sem re-calibracao, sem leakage. AUC macro calculado por roc_auc_score.

    Retorna dict {classe: AUC, 'macro': AUC_macro}.

    Ref: Fawcett (2006) Pattern Recognit. Lett. 27:861-874.
         Hand & Till (2001) Machine Learning 45:171-186.
    """
    from sklearn.metrics import roc_curve, auc as sk_auc, roc_auc_score

    n_cls = Y_bin.shape[1]
    if n_cls < 2:
        return {}

    aucs: Dict[str, float] = {}
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)

    for i, cls in enumerate(classes):
        y_true_i  = Y_bin[:, i]
        y_score_i = Y_cv[:, i]
        if y_true_i.sum() == 0 or y_true_i.sum() == len(y_true_i):
            continue
        fpr, tpr, _ = roc_curve(y_true_i, y_score_i)
        auc_i        = float(sk_auc(fpr, tpr))
        aucs[str(cls)] = auc_i
        ax.plot(fpr, tpr, color=cor(i), lw=1.4,
                label=f"{cls}  (AUC = {auc_i:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Aleatório (AUC = 0.500)")
    ax.set_xlabel("Taxa de Falso Positivo  (1 – Especificidade)")
    ax.set_ylabel("Taxa de Verdadeiro Positivo  (Sensibilidade)")
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.02])

    try:
        macro_auc = float(roc_auc_score(
            Y_bin, Y_cv, average="macro", multi_class="ovr"))
        aucs["macro"] = macro_auc
        ax.set_title(
            f"Curvas ROC — PLS-DA (CV group-aware) | AUC macro = {macro_auc:.3f}",
            loc="left")
    except Exception:
        ax.set_title("Curvas ROC — PLS-DA (CV group-aware)", loc="left")

    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax.grid(color="0.94", lw=0.5); ax.set_axisbelow(True)
    salvar(fig, "fig_roc_auc_multiclasse", pasta, cfg)
    return aucs


def fig_splot_opls(X_proc: np.ndarray, t_pred: np.ndarray,
                   wavenumbers: np.ndarray, cfg: "Config",
                   pasta: str, top_n: int = 15) -> None:
    """S-Plot OPLS-DA: covariancia x correlacao de cada variavel com t_pred.

    Superior-direito: variaveis discriminantes positivas (classe 1 > classe 0).
    Inferior-esquerdo: discriminantes negativas.
    Centro (~0, ~0): ruido espectral sem poder discriminante.

    Anotacao automatica dos top_n wavenumbers por |correlacao|.

    Ref: Bylesjo M. et al. (2006) J. Chemometrics 20:341-351.
         Wiklund S. et al. (2008) Anal. Chem. 80:115-122.
    """
    n = X_proc.shape[0]
    X_c    = X_proc - X_proc.mean(axis=0)
    t_c    = t_pred - float(t_pred.mean())
    cov_xj = (X_c * t_c[:, None]).sum(axis=0) / max(n - 1, 1)

    std_x = X_proc.std(axis=0, ddof=1)
    std_t = float(t_pred.std(ddof=1))
    std_x[std_x < 1e-12] = 1.0
    if std_t < 1e-12:
        std_t = 1.0
    corr_xj = np.clip(cov_xj / (std_x * std_t), -1.0, 1.0)

    fig, ax = plt.subplots(figsize=(9.5, 6.5), constrained_layout=True)
    sc = ax.scatter(cov_xj, corr_xj,
                    c=corr_xj, cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                    s=10, alpha=0.75, edgecolors="none", zorder=3)
    plt.colorbar(sc, ax=ax, label="Correlação($X_j$, $t_p$)",
                 fraction=0.035, pad=0.02)

    ax.axhline(0, color="0.55", lw=0.8, ls="--")
    ax.axvline(0, color="0.55", lw=0.8, ls="--")

    # Anota top_n wavenumbers por |correlacao|
    idx_top = np.argsort(np.abs(corr_xj))[::-1][:top_n]
    for j in idx_top:
        ax.annotate(f"{wavenumbers[j]:.0f}",
                    xy=(cov_xj[j], corr_xj[j]),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=6.5, color="0.2")

    ax.set_xlabel("Covariância($X_j$, $t_p$)")
    ax.set_ylabel("Correlação($X_j$, $t_p$)")
    ax.set_title(
        "S-Plot OPLS-DA — quadrante sup-dir: discriminantes positivos; "
        "inf-esq: negativos; centro: ruído",
        loc="left", fontsize=9)
    ax.grid(color="0.95", lw=0.5); ax.set_axisbelow(True)
    salvar(fig, "fig_splot_opls", pasta, cfg)


def fig_cooman_ddsimca(ddsimca_res: Dict[str, Dict[str, Any]],
                        rotulos: np.ndarray,
                        mapa_cores: Dict[str, str],
                        cfg: "Config", pasta: str,
                        max_pares: int = 6) -> None:
    """Cooman's Plot: distancia normalizada ao modelo de classe A vs classe B.

    Regioes do plano (por par A x B):
      sqrt(dQ_A)<=1 e sqrt(dQ_B)<=1  -> Ambiguo (aceito por ambos)
      sqrt(dQ_A)<=1 e sqrt(dQ_B)>1   -> pertence a A
      sqrt(dQ_A)>1  e sqrt(dQ_B)<=1  -> pertence a B
      sqrt(dQ_A)>1  e sqrt(dQ_B)>1   -> Desconhecido

    Escala raiz-quadrada: preserva estrutura proxima ao UCL sem comprimir
    o eixo para amostras muito distantes (adulteracoes altas).

    Ref: Rodionova & Pomerantsev (2020) Chemom. Intell. Lab. Syst. 200:103958.
    """
    classes_dd = sorted(ddsimca_res.keys())
    pares = [(classes_dd[i], classes_dd[j])
             for i in range(len(classes_dd))
             for j in range(i + 1, len(classes_dd))]
    if not pares:
        return

    pares = pares[:max_pares]
    n_p   = len(pares)
    ncols = min(3, n_p)
    nrows = (n_p + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.2 * ncols, 4.5 * nrows),
                              constrained_layout=True)
    if n_p == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (clsA, clsB) in enumerate(pares):
        ax  = axes[idx // ncols, idx % ncols]
        qA  = np.sqrt(np.clip(np.asarray(ddsimca_res[clsA]["Q_norm"]), 0, None))
        qB  = np.sqrt(np.clip(np.asarray(ddsimca_res[clsB]["Q_norm"]), 0, None))

        for cls in sorted(set(rotulos)):
            mask = rotulos == cls
            ax.scatter(qA[mask], qB[mask],
                       color=mapa_cores.get(cls, "#999999"),
                       s=20, alpha=0.80, label=cls,
                       edgecolors="none", zorder=3)

        ax.axhline(1.0, color="black", lw=0.9, ls="--")
        ax.axvline(1.0, color="black", lw=0.9, ls="--")
        ax.text(0.03, 0.97, clsA, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color="0.35")
        ax.text(0.97, 0.03, clsB, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="0.35")
        ax.set_xlabel(f"$\\sqrt{{d_Q}}$ — modelo {clsA}")
        ax.set_ylabel(f"$\\sqrt{{d_Q}}$ — modelo {clsB}")
        ax.set_title(f"{clsA} × {clsB}", fontsize=9)
        ax.grid(color="0.94", lw=0.5); ax.set_axisbelow(True)

    for idx_extra in range(n_p, nrows * ncols):
        axes[idx_extra // ncols, idx_extra % ncols].set_visible(False)

    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_leg, loc="lower center",
                   ncol=min(len(handles), 5), frameon=False,
                   fontsize=8, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle("Cooman's Plot — DD-SIMCA (escala $\\sqrt{d_Q}$)",
                  fontsize=11, fontweight="bold")
    salvar(fig, "fig_cooman_ddsimca", pasta, cfg, subpasta="ddsimca")


# =========================================================================
#  ETAPA 4 — Selecao de variaveis (iPLS, VIP, SR, sPLS-DA)
# =========================================================================

def _avaliar_subset_cv(X_sel: np.ndarray, Y_bin: np.ndarray, y_int: np.ndarray,
                        cv_indices: list, n_lv: int) -> Dict[str, float]:
    """Avalia um subconjunto de variaveis por CV group-aware (mesmos folds
    do pipeline). Mean-centering re-ajustado por fold (sem leakage). PLS-DA.
    Retorna accuracy, balanced_accuracy, Q2 e n_vars."""
    n_lv_eff = int(max(1, min(n_lv, X_sel.shape[1], X_sel.shape[0] - 1)))

    def _fac():
        return Pipeline([
            ("mc",  StandardScaler(with_std=False)),
            ("pls", PLSRegression(n_components=n_lv_eff, scale=False)),
        ])

    y_cv = _cv_predict_manual(_fac, X_sel, Y_bin, cv_indices)
    ss = float(np.sum((Y_bin - Y_bin.mean(axis=0)) ** 2))
    q2 = 1.0 - float(np.sum((Y_bin - y_cv) ** 2)) / ss if ss > 0 else 0.0
    yhat = np.argmax(y_cv, axis=1)
    return {
        "accuracy":          float(accuracy_score(y_int, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y_int, yhat)),
        "q2":                float(q2),
        "n_vars":            int(X_sel.shape[1]),
        "n_lv":              n_lv_eff,
    }


def selecao_ipls(X_proc, Y_bin, y_int, wavenumbers, cv_indices, n_lv,
                 n_intervalos: int) -> Tuple[list, np.ndarray]:
    """interval-PLS: divide o espectro em n_intervalos contiguos, avalia
    PLS-DA em cada um. Retorna (lista de resultados por intervalo,
    mascara do melhor intervalo)."""
    p = X_proc.shape[1]
    bordas = np.linspace(0, p, n_intervalos + 1).astype(int)
    resultados = []
    for i in range(n_intervalos):
        a, b = bordas[i], bordas[i + 1]
        if b - a < 2:
            continue
        mask = np.zeros(p, dtype=bool); mask[a:b] = True
        m = _avaliar_subset_cv(X_proc[:, mask], Y_bin, y_int, cv_indices, n_lv)
        m["intervalo"] = i + 1
        m["wn_ini"] = float(wavenumbers[a])
        m["wn_fim"] = float(wavenumbers[b - 1])
        m["idx_a"] = int(a)
        m["idx_b"] = int(b)
        resultados.append(m)
    melhor = max(resultados, key=lambda r: r["balanced_accuracy"])
    a, b = int(melhor["idx_a"]), int(melhor["idx_b"])
    mask_melhor = np.zeros(p, dtype=bool); mask_melhor[a:b] = True
    return resultados, mask_melhor


def sparse_plsda_mask(X_proc, Y_bin, n_comp: int,
                      keep_por_comp: int) -> np.ndarray:
    """sPLS-DA (estilo Le Cao et al. 2008): NIPALS com soft-selection —
    mantem apenas as `keep_por_comp` variaveis de maior |peso| por
    componente. Retorna mascara da uniao das variaveis selecionadas."""
    X = np.asarray(X_proc, dtype=float)
    Y = np.asarray(Y_bin, dtype=float)
    Xr = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    p = X.shape[1]
    selecionadas: set = set()
    n_comp = int(max(1, min(n_comp, p)))
    for _ in range(n_comp):
        M = Xr.T @ Yc                      # (p, m)
        try:
            U, _S, _Vt = np.linalg.svd(M, full_matrices=False)
            w = U[:, 0]
        except np.linalg.LinAlgError:
            break
        idx = np.argsort(np.abs(w))[::-1][:keep_por_comp]
        w_sp = np.zeros_like(w); w_sp[idx] = w[idx]
        nw = float(np.linalg.norm(w_sp))
        if nw < 1e-12:
            break
        w_sp /= nw
        t = Xr @ w_sp
        tt = float(t @ t)
        if tt < 1e-12:
            break
        pld = Xr.T @ t / tt
        Xr = Xr - np.outer(t, pld)
        c = Yc.T @ t / tt
        Yc = Yc - np.outer(t, c)
        selecionadas.update(int(i) for i in idx)
    mask = np.zeros(p, dtype=bool)
    if selecionadas:
        mask[list(selecionadas)] = True
    return mask


def fig_etapa4_ipls(resultados, wavenumbers, baseline_bal, cfg, pasta):
    """Barras de balanced_acc por intervalo iPLS, com linha do modelo global."""
    intervalos = [r["intervalo"] for r in resultados]
    bals       = [r["balanced_accuracy"] for r in resultados]
    centros    = [(r["wn_ini"] + r["wn_fim"]) / 2 for r in resultados]
    melhor_i   = int(np.argmax(bals))

    fig, ax = plt.subplots(figsize=(11.0, 4.2), constrained_layout=True)
    cores_b = [cor(2) if k == melhor_i else "0.6" for k in range(len(bals))]
    ax.bar(intervalos, bals, color=cores_b, edgecolor="white", lw=0.5)
    ax.axhline(baseline_bal, color=cor(3), ls="--", lw=1.2,
               label=f"Modelo global ({baseline_bal:.3f})")
    ax.set_xlabel("Intervalo iPLS")
    ax.set_ylabel("Balanced accuracy (CV)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Etapa 4 — iPLS: desempenho por intervalo espectral",
                  loc="left")
    # eixo secundario com faixa de wavenumber do melhor
    r = resultados[melhor_i]
    ax.annotate(f"melhor: {r['wn_ini']:.0f}-{r['wn_fim']:.0f} cm$^{{-1}}$\n"
                f"bal.acc={r['balanced_accuracy']:.3f} ({r['n_vars']} vars)",
                xy=(r["intervalo"], r["balanced_accuracy"]),
                xytext=(0.98, 0.05), textcoords="axes fraction",
                ha="right", va="bottom", fontsize=8.5, color="0.2",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.82"))
    _ticks_x_inteiros(ax, np.array(intervalos))
    ax.grid(axis="y", color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)
    salvar(fig, "fig_etapa4_ipls_intervalos", pasta, cfg)


def fig_etapa4_comparacao(tabela, cfg, pasta):
    """Compara metodos de selecao: balanced_acc (barras) + n_vars (texto)."""
    nomes = [t["metodo"] for t in tabela]
    bals  = [t["balanced_accuracy"] for t in tabela]
    nvars = [t["n_vars"] for t in tabela]
    pos = np.arange(len(nomes))

    fig, ax = plt.subplots(figsize=(10.0, 0.6 * len(nomes) + 2.0),
                            constrained_layout=True)
    cores_b = [cor(0) if n == "Full (todas)" else cor(2) for n in nomes]
    ax.barh(pos, bals, color=cores_b, edgecolor="white", lw=0.5, height=0.7)
    for k, (b, nv) in enumerate(zip(bals, nvars)):
        ax.text(min(b + 0.01, 1.0), k, f" {b:.3f} | {nv} vars",
                va="center", ha="left", fontsize=9)
    ax.set_yticks(pos); ax.set_yticklabels(nomes, fontsize=9.5)
    ax.set_xlim(0, 1.12); ax.invert_yaxis()
    ax.set_xlabel("Balanced accuracy (CV group-aware)")
    ax.set_title("Etapa 4 — comparacao de metodos de selecao de variaveis",
                  loc="left")
    ax.grid(axis="x", color="0.93", lw=0.5); ax.set_axisbelow(True)
    salvar(fig, "fig_etapa4_comparacao_metodos", pasta, cfg)


def etapa4_selecao_variaveis(X_proc, Y_bin, y_int, vip, sr, wavenumbers,
                              cv_indices, n_lv, cfg, pasta, pasta_dados):
    """Orquestra a Etapa 4: avalia Full vs iPLS vs VIP vs SR vs sPLS-DA
    sob o MESMO esquema de CV group-aware. Salva tabela (dados/) e figuras
    (figuras/). Retorna dict-resumo para o resumo_modelo.txt."""
    print("\n[Etapa4] Selecao de variaveis (iPLS, VIP, SR, sPLS-DA)...")
    p = X_proc.shape[1]
    tabela = []

    # 0) Baseline: todas as variaveis
    full = _avaliar_subset_cv(X_proc, Y_bin, y_int, cv_indices, n_lv)
    tabela.append({"metodo": "Full (todas)", **full})
    base_bal = full["balanced_accuracy"]
    print(f"  Full: bal.acc={base_bal:.3f} ({p} vars)")

    # 1) iPLS
    ipls_res, mask_ipls = selecao_ipls(X_proc, Y_bin, y_int, wavenumbers,
                                        cv_indices, n_lv, cfg.ipls_n_intervalos)
    m_ipls = _avaliar_subset_cv(X_proc[:, mask_ipls], Y_bin, y_int,
                                 cv_indices, n_lv)
    tabela.append({"metodo": "iPLS (melhor intervalo)", **m_ipls})
    fig_etapa4_ipls(ipls_res, wavenumbers, base_bal, cfg, pasta)
    pd.DataFrame(ipls_res).to_csv(
        os.path.join(pasta_dados, "etapa4_ipls_intervalos.csv"),
        sep=";", decimal=",", index=False)
    print(f"  iPLS: bal.acc={m_ipls['balanced_accuracy']:.3f} "
          f"({m_ipls['n_vars']} vars)")

    # 2) Selecao por VIP >= limiar
    mask_vip = np.asarray(vip) >= cfg.vip_threshold_sel
    if mask_vip.sum() >= 2:
        m_vip = _avaliar_subset_cv(X_proc[:, mask_vip], Y_bin, y_int,
                                    cv_indices, n_lv)
        tabela.append({"metodo": f"VIP>={cfg.vip_threshold_sel:g}", **m_vip})
        print(f"  VIP: bal.acc={m_vip['balanced_accuracy']:.3f} "
              f"({m_vip['n_vars']} vars)")

    # 3) Selecao por SR (top fracao)
    n_top = max(2, int(round(cfg.sr_top_frac * p)))
    idx_sr = np.argsort(np.asarray(sr))[::-1][:n_top]
    mask_sr = np.zeros(p, dtype=bool); mask_sr[idx_sr] = True
    m_sr = _avaliar_subset_cv(X_proc[:, mask_sr], Y_bin, y_int,
                               cv_indices, n_lv)
    tabela.append({"metodo": f"SR top {cfg.sr_top_frac:.0%}", **m_sr})
    print(f"  SR: bal.acc={m_sr['balanced_accuracy']:.3f} "
          f"({m_sr['n_vars']} vars)")

    # 4) sPLS-DA
    mask_sp = sparse_plsda_mask(X_proc, Y_bin, n_lv, cfg.splsda_keep_por_comp)
    if mask_sp.sum() >= 2:
        m_sp = _avaliar_subset_cv(X_proc[:, mask_sp], Y_bin, y_int,
                                   cv_indices, n_lv)
        tabela.append({"metodo": "sPLS-DA", **m_sp})
        print(f"  sPLS-DA: bal.acc={m_sp['balanced_accuracy']:.3f} "
              f"({m_sp['n_vars']} vars)")

    # Tabela + figura comparativa
    pd.DataFrame(tabela).to_csv(
        os.path.join(pasta_dados, "etapa4_selecao_variaveis.csv"),
        sep=";", decimal=",", index=False)
    fig_etapa4_comparacao(tabela, cfg, pasta)

    # Resumo: melhor metodo por parcimonia (bal.acc dentro de 1% do max, menos vars)
    bal_max = max(t["balanced_accuracy"] for t in tabela)
    candidatos = [t for t in tabela
                  if t["balanced_accuracy"] >= bal_max - 0.01]
    melhor = min(candidatos, key=lambda t: t["n_vars"])
    print(f"  -> Mais parcimonioso (bal.acc>={bal_max-0.01:.3f}): "
          f"{melhor['metodo']} ({melhor['n_vars']} vars, "
          f"bal.acc={melhor['balanced_accuracy']:.3f})")
    return {"tabela": tabela, "melhor": melhor}


# =========================================================================
#  Auto-Benchmark — v27: PLS-DA vs SVM vs RF vs XGBoost
# =========================================================================

class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper sklearn para PLS-DA.
    Permite uso transparente em cross_validate e Pipeline.sklearn.
    Normaliza scores por softmax para predict_proba.
    """
    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.preprocessing import LabelBinarizer as _LB
        self._lb = _LB()
        Y_bin = self._lb.fit_transform(y)
        # Binario: LabelBinarizer retorna (n,1) — expandir para (n,2)
        if Y_bin.ndim == 1 or Y_bin.shape[1] == 1:
            Y_bin = np.hstack([1 - Y_bin.reshape(-1, 1),
                                   Y_bin.reshape(-1, 1)])
        n_comp = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self._pls = PLSRegression(n_components=n_comp, scale=False)
        self._pls.fit(X, Y_bin)
        self.classes_ = self._lb.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._lb.inverse_transform(self._pls.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Y = np.clip(np.asarray(self._pls.predict(X), float), 0.0, 1.0)
        s = Y.sum(axis=1, keepdims=True); s[s < 1e-12] = 1.0
        return Y / s


def fig_benchmark_classificadores(scores_por_clf: Dict[str, np.ndarray],
                                   n_splits: int,
                                   n_classes: int,
                                   cfg: "Config", pasta: str) -> None:
    """Boxplot + swarmplot de balanced accuracy por fold para cada classificador."""
    nomes  = list(scores_por_clf.keys())
    dados  = [scores_por_clf[n] for n in nomes]
    chance = 1.0 / max(n_classes, 1)
    cores  = [cor(i) for i in range(len(nomes))]

    fig, ax = plt.subplots(figsize=(max(7.0, len(nomes) * 1.7), 4.8),
                           constrained_layout=True)

    bp = ax.boxplot(dados, patch_artist=True, notch=False, widths=0.45,
                    medianprops=dict(color="black", lw=2.0),
                    whiskerprops=dict(lw=1.2, color="0.3"),
                    capprops=dict(lw=1.2, color="0.3"),
                    flierprops=dict(marker="x", ms=5, color="0.5", lw=0.8))
    for patch, c in zip(bp["boxes"], cores):
        patch.set_facecolor(c); patch.set_alpha(0.70)

    # Pontos individuais (um por fold) com jitter reprodutivel
    rng = np.random.default_rng(42)
    for i, (nome, dado) in enumerate(zip(nomes, dados), 1):
        jitter = rng.uniform(-0.14, 0.14, len(dado))
        ax.scatter(np.full(len(dado), i) + jitter, dado,
                   color=cores[i - 1], s=38, alpha=0.85, zorder=5,
                   edgecolors="white", linewidths=0.5)

    ax.axhline(chance, color="0.35", ls="--", lw=1.2,
               label=f"Chance = 1/{n_classes} = {chance:.2f}")
    ax.axhline(1.0, color="0.88", ls=":", lw=0.7)

    ax.set_xticks(range(1, len(nomes) + 1))
    ax.set_xticklabels(nomes, fontsize=9)
    ax.set_ylabel("Balanced Accuracy (CV fold)")
    ax.set_title(
        f"Auto-Benchmark — {n_splits}-fold GroupKFold (anti-leakage de replicas)\n"
        f"Preprocessamento: {cfg.preprocessamento_padrao}",
        fontsize=8.5, loc="left")
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="0.94", lw=0.5); ax.set_axisbelow(True)

    salvar(fig, "fig_benchmark_classificadores", pasta, cfg)
    plt.close(fig)


def benchmark_classificadores(X_raw: np.ndarray, y_int: np.ndarray,
                               grupos_cv: Optional[np.ndarray],
                               lb: "LabelBinarizer",
                               n_opt: int, cfg: "Config", pasta: str,
                               wavenumbers: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Compara PLS-DA (n_opt LVs) vs SVM RBF vs Random Forest vs XGBoost
    usando a mesma CV group-aware e o mesmo pre-processamento do pipeline.

    Anti-leakage garantido: cada classificador e encapsulado num sklearn
    Pipeline(preproc + clf), de forma que o MSC/SNV/SG e recalibrado
    dentro de cada fold sem vazamento de informacao do conjunto de validacao.

    Ref: Westerhuis et al. (2008) Chemom. Intell. Lab. Syst. 92:58-64.
    """
    import warnings
    from sklearn.svm import SVC
    from sklearn.ensemble import (RandomForestClassifier,
                                  GradientBoostingClassifier)
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline as _SKPipeline
    from scipy.stats import wilcoxon

    n_splits = min(cfg.n_splits_cv,
                   int(pd.Series(y_int).value_counts().min()))
    n_splits  = max(n_splits, 2)
    n_classes = len(lb.classes_)
    preproc   = construir_preprocessador(cfg)

    # ── Classifiers ──────────────────────────────────────────────────────
    clfs: List[Tuple[str, Any]] = [
        ("PLS-DA",
         PLSDAClassifier(n_components=n_opt)),
        ("SVM RBF",
         SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
             random_state=cfg.seed, class_weight="balanced")),
        ("Random Forest",
         RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                class_weight="balanced_subsample",
                                n_jobs=1, random_state=cfg.seed)),
        ("Grad. Boost.",
         GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                    max_depth=3, subsample=0.8,
                                    random_state=cfg.seed)),
    ]
    try:
        from xgboost import XGBClassifier  # type: ignore
        clfs.append(("XGBoost",
                      XGBClassifier(n_estimators=300, learning_rate=0.05,
                                    max_depth=4, subsample=0.8,
                                    colsample_bytree=0.8,
                                    eval_metric="mlogloss", verbosity=0,
                                    n_jobs=1, random_state=cfg.seed)))
    except ImportError:
        pass

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                               random_state=cfg.seed)
    # Pre-computar os splits (reutilizado em todos os classificadores)
    splits_list = list(cv.split(X_raw, y_int, grupos_cv))

    resultados: List[Dict] = []
    scores_por_clf: Dict[str, np.ndarray] = {}
    # OOF probabilities coletadas no mesmo loop — evita cross_val_predict extra
    oof_probas: Dict[str, np.ndarray] = {}

    for nome, clf in clfs:
        pipe_base = _SKPipeline([("preproc", clone(preproc)), ("clf", clf)])
        print(f"  {nome:<18s} ... ", end="", flush=True)
        t0 = time.time()
        try:
            ba_folds: List[float] = []
            f1_folds: List[float] = []
            proba_oof = np.zeros((len(y_int), n_classes))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for tr_idx, te_idx in splits_list:
                    pipe_i = clone(pipe_base)
                    pipe_i.fit(X_raw[tr_idx], y_int[tr_idx])
                    y_pred = pipe_i.predict(X_raw[te_idx])
                    ba_folds.append(
                        balanced_accuracy_score(y_int[te_idx], y_pred))
                    f1_folds.append(
                        f1_score(y_int[te_idx], y_pred,
                                 average="macro", zero_division=0))
                    # Coleta OOF proba em um unico passe (sem cross_val_predict extra)
                    if hasattr(pipe_i[-1], "predict_proba"):
                        try:
                            p = pipe_i.predict_proba(X_raw[te_idx])
                            nc = min(p.shape[1], n_classes)
                            proba_oof[te_idx, :nc] = p[:, :nc]
                        except Exception:
                            pass
            elapsed = time.time() - t0
            ba = np.array(ba_folds)
            f1 = np.array(f1_folds)
            scores_por_clf[nome] = ba
            oof_probas[nome]     = proba_oof
            resultados.append({
                "Classificador":      nome,
                "Bal.Acc media":      round(float(ba.mean()), 4),
                "Bal.Acc std":        round(float(ba.std()),  4),
                "F1 macro media":     round(float(f1.mean()), 4),
                "F1 macro std":       round(float(f1.std()),  4),
                "Tempo total (s)":    round(elapsed, 2),
            })
            print(f"bal.acc={ba.mean():.4f} ± {ba.std():.4f}  [{elapsed:.1f}s]")
        except Exception as _e:
            print(f"FALHA ({_e})")

    # ── Wilcoxon vs PLS-DA ────────────────────────────────────────────────
    ref = scores_por_clf.get("PLS-DA")
    for r in resultados:
        nome_r = r["Classificador"]
        if nome_r == "PLS-DA" or ref is None:
            r["p Wilcoxon (vs PLS-DA)"] = "-"
        else:
            try:
                alt = scores_por_clf.get(nome_r)
                if alt is not None and len(alt) == len(ref):
                    _, pval = wilcoxon(ref, alt, alternative="two-sided",
                                       zero_method="wilcox")
                    r["p Wilcoxon (vs PLS-DA)"] = round(pval, 4)
                else:
                    r["p Wilcoxon (vs PLS-DA)"] = "n/a"
            except Exception:
                r["p Wilcoxon (vs PLS-DA)"] = "n/a"

    df_bench = pd.DataFrame(resultados)

    # ── Salvar CSV ────────────────────────────────────────────────────────
    cam_csv = os.path.join(pasta, "dados", "benchmark_classificadores.csv")
    df_bench.to_csv(cam_csv, index=False, sep=";", decimal=",")
    print(f"  -> {cam_csv}")

    # ── Figura boxplot ────────────────────────────────────────────────────
    if scores_por_clf:
        fig_benchmark_classificadores(scores_por_clf, n_splits, n_classes, cfg, pasta)

    # ── Curvas DET (OOF coletados no loop principal — sem re-execucao) ───
    if len(oof_probas) >= 2:
        try:
            fig_det_curvas(oof_probas, y_int, n_classes, cfg, pasta)
        except Exception as _e_det:
            print(f"\n  [AVISO] DET curves falhou: {_e_det}")

    # ── SHAP values (opcional) ────────────────────────────────────────────
    if cfg.executar_shap:
        # Guarda: RF multiclass (14 classes x 500 amostras x n_feat) ~600 MB
        if _verificar_ram(0.8, "SHAP TreeExplainer"):
            fig_shap_benchmark(X_raw, y_int, n_opt, cfg, pasta, wavenumbers)

    return df_bench


# =========================================================================
#  v28: Monte Carlo CV — IC95% por percentil
# =========================================================================

def fig_monte_carlo_distribuicao(scores_mc: Dict[str, List[float]],
                                  cfg: "Config", pasta: str) -> None:
    """Violin + IC95% percentil das distribuicoes Monte Carlo CV."""
    nomes = list(scores_mc.keys())
    dados = [np.array(scores_mc[n]) for n in nomes]
    cores = [cor(i) for i in range(len(nomes))]

    fig, ax = plt.subplots(figsize=(max(5.5, len(nomes) * 1.9), 5.0),
                           constrained_layout=True)
    parts = ax.violinplot(dados, positions=range(1, len(nomes) + 1),
                          showmedians=True, showextrema=False)
    for pc, c in zip(parts["bodies"], cores):
        pc.set_facecolor(c); pc.set_alpha(0.50)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2.0)

    for i, d in enumerate(dados, 1):
        ci_lo = float(np.percentile(d, 2.5))
        ci_hi = float(np.percentile(d, 97.5))
        ax.plot([i, i], [ci_lo, ci_hi], color="0.25", lw=2.0, zorder=5)
        ax.plot(i, float(np.mean(d)), "o", ms=6, color="white",
                markeredgecolor="0.2", markeredgewidth=1.2, zorder=6)
        ax.annotate(
            f"IC95%\n[{ci_lo:.3f},{ci_hi:.3f}]",
            xy=(i, ci_hi), xytext=(i + 0.18, ci_hi),
            fontsize=6.5, va="center", color="0.3")

    n_iter = max(len(d) for d in dados)
    ax.set_xticks(range(1, len(nomes) + 1))
    ax.set_xticklabels(nomes, fontsize=9)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(
        f"Monte Carlo CV ({n_iter} iteracoes, test={cfg.monte_carlo_test_size:.0%})\n"
        f"IC95% percentil — pre-processamento: {cfg.preprocessamento_padrao}",
        fontsize=8.5, loc="left")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", color="0.93", lw=0.5)
    ax.set_axisbelow(True)

    salvar(fig, "fig_monte_carlo_cv", pasta, cfg)
    plt.close(fig)


def _stratified_group_shuffle_splits(
        y_int: np.ndarray,
        grupos_cv: np.ndarray,
        n_splits: int,
        test_size: float,
        random_state: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split estratificado no nivel de grupo: cada grupo recebe a classe da maioria
    de suas amostras e o split e feito via StratifiedShuffleSplit sobre grupos.
    Garante (1) anti-leakage de replicas e (2) proporcao de classes no teste.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    grupos_u = np.unique(grupos_cv)
    # Classe dominante por grupo (votacao por maioria)
    cls_por_grupo = np.array([
        int(pd.Series(y_int[grupos_cv == g]).mode().iloc[0])
        for g in grupos_u
    ])

    sss = StratifiedShuffleSplit(n_splits=n_splits,
                                  test_size=test_size,
                                  random_state=random_state)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for grp_tr, grp_te in sss.split(grupos_u, cls_por_grupo):
        grupos_train = set(grupos_u[grp_tr])
        grupos_test  = set(grupos_u[grp_te])
        tr_idx = np.where(np.isin(grupos_cv, list(grupos_train)))[0]
        te_idx = np.where(np.isin(grupos_cv, list(grupos_test)))[0]
        splits.append((tr_idx, te_idx))
    return splits


def monte_carlo_cv(X_raw: np.ndarray, y_int: np.ndarray,
                   grupos_cv: Optional[np.ndarray],
                   lb: "LabelBinarizer",
                   n_opt: int, cfg: "Config", pasta: str) -> pd.DataFrame:
    """
    Monte Carlo CV com split estratificado por grupo (N repeticoes).
    Gera distribuicao empirica de Balanced Accuracy com IC95% por percentil.

    Se cfg.monte_carlo_incluir_todos=True, roda tambem SVM RBF, RF e XGBoost
    (mesmos hiperparametros do benchmark); caso contrario, apenas PLS-DA.

    Ref: Filzmoser et al. (2009) Anal. Chim. Acta 652:133-142.
    """
    import warnings
    from sklearn.base import clone
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.pipeline import Pipeline as _SKPipeline
    from sklearn.svm import SVC
    from sklearn.ensemble import (RandomForestClassifier,
                                  GradientBoostingClassifier)

    n_iter  = cfg.n_monte_carlo
    test_sz = cfg.monte_carlo_test_size
    preproc = construir_preprocessador(cfg)

    # Montar lista de modelos
    mc_clfs: List[Tuple[str, Any]] = [
        ("PLS-DA", PLSDAClassifier(n_components=n_opt)),
    ]
    if cfg.monte_carlo_incluir_todos:
        mc_clfs += [
            ("SVM RBF",
             SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                 random_state=cfg.seed, class_weight="balanced")),
            ("Random Forest",
             RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                    class_weight="balanced_subsample",
                                    n_jobs=1, random_state=cfg.seed)),
            ("Grad. Boost.",
             GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                        max_depth=3, random_state=cfg.seed)),
        ]
        try:
            from xgboost import XGBClassifier  # type: ignore
            mc_clfs.append(("XGBoost",
                             XGBClassifier(n_estimators=300, learning_rate=0.05,
                                           max_depth=4, subsample=0.8,
                                           colsample_bytree=0.8,
                                           eval_metric="mlogloss", verbosity=0,
                                           n_jobs=1, random_state=cfg.seed)))
        except ImportError:
            pass

    # Gerar splits estratificados por grupo (risco 3 resolvido)
    if grupos_cv is not None:
        splits = _stratified_group_shuffle_splits(
            y_int, grupos_cv, n_iter, test_sz, cfg.seed)
    else:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=n_iter,
                                     test_size=test_sz,
                                     random_state=cfg.seed)
        splits = list(sss.split(X_raw, y_int))

    # Correr modelos
    resultados_mc: List[Dict] = []
    scores_mc: Dict[str, List[float]] = {}

    t0_total = time.time()
    for nome, clf in mc_clfs:
        pipe = _SKPipeline([("preproc", clone(preproc)), ("clf", clf)])
        ba_list: List[float] = []
        f1_list: List[float] = []
        print(f"  MC CV [{nome}]: {n_iter} iteracoes ... ", end="", flush=True)
        t0 = time.time()
        for tr_idx, te_idx in splits:
            X_tr, X_te = X_raw[tr_idx], X_raw[te_idx]
            y_tr, y_te = y_int[tr_idx], y_int[te_idx]
            # Garantir todas as classes no treino e pelo menos 1 no teste
            if len(np.unique(y_tr)) < len(lb.classes_):
                continue
            if len(np.unique(y_te)) < 2:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe_i = clone(pipe)
                    pipe_i.fit(X_tr, y_tr)
                    y_pred = pipe_i.predict(X_te)
                    ba_list.append(balanced_accuracy_score(y_te, y_pred))
                    f1_list.append(
                        f1_score(y_te, y_pred, average="macro", zero_division=0))
            except Exception:
                pass

        elapsed = time.time() - t0
        arr_ba = np.array(ba_list) if ba_list else np.array([np.nan])
        arr_f1 = np.array(f1_list) if f1_list else np.array([np.nan])
        ci_lo  = float(np.nanpercentile(arr_ba, 2.5))
        ci_hi  = float(np.nanpercentile(arr_ba, 97.5))
        scores_mc[nome] = ba_list

        print(f"media={np.nanmean(arr_ba):.4f}  IC95%=[{ci_lo:.4f},{ci_hi:.4f}]"
              f"  [{elapsed:.1f}s]")
        resultados_mc.append({
            "Classificador":    nome,
            "Iteracoes validas": len(ba_list),
            "Media BA":         round(float(np.nanmean(arr_ba)), 4),
            "Mediana BA":       round(float(np.nanmedian(arr_ba)), 4),
            "Std BA":           round(float(np.nanstd(arr_ba)), 4),
            "IC95% inf":        round(ci_lo, 4),
            "IC95% sup":        round(ci_hi, 4),
            "Media F1 macro":   round(float(np.nanmean(arr_f1)), 4),
            "IC95% F1 inf":     round(float(np.nanpercentile(arr_f1, 2.5)), 4),
            "IC95% F1 sup":     round(float(np.nanpercentile(arr_f1, 97.5)), 4),
            "Fracao teste":     test_sz,
        })

    print(f"  [MC CV total: {time.time()-t0_total:.1f}s]")
    df_mc = pd.DataFrame(resultados_mc)

    cam_csv = os.path.join(pasta, "dados", "monte_carlo_cv.csv")
    df_mc.to_csv(cam_csv, index=False, sep=";", decimal=",")
    print(f"  -> {cam_csv}")

    # Figura violin — apenas modelos com >= 5 iteracoes validas
    scores_plot = {n: v for n, v in scores_mc.items() if len(v) >= 5}
    if scores_plot:
        fig_monte_carlo_distribuicao(scores_plot, cfg, pasta)

    return df_mc


# =========================================================================
#  v28: Curvas DET — Detection Error Tradeoff
# =========================================================================

def fig_det_curvas(oof_probas: Dict[str, np.ndarray],
                   y_int: np.ndarray,
                   n_classes: int,
                   cfg: "Config", pasta: str) -> None:
    """
    Curvas DET (Detection Error Tradeoff) macro-OvR por classificador.
    Cada curva e a media das curvas binarias One-vs-Rest por classe.

    Ref: Martin et al. (1997) NIST/SEMATECH Engineering Statistics Handbook.
    """
    try:
        from sklearn.metrics import det_curve
        from sklearn.preprocessing import label_binarize
    except ImportError:
        print("  [AVISO] det_curve indisponivel (sklearn < 0.24) — DET pulada.")
        return

    nomes = list(oof_probas.keys())
    cores = [cor(i) for i in range(len(nomes))]

    y_bin = label_binarize(y_int, classes=range(n_classes))
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    # Gerar duas versoes: escala linear (0-100%) + escala log (padrao NIST)
    for log_scale in (False, True):
        eps  = 0.1   # evitar log(0)
        lo   = eps if log_scale else 0.0
        hi   = 100.0
        grid = np.logspace(np.log10(eps), np.log10(hi), 500) if log_scale \
               else np.linspace(0.0, hi, 500)
        fmr_grid_pct = grid
        fmr_grid_frac = fmr_grid_pct / 100.0

        fig, ax = plt.subplots(figsize=(6.0, 5.5), constrained_layout=True)

        for nome, c in zip(nomes, cores):
            proba = oof_probas[nome]
            fnmr_acum = np.zeros(len(fmr_grid_frac))
            n_valid = 0
            for k in range(n_classes):
                y_k = y_bin[:, k]
                if y_k.sum() < 2 or k >= proba.shape[1]:
                    continue
                try:
                    fmr, fnmr, _ = det_curve(y_k, proba[:, k])
                    fnmr_acum += np.interp(fmr_grid_frac, fmr, fnmr,
                                           left=fnmr[0], right=fnmr[-1])
                    n_valid += 1
                except Exception:
                    pass
            if n_valid == 0:
                continue
            fnmr_media = fnmr_acum / n_valid
            ax.plot(fmr_grid_pct, fnmr_media * 100,
                    lw=1.8, color=c, label=nome, alpha=0.85)

        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.35,
                label="Ref. diagonal")
        ax.set_xlabel("False Match Rate — FMR (%)")
        ax.set_ylabel("False Non-Match Rate — FNMR (%)")
        escala_str = "log" if log_scale else "linear"
        ax.set_title(
            f"Curvas DET (Detection Error Tradeoff) — macro OvR [{escala_str}]\n"
            f"OOF predictions, {cfg.n_splits_cv}-fold GroupKFold",
            fontsize=8.5, loc="left")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        if log_scale:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(color="0.93", lw=0.5, which="both"); ax.set_axisbelow(True)

        sufixo = "_log" if log_scale else ""
        salvar(fig, f"fig_det_curvas{sufixo}", pasta, cfg)
        plt.close(fig)


# =========================================================================
#  v28: SHAP values — interpretabilidade espectral dos ensembles
# =========================================================================

def fig_shap_benchmark(X_raw: np.ndarray, y_int: np.ndarray,
                        n_opt: int, cfg: "Config", pasta: str,
                        wavenumbers: Optional[np.ndarray] = None) -> None:
    """
    SHAP TreeExplainer para RF, GBM e XGBoost (se disponivel).
    Plota barplot horizontal dos top-20 wavenumbers por mean |SHAP|.

    Ref: Lundberg & Lee (2017) NeurIPS — SHAP (SHapley Additive exPlanations).
    """
    try:
        import shap  # type: ignore
    except ImportError:
        print("  [AVISO] shap nao instalado — pip install shap. SHAP pulado.")
        return

    import warnings
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    preproc   = construir_preprocessador(cfg)
    X_proc    = clone(preproc).fit(X_raw).transform(X_raw)
    feat_names = ([f"{w:.0f}" for w in wavenumbers]
                  if wavenumbers is not None
                  else [f"X{i}" for i in range(X_proc.shape[1])])

    # Cap de memoria: subsample aleatorio de shap_max_amostras
    n_max = cfg.shap_max_amostras
    if X_proc.shape[0] > n_max:
        rng_shap = np.random.default_rng(cfg.seed)
        idx_shap = rng_shap.choice(X_proc.shape[0], n_max, replace=False)
        X_shap = X_proc[idx_shap]
        y_shap = y_int[idx_shap]
    else:
        X_shap = X_proc
        y_shap = y_int

    tree_clfs: List[Tuple[str, Any]] = [
        ("Random Forest",
         RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                class_weight="balanced_subsample",
                                n_jobs=1, random_state=cfg.seed)),
        ("Grad. Boost.",
         GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                    max_depth=3, random_state=cfg.seed)),
    ]
    try:
        from xgboost import XGBClassifier  # type: ignore
        tree_clfs.append(("XGBoost",
                           XGBClassifier(n_estimators=300, learning_rate=0.05,
                                         max_depth=4, subsample=0.8,
                                         colsample_bytree=0.8,
                                         eval_metric="mlogloss", verbosity=0,
                                         n_jobs=1, random_state=cfg.seed)))
    except ImportError:
        pass

    for idx, (nome, clf) in enumerate(tree_clfs):
        print(f"  SHAP {nome} ({len(X_shap)} amostras) ... ", end="", flush=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Treinar no conjunto completo; calcular SHAP no subconjunto
                clf.fit(X_proc, y_int)
                explainer = shap.TreeExplainer(clf)
                sv = explainer.shap_values(X_shap)

                # sv e lista (multiclass RF) ou ndarray 2D/3D (XGBoost/GBM)
                if isinstance(sv, list):
                    importance = np.mean([np.abs(s) for s in sv], axis=0).mean(axis=0)
                else:
                    sv_arr = np.array(sv)
                    if sv_arr.ndim == 3:        # (n_classes, n_samples, n_features)
                        importance = np.abs(sv_arr).mean(axis=(0, 1))
                    else:                       # (n_samples, n_features)
                        importance = np.abs(sv_arr).mean(axis=0)

                top_n   = min(20, len(importance))
                top_idx = np.argsort(importance)[-top_n:][::-1]
                top_imp = importance[top_idx]
                top_lbl = [feat_names[i] for i in top_idx]

                # ── Barplot de importancia SHAP ───────────────────────────
                fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
                ax.barh(range(top_n), top_imp[::-1],
                        color=cor(idx), alpha=0.76, edgecolor="white", lw=0.5)
                ax.set_yticks(range(top_n))
                ax.set_yticklabels(top_lbl[::-1], fontsize=8)
                ax.set_xlabel("Mean |SHAP value|")
                ax.set_title(
                    f"SHAP — {nome} (top-{top_n} bandas espectrais)\n"
                    f"Unidade: cm⁻¹  |  pre-proc: {cfg.preprocessamento_padrao}"
                    f"  |  n={len(X_shap)}",
                    fontsize=8.5, loc="left")
                ax.grid(axis="x", color="0.93", lw=0.5); ax.set_axisbelow(True)
                tag = nome.lower().replace(" ", "_").replace(".", "").replace("/", "_")
                salvar(fig, f"fig_shap_{tag}", pasta, cfg)
                plt.close(fig)

                # ── Dependence plots: top-3 features × classe ─────────────
                n_dep = min(3, top_n)
                unique_cls = np.unique(y_shap)
                cores_dep  = [cor(c) for c in unique_cls]
                for rank, feat_i in enumerate(top_idx[:n_dep]):
                    feat_vals = X_shap[:, feat_i]
                    # Para multiclass, usar shap medio sobre classes
                    if isinstance(sv, list):
                        shap_feat = np.mean([sv_k[:, feat_i] for sv_k in sv], axis=0)
                    else:
                        sv_arr2 = np.array(sv)
                        if sv_arr2.ndim == 3:
                            shap_feat = sv_arr2.mean(axis=0)[:, feat_i]
                        else:
                            shap_feat = sv_arr2[:, feat_i]

                    fig2, ax2 = plt.subplots(figsize=(5.5, 4.0),
                                             constrained_layout=True)
                    for ci, (cls_v, cc) in enumerate(zip(unique_cls, cores_dep)):
                        mask_c = y_shap == cls_v
                        ax2.scatter(feat_vals[mask_c], shap_feat[mask_c],
                                    color=cc, alpha=0.60, s=18, lw=0,
                                    label=f"Cls {cls_v}")
                    ax2.axhline(0, color="0.5", lw=0.8, ls="--")
                    ax2.set_xlabel(f"{top_lbl[rank]} cm⁻¹")
                    ax2.set_ylabel("SHAP value (medio)")
                    ax2.set_title(
                        f"SHAP Dependence — {nome}\n"
                        f"Feature #{rank+1}: {top_lbl[rank]}",
                        fontsize=8.5, loc="left")
                    if len(unique_cls) <= 10:
                        ax2.legend(fontsize=6.5, ncol=2, markerscale=1.2)
                    ax2.grid(color="0.93", lw=0.5); ax2.set_axisbelow(True)
                    salvar(fig2, f"fig_shap_dep_{tag}_feat{rank+1}", pasta, cfg)
                    plt.close(fig2)

                print("salvo.")
        except Exception as _e:
            print(f"FALHA ({_e})")


# =========================================================================
#  Orquestrador
# =========================================================================

# =========================================================================
#  v29: Compatibilidade de hardware — probe, auto-ajuste, guardas de RAM
# =========================================================================

def hardware_probe() -> Dict[str, Any]:
    """
    Detecta RAM, CPU e disco disponiveis.
    Usa psutil se disponivel; caso contrario retorna estimativas conservadoras.
    Nunca lanca excecao — falha silenciosamente com valores defaults seguros.
    """
    info: Dict[str, Any] = {
        "ram_total_gb":  4.0,   # defaults conservadores
        "ram_livre_gb":  2.0,
        "cpu_logicos":   2,
        "cpu_fisicos":   1,
        "disco_livre_gb": 5.0,
        "psutil_ok":     False,
    }
    try:
        import psutil as _ps
        mem = _ps.virtual_memory()
        info["ram_total_gb"]   = round(mem.total   / 1024**3, 1)
        info["ram_livre_gb"]   = round(mem.available / 1024**3, 1)
        info["cpu_logicos"]    = _ps.cpu_count(logical=True)  or 2
        info["cpu_fisicos"]    = _ps.cpu_count(logical=False) or 1
        try:
            info["disco_livre_gb"] = round(
                _ps.disk_usage(os.path.abspath(".")).free / 1024**3, 1)
        except Exception:
            pass
        info["psutil_ok"] = True
    except ImportError:
        # Fallback Windows: GlobalMemoryStatusEx via ctypes
        try:
            import ctypes
            class _MEMSTATUS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                             ("dwMemoryLoad", ctypes.c_ulong),
                             ("ullTotalPhys", ctypes.c_ulonglong),
                             ("ullAvailPhys", ctypes.c_ulonglong),
                             ("ullTotalPageFile", ctypes.c_ulonglong),
                             ("ullAvailPageFile", ctypes.c_ulonglong),
                             ("ullTotalVirtual", ctypes.c_ulonglong),
                             ("ullAvailVirtual", ctypes.c_ulonglong),
                             ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MEMSTATUS()
            ms.dwLength = ctypes.sizeof(ms)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))  # type: ignore
            info["ram_total_gb"] = round(ms.ullTotalPhys / 1024**3, 1)
            info["ram_livre_gb"] = round(ms.ullAvailPhys / 1024**3, 1)
        except Exception:
            pass
        # CPU via os
        try:
            info["cpu_logicos"] = os.cpu_count() or 2
            info["cpu_fisicos"] = max(1, (os.cpu_count() or 2) // 2)
        except Exception:
            pass
    except Exception:
        pass
    return info


def auto_ajustar_config_hardware(cfg: "Config",
                                  hw: Dict[str, Any]) -> List[str]:
    """
    Ajusta automaticamente limites do cfg com base na RAM livre detectada.
    Previne travamentos em maquinas com < 8 GB RAM disponivel.
    Retorna lista de mensagens de aviso para impressao.
    """
    avisos: List[str] = []
    ram = float(hw.get("ram_livre_gb", 16.0))

    if ram < 2.0:
        # Modo minimo absoluto: desabilitar tudo pesado
        if cfg.executar_shap:
            cfg.executar_shap = False
            avisos.append("SHAP desabilitado (RAM livre < 2 GB)")
        if cfg.executar_benchmark:
            cfg.executar_benchmark = False
            avisos.append("Benchmark desabilitado (RAM livre < 2 GB)")
        if cfg.executar_monte_carlo:
            cfg.executar_monte_carlo = False
            avisos.append("Monte Carlo CV desabilitado (RAM livre < 2 GB)")
        if cfg.n_splits_cv > 3:
            cfg.n_splits_cv = 3
            avisos.append("CV reduzido para 3 folds (RAM livre < 2 GB)")

    elif ram < 4.0:
        # RAM 2-4 GB: desabilitar SHAP e benchmark, limitar MC CV
        if cfg.executar_shap:
            cfg.executar_shap = False
            avisos.append("SHAP desabilitado (RAM livre < 4 GB)")
        if cfg.executar_benchmark:
            cfg.executar_benchmark = False
            avisos.append("Benchmark desabilitado (RAM livre < 4 GB). "
                          "Habilite manualmente se necessario.")
        if cfg.n_monte_carlo > 30:
            cfg.n_monte_carlo = 30
            avisos.append("Monte Carlo CV limitado a 30 iteracoes (RAM livre < 4 GB)")

    elif ram < 6.0:
        # RAM 4-6 GB: SHAP com amostragem reduzida, benchmark sem XGBoost via flag
        if cfg.executar_shap and cfg.shap_max_amostras > 150:
            cfg.shap_max_amostras = 150
            avisos.append(f"SHAP max_amostras reduzido para 150 (RAM livre < 6 GB)")
        if cfg.n_monte_carlo > 60:
            cfg.n_monte_carlo = 60
            avisos.append("Monte Carlo CV limitado a 60 iteracoes (RAM livre < 6 GB)")
        if cfg.monte_carlo_incluir_todos:
            cfg.monte_carlo_incluir_todos = False
            avisos.append("MC CV multi-modelo desabilitado (RAM livre < 6 GB)")

    elif ram < 8.0:
        # RAM 6-8 GB: reducoes moderadas
        if cfg.executar_shap and cfg.shap_max_amostras > 300:
            cfg.shap_max_amostras = 300
            avisos.append(f"SHAP max_amostras reduzido para 300 (RAM livre < 8 GB)")
        if cfg.n_monte_carlo > 80:
            cfg.n_monte_carlo = 80
            avisos.append("Monte Carlo CV limitado a 80 iteracoes (RAM livre < 8 GB)")

    return avisos


def _verificar_ram(min_gb: float, operacao: str) -> bool:
    """
    Verifica RAM livre antes de operacao pesada.
    Retorna True se seguro prosseguir, False se insuficiente.
    Nunca lanca excecao — fail-safe assume OK se psutil indisponivel.
    """
    try:
        import psutil as _ps
        livre_gb = _ps.virtual_memory().available / 1024**3
        if livre_gb < min_gb:
            print(f"  [AVISO RAM] '{operacao}' pulada: "
                  f"{livre_gb:.1f} GB livre < {min_gb:.1f} GB necessario. "
                  f"Reduza n_amostras, desabilite SHAP ou feche outros programas.")
            return False
    except ImportError:
        pass   # sem psutil: assume que ha memoria (falha graciosamente)
    except Exception:
        pass
    return True


def limpar_resultados_antigos(pasta_base: str,
                               manter_n: int = 3) -> Dict[str, Any]:
    """
    Remove as pastas de resultados mais antigas dentro de pasta_base,
    mantendo apenas as manter_n mais recentes (por data de modificacao).
    Retorna {'removidas': [paths], 'liberado_mb': float, 'erro': str|None}.
    """
    import shutil
    from pathlib import Path

    resultado: Dict[str, Any] = {"removidas": [], "liberado_mb": 0.0, "erro": None}
    base = Path(pasta_base)
    if not base.exists():
        resultado["erro"] = f"Pasta '{pasta_base}' nao encontrada."
        return resultado

    pastas = sorted(
        [p for p in base.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    a_remover = pastas[manter_n:]
    if not a_remover:
        return resultado

    for p in a_remover:
        try:
            tam = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            shutil.rmtree(p)
            resultado["removidas"].append(str(p))
            resultado["liberado_mb"] += tam / (1024 * 1024)
        except Exception as _e:
            resultado["erro"] = (resultado["erro"] or "") + f"\n{p}: {_e}"
    resultado["liberado_mb"] = round(resultado["liberado_mb"], 1)
    return resultado


def rmse_flat(a, b):
    return float(np.sqrt(np.mean((np.asarray(a).flatten()
                                  - np.asarray(b).flatten()) ** 2)))


def executar(cfg: Config):
    setup_matplotlib(cfg)

    # --- 0. Hardware probe + auto-ajuste preventivo -------------------------
    _hw = hardware_probe()
    _avisos_hw = auto_ajustar_config_hardware(cfg, _hw)
    print(f"[HARDWARE] RAM total: {_hw['ram_total_gb']:.1f} GB  "
          f"livre: {_hw['ram_livre_gb']:.1f} GB  "
          f"CPU: {_hw['cpu_fisicos']}f/{_hw['cpu_logicos']}l")
    for _av in _avisos_hw:
        print(f"  [AUTO-AJUSTE] {_av}")

    # --- 1. Carregamento (6-tupla com mae_id + metadados) ------------------
    wavenumbers, X_raw, rotulos, conc, mae_id, metadados_df = carregar_dados(cfg)
    X_raw   = np.asarray(X_raw,   dtype=float)
    rotulos = np.asarray(rotulos, dtype=str)
    if conc is not None:
        conc = np.asarray(conc, dtype=float)
    if mae_id is not None:
        mae_id = np.asarray(mae_id, dtype=str)

    # --- 1a. Truncamento espectral: remove ruido de borda da FFT ----------
    # SG derivativo amplifica os ultimos pontos da FFT (proximos a 0 cm-1
    # e ao final do interferograma). Sem truncar, esses pontos viram falsos
    # top-VIP com estabilidade 100% (artefato, nao quimica).
    n_orig = X_raw.shape[1]
    mask_wn = (wavenumbers >= cfg.wn_min) & (wavenumbers <= cfg.wn_max)
    if not mask_wn.any():
        raise ValueError(
            f"Nenhuma variavel sobrevive ao filtro "
            f"[{cfg.wn_min}, {cfg.wn_max}] cm-1. "
            f"Faixa real dos dados: [{wavenumbers.min():.1f}, "
            f"{wavenumbers.max():.1f}].")
    n_removidas = n_orig - int(mask_wn.sum())
    X_raw       = X_raw[:, mask_wn]
    wavenumbers = wavenumbers[mask_wn]
    print(f"[INFO] Truncamento espectral [{cfg.wn_min:.0f}, {cfg.wn_max:.0f}]"
          f" cm-1: {n_removidas}/{n_orig} variaveis removidas, "
          f"restam {len(wavenumbers)} ({wavenumbers.min():.1f}-"
          f"{wavenumbers.max():.1f}).")

    # --- 1a2. Exclusao de classes (ex: Copaiba com lote anomalo) ----------
    if cfg.excluir_classes:
        excl = set(str(c) for c in cfg.excluir_classes)
        mask_keep = ~np.isin(rotulos, list(excl))
        n_rem = int((~mask_keep).sum())
        if n_rem > 0:
            print(f"[INFO] Excluindo classes {sorted(excl)}: "
                  f"{n_rem} amostras removidas.")
            X_raw   = X_raw[mask_keep]
            rotulos = rotulos[mask_keep]
            conc    = conc[mask_keep] if conc is not None else None
            mae_id  = mae_id[mask_keep] if mae_id is not None else None

    # --- 1b. Pasta de saida descritiva -------------------------------------
    cfg.pasta_saida = gerar_nome_saida(cfg, len(np.unique(rotulos)),
                                         X_raw.shape[0])
    pasta = cfg.pasta_saida
    # Estrutura de subpastas (v20): dados/ figuras/ modelos/ logs/
    pasta_dados   = os.path.join(pasta, "dados")
    pasta_modelos = os.path.join(pasta, "modelos")
    pasta_logs    = os.path.join(pasta, "logs")
    for _p in (pasta, pasta_dados, os.path.join(pasta, "figuras"),
               pasta_modelos, pasta_logs):
        os.makedirs(_p, exist_ok=True)
    print(f"[INFO] Saida: {pasta}")
    print(f"[INFO] Subpastas: dados/ figuras/ modelos/ logs/")
    if metadados_df is not None:
        cam_meta = os.path.join(pasta_dados, "metadados.csv")
        metadados_df.to_csv(cam_meta, index=False, sep=";", decimal=",")
        print(f"[INFO] Metadados salvos: {cam_meta}")

    # --- 1c. Validacao de integridade --------------------------------------
    print("\n[0/7] Validacao de integridade da entrada...")
    X_raw, wavenumbers, rotulos, conc, mae_id, relatorio_entrada = validar_entrada(
        X_raw, wavenumbers, rotulos, conc, mae_id)
    relatorio_balanco = verificar_balanceamento(rotulos)

    # B1: mae_id agora e sincronizado DENTRO de validar_entrada (mesma mascara
    # de remocao de NaN/Inf). A validacao group-aware sobrevive a remocoes —
    # nao ha mais desligamento silencioso do GroupKFold por causa de 1 NaN.
    if mae_id is not None:
        mae_id = np.asarray(mae_id, dtype=str)

    # --- Estrategia de validacao: group-aware se mae_id disponivel ---------
    usar_grupos = (cfg.agrupar_por_mae_id and mae_id is not None
                   and len(np.unique(mae_id)) >= 3)
    if cfg.agrupar_por_mae_id and not usar_grupos:
        print("[INFO] GroupKFold desabilitado: mae_id indisponivel ou "
              "grupos insuficientes — usando StratifiedKFold (estratificada).")
    if usar_grupos and mae_id is not None:
        n_grupos = int(len(np.unique(mae_id)))
        print(f"[INFO] Validacao group-aware ATIVA: {n_grupos} grupos "
              f"unicos via mae_id.")

    # --- 1d. Hold-out externo (group-aware se possivel) --------------------
    X_holdout = None; rotulos_holdout = None; conc_holdout = None
    mae_id_holdout: Optional[np.ndarray] = None
    n_holdout = 0
    if cfg.frac_holdout > 0:
        try:
            # v15: opcionalmente exclui puros do sorteio — eles sempre ficam
            # no treino. O split roda so no subconjunto elegivel (adulterados).
            n_all = len(rotulos)
            if (cfg.holdout_preserva_puros and conc is not None):
                elegiveis = np.where(np.asarray(conc, dtype=float) > 0)[0]
                n_puros_reserv = n_all - len(elegiveis)
            else:
                elegiveis = np.arange(n_all)
                n_puros_reserv = 0

            Xe, rote = X_raw[elegiveis], rotulos[elegiveis]
            maee = mae_id[elegiveis] if mae_id is not None else None

            if usar_grupos and maee is not None:
                gss = GroupShuffleSplit(
                    n_splits=1, test_size=cfg.frac_holdout,
                    random_state=cfg.seed_holdout)
                tr_e, ho_e = next(gss.split(Xe, rote, groups=maee))
                tipo_ho = "GroupShuffleSplit (replicas juntas)"
            else:
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=cfg.frac_holdout,
                    random_state=cfg.seed_holdout)
                tr_e, ho_e = next(sss.split(Xe, rote))
                tipo_ho = "StratifiedShuffleSplit"
            # Remapeia indices do subconjunto elegivel para indices globais;
            # puros (nao-elegiveis) entram inteiros no treino.
            nao_elegiveis = np.setdiff1d(np.arange(n_all), elegiveis)
            tr_idx = np.concatenate([elegiveis[tr_e], nao_elegiveis])
            ho_idx = elegiveis[ho_e]
            if n_puros_reserv > 0:
                tipo_ho += f" + {n_puros_reserv} puros preservados"
            X_holdout       = X_raw[ho_idx]
            rotulos_holdout = rotulos[ho_idx]
            conc_holdout    = conc[ho_idx] if conc is not None else None
            if mae_id is not None:
                mae_id_holdout = mae_id[ho_idx]
            X_raw   = X_raw[tr_idx]
            rotulos = rotulos[tr_idx]
            conc    = conc[tr_idx] if conc is not None else None
            if mae_id is not None:
                mae_id = mae_id[tr_idx]
            n_holdout = int(len(ho_idx))
            print(f"[INFO] Hold-out ({tipo_ho}): {n_holdout} amostras "
                  f"reservadas (frac={cfg.frac_holdout:.2f}). "
                  f"Pipeline rodara em {len(tr_idx)} amostras.")
        except Exception as e:
            print(f"[AVISO] Hold-out falhou ({e}). Continuando sem holdout.")
            X_holdout = None

    if cfg.aplicar_sg and cfg.sg_window >= X_raw.shape[1]:
        raise ValueError(
            f"sg_window ({cfg.sg_window}) deve ser menor que o numero de "
            f"pontos espectrais ({X_raw.shape[1]}).")

    classes_unicas = np.unique(rotulos)
    mapa_cores     = mapear_cores_classes(classes_unicas)
    mapa_marcadores = mapear_marcadores_classes(classes_unicas)

    print(f"\n[INFO] Amostras : {X_raw.shape[0]}")
    print(f"[INFO] Variaveis: {X_raw.shape[1]}")
    print(f"[INFO] Classes  : {classes_unicas.tolist()}")
    print(f"[INFO] Conc.    : {'sim' if conc is not None else 'nao'}")
    _mae_info = (f"sim ({int(len(np.unique(mae_id)))} grupos)"
                  if mae_id is not None else "nao")
    print(f"[INFO] mae_id   : {_mae_info}")
    print(f"[INFO] Imbalance ratio: {relatorio_balanco['imbalance_ratio']:.2f}  "
          f"(max={relatorio_balanco['n_max']}, min={relatorio_balanco['n_min']})")

    contagem = {cls: int(np.sum(rotulos == cls)) for cls in classes_unicas}
    min_por_classe = min(contagem.values())
    n_splits = min(cfg.n_splits_cv, min_por_classe)
    if usar_grupos and mae_id is not None:
        # Numero de grupos por classe limita k
        n_grupos_por_classe = []
        for cls in classes_unicas:
            n_grupos_por_classe.append(
                int(len(np.unique(mae_id[rotulos == cls]))))
        min_grupos_classe = min(n_grupos_por_classe)
        n_splits = min(n_splits, min_grupos_classe)
        if n_splits < cfg.n_splits_cv:
            print(f"[AVISO] n_splits ajustado para {n_splits} "
                  f"(min de grupos/classe = {min_grupos_classe}).")
    elif n_splits < cfg.n_splits_cv:
        print(f"[AVISO] n_splits_cv={cfg.n_splits_cv} excede minimo por "
              f"classe ({min_por_classe}). Usando {n_splits}.")

    if usar_grupos:
        # StratifiedGroupKFold: estratifica por classe E agrupa por mae_id
        cv = StratifiedGroupKFold(n_splits=max(n_splits, 2), shuffle=True,
                                    random_state=cfg.seed)
        cv_label = f"StratifiedGroupKFold n_splits={n_splits}"
    elif cfg.n_repeats_cv > 1:
        cv = RepeatedStratifiedKFold(n_splits=n_splits,
                                      n_repeats=cfg.n_repeats_cv,
                                      random_state=cfg.seed)
        cv_label = f"RepeatedStratifiedKFold n_splits={n_splits} repeats={cfg.n_repeats_cv}"
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=cfg.seed)
        cv_label = f"StratifiedKFold n_splits={n_splits}"

    lb = LabelBinarizer()
    Y_bin = np.asarray(lb.fit_transform(rotulos), dtype=float)
    if Y_bin.ndim == 1:
        Y_bin = np.column_stack([1 - Y_bin, Y_bin])
    y_int = np.argmax(Y_bin, axis=1)

    # --- 2. Pre-processamento (uma vez, para visualizacao e PCA) -----------
    print(f"\n[1/7] Pre-processamento (preset='{cfg.preprocessamento_padrao}')")
    preproc_full = construir_preprocessador(cfg).fit(X_raw)
    X_processed  = np.asarray(preproc_full.transform(X_raw), dtype=float)

    # --- 3. Selecao de LVs por CV (sem leakage, group-aware se possivel) ---
    print(f"\n[2/7] Selecao de LVs ({cv_label})")

    def fabrica_pipeline(n_lv: int):
        return Pipeline([
            ("preproc", construir_preprocessador(cfg)),
            ("pls", PLSRegression(n_components=n_lv, scale=False)),
        ])

    erros_rmsecv: List[float] = []
    metricas_por_lv: List[Dict[str, float]] = []
    preds_por_lv: Dict[int, np.ndarray] = {}

    grupos_cv = mae_id if usar_grupos else None
    cv_indices = list(cv.split(X_raw, y_int, groups=grupos_cv))

    for n in range(1, cfg.max_lvs + 1):
        y_hat = np.zeros_like(Y_bin)
        contador = np.zeros(len(Y_bin), dtype=int)
        for tr, va in cv_indices:
            pipe = fabrica_pipeline(n)
            pipe.fit(X_raw[tr], Y_bin[tr])
            y_hat[va] += pipe.predict(X_raw[va])
            contador[va] += 1
        contador[contador == 0] = 1
        y_hat = y_hat / contador[:, None]

        erros_rmsecv.append(rmse_flat(Y_bin, y_hat))
        y_hat_int = np.argmax(y_hat, axis=1)
        m = metricas_classificacao(y_int, y_hat_int, np.arange(len(classes_unicas)))
        metricas_por_lv.append(m)
        preds_por_lv[n] = y_hat

    n_opt = int(np.argmin(erros_rmsecv)) + 1
    Y_cv  = preds_por_lv[n_opt]
    pred_lab = lb.classes_[np.argmax(Y_cv, axis=1)]
    print(f"  LVs otimas: {n_opt}")
    lvs_no_teto = (n_opt >= cfg.max_lvs)
    if lvs_no_teto:
        print(f"  [ATENCAO] LVs otimas ({n_opt}) == max_lvs ({cfg.max_lvs}): "
              f"RMSECV ainda nao atingiu plateau. Aumente max_lvs "
              f"(ex: {cfg.max_lvs + 10}) e rode novamente.")

    # --- 4. Modelo final (para VIP, scores, T2/Q) --------------------------
    pls_final = PLSRegression(n_components=n_opt, scale=False)
    pls_final.fit(X_processed, Y_bin)
    T_pls = np.asarray(pls_final.x_scores_,  dtype=float)
    P_pls = np.asarray(pls_final.x_loadings_, dtype=float).T
    var_lv_pls = variancia_explicada(X_processed, T_pls)
    vip = vip_scores(pls_final)

    # --- 5. PCA exploratoria -----------------------------------------------
    print("\n[3/7] PCA exploratoria")
    pca = PCA(n_components=min(cfg.n_pcs_pca, X_processed.shape[1],
                                X_processed.shape[0]))
    scores_pca = pca.fit_transform(X_processed)
    var_pca = pca.explained_variance_ratio_ * 100
    print(f"  PC1: {var_pca[0]:.2f}%  PC2: {var_pca[1]:.2f}%  "
          f"acumulado(2): {sum(var_pca[:2]):.2f}%")

    # --- 6. Teste de permutacao --------------------------------------------
    print(f"\n[4/7] Teste de permutacao (Y-randomization, "
          f"n={cfg.n_permutacoes})")
    if usar_grupos:
        cv_perm = StratifiedGroupKFold(n_splits=max(n_splits, 2),
                                         shuffle=True,
                                         random_state=cfg.seed)
    else:
        cv_perm = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=cfg.seed)
    perm_res = teste_permutacao(
        lambda: fabrica_pipeline(n_opt),
        X_raw, Y_bin, y_int, cv_perm, cfg.n_permutacoes, cfg.seed,
        groups=grupos_cv)
    perm_obs : float      = cast(float, perm_res["acc_observada"])
    perm_dist: np.ndarray = cast(np.ndarray, perm_res["accs_permutadas"])
    perm_p   : float      = cast(float, perm_res["p_value"])
    media_h0 = float(perm_dist.mean()) if len(perm_dist) > 0 else float("nan")
    print(f"  Acc observada = {perm_obs:.4f}  |  p = {perm_p:.4f}  "
          f"|  acc media H0 = {media_h0:.4f}")
    print(f"  Iteracoes validas: {cast(int, perm_res['n_validos'])}/"
          f"{cfg.n_permutacoes}  "
          f"(failure_rate = {cast(float, perm_res['failure_rate']):.1%})")

    # --- 6b. Teste de Wold (R2Y / Q2Y intercept) --------------------------
    wold_res: Optional[Dict[str, object]] = None
    if cfg.executar_wold:
        print(f"\n[4b/7] Teste de Wold (R2Y/Q2Y intercept, "
              f"n={cfg.n_permutacoes_wold})")
        wold_res = teste_wold(
            lambda: fabrica_pipeline(n_opt),
            X_raw, Y_bin, y_int, cv_perm, cfg.n_permutacoes_wold, cfg.seed,
            groups=grupos_cv)
        print(f"  R2Y obs = {cast(float, wold_res['r2_obs']):.4f}  |  "
              f"intercepto = {cast(float, wold_res['intercept_r2']):.4f}  "
              f"{'VALIDO' if wold_res['valid_r2'] else 'FALHA'} (limiar < 0.40)")
        print(f"  Q2Y obs = {cast(float, wold_res['q2_obs']):.4f}  |  "
              f"intercepto = {cast(float, wold_res['intercept_q2']):.4f}  "
              f"{'VALIDO' if wold_res['valid_q2'] else 'FALHA'} (limiar < 0.05)")

    # --- 6c. CV-ANOVA Eriksson --------------------------------------------
    cv_anova_res: Optional[Dict[str, float]] = None
    if cfg.executar_cv_anova:
        cv_anova_res = cv_anova_eriksson(Y_bin, Y_cv, n_opt)
        print(f"\n[4c/7] CV-ANOVA (Eriksson): F = {cv_anova_res['F']:.3f}  "
              f"p = {cv_anova_res['p_value']:.4g}  "
              f"(df = {cv_anova_res['df_model']}, {cv_anova_res['df_resid']})")

    # --- 7. Metricas e relatorio -------------------------------------------
    cm_mat = confusion_matrix(rotulos, pred_lab, labels=lb.classes_)
    metricas_finais = metricas_classificacao(rotulos, pred_lab, lb.classes_)
    print("\n[5/7] Metricas finais (CV):")
    for k, v in metricas_finais.items():
        print(f"  {k:>22s}: {v:.4f}")

    # --- 5b. BCa CI 95% para metricas via bootstrap estratificado ----------
    print(f"\n[5b/7] BCa CI 95% (n_boot={cfg.n_bootstrap_bca})")
    bca: Dict[str, Tuple[float, float, float]] = {}
    cls_arr = np.asarray(lb.classes_)
    metricas_funcoes = {
        "accuracy":          lambda yt, yp: accuracy_score(yt, yp),
        "balanced_accuracy": lambda yt, yp: balanced_accuracy_score(yt, yp),
        "f1_macro":          lambda yt, yp: f1_score(yt, yp, labels=cls_arr,
                                                       average="macro", zero_division=0),
        "cohen_kappa":       lambda yt, yp: cohen_kappa_score(yt, yp),
    }
    for nome, fn in metricas_funcoes.items():
        lo, hi, obs = bootstrap_bca_ci(rotulos, pred_lab, fn,
                                         n_boot=cfg.n_bootstrap_bca,
                                         alpha=0.05, seed=cfg.seed)
        bca[nome] = (lo, hi, obs)
        print(f"  {nome:>22s}: {obs:.4f}  [{lo:.4f}, {hi:.4f}]")
    print("\n" + str(classification_report(rotulos, pred_lab,
                                             target_names=lb.classes_,
                                             zero_division=0)))

    # --- 7b. R²X, R²Y, Q² ---------------------------------------------------
    r2x, r2y, q2 = metricas_modelo_pls(pls_final, X_processed, Y_bin, Y_cv)
    print(f"\n[5b/7] R2X = {r2x:.4f}  |  R2Y = {r2y:.4f}  |  Q2 = {q2:.4f}")

    # --- 8. Figuras --------------------------------------------------------
    print("\n[6/7] Gerando figuras...")
    aucs_roc: Dict[str, float] = {}
    # M1: mascara de puros (conc==0) para marcadores diferenciados
    puros_mask_fig = ((np.asarray(conc, dtype=float) == 0.0)
                       if conc is not None else None)
    # Flag de simbolos por classe (None -> todos circulo 'o')
    marcadores_fig = (mapa_marcadores if cfg.mostrar_marcadores_classe
                      else None)
    fig1_pca_scores(scores_pca, var_pca, rotulos, mapa_cores, cfg, pasta,
                     puros_mask=puros_mask_fig, mapa_marcadores=marcadores_fig)
    fig_hca_dendrograma(X_processed, rotulos, mapa_cores, cfg, pasta)
    fig_loadings_pca(pca, wavenumbers, cfg, pasta, n_pcs=2)
    if cfg.comparar_hca_pipelines:
        fig_hca_comparacao_pipelines(X_raw, rotulos, mapa_cores, cfg, pasta)
    fig2_plsda_scores(T_pls, var_lv_pls, rotulos, mapa_cores, cfg, pasta,
                       puros_mask=puros_mask_fig, mapa_marcadores=marcadores_fig)
    T2, Q, t2_lim, q_lim, out_t2, out_q = fig3_outliers(
        T_pls, P_pls, X_processed, rotulos, mapa_cores, n_opt, cfg, pasta)
    fig4_confusao(cm_mat, lb.classes_, rotulos, pred_lab, cfg, pasta)
    try:
        aucs_roc = fig_roc_auc(Y_bin, Y_cv, lb.classes_, cfg, pasta)
    except Exception as _e_roc:
        print(f"  [AVISO] ROC/AUC: {_e_roc}")
    # fig4b_metricas_globais — REMOVIDA do output principal (redundante
    # com resumo_modelo.txt). Funcao preservada para uso opcional.
    fig5_vip(vip, wavenumbers, top_n=20, cfg=cfg, pasta=pasta)

    if cfg.n_bootstrap_vip > 0:
        print(f"  [bootstrap VIP estratificado, n={cfg.n_bootstrap_vip}]")
        boot = bootstrap_vip_estratificado(
            X_processed, Y_bin, y_int, n_opt,
            cfg.n_bootstrap_vip, cfg.seed)
        boot_validos = cast(int, boot["n_validos"])
        print(f"  Iteracoes validas: {boot_validos}/"
              f"{cfg.n_bootstrap_vip}  (falhos: {cast(int, boot['n_falhos'])})")
        if boot_validos > 0:
            fig5b_vip_estabilidade(boot, wavenumbers, top_n=20,
                                     cfg=cfg, pasta=pasta)
        else:
            print("  [AVISO] Bootstrap VIP: 0 iteracoes validas — fig5b pulada.")

    fig6_preprocessamento(wavenumbers, X_raw, X_processed, rotulos,
                           mapa_cores, cfg, pasta)
    fig1_selecao_lvs(erros_rmsecv, metricas_por_lv, n_opt, cfg, pasta)

    # ---- Sprint 3 — SR, Score Contribution, DD-SIMCA, OPLS-DA -----------
    print("\n[Sprint3] Selectivity Ratio + Score Contribution...")
    sr = calcular_selectivity_ratio(pls_final, X_processed)
    fig_sprint3_sr_vip(vip, sr, wavenumbers, top_n=20, cfg=cfg, pasta=pasta)
    fig_sprint3_score_contribution(pls_final, X_processed, rotulos,
                                    wavenumbers, mapa_cores, top_n=20,
                                    cfg=cfg, pasta=pasta)

    # DD-SIMCA — modo de treino configuravel (v14).
    #   'todos' (default): treina cada modelo com todas as amostras da
    #     classe (exploratorio; robusto com poucos puros). sens = fracao
    #     da propria classe aceita.
    #   'puros': one-class N2 verdadeiro (treina so em puros) — requer
    #     >=15 puros/classe. Com 3/classe gera regiao minuscula e ~todos
    #     viram 'Desconhecido'. sens = puros aceitos; esp = adult rejeitados.
    ddsimca_res: Optional[Dict[str, Dict[str, Any]]] = None
    simca_pred: np.ndarray = np.array([], dtype=str)
    ddsimca_sens_esp: Dict[str, Tuple[float, float, int, int]] = {}
    if cfg.executar_ddsimca:
        modo_dd = (cfg.ddsimca_treinar_em or "todos").lower()
        if conc is not None:
            mask_puros = (np.asarray(conc, dtype=float) == 0.0)
        else:
            mask_puros = np.ones(len(rotulos), dtype=bool)

        if modo_dd == "puros":
            mask_treino = mask_puros
            n_treino = int(mask_puros.sum())
        else:
            mask_treino = np.ones(len(rotulos), dtype=bool)
            n_treino = len(rotulos)

        print(f"\n[Sprint3] DD-SIMCA "
              f"(n_components={cfg.ddsimca_n_components}, alpha=0.05, "
              f"ucl={cfg.ddsimca_ucl_method}, treino='{modo_dd}') "
              f"— {n_treino} amostras de treino")
        if modo_dd == "puros" and n_treino < (cfg.ddsimca_n_components + 2) * 2:
            print(f"  [AVISO] Poucos puros ({n_treino}) para one-class robusto"
                  f" (recomendado >=15/classe). Resultados exploratorios.")

        ddsimca = DDSimca(n_components=cfg.ddsimca_n_components, alpha=0.05,
                           ucl_method=cfg.ddsimca_ucl_method)
        ddsimca.fit(X_processed[mask_treino], rotulos[mask_treino])
        ddsimca_res   = ddsimca.score_matrix(X_processed)   # prediz em TODOS
        simca_pred    = ddsimca.predict(X_processed)
        n_unknown     = int(np.sum(simca_pred == "Desconhecido"))
        n_ambig       = int(np.sum(simca_pred == "Ambiguo"))

        print(f"  {'Classe':18s} {'sens':>7s} {'esp(adult)':>11s}")
        for cls in classes_unicas:
            if cls not in ddsimca_res:
                continue
            m = ddsimca_res[cls]
            aceito = ((np.asarray(m["T2_norm"]) <= 1.0) &
                      (np.asarray(m["Q_norm"])  <= 1.0))
            idx_puro_c  = (rotulos == cls) & mask_puros
            idx_adult_c = (rotulos == cls) & (~mask_puros)
            idx_cls     = (rotulos == cls)
            n_puro_c    = int(idx_puro_c.sum())
            n_adult_c   = int(idx_adult_c.sum())
            # sensibilidade: em 'puros' usa puros; em 'todos' usa a classe toda
            if modo_dd == "puros" and n_puro_c > 0:
                sens = float(np.mean(aceito[idx_puro_c]))
            else:
                sens = float(np.mean(aceito[idx_cls]))
            # B4: no modo 'todos' as adulteradas estao NO TREINO, logo a
            # "esp" seria in-sample (nao-autenticacao) e enganosa. So a
            # reportamos no modo 'puros' (one-class verdadeiro).
            if modo_dd == "puros":
                esp = (float(np.mean(~aceito[idx_adult_c]))
                       if n_adult_c > 0 else float("nan"))
            else:
                esp = float("nan")
            ddsimca_sens_esp[cls] = (sens, esp, n_puro_c, n_adult_c)
            esp_txt = f"{esp*100:9.1f}%" if esp == esp else "      n/a"
            print(f"  {cls:18s} {sens*100:6.1f}% {esp_txt}"
                  f"   (puros={n_puro_c}, adult={n_adult_c})")
        print(f"  Desconhecidos: {n_unknown}  |  Ambiguos: {n_ambig}")
        if ddsimca_res:
            fig_sprint3_ddsimca_acceptance(
                ddsimca_res, rotulos, mapa_cores, cfg, pasta,
                sens_esp=ddsimca_sens_esp)
            # Plots individuais por classe em subpasta ddsimca/
            fig_ddsimca_individuais(
                ddsimca_res, rotulos, mapa_cores, cfg, pasta,
                sens_esp=ddsimca_sens_esp)
            if len(ddsimca_res) >= 2:
                try:
                    fig_cooman_ddsimca(ddsimca_res, rotulos, mapa_cores,
                                       cfg, pasta)
                except Exception as _e_coom:
                    print(f"  [AVISO] Cooman's Plot: {_e_coom}")

    # OPLS-DA
    _opls_n_ortho: Optional[int] = None
    if cfg.executar_opls:
        n_cls_opls = len(classes_unicas)
        print(f"\n[Sprint3] OPLS-DA "
              f"(n_ortho={cfg.n_ortho_opls}, {n_cls_opls} classes)...")
        if n_cls_opls < 2:
            print("  [AVISO] OPLS-DA requer >= 2 classes.")
        else:
            try:
                opls = OPLSDAWrapper(n_ortho=cfg.n_ortho_opls)
                opls.fit(X_processed, Y_bin)
                t_pred_opls, t_orth_opls = opls.transform(X_processed)
                fig_sprint3_opls_scores(
                    t_pred_opls, t_orth_opls, rotulos, mapa_cores,
                    opls.n_ortho_fitted_, cfg=cfg, pasta=pasta)
                fig_splot_opls(X_processed, t_pred_opls, wavenumbers, cfg, pasta)
                _opls_n_ortho = opls.n_ortho_fitted_
                print(f"  Componentes ortogonais ajustados: {_opls_n_ortho}")
            except Exception as _e_opls:
                print(f"  [ERRO] OPLS-DA: {_e_opls}")

    # --- ETAPA 4: Selecao de variaveis ------------------------------------
    etapa4_res: Optional[Dict[str, Any]] = None
    if cfg.executar_etapa4:
        try:
            etapa4_res = etapa4_selecao_variaveis(
                X_processed, Y_bin, y_int, vip, sr, wavenumbers,
                cv_indices, n_opt, cfg, pasta, pasta_dados)
        except Exception as _e_e4:
            print(f"  [ERRO] Etapa 4: {_e_e4}")

    if cfg.comparar_pipelines:
        print("\n[6b/7] Comparacao de pipelines de pre-processamento...")
        comp = comparar_pipelines(cfg, X_raw, Y_bin, y_int, cv_indices,
                                    max_lv=cfg.max_lvs)
        fig_extra_comparacao_pipelines(comp, cfg, pasta)
        pd.DataFrame(comp).T.to_csv(
            os.path.join(pasta_dados, "comparacao_pipelines.csv"),
            sep=";", decimal=",")

    if wold_res is not None and cast(int, wold_res["n_validos"]) > 2:
        fig_extra_wold(wold_res, cfg, pasta)

    # --- 8b. Avaliacao em holdout independente ----------------------------
    metricas_holdout: Optional[Dict[str, float]] = None
    bca_holdout:      Optional[Dict[str, Tuple[float, float, float]]] = None
    if X_holdout is not None and rotulos_holdout is not None:
        rot_ho: np.ndarray = rotulos_holdout
        print(f"\n[6c/7] Avaliacao em holdout ({n_holdout} amostras)...")
        try:
            X_holdout_proc = preproc_full.transform(X_holdout)
            Y_holdout_hat = pls_final.predict(X_holdout_proc)
            pred_holdout = lb.classes_[np.argmax(Y_holdout_hat, axis=1)]
            cm_holdout = confusion_matrix(rot_ho, pred_holdout,
                                            labels=lb.classes_)
            metricas_holdout = metricas_classificacao(
                rot_ho, pred_holdout, lb.classes_)
            for k, v in metricas_holdout.items():
                print(f"  {k:>22s}: {v:.4f}")
            fig_extra_holdout(metricas_finais, metricas_holdout,
                                cm_holdout, lb.classes_, n_holdout, cfg, pasta)
            # BCa CI no holdout tambem
            bca_holdout = {}
            for nome, fn in metricas_funcoes.items():
                lo, hi, obs = bootstrap_bca_ci(
                    rot_ho, pred_holdout, fn,
                    n_boot=cfg.n_bootstrap_bca, alpha=0.05,
                    seed=cfg.seed + 1)
                bca_holdout[nome] = (lo, hi, obs)
        except Exception as e:
            print(f"  [ERRO] Avaliacao em holdout falhou: {e}")
            metricas_holdout = None

    # --- 9. Identificadores e resumo (separados dos graficos) --------------
    salvar_identificadores(rotulos, pred_lab, T_pls, T2, Q,
                            t2_lim, q_lim, pasta_dados)
    print(f"  -> {os.path.join(pasta_dados, 'amostras_identificadores.csv')}")

    # Formata strings de CI para o resumo
    def _ci_str(b):
        if b is None: return "-"
        lo, hi, obs = b
        if not np.isfinite(lo): return f"{obs:.4f} [CI indisponivel]"
        return f"{obs:.4f} [{lo:.4f}, {hi:.4f}]"

    _preset_str = (cfg.preprocessamento_padrao or "custom").lower()
    if _preset_str in ("snv_sg_mc", "msc_sg_mc"):
        _scat = "SNV" if _preset_str == "snv_sg_mc" else "MSC"
        _pp_descr = (f"{_scat} -> SG(w={cfg.sg_window},p={cfg.sg_polyorder},"
                      f"d={cfg.sg_deriv}) -> mean-centering")
    elif _preset_str == "autoscaling":
        _pp_descr = "Autoscaling (mean + unit variance)"
    elif _preset_str == "mc":
        _pp_descr = "Mean-centering"
    else:
        _pp_descr = " -> ".join(
            (["SNV"] if cfg.aplicar_snv else []) +
            ([f"SG(w={cfg.sg_window},p={cfg.sg_polyorder},d={cfg.sg_deriv})"]
             if cfg.aplicar_sg else []) +
            (["mean-centering"] if cfg.aplicar_mc else []))
    resumo = {
        "Total de amostras":      int(X_raw.shape[0]),
        "Total de variaveis":     int(X_raw.shape[1]),
        "Total de classes":       int(len(classes_unicas)),
        "Metodo":                 "PLS-DA",
        "Pre-processamento":      _pp_descr,
        "Faixa espectral (cm-1)": f"[{cfg.wn_min:.0f}, {cfg.wn_max:.0f}]",
        "LVs otimas":             int(n_opt),
        "LVs no teto (max_lvs)":  ("SIM - aumente max_lvs" if lvs_no_teto
                                    else "nao"),
        "Validacao":              cv_label + (
            f", repeats={cfg.n_repeats_cv}" if cfg.n_repeats_cv > 1 and
            not usar_grupos else ""),
        "Group-aware (mae_id)":   "sim" if usar_grupos else "nao",
        "N grupos mae_id":        (int(len(np.unique(mae_id)))
                                    if mae_id is not None else 0),
        "Nivel":                  cfg.nivel,
        "Tag":                    cfg.tag if cfg.tag else "-",
        "Accuracy (CV)":          float(metricas_finais["accuracy"]),
        "Balanced accuracy":      float(metricas_finais["balanced_accuracy"]),
        "F1 (macro)":             float(metricas_finais["f1_macro"]),
        "Cohen's kappa":          float(metricas_finais["cohen_kappa"]),
        "R2X":                    float(r2x),
        "R2Y":                    float(r2y),
        "Q2":                     float(q2),
        "Permutation p-value":    float(perm_p),
        "Permutation n_validos":  cast(int, perm_res["n_validos"]),
        "Permutation n_falhos":   cast(int, perm_res["n_falhos"]),
        "Permutation failure_rate": cast(float, perm_res["failure_rate"]),
        "Hotelling T2 (95%)":     float(t2_lim),
        # B7: notacao adaptativa — com SG-derivada + muitas LVs, Q_lim e
        # minusculo e ":.4f" exibia 0.0000 (mascarando o valor real).
        "Q-residual (95%)":       (f"{q_lim:.4g}" if abs(q_lim) < 1e-3
                                    else f"{q_lim:.4f}"),
        "N outliers T2":          int(out_t2.size),
        "N outliers Q":           int(out_q.size),
        "Imbalance ratio":        cast(float, relatorio_balanco["imbalance_ratio"]),
        "Classe maior":           cast(int, relatorio_balanco["n_max"]),
        "Classe menor":           cast(int, relatorio_balanco["n_min"]),
        "Integridade NaN":        cast(int, relatorio_entrada["n_nan_amostras"]),
        "Integridade Inf":        cast(int, relatorio_entrada["n_inf_amostras"]),
        "Variaveis constantes":   cast(int, relatorio_entrada["n_constantes_removidas"]),
        "Duplicatas exatas":      cast(int, relatorio_entrada["n_duplicatas_exatas"]),
        "Duplicatas aproximadas": cast(int, relatorio_entrada["n_duplicatas_aproximadas"]),
        "BCa Accuracy":           _ci_str(bca.get("accuracy")),
        "BCa Balanced acc.":      _ci_str(bca.get("balanced_accuracy")),
        "BCa F1 (macro)":         _ci_str(bca.get("f1_macro")),
        "BCa Cohen's kappa":      _ci_str(bca.get("cohen_kappa")),
    }
    # ROC/AUC (v24)
    if aucs_roc:
        resumo["ROC AUC macro (OvR)"] = float(aucs_roc.get("macro", float("nan")))
        resumo["--- ROC AUC por classe ---"] = ""
        for _cls, _auc in aucs_roc.items():
            if _cls != "macro":
                resumo[f"  AUC {_cls}"] = float(_auc)
    if wold_res is not None:
        resumo["Wold R2Y intercept"]    = cast(float, wold_res["intercept_r2"])
        resumo["Wold Q2Y intercept"]    = cast(float, wold_res["intercept_q2"])
        resumo["Wold R2Y valido (<.40)"] = cast(bool,  wold_res["valid_r2"])
        resumo["Wold Q2Y valido (<.05)"] = cast(bool,  wold_res["valid_q2"])
    if cv_anova_res is not None:
        resumo["CV-ANOVA F"]      = float(cv_anova_res["F"])
        resumo["CV-ANOVA p"]      = float(cv_anova_res["p_value"])
    if X_holdout is not None and metricas_holdout is not None:
        resumo["Holdout n"]          = int(n_holdout)
        resumo["Holdout accuracy"]   = float(metricas_holdout["accuracy"])
        resumo["Holdout balanced acc"] = float(metricas_holdout["balanced_accuracy"])
        resumo["Holdout F1 (macro)"] = float(metricas_holdout["f1_macro"])
        resumo["Holdout Cohen kappa"] = float(metricas_holdout["cohen_kappa"])
        if bca_holdout is not None:
            resumo["BCa Holdout Accuracy"] = _ci_str(bca_holdout.get("accuracy"))
            resumo["BCa Holdout Bal.acc"]  = _ci_str(bca_holdout.get("balanced_accuracy"))
    # Sprint 3 — adicionar ao resumo depois que o dict ja existe
    if cfg.executar_ddsimca and ddsimca_res is not None:
        resumo["DD-SIMCA n_components"]    = int(cfg.ddsimca_n_components)
        resumo["DD-SIMCA n_desconhecidos"] = int(np.sum(simca_pred == "Desconhecido"))
        resumo["DD-SIMCA n_ambiguos"]      = int(np.sum(simca_pred == "Ambiguo"))
        # B4 — rotulo honesto do modo de treino. No modo 'todos' sens/esp
        # sao aceitacao IN-SAMPLE, NAO metricas de autenticacao one-class.
        resumo["DD-SIMCA modo treino"] = (
            modo_dd + (" (one-class)" if modo_dd == "puros"
                       else " (in-sample; sens/esp NAO sao autenticacao)"))
        # C4 — sensibilidade/especificidade one-class por classe
        for cls in classes_unicas:
            if cls in ddsimca_sens_esp:
                s_c, e_c, npc, nac = ddsimca_sens_esp[cls]
                esp_s = f"{e_c*100:.1f}%" if e_c == e_c else "n/a"
                resumo[f"DD-SIMCA {cls} sens/esp"] = (
                    f"{s_c*100:.1f}% / {esp_s} "
                    f"(puros={npc}, adult={nac})")
    if _opls_n_ortho is not None:
        resumo["OPLS-DA n_ortho"] = int(_opls_n_ortho)

    # Etapa 4 — selecao de variaveis
    if etapa4_res is not None:
        resumo["--- Etapa 4: selecao de variaveis ---"] = ""
        for t in etapa4_res["tabela"]:
            resumo[f"  {t['metodo']}"] = (
                f"bal.acc={t['balanced_accuracy']:.3f} | "
                f"Q2={t['q2']:.3f} | {t['n_vars']} vars | {t['n_lv']} LVs")
        mlhr = etapa4_res["melhor"]
        resumo["  >> Mais parcimonioso"] = (
            f"{mlhr['metodo']} ({mlhr['n_vars']} vars, "
            f"bal.acc={mlhr['balanced_accuracy']:.3f})")

    # C6 — Outliers T2 por classe (diagnostico de batch/lote)
    resumo["--- Outliers T2 por classe ---"] = ""
    set_out_t2 = set(out_t2.tolist())
    for cls in classes_unicas:
        idx_cls = np.where(rotulos == cls)[0]
        n_out_c = int(sum(1 for i in idx_cls if i in set_out_t2))
        n_tot_c = int(len(idx_cls))
        frac_c  = n_out_c / max(n_tot_c, 1)
        flag = "  <-- ANOMALO (>30%)" if frac_c > 0.30 else ""
        resumo[f"  T2-outliers {cls}"] = (
            f"{n_out_c}/{n_tot_c} ({frac_c*100:.1f}%){flag}")
        if frac_c > 0.30:
            print(f"[ATENCAO] Classe '{cls}': {frac_c*100:.0f}% outliers T2 "
                  f"({n_out_c}/{n_tot_c}) — possivel batch/lote anomalo.")

    # M4 — Accuracy (recall) por classe
    resumo["--- Accuracy por classe ---"] = ""
    rec_por_classe = recall_score(rotulos, pred_lab, labels=lb.classes_,
                                   average=None, zero_division=0)
    for cls, rec_c in zip(lb.classes_, np.asarray(rec_por_classe)):
        resumo[f"  Acc {cls}"] = float(rec_c)

    salvar_resumo_modelo(pasta_logs, resumo)
    print(f"  -> {os.path.join(pasta_logs, 'resumo_modelo.txt')}")

    # --- 9a. Auto-Benchmark (opcional) ─────────────────────────────────────
    if cfg.executar_benchmark:
        print("\n[7b/7] Auto-Benchmark (SVM / RF / XGBoost vs PLS-DA)...")
        # Guarda: ~1.2 GB pico (SVM kernel matrix + OOF proba)
        if _verificar_ram(1.2, "Auto-Benchmark"):
            try:
                bench_df = benchmark_classificadores(
                    X_raw, y_int, grupos_cv, lb, n_opt, cfg, pasta,
                    wavenumbers=wavenumbers)
                print(bench_df.to_string(index=False))
            except Exception as _e_bench:
                print(f"  [AVISO] Benchmark falhou: {_e_bench}")

    # --- 9a2. Monte Carlo CV (opcional) ────────────────────────────────────
    if cfg.executar_monte_carlo:
        print("\n[7c/7] Monte Carlo CV (IC95% por percentil)...")
        # Guarda: ~400 MB (PLS-DA x N splits em serie)
        if _verificar_ram(0.5, "Monte Carlo CV"):
            try:
                mc_df = monte_carlo_cv(
                    X_raw, y_int, grupos_cv, lb, n_opt, cfg, pasta)
                print(mc_df.to_string(index=False))
            except Exception as _e_mc:
                print(f"  [AVISO] Monte Carlo CV falhou: {_e_mc}")

    # --- 9b. Exportar modelo final (modelos/) — joblib opcional -----------
    try:
        import joblib
        preproc_export = construir_preprocessador(cfg).fit(X_raw)
        pacote_modelo = {
            "preprocessador": preproc_export,
            "pls_final":      pls_final,
            "label_binarizer": lb,
            "classes":        list(lb.classes_),
            "wavenumbers":    wavenumbers,
            "n_opt":          int(n_opt),
            "preset":         cfg.preprocessamento_padrao,
            "wn_min":         cfg.wn_min, "wn_max": cfg.wn_max,
            # v25: limites para diagnosticos em novos dados (Aba Predicao)
            "t2_ucl":         float(t2_lim),
            "q_ucl":          float(q_lim),
        }
        cam_modelo = os.path.join(pasta_modelos, "modelo_plsda.joblib")
        joblib.dump(pacote_modelo, cam_modelo)
        print(f"  -> {cam_modelo}")
    except Exception as _e_mod:
        print(f"  [AVISO] Exportacao do modelo pulada: {_e_mod}")

    if out_t2.size or out_q.size:
        print(f"\n[INFO] Outliers (T2 > lim): {out_t2.tolist()}")
        print(f"[INFO] Outliers (Q  > lim): {out_q.tolist()}")

    # --- 9. PLS regressao (opcional) ---------------------------------------
    # Guarda: so executa se (1) ha teores, (2) sem NaN, (3) variancia > 0,
    # (4) pelo menos 10 amostras com teor > 0 (caso contrario regressao em
    # quase-puros nao tem sinal).
    _pls_reg_ok = False
    if conc is not None:
        conc_arr = np.asarray(conc, dtype=float)
        n_nan = int(np.isnan(conc_arr).sum())
        n_nonzero = int(np.sum(conc_arr > 0))
        if n_nan > 0:
            print(f"\n[7/7] PLS regressao — PULADA: {n_nan} amostras com "
                  f"conc=NaN.")
        elif float(conc_arr.std()) < 1e-8:
            print(f"\n[7/7] PLS regressao — PULADA: variancia de conc ~= 0.")
        elif n_nonzero < 10:
            print(f"\n[7/7] PLS regressao — PULADA: apenas {n_nonzero} "
                  f"amostras com teor > 0 (precisa >= 10).")
        else:
            _pls_reg_ok = True

    if _pls_reg_ok and conc is not None:
        print(f"\n[7/7] PLS regressao "
              f"(target=teor%, {int(np.sum(conc > 0))} adulterados + "
              f"{int(np.sum(conc == 0))} puros)")
        Y_reg = np.asarray(conc, dtype=float).reshape(-1, 1)

        # Split calibracao/validacao — group-aware se mae_id disponivel
        # (replicas T1/T2/T3 do mesmo ponto nunca separadas entre cal/val).
        if mae_id is not None:
            gss_reg = GroupShuffleSplit(n_splits=1, train_size=cfg.frac_cal,
                                         random_state=cfg.seed)
            ic, iv = next(gss_reg.split(X_raw, Y_reg, groups=mae_id))
            print(f"  Split cal/val: GroupShuffleSplit por mae_id "
                  f"({len(ic)} cal / {len(iv)} val)")
        else:
            rng   = np.random.default_rng(cfg.seed)
            idx_p = rng.permutation(len(conc))
            n_cal = int(cfg.frac_cal * len(conc))
            ic, iv = idx_p[:n_cal], idx_p[n_cal:]
        Xc_raw, Yc = X_raw[ic], Y_reg[ic]
        Xv_raw, Yv = X_raw[iv], Y_reg[iv]

        lv_max = min(cfg.max_lvs, max(2, Xc_raw.shape[0] // 5))

        # CV interna — GroupKFold por mae_id (C5: nao vaza replicas)
        if mae_id is not None:
            grupos_cal   = mae_id[ic]
            n_grupos_cal = int(len(np.unique(grupos_cal)))
            n_splits_reg = max(2, min(n_splits, n_grupos_cal))
            cv_reg = GroupKFold(n_splits=n_splits_reg)
            grupos_cv_reg: Optional[np.ndarray] = grupos_cal
            print(f"  CV interna: GroupKFold n_splits={n_splits_reg} "
                  f"({n_grupos_cal} grupos na calibracao)")
        else:
            n_splits_reg = max(2, min(n_splits, Xc_raw.shape[0] // 2))
            cv_reg = KFold(n_splits=n_splits_reg, shuffle=True,
                            random_state=cfg.seed)
            grupos_cv_reg = None

        erros_reg = []
        preds_reg = []
        for n in range(1, lv_max + 1):
            pipe = Pipeline([
                ("preproc", construir_preprocessador(cfg)),
                ("pls", PLSRegression(n_components=n, scale=False)),
            ])
            Y_hat = cross_val_predict(pipe, Xc_raw, Yc, cv=cv_reg,
                                       groups=grupos_cv_reg)
            erros_reg.append(rmse_flat(Yc, Y_hat))
            preds_reg.append(Y_hat)

        n_opt_reg = int(np.argmin(erros_reg)) + 1
        pipe_final = Pipeline([
            ("preproc", construir_preprocessador(cfg)),
            ("pls", PLSRegression(n_components=n_opt_reg, scale=False)),
        ]).fit(Xc_raw, Yc)
        Yc_hat = pipe_final.predict(Xc_raw)
        Yv_hat = pipe_final.predict(Xv_raw)
        Yc_cv  = preds_reg[n_opt_reg - 1]

        rmsec  = rmse_flat(Yc, Yc_hat)
        rmsecv = rmse_flat(Yc, Yc_cv)
        rmsep  = rmse_flat(Yv, Yv_hat)
        bias_v = float(np.mean(np.asarray(Yv_hat).flatten()
                                - np.asarray(Yv).flatten()))
        r2c    = float(r2_score(Yc, Yc_hat))
        r2v    = float(r2_score(Yv, Yv_hat))

        fig7_pls_regressao(Yc, Yc_hat, Yv, Yv_hat, erros_reg, n_opt_reg,
                            r2c, r2v, rmsec, rmsecv, rmsep, bias_v, cfg, pasta)

        print(f"  LVs    : {n_opt_reg}")
        print(f"  RMSEC  : {rmsec:.3f}  |  RMSECV: {rmsecv:.3f}  "
              f"|  RMSEP: {rmsep:.3f}")
        print(f"  R2cal  : {r2c:.4f}  |  R2val : {r2v:.4f}  "
              f"|  Bias: {bias_v:.4f}")
    elif conc is None:
        print("\n[7/7] PLS regressao — pulado (sem coluna de concentracao)")

    print(f"\n{'=' * 60}")
    print(f"  Pipeline concluido.")
    print(f"  Resultados em: {pasta}")
    print(f"{'=' * 60}")


# =========================================================================
#  CAMADA ACESSIVEL (v23) — configuracao em YAML + assistente de terminal
#  Objetivo: usar o pipeline SEM editar o codigo. Toda a configuracao de
#  usuario fica em config.yaml (linguagem simples). O assistente CMD
#  (menu_interativo) le/edita/salva esse arquivo e dispara executar().
#  Fonte UNICA de verdade: _CONFIG_SPEC mapeia campo amigavel <-> Config,
#  e alimenta tanto o YAML quanto o menu (e, depois, o app web).
# =========================================================================

_PRE_PROC_FRIENDLY: Dict[str, str] = {
    "MSC+SG+MC":      "msc_sg_mc",
    "SNV+SG+MC":      "snv_sg_mc",
    "Autoscaling":    "autoscaling",
    "Mean-centering": "mc",
}
_PRE_PROC_INV: Dict[str, str] = {v: k for k, v in _PRE_PROC_FRIENDLY.items()}

# Cada campo: chave_yaml, atributo da Config, tipo, descricao, opcoes.
#   tipo in {"str","int","float","bool","list","choice","preproc"}
_CONFIG_SPEC: List[Dict[str, Any]] = [
    {"key": "modo_entrada", "attr": "modo", "tipo": "choice",
     "desc": "Origem dos dados: dx (JCAMP-DX, FT-NIR) | csv (tabela generica) | sintetico (teste)",
     "opcoes": ["dx", "csv", "sintetico"]},
    {"key": "pasta_dados", "attr": "pasta_entrada", "tipo": "str",
     "desc": "Pasta com os arquivos .dx (modo dx; uma subpasta por classe)", "opcoes": None},
    {"key": "arquivo_csv", "attr": "arquivo_csv", "tipo": "str",
     "desc": "Caminho do CSV (modo csv): colunas espectrais/variaveis + 1 coluna de classe", "opcoes": None},
    {"key": "coluna_classe", "attr": "coluna_classe", "tipo": "str",
     "desc": "Nome da coluna de classe/rotulo no CSV (modo csv)", "opcoes": None},
    {"key": "coluna_concentracao", "attr": "coluna_conc", "tipo": "str_opcional",
     "desc": "Nome da coluna de concentracao no CSV (vazio se nao houver; modo csv)", "opcoes": None},
    {"key": "pasta_saida", "attr": "pasta_saida_raiz", "tipo": "str",
     "desc": "Pasta onde os resultados serao gravados", "opcoes": None},
    {"key": "nivel", "attr": "nivel", "tipo": "choice",
     "desc": "Nivel: N1=especie | N2=puro/adulterado | N3=teor", "opcoes": ["N1", "N2", "N3"]},
    {"key": "pre_processamento", "attr": "preprocessamento_padrao", "tipo": "preproc",
     "desc": "Pre-processamento espectral", "opcoes": list(_PRE_PROC_FRIENDLY)},
    {"key": "faixa_min_cm", "attr": "wn_min", "tipo": "float",
     "desc": "Inicio da faixa espectral util (cm-1)", "opcoes": None},
    {"key": "faixa_max_cm", "attr": "wn_max", "tipo": "float",
     "desc": "Fim da faixa espectral util (cm-1)", "opcoes": None},
    {"key": "excluir_classes", "attr": "excluir_classes", "tipo": "list",
     "desc": "Especies a remover da analise (ex: [Copaiba])", "opcoes": None},
    {"key": "max_lvs", "attr": "max_lvs", "tipo": "int",
     "desc": "Numero maximo de variaveis latentes (LVs) testadas", "opcoes": None},
    {"key": "holdout_fracao", "attr": "frac_holdout", "tipo": "float",
     "desc": "Fracao reservada para teste externo (0 a 0.5)", "opcoes": None},
    {"key": "validacao_group_aware", "attr": "agrupar_por_mae_id", "tipo": "bool",
     "desc": "Manter replicas (T1/T2/T3) juntas na validacao (evita vazamento)", "opcoes": None},
    {"key": "n_permutacoes", "attr": "n_permutacoes", "tipo": "int",
     "desc": "Iteracoes do teste de permutacao", "opcoes": None},
    {"key": "teste_wold", "attr": "executar_wold", "tipo": "bool",
     "desc": "Rodar teste de Wold (intercepts R2Y/Q2Y)", "opcoes": None},
    {"key": "teste_cv_anova", "attr": "executar_cv_anova", "tipo": "bool",
     "desc": "Rodar CV-ANOVA (Eriksson)", "opcoes": None},
    {"key": "selecao_variaveis_etapa4", "attr": "executar_etapa4", "tipo": "bool",
     "desc": "Rodar Etapa 4 (iPLS / VIP / SR / sPLS-DA)", "opcoes": None},
    {"key": "ddsimca", "attr": "executar_ddsimca", "tipo": "bool",
     "desc": "Rodar DD-SIMCA (classificacao one-class)", "opcoes": None},
    {"key": "opls_da", "attr": "executar_opls", "tipo": "bool",
     "desc": "Rodar OPLS-DA", "opcoes": None},
    {"key": "comparar_pre_processamentos", "attr": "comparar_pipelines", "tipo": "bool",
     "desc": "Comparar varios pre-processamentos", "opcoes": None},
    {"key": "benchmark", "attr": "executar_benchmark", "tipo": "bool",
     "desc": "Auto-Benchmark: SVM RBF / RF / XGBoost vs PLS-DA (mesma CV group-aware)", "opcoes": None},
    {"key": "monte_carlo", "attr": "executar_monte_carlo", "tipo": "bool",
     "desc": "Monte Carlo CV: IC95% por percentil (N repeticoes estratificadas por grupo)", "opcoes": None},
    {"key": "n_monte_carlo", "attr": "n_monte_carlo", "tipo": "int",
     "desc": "Numero de repeticoes do Monte Carlo CV", "opcoes": None},
    {"key": "monte_carlo_incluir_todos", "attr": "monte_carlo_incluir_todos", "tipo": "bool",
     "desc": "MC CV: incluir SVM RBF / RF / XGBoost alem do PLS-DA (mais lento)", "opcoes": None},
    {"key": "shap_benchmark", "attr": "executar_shap", "tipo": "bool",
     "desc": "SHAP values (TreeExplainer) para RF/XGBoost/GBM — interpretabilidade espectral", "opcoes": None},
    {"key": "shap_max_amostras", "attr": "shap_max_amostras", "tipo": "int",
     "desc": "Limite de amostras para calculo de SHAP (controle de memoria)", "opcoes": None},
    {"key": "figuras_mostrar_marcadores", "attr": "mostrar_marcadores_classe", "tipo": "bool",
     "desc": "Usar formas diferentes por classe nos graficos de score", "opcoes": None},
    {"key": "figuras_mostrar_elipses", "attr": "mostrar_elipses_grupo", "tipo": "bool",
     "desc": "Desenhar elipses de confianca por grupo", "opcoes": None},
    {"key": "formato_figura", "attr": "formato_saida", "tipo": "choice",
     "desc": "Formato das figuras", "opcoes": ["png", "pdf", "svg"]},
    {"key": "dpi", "attr": "dpi_salvar", "tipo": "int",
     "desc": "Resolucao das figuras (DPI)", "opcoes": None},
    {"key": "abrir_figuras_na_tela", "attr": "mostrar_graficos", "tipo": "bool",
     "desc": "Abrir cada figura na tela ao gerar (alem de salvar)", "opcoes": None},
]


def _attr_para_yaml(spec: Dict[str, Any], cfg: Config) -> Any:
    """Le o atributo da Config e converte para a forma amigavel do YAML."""
    v = getattr(cfg, spec["attr"])
    if spec["tipo"] == "preproc":
        return _PRE_PROC_INV.get(str(v).lower(), str(v))
    if spec["tipo"] == "list":
        return list(v)
    if spec["tipo"] == "str_opcional":
        return "" if v is None else str(v)
    return v


def _coagir_valor(spec: Dict[str, Any], val: Any) -> Any:
    """Converte um valor do YAML/menu para o tipo correto do atributo Config,
    validando opcoes. Lanca ValueError com mensagem amigavel se invalido."""
    t = spec["tipo"]
    if t == "str_opcional":
        s2 = str(val).strip()
        return s2 if s2 else None
    if t == "bool":
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in ("true", "sim", "1", "yes", "s", "v")
    if t == "int":
        return int(val)
    if t == "float":
        return float(val)
    if t == "list":
        if val is None:
            return ()
        if isinstance(val, (list, tuple)):
            return tuple(str(x).strip() for x in val if str(x).strip())
        return tuple(x.strip() for x in str(val).split(",") if x.strip())
    if t == "choice":
        sv = str(val)
        if spec["opcoes"] and sv not in spec["opcoes"]:
            raise ValueError(f"valor '{sv}' invalido; use {spec['opcoes']}")
        return sv
    if t == "preproc":
        sv = str(val)
        if sv in _PRE_PROC_FRIENDLY:
            return _PRE_PROC_FRIENDLY[sv]
        if sv.lower() in _PRE_PROC_INV:
            return sv.lower()
        raise ValueError(
            f"pre-processamento '{sv}' invalido; use {list(_PRE_PROC_FRIENDLY)}")
    return str(val)


def _fmt_yaml(v: Any) -> str:
    """Formata um valor Python como YAML simples (para o arquivo comentado).
    Usa ASPAS SIMPLES quando precisa citar: em YAML, dentro de aspas simples a
    barra invertida e literal — essencial p/ caminhos do Windows (C:\\Users\\...).
    Em aspas duplas, '\\U' / '\\D' seriam escapes e quebrariam a leitura."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_yaml(x) for x in v) + "]"
    if isinstance(v, str):
        precisa = (v == "" or v != v.strip()
                   or v.lower() in ("true", "false", "null", "yes", "no", "~")
                   or any(c in v for c in ':#,[]{}&*!|>%@`"\''))
        if precisa:
            return "'" + v.replace("'", "''") + "'"
        return v
    return str(v)


def salvar_config(cfg: Config, caminho: str) -> None:
    """Escreve config.yaml em linguagem simples, com um comentario explicativo
    acima de cada campo. Regenera os comentarios a cada salvamento."""
    linhas = [
        "# " + "=" * 60,
        "#  CONFIGURACAO DO PIPELINE QUIMIOMETRICO",
        "#  Edite os valores abaixo. Cada campo tem uma explicacao no",
        "#  comentario logo acima. Voce NAO precisa abrir o codigo.",
        "#  Para rodar:   python pineline_quimiometria_14.py --rodar",
        "#  Assistente:   python pineline_quimiometria_14.py",
        "# " + "=" * 60,
        "",
    ]
    for s in _CONFIG_SPEC:
        linhas.append(f"# {s['desc']}")
        if s["opcoes"]:
            linhas.append(f"#   opcoes: {' | '.join(map(str, s['opcoes']))}")
        linhas.append(f"{s['key']}: {_fmt_yaml(_attr_para_yaml(s, cfg))}")
        linhas.append("")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))


def carregar_config(caminho: str, base: Optional[Config] = None) -> Config:
    """Le config.yaml e devolve uma Config. Mantem os defaults para chaves
    ausentes; ignora chaves desconhecidas; reune erros numa mensagem clara."""
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML nao instalado. Rode: pip install pyyaml")
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Config nao encontrado: {caminho}. Gere um pelo assistente "
            f"(opcao [S] salvar) antes de usar --rodar.")
    with open(caminho, "r", encoding="utf-8") as f:
        dados = yaml.safe_load(f) or {}
    cfg = base if base is not None else Config()
    spec_por_key = {s["key"]: s for s in _CONFIG_SPEC}
    erros: List[str] = []
    for key, val in dados.items():
        s = spec_por_key.get(key)
        if s is None:
            continue
        try:
            setattr(cfg, s["attr"], _coagir_valor(s, val))
        except Exception as e:
            erros.append(f"  - '{key}': {e}")
    if erros:
        raise ValueError("Problemas no config.yaml:\n" + "\n".join(erros))
    return cfg


def _validar_pasta_dados(cfg: Config) -> Tuple[bool, str]:
    """Checagem amigavel da fonte de dados, ciente do modo de entrada.
    Generico: serve para .dx (FT-NIR), CSV (qualquer dado tabular) ou
    dados sinteticos de teste."""
    modo = getattr(cfg, "modo", "dx")
    if modo == "sintetico":
        return True, "OK — modo sintetico (dados gerados em memoria, sem arquivo)"
    if modo == "csv":
        cam = cfg.arquivo_csv
        if not cam or not os.path.isfile(cam):
            return False, f"CSV nao encontrado: '{cam}' (confira o caminho)"
        return True, f"OK — CSV: {os.path.basename(cam)}"
    # modo dx (padrao)
    p = cfg.pasta_entrada
    if not p or not os.path.isdir(p):
        return False, f"pasta nao encontrada: '{p}' (confira o caminho)"
    n_dx = len(glob.glob(os.path.join(p, "**", "*.dx"), recursive=True))
    if n_dx == 0:
        return False, f"nenhum arquivo .dx em '{p}' (nem nas subpastas)"
    return True, f"OK — {n_dx} arquivos .dx encontrados"


def _editar_campo(cfg: Config, s: Dict[str, Any]) -> None:
    """Edita um campo via terminal, com validacao."""
    print(f"\n  {s['desc']}")
    print(f"  valor atual: {_fmt_yaml(_attr_para_yaml(s, cfg))}")
    if s["tipo"] == "bool":
        setattr(cfg, s["attr"], not bool(getattr(cfg, s["attr"])))
        print(f"  -> alterado para {getattr(cfg, s['attr'])}")
        return
    if s["tipo"] in ("choice", "preproc"):
        ops = s["opcoes"] or []
        for i, o in enumerate(ops, 1):
            print(f"    ({i}) {o}")
        r = input("  escolha o numero (Enter cancela): ").strip()
        if r.isdigit() and 1 <= int(r) <= len(ops):
            try:
                setattr(cfg, s["attr"], _coagir_valor(s, ops[int(r) - 1]))
                print("  -> ok")
            except Exception as e:
                print(f"  erro: {e}")
        else:
            print("  cancelado.")
        return
    novo = input("  novo valor (Enter cancela): ").strip()
    if novo == "":
        print("  cancelado.")
        return
    try:
        setattr(cfg, s["attr"], _coagir_valor(s, novo))
        print("  -> ok")
    except Exception as e:
        print(f"  erro: {e}")


def menu_interativo(cfg: Optional[Config] = None,
                    caminho_cfg: str = "config.yaml") -> None:
    """Assistente de terminal: visualiza/edita a configuracao, salva/carrega
    o config.yaml e dispara o pipeline — tudo sem editar o codigo."""
    cfg = cfg if cfg is not None else Config()
    if os.path.exists(caminho_cfg):
        try:
            cfg = carregar_config(caminho_cfg, base=cfg)
            print(f"[config] carregado de {caminho_cfg}")
        except Exception as e:
            print(f"[config] nao foi possivel carregar ({e}). Usando padroes.")

    while True:
        print("\n" + "=" * 62)
        print("  ASSISTENTE DO PIPELINE QUIMIOMETRICO")
        print("=" * 62)
        for i, s in enumerate(_CONFIG_SPEC, 1):
            print(f"  [{i:2d}] {s['key']:<28s}: "
                  f"{_fmt_yaml(_attr_para_yaml(s, cfg))}")
        ok, msg = _validar_pasta_dados(cfg)
        print("-" * 62)
        print(f"  dados: {msg}")
        print("-" * 62)
        print("  digite o NUMERO p/ editar  |  [S]alvar  [L]carregar  "
              "[R]odar  [Q]sair")
        escolha = input("  > ").strip().lower()

        if escolha in ("q", "sair", "quit"):
            print("  encerrado."); return
        if escolha in ("s", "salvar"):
            salvar_config(cfg, caminho_cfg)
            print(f"  salvo em {caminho_cfg}"); continue
        if escolha in ("l", "carregar"):
            try:
                cfg = carregar_config(caminho_cfg, base=cfg)
                print("  recarregado.")
            except Exception as e:
                print(f"  erro: {e}")
            continue
        if escolha in ("r", "rodar", "run"):
            ok, msg = _validar_pasta_dados(cfg)
            if not ok:
                print(f"  [!] {msg}. Corrija antes de rodar."); continue
            salvar_config(cfg, caminho_cfg)
            print("  iniciando pipeline...\n")
            executar(cfg); return
        if escolha.isdigit() and 1 <= int(escolha) <= len(_CONFIG_SPEC):
            _editar_campo(cfg, _CONFIG_SPEC[int(escolha) - 1])
        else:
            print("  opcao invalida.")


if __name__ == "__main__":
    import sys
    _CFG_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    if "--rodar" in sys.argv:
        # Modo direto: usa config.yaml se existir, senao a Config do codigo.
        _cfg = carregar_config(_CFG_PATH) if os.path.exists(_CFG_PATH) else CFG
        executar(_cfg)
    elif "--codigo" in sys.argv:
        executar(CFG)                       # modo legado (Config embutida)
    elif sys.stdin is not None and sys.stdin.isatty():
        menu_interativo(CFG, _CFG_PATH)     # assistente (default interativo)
    else:
        _cfg = carregar_config(_CFG_PATH) if os.path.exists(_CFG_PATH) else CFG
        executar(_cfg)
