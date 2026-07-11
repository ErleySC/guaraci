"""config_io.py — Serialização, validação e schema (_CONFIG_SPEC) da Config.

Extraído de pipeline.py (dívida técnica pós-auditoria: pipeline.py era fachada
+ orquestrador + IO de config num arquivo só). Este módulo concentra a
"fonte única de verdade" da configuração — o _CONFIG_SPEC (mapeia campo
amigável <-> atributo da Config, alimenta YAML/menu/app web) e as funções de
ler/gravar/validar/coagir. Depende só de config.Config; nenhuma dependência
do motor (executar) — por isso não há import circular.

pipeline.py reexporta todos estes nomes, então `pipeline._CONFIG_SPEC`,
`pipeline.carregar_config(...)`, `pq._coagir_valor(...)` etc. seguem
funcionando sem alteração (o menu de terminal legado em pipeline.py e o app
web consomem daqui via a fachada).
"""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

from guaraci.config import Config

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
     "desc": "Origem dos dados: dx (JCAMP-DX, FT-NIR) | csv (tabela generica) | "
             "imagem (colorimetria digital, prototipo) | sintetico (teste)",
     "opcoes": ["dx", "csv", "imagem", "sintetico"]},
    {"key": "pasta_dados", "attr": "pasta_entrada", "tipo": "str",
     "desc": "Pasta com os arquivos .dx OU imagens (modo dx/imagem; uma subpasta por classe)",
     "opcoes": None},
    {"key": "imagem_incluir_textura", "attr": "imagem_incluir_textura", "tipo": "bool",
     "desc": "Modo imagem: incluir features de textura (GLCM) alem de cor "
             "(media/desvio RGB+HSV+Lab) — requer 'pip install scikit-image'",
     "opcoes": None},
    {"key": "arquivo_csv", "attr": "arquivo_csv", "tipo": "str",
     "desc": "Caminho do CSV (modo csv): colunas espectrais/variaveis + 1 coluna de classe", "opcoes": None},
    {"key": "coluna_classe", "attr": "coluna_classe", "tipo": "str",
     "desc": "Nome da coluna de classe/rotulo no CSV (modo csv)", "opcoes": None},
    {"key": "coluna_concentracao", "attr": "coluna_conc", "tipo": "str_opcional",
     "desc": "Nome da coluna de concentracao no CSV (vazio se nao houver; modo csv)", "opcoes": None},
    {"key": "pasta_saida", "attr": "pasta_saida_raiz", "tipo": "str",
     "desc": "Pasta onde os resultados serao gravados", "opcoes": None},
    {"key": "nivel", "attr": "nivel", "tipo": "choice",
     "desc": "Modo de analise: Classificacao (especie) | Discriminacao "
             "(puro vs. adulterado) | Quantificacao (teor de adulterante)",
     "opcoes": ["N1", "N2", "N3"]},
    {"key": "objetivo", "attr": "objetivo", "tipo": "choice",
     "desc": "Objetivo cientifico que filtra QUAIS figuras/relatorios sao "
             "gerados: auto (deriva do nivel — N1/N2=Classificacao, "
             "N3=Quantificacao) | exploratorio (so PCA/HCA/loadings/pre-proc, "
             "sem PLS-DA) | classificacao (PLS-DA e derivados) | "
             "quantificacao (regressao PLS + figuras de merito)",
     "opcoes": ["auto", "exploratorio", "classificacao", "quantificacao"]},
    {"key": "pre_processamento", "attr": "preprocessamento_padrao", "tipo": "preproc",
     "desc": "Pre-processamento espectral", "opcoes": list(_PRE_PROC_FRIENDLY)},
    {"key": "faixa_min_cm", "attr": "wn_min", "tipo": "float",
     "desc": "Inicio da faixa espectral util (cm-1)", "opcoes": None, "min": 0.0},
    {"key": "faixa_max_cm", "attr": "wn_max", "tipo": "float",
     "desc": "Fim da faixa espectral util (cm-1)", "opcoes": None, "min": 0.0},
    {"key": "excluir_classes", "attr": "excluir_classes", "tipo": "list",
     "desc": "Especies a remover da analise (ex: [Copaiba])", "opcoes": None},
    {"key": "max_lvs", "attr": "max_lvs", "tipo": "int",
     "desc": "Numero maximo de variaveis latentes (LVs) testadas", "opcoes": None,
     "min": 1, "max": 200},
    {"key": "holdout_fracao", "attr": "frac_holdout", "tipo": "float",
     "desc": "Fracao reservada para teste externo (0 a 0.5)", "opcoes": None,
     "min": 0.0, "max": 0.5},
    {"key": "validacao_group_aware", "attr": "agrupar_por_mae_id", "tipo": "bool",
     "desc": "Manter replicas (T1/T2/T3) juntas na validacao (evita vazamento)", "opcoes": None},
    {"key": "n_permutacoes", "attr": "n_permutacoes", "tipo": "int",
     "desc": "Iteracoes do teste de permutacao", "opcoes": None, "min": 1, "max": 100000},
    {"key": "teste_wold", "attr": "executar_wold", "tipo": "bool",
     "desc": "Rodar teste de Wold (intercepts R2Y/Q2Y)", "opcoes": None},
    {"key": "n_jobs_permutacao", "attr": "n_jobs_permutacao", "tipo": "int",
     "desc": "Processos paralelos para os testes de permutacao/Wold "
             "(1 = sequencial, resultado identico; so muda o tempo). "
             "Medido: 4 processos = ~2x mais rapido no pipeline completo; "
             "acima disso o ganho cai (overhead de criar processos). "
             "Use 1 em ambientes com pouca RAM/CPU (ex.: Streamlit Cloud gratuito)",
     "opcoes": None, "min": 1, "max": 64},
    {"key": "teste_cv_anova", "attr": "executar_cv_anova", "tipo": "bool",
     "desc": "Rodar CV-ANOVA (Eriksson)", "opcoes": None},
    {"key": "teste_martens", "attr": "executar_martens", "tipo": "bool",
     "desc": "Teste de incerteza de Martens: jackknifing dos coeficientes "
             "PLS, p-valor de significancia por variavel", "opcoes": None},
    {"key": "selecao_variaveis_etapa4", "attr": "executar_etapa4", "tipo": "bool",
     "desc": "Rodar Etapa 4 (iPLS / VIP / SR / sPLS-DA)", "opcoes": None},
    {"key": "selecao_spa", "attr": "executar_spa", "tipo": "bool",
     "desc": "Etapa 4: rodar tambem SPA/APS (Algoritmo das Projecoes "
             "Sucessivas, Araujo et al. 2001) — mais lento que iPLS/VIP/SR",
     "opcoes": None},
    {"key": "selecao_ag", "attr": "executar_ag", "tipo": "bool",
     "desc": "Etapa 4: rodar tambem AG (Algoritmo Genetico, GA-PLS) — o mais "
             "lento dos metodos de selecao de variaveis (populacao x geracoes "
             "avaliacoes de CV)",
     "opcoes": None},
    {"key": "ddsimca", "attr": "executar_ddsimca", "tipo": "bool",
     "desc": "Rodar DD-SIMCA (classificacao one-class)", "opcoes": None},
    {"key": "modo_ddsimca", "attr": "ddsimca_treinar_em", "tipo": "choice",
     "desc": "Modo de treino do DD-SIMCA: 'puros' treina SO com amostras puras "
             "(o resto conta como contaminante/adulterado -- autenticacao de "
             "verdade); 'todos' treina com toda a classe (exploratorio, mais "
             "robusto com poucas amostras puras, porem menos rigoroso)",
     "opcoes": ["puros", "todos"]},
    {"key": "opls_da", "attr": "executar_opls", "tipo": "bool",
     "desc": "Rodar OPLS-DA", "opcoes": None},
    {"key": "comparar_pre_processamentos", "attr": "comparar_pipelines", "tipo": "bool",
     "desc": "Comparar varios pre-processamentos", "opcoes": None},
    {"key": "benchmark", "attr": "executar_benchmark", "tipo": "bool",
     "desc": "Auto-Benchmark: SVM RBF / RF / XGBoost vs PLS-DA (mesma CV group-aware)", "opcoes": None},
    {"key": "benchmark_regressao", "attr": "executar_benchmark_regressao", "tipo": "bool",
     "desc": "Auto-Benchmark de regressao: Ridge/Lasso/Elastic Net/SVR/RF vs PLS-R "
             "(N2/N3, por especie, mesmo split cal/val)", "opcoes": None},
    {"key": "monte_carlo", "attr": "executar_monte_carlo", "tipo": "bool",
     "desc": "Monte Carlo CV: IC95% por percentil (N repeticoes estratificadas por grupo)", "opcoes": None},
    {"key": "n_monte_carlo", "attr": "n_monte_carlo", "tipo": "int",
     "desc": "Numero de repeticoes do Monte Carlo CV", "opcoes": None, "min": 1, "max": 100000},
    {"key": "monte_carlo_incluir_todos", "attr": "monte_carlo_incluir_todos", "tipo": "bool",
     "desc": "MC CV: incluir SVM RBF / RF / XGBoost alem do PLS-DA (mais lento)", "opcoes": None},
    {"key": "shap_benchmark", "attr": "executar_shap", "tipo": "bool",
     "desc": "SHAP values (TreeExplainer) para RF/XGBoost/GBM — interpretabilidade espectral", "opcoes": None},
    {"key": "shap_max_amostras", "attr": "shap_max_amostras", "tipo": "int",
     "desc": "Limite de amostras para calculo de SHAP (controle de memoria)", "opcoes": None,
     "min": 1, "max": 1000000},
    {"key": "figuras_detalhadas", "attr": "figuras_detalhadas", "tipo": "bool",
     "desc": "Gerar tambem as figuras exploratorias/detalhadas (HCA, loadings PCA, "
             "pre-processamento, contribuicao de score, DD-SIMCA por classe, Cooman). "
             "Desligado = so o conjunto essencial (mais rapido, menos arquivos)",
     "opcoes": None},
    {"key": "figuras_mostrar_marcadores", "attr": "mostrar_marcadores_classe", "tipo": "bool",
     "desc": "Usar formas diferentes por classe nos graficos de score", "opcoes": None},
    {"key": "figuras_mostrar_elipses", "attr": "mostrar_elipses_grupo", "tipo": "bool",
     "desc": "Desenhar elipses de confianca por grupo", "opcoes": None},
    {"key": "formato_figura", "attr": "formato_saida", "tipo": "choice",
     "desc": "Formato das figuras", "opcoes": ["png", "pdf", "svg"]},
    {"key": "dpi", "attr": "dpi_salvar", "tipo": "int",
     "desc": "Resolucao das figuras (DPI)", "opcoes": None, "min": 50, "max": 1200},
    {"key": "abrir_figuras_na_tela", "attr": "mostrar_graficos", "tipo": "bool",
     "desc": "[Nao disponivel: o backend grafico e sempre headless/Agg, por "
             "estabilidade em execucao paralela e no servidor web] Figuras "
             "continuam sendo sempre salvas em disco normalmente",
     "opcoes": None},
]


def _attr_para_yaml(spec: Dict[str, Any], cfg: Config) -> Any:
    """Le o atributo da Config e converte para a forma amigavel do YAML."""
    v = getattr(cfg, spec["attr"], getattr(Config(), spec["attr"], None))
    if spec["tipo"] == "preproc":
        return _PRE_PROC_INV.get(str(v).lower(), str(v))
    if spec["tipo"] == "list":
        return list(v) if v is not None else []
    if spec["tipo"] == "str_opcional":
        return "" if v is None else str(v)
    return v


def _checar_faixa(spec: Dict[str, Any], v: float) -> float:
    """Valida um numero contra 'min'/'max' opcionais do spec, com mensagem
    amigavel. Impede que valores impossiveis (fracao negativa, contagem zero)
    passem silenciosamente e so quebrem — ou pior, distorcam o resultado —
    no meio da execucao."""
    lo, hi = spec.get("min"), spec.get("max")
    if lo is not None and v < lo:
        raise ValueError(
            f"{spec['key']}={v}: valor minimo permitido e {lo}")
    if hi is not None and v > hi:
        raise ValueError(
            f"{spec['key']}={v}: valor maximo permitido e {hi}")
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
        return int(_checar_faixa(spec, int(val)))
    if t == "float":
        return _checar_faixa(spec, float(val))
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


def _validar_semantico(cfg: "Config") -> List[str]:
    """Validacoes CRUZADAS entre campos, que 'min'/'max' isolados nao pegam.

    Retorna lista de mensagens amigaveis (vazia = tudo certo). Deve ser chamada
    ANTES de rodar (app e CLI) para falhar cedo, com mensagem clara, em vez de
    quebrar so no meio da execucao (depois de minutos carregando/processando).
    """
    erros: List[str] = []

    # Faixa espectral: o inicio precisa ser menor que o fim. Se invertido, o
    # filtro [wn_min, wn_max] remove TODAS as variaveis e o pipeline so
    # quebraria depois de carregar os dados (crash tardio, confuso).
    try:
        if float(cfg.wn_min) >= float(cfg.wn_max):
            erros.append(
                f"Faixa espectral invalida: o inicio ({cfg.wn_min:.0f}) deve "
                f"ser MENOR que o fim ({cfg.wn_max:.0f}) cm-1.")
    except (TypeError, ValueError):
        pass

    # Holdout: precisa sobrar treino suficiente. 'max'=0.5 ja barra pelo widget,
    # mas um config.yaml editado a mao pode trazer valor fora da faixa — aqui e
    # o backstop para esse caminho (Config cru, sem passar por _coagir_valor).
    try:
        fh = float(cfg.frac_holdout)
        if fh < 0.0 or fh > 0.5:
            erros.append(
                f"Fracao de holdout invalida ({fh}): use um valor entre 0 e 0.5 "
                "(0 = sem teste externo).")
    except (TypeError, ValueError):
        pass

    return erros


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
        "#  Para rodar:   python pipeline.py --rodar",
        "#  Assistente:   python guaraci.py",
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
    if modo == "imagem":
        p_img = cfg.pasta_entrada
        if not p_img or not os.path.isdir(p_img):
            return False, f"pasta nao encontrada: '{p_img}' (confira o caminho)"
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        n_img = sum(len(glob.glob(os.path.join(p_img, "**", f"*{e}"), recursive=True))
                    for e in exts)
        if n_img == 0:
            return False, f"nenhuma imagem em '{p_img}' (nem nas subpastas)"
        return True, f"OK — {n_img} imagens encontradas"
    # modo dx (padrao)
    p = cfg.pasta_entrada
    if not p or not os.path.isdir(p):
        return False, f"pasta nao encontrada: '{p}' (confira o caminho)"
    n_dx = len(glob.glob(os.path.join(p, "**", "*.dx"), recursive=True))
    if n_dx == 0:
        return False, f"nenhum arquivo .dx em '{p}' (nem nas subpastas)"
    return True, f"OK — {n_dx} arquivos .dx encontrados"
