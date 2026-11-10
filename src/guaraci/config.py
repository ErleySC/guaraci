"""config.py — Objeto de configuração central do pipeline (dataclass Config).

Extraído de pipeline.py (item 10 da auditoria) para isolar o schema de
configuração da lógica de execução. É uma dataclass PURA (só campos com
defaults, sem métodos e sem dependência de pipeline/numpy/sklearn), então
pode ser importada por qualquer módulo sem risco de import circular.

pipeline.py reexporta `Config`, então `pipeline.Config` e
`from pipeline import Config` continuam funcionando sem alteração.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any, Dict, List

__all__ = [
    "__version__",
    "NOME_GRAFICOS",
    "NOME_TABELAS",
    "NOME_RELATORIOS",
    "NOME_MODELOS",
    "Config",
]

# Versão canônica do pacote. Mora aqui (módulo sem dependências) para ser
# importável por qualquer módulo — pipeline.py e guaraci/__init__.py a
# reexportam, então `pipeline.__version__` e `from guaraci import __version__`
# seguem funcionando. Fonte ÚNICA: nunca duplicar um literal "vXX.Y".
__version__ = "1.0.0"

# Identidade do projeto: FORMA UNICA do nome do autor, ORCID e licenca. Usadas por `--version`,
# pela tela Sobre (CLI e web) e conferidas contra pyproject.toml/CITATION.cff por teste.
AUTOR_NOME = "Erley Silva da Costa"
AUTOR_ORCID = "0009-0005-9655-6349"
LICENCA_SPDX = "GPL-3.0-or-later"
# Linha de copyright: ano do primeiro commit do historico ate o ano corrente (um ano so' quando coincidem).
COPYRIGHT_ANO_INICIAL = 2026
COPYRIGHT_ANO_ATUAL = 2026
_ANOS = (str(COPYRIGHT_ANO_INICIAL) if COPYRIGHT_ANO_INICIAL == COPYRIGHT_ANO_ATUAL
         else f"{COPYRIGHT_ANO_INICIAL}–{COPYRIGHT_ANO_ATUAL}")
COPYRIGHT_LINHA = f"Copyright (C) {_ANOS} {AUTOR_NOME}"
# Onde o projeto vive. FONTE UNICA no codigo: se o repositorio mudar de nome, troque so' aqui e rode
# `python scripts/gerar_docs_pacote.py`; os documentos (.md/.toml/.cff/.yml/.ipynb) repetem a URL e
# `tests/test_identidade_e_contato.py` lista e confere todos os lugares.
AUTOR_GITHUB_URL = "https://github.com/ErleySC"
REPO_URL = f"{AUTOR_GITHUB_URL}/guaraci"
DISCUSSIONS_URL = f"{REPO_URL}/discussions"
SEGURANCA_URL = f"{REPO_URL}/security/advisories/new"   # canal privado do GitHub (relato de vulnerabilidade)

# Nome amigavel de cada nivel de analise (valor interno N1/N2/N3 inalterado).
_NIVEL_NOME = {
    "N1": "Classificacao por especie",
    "N2": "Discriminacao puro/adulterado",
    "N3": "Quantificacao de teor",
}

# Slug curto p/ NOME DE PASTA de saida (auditoria de 2026-07-12, P8 residual:
# o nome de pasta ainda expunha N1/N2/N3 cru, ex. PLSDA_OE_N2_..., mesmo com
# os menus ja mostrando o nome amigavel). cfg.level continua sendo "N1"/"N2"/
# "N3" internamente -- so' o SLUG usado no nome do diretorio muda; nao afeta
# nenhuma logica de negocio, so' o texto do caminho gravado em disco.
_NIVEL_SLUG_PASTA = {
    "N1": "PorEspecie",
    "N2": "Autenticacao",
    "N3": "Quantificacao",
}

# Nomes de subpasta da saida de cada execucao (auditoria jul/2026, item 4:
# reestruturacao Amostra/Modo/{Graficos,Tabelas,Relatorios}). Fonte UNICA —
# writers (pipeline.py/figuras.py/avaliacao_modelos.py/selecao_variaveis.py)
# e readers (reports.py/app_logic.py/app_tabs/) importam daqui, nunca com a
# string literal, para que um rename futuro seja de 1 lugar so.
NOME_GRAFICOS   = "Graficos"     # antigo "figuras"
NOME_TABELAS    = "Tabelas"      # antigo "dados"
NOME_RELATORIOS = "Relatorios"   # antigo "logs"
NOME_MODELOS    = "Modelos"      # antigo "modelos" (so' capitalizacao)

#: Subpastas da execucao no idioma da sessao (pedido de 2026-11-07): nome canonico (os NOME_*
#: acima, em PT) -> (PT, EN). Grava-se no idioma da sessao e le-se aceitando os DOIS nomes, para
#: que uma rodada antiga, ou feita no outro idioma, continue abrindo (modelo, resumo, tabelas).
PASTAS_EXECUCAO: Dict[str, Tuple[str, str]] = {
    NOME_GRAFICOS: ("Graficos", "Figures"),
    NOME_TABELAS: ("Tabelas", "Tables"),
    NOME_RELATORIOS: ("Relatorios", "Reports"),
    NOME_MODELOS: ("Modelos", "Models"),
    # ferramentas avulsas (anova, vip, monitorar, transferir): subpastas e pasta de saida padrao
    "tabelas": ("tabelas", "tables"),
    "figuras": ("figuras", "figures"),
    "relatorio": ("relatorio", "report"),
    "resultados": ("resultados", "results"),          # pasta_saida dos exemplos
    "resultados_anova": ("resultados_anova", "anova_results"),
    "resultados_vip": ("resultados_vip", "vip_results"),
    "resultados_monitoramento": ("resultados_monitoramento", "monitoring_results"),
    "resultados_transferencia": ("resultados_transferencia", "transfer_results"),
}


def nome_pasta(chave: str, lang: Optional[str] = None) -> str:
    """Nome da pasta `chave` no idioma da sessao (para quem so' cria, ex.: padrao de --saida)."""
    from guaraci.idioma_relatorio import idioma
    pt, en = nomes_pasta_execucao(chave)
    return en if idioma(lang) == "en" else pt


def nomes_pasta_execucao(chave: str) -> Tuple[str, str]:
    """(PT, EN) da subpasta `chave` (um dos NOME_*)."""
    return PASTAS_EXECUCAO.get(chave, (chave, chave))


def pasta_execucao(raiz: str, chave: str, lang: Optional[str] = None) -> str:
    """Caminho da subpasta `chave` (NOME_GRAFICOS, NOME_TABELAS...) da execucao em `raiz`.

    A que ja' existe vence, em qualquer idioma (quem le e quem completa uma rodada antiga caem na
    mesma pasta); se nenhuma existe, o nome no idioma da sessao (quem cria uma rodada nova)."""
    pt, en = nomes_pasta_execucao(chave)
    for nome in (pt, en):
        candidata = os.path.join(raiz, nome)
        if os.path.isdir(candidata):
            return candidata
    from guaraci.idioma_relatorio import idioma
    return os.path.join(raiz, en if idioma(lang) == "en" else pt)


@dataclass
class Config:
    mode: str = "dx"                          # "dx" | "csv" | "imagem" | "sintetico" | "hsi"

    # ---- INPUT ----
    # Root folder with subfolders per species (each subfolder = 1 class) OR
    # single folder with .dx files (legacy mode). Auto-detected.
    # DO NOT version personal paths: the real path goes in config.yaml
    # (gitignored). This default is just a portable example.
    input_folder: str = r"dados"
    class_part: int = 0                     # fallback if no subfolders
    extract_conc_from_filename: bool = True        # only used in old fallback
    csv_file: str = "seus_espectros.csv"
    class_column: str = "classe"
    conc_column: Optional[str] = None
    # Fonte de GRUPO (amostra fisica independente) para modo csv e para os
    # leitores de instrumento: sem ela nao ha'
    # validacao agrupada e a CV cai em StratifiedKFold -- declarado, nunca
    # silencioso. `group_column` = coluna do proprio CSV; `group_file` =
    # CSV/TSV (id, grupo) -- ver dados_io.carregar_mapa_grupos.
    group_column: Optional[str] = None
    group_file: Optional[str] = None
    # Unidade independente da validacao agrupada (B4c): "auto" (padrao: vira
    # "sessao" so' quando cada sessao tem uma classe so' e validar por sessao e'
    # possivel; senao "mae_id"), "mae_id" ou "sessao" (session_from_mae_id --
    # p.ex. diluicoes da mesma amostra-base sao 1 unidade). Ver MANUAL.md.
    group_unit: str = "auto"
    # Holdout agrupado ESTRATIFICADO por classe:
    # o GroupShuffleSplit sorteava GRUPOS sem olhar a classe e, em 200 de 200
    # sementes, pelo menos uma classe ficava fora do holdout sem aviso (resumo
    # com bal.acc 1,0 ao lado de F1 0,5). False = comportamento legado (so'
    # para reproduzir numeros anteriores a 2026-09-25).
    holdout_estratificado: bool = True
    # Correcoes de metadado LOCAIS: CSVs na pasta local de correcoes
    # (`dados_io._DIR_LOCAL`, no diretorio do usuario) que reclassificam rotulos/mae_id/pureza. Antes eram
    # lidos NA IMPORTACAO e aplicados sem rastro: o mesmo dado bruto dava
    # rotulos diferentes em outra maquina. Agora so' valem com opcao EXPLICITA
    # (esta flag ou GUARACI_APLICAR_CORRECOES_LOCAIS=1) e o que foi aplicado
    # (arquivo, sha256, n de amostras alteradas) vai para o resumo, o model
    # card e o manifesto.
    apply_local_corrections: bool = False
    correcoes_locais_aplicadas: Dict[str, Any] = field(default_factory=dict)
    use_parse_title: bool = True             # uses ##TITLE= from JCAMP-DX
    # mode="imagem" (colorimetria digital, prototipo): extrai estatisticas de
    # cor (RGB/HSV/Lab) de cada imagem, reaproveitando input_folder (mesma
    # convencao de 1 subpasta por classe do mode dx). Use preprocessamento_
    # padrao="autoscaling" (NAO "msc_sg_mc"/"snv_sg_mc" — SG exige janela <=
    # n_variaveis e MSC/SG pressupoem sinal espectral continuo, sem sentido
    # p/ um vetor curto de estatisticas de cor) e ajuste wn_min/wn_max p/
    # cobrir o eixo simbolico 0..n_features-1 (ver dados_imagem.py).
    image_crop: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    include_image_texture: bool = False       # GLCM — requer scikit-image
    # R12 (fase 2): normalizacao por cartao cinza/branco e segmentacao do objeto.
    # image_reference = (x0, y0, x1, y1) relativos da regiao do cartao; () = desligado.
    image_reference: Tuple[float, ...] = ()
    image_segmentation: str = "nenhuma"       # "nenhuma" | "otsu" | "distancia_fundo"
    # mode="hsi" (imageamento HIPERESPECTRAL, prototipo "minimo viavel",
    # Passos 92-102 da INSTRUCAO_HSI_MINIMO_VIAVEL.md): DISTINTO do mode
    # "imagem" acima -- "imagem" e' colorimetria de foto comum (1 vetor de
    # features por foto, reusa pipeline.executar() sem alteracao); "hsi" e'
    # POR PIXEL de um cubo hiperespectral (ENVI .hdr+.bin), com agregacao
    # por objeto fisico -- forma de dado diferente, tem sua propria
    # orquestracao em hsi_pipeline.run_hsi_pipeline (nao usa
    # pipeline.executar()). hsi_dataset_folder aponta para a pasta de
    # cubos ENVI (uma subpasta por classe) --
    # nao reusa input_folder para nao confundir com o mode "imagem"/"dx".
    hsi_dataset_folder: str = ""
    # R11 (fase 2): calibracao branco/escuro (R = (I-D)/(W-D)), eixo de bandas entre
    # cubos e validacao por objeto. Padroes = comportamento anterior, exceto
    # hsi_validation (CV por objeto no lugar do holdout unico; "holdout" volta ao antigo).
    hsi_calibration: str = "nenhuma"          # "nenhuma" | "branco_escuro"
    hsi_white_ref: str = ""                   # arquivo/pasta do WHITEREF ("" = procurar ao lado dos cubos)
    hsi_dark_ref: str = ""                    # idem DARKREF
    hsi_band_axis: str = "recusar"            # "recusar" | "reamostrar" (eixos de bandas diferentes)
    hsi_validation: str = "cv_objetos"        # "cv_objetos" | "holdout"
    hsi_cv_folds: int = 5                     # dobras da CV por objeto (0 = leave-one-object-out)
    # mode="hplc" (cromatograma bruto Agilent/Waters via `rainbow-api`,
    # `importadores_proprietarios.parse_cromatograma_hplc`): qual TIPO de
    # detector ler quando o diretorio tiver mais de um ("UV" p/ UV/DAD --
    # caso mais comum de HPLC --, "FID" p/ GC-FID). Reusa input_folder
    # (mesma convencao 1-subpasta-por-classe de dx/imagem).
    hplc_detector: str = "UV"
    # Canais DAD do cromatograma (Agilent/Waters via rainbow): "" = comportamento historico (1o canal do
    # detector); "todos" ou lista de comprimentos de onda ("254,280") = matriz tempo x comprimento de onda
    # achatada (um vetor por amostra). Ver `importadores_proprietarios.parse_cromatograma_hplc`.
    cromatografia_canal: str = ""
    # Teto de pontos do eixo comum. Espectros (dx/opus/spc/sp/rmn): 2000 (historico; reducao por media por
    # bin, com AVISO). Cromatogramas (hplc/gcms): 0 = sem limite (decimar fazia a altura do pico depender da
    # fase da amostragem). 0 = sem limite nos dois.
    espectro_max_pontos: int = 2000
    cromatograma_max_pontos: int = 0
    # mode="eem" (fluorescencia excitacao-emissao): tratamento do espalhamento Rayleigh/Raman de 1a e 2a ordem na leitura
    # (`eem_io.tratar_espalhamento`). "nenhum" = nao toca no dado (padrao: nao muda resultados existentes; a leitura avisa se a
    # EEM bruta tem espalhamento forte). Larguras TOTAIS em nm por ordem (Rayleigh 1a, Rayleigh 2a, Raman 1a, Raman 2a);
    # vazio = 30 nm para todas. PARAFAC: nao-negativo (HALS) por padrao; False = ALS sem restricao.
    eem_espalhamento: str = "nenhum"
    eem_espalhamento_largura_nm: Tuple[str, ...] = ()
    eem_raman: bool = False
    parafac_nao_negativo: bool = True
    # mode="multielementar" (ICP-OES / ICP-MS, tabela amostra x elemento,
    # `icp_multielementar.py`): reusa csv_file (arquivo), class_column (o
    # GRUPO a comparar: origem, produtor...) e group_column (a UNIDADE
    # independente, obrigatoria neste mode). Unidade de concentracao
    # obrigatoria; politica de censura declarada, com analise de
    # sensibilidade sempre feita com as duas politicas.
    icp_unidade: str = ""
    icp_coluna_amostra: str = "Amostra"
    icp_coluna_tipo: str = ""
    icp_coluna_sessao: str = ""
    icp_padrao_interno: str = ""
    icp_politica_censura: str = "substituir"
    icp_limiar_censura: float = 0.5
    icp_log: bool = False
    # Ajuste automatico por contexto (guaraci.contexto): so' compatibilidade, com aviso e reversao
    auto_adjust: bool = True
    # preenchido em tempo de execucao pelo relatorio multielementar (nao vem do YAML)
    multielementar_resumo: Dict[str, Any] = field(default_factory=dict)

    # ---- OUTPUT (auto-generated by generate_output_name) ----
    output_root_folder: str = "resultados_guaraci"
    level: str = "N1"                         # "N1" | "N2" | "N3"
    # Objetivo cientifico do run — controla QUAIS figuras/relatorios sao
    # gerados (ver modos_analise.py). "auto" (default) deriva do nivel
    # preservando o comportamento historico (N1/N2 -> classificacao,
    # N3 -> quantificacao). Valores explicitos: "exploratorio" |
    # "classificacao" | "quantificacao" sobrepoem a derivacao. O Modo
    # Exploratorio gera apenas PCA/HCA/loadings/pre-proc (sem PLS-DA).
    objective: str = "auto"                    # "auto" | "exploratorio" | "classificacao" | "quantificacao"
    tag:   str = ""                            # free label e.g. "pure", "soybean"
    output_folder: str = ""                     # DO NOT edit manually
    # Confianca do agrupamento group-aware desta execucao -- "high" (mae_id
    # real, ex.: .dx/sintetico), "medium" (associacao manual via CSV, modo
    # imagem), "none" (sem fonte de agrupamento, fallback StratifiedKFold).
    # DO NOT edit manually -- escrito por dados_io._leitor_imagem apos
    # detectar a estrutura de dados; default "high" cobre todo mode que
    # nao sobrescreve isto (dx/sintetico ja' tem mae_id real por natureza).
    grouping_guarantee: str = "high"
    # Ajustes de EXTRAS por RAM aplicados nesta execucao (M-03 do parecer de
    # 2026-09-25): lista de {parametro, pedido, efetivo, motivo}. DO NOT edit
    # manually -- escrito por hardware.auto_adjust_hardware_config. Nunca
    # contem parametro cientifico (n_splits_cv, permutacoes, repeticoes).
    ajustes_hardware: List[Dict[str, Any]] = field(default_factory=list)
    output_format: str = "png"                # png | pdf | svg
    save_dpi: int = 600
    show_plots: bool = False

    # ---- Score plot style (PCA / PLS-DA / OPLS) ----
    # Disable for "clean" scatter (color by class only, no shapes/contours).
    show_class_markers: bool = True    # True (padrao, M-18a): forma por classe alem da cor; False -> todos circulos 'o'
    show_group_ellipses:     bool = False   # False -> no T2/convex hull ellipse

    # ---- Figure detail level ----
    # False (default): gera apenas o conjunto ESSENCIAL de figuras (scores PCA/
    #   PLS-DA, confusao, ROC/AUC, selecao de LVs, SR+VIP, outliers T2/Q, e a
    #   aceitacao DD-SIMCA quando aplicavel). Mais rapido e menos arquivos.
    # True: gera tambem as figuras EXPLORATORIAS/detalhadas (HCA, loadings PCA,
    #   pre-processamento, contribuicao de score, DD-SIMCA por classe, Cooman).
    detailed_figures: bool = False

    # ---- Group-aware validation (Q1 differentiator) ----
    # When True and mae_id is available, uses GroupKFold/GroupShuffleSplit
    # so that T1/T2/T3 of the same physical point stay in the same fold/holdout.
    group_by_mae_id: bool = True

    # ---- Perfil da matriz ------------------------------------------------
    # Junta num so' lugar tudo que e' propriedade da MATRIZ (faixa do eixo,
    # pre-processamento padrao, unidade, vocabulario da saida) -- ver
    # `guaraci/perfis_matriz/*.yaml` e `perfil_matriz.py`. Trocar de matriz
    # e' trocar este nome, nunca editar codigo-fonte. Aceita tambem o
    # caminho de um YAML proprio. Nome desconhecido levanta erro com a
    # lista do que existe, em vez de cair num padrao errado em silencio.
    matrix_profile: str = "generico"

    # ---- Eixo e tecnica declarados pela tela [2.4] (R6) -------------------
    # Unidade do eixo espectral dos DADOS: "cm-1", "nm", "raman_shift", "min",
    # "ppm", "ms" ou "indice". Dela saem o rotulo do eixo das figuras, a
    # orientacao (decrescente so' para cm-1 e ppm) e a unidade da faixa no
    # resumo. "" = nao declarada: vale o perfil (matrix_profile) e, sem perfil,
    # o comportamento antigo (rotulo neutro; eixo decrescente por modo de
    # entrada). Nunca altera o calculo: so' rotulo e orientacao.
    axis_unit: str = ""
    # Chave da tecnica escolhida na tela [2.4] (ex. "raman"); so' informativa
    # (campo "Tecnica (perfil)" do resumo quando nenhum perfil a descreve).
    technique: str = ""

    # ---- Perfil de TECNICA DE AQUISICAO (Bloco 8a/5B, so' relevante p/
    # mode="imagem") -------------------------------------------------------
    # Dimensao INDEPENDENTE de matrix_profile: matrix_profile diz "isto e'
    # esta matriz" (vocabulario/faixa do relatorio); acquisition_profile diz "foi
    # fotografado com celular" (resolucao esperada/formatos aceitos/nivel de
    # garantia de agrupamento tipico -- so' informativo, nunca restritivo; a
    # garantia REAL vem da estrutura dos dados, ver dados_imagem.py). Os
    # dois podem ser combinados ao mesmo tempo -- por isso e' um campo
    # separado, nao uma reutilizacao de matrix_profile. "" (nao Optional,
    # mesma convencao de matrix_profile) = nao declarado -- so' significa
    # que a tela nao mostra a dica de resolucao/formato/garantia tipica.
    acquisition_profile: str = ""

    # ---- Modo de rotulo na QUANTIFICACAO ---------------------------------
    # "cego" (PADRAO): a calibracao por classe usa a classe PREDITA pelo
    #     classificador, nao a verdadeira. E' o unico mode que corresponde
    #     ao uso real -- quem manda uma amostra desconhecida nao sabe a
    #     classe dela, entao um R2 obtido com a classe verdadeira mede algo
    #     que o usuario nunca tera' em maos.
    # "controle": usa a classe verdadeira. Valido apenas para diagnostico
    #     interno (isolar erro de quantificacao do erro de classificacao).
    #     Numeros obtidos assim NAO representam o desempenho de uso.
    label_mode: str = "cego"
    # Rotulo usado para TREINAR o modelo de teor de cada classe (so' com mais de
    # uma classe; ver `pipeline.pls_regression_by_species`):
    # "previsto" (PADRAO, comportamento da v1.0): cada modelo de teor treina com
    #     as amostras que o classificador atribuiu a classe. Com classificacao
    #     fraca os modelos recebem amostras misturadas e poucas amostras cada.
    # "verdadeiro": treina com a classe VERDADEIRA e valida roteando cada amostra
    #     de validacao pela classe PREVISTA (o uso real). Opt-in: muda os numeros.
    quant_rotulo_treino: str = "previsto"

    # ---- Spectral truncation ---------------------------------------------
    # Padrao herdado do caso de uso de origem (FT-NIR difuso, 4000-10000
    # cm-1: remove ruido de borda de FFT, onde a derivada SG cria VIP
    # falso). Um perfil de tecnica sobrescreve estes valores quando eles
    # estao no default. Aplicado ANTES de qualquer pre-processamento.
    wn_min: float = 4000.0
    wn_max: float = 10000.0
    # Faixas EXCLUIDAS do eixo (ex.: agua e TSP em RMN), "a-b" na unidade do eixo.
    # Aplicadas logo depois do corte wn_min/wn_max e ANTES de qualquer pre-
    # processamento. Vazio no motor geral (nenhum resultado existente muda); a
    # tela [2.4] preenche para a tecnica `nmr`. Vai no pacote do modelo.
    exclude_ranges: Tuple[str, ...] = ()
    # Normalizacao por espectro ANTES do pre-processamento (opt-in): "nenhuma",
    # "rip" (divide pela intensidade maxima na janela `rip_window`, o RIP do IMS)
    # ou "area" (divide pela area do espectro). Sem estado: nao vaza entre dobras.
    # Saturacao do detector (teto de absorbancia): "auto" remove variaveis/espectros no teto e
    # registra; "avisar" so' avisa; "nenhuma" nao procura. Independe de `ajuste_automatico`.
    saturacao: str = "auto"
    normalization: str = "nenhuma"
    rip_window: Optional[str] = None

    # ---- Preprocessing ---------------------------------------------------
    # Quick preset. When != 'custom', overrides individual flags.
    #   'msc_sg_mc'   : MSC -> SG -> mean-centering (padrao: FT-NIR difuso
    #                   com espalhamento forte; ver comparar_pre_processamentos)
    #   'snv_sg_mc'   : SNV -> SG -> mean-centering (Rinnan et al.)
    #   'autoscaling' : StandardScaler only (good on subset, poor on full: 0.472)
    #   'mc'          : mean-centering only
    #   'airpls_sg_mc'     : AirPLS -> SG -> mean-centering (Raman)
    #   'airpls_snv_sg_mc' : AirPLS -> SNV -> SG -> mean-centering (Raman quantitativo)
    #   'custom'      : honors apply_snv / apply_sg / apply_mc below
    default_preprocessing: str = "msc_sg_mc"   # FT-NIR difuso c/ espalhamento

    apply_snv: bool = True
    apply_sg: bool = True
    sg_window: int = 25
    sg_polyorder: int = 2
    sg_deriv: int = 1
    apply_mc: bool = True
    # EMSC (Martens & Stark 1991) e OSC (Wold et al. 1998) -- so' tem efeito
    # com default_preprocessing='custom' (mesmo padrao de apply_snv/apply_sg/
    # apply_mc acima). No config.yaml: `pre_processamento: Personalizado` +
    # `etapas_pre_processamento: [...]` (propriedade no fim desta classe).
    apply_emsc: bool = False
    emsc_ordem_polinomial: int = 2
    apply_osc: bool = False
    osc_n_componentes: int = 1
    # AirPLS (Zhang, Chen & Liang 2010) -- correcao de linha de base,
    # proposta p/ fluorescencia de fundo em Raman (Passo 144/145, mesmo
    # padrao de apply_emsc/apply_osc acima: so' com default_preprocessing=
    # 'custom', ainda sem preset default -- ver veredito do portao de
    # aceite no docstring de AirPLS antes de usar como
    # padrao).
    apply_airpls: bool = False
    airpls_lam: float = 100.0
    # Remocao de raios cosmicos (Whitaker & Hayes 2018, z-score modificado
    # da primeira diferenca; `preprocessamento.RemocaoRaiosCosmicos`). Ao
    # contrario das etapas acima, vale para QUALQUER preset: quando ligada,
    # entra como primeira etapa do pre-processamento. Desligada por padrao
    # (nenhum resultado existente muda). Limiar 6 e janela 5 sao os valores
    # do artigo.
    apply_despike: bool = False
    despike_limiar: float = 6.0
    despike_janela: int = 5
    # Alinhamento de retencao COW (Nielsen, Carstensen & Smedsgaard 1998;
    # `alinhamento_retencao.AlinhadorCOW`) para cromatogramas. 'nenhum'
    # (PADRAO) nao mexe em nada; 'cow' entra como etapa do pre-processamento,
    # logo DENTRO de cada dobra: a referencia e' escolhida so' no treino da
    # dobra e a mesma referencia alinha o teste e as amostras novas.
    alinhamento: str = "nenhum"
    alinhamento_segmentos: int = 30
    alinhamento_folga: int = 10
    # PQN (Dieterle et al. 2006) -- normalizacao por quociente probabilistico,
    # robusta a diluicao. Proposta T6 da rodada multiagente 2026-09-10:
    # `cli_assistente.py` ja recomendava "SNV ou PQN + MC" p/ RMN, mas PQN
    # nao existia -- so' com default_preprocessing='custom' (mesmo padrao
    # de apply_emsc/apply_osc/apply_airpls acima).
    apply_pqn: bool = False
    # VRM (aumento de dados espectral, `guaraci.aumento_dados.AumentoVRM`,
    # inspirado em Tumoine, Metz, Abdelghafour, Esteve, Grotus, Bendoula &
    # Roger 2026, DOI 10.1016/j.chemolab.2026.105769 -- ver ressalva de
    # fidelidade ao artigo no docstring do modulo) NAO tem flag `apply_*`
    # aqui, ao contrario de EMSC/OSC/AirPLS/PQN acima: aqueles sao
    # transformers que PRESERVAM o numero de linhas (encaixam no Pipeline
    # de `build_preprocessor`); VRM e' um AUMENTO DE DADOS que MUDA o
    # numero de linhas do treino (gera copias extras), entao nao se
    # encaixa no mesmo mecanismo de toggle dentro de um unico Pipeline
    # sklearn. Padroes sao ponto de partida (nao universais), com efeito
    # esperado pequeno (ver docstring de `aumento_dados.py`). Disponivel para
    # uso EXPERIMENTAL chamando `AumentoVRM`/`aumentar_vrm` diretamente
    # sobre o fold de treino de uma CV group-aware --
    # NAO integrado a nenhum caminho de treino automatico desta classe
    # `Config` nesta rodada (Parte A do Bloco de aumento de dados).

    # CV aninhada p/ selecao do numero de variaveis latentes (correcao do
    # achado #12, rodada multiagente 2026-09-10 -- ver docs/BACKLOG_
    # MULTIAGENTE.md #12 e Passo 211/213 de PROGRESSO.md). Sem isso, a
    # metrica de CV reportada e o n_opt sao escolhidos com os MESMOS folds,
    # o que infla a metrica de forma otimista -- medido: 10 seeds
    # replicadas, naive > aninhada em 10/10, Wilcoxon p=0,0020. Default True
    # a partir da v1.0 (metrica honesta passa a ser o padrao); custa ~4-5x
    # mais tempo na etapa de selecao de LVs -- desligue p/ iteracao rapida
    # se o custo for proibitivo (aceita a metrica otimista conhecida).
    # NAO afeta o `n_opt` do modelo final implantado (continua escolhido
    # pela CV externa sobre todo o dado -- CV aninhada e' avaliacao, nao
    # selecao de hiperparametro de um unico modelo).
    selecao_lv_cv_aninhada: bool = True

    max_lvs: int = 40
    n_pcs_pca: int = 10
    # HCA: dendrogram uses PCA scores with hca_n_pcs components
    # (reduces spectral noise before clustering). compare_hca_pipelines
    # generates a dendrogram panel per preprocessing method.
    hca_n_pcs: int = 65
    # Extra opt-in (fora do conjunto padrao de ~7 figuras "core"; ligar na mao).
    compare_hca_pipelines: bool = False
    n_splits_cv: int = 5
    n_repeats_cv: int = 3

    frac_cal: float = 0.70
    # Metodo de split calibracao/validacao da regressao PLS (N2/N3):
    # 'aleatoria' (default, comportamento historico -- GroupShuffleSplit
    # group-aware por mae_id, ou permutacao simples sem mae_id) |
    # 'kennard_stone' (Kennard & Stone 1969 -- cobertura maximamente
    # representativa do espaco espectral em vez de aleatoria; group-aware,
    # colapsa replicas fisicas num espectro medio por grupo antes de
    # selecionar). Hiperparametro avancado, config-only (mesmo precedente
    # de ipls_n_intervalos/vip_threshold_sel/etc -- nao exposto no app/CLI).
    cal_val_split: str = "aleatoria"   # 'aleatoria' | 'kennard_stone'

    n_permutations: int = 200
    n_permutations_wold: int = 200        # 200 for publication; 50 is minimum for diagnostic
    n_bootstrap_vip: int = 30
    n_bootstrap_bca: int = 500
    # Os 3 abaixo sao "extra" opt-in (fora do conjunto padrao de ~7 figuras
    # "core"): validacoes/comparacoes adicionais, uteis mas nao essenciais
    # para o resultado principal. Ligar na mao quando quiser mais rigor.
    comparar_pipelines: bool = False
    run_wold: bool = False
    run_cv_anova: bool = False
    # Teste de incerteza de Martens (Martens & Martens 2000): jackknifing
    # group-aware dos coeficientes PLS -- teste de hipotese formal (p-valor)
    # de significancia por variavel, mais rigoroso que VIP/SR (magnitude).
    run_martens: bool = False
    # Paralelismo (processos, via joblib/loky) para os testes de permutacao/
    # Wold — cada iteracao e independente (mesma X, so o rotulo e
    # reembaralhado), entao rodar em paralelo NAO altera nenhum resultado
    # numerico, so o tempo de execucao (verificado por teste de regressao).
    # Padrao 1 (sequencial) preserva o comportamento atual sem mudanca.
    # Medido: n_jobs=4 ~2x mais rapido no pipeline completo (threading NAO
    # ajuda aqui — overhead do sklearn e Python/GIL-bound; processos sim).
    # Em ambientes com pouca RAM/CPU (ex.: Streamlit Cloud gratuito), deixe em 1
    # (processos extras multiplicam o uso de memoria).
    n_jobs_permutation: int = 1

    frac_holdout: float = 0.20         # v14: external holdout by default
    # v15: pure samples (conc==0) ALWAYS stay in training — they are scarce (3/class)
    # and precious for DD-SIMCA/interpretation. Only adulterated go to holdout.
    holdout_preserves_pure: bool = True
    seed_holdout: int = 42

    n_per_class: int = 20
    n_synthetic_points: int = 1000
    n_synthetic_replicates: int = 3   # replicas fisicas/ponto (estilo T1/T2/T3) no mode sintetico
    # Modo sintetico: adulterantes a gerar por especie (letras A/M/S). Vazio =
    # so o gradiente puro legado (nao muda dados de teste existentes). Quando
    # preenchido, cada especie ganha puros + esses adulterantes em varios teores
    # -> permite exercitar o heatmap especie x adulterante (N3) sinteticamente.
    synthetic_adulterants: Tuple[str, ...] = ()
    # Cenario do modo sintetico (`simulador_espectral`): "nir", "raman" e
    # "rmn" imitam um conjunto real (espalhamento, linha de base, efeito de
    # dia, composicao variando entre amostras, replicas correlacionadas);
    # "classico" e' o gerador antigo, sem nada disso. Padrao "nir"; a suite
    # de testes fixa "classico" (GUARACI_SINTETICO_CENARIO no conftest),
    # porque os limiares dela foram calibrados nele.
    synthetic_scenario: str = field(default_factory=lambda: os.environ.get(
        "GUARACI_SINTETICO_CENARIO", "nir"))

    seed: int = 42

    # Sprint 3
    # DD-SIMCA e' o "core" automatico de N2 (executar() forca True quando
    # level=="N2", e ignora o toggle quando level=="N1" -- ver executar()).
    # Default False aqui so' afeta N3, onde vira extra opt-in de verdade.
    run_ddsimca: bool = False
    ddsimca_n_components: int = 7       # PCA LVs per DD-SIMCA model (3 was insufficient — UCL poorly calibrated)
    ddsimca_ucl_method: str = "empirical"  # 'empirical' | 'theoretical' | 'chi2'
    # v14: 'todos' trains each model with ALL samples of the class
    # (exploratory; works with few pure samples). 'puros' = true one-class N2
    # but requires >=15 pure/class (current data: 3/class).
    ddsimca_train_on: str = "puros"   # 'todos' | 'puros'
    # Diagnostico complementar ao LOGO via Procrustes Cross-Validation
    # (Kucheryavskiy/Rodionova/Pomerantsev) -- exige o extra opcional
    # [robusto] (pacote 'prcv'). NUNCA substitui o aviso "nao validado" do
    # LOGO quando ha' so' 1 grupo de replica -- ver
    # classificadores.ddsimca_pcv_sensitivity().
    ddsimca_pcv: bool = False
    # Classificador principal. 'plsda' (PADRAO): PLS com resposta dummy e argmax.
    # 'plslda': LDA sobre os escores do PLS (PLS-LDA); resolve o mascaramento da
    # classe do meio quando as classes sao niveis ordenados (ver
    # `classificadores.PLSLDAClassifier`). Opt-in.
    classificador: str = "plsda"
    # Extra opt-in (fora do conjunto padrao de ~7 figuras "core").
    run_opls: bool = False
    n_ortho_opls: int = 1               # OPLS-DA orthogonal components
    run_benchmark: bool = False    # v27: SVM / RF / XGBoost vs PLS-DA (same CV)
    # Auto-Benchmark de REGRESSAO (N2/N3): Ridge/Lasso/Elastic Net/
    # Dual-sPLS/SVR/RF vs PLS-R, por especie, mesmo split cal/val e
    # pre-processamento (ver
    # avaliacao_modelos.benchmark_regression_by_species). So' roda quando
    # ha regressao multi-especie (mesma condicao de pls_regression_by_species).
    run_benchmark_regression: bool = False
    run_monte_carlo: bool = False      # v28: MC CV (N × GroupShuffleSplit) for 95% CI
    n_monte_carlo: int = 100               # number of MC repetitions
    monte_carlo_test_size: float = 0.25    # test fraction per MC repetition
    monte_carlo_include_all: bool = False # v28: run all benchmark models in MC CV
    run_shap: bool = False            # v28: SHAP values (TreeExplainer) for ensemble
    shap_max_samples: int = 500        # sample limit for SHAP (memory control)

    # ---- Class exclusion (e.g. a class with an anomalous batch) ----
    exclude_classes: Tuple[str, ...] = ()

    # ---- STAGE 4: Variable selection ----
    # Extra opt-in (fora do conjunto padrao de ~7 figuras "core").
    executar_etapa4: bool = False
    ipls_n_intervalos: int = 20         # iPLS: number of intervals
    vip_threshold_sel: float = 1.0      # selection by VIP >= threshold
    sr_top_frac: float = 0.20           # selection by SR: top fraction
    splsda_keep_by_comp: int = 50      # sPLS-DA: non-zero variables/component
    # SPA/APS (Successive Projections Algorithm, Araujo et al. 2001) and
    # AG (Genetic Algorithm, GA-PLS) — opt-in (default False) because they
    # run many more CV evaluations than iPLS/VIP/SR/sPLS-DA (heavier, like
    # run_benchmark/run_shap).
    run_spa: bool = False
    spa_n_vars_max: int = 20            # SPA: max chain length (variables)
    spa_n_starts: int = 25              # SPA: starting variables sampled (evenly across spectrum)
    executar_ag: bool = False
    ag_tam_populacao: int = 20          # AG: chromosomes per generation
    ag_n_geracoes: int = 15             # AG: number of generations
    ag_prob_mutacao: float = 0.02       # AG: per-bit mutation probability
    ag_frac_inicial: float = 0.10       # AG: initial fraction of variables "on" per chromosome
    # CARS (Li, Liang, Xu & Cao 2009, DOI 10.1016/j.aca.2009.06.046) and UVE
    # (Centner et al. 1996, DOI 10.1021/ac960321m) — opt-in, mesmo motivo do
    # SPA/AG (mais avaliacoes de CV que iPLS/VIP/SR/sPLS-DA).
    run_cars: bool = False
    cars_n_iteracoes: int = 50          # CARS: EDF sampling runs
    cars_frac_amostragem: float = 0.80  # CARS: fracao de amostras por Monte Carlo run
    # Protocolo de robustez (robustness.protocolo_padrao): opt-in, refaz o modelo sob ruido,
    # deriva e variantes de pre-processamento e anexa o intervalo ao model card.
    robustness_protocol: bool = False
    # Selecao de graficos: "padrao" (essenciais do objetivo, catalogo), "todos" ou lista.
    figure_selection: str = "padrao"
    run_uve: bool = False
    uve_n_repeticoes: int = 50          # UVE: Monte Carlo runs p/ estabilidade do coeficiente
    uve_frac_amostragem: float = 0.80   # UVE: fracao de amostras por Monte Carlo run

    # ---- visoes de configuracao sobre campos existentes (uma fonte so') ----
    # `etapas_pre_processamento` (config.yaml) le/escreve os apply_* acima;
    # so' tem efeito com default_preprocessing='custom' ("Personalizado").
    @property
    def etapas_pre_processamento(self) -> Tuple[str, ...]:
        return tuple(nome for nome, attr in ETAPAS_PRE_PROCESSAMENTO if getattr(self, attr))

    @etapas_pre_processamento.setter
    def etapas_pre_processamento(self, etapas: Any) -> None:
        pedidas = {str(e).strip().lower() for e in (etapas or ()) if str(e).strip()}
        validas = {nome for nome, _ in ETAPAS_PRE_PROCESSAMENTO}
        if pedidas - validas:
            raise ValueError(f"etapa(s) desconhecida(s) {sorted(pedidas - validas)}; use {[n for n, _ in ETAPAS_PRE_PROCESSAMENTO]}")
        for nome, attr in ETAPAS_PRE_PROCESSAMENTO:
            setattr(self, attr, nome in pedidas)

    # `imagem_recorte` (config.yaml): x0, y0, x1, y1 em fracao da foto -> image_crop
    @property
    def imagem_recorte(self) -> Tuple[float, ...]:
        return tuple(float(v) for v in self.image_crop)

    @imagem_recorte.setter
    def imagem_recorte(self, valores: Any) -> None:
        v = tuple(float(x) for x in valores)
        if len(v) != 4 or not all(0.0 <= x <= 1.0 for x in v) or not (v[0] < v[2] and v[1] < v[3]):
            raise ValueError("use 4 valores entre 0 e 1 (x0, y0, x1, y1) com x0 < x1 e y0 < y1")
        self.image_crop = (v[0], v[1], v[2], v[3])

    # `imagem_referencia` (config.yaml): regiao do cartao cinza/branco, x0, y0, x1, y1 em fracao da
    # foto; vazio = sem normalizacao -> image_reference
    @property
    def imagem_referencia(self) -> Tuple[float, ...]:
        return tuple(float(v) for v in self.image_reference)

    @imagem_referencia.setter
    def imagem_referencia(self, valores: Any) -> None:
        v = tuple(float(x) for x in (valores or ()))
        if v and (len(v) != 4 or not all(0.0 <= x <= 1.0 for x in v) or not (v[0] < v[2] and v[1] < v[3])):
            raise ValueError("use 4 valores entre 0 e 1 (x0, y0, x1, y1) com x0 < x1 e y0 < y1, ou vazio")
        self.image_reference = v


#: Etapas do pre-processamento personalizado, na ORDEM em que build_preprocessor as aplica.
ETAPAS_PRE_PROCESSAMENTO: Tuple[Tuple[str, str], ...] = (
    ("airpls", "apply_airpls"), ("snv", "apply_snv"), ("pqn", "apply_pqn"), ("emsc", "apply_emsc"),
    ("sg", "apply_sg"), ("mc", "apply_mc"), ("osc", "apply_osc"))

#: Nome amigavel de cada nivel (N1/N2/N3). Publico para as interfaces.
NIVEL_NOME = _NIVEL_NOME
