"""technique_registry.py — Catalogo dos metodos cientificos do GUARACI
(Agente 6, item "d": listar/explicar tecnicas disponiveis pelo assistente).

Fonte UNICA da lista que o assistente `G` mostra quando o usuario pergunta
"o que o GUARACI sabe fazer?"/"quais metodos existem para X?". Mesmo padrao
ja usado por `model_registry.py`/`io_registry.py`: uma entrada por metodo,
num so lugar, em vez de uma lista escrita a mao espalhada pela UI (o
problema que `_guaraci_navegar_secoes` tinha antes desta rodada -- ver
docs/DESIGN.md, secao do Agente 6).

"Nunca fica desatualizado" nao vem de introspeccao de AST (fragil para
codigo cientifico com muitos helpers internos) e sim de
`tests/test_technique_registry.py`: para os modulos de proposito UNICO
(cada simbolo publico e' de fato uma tecnica, nao um helper interno), o
teste falha se `__all__` tiver algo sem entrada aqui. Adicionar um metodo
novo a um desses modulos sem atualizar este arquivo quebra a suite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

__all__ = [
    "TechniqueEntry",
    "REGISTRY",
    "MODULOS_COBERTURA_TOTAL",
]


@dataclass(frozen=True)
class TechniqueEntry:
    """Uma entrada do catalogo. `referencia` e' `"modulo.simbolo"` (string,
    nunca o objeto importado -- mantem este arquivo sem custo de import de
    numpy/scipy/sklearn so' para listar nomes)."""
    id: str
    categoria: str
    nome: str
    referencia: str
    quando_usar: str
    limitacao: str


REGISTRY: List[TechniqueEntry] = [
    # ---- Classificacao / deteccao (autenticacao pura x adulterada) -------
    TechniqueEntry(
        id="ddsimca",
        categoria="classificacao_deteccao",
        nome="DD-SIMCA",
        referencia="guaraci.classificadores.DDSimca",
        quando_usar=(
            "Autenticacao one-class por especie: aprende so' o espaco da "
            "classe PURA (PCA por classe) e testa se uma amostra nova cabe "
            "nele -- nao precisa de exemplo de adulterante no treino."),
        limitacao=(
            "Precisa de amostras FISICAS suficientes por classe (nao "
            "espectros); com 1 so' ponto de amostragem fisico, o limite e' "
            "calibrado contra 1 unica observacao independente e o software "
            "reporta isso em vez de um numero confiante. Sensibilidade "
            "LOGO nao-validavel quando so' ha' 1 grupo de puros/especie."),
    ),
    TechniqueEntry(
        id="conformal_one_class",
        categoria="classificacao_deteccao",
        nome="Predicao conformal (one-class)",
        referencia="guaraci.conformal.ConformalOneClass",
        quando_usar=(
            "Alternativa a DD-SIMCA com garantia de cobertura estatistica "
            "explicita e calibravel (alpha) -- devolve aceita/rejeita com "
            "taxa de erro CONTROLADA, nao so' um score."),
        limitacao=(
            "alpha minimo garantivel e' limitado por n (alpha_min = "
            "1/(n+1), ver achievable_alpha) -- com n pequeno, o alpha "
            "alcancavel pode ser maior do que o desejado."),
    ),
    TechniqueEntry(
        id="pls_da",
        categoria="classificacao_deteccao",
        nome="PLS-DA",
        referencia="guaraci.avaliacao_modelos.PLSDAClassifier",
        quando_usar=(
            "Classificacao multiclasse supervisionada (qual especie/classe "
            "esta amostra pertence) -- o classificador PADRAO do fluxo N1."),
        limitacao=(
            "Supervisionado: precisa de exemplo rotulado de TODAS as "
            "classes no treino; nao detecta uma classe nunca vista (isso e' "
            "papel do DD-SIMCA/conformal, nao do PLS-DA)."),
    ),
    # ---- Quantificacao -----------------------------------------------------
    TechniqueEntry(
        id="pls_r_pooled",
        categoria="quantificacao",
        nome="PLS-R (regressao pooled)",
        referencia="guaraci.pipeline.pls_regressao_pooled",
        quando_usar=(
            "Estima teor de adulterante (%) com UM modelo de regressao "
            "treinado em todas as especies juntas -- mais simples, precisa "
            "de menos dado por especie."),
        limitacao=(
            "Assume que a relacao espectro->teor e' aproximadamente igual "
            "entre especies; se a matriz-hospedeira dominar o sinal mais "
            "que o adulterante (medido: 21-175x mais variancia explicada "
            "pela especie que pelo adulterante), o pooled mistura dois "
            "efeitos diferentes num modelo so'."),
    ),
    TechniqueEntry(
        id="pls_r_por_especie",
        categoria="quantificacao",
        nome="PLS-R por especie",
        referencia="guaraci.avaliacao_modelos.benchmark_regression_by_species",
        quando_usar=(
            "Mesma ideia do pooled, mas 1 modelo de regressao POR ESPECIE "
            "-- evita a matriz-hospedeira contaminar a estimativa do teor "
            "quando ha' dado suficiente por especie para treinar cada um."),
        limitacao=(
            "Precisa de n minimo POR ESPECIE (nao so' no total); com "
            "especie rara, o modelo daquela especie fica subdimensionado "
            "ou nem chega a treinar."),
    ),
    # ---- Identificacao de conjunto aberto (Detectar->Identificar->Quantificar) --
    TechniqueEntry(
        id="identificacao_conjunto_aberto",
        categoria="identificacao_conjunto_aberto",
        nome="Identificacao de adulterante (conjunto aberto)",
        referencia="guaraci.identificacao.identify_sample",
        quando_usar=(
            "Depois de detectar que uma amostra e' adulterada (DD-SIMCA/"
            "conformal), identifica QUAL adulterante e' -- por combinacao "
            "especie x adulterante, calibrada por predicao conforme, com "
            "garantia de cobertura formal por combinacao. So' produz "
            "rotulo quando a combinacao tem garantia estatistica VALIDADA "
            "(>=2 sessoes de coleta independentes); senao, reporta "
            "DESCONHECIDO em vez de arriscar um palpite sem garantia."),
        limitacao=(
            "Cobertura so' e' validada por combinacao com sessoes de "
            "coleta independentes suficientes -- no dataset privado do "
            "TCC, 36 de 38 combinacoes especie x adulterante tem so' 1 "
            "sessao (nao-validavel). Fluxo completo em "
            "guaraci.predicao.predict_blind."),
    ),
    # ---- Selecao de amostras (Bloco K) -------------------------------------
    TechniqueEntry(
        id="kennard_stone",
        categoria="selecao_amostras",
        nome="Kennard-Stone",
        referencia="guaraci.dados_io.kennard_stone_split",
        quando_usar=(
            "Escolhe um subconjunto de calibracao que cobre bem o espaco "
            "espectral (maximiza distancia entre amostras selecionadas) -- "
            "bom default para selecao representativa sem depender do alvo Y."),
        limitacao=(
            "So' olha X (espectro), nunca Y (concentracao/classe) -- pode "
            "deixar uma faixa de concentracao mal coberta se ela nao "
            "corresponder a uma regiao distinta do espectro."),
    ),
    TechniqueEntry(
        id="duplex",
        categoria="selecao_amostras",
        nome="Duplex",
        referencia="guaraci.dados_io.duplex_split",
        quando_usar=(
            "Como Kennard-Stone, mas distribui as amostras mais distantes "
            "ALTERNANDO entre os dois conjuntos (calibracao/validacao) em "
            "vez de encher um primeiro -- validacao fica mais representativa."),
        limitacao="Mesma limitacao do Kennard-Stone: ignora Y.",
    ),
    TechniqueEntry(
        id="spxy",
        categoria="selecao_amostras",
        nome="SPXY",
        referencia="guaraci.dados_io.spxy_split",
        quando_usar=(
            "Kennard-Stone considerando X E Y juntos (distancia combinada "
            "espectro+concentracao) -- preferivel quando o objetivo e' "
            "quantificacao e a faixa de Y precisa ficar bem coberta."),
        limitacao="Precisa de Y numerico (concentracao) -- nao serve para selecao pre-classificacao sem alvo continuo.",
    ),
    # ---- Transferencia de calibracao (Bloco 12) ----------------------------
    TechniqueEntry(
        id="direct_standardization",
        categoria="transferencia_calibracao",
        nome="Direct Standardization (DS)",
        referencia="guaraci.transferencia_calibracao.direct_standardization",
        quando_usar=(
            "Move um modelo calibrado num instrumento para funcionar em "
            "outro (equipamento escravo), aprendendo uma transformacao "
            "GLOBAL entre os dois a partir de amostras medidas nos dois."),
        limitacao=(
            "Transformacao densa (todas as variaveis afetam todas) -- mais "
            "sujeita a overfitting com poucas amostras de transferencia "
            "que a versao piecewise (PDS)."),
    ),
    TechniqueEntry(
        id="piecewise_direct_standardization",
        categoria="transferencia_calibracao",
        nome="Piecewise Direct Standardization (PDS)",
        referencia="guaraci.transferencia_calibracao.piecewise_direct_standardization",
        quando_usar=(
            "Mesma ideia do DS, mas a transformacao usa so' uma JANELA "
            "local de variaveis vizinhas por vez -- geralmente mais "
            "estavel que o DS global com poucas amostras de transferencia."),
        limitacao=(
            "Requer escolher o tamanho da janela; no benchmark publico "
            "(Corn), a reducao de erro entre instrumentos foi medida e "
            "documentada -- nem sempre supera o DS, depende do par de "
            "instrumentos (ver docs/VALIDACAO_PUBLICA.md)."),
    ),
    # ---- Figuras de merito --------------------------------------------------
    TechniqueEntry(
        id="figuras_de_merito_regressao",
        categoria="figuras_de_merito",
        nome="LOD/LOQ/sensibilidade (figuras de merito)",
        referencia="guaraci.chemometric_stats.regression_figures_of_merit",
        quando_usar=(
            "Reporta limite de deteccao/quantificacao, sensibilidade e "
            "seletividade de um modelo de regressao PLS, seguindo "
            "Valderrama, Braga & Poppi (2009) -- acompanha toda "
            "quantificacao, nao e' uma etapa que se escolhe rodar ou nao."),
        limitacao=(
            "LOD/LOQ dependem de estimar ruido instrumental a partir de "
            "REPLICAS FISICAS (>=2 por ponto amostral via mae_id); sem "
            "replica suficiente, os campos voltam N/A -- nunca um numero "
            "inventado."),
    ),
    TechniqueEntry(
        id="rpd_rer",
        categoria="figuras_de_merito",
        nome="RPD/RER",
        referencia="guaraci.chemometric_stats.rpd_rer",
        quando_usar=(
            "Interpreta se um RMSEP de quantificacao e' bom ou ruim em "
            "termos relativos (RPD = desvio padrao da referencia / RMSEP; "
            "RER = faixa de referencia / erro) -- acompanha SEMPRE o RMSEP, "
            "nunca aparece sozinho."),
        limitacao=(
            "Faixa de interpretacao (Williams 2014; AACC 39-00.01) e' "
            "generica por literatura, nao especifica da matriz analisada."),
    ),
    # ---- Robustez / linearidade ---------------------------------------------
    TechniqueEntry(
        id="linearidade",
        categoria="robustez_linearidade",
        nome="Teste de falta de ajuste (linearidade)",
        referencia="guaraci.linearity.lack_of_fit_test",
        quando_usar=(
            "Testa se a relacao espectro->concentracao e' de fato linear "
            "no intervalo calibrado -- opcional, so' roda quando ha' "
            "replica verdadeira suficiente por nivel de concentracao."),
        limitacao=(
            "Sem replica fisica suficiente por nivel, o teste NAO e' "
            "computavel -- `computavel=False` com `motivo`, nunca forca o "
            "teste nem inventa replica."),
    ),
    TechniqueEntry(
        id="robustez",
        categoria="robustez_linearidade",
        nome="Protocolo de robustez",
        referencia="guaraci.robustness.run_robustness_protocol",
        quando_usar=(
            "Mede o quanto o resultado varia sob perturbacoes pequenas e "
            "plausiveis (ruido gaussiano, deriva de linha de base, "
            "variantes de pre-processamento) -- opcional, para reforcar "
            "confianca antes de publicar/decidir."),
        limitacao=(
            "Reporta min/mediana/max da variacao -- NUNCA um veredito "
            "aprovado/reprovado automatico (R2); a decisao de aceitar a "
            "variacao e' sempre humana."),
    ),
    # ---- Perfis (ligacao direta com o Agente 5B) ----------------------------
    TechniqueEntry(
        id="perfil_matriz",
        categoria="perfis",
        nome="Perfil de matriz quimica",
        referencia="guaraci.perfil_matriz.load_profile",
        quando_usar=(
            "Declara QUAL matriz esta sendo analisada (oleo, milho, mel, "
            "oleos comestiveis, generico) -- define faixa espectral "
            "padrao, pre-processamento padrao e o vocabulario usado nos "
            "relatorios (nunca afirma 'oleo' analisando milho)."),
        limitacao=(
            "So' cobre as matrizes com perfil escrito; uma matriz nova "
            "precisa de um YAML proprio (ver perfis_matriz/generico.yaml "
            "como modelo) -- carregar o perfil errado produziria "
            "vocabulario/faixa incorretos."),
    ),
    TechniqueEntry(
        id="perfil_tecnica_aquisicao",
        categoria="perfis",
        nome="Perfil de tecnica de aquisicao (modo imagem)",
        referencia="guaraci.perfil_matriz.load_profile",
        quando_usar=(
            "So' modo_entrada=imagem: declara COMO a imagem foi capturada "
            "(bancada/celular/scanner) -- mostra resolucao esperada, "
            "formatos aceitos e o nivel de garantia de agrupamento "
            "TIPICO dessa tecnica."),
        limitacao=(
            "So' informativo -- a garantia REAL de agrupamento vem da "
            "estrutura de dados fornecida (subpasta por amostra/CSV de "
            "associacao), nunca deste campo; nenhum perfil de tecnica "
            "de imagem tem validacao publicada com dado real ainda."),
    ),
    # ---- Resolucao de mistura (Bloco 14, 2026-09-04) -----------------------
    TechniqueEntry(
        id="mcr_als",
        categoria="resolucao_mistura",
        nome="MCR-ALS",
        referencia="guaraci.mcr_als.mcr_als",
        quando_usar=(
            "Quando o objetivo e' recuperar OS PROPRIOS espectros puros e "
            "as proporcoes de cada componente numa mistura (nao so' "
            "classificar/quantificar contra um alvo ja conhecido) -- "
            "decompoe a matriz de espectros em perfis de concentracao (C) "
            "e espectrais (S) via minimos quadrados alternados, com "
            "restricoes de nao-negatividade/normalizacao."),
        limitacao=(
            "Solucao SEM garantia de unicidade (ambiguidade rotacional -- "
            "toda chamada carrega esse aviso no retorno). Use "
            "`avaliar_incerteza_rotacional` para medir sensibilidade a' "
            "inicializacao antes de tratar C/S como definitivos."),
    ),
]

#: Modulos de proposito UNICO (todo simbolo em __all__ e' de fato uma
#: tecnica) -- tests/test_technique_registry.py exige cobertura completa
#: so' destes, menos o `excecoes` (dataclass/enum/excecao, nao "metodo").
MODULOS_COBERTURA_TOTAL = {
    "guaraci.classificadores": {
        "incluir": {"DDSimca"},
        "excecoes": {"OPLSDAWrapper", "ddsimca_logo_sensitivity",
                     "ddsimca_pcv_sensitivity"},
    },
    "guaraci.conformal": {
        "incluir": {"ConformalOneClass"},
        "excecoes": {"achievable_alpha", "n_minimum_for_alpha",
                     "conformal_threshold"},
    },
    "guaraci.transferencia_calibracao": {
        "incluir": {"direct_standardization", "piecewise_direct_standardization"},
        "excecoes": {"StandardizationTransform", "apply_standardization"},
    },
    "guaraci.linearity": {
        "incluir": {"lack_of_fit_test"},
        "excecoes": {"LackOfFitResult"},
    },
    "guaraci.robustness": {
        "incluir": {"run_robustness_protocol"},
        "excecoes": {"RobustnessResult", "gaussian_noise_variants",
                     "baseline_drift_variants", "preprocessing_config_variants",
                     "avaliar_rmsep_plsr", "avaliar_bal_acc_plsda"},
    },
}
