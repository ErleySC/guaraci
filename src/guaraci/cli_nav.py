"""cli_nav.py — navegacao da CLI como dado puro (sem Rich, sem I/O).

Fonte unica do guia de design v2 da CLI (`docs/DESIGN.md` secao 7):

- teclas globais (`?`, `G`, `I`, `V`, `Q`, `R`, `M`) e como uma entrada
  digitada num prompt de MENU e' interpretada (`interpretar`);
- o rodape de toda tela, gerado a partir do registro de teclas -- nunca
  escrito a mao tela a tela (foi assim que o bug B2 de
  `docs/AUDITORIA_UX_2026-09-24.md` aconteceu);
- a arvore de menus (grupos do menu principal e suas telas), com o que
  aparece no modo Basico e o que so' aparece no Avancado;
- a ajuda curta de cada tela (o que faz / quando usar / o que fazer
  depois), que alimenta `?` e o assistente `G`.

`guaraci.py` desenha a partir daqui; os testes leem daqui. Mesmo padrao de
`app_nav.py` (navegacao do app web como dado puro).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Teclas globais
# ---------------------------------------------------------------------------
AJUDA = "?"
GUARACI = "G"
IDIOMA = "I"
VOLTAR = "V"
SAIR = "Q"
RODAR = "R"
AVANCADOS = "M"

TECLAS_GLOBAIS: Tuple[str, ...] = (AJUDA, GUARACI, IDIOMA, VOLTAR, SAIR,
                                   RODAR, AVANCADOS)

# Aceitas como "voltar" nas telas internas sem aparecer no rodape:
# Enter vazio (o mais rapido), "0" (rodape antigo) e "Q" (antes voltava de
# qualquer submenu -- quem decorou nao deve cair em "opcao invalida").
_ALIASES_VOLTAR = ("", "0", VOLTAR, SAIR)


def interpretar(raw: str, *, raiz: bool = False) -> Tuple[str, Optional[int]]:
    """Classifica o que foi digitado num prompt de MENU.

    Retorna `(acao, n)`:
      - ("item", n)         -> numero n (1-based) digitado
      - ("ajuda", None)     -> `?`
      - ("ajuda", n)        -> `?n` / `? n`: ajuda direta do campo n
      - ("guaraci" | "idioma" | "rodar" | "avancados", None)
      - ("voltar", None)    -> tela interna: V, Enter vazio, 0 ou Q
      - ("sair", None)      -> menu principal: Q
      - ("nada", None)      -> menu principal: Enter vazio/V/0 (redesenha)
      - ("outro", None)     -> qualquer outra coisa (a tela decide; ex.:
                               atalho legado no menu principal)

    So' vale para prompts de menu. Prompts de VALOR (pasta, nome, classes)
    nunca passam por aqui: la' o texto digitado e' sempre um valor (uma
    classe chamada "G" nao pode abrir o assistente).
    """
    r = (raw or "").strip().upper()
    if r.startswith(AJUDA):
        resto = r[1:].strip()
        if not resto:
            return ("ajuda", None)
        if resto.isdigit():
            return ("ajuda", int(resto))
        return ("outro", None)
    if r.isdigit():
        n = int(r)
        if n == 0:
            return ("nada", None) if raiz else ("voltar", None)
        return ("item", n)
    if raiz:
        if r == SAIR:
            return ("sair", None)
        if r in ("", VOLTAR):
            return ("nada", None)
    elif r in _ALIASES_VOLTAR:
        return ("voltar", None)
    return {
        GUARACI: ("guaraci", None),
        IDIOMA: ("idioma", None),
        RODAR: ("rodar", None),
        AVANCADOS: ("avancados", None),
    }.get(r, ("outro", None))


_ROTULOS_TECLAS: Dict[str, Dict[str, str]] = {
    AJUDA: {"PT": "Ajuda", "EN": "Help"},
    GUARACI: {"PT": "Guaraci", "EN": "Guaraci"},
    IDIOMA: {"PT": "Idioma", "EN": "Language"},
    VOLTAR: {"PT": "Voltar", "EN": "Back"},
    SAIR: {"PT": "Sair", "EN": "Quit"},
    RODAR: {"PT": "Rodar análise", "EN": "Run analysis"},
}


def rodape(lang: str, *, raiz: bool = False) -> List[Tuple[str, str]]:
    """Rodape fixo: `? Ajuda  G Guaraci  I Idioma  V Voltar` (ou `Q Sair`
    no menu principal, onde nao ha' para onde voltar)."""
    lang = "EN" if lang == "EN" else "PT"
    ultima = SAIR if raiz else VOLTAR
    return [(k, _ROTULOS_TECLAS[k][lang]) for k in (AJUDA, GUARACI, IDIOMA, ultima)]


def linha_acoes(lang: str, *, rodar: bool, n_ocultos: Optional[int],
                modo_avancado: bool) -> List[Tuple[str, str]]:
    """Linha de acoes acima do rodape: `R` so' onde ha' configuracao e `M`
    so' onde ha' algo a esconder/mostrar. `n_ocultos=None` = a tela nao tem
    nada avancado (sem `M`)."""
    lang = "EN" if lang == "EN" else "PT"
    out: List[Tuple[str, str]] = []
    if rodar:
        out.append((RODAR, _ROTULOS_TECLAS[RODAR][lang]))
    if n_ocultos is not None:
        if modo_avancado:
            out.append((AVANCADOS, "Ocultar avançados" if lang == "PT"
                        else "Hide advanced"))
        elif n_ocultos > 0:
            if lang == "PT":
                txt = (f"Mostrar avançados ({n_ocultos} oculto)" if n_ocultos == 1
                       else f"Mostrar avançados ({n_ocultos} ocultos)")
            else:
                txt = f"Show advanced ({n_ocultos} hidden)"
            out.append((AVANCADOS, txt))
    return out


# ---------------------------------------------------------------------------
# Arvore de menus
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Item:
    """Uma tela alcancavel a partir de um grupo."""
    id: str
    pt: str
    en: str
    avancado: bool = False
    # Aparece mesmo no modo Basico quando uma destas condicoes de contexto
    # vale (ex.: "hsi" -> tecnica selecionada e' HSI). Tipo de dado nao e'
    # nivel de experiencia (guia v2, secao 2.4).
    sempre_se: FrozenSet[str] = field(default_factory=frozenset)

    def rotulo(self, lang: str) -> str:
        return self.en if lang == "EN" else self.pt


@dataclass(frozen=True)
class Grupo:
    """Uma entrada do menu principal. Grupo com um unico item abre a tela
    direto (sem submenu de uma linha)."""
    id: str
    pt: str
    en: str
    itens: Tuple[Item, ...]
    avancado: bool = False
    sempre_se: FrozenSet[str] = field(default_factory=frozenset)
    # O grupo tem campos de configuracao -> mostra `R` na linha de acoes.
    rodar: bool = True

    def rotulo(self, lang: str) -> str:
        return self.en if lang == "EN" else self.pt

    @property
    def direto(self) -> bool:
        return len(self.itens) == 1


_HSI = frozenset({"hsi"})

MENU: Tuple[Grupo, ...] = (
    Grupo("comecar", "Começar", "Get started", rodar=False, itens=(
        Item("exemplo", "Rodar um exemplo (dados sintéticos, nada a configurar)",
             "Run an example (synthetic data, nothing to configure)"),
        Item("perfis", "Usar um perfil pronto para a minha amostra",
             "Use a ready profile for my sample"),
        Item("passo_a_passo", "Passo a passo: o que preciso decidir?",
             "Step by step: what do I need to decide?"),
    )),
    Grupo("preparar", "Preparar dados", "Prepare data", itens=(
        Item("projeto", "Projeto (pastas e nome)", "Project (folders and name)"),
        Item("dados", "Dados", "Data"),
        Item("tecnica", "Técnica analítica", "Analytical technique"),
        Item("preproc", "Pré-processamento", "Preprocessing"),
        Item("auditoria", "Verificar delineamento (auditoria anti-vazamento)",
             "Check study design (anti-leakage audit)"),
        Item("planejamento", "Planejar coleta", "Plan sample collection",
             avancado=True),
        Item("selecao_amostras", "Selecionar amostras (Kennard-Stone, Duplex, SPXY)",
             "Select samples (Kennard-Stone, Duplex, SPXY)", avancado=True),
        Item("codificacao", "Codificação DX", "DX encoding", avancado=True),
    )),
    Grupo("modelar", "Modelar e validar", "Model and validate", itens=(
        Item("objetivo_lvs", "Objetivo e variáveis latentes",
             "Goal and latent variables"),
        Item("validacao", "Validação", "Validation"),
        Item("metodos_classe", "Métodos de classe (DD-SIMCA, OPLS-DA)",
             "Class methods (DD-SIMCA, OPLS-DA)", avancado=True),
        Item("selecao_variaveis", "Seleção de variáveis", "Variable selection",
             avancado=True),
        Item("modulos_pesados", "Módulos pesados (Benchmark, Monte Carlo, SHAP)",
             "Heavy modules (Benchmark, Monte Carlo, SHAP)", avancado=True),
    )),
    Grupo("prever", "Prever novas amostras", "Predict new samples", rodar=False,
          itens=(Item("predicao", "Predição em lote", "Batch prediction"),)),
    Grupo("graficos", "Gráficos", "Charts",
          itens=(Item("visualizacao", "Gráficos", "Charts"),)),
    Grupo("sistema", "Sistema", "System", rodar=False, itens=(
        Item("salvar", "Salvar configuração", "Save configuration"),
        Item("carregar", "Carregar configuração", "Load configuration"),
        Item("nome_execucao", "Nome da execução", "Run name"),
        Item("modo", "Modo", "Mode"),
        Item("idioma", "Idioma", "Language"),
        Item("sobre", "Sobre o Guaraci", "About Guaraci"),
        Item("hardware", "Hardware", "Hardware", avancado=True),
    )),
    Grupo("tecnicas_especiais", "Técnicas especiais (ASCA, EPO/GLSW, MCR-ALS, fusão multibloco)",
          "Special techniques (ASCA, EPO/GLSW, MCR-ALS, multiblock fusion)",
          avancado=True, rodar=False,
          itens=(Item("tecnicas_especiais", "Técnicas especiais", "Special techniques"),)),
    Grupo("hsi", "Imageamento hiperespectral", "Hyperspectral imaging",
          avancado=True, sempre_se=_HSI, rodar=False,
          itens=(Item("hsi", "Imageamento hiperespectral", "Hyperspectral imaging",
                      sempre_se=_HSI),)),
)

GRUPOS: Dict[str, Grupo] = {g.id: g for g in MENU}

# Atalhos do menu antigo que continuam funcionando no menu principal, sem
# aparecer na tela (promessa do DESIGN.md secao 4: quem decorou nao perde o
# atalho). So' letras que nao colidem com as globais.
ATALHOS_LEGADOS: Dict[str, str] = {
    "B": "predicao", "X": "hsi", "J": "planejamento", "U": "auditoria",
    "K": "selecao_amostras", "T": "tecnicas_especiais", "P": "perfis",
    "H": "hardware", "A": "sobre", "S": "salvar", "L": "carregar",
    "N": "nome_execucao",
}


def _visivel(avancado: bool, sempre_se: FrozenSet[str], modo_avancado: bool,
             contexto: FrozenSet[str]) -> bool:
    return (not avancado) or modo_avancado or bool(sempre_se & contexto)


def ordenar_visiveis(entradas: Sequence, modo_avancado: bool,
                     contexto: FrozenSet[str] = frozenset()) -> Tuple[list, int]:
    """Basicos primeiro, avancados depois (numeracao continua): o numero de
    um item basico nunca muda entre os modos. Retorna (visiveis, n_ocultos).
    Funciona para `Item` e `Grupo`."""
    basicos = [e for e in entradas if not e.avancado]
    avancados = [e for e in entradas if e.avancado]
    vis_av = [e for e in avancados
              if _visivel(True, e.sempre_se, modo_avancado, contexto)]
    return basicos + vis_av, len(avancados) - len(vis_av)


def localizar(tela_id: str) -> Tuple[Optional[Grupo], Optional[int]]:
    """(grupo, posicao 1-based no modo avancado) da tela `tela_id`."""
    for g in MENU:
        visiveis, _ = ordenar_visiveis(g.itens, True)
        for i, it in enumerate(visiveis, 1):
            if it.id == tela_id:
                return g, i
    return None, None


def caminho(tela_id: str, lang: str) -> str:
    """Caminho legivel ate uma tela, ex. 'Preparar dados › Dados'. Usado em
    todo texto que manda o usuario a outra tela -- nenhum texto cita tecla
    literal ('[2] Dados'), que quebraria a cada renumeracao."""
    g, _ = localizar(tela_id)
    if g is None:
        return tela_id
    if g.direto:
        return g.rotulo(lang)
    it = next(i for i in g.itens if i.id == tela_id)
    return f"{g.rotulo(lang)} › {it.rotulo(lang)}"


def proximo_passo(lang: str, *, dados_ok: bool, hsi: bool) -> str:
    """Uma frase com o proximo passo provavel (padrao `git status`)."""
    pt = lang != "EN"
    if hsi:
        return (f"abrir {caminho('hsi', lang)}" if pt
                else f"open {caminho('hsi', lang)}")
    if not dados_ok:
        return ("1 Começar, ou escolher a pasta em " + caminho("dados", lang)
                if pt else
                "1 Get started, or choose the folder in " + caminho("dados", lang))
    return "[R] rodar a análise" if pt else "[R] run the analysis"


# ---------------------------------------------------------------------------
# Ajuda curta por tela: (o que faz, quando usar, o que fazer depois)
# ---------------------------------------------------------------------------
AJUDA_TELAS: Dict[str, Dict[str, Tuple[str, str, str]]] = {
    "menu_principal": {
        "PT": ("Ponto de partida. Cada número abre uma etapa do trabalho, na ordem em que costuma ser feita.",
               "Sempre que não souber por onde seguir.",
               "Primeira vez: 1 Começar. Já tem dados: 2 Preparar dados, depois [R]."),
        "EN": ("Starting point. Each number opens one step of the work, in the order it is usually done.",
               "Whenever you are not sure where to go next.",
               "First time: 1 Get started. Have data already: 2 Prepare data, then [R]."),
    },
    "comecar": {
        "PT": ("Caminhos para o primeiro contato: um exemplo pronto, um perfil para a sua amostra ou um roteiro de decisões.",
               "Na primeira vez, ou para conferir se a instalação funciona.",
               "Depois do exemplo, aponte seus dados em Preparar dados › Dados."),
        "EN": ("Ways in for a first contact: a ready example, a profile for your sample or a decision guide.",
               "The first time, or to check that the installation works.",
               "After the example, point to your data in Prepare data › Data."),
    },
    "preparar": {
        "PT": ("Tudo o que vem antes do modelo: onde estão os dados, que técnica gerou os espectros, como tratá-los e se o delineamento é seguro.",
               "Antes da primeira análise e sempre que mudar de conjunto de dados.",
               "Rode a verificação de delineamento (item 5) antes de [R]."),
        "EN": ("Everything before the model: where the data is, which technique produced the spectra, how to treat them and whether the design is safe.",
               "Before the first analysis and whenever the dataset changes.",
               "Run the design check (item 5) before [R]."),
    },
    "modelar": {
        "PT": ("O que o modelo deve responder (classificar, discriminar ou quantificar) e como ele é validado.",
               "Depois de preparar os dados. Os padrões já servem para a maioria dos casos.",
               "Confira a Validação e rode com [R]."),
        "EN": ("What the model must answer (classify, discriminate or quantify) and how it is validated.",
               "After preparing the data. The defaults fit most cases.",
               "Check Validation and run with [R]."),
    },
    "sistema": {
        "PT": ("Salvar e carregar configurações, nome da execução, modo, idioma e informações do Guaraci.",
               "Para guardar uma configuração que deu certo ou reabrir uma antiga.",
               "Salve antes de sair se quiser reaproveitar a configuração."),
        "EN": ("Save and load configurations, run name, mode, language and information about Guaraci.",
               "To keep a configuration that worked or reopen an old one.",
               "Save before quitting if you want to reuse the configuration."),
    },
    "exemplo": {
        "PT": ("Roda o pipeline inteiro com espectros sintéticos. Nenhum dado seu é usado.",
               "Para ver o que o Guaraci entrega antes de preparar os seus dados.",
               "Abra a pasta GUARACI_Demo e veja as figuras e o relatório."),
        "EN": ("Runs the whole pipeline on synthetic spectra. None of your data is used.",
               "To see what Guaraci delivers before preparing your own data.",
               "Open the GUARACI_Demo folder and look at the figures and report."),
    },
    "perfis": {
        "PT": ("Configurações prontas por tipo de amostra e objetivo. Aplicar um perfil ajusta faixa, pré-processamento e validação de uma vez.",
               "Quando sua amostra se parece com um dos perfis listados.",
               "Depois de aplicar, confira a pasta de dados e rode com [R]."),
        "EN": ("Ready configurations by sample type and goal. Applying one sets range, preprocessing and validation at once.",
               "When your sample resembles one of the listed profiles.",
               "After applying, check the data folder and run with [R]."),
    },
    "passo_a_passo": {
        "PT": ("Perguntas curtas que levam ao método e às configurações adequadas ao seu objetivo.",
               "Quando não sabe qual nível de análise ou método escolher.",
               "Aplique o que foi sugerido em Modelar e validar."),
        "EN": ("Short questions that lead to the method and settings that fit your goal.",
               "When you do not know which analysis level or method to pick.",
               "Apply what was suggested in Model and validate."),
    },
    "projeto": {
        "PT": ("Pasta de entrada, pasta de saída e nome desta análise.",
               "Logo no início, para dizer onde estão os dados e onde gravar os resultados.",
               "Siga para Dados."),
        "EN": ("Input folder, output folder and name of this analysis.",
               "At the very start, to say where the data is and where to write results.",
               "Go on to Data."),
    },
    "dados": {
        "PT": ("Como ler os seus dados: formato de entrada, perfil da matriz, colunas de classe e de teor, faixa espectral e classes a excluir.",
               "Sempre que trocar de conjunto de dados ou de formato.",
               "Escolha a Técnica analítica e depois verifique o delineamento."),
        "EN": ("How to read your data: input format, matrix profile, class and content columns, spectral range and classes to exclude.",
               "Whenever the dataset or its format changes.",
               "Pick the Analytical technique, then check the design."),
    },
    "tecnica": {
        "PT": ("Técnica que gerou os espectros. Escolher uma ajusta faixa, pré-processamento e formato de leitura.",
               "Antes da primeira análise com um equipamento novo.",
               "Confira o Pré-processamento sugerido."),
        "EN": ("Technique that produced the spectra. Picking one sets range, preprocessing and reading format.",
               "Before the first analysis with a new instrument.",
               "Check the suggested Preprocessing."),
    },
    "preproc": {
        "PT": ("Tratamentos aplicados aos espectros antes do modelo (MSC, SNV, Savitzky-Golay, centragem).",
               "Quando o padrão da técnica não serve ou você quer comparar alternativas.",
               "Volte e rode com [R]."),
        "EN": ("Treatments applied to the spectra before modelling (MSC, SNV, Savitzky-Golay, centring).",
               "When the technique default does not fit or you want to compare alternatives.",
               "Go back and run with [R]."),
    },
    "auditoria": {
        "PT": ("Verifica o delineamento sem rodar o modelo: agrupamento por amostra física, confundimento classe × sessão, duplicatas, N insuficiente e faixa de validação.",
               "Antes de confiar em qualquer número. É o que impede um resultado inflado por vazamento.",
               "Corrija os itens críticos e rode com [R]."),
        "EN": ("Checks the study design without running the model: grouping by physical sample, class × session confounding, duplicates, insufficient N and validation range.",
               "Before trusting any number. It is what prevents a result inflated by leakage.",
               "Fix the critical items, then run with [R]."),
    },
    "planejamento": {
        "PT": ("Calcula quantas amostras coletar por classe e gera um plano de sessões com ordem de leitura aleatorizada.",
               "Antes de ir ao laboratório, quando ainda dá para mudar a coleta.",
               "Siga o plano e depois aponte os dados em Preparar dados › Dados."),
        "EN": ("Calculates how many samples to collect per class and builds a session plan with randomised reading order.",
               "Before going to the lab, while the collection can still change.",
               "Follow the plan, then point to the data in Prepare data › Data."),
    },
    "selecao_amostras": {
        "PT": ("Escolhe quais amostras de um CSV vão para calibração com Kennard-Stone, Duplex ou SPXY. Não altera o CSV original.",
               "Quando já mediu as amostras e quer um conjunto de calibração representativo.",
               "Use o CSV gerado como entrada em Dados."),
        "EN": ("Chooses which samples from a CSV go into calibration with Kennard-Stone, Duplex or SPXY. Never alters the original CSV.",
               "When the samples are measured and you want a representative calibration set.",
               "Use the generated CSV as input in Data."),
    },
    "codificacao": {
        "PT": ("Mapeia o código no início do nome de cada arquivo .dx para o nome da classe usado nos resultados.",
               "Só se os seus arquivos .dx seguem o padrão COD-DD-MM-AAAA_Tn.dx.",
               "Cadastre os códigos e volte a Dados."),
        "EN": ("Maps the code at the start of each .dx file name to the class name used in the results.",
               "Only if your .dx files follow the COD-DD-MM-YYYY_Tn.dx pattern.",
               "Register the codes and go back to Data."),
    },
    "objetivo_lvs": {
        "PT": ("Nível de análise (o que o modelo responde) e número máximo de variáveis latentes.",
               "Sempre: é a primeira decisão de modelagem.",
               "Confira a Validação."),
        "EN": ("Analysis level (what the model answers) and maximum number of latent variables.",
               "Always: it is the first modelling decision.",
               "Check Validation."),
    },
    "validacao": {
        "PT": ("Como o modelo é testado: fração de teste externo e validação cruzada por grupo (GroupKFold), que impede vazamento entre réplicas.",
               "Sempre confira. Desligar a validação por grupo infla os resultados.",
               "Rode com [R]."),
        "EN": ("How the model is tested: external test fraction and group-aware cross-validation (GroupKFold), which prevents leakage between replicates.",
               "Always check it. Turning group-aware validation off inflates the results.",
               "Run with [R]."),
    },
    "metodos_classe": {
        "PT": ("Métodos adicionais de classe: DD-SIMCA (autenticação de uma classe) e OPLS-DA (separação em componente preditiva e ortogonal).",
               "Para autenticação de uma classe-alvo ou para interpretar a separação entre duas classes.",
               "Rode com [R] e veja as figuras do método."),
        "EN": ("Additional class methods: DD-SIMCA (one-class authentication) and OPLS-DA (predictive and orthogonal split).",
               "For authenticating one target class or interpreting the split between two classes.",
               "Run with [R] and look at the method's figures."),
    },
    "selecao_variaveis": {
        "PT": ("Seleção de regiões e variáveis espectrais: iPLS, VIP e Selectivity Ratio juntos, mais SPA, AG, CARS e UVE.",
               "Quando quer saber quais regiões do espectro sustentam o modelo.",
               "Rode com [R] e compare com o modelo sem seleção."),
        "EN": ("Selection of spectral regions and variables: iPLS, VIP and Selectivity Ratio together, plus SPA, GA, CARS and UVE.",
               "When you want to know which spectral regions support the model.",
               "Run with [R] and compare with the model without selection."),
    },
    "modulos_pesados": {
        "PT": ("Comparações extras que demoram: benchmark contra outros algoritmos, Monte Carlo e SHAP.",
               "Para um artigo ou relatório completo, com tempo e memória disponíveis.",
               "Confira o Hardware antes de ligar e rode com [R]."),
        "EN": ("Extra comparisons that take time: benchmark against other algorithms, Monte Carlo and SHAP.",
               "For a paper or full report, with time and memory available.",
               "Check Hardware before enabling, then run with [R]."),
    },
    "predicao": {
        "PT": ("Aplica um modelo já treinado (.joblib) a espectros novos e indica se cada amostra está dentro do domínio do modelo.",
               "Depois de treinar e salvar um modelo, para analisar amostras desconhecidas.",
               "Abra o CSV de resultados gerado."),
        "EN": ("Applies an already trained model (.joblib) to new spectra and flags whether each sample is inside the model's domain.",
               "After training and saving a model, to analyse unknown samples.",
               "Open the generated results CSV."),
    },
    "visualizacao": {
        "PT": ("Aparência das figuras: detalhamento, marcadores, elipses, formato, resolução, paleta, fonte, grade e transparência.",
               "Antes de gerar figuras para apresentação ou publicação.",
               "Rode com [R] para gerar as figuras com o novo visual."),
        "EN": ("Figure appearance: detail, markers, ellipses, format, resolution, palette, font, grid and transparency.",
               "Before producing figures for a talk or paper.",
               "Run with [R] to produce the figures with the new look."),
    },
    "salvar": {
        "PT": ("Grava a configuração atual com um nome, para reabrir depois.",
               "Quando uma configuração deu certo.",
               "Use Carregar configuração para reabri-la."),
        "EN": ("Stores the current configuration under a name, to reopen later.",
               "When a configuration worked.",
               "Use Load configuration to reopen it."),
    },
    "carregar": {
        "PT": ("Reabre uma configuração salva antes.", "Para repetir uma análise.",
               "Confira a pasta de dados e rode com [R]."),
        "EN": ("Reopens a previously saved configuration.", "To repeat an analysis.",
               "Check the data folder and run with [R]."),
    },
    "nome_execucao": {
        "PT": ("Nome usado na pasta de resultados desta execução.",
               "Para encontrar os resultados depois.", "Rode com [R]."),
        "EN": ("Name used for this run's results folder.",
               "To find the results later.", "Run with [R]."),
    },
    "modo": {
        "PT": ("Alterna entre o modo Básico (só o essencial) e o Avançado (todas as opções). O mesmo que a tecla [M].",
               "Quando precisa de uma opção que não aparece.", "Volte à tela que estava usando."),
        "EN": ("Switches between Basic mode (essentials only) and Advanced (all options). Same as the [M] key.",
               "When you need an option that is not shown.", "Go back to the screen you were using."),
    },
    "idioma": {
        "PT": ("Alterna o idioma da interface entre português e inglês. O mesmo que a tecla [I].",
               "Quando preferir o outro idioma.", "Volte à tela que estava usando."),
        "EN": ("Switches the interface language between Portuguese and English. Same as the [I] key.",
               "When you prefer the other language.", "Go back to the screen you were using."),
    },
    "sobre": {
        "PT": ("Diferenciais, autoria, como citar e referências do Guaraci.",
               "Para citar o Guaraci ou entender o que o diferencia.", "Nada a fazer depois."),
        "EN": ("Differentiators, authorship, how to cite and references of Guaraci.",
               "To cite Guaraci or understand what sets it apart.", "Nothing to do afterwards."),
    },
    "hardware": {
        "PT": ("Memória, processador e o que esta máquina aguenta rodar.",
               "Antes de ligar os módulos pesados.", "Ajuste os Módulos pesados conforme a recomendação."),
        "EN": ("Memory, processor and what this machine can handle.",
               "Before enabling the heavy modules.", "Adjust the Heavy modules as recommended."),
    },
    "tecnicas_especiais": {
        "PT": ("ASCA, EPO/GLSW, MCR-ALS e fusão multibloco: técnicas exploratórias, fora do pipeline automático.",
               "Para efeitos de fatores experimentais (ASCA), remover interferências (EPO/GLSW), resolver misturas (MCR-ALS) ou combinar dois blocos de dados.",
               "Leia a ressalva de cada técnica antes de usar o resultado."),
        "EN": ("ASCA, EPO/GLSW, MCR-ALS and multiblock fusion: exploratory techniques, outside the automatic pipeline.",
               "For experimental factor effects (ASCA), removing interferences (EPO/GLSW), resolving mixtures (MCR-ALS) or combining two data blocks.",
               "Read each technique's caveat before using the result."),
    },
    "hsi": {
        "PT": ("Pipeline próprio para cubos hiperespectrais (ENVI .hdr/.bin, uma subpasta por classe), com mapa espacial por pixel.",
               "Quando seus dados são imagens hiperespectrais, não espectros pontuais.",
               "Veja o mapa e a confiança por objeto na pasta de resultados."),
        "EN": ("Dedicated pipeline for hyperspectral cubes (ENVI .hdr/.bin, one subfolder per class), with a per-pixel spatial map.",
               "When your data are hyperspectral images, not point spectra.",
               "Look at the map and per-object confidence in the results folder."),
    },
}


def ajuda_tela(tela_id: str, lang: str) -> Optional[Tuple[str, str, str]]:
    d = AJUDA_TELAS.get(tela_id)
    if d is None:
        return None
    return d.get("EN" if lang == "EN" else "PT")


def ids_telas() -> List[str]:
    """Todas as telas e grupos que precisam de ajuda registrada."""
    ids = ["menu_principal"]
    for g in MENU:
        if not g.direto:
            ids.append(g.id)
        ids.extend(it.id for it in g.itens)
    return ids
