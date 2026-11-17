"""catalogo_perguntas.py -- o que cada PERGUNTA solta do menu quer dizer (tecla "?" num prompt).

Os prompts que nao sao lista numerada de um campo (nome de classe, numero de componentes, metodo do
EPO/GLSW...) respondem "?" com o texto daqui, escolhido por um trecho do proprio rotulo da pergunta
(sem acento, minusculo). Pergunta sem texto proprio mostra o rotulo e a ajuda da tela. ASCII por convencao do CLI.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

Texto = Tuple[str, str]      # (PT, EN)

#: (trechos do rotulo em PT ou EN, texto). O primeiro que casar vale.
PERGUNTAS: List[Tuple[Tuple[str, ...], Texto]] = [
    (("classes (separadas", "classes (comma"),
     ("Os nomes das classes que voce vai coletar, separados por virgula (ex.: A,B,C). O planejamento calcula quantas amostras fisicas e sessoes cada uma precisa.",
      "The names of the classes you will collect, comma-separated (e.g. A,B,C). The plan computes how many physical samples and sessions each one needs.")),
    (("numero de sessoes", "number of collection sessions"),
     ("Quantas sessoes de coleta (dias, lotes, reagentes novos) voce pretende fazer. Sessoes diferentes sao o que a validacao trata como independente; Enter = 2.",
      "How many collection sessions (days, batches, fresh reagents) you intend to run. Different sessions are what validation treats as independent; Enter = 2.")),
    (("alvo: (c)onformal", "target: (c)onformal"),
     ("C = predicao conformal (identificar entre varias classes, com taxa de erro alpha); D = DD-SIMCA (pureza de cada especie, com cobertura-alvo).",
      "C = conformal prediction (identify among several classes, with error rate alpha); D = DD-SIMCA (purity of each species, with a target coverage).")),
    (("alpha desejado", "desired alpha"),
     ("Taxa de erro aceita pela predicao conformal (ex.: 0.05 = 5%). Menor alpha pede mais amostras por classe.",
      "Error rate accepted by conformal prediction (e.g. 0.05 = 5%). A smaller alpha needs more samples per class.")),
    (("cobertura-alvo", "target coverage"),
     ("Fracao das amostras genuinas que o DD-SIMCA deve aceitar (ex.: 0.90 = 90%).",
      "Fraction of genuine samples DD-SIMCA should accept (e.g. 0.90 = 90%).")),
    (("coluna que identifica a amostra fisica", "column identifying the physical sample"),
     ("Coluna com o identificador da amostra fisica (replicas da mesma amostra repetem o valor). Serve para nao separar replicas entre calibracao e validacao. Enter usa a coluna detectada.",
      "Column with the physical-sample identifier (replicates of one sample repeat the value). It keeps replicates from being split between calibration and validation. Enter uses the detected column.")),
    (("fracao para calibracao", "calibration fraction"),
     ("Parte das amostras que vai para a calibracao; o resto fica para validar (entre 0 e 1; Enter = 0.7).",
      "Share of the samples that goes to calibration; the rest is kept for validation (between 0 and 1; Enter = 0.7).")),
    (("metodo (1=epo", "method (1=epo"),
     ("1 = EPO: projeta fora do subespaco do fator indesejado, calculado de pares de diferenca. 2 = GLSW: pesa os espectros contra a variacao do fator (mais suave). Enter = EPO.",
      "1 = EPO: projects out the unwanted-factor subspace computed from difference pairs. 2 = GLSW: weights the spectra against the factor's variation (smoother). Enter = EPO.")),
    (("numero de componentes", "number of components"),
     ("Quantos componentes (fatores) manter. Poucos perdem informacao; muitos ajustam ruido. Enter usa o numero indicado entre parenteses.",
      "How many components (factors) to keep. Too few lose information; too many fit noise. Enter uses the number shown in brackets.")),
    (("usar restricao de correlacao", "use correlation restriction"),
     ("Faz um dos componentes do MCR-ALS acompanhar o teor conhecido (correlacao com a referencia). Use quando um componente deve representar o analito.",
      "Makes one MCR-ALS component follow the known content (correlation with the reference). Use when one component should represent the analyte.")),
    (("indice do componente-alvo", "target component index"),
     ("Qual componente (contando de 0) deve acompanhar o teor. Enter = 0, o primeiro.",
      "Which component (counting from 0) should follow the content. Enter = 0, the first.")),
    (("segmentos", "segments"),
     ("Em quantos trechos o COW divide o cromatograma para alinhar. Mais trechos seguem deslocamentos locais, mas podem deformar picos; Enter = 30.",
      "Into how many pieces COW splits the chromatogram to align it. More pieces follow local shifts but can distort peaks; Enter = 30.")),
    (("folga em pontos", "slack in points"),
     ("O quanto, em pontos, cada trecho pode se esticar ou encolher no alinhamento COW; Enter = 10.",
      "How much, in points, each piece may stretch or shrink in the COW alignment; Enter = 10.")),
    (("correcao (emsc/osc)", "correction (emsc/osc)"),
     ("EMSC: corrige espalhamento e linha de base com polinomio. OSC: remove direcoes sem relacao com o alvo. A tela so aceita a correcao se ela vencer nas mesmas particoes. Enter = emsc.",
      "EMSC: corrects scatter and a polynomial baseline. OSC: removes directions unrelated to the target. The screen only accepts the correction if it wins on the same partitions. Enter = emsc.")),
    (("nome do bloco", "block name"),
     ("Um nome curto para este bloco de dados (ex.: NIR, Raman). Aparece nas tabelas e figuras da fusao.",
      "A short name for this data block (e.g. NIR, Raman). Appears in the fusion tables and figures.")),
    (("modo de entrada", "input mode"),
     ("Tipo de dado deste bloco: dx (pasta de JCAMP), csv (tabela), spc, sp, opus, rmn, hplc ou gcms. Veja [2.3] Dados, campo Modo de entrada, para o que cada um le.",
      "Data type of this block: dx (JCAMP folder), csv (table), spc, sp, opus, rmn, hplc or gcms. See [2.3] Data, Input mode field, for what each one reads.")),
    (("pasta (ou arquivo csv)", "folder (or csv file)"),
     ("Onde estao os dados deste bloco: uma pasta (dx e demais formatos) ou o arquivo CSV. As amostras dos dois blocos devem ser as mesmas, na mesma ordem.",
      "Where this block's data are: a folder (dx and other formats) or the CSV file. The samples of both blocks must be the same, in the same order.")),
    (("coluna de classe", "class column"),
     ("Coluna do CSV que diz a classe de cada amostra.", "CSV column that gives each sample's class.")),
    (("coluna de teor", "content column", "coluna alvo", "target column"),
     ("Coluna do CSV com o teor (valor numerico) a quantificar.", "CSV column with the content (numeric value) to quantify.")),
    (("pasta de destino", "destination folder"),
     ("Onde o exemplo sera criado. Enter usa a pasta sugerida entre colchetes.", "Where the example will be created. Enter uses the suggested folder in brackets.")),
    (("novo identificador", "new identifier"),
     ("Identificador curto desta execucao; entra no nome da pasta de resultados. '-' limpa, Enter mantem.",
      "Short identifier of this run; goes into the results folder name. '-' clears, Enter keeps.")),
    (("caminho do csv (colunas: codigo", "csv path (columns: code"),
     ("CSV com duas colunas, codigo e nome da classe (separador , ou ;). Cada linha vira um codigo cadastrado.",
      "CSV with two columns, code and class name (separator , or ;). Each row becomes a registered code.")),
    (("pasta com seus cubos", "folder with your"),
     ("Pasta com os dados desta analise (uma subpasta por classe, no caso de cubos ou fotos).",
      "Folder with this analysis' data (one subfolder per class, for cubes or photos).")),
    (("pasta de entrada", "input folder"),
     ("Pasta onde estao os arquivos de dados. Uma subpasta por classe (ou os arquivos soltos, com a classe no nome).",
      "Folder where the data files are. One subfolder per class (or loose files with the class in the name).")),
]


def _normalizar(t: str) -> str:
    t = re.sub(r"\[/?[^\]]*\]", " ", t)         # markup do Rich
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", t).strip().lower()


def texto_pergunta(prompt: str, lang: str = "PT") -> Optional[str]:
    """Explicacao da pergunta que o prompt faz, ou None."""
    p = _normalizar(prompt)
    if not p:
        return None
    for trechos, texto in PERGUNTAS:
        if any(t in p for t in trechos):
            return texto[1 if lang == "EN" else 0]
    return None
