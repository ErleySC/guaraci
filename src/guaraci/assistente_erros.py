"""assistente_erros.py -- codigos estaveis para erros de execucao, com causa provavel e o que fazer.

Modelo: o `rustc --explain` (rustc-dev-guide, Diagnostics, https://rustc-dev-guide.rust-lang.org/diagnostics):
cada erro conhecido tem um codigo que nao muda (GUA-Ennn) e uma explicacao longa. O codigo e' achado pela
excecao (tipo + texto), entao o mesmo diagnostico vale para um erro capturado pela CLI e para um texto colado
pelo usuario no assistente. Padroes casam mensagens reais do scikit-learn/numpy/Python (conferidas com
scikit-learn 1.9) e do proprio Guaraci (dados_io, pipeline).

Uso:
    codigos("ValueError: Input X contains NaN.")   -> ["GUA-E001"]
    explicar("GUA-E001")                          -> texto longo
    diagnosticar_excecao(exc)                     -> texto pronto para o painel de erro
Teste: tests/test_assistente_erros.py.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["CODIGOS", "codigos", "explicar", "diagnosticar_excecao", "eh_codigo"]

# codigo -> titulo, padroes (regex, sem caixa), causa, fazer (PT, EN)
CODIGOS: Dict[str, Dict[str, Any]] = {
    "GUA-E001": dict(
        titulo=("Valores faltantes (NaN) nos dados", "Missing values (NaN) in the data"),
        padroes=(r"contains nan", r"input contains nan"),
        causa=("Ha celula vazia, texto ou divisao por zero em alguma amostra; PLS e PCA nao aceitam NaN.",
               "There is an empty cell, text or a division by zero in some sample; PLS and PCA do not accept NaN."),
        fazer=("Procure celulas vazias no arquivo de origem. Se a falta e' real, retire a amostra ou a variavel; "
               "se veio da autoescala, retire variaveis constantes cortando a faixa.",
               "Look for empty cells in the source file. If the gap is real, drop the sample or the variable; "
               "if it came from autoscaling, drop constant variables by cutting the range.")),
    "GUA-E002": dict(
        titulo=("Texto numa coluna numerica", "Text in a numeric column"),
        padroes=(r"could not convert string to float", r"valores espectrais nao numericos", r"tem valores nao\s+numericos"),
        causa=("Ha texto, unidade ou simbolo numa celula que deveria ser numero. O leitor de CSV do Guaraci ja aceita "
               "virgula decimal; o que sobra e' celula com texto.",
               "There is text, a unit or a symbol in a cell that should be a number. Guaraci's CSV reader already accepts "
               "comma decimals; what is left is a cell with text."),
        fazer=("A mensagem original mostra a linha e a coluna; corrija essa celula no arquivo de origem ou retire a amostra.",
               "The original message shows the row and column; fix that cell in the source file or drop the sample.")),
    "GUA-E003": dict(
        titulo=("Codificacao de texto do arquivo", "File text encoding"),
        padroes=(r"codec can't decode", r"unicodedecodeerror"),
        causa=("O arquivo nao esta em UTF-8 (comum em CSV salvo pelo Excel no Windows) ou e' binario.",
               "The file is not UTF-8 (common for CSV saved by Excel on Windows) or is binary."),
        fazer=("Salve como 'CSV UTF-8' no Excel. Se o arquivo for de um equipamento, exporte como JCAMP-DX ou texto.",
               "Save as 'CSV UTF-8' in Excel. If the file comes from an instrument, export it as JCAMP-DX or text.")),
    "GUA-E004": dict(
        titulo=("Memoria insuficiente", "Not enough memory"),
        padroes=(r"memoryerror", r"unable to allocate"),
        causa=("A matriz (amostras x variaveis x repeticoes) nao cabe na memoria desta maquina.",
               "The matrix (samples x variables x repeats) does not fit in this machine's memory."),
        fazer=("Corte a faixa espectral, reduza n_permutacoes e repeticoes da validacao, ou feche outros programas; "
               "`guaraci doctor` mostra a memoria disponivel.",
               "Cut the spectral range, lower n_permutacoes and validation repeats, or close other programs; "
               "`guaraci doctor` shows available memory.")),
    "GUA-E005": dict(
        titulo=("Mais dobras do que grupos", "More folds than groups"),
        padroes=(r"cannot have number of splits n_splits=\d+ greater than the number of groups",
                 r"validacao cruzada impossivel"),
        causa=("A validacao por grupo precisa de pelo menos uma unidade independente por dobra (e por classe).",
               "Group validation needs at least one independent unit per fold (and per class)."),
        fazer=("Baixe cv_dobras, diminua holdout_fracao ou meca mais amostras fisicas. Confira se a coluna de grupo "
               "nao junta amostras diferentes no mesmo grupo.",
               "Lower cv_dobras, reduce holdout_fracao or measure more physical samples. Check that the group column "
               "does not merge different samples into one group.")),
    "GUA-E006": dict(
        titulo=("Componentes demais para os dados", "Too many components for the data"),
        padroes=(r"n_components.{0,3} upper bound is \d+", r"n_components=\d+ must be between"),
        causa=("O numero de componentes pedido passa do posto da matriz (minimo entre amostras-1 e variaveis).",
               "The requested number of components exceeds the matrix rank (minimum of samples-1 and variables)."),
        fazer=("Baixe max_lvs (ou os componentes do DD-SIMCA) para menos que o numero de amostras do menor conjunto de treino.",
               "Lower max_lvs (or the DD-SIMCA components) below the number of samples in the smallest training set.")),
    "GUA-E007": dict(
        titulo=("Nenhuma variavel depois do corte", "No variable left after the cut"),
        padroes=(r"found array with 0 feature", r"deixou apenas \d+ de \d+ pontos"),
        causa=("A faixa escolhida (faixa_min_cm/faixa_max_cm) nao cruza o eixo dos dados, ou o eixo esta em outra "
               "unidade (nm, ppm, minutos...).",
               "The chosen range (faixa_min_cm/faixa_max_cm) does not overlap the data axis, or the axis is in "
               "another unit (nm, ppm, minutes...)."),
        fazer=("Confira faixa_min_cm e faixa_max_cm contra o eixo dos arquivos. Para uma tabela de variaveis com "
               "NOMES no cabecalho (nao espectral), o eixo e' a posicao 1..k: use faixa 0 a 999999.",
               "Check faixa_min_cm and faixa_max_cm against the file axis. For a table of variables with NAMES "
               "in the header (non-spectral), the axis is the position 1..k: use range 0 to 999999.")),
    "GUA-E008": dict(
        titulo=("Pasta ou arquivo de dados nao encontrado", "Data folder or file not found"),
        padroes=(r"path does not exist", r"csv nao encontrado", r"contains no known spectral files",
                 r"no valid spectra loaded", r"pasta nao encontrada", r"pasta nao existe",
                 r"nenhum arquivo candidato", r"nenhuma pasta-amostra", r"nenhuma amostra .{0,40}valida carregada"),
        causa=("O caminho em pasta_dados ou arquivo_csv nao existe, ou a pasta nao tem arquivos de formato conhecido.",
               "The path in pasta_dados or arquivo_csv does not exist, or the folder has no files of a known format."),
        fazer=("Confira o caminho (relativo a pasta onde o Guaraci foi aberto) e a extensao dos arquivos; "
               "`guaraci exemplo` cria uma pasta de exemplo para comparar.",
               "Check the path (relative to the folder Guaraci was opened in) and the file extension; "
               "`guaraci exemplo` creates an example folder to compare.")),
    "GUA-E009": dict(
        titulo=("Formato binario do fabricante", "Vendor binary format"),
        padroes=(r"proprietary binary format", r"unrecognized binary file", r"abb bomem horizon mb"),
        causa=("O arquivo esta no formato interno do software do equipamento.",
               "The file is in the instrument software's internal format."),
        fazer=("Exporte do software original como JCAMP-DX (.dx) ou texto (.txt/.csv).",
               "Export from the original software as JCAMP-DX (.dx) or text (.txt/.csv).")),
    "GUA-E010": dict(
        titulo=("Caminho longo demais (Windows)", "Path too long (Windows)"),
        padroes=(r"winerror 206", r"passaria de 260"),
        causa=("O Windows limita caminhos a 260 caracteres e a pasta de saida ja esta perto disso.",
               "Windows limits paths to 260 characters and the output folder is already close to that."),
        fazer=("Use uma pasta de saida curta, como C:\\guaraci\\resultados, fora de pasta sincronizada com a nuvem.",
               "Use a short output folder, such as C:\\guaraci\\resultados, outside any cloud-synced folder.")),
    "GUA-E011": dict(
        titulo=("Coluna do CSV nao existe", "CSV column does not exist"),
        padroes=(r"a coluna de (classe|concentracao|grupo) '.*' nao existe",),
        causa=("O nome em coluna_classe, coluna_concentracao ou coluna_grupo nao bate com o cabecalho do CSV.",
               "The name in coluna_classe, coluna_concentracao or coluna_grupo does not match the CSV header."),
        fazer=("Copie o nome exatamente como aparece na lista de colunas da mensagem (maiusculas e espacos contam).",
               "Copy the name exactly as it appears in the message's column list (case and spaces matter).")),
    "GUA-E012": dict(
        titulo=("Arquivo bloqueado ou sem permissao", "File locked or no permission"),
        padroes=(r"permission denied", r"errno 13", r"winerror 32", r"controle de aplicativo", r"winerror 4551"),
        causa=("Outro programa (Excel, sincronizacao com a nuvem, antivirus) esta com o arquivo aberto, ou a pasta nao permite escrita.",
               "Another program (Excel, cloud sync, antivirus) has the file open, or the folder does not allow writing."),
        fazer=("Feche o arquivo no Excel, pause a sincronizacao com a nuvem ou escolha outra pasta de saida.",
               "Close the file in Excel, pause cloud sync or choose another output folder.")),
    "GUA-E013": dict(
        titulo=("Uma classe so'", "A single class"),
        padroes=(r"contains only one class", r"number of classes has to be greater than one"),
        causa=("Depois da leitura (ou do corte do holdout) sobrou uma classe; classificar exige pelo menos duas.",
               "After loading (or the holdout cut) a single class is left; classification needs at least two."),
        fazer=("Confira se as subpastas ou a coluna de classe tem mais de um valor; para uma classe so', use o "
               "DD-SIMCA (autenticacao).",
               "Check that the subfolders or the class column have more than one value; for a single class, use "
               "DD-SIMCA (authentication).")),
    "GUA-E014": dict(
        titulo=("CSV com menos de 2 colunas de variaveis", "CSV with fewer than 2 variable columns"),
        padroes=(r"menos de 2 colunas numericas de variaveis",),
        causa=("Depois de tirar classe, grupo, teor e colunas de texto, sobrou menos de 2 coluna numerica no CSV. "
               "Cabecalhos numericos viram o eixo (espectro); cabecalhos com nomes viram variaveis nomeadas.",
               "After removing class, group, content and text columns, fewer than 2 numeric columns are left in the "
               "CSV. Numeric headers become the axis (spectrum); named headers become named variables."),
        fazer=("Confira o separador e o decimal do CSV, e se as variaveis estao em colunas (uma amostra por linha). "
               "Colunas de teor/lote/id nao contam como variavel.",
               "Check the CSV separator and decimal, and that the variables are in columns (one sample per row). "
               "Content/batch/id columns do not count as variables.")),
    "GUA-E015": dict(
        titulo=("Faixa excluida ou janela fora do eixo", "Excluded range or window outside the axis"),
        padroes=(r"faixas_excluir .{0,80}nao cruza", r"faixas_excluir .{0,80}removem todas as variaveis",
                 r"ims_rip_janela .{0,40}nao cruza"),
        causa=("faixas_excluir (ou a janela do RIP, ims_rip_janela) esta numa unidade diferente da do eixo dos dados "
               "(ex.: ppm contra pontos, nm contra cm-1), ou apaga todas as variaveis que sobraram da faixa usada.",
               "faixas_excluir (or the RIP window, ims_rip_janela) is in a unit different from the data axis "
               "(e.g. ppm vs points, nm vs cm-1), or it removes every variable left in the range in use."),
        fazer=("Escreva as faixas na unidade do eixo mostrada na mensagem, ou esvazie faixas_excluir (faixas_excluir: []). "
               "A tela [2.4] preenche o valor certo para RMN (ppm).",
               "Write the ranges in the axis unit shown in the message, or empty faixas_excluir (faixas_excluir: []). "
               "Screen [2.4] fills the right value for NMR (ppm).")),
}

_RE = {k: [re.compile(p, re.I) for p in v["padroes"]] for k, v in CODIGOS.items()}
_COD = re.compile(r"\bGUA-E\d{3}\b", re.I)


def _L(lang: str) -> int:
    return 1 if str(lang).upper() == "EN" else 0


def eh_codigo(texto: str) -> Optional[str]:
    m = _COD.search(texto or "")
    return m.group(0).upper() if m and m.group(0).upper() in CODIGOS else None


def _texto(exc_ou_texto: Any) -> str:
    if isinstance(exc_ou_texto, BaseException):
        return f"{type(exc_ou_texto).__name__}: {exc_ou_texto}"
    return str(exc_ou_texto or "")


def codigos(exc_ou_texto: Any) -> List[str]:
    """Codigos cujo padrao aparece na excecao ou no texto colado (ou o codigo digitado)."""
    t = _texto(exc_ou_texto)
    c = eh_codigo(t)
    if c:
        return [c]
    return [k for k, regs in _RE.items() if any(r.search(t) for r in regs)]


def explicar(codigo: str, lang: str = "PT") -> Optional[str]:
    d = CODIGOS.get((codigo or "").strip().upper())
    if d is None:
        return None
    L = _L(lang)
    rot = (("Causa provavel", "O que fazer"), ("Likely cause", "What to do"))[L]
    return f"{codigo.upper()}: {d['titulo'][L]}\n{rot[0]}: {d['causa'][L]}\n{rot[1]}: {d['fazer'][L]}"


def diagnosticar_excecao(exc_ou_texto: Any, lang: str = "PT") -> str:
    """Texto do painel de erro: a mensagem original e, para cada codigo reconhecido, causa e acao.
    Nunca esconde a mensagem original."""
    L = _L(lang)
    orig = _texto(exc_ou_texto)
    cods = codigos(exc_ou_texto)
    if not cods:
        return orig + ("\n\nErro sem codigo conhecido. Copie a mensagem acima em `guaraci assistente` ou num relato de erro."
                       if L == 0 else
                       "\n\nError with no known code. Paste the message above into `guaraci assistente` or a bug report.")
    corpo = "\n\n".join(explicar(c, lang) or c for c in cods)
    dica = ("Mais detalhes: guaraci assistente " if L == 0 else "More detail: guaraci assistente ") + cods[0]
    return f"{orig}\n\n{corpo}\n\n{dica}"


def lista(lang: str = "PT") -> List[Tuple[str, str]]:
    L = _L(lang)
    return [(k, v["titulo"][L]) for k, v in CODIGOS.items()]
