# -*- coding: utf-8 -*-
"""plano_coleta.py -- Distribuicao de amostras entre sessoes de coleta e
ordem de leitura aleatorizada (Bloco 10, `guaraci plan`).

POR QUE ESTE MODULO EXISTE
---------------------------
Duas formas de confundimento ja medidas neste projeto (classe x sessao e
ordem de leitura x teor, em conjuntos de dados do autor; sem script publicado) sao
EVITAVEIS no momento do PLANEJAMENTO da coleta, antes de virarem dado a
corrigir depois:

1. CLASSE x SESSAO: se uma sessao de coleta contem so' uma classe (ou
   uma classe fortemente dominante), qualquer deriva instrumental/
   temporal daquela sessao fica indistinguivel de efeito de classe --
   exatamente o confundimento que motivou `dados_io.session_from_mae_id`
   e a separacao treino/teste por sessao no Bloco 9b, so' que agora
   evitado na COLETA em vez de mitigado depois na analise.
2. ORDEM DE LEITURA x TEOR: se as amostras sao lidas em ordem crescente
   de teor (ou qualquer ordem correlacionada com uma variavel de
   interesse), qualquer deriva do instrumento ao longo da sessao (fonte
   esquentando, purga de CO2/H2O, sujeira acumulando na janela) entra no
   modelo como se fosse sinal quimico -- medido em sessao anterior como
   rho de Spearman proximo de 1 quando a ordem de leitura segue o teor
   dentro do bloco.

Este modulo distribui as amostras por sessao de forma BALANCEADA
(round-robin determinstico -- o objetivo e' balanceamento, nao
aleatoriedade nessa etapa) e ALEATORIZA a ordem de leitura DENTRO de
cada sessao (aqui sim, aleatoriedade e' o ponto).

Integra com `plano_amostral.py` (P1, Bloco 10): aquele modulo decide
QUANTAS amostras sao necessarias (`n_minimum_conformal`/
`ddsimca_sample_size_guidance`); este decide COMO distribui-las
entre sessoes e em que ordem le-las.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from guaraci.plano_amostral import (
    n_minimum_conformal,
    ddsimca_sample_size_guidance,
)

__all__ = [
    "CollectionItem",
    "CollectionPlan",
    "plan_collection",
    "plan_from_statistical_target",
    "lacunas_de_sessoes",
    "texto_lacunas",
    "export_markdown",
    "export_excel",
    "export_pdf",
]


@dataclass
class CollectionItem:
    """Uma AMOSTRA FISICA no plano -- ver alerta sobre replica tecnica em
    `plan_collection`. `ordem_na_sessao` ja' e' a ordem de leitura
    ALEATORIZADA (0-based), nao a ordem de alocacao."""

    classe: str
    replica_idx: int
    sessao: int
    ordem_na_sessao: int


@dataclass
class CollectionPlan:
    classes: List[str]
    n_por_classe: int
    n_sessoes: int
    itens: List[CollectionItem]
    alertas: List[str] = field(default_factory=list)

    def itens_da_sessao(self, sessao: int) -> List[CollectionItem]:
        return sorted((i for i in self.itens if i.sessao == sessao),
                       key=lambda i: i.ordem_na_sessao)


def plan_collection(classes: Sequence[str], n_por_classe: int,
                     n_sessoes: int, *, n_brancos_por_sessao: int = 1,
                     seed: Optional[int] = None) -> CollectionPlan:
    """Distribui `n_por_classe` amostras de cada classe entre `n_sessoes`
    sessoes, balanceado (round-robin), e aleatoriza a ordem de leitura
    DENTRO de cada sessao. Ver docstring do modulo para o motivo dos dois
    confundimentos que isso evita.

    `seed`: None usa entropia do sistema (plano de coleta real -- cada
    geracao deve ser genuinamente aleatoria); um inteiro fixo e' so' para
    reprodutibilidade em teste.
    """
    if n_por_classe < 1:
        raise ValueError("n_por_classe deve ser >= 1")
    if n_sessoes < 1:
        raise ValueError("n_sessoes deve ser >= 1")
    if not classes:
        raise ValueError("classes nao pode ser vazio")

    rng = np.random.default_rng(seed)
    itens: List[CollectionItem] = []

    for classe in classes:
        for replica_idx in range(n_por_classe):
            sessao = replica_idx % n_sessoes
            itens.append(CollectionItem(classe=classe, replica_idx=replica_idx,
                                         sessao=sessao, ordem_na_sessao=-1))

    for sessao in range(n_sessoes):
        idx_sessao = [i for i, item in enumerate(itens)
                       if item.sessao == sessao]
        ordem = rng.permutation(len(idx_sessao))
        for pos, idx in zip(ordem, idx_sessao):
            itens[idx].ordem_na_sessao = int(pos)

    alertas: List[str] = []
    if n_por_classe % n_sessoes != 0:
        resto = n_por_classe % n_sessoes
        alertas.append(
            f"n_por_classe ({n_por_classe}) nao e' multiplo de n_sessoes "
            f"({n_sessoes}): {resto} sessao(oes) recebem 1 amostra a mais "
            "de cada classe (balanceamento residual pequeno -- confira a "
            "tabela por sessao antes de coletar).")
    if n_sessoes == 1:
        alertas.append(
            "APENAS 1 SESSAO: nao ha' como separar efeito de classe de "
            "deriva instrumental/temporal -- o confundimento classe x "
            "sessao fica estruturalmente impossivel de descartar com "
            "este plano. Distribua a coleta em >=2 sessoes se possivel.")
    alertas.append(
        "REPLICA TECNICA != AMOSTRA FISICA: cada item deste plano e' UMA "
        "amostra fisica independente (novo preparo), nao uma releitura do "
        "mesmo vial. Replicas tecnicas (T1/T2/T3 da MESMA amostra) nao "
        "contam como amostras adicionais para fins de tamanho amostral "
        "-- leia cada item quantas vezes o "
        "protocolo do laboratorio pedir, mas isso nao aumenta o `n` "
        "usado para a garantia estatistica.")
    alertas.append(
        "BRANCOS/CONTROLES: inclua pelo menos "
        f"{max(1, n_brancos_por_sessao)} leitura(s) de branco/referencia "
        "por sessao, intercalada(s) ao longo da ordem de leitura (nao so' "
        "no inicio/fim) -- e' o que permite detectar deriva instrumental "
        "depois, sem precisar reler amostras.")

    return CollectionPlan(classes=list(classes), n_por_classe=n_por_classe,
                           n_sessoes=n_sessoes, itens=itens, alertas=alertas)


def plan_from_statistical_target(
        classes: Sequence[str], n_sessoes: int, *,
        alpha_conformal: Optional[float] = None,
        cobertura_ddsimca: Optional[float] = None,
        n_brancos_por_sessao: int = 1,
        seed: Optional[int] = None) -> Tuple[CollectionPlan, Dict[str, Any]]:
    """Combina `plano_amostral.py` (quanto coletar) com `plan_collection`
    (como distribuir/ordenar). Passe exatamente UM alvo estatistico:
    `alpha_conformal` (gate conformal, Identificar/agrupado) OU
    `cobertura_ddsimca` (DD-SIMCA por especie, com o teto de plato ja
    embutido em `ddsimca_sample_size_guidance` -- ver esse modulo
    para o motivo de nao prometer cobertura inalcancavel).

    Retorna (plano, metadados) -- metadados inclui `n_por_classe`,
    `origem` (qual alvo foi usado) e, se DD-SIMCA, a `DDSimcaGuidance`
    completa (para quem quiser inspecionar a ressalva original).
    """
    if (alpha_conformal is None) == (cobertura_ddsimca is None):
        raise ValueError(
            "passe exatamente um de alpha_conformal ou cobertura_ddsimca, "
            "nunca os dois nem nenhum")

    metadados: Dict[str, Any] = {}
    if alpha_conformal is not None:
        n = n_minimum_conformal(alpha_conformal)
        metadados["origem"] = f"conformal (alpha={alpha_conformal})"
        metadados["n_por_classe"] = n
    else:
        orientacao = ddsimca_sample_size_guidance(cobertura_ddsimca)
        if not orientacao.alcancavel:
            raise ValueError(
                f"cobertura-alvo {cobertura_ddsimca} NAO alcancavel via "
                f"DD-SIMCA -- {orientacao.ressalva}")
        n = orientacao.n_sugerido
        metadados["origem"] = f"DD-SIMCA (cobertura={cobertura_ddsimca})"
        metadados["n_por_classe"] = n
        metadados["orientacao_ddsimca"] = orientacao

    plano = plan_collection(classes, n, n_sessoes,
                             n_brancos_por_sessao=n_brancos_por_sessao,
                             seed=seed)
    plano.alertas.insert(
        0, f"n por classe ({n}) calculado via {metadados['origem']}.")
    if "orientacao_ddsimca" in metadados:
        plano.alertas.insert(1, metadados["orientacao_ddsimca"].ressalva)
    return plano, metadados


#: Sessoes independentes por classe para um resultado HONESTO (sem garantia): o
#: mesmo piso que o DD-SIMCA do GUARACI usa (`sensibilidade LOGO >= 0,90 com 5 ou
#: mais grupos`).
SESSOES_MINIMAS_HONESTO = 5


def lacunas_de_sessoes(rotulos: Sequence[str], mae_id: Optional[Sequence[str]],
                       alpha: float = 0.05,
                       minimo_honesto: int = SESSOES_MINIMAS_HONESTO
                       ) -> Dict[str, Any]:
    """Quantas sessoes de coleta INDEPENDENTES cada classe tem nos SEUS dados e
    quantas faltam para (a) um resultado honesto e (b) a garantia conformal em
    `alpha`. E' o plano de coleta a partir do que ja' foi medido.

    Devolve `{"classes": [{classe, sessoes, faltam_honesto, faltam_conformal}],
    "n_sessoes_total", "minimo_honesto", "minimo_conformal", "confundidas"}`.
    `confundidas` lista as sessoes que tem UMA classe so' (cada uma e'
    indistinguivel da classe). Sem `mae_id` nao ha' como contar sessoes:
    devolve `classes=[]`.
    """
    from guaraci.dados_io import session_from_mae_id
    minimo_conformal = n_minimum_conformal(alpha)
    if mae_id is None:
        return {"classes": [], "n_sessoes_total": 0, "minimo_honesto": minimo_honesto,
                "minimo_conformal": minimo_conformal, "confundidas": []}
    rot = [str(r) for r in rotulos]
    ses = [str(session_from_mae_id(m)) for m in mae_id]
    por_classe: Dict[str, set] = {}
    por_sessao: Dict[str, set] = {}
    for r, s in zip(rot, ses):
        por_classe.setdefault(r, set()).add(s)
        por_sessao.setdefault(s, set()).add(r)
    classes = [{"classe": c, "sessoes": len(por_classe[c]),
                "faltam_honesto": max(0, minimo_honesto - len(por_classe[c])),
                "faltam_conformal": max(0, minimo_conformal - len(por_classe[c]))}
               for c in sorted(por_classe)]
    return {"classes": classes, "n_sessoes_total": len(por_sessao),
            "minimo_honesto": minimo_honesto, "minimo_conformal": minimo_conformal,
            "confundidas": sorted(s for s, cs in por_sessao.items() if len(cs) == 1)}


def texto_lacunas(lac: Dict[str, Any], lang: str = "PT") -> List[str]:
    """Linhas de texto (PT/EN) do plano de coleta a partir dos dados. Lista vazia
    quando nao ha' o que dizer (sem mae_id)."""
    cls = lac["classes"]
    if not cls:
        return []
    pt = lang == "PT"
    h, c = lac["minimo_honesto"], lac["minimo_conformal"]
    n_conf = len(lac["confundidas"])
    n_tot = lac["n_sessoes_total"]
    fracas = [x for x in cls if x["faltam_honesto"] > 0]
    pior = max(x["faltam_honesto"] for x in cls)
    linhas = []
    if n_conf == n_tot:
        linhas.append(
            f"Todas as {n_tot} sessoes tem UMA classe so': o modelo nao consegue separar classe de sessao. "
            "Meca varias classes em cada sessao." if pt else
            f"All {n_tot} sessions hold a SINGLE class: the model cannot tell class from session. "
            "Measure several classes in each session.")
    elif n_conf:
        linhas.append(
            f"{n_conf} de {n_tot} sessoes tem uma classe so'. Misture classes nessas sessoes." if pt else
            f"{n_conf} of {n_tot} sessions hold a single class. Mix classes in those sessions.")
    if fracas:
        linhas.append(
            f"{len(fracas)} de {len(cls)} classes tem menos de {h} sessoes independentes (resultado honesto, sem garantia): "
            f"faltam ate' {pior} sessao(oes) na pior. Se TODAS as classes forem medidas em cada sessao nova, "
            f"{pior} sessao(oes) novas resolvem." if pt else
            f"{len(fracas)} of {len(cls)} classes have fewer than {h} independent sessions (honest result, no guarantee): "
            f"up to {pior} session(s) missing in the worst. If ALL classes are measured in each new session, "
            f"{pior} new session(s) fix it.")
    else:
        linhas.append(f"Toda classe tem {h} ou mais sessoes: resultado honesto possivel." if pt else
                      f"Every class has {h} or more sessions: an honest result is possible.")
    pior_c = max(x["faltam_conformal"] for x in cls)
    if pior_c:
        linhas.append(
            f"Para a garantia conformal em alpha={1.0 / (c + 1):.2f} sao {c} sessoes por classe: faltam ate' {pior_c} na pior." if pt else
            f"For the conformal guarantee at alpha={1.0 / (c + 1):.2f} you need {c} sessions per class: up to {pior_c} missing in the worst.")
    linhas.append("Replicas e diluicoes da mesma amostra nao contam como sessoes." if pt else
                  "Replicates and dilutions of the same sample do not count as sessions.")
    return linhas


def export_markdown(plano: CollectionPlan) -> str:
    """Formato PRIMARIO de saida (P4, Bloco 10) -- simples de gerar e
    versionar, sem dependencia nova."""
    linhas = [
        "# Plano de Coleta -- GUARACI", "",
        f"**Classes:** {', '.join(plano.classes)}  ",
        f"**Amostras por classe:** {plano.n_por_classe}  ",
        f"**Sessoes:** {plano.n_sessoes}  ",
        f"**Total de amostras fisicas:** {len(plano.itens)}",
        "", "## Alertas", "",
    ]
    for a in plano.alertas:
        linhas.append(f"- {a}")
    linhas.append("")
    for sessao in range(plano.n_sessoes):
        itens_sessao = plano.itens_da_sessao(sessao)
        linhas.append(f"## Sessao {sessao + 1} ({len(itens_sessao)} amostras)")
        linhas.append("")
        linhas.append("| Ordem de leitura | Classe | Replica |")
        linhas.append("|---|---|---|")
        for item in itens_sessao:
            linhas.append(f"| {item.ordem_na_sessao + 1} | {item.classe} | "
                           f"{item.replica_idx + 1} |")
        linhas.append("")
    return "\n".join(linhas)


def export_excel(plano: CollectionPlan, caminho: str) -> str:
    """Planilha (P4, Bloco 10) -- reaproveita `openpyxl`, ja' dependencia
    do projeto (`reports.py`), nenhuma dependencia nova. 2 abas: ordem de
    leitura completa (para levar ao laboratorio) e alertas."""
    import openpyxl
    from openpyxl.styles import Font

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Ordem de Leitura"
    ws.append(["Sessao", "Ordem de leitura", "Classe", "Replica"])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for item in sorted(plano.itens, key=lambda i: (i.sessao, i.ordem_na_sessao)):
        ws.append([item.sessao + 1, item.ordem_na_sessao + 1, item.classe,
                   item.replica_idx + 1])

    ws_alertas = wb.create_sheet("Alertas")
    ws_alertas.append(["Alerta"])
    ws_alertas["A1"].font = Font(bold=True)
    for a in plano.alertas:
        ws_alertas.append([a])
    ws_alertas.column_dimensions["A"].width = 100
    ws.column_dimensions["C"].width = 20

    wb.save(caminho)
    return caminho


def export_pdf(plano: CollectionPlan, caminho: str) -> str:
    """PDF opcional (P4, Bloco 10) -- reaproveita `fpdf2`, ja' dependencia
    do projeto (`reports.generate_pdf_report`), nenhuma dependencia nova.
    Mesmo conteudo de `export_markdown`/`export_excel` (tamanho amostral,
    alertas, ordem de leitura por sessao), em layout imprimivel para levar
    ao laboratorio. Usa `FPDF.table()` (fpdf2 >= 2.7, nativo -- sem
    gambiarra de celula-a-celula)."""
    import unicodedata

    from fpdf import FPDF

    def _a(txt: str) -> str:
        """Remove acentos p/ as fontes Latin-1 padrao do fpdf2 -- mesmo
        helper de `reports.generate_pdf_report` (_a), reimplementado aqui
        para nao criar acoplamento entre os dois modulos por uma funcao
        de 1 linha."""
        return unicodedata.normalize("NFKD", str(txt)).encode(
            "ascii", "ignore").decode("ascii")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Plano de Coleta -- GUARACI", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _a(f"Classes: {', '.join(plano.classes)}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Amostras por classe: {plano.n_por_classe}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Sessoes: {plano.n_sessoes}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Total de amostras fisicas: {len(plano.itens)}",
             new_x="LMARGIN", new_y="NEXT")

    if plano.alertas:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Alertas", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for alerta in plano.alertas:
            pdf.multi_cell(0, 5, _a(f"- {alerta}"),
                           new_x="LMARGIN", new_y="NEXT")

    for sessao in range(plano.n_sessoes):
        itens_sessao = plano.itens_da_sessao(sessao)
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"Sessao {sessao + 1} ({len(itens_sessao)} amostras)",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)

        linhas = [["Ordem de leitura", "Classe", "Replica"]]
        for item in itens_sessao:
            linhas.append([str(item.ordem_na_sessao + 1),
                            _a(item.classe), str(item.replica_idx + 1)])
        with pdf.table(text_align="LEFT") as table:
            for linha in linhas:
                row = table.row()
                for valor in linha:
                    row.cell(valor)

    pdf.output(caminho)
    return caminho
