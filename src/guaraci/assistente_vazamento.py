"""assistente_vazamento.py -- ficha anti-vazamento de uma execucao (offline, deterministica).

Inspirada nas "model info sheets" de Kapoor & Narayanan (2023, Patterns 4:100804,
https://doi.org/10.1016/j.patter.2023.100804): para cada fonte de vazamento a ficha diz o que a
execucao REALMENTE fez, a partir da configuracao, do resumo dos dados e do resumo_modelo.txt.
O que o programa nao tem como saber (o holdout foi olhado antes da ultima mudanca? o teste
representa a populacao do artigo?) vira pergunta ao autor, nunca "ok".

Estados: ok | atencao | declarar. A ficha nao alega validacao: so' descreve o procedimento.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from guaraci.assistente_regras import ResumoDados, _num
from guaraci.resumo_parse import N_TESTE_HSI_INDICATIVO

__all__ = ["Item", "itens", "itens_hsi", "ficha", "OK", "ATENCAO", "DECLARAR"]

OK, ATENCAO, DECLARAR = "ok", "atencao", "declarar"
_SIMBOLO = {OK: "[ok]", ATENCAO: "[!]", DECLARAR: "[?]"}
_ROTULO = {OK: ("ok", "ok"), ATENCAO: ("atencao", "attention"), DECLARAR: ("declare", "declare")}


@dataclass
class Item:
    id: str
    titulo: str
    estado: str
    detalhe: str


def _sim(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "sim", "yes", "1"):
        return True
    if s in ("false", "nao", "no", "0"):
        return False
    return None


def _ler_hsi(cfg: Any) -> Dict[str, Any]:
    """Dados da execucao HSI (Relatorios/hsi_relatorio.json; o .txt nas execucoes antigas). Vazio se a pasta nao
    existe ou nao tem o relatorio."""
    pasta = str(getattr(cfg, "output_folder", "") or "") if cfg is not None else ""
    if not pasta:
        return {}
    try:
        from guaraci.config import NOME_RELATORIOS, pasta_execucao
        from guaraci.resumo_parse import ler_hsi_pasta
        return dict(ler_hsi_pasta(pasta_execucao(pasta, NOME_RELATORIOS)))
    except (OSError, ValueError):
        return {}


def _n_avaliados(hsi: Dict[str, Any]) -> Optional[int]:
    """Objetos avaliados fora da amostra: todos na CV por objeto, so' os de teste no holdout."""
    n = hsi.get("n_objetos_avaliados")
    if not isinstance(n, int):
        n = hsi.get("n_teste_interno")
    return n if isinstance(n, int) else None


def itens_hsi(cfg: Any = None, resultado: Optional[Dict[str, Any]] = None, lang: str = "PT") -> List[Item]:
    """Itens da ficha para `mode == "hsi"`: so' o que `hsi_pipeline`/`hsi_validation` fazem de fato.

    O fluxo HSI e' outro: PLS-DA por pixel, decisao por OBJETO fisico (voto majoritario), sem MSC/EMSC, sem o holdout
    de 20% do fluxo tabular. A validacao e' a CV por objeto (padrao, `hsi_validation: cv_objetos`: todos os objetos
    sao previstos por um modelo que nao os viu, VLs reescolhidas em cada treino) ou o holdout unico de objetos
    (`hsi_validation: holdout`); o tipo vem de `hsi_relatorio.json`. O que nao protege contra vazamento aparece
    como limite, nao como "ok"."""
    L = 1 if str(lang).upper() == "EN" else 0
    g = lambda a, p=None: getattr(cfg, a, p) if cfg is not None else p   # noqa: E731
    hsi = {**_ler_hsi(cfg), **{k[4:]: v for k, v in (resultado or {}).items() if str(k).startswith("hsi_")}}
    garantia = str(hsi.get("garantia") or "").lower()
    tipo = str(hsi.get("tipo") or "")
    cv = tipo == "cv_objetos"
    n_av = _n_avaliados(hsi)
    n_tot = hsi.get("n_objetos_total") if isinstance(hsi.get("n_objetos_total"), int) else None
    out: List[Item] = []

    # 1. separacao por objeto fisico
    quem = (("a CV por objeto", "the per-object CV") if cv else ("o holdout", "the holdout"))[L]
    if garantia == "high":
        e, d = OK, (f"{quem} agrupa por OBJETO fisico (subpasta por objeto): todos os pixels de um objeto ficam do mesmo "
                    f"lado, treino ou teste (hsi_validation).",
                    f"{quem} groups by physical OBJECT (one subfolder per object): all pixels of an object stay on the same "
                    f"side, training or test (hsi_validation).")
    elif garantia == "medium":
        e, d = ATENCAO, (f"{quem} agrupa por objeto inteiro, mas o agrupamento vem do CSV de associacao que voce forneceu: "
                         "o programa nao verifica que os grupos sao de fato objetos independentes.",
                         f"{quem} groups by whole object, but the grouping comes from the association CSV you provided: "
                         "the program does not check that the groups are really independent objects.")
    elif garantia == "none":
        e, d = ATENCAO, ("sem fonte de agrupamento: cada gravacao conta como um objeto proprio. Se duas gravacoes forem do "
                         "MESMO objeto fisico, uma pode cair no treino e outra no teste (vazamento).",
                         "no grouping source: each recording counts as its own object. If two recordings are of the SAME "
                         "physical object, one can fall in training and the other in test (leakage).")
    else:
        e, d = DECLARAR, ("garantia de agrupamento desconhecida (sem Relatorios/hsi_relatorio.json): informe se a validacao "
                          "separa objetos fisicos inteiros.",
                          "grouping guarantee unknown (no Reports/hsi_relatorio.json): state whether the validation "
                          "separates whole physical objects.")
    out.append(Item("separacao_por_objeto", ("Separacao por objeto fisico", "Split by physical object")[L], e, d[L]))

    # 2. pixels nao sao amostras independentes
    out.append(Item("pixels", ("Pixels do mesmo objeto", "Pixels of the same object")[L], OK,
                    ("pixels vizinhos sao fortemente correlacionados; por isso nunca sao separados entre treino e teste "
                     "(ate 5000 pixels por gravacao, sorteados com semente fixa) e a decisao e' tomada por OBJETO "
                     "(voto majoritario dos pixels). O n efetivo e' o numero de objetos, nao o de pixels.",
                     "neighbouring pixels are strongly correlated; they are therefore never split between training and test "
                     "(up to 5000 pixels per recording, drawn with a fixed seed) and the decision is made per OBJECT "
                     "(majority vote of its pixels). The effective n is the number of objects, not of pixels.")[L]))

    # 3. calibracao radiometrica e eixo de bandas (pre-etapas do cubo)
    cal = hsi.get("calibracao") if isinstance(hsi.get("calibracao"), dict) else {}
    modo_cal = str(cal.get("modo") or "")
    if modo_cal == "branco_escuro":
        nc = cal.get("n_cubos_calibrados")
        av = [str(a) for a in (cal.get("avisos") or [])]
        e = ATENCAO if av else OK
        d = (f"R = (I - D)/(W - D) com a referencia de branco e de escuro informada, em {nc if nc is not None else 'todos os'} "
             "cubo(s) (tudo-ou-nada: um cubo sem referencia recusa o estudo). A conta usa so' as referencias e o proprio "
             "cubo, nunca rotulos nem outros objetos; uma referencia por sessao tira a diferenca de iluminacao entre sessoes."
             + (" Avisos da calibracao: " + "; ".join(av) + "." if av else ""),
             f"R = (I - D)/(W - D) with the white and dark references provided, in {nc if nc is not None else 'all'} "
             "cube(s) (all-or-nothing: a cube without a reference refuses the study). The computation uses only the "
             "references and the cube itself, never labels or other objects; one reference per session removes the "
             "illumination difference between sessions."
             + (" Calibration warnings: " + "; ".join(av) + "." if av else ""))
    elif modo_cal == "nenhuma":
        e, d = DECLARAR, ("sem calibracao (hsi_calibration: nenhuma): os cubos sao usados como estao e se presume "
                          "reflectancia. Declare se ja estao em reflectancia calibrada; em contagens brutas, uma diferenca de "
                          "iluminacao/sessao entre classes pode ser aprendida como se fosse a classe.",
                          "no calibration (hsi_calibration: nenhuma): cubes are used as they are and reflectance is assumed. "
                          "State whether they are already calibrated reflectance; in raw counts, an illumination/session "
                          "difference between classes can be learned as if it were the class.")
    else:
        e, d = DECLARAR, ("calibracao desconhecida (sem hsi_relatorio.json): declare se os cubos estao em reflectancia "
                          "calibrada.",
                          "calibration unknown (no hsi_relatorio.json): state whether the cubes are in calibrated reflectance.")
    out.append(Item("calibracao_hsi", ("Calibracao radiometrica (branco/escuro)", "Radiometric calibration (white/dark)")[L],
                    e, d[L]))
    eixo = hsi.get("eixo_bandas") if isinstance(hsi.get("eixo_bandas"), dict) else {}
    if eixo:
        nb, fx = eixo.get("n_bandas"), eixo.get("faixa_nm")
        faixa = (f", {fx[0]:g}-{fx[1]:g} nm" if isinstance(fx, (list, tuple)) and len(fx) == 2 else "")
        tol = eixo.get("tolerancia_nm", 1.0)
        nr = eixo.get("n_reamostrados", 0)
        if str(eixo.get("modo")) == "reamostrar" and nr:
            d = (f"{nb} bandas{faixa}; {nr} cubo(s) reamostrado(s) por interpolacao linear para o eixo comum (desvio maximo "
                 f"{eixo.get('desvio_max_nm', 0):g} nm). A interpolacao suaviza um pouco o espectro desses cubos.",
                 f"{nb} bands{faixa}; {nr} cube(s) resampled by linear interpolation to the common axis (maximum shift "
                 f"{eixo.get('desvio_max_nm', 0):g} nm). Interpolation smooths the spectra of those cubes slightly.")
        else:
            d = (f"{nb} bandas{faixa}; todos os cubos tem o mesmo eixo de comprimentos de onda (tolerancia {tol:g} nm); "
                 "eixos diferentes sao recusados em vez de misturados.",
                 f"{nb} bands{faixa}; all cubes share the same wavelength axis (tolerance {tol:g} nm); different axes are "
                 "refused instead of mixed.")
        out.append(Item("eixo_bandas_hsi", ("Eixo de bandas", "Band axis")[L], OK, d[L]))

    # 4. pre-processamento
    preset = str(g("default_preprocessing", "") or "").lower()
    if "snv" in preset:
        d = ("SNV por pixel: cada espectro e' normalizado so' por si mesmo, entao nenhum parametro vem de outros pixels ou "
             "objetos. A centragem na media do PLS e' ajustada so' nos pixels de treino. A faixa espectral vem da "
             "configuracao, nao dos dados. MSC e EMSC nao se aplicam ao modo HSI.",
             "per-pixel SNV: each spectrum is normalised by itself only, so no parameter comes from other pixels or "
             "objects. The PLS mean-centring is fitted on the training pixels only. The spectral range comes from the "
             "configuration, not from the data. MSC and EMSC do not apply to the HSI mode.")
    else:
        _nao = "" if preset == "mc" else f" (preset '{preset}' nao se aplica ao modo HSI)"
        _not = "" if preset == "mc" else f" (preset '{preset}' does not apply to the HSI mode)"
        d = (f"sem SNV{_nao}: so' a centragem na media do PLS, ajustada nos pixels de treino. MSC e EMSC nao se aplicam "
             "ao modo HSI.",
             f"no SNV{_not}: only the PLS mean-centring, fitted on the training pixels. MSC and EMSC do not apply to the "
             "HSI mode.")
    out.append(Item("preproc_hsi", ("Pre-processamento sem parametros aprendidos entre objetos",
                                    "Pre-processing with no parameters learned across objects")[L], OK, d[L]))
    out.append(Item("segmentacao", ("Segmentacao objeto/fundo", "Object/background segmentation")[L], OK,
                    ("PCA + limiar de Otsu calculados em cada cubo separadamente, sem rotulos e sem usar outros cubos.",
                     "PCA + Otsu threshold computed on each cube separately, without labels and without using other cubes.")[L]))

    # 5. escolha de VLs
    if cv:
        out.append(Item("selecao_lvs_hsi", ("Escolha de variaveis latentes dentro de cada treino",
                                            "Latent variable choice inside each training set")[L], OK,
                        ("em cada dobra as variaveis latentes sao reescolhidas so' com os objetos de treino (CV por objeto, "
                         "regra de parcimonia de 2%): o objeto previsto nao participa da escolha. Nao e' uma validacao "
                         "aninhada completa com repeticoes, mas a metrica reportada vem de previsoes de modelos que nao "
                         "viram o objeto. O modelo final usa VLs escolhidas com todos os objetos.",
                         "in each fold the latent variables are re-chosen with the training objects only (per-object CV, "
                         "2% parsimony rule): the predicted object takes no part in the choice. It is not a full nested "
                         "validation with repetitions, but the reported metric comes from predictions of models that did "
                         "not see the object. The final model uses LVs chosen with all objects.")[L]))
    else:
        out.append(Item("selecao_lvs_hsi", ("Escolha de variaveis latentes (sem validacao aninhada)",
                                            "Latent variable choice (no nested validation)")[L], ATENCAO,
                        ("nao ha validacao aninhada. Para o holdout, as variaveis latentes sao escolhidas so' com os objetos "
                         "de treino (CV por objeto, regra de parcimonia de 2%), entao os objetos de teste nao participam da "
                         "escolha. O modelo final usa VLs escolhidas com todos os objetos.",
                         "there is no nested validation. For the holdout, the latent variables are chosen with the training "
                         "objects only (per-object CV, 2% parsimony rule), so the test objects take no part in the choice. "
                         "The final model uses LVs chosen with all objects.")[L]))

    # 6. validacao: CV por objeto (padrao) ou holdout unico; sem validacao externa
    peq = n_av is not None and n_av < N_TESTE_HSI_INDICATIVO
    indic_pt = (f"Com menos de {N_TESTE_HSI_INDICATIVO} objetos avaliados, sensibilidade e especificidade sao INDICATIVAS: "
                f"um objeto a mais ou a menos move a metrica em 1/{max(n_av or 1, 1)}. " if peq else "")
    indic_en = (f"With fewer than {N_TESTE_HSI_INDICATIVO} evaluated objects, sensitivity and specificity are INDICATIVE only: "
                f"one object more or less moves the metric by 1/{max(n_av or 1, 1)}. " if peq else "")
    if cv:
        nd = hsi.get("n_dobras")
        forma_pt = ("leave-one-object-out" if isinstance(nd, int) and isinstance(n_tot, int) and nd >= n_tot
                    else f"{nd} dobras" if nd else "k dobras")
        forma_en = ("leave-one-object-out" if isinstance(nd, int) and isinstance(n_tot, int) and nd >= n_tot
                    else f"{nd} folds" if nd else "k folds")
        e = ATENCAO if (peq or hsi.get("avisos")) else OK
        d = (f"CV por objeto ({forma_pt}, estratificada por classe): {n_av} de {n_tot} objetos previstos por um modelo "
             f"ajustado sem eles. Sensibilidade/especificidade com IC de Clopper-Pearson a 95%; o IC trata cada objeto como "
             f"tentativa independente e, como as dobras compartilham treino, e' aproximado (um pouco otimista). "
             + indic_pt + "Nao ha validacao externa (outro dia, lote ou equipamento)."
             if n_av is not None and n_tot is not None else
             "CV por objeto: cada objeto e' previsto por um modelo ajustado sem ele, com IC de Clopper-Pearson. Nao ha "
             "validacao externa (outro dia, lote ou equipamento).",
             f"per-object CV ({forma_en}, stratified by class): {n_av} of {n_tot} objects predicted by a model fitted "
             f"without them. Sensitivity/specificity with 95% Clopper-Pearson CI; the CI treats each object as an "
             f"independent trial and, since the folds share training objects, is approximate (slightly optimistic). "
             + indic_en + "There is no external validation (another day, batch or instrument)."
             if n_av is not None and n_tot is not None else
             "per-object CV: each object is predicted by a model fitted without it, with Clopper-Pearson CI. There is no "
             "external validation (another day, batch or instrument).")
        out.append(Item("cv_objetos_hsi", ("Validacao cruzada por objeto (padrao)", "Per-object cross-validation (default)")[L],
                        e, d[L]))
    elif n_av is not None:
        d = (f"um unico holdout de {n_av} objeto(s) de teste, sorteado sem estratificar por classe (uma classe pode ficar "
             f"sem objeto no teste: sensibilidade n/d) e sem repeticao. " + indic_pt
             + "Nao ha validacao externa (outro dia, lote ou equipamento). hsi_validation: cv_objetos preve todos os objetos.",
             f"a single holdout of {n_av} test object(s), drawn without stratifying by class (a class can end up with no "
             f"object in the test: sensitivity n/a) and without repetition. " + indic_en
             + "There is no external validation (another day, batch or instrument). hsi_validation: cv_objetos predicts "
             "every object.")
        out.append(Item("holdout_hsi", ("Holdout de objetos (interno, unico)", "Object holdout (internal, single)")[L],
                        ATENCAO, d[L]))
    else:
        out.append(Item("holdout_hsi", ("Validacao de objetos (interna)", "Object validation (internal)")[L], DECLARAR,
                        ("numero de objetos avaliados desconhecido (sem hsi_relatorio.json): com poucos objetos a metrica e' "
                         "so' indicativa. Nao ha validacao externa.",
                         "number of evaluated objects unknown (no hsi_relatorio.json): with few objects the metric is only "
                         "indicative. There is no external validation.")[L]))

    # 7. confianca por objeto e mapa
    fora = hsi.get("fora_da_amostra")
    if fora is True:
        out.append(Item("confianca_in_sample", ("Confianca por objeto", "Per-object confidence")[L], OK,
                        ("a confianca/heterogeneidade por objeto vem da previsao fora da amostra (modelo que nao viu o "
                         "objeto). O mapa espacial de uma gravacao tambem e' predito por um modelo ajustado sem aquele objeto.",
                         "the per-object confidence/heterogeneity comes from the out-of-sample prediction (a model that did "
                         "not see the object). The spatial map of a recording is also predicted by a model fitted without "
                         "that object.")[L]))
    else:
        out.append(Item("confianca_in_sample", ("Confianca por objeto", "Per-object confidence")[L], ATENCAO,
                        ("a heterogeneidade/confianca por objeto vem do modelo final aplicado aos mesmos pixels com que foi "
                         "ajustado (dentro da amostra): nao e' desempenho fora da amostra. So' o mapa espacial de uma "
                         "gravacao e' predito por um modelo ajustado sem aquele objeto.",
                         "the per-object heterogeneity/confidence comes from the final model applied to the same pixels it "
                         "was fitted on (in-sample): it is not out-of-sample performance. Only the spatial map of one "
                         "recording is predicted by a model fitted without that object.")[L]))

    # 8-10. o que o programa nao sabe
    resultado_pt, resultado_en = (("da CV", "of the CV") if cv else ("do holdout", "of the holdout"))
    out.append(Item("holdout_intocado", ("Resultado da validacao sem uso nas escolhas", "Validation result unused in choices")[L],
                    DECLARAR,
                    (f"declare se alguma configuracao (faixa, pre-processamento, calibracao, objetos retirados) foi mudada "
                     f"depois de ver o resultado {resultado_pt}; se foi, ele virou parte do ajuste.",
                     f"state whether any setting (range, pre-processing, calibration, removed objects) was changed after "
                     f"seeing the result {resultado_en}; if so, it became part of the fit.")[L]))
    out.append(Item("sessao_hsi", ("Classe x sessao/dia/iluminacao", "Class x session/day/lighting")[L], DECLARAR,
                    ("o fluxo HSI nao recebe a sessao nem o dia de cada cubo e nao checa esse confundimento. Declare se cada "
                     "classe foi imageada em varias sessoes e com a mesma iluminacao.",
                     "the HSI flow is not given the session or day of each cube and does not check this confounding. State "
                     "whether each class was imaged in several sessions and under the same lighting.")[L]))
    out.append(Item("populacao", ("Teste representa a populacao do estudo", "Test represents the study population")[L], DECLARAR,
                    ("declare de onde vem o teste (lotes, safras, camera) e se e' a mesma populacao sobre a qual o artigo "
                     "conclui.",
                     "state where the test comes from (batches, harvests, camera) and whether it is the population the paper "
                     "draws conclusions about.")[L]))
    return out


def itens(cfg: Any = None, dados: Optional[ResumoDados] = None, resultado: Optional[Dict[str, Any]] = None,
          lang: str = "PT") -> List[Item]:
    L = 1 if str(lang).upper() == "EN" else 0
    res = resultado or {}
    g = lambda a, p=None: getattr(cfg, a, p) if cfg is not None else p   # noqa: E731
    out: List[Item] = []
    if str(g("mode", "") or "").lower() == "hsi":
        return itens_hsi(cfg, resultado, lang)

    # 1. unidade independente separada entre treino e teste
    agrupa = _sim(res.get("Group-aware (mae_id)"))
    if agrupa is None:
        agrupa = bool(g("group_by_mae_id", True))
    garantia = str(res.get("Grouping guarantee", g("grouping_guarantee", "")) or "").lower()
    if not agrupa:
        e, d = ATENCAO, ("validacao por grupo desligada: leituras da mesma amostra podem cair em treino e teste.",
                         "group validation off: readings of the same sample may fall in training and test.")
    elif garantia in ("medium", "none", "media", "nenhuma"):
        e, d = ATENCAO, (f"validacao por grupo ligada, mas a garantia de agrupamento e' '{garantia}'.",
                         f"group validation on, but the grouping guarantee is '{garantia}'.")
    else:
        e, d = OK, ("treino e teste separados por unidade independente.", "training and test split by independent unit.")
    out.append(Item("separacao_por_unidade", ("Separacao por unidade independente", "Split by independent unit")[L], e, d[L]))

    # 2. replicas
    if dados is not None:
        k = dados.leituras_por_unidade_max
        if k <= 1:
            e, d = OK, ("uma leitura por unidade.", "one reading per unit.")
        elif agrupa:
            e, d = OK, (f"ate {k} leituras por unidade, mantidas juntas pela validacao por grupo.",
                        f"up to {k} readings per unit, kept together by group validation.")
        else:
            e, d = ATENCAO, (f"ate {k} leituras por unidade sem validacao por grupo.",
                             f"up to {k} readings per unit without group validation.")
    else:
        e, d = DECLARAR, ("resumo dos dados indisponivel: informe quantas leituras ha por amostra fisica.",
                          "data summary unavailable: state how many readings there are per physical sample.")
    out.append(Item("replicas", ("Replicas", "Replicates")[L], e, d[L]))

    # 3. pre-processamento dentro das dobras
    out.append(Item("preproc_nas_dobras", ("Pre-processamento ajustado so' no treino", "Pre-processing fitted on training only")[L], OK,
                    ("o pre-processamento roda dentro do Pipeline do scikit-learn em cada dobra; MSC e EMSC usam a "
                     "referencia do treino (guaraci.preprocessamento).",
                     "pre-processing runs inside the scikit-learn Pipeline in each fold; MSC and EMSC use the "
                     "training reference (guaraci.preprocessamento).")[L]))

    # 4. escolha do numero de variaveis latentes
    if bool(g("selecao_lv_cv_aninhada", True)):
        e, d = OK, ("o numero de variaveis latentes e' escolhido dentro de cada dobra externa (validacao aninhada): "
                    "a dobra que mede o erro nao participa da escolha.",
                    "the number of latent variables is chosen inside each outer fold (nested validation): "
                    "the fold that measures the error takes no part in the choice.")
    else:
        e, d = ATENCAO, ("numero de variaveis latentes escolhido na mesma validacao que mede o erro: a metrica sai otimista.",
                         "number of latent variables chosen on the same validation that measures the error: the metric is optimistic.")
    out.append(Item("selecao_lvs", ("Escolha de variaveis latentes", "Latent variable choice")[L], e, d[L]))

    # 5. holdout externo
    n_ho = _num(res.get("Holdout n"))
    frac = _num(g("frac_holdout", 0.0)) or 0.0
    if n_ho:
        e, d = OK, (f"holdout externo com {int(n_ho)} leituras.", f"external holdout with {int(n_ho)} readings.")
    elif frac > 0:
        e, d = OK, (f"holdout externo configurado ({frac:.0%} das unidades).", f"external holdout configured ({frac:.0%} of units).")
    else:
        e, d = ATENCAO, ("sem holdout externo: o desempenho vem so' da validacao cruzada.",
                         "no external holdout: performance comes from cross-validation only.")
    out.append(Item("holdout_externo", ("Holdout externo", "External holdout")[L], e, d[L]))

    # 6. holdout tocado (o programa nao sabe)
    out.append(Item("holdout_intocado", ("Holdout sem uso nas escolhas", "Holdout unused in choices")[L], DECLARAR,
                    ("declare se alguma configuracao foi mudada depois de ver o resultado do holdout; se foi, "
                     "ele virou parte do ajuste.",
                     "state whether any setting was changed after seeing the holdout result; if so, it became "
                     "part of the fit.")[L]))

    # 7. confundimento classe x sessao (vem da auditoria)
    criticos = [x.lower() for x in res.get("_auditoria_critica", []) or []]
    if any("confund" in x for x in criticos):
        e, d = ATENCAO, ("a auditoria encontrou classe confundida com sessao: o modelo pode separar dias, nao amostras.",
                         "the audit found class confounded with session: the model may separate days, not samples.")
    elif resultado is not None:
        e, d = OK, ("a auditoria nao marcou confundimento critico entre classe e sessao.",
                    "the audit flagged no critical class-session confounding.")
    else:
        e, d = DECLARAR, ("sem resultado: informe se cada classe passou pelo equipamento em varias sessoes.",
                          "no result: state whether each class was measured in several sessions.")
    out.append(Item("confundimento", ("Classe x sessao", "Class x session")[L], e, d[L]))

    # 8. populacao de interesse
    out.append(Item("populacao", ("Teste representa a populacao do estudo", "Test represents the study population")[L], DECLARAR,
                    ("declare de onde vem o teste (lotes, safras, equipamento) e se e' a mesma populacao sobre a "
                     "qual o artigo conclui.",
                     "state where the test comes from (batches, harvests, instrument) and whether it is the "
                     "population the paper draws conclusions about.")[L]))
    return out


def ficha(cfg: Any = None, dados: Optional[ResumoDados] = None, resultado: Optional[Dict[str, Any]] = None,
          lang: str = "PT") -> str:
    """Ficha em Markdown, pronta para gravar ao lado do resumo_modelo.txt."""
    L = 1 if str(lang).upper() == "EN" else 0
    its = itens(cfg, dados, resultado, lang)
    tit = ("# Ficha anti-vazamento", "# Leakage sheet")[L]
    intro = ("Descreve o que esta execucao fez em cada fonte de vazamento conhecida (Kapoor & Narayanan 2023, "
             "https://doi.org/10.1016/j.patter.2023.100804). `[?]` e' pergunta ao autor: o programa nao tem como saber.",
             "Describes what this run did for each known leakage source (Kapoor & Narayanan 2023, "
             "https://doi.org/10.1016/j.patter.2023.100804). `[?]` is a question for the author: the program cannot know.")[L]
    linhas = [tit, "", intro, ""]
    for i in its:
        linhas.append(f"- {_SIMBOLO[i.estado]} **{i.titulo}** ({_ROTULO[i.estado][L]}): {i.detalhe}")
    n_at = sum(i.estado == ATENCAO for i in its)
    n_dec = sum(i.estado == DECLARAR for i in its)
    linhas += ["", (f"Resumo: {n_at} item(ns) de atencao, {n_dec} a declarar.",
                    f"Summary: {n_at} attention item(s), {n_dec} to declare.")[L]]
    return "\n".join(linhas) + "\n"
