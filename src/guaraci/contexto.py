"""contexto.py -- ajuste automatico por contexto (regras declarativas, offline).

O contexto (modo de entrada, nivel, objetivo, colunas declaradas) decide quatro
coisas, e SO' elas:

  ocultar   opcoes que nao se aplicam ao modo de entrada atual (so' visual)
  ajustar   AJUSTE AUTOMATICO, so' por compatibilidade (lista branca `AJUSTES`):
            algo que o motor ignoraria ou recusaria de qualquer forma
  sugerir   escolha que MUDA o resultado cientifico: so' sugestao, nunca aplicada
  recusar   combinacao sem remedio seguro: erro com causa e acao

Nunca e' automatico: alpha, esquema de validacao, unidade de analise, politica de
censura, selecao de variaveis, metodo final. Todo ajuste mostra `Ajustado: X ->
Y. Motivo: Z. Desfazer: ...`, fica na lista `cfg.ajustes_automaticos` e vai para
a configuracao efetiva gravada com os resultados. `ajuste_automatico: false`
(ou `--sem-ajuste-automatico`) desliga os ajustes.

Importa so' stdlib (e `modos_analise`, que e' leve): carrega em milissegundos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["Contexto", "Ajuste", "Resultado", "AJUSTES", "SUGESTOES", "RECUSAS", "contexto_de",
           "aplicar", "campos_visiveis", "desfazer", "texto_efetiva", "gravar_efetiva", "MODOS_TABULARES", "PREPROC_TABULAR"]

#: Modos de entrada que leem uma tabela de elementos (nao espectro).
MODOS_TABULARES = ("multielementar",)
PREPROC_TABULAR = ("autoscaling", "mc")

# Opcoes que so' valem em certos modos de entrada (ocultas nos demais). Chave de opcao -> modos.
APLICA_A: Dict[str, Tuple[str, ...]] = {
    "imagem_incluir_textura": ("imagem",), "perfil_tecnica": ("imagem",),
    "hsi_pasta_dataset": ("hsi",), "hplc_detector": ("hplc",),
    "cromatografia_canal": ("hplc",), "cromatograma_max_pontos": ("hplc", "gcms"),
    "espectro_max_pontos": ("dx", "opus", "spc", "sp", "rmn"),
    "eem_espalhamento": ("eem",), "eem_espalhamento_largura_nm": ("eem",), "eem_raman": ("eem",), "parafac_nao_negativo": ("eem",),
    "hsi_calibracao": ("hsi",), "hsi_referencia_branco": ("hsi",), "hsi_referencia_escuro": ("hsi",),
    "hsi_eixo_bandas": ("hsi",), "hsi_validacao": ("hsi",), "hsi_cv_dobras": ("hsi",),
    "imagem_referencia": ("imagem",), "imagem_segmentacao": ("imagem",),
    "icp_unidade": ("multielementar",), "icp_coluna_amostra": ("multielementar",),
    "icp_coluna_tipo": ("multielementar",), "icp_coluna_sessao": ("multielementar",),
    "icp_padrao_interno": ("multielementar",), "icp_politica_censura": ("multielementar",),
    "icp_limiar_censura": ("multielementar",), "icp_log": ("multielementar",),
    "arquivo_csv": ("csv", "multielementar"), "coluna_concentracao": ("csv",),
    "coluna_classe": ("csv", "multielementar"), "coluna_grupo": ("csv", "multielementar"),
    "arquivo_grupos": ("dx", "opus", "spc", "sp", "rmn", "hplc", "gcms", "eem"),
    "faixa_min_cm": ("dx", "csv", "opus", "spc", "sp", "rmn", "hplc", "gcms", "eem", "sintetico"),
    "faixa_max_cm": ("dx", "csv", "opus", "spc", "sp", "rmn", "hplc", "gcms", "eem", "sintetico"),
    "saturacao": ("dx", "csv", "opus", "spc", "sp"),
    "faixas_excluir": ("dx", "csv", "opus", "spc", "sp", "rmn", "hplc", "gcms", "sintetico"),
    "normalizacao": ("dx", "csv", "opus", "spc", "sp", "rmn", "hplc", "gcms", "sintetico"),
    "ims_rip_janela": ("dx", "csv", "opus", "spc", "sp", "rmn", "hplc", "gcms", "sintetico"),
}

_TOGGLES_SO_CLASSIFICACAO = ("opls_da", "selecao_variaveis_etapa4", "comparar_pre_processamentos",
                             "teste_wold", "teste_cv_anova", "teste_martens",
                             "benchmark", "monte_carlo", "shap_benchmark")
_TOGGLES_SO_QUANTIFICACAO = ("benchmark_regressao",)


@dataclass
class Contexto:
    modo_entrada: str
    nivel: str
    objetivo: str                      # ja resolvido (exploratorio/classificacao/quantificacao)
    pre_processamento: str
    tem_grupo: bool
    tem_concentracao: bool
    espectral: bool
    faixa: Tuple[float, float] = (0.0, 0.0)   # (faixa_min_cm, faixa_max_cm) atuais


@dataclass
class Ajuste:
    regra: str
    campo: str                         # chave de configuracao (YAML)
    de: Any
    para: Any
    motivo: str

    def linha(self) -> str:
        return (f"Ajustado: {self.campo} {_fmt(self.de)} -> {_fmt(self.para)}. Motivo: {self.motivo}. "
                f"Desfazer: defina `{self.campo}: {_fmt(self.de)}` e rode com --sem-ajuste-automatico.")


@dataclass
class Resultado:
    ajustes: List[Ajuste] = field(default_factory=list)
    sugestoes: List[str] = field(default_factory=list)
    erros: List[str] = field(default_factory=list)

    def linhas(self) -> List[str]:
        return ([a.linha() for a in self.ajustes] + [f"Sugestao: {s}" for s in self.sugestoes])


def _fmt(v: Any) -> str:
    return str(v).lower() if isinstance(v, bool) else str(v)


def _attr(chave: str) -> str:
    from guaraci.config_io import _CONFIG_SPEC
    for s in _CONFIG_SPEC:
        if s["key"] == chave:
            return s["attr"]
    raise KeyError(chave)


def contexto_de(cfg: Any) -> Contexto:
    from guaraci import modos_analise as ma
    modo = str(getattr(cfg, "mode", "") or "")
    pre = str(getattr(cfg, "default_preprocessing", "") or "")
    return Contexto(
        modo_entrada=modo, nivel=str(getattr(cfg, "level", "") or ""), objetivo=ma.resolve_objective(cfg),
        pre_processamento=pre,
        tem_grupo=bool(getattr(cfg, "group_column", None) or getattr(cfg, "group_file", None)),
        tem_concentracao=bool(getattr(cfg, "conc_column", None)),
        espectral=modo not in MODOS_TABULARES and modo not in ("imagem", "hsi"),
        faixa=(float(getattr(cfg, "wn_min", 0.0)), float(getattr(cfg, "wn_max", 0.0))))


#: Faixa padrao da Config (FT-NIR, cm-1). Nos modos cujo eixo nao e' numero de onda ela corta tudo.
FAIXA_NIR_PADRAO = (4000.0, 10000.0)
FAIXA_ABERTA = (-5000.0, 1.0e9)
#: Modos cujo eixo e' indice, tempo de retencao, ppm ou grade de fluorescencia (nao cm-1).
MODOS_EIXO_NAO_CM = ("imagem", "gcms", "hplc", "eem", "rmn")
_MOTIVO_FAIXA = ("o eixo deste modo nao e' numero de onda: a faixa padrao de NIR (4000 a 10000 cm-1) cortaria todas as variaveis",
                 "this mode's axis is not wavenumber: the default NIR range (4000 to 10000 cm-1) would cut every variable")


# --------------------------------------------------------------------------
# Regras. Cada ajuste: (id, quando(ctx), campo, valor-alvo(ctx), motivo). So' o que esta
# aqui pode ser alterado sem o usuario pedir. Teste: tests/test_contexto.py.
# --------------------------------------------------------------------------
AJUSTES: Tuple[Tuple[str, Callable[[Contexto], bool], str, Any, Tuple[str, str]], ...] = (
    ("tabular_preproc", lambda c: c.modo_entrada in MODOS_TABULARES + ("imagem",) and c.pre_processamento not in PREPROC_TABULAR,
     "pre_processamento", "autoscaling",
     ("uma tabela de elementos ou de medidas de imagem nao e espectro: MSC, SNV e Savitzky-Golay nao se aplicam",
      "an element table or image measures are not a spectrum: MSC, SNV and Savitzky-Golay do not apply")),
    ("n1_sem_ddsimca", lambda c: c.nivel == "N1", "ddsimca", False,
     ("o nivel N1 (identificar a classe) nao usa o DD-SIMCA", "level N1 (identify the class) does not use DD-SIMCA")),
    # so' quando a faixa ainda e' a padrao (o usuario nao escolheu); a 2a regra ve o efeito da 1a
    ("faixa_eixo_nao_cm_min", lambda c: c.modo_entrada in MODOS_EIXO_NAO_CM and c.faixa == FAIXA_NIR_PADRAO,
     "faixa_min_cm", FAIXA_ABERTA[0], _MOTIVO_FAIXA),
    ("faixa_eixo_nao_cm_max", lambda c: c.modo_entrada in MODOS_EIXO_NAO_CM and c.faixa == (FAIXA_ABERTA[0], FAIXA_NIR_PADRAO[1]),
     "faixa_max_cm", FAIXA_ABERTA[1], _MOTIVO_FAIXA),
    ("n2_com_ddsimca", lambda c: c.nivel == "N2", "ddsimca", True,
     ("o nivel N2 (puro x adulterado) roda o DD-SIMCA", "level N2 (pure vs adulterated) runs DD-SIMCA")),
) + tuple(
    (f"objetivo_{k}", (lambda c: c.objetivo != "classificacao"), k, False,
     ("este objetivo nao e classificacao: o motor ignoraria esta etapa",
      "this objective is not classification: the engine would skip this step"))
    for k in _TOGGLES_SO_CLASSIFICACAO
) + tuple(
    (f"objetivo_{k}", (lambda c: c.objetivo != "quantificacao"), k, False,
     ("este objetivo nao e quantificacao: o motor ignoraria esta etapa",
      "this objective is not quantification: the engine would skip this step"))
    for k in _TOGGLES_SO_QUANTIFICACAO
)

#: Sugestoes (nunca aplicadas): (id, quando, texto)
SUGESTOES: Tuple[Tuple[str, Callable[[Contexto, Any], bool], Tuple[str, str]], ...] = (
    ("grupo_sem_validacao_agrupada",
     lambda c, cfg: c.tem_grupo and not bool(getattr(cfg, "group_by_mae_id", True)),
     ("ha coluna de grupo declarada mas a validacao por grupo esta desligada: com replicas, "
      "a metrica sai otimista. Para aceitar: `validacao_group_aware: true`.",
      "a group column is declared but group-aware validation is off: with replicates the metric "
      "comes out optimistic. To accept: `validacao_group_aware: true`.")),
    ("censura_padrao_visivel",
     lambda c, cfg: c.modo_entrada == "multielementar",
     ("politica de censura em uso: `icp_politica_censura` (a analise e refeita com as duas "
      "politicas e o relatorio diz se a conclusao muda).",
      "censoring policy in use: `icp_politica_censura` (the analysis is redone with both policies "
      "and the report says whether the conclusion changes).")),
)

#: Recusas: (id, quando, mensagem com causa e acao)
RECUSAS: Tuple[Tuple[str, Callable[[Contexto, Any], bool], Tuple[str, str]], ...] = (
    ("n3_sem_concentracao",
     lambda c, cfg: c.nivel == "N3" and c.modo_entrada == "csv" and not c.tem_concentracao,
     ("o nivel N3 (quantificar) precisa de `coluna_concentracao` no modo csv. Informe a coluna com o "
      "teor, ou troque o nivel para N1 (classificar) ou N2 (puro x adulterado).",
      "level N3 (quantify) needs `coluna_concentracao` in csv mode. Give the content column, or "
      "change the level to N1 (classify) or N2 (pure vs adulterated).")),
)


def aplicar(cfg: Any, *, auto: Optional[bool] = None, lang: str = "PT") -> Resultado:
    """Avalia as regras no contexto de `cfg`. Com ajuste automatico ligado,
    aplica a lista branca em `cfg` (e guarda em `cfg.ajustes_automaticos`).
    `lang` escolhe o idioma das mensagens (o log do motor usa PT)."""
    L = 1 if str(lang).upper() == "EN" else 0
    if auto is None:
        auto = bool(getattr(cfg, "auto_adjust", True))
    ctx = contexto_de(cfg)
    r = Resultado()
    for rid, quando, msg in RECUSAS:
        if quando(ctx, cfg):
            r.erros.append(msg[L])
    if auto:
        for rid, quando_aj, campo, alvo, motivo in AJUSTES:
            if not quando_aj(ctx):
                continue
            attr = _attr(campo)
            atual = getattr(cfg, attr, None)
            if _igual(atual, alvo):
                continue
            setattr(cfg, attr, alvo)
            r.ajustes.append(Ajuste(rid, campo, _legivel(atual), _legivel(alvo), motivo[L]))
            ctx = contexto_de(cfg)             # regras seguintes enxergam o ajuste ja feito
    for sid, quando, texto in SUGESTOES:
        if quando(ctx, cfg):
            r.sugestoes.append(texto[L])
    cfg.ajustes_automaticos = [dict(regra=a.regra, campo=a.campo, de=a.de, para=a.para, motivo=a.motivo)
                               for a in r.ajustes]
    cfg.sugestoes_contexto = list(r.sugestoes)
    return r


def desfazer(cfg: Any, *, manter_registro: bool = False) -> int:
    """Reverte os ajustes automaticos ja registrados em `cfg`. Devolve quantos.
    `manter_registro=True` volta os valores mas deixa a lista (para relatorio)."""
    n = 0
    for a in reversed(list(getattr(cfg, "ajustes_automaticos", []) or [])):
        setattr(cfg, _attr(a["campo"]), a["de"])
        n += 1
    if not manter_registro:
        cfg.ajustes_automaticos = []
    return n


# Campos cujo valor e' caminho do usuario: a configuracao efetiva grava so' o nome final.
_CAMPOS_CAMINHO = ("pasta_dados", "pasta_saida", "arquivo_csv", "arquivo_grupos", "hsi_pasta_dataset",
                   "hsi_referencia_branco", "hsi_referencia_escuro")


def texto_efetiva(cfg: Any) -> str:
    """config_efetiva.yaml: o que a execucao REALMENTE usou (depois do ajuste
    automatico), sem caminhos completos da maquina."""
    import os
    from guaraci.config_io import _CONFIG_SPEC, _attr_para_yaml, _fmt_yaml
    linhas = ["# Configuracao EFETIVA desta execucao (depois do ajuste automatico).",
              "# Caminhos de arquivo aparecem so' pelo nome final.", ""]
    for a in getattr(cfg, "ajustes_automaticos", []) or []:
        linhas.append(f"# Ajustado: {a['campo']} {_fmt(a['de'])} -> {_fmt(a['para'])}. Motivo: {a['motivo']}.")
    for s in _CONFIG_SPEC:
        v = _attr_para_yaml(s, cfg)
        if s["key"] in _CAMPOS_CAMINHO and isinstance(v, str) and v:
            v = os.path.basename(v.rstrip("/\\")) or "(definido)"
        linhas.append(f"{s['key']}: {_fmt_yaml(v)}")
    return "\n".join(linhas) + "\n"


def gravar_efetiva(cfg: Any, pasta: str, pasta_relatorios: str) -> None:
    """Grava config_efetiva.yaml na pasta do resultado e os ajustes/sugestoes em JSON."""
    import json
    import os
    with open(os.path.join(pasta, "config_efetiva.yaml"), "w", encoding="utf-8") as fh:
        fh.write(texto_efetiva(cfg))
    with open(os.path.join(pasta_relatorios, "ajustes_automaticos.json"), "w", encoding="utf-8") as fh:
        json.dump({"ajuste_automatico": bool(getattr(cfg, "auto_adjust", True)),
                   "ajustes": list(getattr(cfg, "ajustes_automaticos", []) or []),
                   "ajustes_hardware": list(getattr(cfg, "ajustes_hardware", []) or []),
                   "sugestoes": list(getattr(cfg, "sugestoes_contexto", []) or [])},
                  fh, ensure_ascii=False, indent=2, default=str)


def _igual(a: Any, b: Any) -> bool:
    if isinstance(b, bool):
        return bool(a) == b
    return str(a).strip().lower() == str(b).strip().lower()


def _legivel(v: Any) -> Any:
    return v


def campos_visiveis(cfg: Any, campos: Sequence[str]) -> List[str]:
    """Remove as opcoes que nao se aplicam ao modo de entrada de `cfg`."""
    modo = str(getattr(cfg, "mode", "") or "")
    return [k for k in campos if k not in APLICA_A or modo in APLICA_A[k]]
