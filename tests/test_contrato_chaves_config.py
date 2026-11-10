"""Contrato das chaves de configuracao, nos dois sentidos.

1. Todo campo da Config tem chave no config.yaml (direta ou por uma propriedade sobre ele) ou
   esta em CAMPOS_INTERNOS com motivo -- nunca os dois. Campo sem rota e' inacessivel.
2. Toda chave do config.yaml esta no catalogo (texto, tela, nivel) e toda opcao do catalogo
   esta no config.yaml ou em CHAVES_FORA_DO_YAML com motivo.
3. Toda chave do config.yaml e' usada de verdade pelo motor (busca no codigo fora das telas,
   do catalogo e do proprio config). Chave sem uso e' codigo morto.
4. As listas de excecao nao crescem sem justificativa (teto explicito).
"""
import dataclasses
import re
from pathlib import Path

from guaraci import catalogo as cat
from guaraci.config import ETAPAS_PRE_PROCESSAMENTO, Config
from guaraci.config_io import _CONFIG_SPEC, CAMPOS_INTERNOS, CHAVES_FORA_DO_YAML, load_config, save_config

SRC = Path(__file__).resolve().parents[1] / "src" / "guaraci"
#: chave -> campos reais por tras de uma propriedade
VIA_PROPRIEDADE = {"etapas_pre_processamento": [a for _, a in ETAPAS_PRE_PROCESSAMENTO], "imagem_recorte": ["image_crop"],
                  "imagem_referencia": ["image_reference"]}
#: arquivos de interface/definicao: uso ali nao conta como uso pelo motor
_INTERFACE = {"config.py", "config_io.py", "cli_assistente.py", "guaraci.py", "app_logic.py", "cli_logic.py"}
#: chaves cujo efeito e' so' na interface, com motivo
USO_SO_NA_INTERFACE = {
    "perfil_tecnica": "informativo: mostra resolucao e formatos esperados e serve para combinar perfis (catalogo diz que nao muda o calculo)",
    "ajuste_automatico": "lido pelas regras de contexto (contexto.py), que rodam antes do motor",
}


def _campos_com_rota():
    out = set()
    for s in _CONFIG_SPEC:
        out.update(VIA_PROPRIEDADE.get(s["attr"], [s["attr"]]))
    return out


def test_todo_campo_da_config_tem_chave_ou_motivo_declarado():
    campos = {f.name for f in dataclasses.fields(Config)}
    com_rota = _campos_com_rota()
    sem_rota = campos - com_rota - set(CAMPOS_INTERNOS)
    assert not sem_rota, f"campo(s) sem chave no config.yaml nem motivo em CAMPOS_INTERNOS: {sorted(sem_rota)}"
    assert not (com_rota & set(CAMPOS_INTERNOS)), "campo com chave nao pode estar em CAMPOS_INTERNOS"
    assert set(CAMPOS_INTERNOS) <= campos, "CAMPOS_INTERNOS cita campo que nao existe"
    assert all(len(m) > 20 for m in CAMPOS_INTERNOS.values())


def test_toda_chave_esta_no_catalogo_e_toda_opcao_esta_no_yaml_ou_declarada():
    chaves = {s["key"] for s in _CONFIG_SPEC}
    assert chaves <= set(cat.OPCOES), sorted(chaves - set(cat.OPCOES))
    fora = set(cat.OPCOES) - chaves
    assert fora == set(CHAVES_FORA_DO_YAML), (sorted(fora - set(CHAVES_FORA_DO_YAML)), sorted(set(CHAVES_FORA_DO_YAML) - fora))
    for k in chaves:
        o = cat.OPCOES[k]
        assert o["tela"] in cat.TELAS and "cli" in o["canais"], f"{k}: sem rota no CLI"


def test_toda_chave_e_usada_pelo_motor():
    fontes = [p.read_text(encoding="utf-8") for p in SRC.rglob("*.py")
              if p.name not in _INTERFACE and not p.name.startswith("catalogo") and "app_tabs" not in p.parts]
    sem_uso = []
    for s in _CONFIG_SPEC:
        if s["key"] in USO_SO_NA_INTERFACE:
            continue
        alvos = VIA_PROPRIEDADE.get(s["attr"], [s["attr"]])
        pat = re.compile("|".join(rf"\.{a}\b|['\"]{a}['\"]" for a in alvos))
        if not any(pat.search(t) for t in fontes):
            sem_uso.append(s["key"])
    assert not sem_uso, f"chave(s) do config.yaml que o motor nunca le: {sem_uso}"


def test_listas_de_excecao_tem_teto():
    assert len(CAMPOS_INTERNOS) <= 6 and len(CHAVES_FORA_DO_YAML) <= 6 and len(USO_SO_NA_INTERFACE) <= 2


def test_chaves_novas_vao_e_voltam_pelo_config_yaml(tmp_path):
    cfg = Config()
    cfg.default_preprocessing = "custom"
    cfg.etapas_pre_processamento = ["emsc", "sg", "mc"]
    cfg.imagem_recorte = (0.1, 0.1, 0.9, 0.8)
    cfg.n_splits_cv, cfg.seed, cfg.tag, cfg.cal_val_split = 4, 7, "rodada_x", "kennard_stone"
    caminho = tmp_path / "c.yaml"
    save_config(cfg, str(caminho))
    volta = load_config(str(caminho))
    assert volta.etapas_pre_processamento == ("emsc", "sg", "mc") and volta.apply_emsc and not volta.apply_snv
    assert volta.imagem_recorte == (0.1, 0.1, 0.9, 0.8)
    assert (volta.n_splits_cv, volta.seed, volta.tag, volta.cal_val_split) == (4, 7, "rodada_x", "kennard_stone")
    assert volta.default_preprocessing == "custom"


def test_etapa_ou_recorte_invalidos_sao_erro_com_mensagem(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("etapas_pre_processamento: [snv, xyz]\nimagem_recorte: [0.5, 0, 0.2, 1]\n", encoding="utf-8")
    try:
        load_config(str(p))
    except ValueError as e:
        assert "xyz" in str(e) and "imagem_recorte" in str(e)
    else:
        raise AssertionError("config invalido aceito")


def test_preset_personalizado_monta_as_etapas_pedidas():
    from guaraci.preprocessamento import build_preprocessor
    cfg = Config(default_preprocessing="custom")
    cfg.etapas_pre_processamento = ["airpls", "emsc", "mc"]
    assert [n for n, _ in build_preprocessor(cfg).steps] == ["airpls", "emsc", "mc"]
    cfg2 = Config(default_preprocessing="airpls_sg_mc")
    assert [n for n, _ in build_preprocessor(cfg2).steps] == ["airpls", "sg", "mc"]


def test_toda_tecnica_analitica_da_tela_8_aplica_e_volta_pelo_config(tmp_path):
    """Antes: UV-Vis, fluorescencia e RMN aplicavam o preset 'snv_mc', que o motor nao
    conhecia -- a tela aplicava a faixa e falhava no pre-processamento e no modo."""
    from guaraci.config_io import _coagir_valor
    _SPEC_BY_KEY = {s["key"]: s for s in _CONFIG_SPEC}
    from guaraci.preprocessamento import build_preprocessor, descrever_preprocessamento
    for k, ins in cat.INSTRUMENTOS.items():
        cfg = Config()
        for chave, valor in (("faixa_min_cm", ins["faixa_min"]), ("faixa_max_cm", ins["faixa_max"]),
                             ("pre_processamento", ins["preproc"]), ("modo_entrada", ins["mode"])):
            setattr(cfg, _SPEC_BY_KEY[chave]["attr"], _coagir_valor(_SPEC_BY_KEY[chave], valor))
        caminho = tmp_path / f"{k}.yaml"
        save_config(cfg, str(caminho))
        volta = load_config(str(caminho))
        assert volta.default_preprocessing == ins["preproc"], k
        assert build_preprocessor(volta).steps and descrever_preprocessamento(volta)[0] != ["raw"], k
