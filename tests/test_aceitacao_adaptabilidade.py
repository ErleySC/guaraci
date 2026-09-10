"""test_aceitacao_adaptabilidade.py -- Passo 117
(INSTRUCAO_PUSH_HIPOTESE_D_...md): auditoria de adaptabilidade a
multiplas matrizes/tecnicas, medida por comando direto, nao alegada.

Repete o teste de aceite multimatriz de `test_perfil_matriz.py` (trocar
de matriz e' trocar de perfil, zero linha de codigo) para os 3 modos
que faltavam: tabular com um perfil FICTICIO nunca visto pelo pacote
(prova que um usuario pode escrever seu proprio YAML sem tocar em
codigo-fonte, nao so' reusar os que ja vem embutidos), imagem
colorimetrica com uma TECNICA ficticia nova, e HSI com um dominio sem
nenhuma relacao com fruta (prova que o carregador generico do Passo 111
nao esta amarrado ao vocabulario/classes do DeepHS Fruit).

Tambem documenta 2 achados de arquitetura reais medidos durante esta
auditoria (reportados, NAO corrigidos sozinhos -- a instrucao pede
decisao de escopo antes de qualquer refatoracao):

  1. `identificacao.py` (Bloco 9b, Identificacao especie x adulterante)
     e' estruturalmente amarrado ao conceito de "adulterante" via
     `dados_io.adulterant_from_mae_id` (parsing de um padrao de nome
     especifico do dataset original de oleo, letras A/M/S) -- para
     qualquer matriz/mae_id que nao siga essa convencao (mode="imagem"
     incluso, mesmo com mae_id REAL de nivel "high"), a Identificacao
     roda sem erro mas produz SEMPRE 0 combinacoes, silenciosamente. Nao
     e' so' vocabulario -- e' a LOGICA de particionamento que esta
     amarrada. O proprio `model_card.md` ja' documenta a causa ("sem
     adulterante nomeavel") no addendum gerado por
     `resultados_io.append_identification_model_card`. Achado colateral
     de ORDEM (nao de acoplamento): `resumo["Identificacao (Bloco 9b)
     ..."]` (pipeline.py) e' escrito DEPOIS que `resumo_modelo.txt` ja'
     foi salvo em disco -- nunca aparece la', so' no model_card.md.
  2. `perfil_matriz.PERFIS_TECNICA` e' um frozenset de 3 nomes fixos
     ("bancada"/"celular"/"scanner") usado so' pra' filtrar
     `perfis_disponiveis(apenas="tecnica")` -- uma tecnica NOVA (ex.
     escrita por um usuario) carrega e funciona normalmente via
     `load_profile`/`combine_profiles`, mas nao aparece na listagem
     filtrada por "tecnica" (cai errado na lista de "matriz"). So'
     afeta listagem/descoberta em menu, nao a execucao do pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from guaraci.perfil_matriz import (PERFIS_TECNICA, combine_profiles,
                                   load_profile, perfis_disponiveis,
                                   save_profile)


# ── 1. Tabular: perfil FICTICIO, nunca embutido no pacote ──────────────────

def _csv_espectral_ficticio(caminho: Path, eixo: np.ndarray) -> None:
    import pandas as pd
    rng = np.random.default_rng(11)
    linhas, classes, concs = [], [], []
    for k, cls in enumerate(("Silvestre", "Cultivado")):
        for i in range(12):
            teor = float(i)
            base = 0.5 + 0.15 * k + 0.03 * np.sin(eixo / (eixo.max() / 5.0))
            linhas.append(base + 0.003 * teor + rng.normal(0, 0.002, eixo.size))
            classes.append(cls)
            concs.append(teor)
    df = pd.DataFrame(np.array(linhas), columns=[f"{v:.1f}" for v in eixo])
    df.insert(0, "classe", classes)
    df.insert(1, "conc", concs)
    df.to_csv(caminho, index=False)


def test_aceitacao_tabular_perfil_ficticio_escrito_pelo_usuario(pq, tmp_path):
    """Um perfil que NUNCA existiu no pacote (matriz inventada pra' este
    teste -- "resina de cupuacu"), escrito como YAML solto por um
    usuario hipotetico, carregado pelo CAMINHO (nao pelo nome -- prova
    que nao precisa estar dentro de src/guaraci/perfis_matriz/ pra'
    funcionar) e aplicado ao pipeline sem alterar NADA em src/guaraci/."""
    from conftest import achar_pastas_run

    caminho_perfil = tmp_path / "resina_cupuacu.yaml"
    caminho_perfil.write_text(yaml.safe_dump({
        "descricao": "Resina de cupuacu por NIR (perfil ficticio de teste)",
        "unidade_eixo": "cm-1",
        "eixo_min": 5000.0, "eixo_max": 9000.0,
        "default_preprocessing": "snv_sg_mc",
        "vocabulario": {
            "classe": "procedencia", "classe_plural": "procedencias",
            "matriz": "resina de cupuacu",
            "alvo": "o teor de resina sintetica adicionada",
            "conforme": "pura", "nao_conforme": "adulterada",
        },
        "codigos_classe": {}, "faixa_trabalho": None, "referencia": "",
    }), encoding="utf-8")

    eixo = np.linspace(5000.0, 9000.0, 180)
    csv = tmp_path / "espectros.csv"
    _csv_espectral_ficticio(csv, eixo)

    cfg = pq.Config(
        mode="csv", csv_file=str(csv), class_column="classe",
        conc_column="conc", output_root_folder=str(tmp_path / "saida"),
        matrix_profile=str(caminho_perfil), group_by_mae_id=False,
        n_splits_cv=2, n_repeats_cv=1, n_permutations=5,
        n_permutations_wold=5, n_bootstrap_vip=3, n_bootstrap_bca=20,
        n_monte_carlo=3, max_lvs=3, frac_holdout=0.0,
    )
    pq.executar(cfg)

    runs = achar_pastas_run(cfg.output_root_folder)
    assert runs, "executar() nao criou saida para o perfil ficticio"
    card = next(Path(runs[0]).rglob("model_card.md")).read_text(encoding="utf-8")
    assert "resina de cupuacu" in card
    assert "procedencia" in card
    # nunca deve vazar vocabulario de NENHUMA matriz embutida no pacote
    assert "oleo vegetal" not in card and "milho" not in card


# ── 2. Imagem colorimetrica: TECNICA ficticia, nunca embutida ──────────────

def _salvar_imagem(caminho: Path, rgb, tamanho=24, seed=0) -> None:
    rng = np.random.default_rng(seed)
    arr = np.zeros((tamanho, tamanho, 3), dtype=np.uint8)
    for c in range(3):
        arr[..., c] = np.clip(rgb[c] + rng.integers(-3, 4, size=(tamanho, tamanho)),
                              0, 255)
    Image.fromarray(arr, "RGB").save(caminho)


def test_aceitacao_imagem_tecnica_ficticia_nova(pq, tmp_path):
    """Tecnica de aquisicao que NAO existe no pacote ("microscopio_
    digital", inventada pra' este teste) -- combinada com o perfil de
    matriz "generico" (unico que nao trava o eixo espectral fora da
    faixa simbolica de imagem, ver achado documentado no docstring do
    modulo) e salva como YAML combinado, exatamente o fluxo que
    `combine_profiles`/`save_profile` existem pra' viabilizar (Agente
    5B). Roda mode="imagem" ponta-a-ponta sem alterar codigo-fonte."""
    from conftest import achar_pastas_run

    tecnica_ficticia_path = tmp_path / "microscopio_digital.yaml"
    tecnica_ficticia_path.write_text(yaml.safe_dump({
        "descricao": "Colorimetria via microscopio digital USB (tecnica ficticia)",
        "unidade_eixo": "indice",
        "default_preprocessing": "autoscaling",
        "vocabulario": {}, "codigos_classe": {}, "faixa_trabalho": None,
        "referencia": "",
        "resolucao_esperada": "1920x1080 minimo",
        "formatos_aceitos": [".png", ".tif"],
        "nivel_agrupamento_tipico": "high",
    }), encoding="utf-8")
    tecnica_ficticia = load_profile(str(tecnica_ficticia_path))

    matriz_generica = load_profile("generico")
    combinado = combine_profiles("generico_microscopio", matriz_generica,
                                 tecnica_ficticia)
    caminho_combinado = tmp_path / "generico_microscopio.yaml"
    save_profile(combinado, str(caminho_combinado))

    raiz = tmp_path / "fotos"
    for cls, rgb in (("Grupo1", (210, 190, 40)), ("Grupo2", (60, 130, 200))):
        (raiz / cls).mkdir(parents=True)
        for i in range(6):
            _salvar_imagem(raiz / cls / f"{cls}_{i}.png", rgb, seed=i)

    cfg = pq.Config(
        mode="imagem", input_folder=str(raiz),
        output_root_folder=str(tmp_path / "saida"),
        matrix_profile=str(caminho_combinado),
        wn_min=-1.0, wn_max=100.0,  # eixo simbolico de imagem (ver dados_imagem.py)
        n_splits_cv=2, n_repeats_cv=1, max_lvs=3,
        n_permutations=3, n_permutations_wold=3,
        n_bootstrap_vip=2, n_bootstrap_bca=10, n_monte_carlo=2,
        run_ddsimca=False, run_opls=False, executar_etapa4=False,
        run_wold=False, comparar_pipelines=False,
        run_cv_anova=False, run_benchmark=False,
        run_monte_carlo=False, run_shap=False,
    )
    pq.executar(cfg)

    runs = achar_pastas_run(cfg.output_root_folder)
    assert runs, "executar() nao criou saida para a tecnica ficticia"


def test_perfis_tecnica_e_generico_por_conteudo_nao_por_nome(tmp_path):
    """CORRIGIDO no Passo 124 (achado original do Passo 117, ver git
    blame/docs/PROGRESSO.md pra' o antes-e-depois): uma tecnica NOVA
    (nome fora do frozenset PERFIS_TECNICA) FUNCIONA via load_profile/
    combine_profiles (ver teste acima) E agora aparece corretamente na
    listagem filtrada `perfis_disponiveis(apenas="tecnica")` --
    classificacao e' por CONTEUDO (declara resolucao_esperada/
    formatos_aceitos/nivel_agrupamento_tipico), nunca por nome de
    arquivo contra uma lista fixa."""
    nome_tecnica_nova = "microscopio_digital_persistido"
    caminho = Path(__import__("guaraci.perfil_matriz", fromlist=["DIR_PERFIS"])
                  .DIR_PERFIS) / f"{nome_tecnica_nova}.yaml"
    caminho.write_text(yaml.safe_dump({
        "descricao": "tecnica nova, so' pra' medir a listagem",
        "unidade_eixo": "indice", "vocabulario": {},
        "resolucao_esperada": "N/A", "formatos_aceitos": [".png"],
        "nivel_agrupamento_tipico": "high",
    }), encoding="utf-8")
    try:
        tecnica = perfis_disponiveis(apenas="tecnica")
        matriz = perfis_disponiveis(apenas="matriz")
        assert nome_tecnica_nova in tecnica
        assert nome_tecnica_nova not in matriz
        assert PERFIS_TECNICA <= set(tecnica)  # os 3 pre-cadastrados continuam la
    finally:
        caminho.unlink(missing_ok=True)


# ── 3. HSI: dominio sem NENHUMA relacao com fruta ──────────────────────────

def _mascara_objeto_suave(n_lin, n_col, raio_frac=0.35, largura_borda=4.0):
    yy, xx = np.ogrid[:n_lin, :n_col]
    cy, cx = (n_lin - 1) / 2.0, (n_col - 1) / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    raio = raio_frac * min(n_lin, n_col)
    return np.clip((raio - dist) / largura_borda + 0.5, 0.0, 1.0)


def _gravar_cubo(caminho_sem_ext: Path, cubo: np.ndarray, n_bandas: int) -> None:
    n_lin, n_col, _ = cubo.shape
    caminho_sem_ext.parent.mkdir(parents=True, exist_ok=True)
    caminho_sem_ext.with_suffix(".bin").write_bytes(cubo.astype("<f4").tobytes())
    caminho_sem_ext.with_suffix(".hdr").write_text(
        f"ENVI\nsamples = {n_col}\nlines = {n_lin}\nbands = {n_bandas}\n"
        f"header offset = 0\nfile type = ENVI Standard\ndata type = 4\n"
        f"interleave = bip\nbyte order = 0\n", encoding="utf-8")


def test_aceitacao_hsi_dominio_sem_relacao_com_fruta(tmp_path):
    """HSI nao tem sistema de "perfil de matriz" (o rotulo de classe vem
    direto do nome da subpasta, ja generico por construcao -- Passo
    111) -- a prova de adaptabilidade aqui e' rodar um dominio TOTALMENTE
    alheio ao dataset publico usado nas outras validacoes (fruta
    madura/verde/etc.): autenticidade de comprimido farmaceutico
    ("autentico"/"falsificado"), zero relacao com ripeness_state."""
    from guaraci.config import Config
    from guaraci.hsi_pipeline import run_hsi_pipeline

    rng = np.random.default_rng(7)
    raiz = tmp_path / "comprimidos_hsi"
    n_lin, n_col, n_bandas = 64, 64, 6
    alpha = _mascara_objeto_suave(n_lin, n_col)[..., None]
    for classe, nivel in (("autentico", 0.80), ("falsificado", 0.50)):
        for a in range(3):
            fundo = rng.normal(loc=0.05, scale=0.01, size=(n_lin, n_col, n_bandas))
            objeto = rng.normal(loc=nivel, scale=0.01, size=(n_lin, n_col, n_bandas))
            cubo = alpha * objeto + (1.0 - alpha) * fundo
            _gravar_cubo(raiz / classe / f"lote{a}" / "vista0", cubo, n_bandas)

    cfg = Config(mode="hsi", hsi_dataset_folder=str(raiz),
                output_root_folder=str(tmp_path / "saida"),
                output_format="png")
    resumo = run_hsi_pipeline(cfg)

    assert resumo["n_gravacoes_total"] == 6
    assert resumo["grouping_guarantee"] == "high"
    val = resumo["validacao_externa"]
    for classe in val.classes:
        assert classe in ("autentico", "falsificado")
    assert set(val.classes) == {"autentico", "falsificado"}


# ── 4. Achado: Identificacao (Bloco 9b) amarrada ao conceito "adulterante" ──

def test_achado_identificacao_bloco9b_zero_combinacoes_fora_do_padrao_adulterante(
        pq, tmp_path):
    """Achado de arquitetura (reportado, NAO corrigido nesta rodada,
    decisao de escopo fica pro usuario -- ver docstring do modulo):
    `train_identification_ensemble` roda sem erro pra' QUALQUER matriz,
    mas so' produz combinacoes != 0 quando `mae_id` segue a convencao
    ESPECIFICA `{cod}-{data}-{adulterante}{teor}` (letras A/M/S) do
    dataset original de oleo (`dados_io.adulterant_from_mae_id`) -- nao
    e' so' vocabulario amarrado, e' a LOGICA de particionamento.

    mode="imagem" com nivel "high" (Bloco 8) e' o caso mais direto pra'
    medir isso: PRODUZ mae_id real (nao None -- "Classe/amostra0" etc.),
    entao a Identificacao nao para por falta de mae_id (esse seria um
    motivo trivial, ja' avisado em log) -- para porque `adulterant_
    from_mae_id("Classe/amostra0")` nao bate no regex esperado
    (`^[A-Za-z][0-9]` no ULTIMO segmento apos "-") e devolve None pra'
    toda amostra."""
    from conftest import achar_pastas_run

    raiz = tmp_path / "dados_img_identificacao"
    for cls, rgb in (("Esp_A", (210, 190, 40)), ("Esp_B", (60, 130, 200))):
        for a in range(4):
            pasta_amostra = raiz / cls / f"amostra{a}"
            pasta_amostra.mkdir(parents=True)
            for r in range(3):
                _salvar_imagem(pasta_amostra / f"foto{r}.png", rgb,
                               seed=hash((cls, a, r)) % 1000)

    cfg = pq.Config(
        mode="imagem", input_folder=str(raiz),
        output_root_folder=str(tmp_path / "saida"),
        wn_min=-1.0, wn_max=100.0, default_preprocessing="autoscaling",
        n_splits_cv=2, n_repeats_cv=1, max_lvs=3,
        n_permutations=3, n_permutations_wold=3,
        n_bootstrap_vip=2, n_bootstrap_bca=10, n_monte_carlo=2,
        run_ddsimca=False, run_opls=False, executar_etapa4=False,
        run_wold=False, comparar_pipelines=False,
        run_cv_anova=False, run_benchmark=False,
        run_monte_carlo=False, run_shap=False,
    )
    pq.executar(cfg)

    runs = achar_pastas_run(cfg.output_root_folder)
    # O resumo em memoria (`resumo["Identificacao (Bloco 9b) ..."]`) e'
    # escrito DEPOIS que resumo_modelo.txt ja' foi salvo em disco (achado
    # colateral desta auditoria -- ordem de execucao em pipeline.py, ~L2527
    # salva o arquivo ANTES do bloco "9b. Exportar modelo final" comecar
    # em ~L2567) -- entao nunca aparece la'. O addendum de Identificacao
    # vai pro model_card.md via `append_identification_model_card`
    # (resultados_io.py), que e' onde este teste verifica.
    card = (Path(runs[0]) / pq.NOME_RELATORIOS
           / "model_card.md").read_text(encoding="utf-8")
    assert "Identificacao especie x adulterante" in card
    # Achado MEDIDO: com mae_id fora da convencao de adulterante, o
    # proprio model_card.md ja' documenta a causa ("sem adulterante
    # nomeavel") -- confirma que a Identificacao roda sem erro mas fica
    # estruturalmente inoperante fora do padrao especifico de mae_id do
    # dataset original de oleo (nao e' so' vocabulario, ver docstring do
    # modulo). Se isto deixar de aparecer, o achado documentado aqui
    # pode estar obsoleto -- revisar antes de confiar nele.
    assert "Nenhuma combinacao especie x adulterante calibrada" in card
    assert "sem adulterante nomeavel" in card
