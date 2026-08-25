"""Testes de funcoes puras extraidas de figuras.py (matematica de plotagem,
testavel sem renderizar nenhuma figura de verdade).
"""
import numpy as np

from guaraci.figuras import _escala_vetores_biplot


def test_escala_vetores_biplot_respeita_ambos_os_eixos():
    """REGRESSAO: um bug real deixava vetores com componente forte no eixo
    de MENOR alcance (aqui, PC2) desenhados MUITO alem da area visivel --
    rotulos apareciam flutuando fora do grafico renderizado. Reproduz o
    cenario (PC1 com alcance ~10x maior que PC2, tipico quando os
    autovalores sao desiguais) e prova que a escala escolhida mantem TODO
    vetor dentro de `frac` do maior score de CADA eixo, nao so' do eixo
    dominante."""
    rng = np.random.default_rng(0)
    n, p = 60, 40
    scores2 = np.column_stack([
        rng.uniform(-0.05, 0.05, n),     # PC1: alcance grande
        rng.uniform(-0.005, 0.005, n),   # PC2: alcance 10x menor
    ])
    loadings = rng.uniform(-0.5, 0.5, size=(p, 2))
    # Garante que ALGUMAS variaveis tenham componente PC2 dominante (o
    # cenario que disparava o bug).
    loadings[0] = [0.02, 0.45]

    frac = 0.8
    escala = _escala_vetores_biplot(scores2, loadings, frac=frac)
    vx = loadings[:, 0] * escala
    vy = loadings[:, 1] * escala

    max_score_x = np.abs(scores2[:, 0]).max()
    max_score_y = np.abs(scores2[:, 1]).max()
    # Tolerancia de ponto flutuante minima -- a garantia e' "nao ultrapassa
    # frac do maior score daquele eixo", nao um limite exato.
    assert np.all(np.abs(vx) <= frac * max_score_x + 1e-12)
    assert np.all(np.abs(vy) <= frac * max_score_y + 1e-12)


def test_escala_vetores_biplot_escala_positiva_com_dados_degenerados():
    """Scores ou loadings todos zero (caso degenerado) nao devem gerar
    escala negativa, NaN ou ZeroDivisionError."""
    scores2 = np.zeros((5, 2))
    loadings = np.zeros((10, 2))
    escala = _escala_vetores_biplot(scores2, loadings)
    assert np.isfinite(escala)
    assert escala >= 0


# ---------------------------------------------------------------------------
# Biplot: rotulos sobrepostos e loadings redundantes (achado 2026-08-07)
# ---------------------------------------------------------------------------
def _n_sobreposicoes(pos, sep_x, sep_y):
    n = len(pos)
    return sum(1 for a in range(n) for b in range(a + 1, n)
               if abs(pos[a, 0] - pos[b, 0]) < sep_x - 1e-9
               and abs(pos[a, 1] - pos[b, 1]) < sep_y - 1e-9)


def test_afastar_rotulos_elimina_toda_sobreposicao():
    """REGRESSAO: no biplot real os rotulos de numero de onda saiam
    impressos uns por cima dos outros (ilegiveis). A funcao tem que
    garantir separacao para QUALQUER entrada, inclusive o pior caso de
    rotulos exatamente coincidentes."""
    from guaraci.figuras import spread_labels
    rng = np.random.default_rng(0)
    sep_x, sep_y = 0.05, 0.02
    casos = {
        "coincidentes":  np.zeros((12, 2)),
        "linha_vertical": np.column_stack([np.zeros(12),
                                           np.linspace(0, 0.05, 12)]),
        "aglomerado":    rng.normal(scale=0.01, size=(20, 2)),
        "espalhado":     rng.normal(size=(50, 2)) * 0.1,
    }
    for nome, pos in casos.items():
        out = spread_labels(pos, sep_x, sep_y)
        assert _n_sobreposicoes(out, sep_x, sep_y) == 0, (
            f"caso '{nome}': ainda ha rotulos sobrepostos")
        assert out.shape == pos.shape
        assert np.all(np.isfinite(out))


def test_afastar_rotulos_e_deterministico_e_preserva_x():
    """A figura precisa ser reproduzivel (sem RNG), e o passo vertical nao
    pode mexer em x -- e' o que garante a separacao entre colunas."""
    from guaraci.figuras import spread_labels
    pos = np.column_stack([np.tile([0.0, 0.5], 6), np.zeros(12)])
    a = spread_labels(pos, 0.05, 0.02)
    b = spread_labels(pos, 0.05, 0.02)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a[:, 0], pos[:, 0])


def test_afastar_rotulos_casos_degenerados():
    from guaraci.figuras import spread_labels
    for pos in (np.zeros((0, 2)), np.zeros((1, 2))):
        out = spread_labels(pos, 0.05, 0.02)
        assert out.shape == pos.shape


def test_selecionar_loadings_distintos_evita_canais_vizinhos():
    """REGRESSAO: pegar o top-N por magnitude devolvia canais ADJACENTES da
    mesma banda (no espectro real: 5875/5883/5891/5899... = 2 bandas
    contadas 12 vezes). Cada seta do biplot tem que representar uma banda
    espectral distinta."""
    from guaraci.figuras import select_distinct_loadings
    wn = np.linspace(4000, 10000, 759)
    # duas bandas estreitas fortes + uma larga fraca
    mag = (np.exp(-((wn - 5900) / 60) ** 2)
           + 0.9 * np.exp(-((wn - 4450) / 50) ** 2)
           + 0.15 * np.exp(-((wn - 7100) / 200) ** 2))

    ingenuo = np.argsort(mag)[::-1][:12]
    assert len(np.unique(np.round(wn[ingenuo] / 200))) <= 3, (
        "premissa do teste: o top-N ingenuo concentra em poucas bandas")

    idx = select_distinct_loadings(mag, wn, 12)
    assert 1 <= len(idx) <= 12
    assert len(np.unique(idx)) == len(idx), "indices repetidos"
    escolhidos = np.sort(wn[idx])
    if len(escolhidos) > 1:
        sep_min = float(np.diff(escolhidos).min())
        faixa = wn.max() - wn.min()
        assert sep_min >= faixa / (12 * 3.0) - 1e-6, (
            f"bandas ainda coladas (menor separacao: {sep_min:.0f} cm-1)")
    # a banda mais forte tem que continuar entre as escolhidas
    assert np.min(np.abs(wn[idx] - 5900)) < 100


def test_selecionar_loadings_nao_completa_cota_com_ruido():
    """REGRESSAO: exigir separacao espectral SEM piso de magnitude fazia a
    funcao completar a cota com canais de magnitude ~0 -- o biplot ganhava
    setas de comprimento nulo empilhadas na origem ("linhas que nao dizem
    nada"). Com so' 2 bandas reais, tem que devolver ~2 indices, nao 12."""
    from guaraci.figuras import select_distinct_loadings
    wn = np.linspace(4000, 10000, 759)
    mag = (np.exp(-((wn - 5900) / 60) ** 2)
           + 0.9 * np.exp(-((wn - 4450) / 50) ** 2))   # so' 2 bandas
    idx = select_distinct_loadings(mag, wn, 12)
    assert len(idx) <= 4, (
        f"devolveu {len(idx)} vetores para um espectro com 2 bandas — "
        "a cota esta sendo completada com ruido")
    assert (mag[idx] >= 0.15 * mag.max()).all(), "entrou variavel abaixo do piso"


def test_selecionar_loadings_espectro_plano_devolve_ao_menos_um():
    """Espectro sem banda alguma nao pode devolver lista vazia (quebraria a
    figura); devolve o maior, e a figura fica honestamente pobre."""
    from guaraci.figuras import select_distinct_loadings
    wn = np.linspace(4000, 10000, 100)
    idx = select_distinct_loadings(np.ones(100), wn, 12)
    assert len(idx) >= 1


def test_selecionar_loadings_distintos_respeita_n_disponivel():
    from guaraci.figuras import select_distinct_loadings
    wn = np.linspace(4000, 4100, 5)
    mag = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
    idx = select_distinct_loadings(mag, wn, 12)
    assert 1 <= len(idx) <= 5
    assert select_distinct_loadings(np.array([]), np.array([]), 5).size == 0


# ---------------------------------------------------------------------------
# DD-SIMCA acceptance plot: parede de pontos no piso fixo (achado 2026-08-07)
# ---------------------------------------------------------------------------
def test_limites_log_ddsimca_nao_colapsa_maioria_no_piso():
    """REGRESSAO: com n_comp=1 (comum quando so' ha' 3 amostras puras de
    treino), a maioria das amostras de outras classes projeta perto de
    zero em T2 -- medido em cenario real: 91% dos pontos caiam abaixo do
    piso FIXO de 1e-2 antigo, todos empilhados na mesma coluna de pixels
    (parecia bug de renderizacao). O piso dinamico tem que refletir a
    dispersao real dos dados, nao esconder 90%+ deles atras de uma parede."""
    from guaraci.figuras import _limites_log_ddsimca
    rng = np.random.default_rng(0)
    # Simula o padrao medido: a maioria dos valores entre 1e-10 e 1e-2,
    # uma minoria (~10%) acima de 1.0
    baixos = 10 ** rng.uniform(-10, -2, 900)
    altos = 10 ** rng.uniform(-1, 0.5, 100)
    valores = np.concatenate([baixos, altos])

    piso, teto = _limites_log_ddsimca(valores)
    fracao_visivel = float(np.mean(valores >= piso))
    assert fracao_visivel > 0.5, (
        f"piso dinamico ainda esconde a maioria dos pontos "
        f"(so' {fracao_visivel:.0%} visiveis) — parede nao foi eliminada")
    assert piso < 1e-2, "piso deveria descer abaixo do antigo valor fixo"
    assert teto > 1.0


def test_limites_log_ddsimca_nao_estica_por_um_unico_zero():
    """Um unico valor colapsado por underflow numerico (T2~0 exato) nao
    pode esticar o piso ate um extremo absurdo — usa percentil, nao o
    minimo bruto."""
    from guaraci.figuras import _limites_log_ddsimca
    valores = np.concatenate([np.full(99, 0.5), [1e-300]])
    piso, _teto = _limites_log_ddsimca(valores)
    assert piso >= 1e-6, "um unico outlier extremo nao deveria dominar o piso"


def test_limites_log_ddsimca_caso_degenerado():
    from guaraci.figuras import _limites_log_ddsimca
    piso, teto = _limites_log_ddsimca(np.array([]))
    assert np.isfinite(piso) and np.isfinite(teto)
    piso, teto = _limites_log_ddsimca(np.zeros(10))
    assert np.isfinite(piso) and np.isfinite(teto)


def test_limites_log_ddsimca_dados_bem_comportados_nao_alarga_demais():
    """Quando os dados NAO tem o problema (poucos valores extremos, a
    maioria perto da regiao de aceite), o piso nao deve descer
    desnecessariamente -- fica perto do antigo 1e-2."""
    from guaraci.figuras import _limites_log_ddsimca
    rng = np.random.default_rng(1)
    valores = 10 ** rng.uniform(-1.5, 0.3, 500)   # tudo entre ~0.03 e ~2
    piso, _teto = _limites_log_ddsimca(valores)
    assert piso >= 1e-3, "piso desceu sem necessidade para dados bem comportados"


# ---------------------------------------------------------------------------
# Grafico de permutacao de Wold: piso fixo cortava pontos (achado 2026-08-07)
# ---------------------------------------------------------------------------
def test_ylim_permutacao_nunca_corta_ponto():
    """REGRESSAO (bug LATENTE): o piso do eixo Y era fixo em -0.5/-0.6.
    Q2Y de rotulos permutados fica mais negativo quanto MAIS componentes o
    modelo usa. Medido num problema multiclasse tipico: 23 LVs -> minimo
    -0.465 (cabe), mas
    40 LVs (o max_lvs em uso) -> 80% dos pontos abaixo de -0.6, sumindo do
    grafico enquanto a reta de regressao seguia calculada sobre eles."""
    from guaraci.figuras import _ylim_permutacao
    casos = {
        "23_lvs":    (np.array([-0.465, -0.41, -0.38]), 0.43),
        "40_lvs":    (np.array([-0.647, -0.62, -0.70]), 0.43),
        "extremo":   (np.array([-2.5, -1.8, -0.9]), 0.40),
        "positivos": (np.array([0.1, 0.2, 0.05]), 0.90),
    }
    for nome, (vals, obs) in casos.items():
        lo, hi = _ylim_permutacao(vals, obs, base=-0.6)
        assert lo <= vals.min(), f"caso '{nome}': piso corta pontos de permutacao"
        assert lo <= obs, f"caso '{nome}': piso corta o ponto observado"
        assert hi >= obs


def test_ylim_permutacao_preserva_base_quando_cabe():
    """Execucoes normais nao podem mudar de aparencia: se todos os pontos
    cabem no piso padrao, o piso padrao e' mantido."""
    from guaraci.figuras import _ylim_permutacao
    lo, _hi = _ylim_permutacao(np.array([-0.3, -0.2, 0.1]), 0.5, base=-0.6)
    assert lo == -0.6


def test_ylim_permutacao_entrada_degenerada():
    from guaraci.figuras import _ylim_permutacao
    for vals, obs in ((np.array([np.nan, np.nan]), np.nan),
                      (np.array([]), 0.5)):
        lo, hi = _ylim_permutacao(vals, obs)
        assert np.isfinite(lo) and np.isfinite(hi) and lo < hi


# ---------------------------------------------------------------------------
# Fronteira de aceitacao verdadeira do DD-SIMCA (corrigido 2026-08-08)
# ---------------------------------------------------------------------------
def test_fronteira_ddsimca_reta_diagonal_nao_caixa():
    """REGRESSAO: o grafico desenhava DUAS linhas retas perpendiculares
    (T2_norm=1, Q_norm=1) -- uma caixa. A fronteira real do modelo
    corrigido e' UMA reta diagonal unica. A curva devolvida tem que VARIAR
    com T2_norm (nao ser constante em 1.0 -- senao ainda seria a caixa
    antiga disfarcada)."""
    from guaraci.figuras import _fronteira_ddsimca
    m = {"h0": 2.0, "q0": 50.0, "Nh": 3.0, "Nq": 40.0,
         "T2_ucl": 5.0, "Q_ucl": 80.0}
    m["f_crit"] = (5.0 / 2.0) * 3.0 + (80.0 / 50.0) * 40.0   # ponto (1,1) na fronteira
    grid = np.array([0.1, 0.5, 1.0, 2.0])
    q_front = _fronteira_ddsimca(m, grid)
    validos = q_front[np.isfinite(q_front)]
    assert validos.size >= 2
    assert float(np.ptp(validos)) > 1e-6, "fronteira constante -- ainda e' a caixa antiga"
    # Ponto de calibracao: em T2_norm=1, a fronteira passa por Q_norm=1
    idx1 = list(grid).index(1.0)
    assert abs(q_front[idx1] - 1.0) < 1e-9


def test_fronteira_ddsimca_decrescente():
    """A fronteira tem inclinacao negativa: quanto mais T2 'gasta' do
    orcamento de f_crit, menos sobra para Q -- e' a mesma logica de
    trade-off de qualquer reta A*x+B*y=const com A,B>0."""
    from guaraci.figuras import _fronteira_ddsimca
    m = {"h0": 2.0, "q0": 50.0, "Nh": 3.0, "Nq": 40.0,
         "T2_ucl": 5.0, "Q_ucl": 80.0, "f_crit": 100.0}
    grid = np.linspace(0.05, 5.0, 50)
    q_front = _fronteira_ddsimca(m, grid)
    validos = q_front[np.isfinite(q_front)]
    idx_validos = np.where(np.isfinite(q_front))[0]
    assert np.all(np.diff(validos) <= 1e-9), "fronteira nao e' monotona decrescente"
    assert len(idx_validos) >= 2


def test_fronteira_ddsimca_campos_ausentes_nao_quebra():
    from guaraci.figuras import _fronteira_ddsimca
    grid = np.array([0.1, 1.0, 10.0])
    out = _fronteira_ddsimca({}, grid)
    assert out.shape == grid.shape
    assert np.all(np.isnan(out))


def test_salvar_carimba_prototipo_apenas_no_modo_imagem(tmp_path):
    """Achado B4-1 (residual): os relatorios PDF/Word/LaTeX ja saem
    carimbados no mode imagem, mas uma figura .png exportada solta da pasta
    Graficos/ circulava sem nenhum contexto -- e' justamente o arquivo que
    acaba colado num slide. `save()` e' o ponto unico por onde toda figura
    passa, entao a marca entra la'.

    Verifica a PROPRIEDADE (um texto de aviso foi adicionado a figura), nao
    pixels: o teste falha se a marca sumir ou vazar para o mode normal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from guaraci.figuras import save
    from guaraci.config import Config, NOME_GRAFICOS

    def _texts_da_figura(mode):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        cfg = Config(mode=mode, output_format="png")
        # captura os textos ANTES de salvar fechar a figura
        registrados = {}
        orig_savefig = fig.savefig

        def _spy(caminho, *a, **kw):
            registrados["textos"] = [t.get_text() for t in fig.texts]
            return orig_savefig(caminho, *a, **kw)

        fig.savefig = _spy   # type: ignore[method-assign]
        save(fig, "fig_teste", str(tmp_path / mode), cfg)
        assert (tmp_path / mode / NOME_GRAFICOS / "fig_teste.png").exists()
        return registrados.get("textos", [])

    textos_img = " ".join(_texts_da_figura("imagem"))
    textos_dx = " ".join(_texts_da_figura("dx"))

    assert "PROTOTIPO" in textos_img and "NAO VALIDADO" in textos_img
    assert "PROTOTIPO" not in textos_dx
