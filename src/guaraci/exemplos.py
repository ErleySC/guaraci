"""exemplos.py -- `guaraci exemplo <cenario>`: dado sintetico pequeno + config pronta.

Caminho dos primeiros 5 minutos: gera numa pasta um CSV sintetico (nenhum dado
real) e o `config.yaml` que roda aquele cenario de ponta a ponta com
`guaraci run config.yaml`. Sem numpy/pandas no import (so' dentro das funcoes).
ASCII por convencao do CLI.
"""
from __future__ import annotations

import os

from guaraci.config import nome_pasta
from typing import Any, Dict, List

__all__ = ["EXEMPLOS", "gerar", "comando_para_rodar", "listar"]

COMANDO_TRANSFERIR = "guaraci transferir --mestre mestre.csv --escravo escravo.csv --novos novos_escravo.csv"
COMANDO_VIP = "guaraci vip tabela.csv --classe classe --grupo amostra"
COMANDO_MONITORAR = "guaraci monitorar --referencia referencia.csv --novos novos.csv --lote lote --id amostra"
#: Explica o resultado do exemplo `vip`: so' v00-v02 foram construidas para separar as classes;
#: as demais variaveis sao ruido e algumas (v08, v11) passam de 1 por acaso.
NOTA_VIP = ("Ao rodar, v00, v01 e v02 sao as variaveis construidas para separar as classes. As outras 9 sao so' ruido;\n"
            "algumas (por exemplo v08 e v11) tambem podem passar de VIP 1 por acaso: o VIP e' relativo (a soma dos\n"
            "quadrados e' sempre igual ao numero de variaveis) e ha so' 24 amostras fisicas. Por isso VIP > 1 e'\n"
            "regra pratica, nao teste.")

# id -> grupo do catalogo, nome, faz, esperado (arquivos que a execucao cria)
EXEMPLOS: Dict[str, Dict[str, Any]] = {
    "explorar": dict(grupo="explorar", nome=('Explorar espectros (PCA, loadings)', 'Explore spectra (PCA, loadings)'),
        faz=('Olha a estrutura de espectros sinteticos de 3 classes, sem treinar modelo.', 'Looks at the structure of synthetic spectra of 3 classes, without training a model.'),
        comando="guaraci run config.yaml", esperado=("Relatorios/resumo_modelo.txt",)),
    "classificar": dict(grupo="classificar", nome=('Classificar 3 classes (PLS-DA)', 'Classify 3 classes (PLS-DA)'),
        faz=('Classifica espectros sinteticos de 3 especies com validacao por grupo (lote).', 'Classifies synthetic spectra of 3 species with group (batch) validation.'),
        comando="guaraci run config.yaml", esperado=("Relatorios/resumo_modelo.txt",)),
    "quantificar": dict(grupo="quantificar", nome=('Quantificar teor (PLS-R)', 'Quantify content (PLS-R)'),
        faz=('Estima um teor sintetico de 0 a 20 por regressao PLS com validacao por grupo.', 'Estimates a synthetic content from 0 to 20 by PLS regression with group validation.'),
        comando="guaraci run config.yaml", esperado=("Relatorios/resumo_modelo.txt",)),
    "multielementar": dict(grupo="comparar", nome=('Perfil multielementar (ICP)', 'Multi-element profile (ICP)'),
        faz=('Classifica origens e compara elementos sobre o lote, com censura e QC.', 'Classifies origins and compares elements over the batch, with censoring and QC.'),
        comando="guaraci run config.yaml", esperado=("Relatorios/multielementar.json",)),
    "comparar": dict(grupo="comparar", nome=('Comparar grupos (ANOVA sobre a unidade)', 'Compare groups (ANOVA over the unit)'),
        faz=('Compara a media de uma resposta entre 3 grupos, agregando por unidade independente.', 'Compares the mean of a response between 3 groups, aggregating by independent unit.'),
        comando="guaraci anova tabela.csv --resposta teor --grupo grupo --unidade amostra", esperado=("anova_resultado.csv",)),
    "imagem": dict(grupo="imagens", nome=("Classificar fotos por cor (modo imagem)", "Classify photos by colour (image mode)"),
        faz=("Gera 36 fotos sinteticas de 3 classes (6 amostras com 2 fotos cada por classe) e classifica pela cor. As classes (clara, media, escura) sao niveis ordenados de cor: a acuracia balanceada abaixo de 1,00 (cerca de 0,9) vem do PLS-DA, nao dos dados; com classificador: plslda (LDA sobre os escores) as 3 classes saem com 1,00.",
             "Generates 36 synthetic photos of 3 classes (6 samples with 2 photos each per class) and classifies by colour. The classes (light, medium, dark) are ordered colour levels: the balanced accuracy below 1.00 (about 0.9) comes from PLS-DA, not from the data; with classificador: plslda (LDA on the scores) the 3 classes reach 1.00."),
        comando="guaraci run config.yaml", esperado=("Relatorios/resumo_modelo.txt",)),
    "hsi": dict(grupo="imagens", nome=("Imageamento hiperespectral (cubos ENVI)", "Hyperspectral imaging (ENVI cubes)"),
        faz=("Gera 16 cubos ENVI sinteticos de 2 classes (4 objetos por classe, 2 vistas cada) e roda o fluxo HSI.",
             "Generates 16 synthetic ENVI cubes of 2 classes (4 objects per class, 2 views each) and runs the HSI flow."),
        comando="guaraci run config.yaml", esperado=("hsi_relatorio",)),
    "transferir": dict(grupo="quantificar", nome=("Transferir calibracao entre instrumentos (PDS)", "Transfer calibration between instruments (PDS)"),
        faz=("Gera 12 amostras lidas nos dois instrumentos e 20 espectros novos do segundo, e os converte para o primeiro.",
             "Generates 12 samples measured on two instruments and 20 new spectra from the second, and converts them to the first."),
        comando=COMANDO_TRANSFERIR, esperado=("resultados_transferencia/novos_no_espaco_do_mestre.csv",)),
    "monitorar": dict(grupo="monitorar", nome=('Monitorar lotes de um processo (T2/Q)', 'Monitor batches of a process (T2/Q)'),
        faz=('Calibra em 5 lotes em controle e acompanha 6 lotes novos, os 2 ultimos com deriva simulada.', 'Calibrates on 5 in-control batches and follows 6 new batches, the last 2 with simulated drift.'),
        comando=COMANDO_MONITORAR, esperado=("resultados_monitoramento/tabelas/monitor_lotes.csv",)),
    "fluorescencia_eem": dict(grupo="classificar", nome=('Classificar por fluorescencia excitacao-emissao (EEM)', 'Classify by excitation-emission fluorescence (EEM)'),
        faz=('Gera 18 matrizes excitacao-emissao sinteticas de 3 classes (2 fluoroforos, a classe muda a proporcao) e classifica o vetor achatado.',
             'Generates 18 synthetic excitation-emission matrices of 3 classes (2 fluorophores, the class changes their ratio) and classifies the flattened vector.'),
        comando="guaraci run config.yaml", esperado=("Relatorios/resumo_modelo.txt",)),
    "cromatograma_gcms": dict(grupo="classificar", nome=('Classificar cromatogramas (TIC de GC-MS)', 'Classify chromatograms (GC-MS TIC)'),
        faz=('Gera 18 cromatogramas de TIC sinteticos de 3 classes (razao entre dois picos e pequeno deslocamento de retencao) e classifica.',
             'Generates 18 synthetic TIC chromatograms of 3 classes (ratio between two peaks and a small retention shift) and classifies them.'),
        comando="guaraci run config.yaml", esperado=("Relatorios/resumo_modelo.txt",)),
    "vip": dict(grupo="modelar", nome=('VIP do PLS com estabilidade', 'PLS VIP with stability'),
        faz=('Gera 3 classes com 12 variaveis (so as 3 primeiras separam as classes) e mostra o VIP e a estabilidade nas dobras agrupadas.',
             'Generates 3 classes with 12 variables (only the first 3 separate the classes) and shows the VIP and its stability across grouped folds.'),
        comando=COMANDO_VIP, esperado=("resultados_vip/tabelas/vip_estabilidade.csv",)),
}


def listar() -> List[str]:
    return list(EXEMPLOS)


def _escrever(pasta: str, nome: str, texto: str) -> str:
    caminho = os.path.join(pasta, nome)
    with open(caminho, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(texto)
    return caminho


def _config(**kv: Any) -> str:
    linhas = []
    for k, v in kv.items():
        if isinstance(v, bool):
            v = "true" if v else "false"
        linhas.append(f"{k}: {v}")
    return "\n".join(linhas) + "\n"


def _csv_espectral(pasta: str, *, quantificar: bool, seed: int = 7) -> None:
    import numpy as np
    rng = np.random.default_rng(seed)
    eixo = np.arange(4000.0, 10000.0, 30.0)
    cab = ["classe", "lote"] + (["conc"] if quantificar else []) + [f"{v:.1f}" for v in eixo]
    linhas = [",".join(cab)]
    classes = ("especie_A", "especie_B") if quantificar else ("especie_A", "especie_B", "especie_C")
    for k, cls in enumerate(classes):
        picos = (5200 + 400 * k, 6900 - 250 * k, 8300 + 150 * k)
        for g in range(10 if quantificar else 8):
            conc = float(rng.uniform(0, 20)) if quantificar else 0.0
            amp = rng.normal(1.0, 0.05, 3)
            base = sum(a * np.exp(-((eixo - p) / 380.0) ** 2) for a, p in zip(amp, picos))
            if quantificar:
                base = base + 0.04 * conc * np.exp(-((eixo - 7600.0) / 300.0) ** 2)
            for _ in range(3):
                esp = base * rng.normal(1.0, 0.01) + rng.normal(0, 0.004, eixo.size)
                cel = [cls, f"{cls}_L{g:02d}"] + ([f"{conc:.3f}"] if quantificar else []) + [f"{x:.5f}" for x in esp]
                linhas.append(",".join(cel))
    _escrever(pasta, "dados.csv", "\n".join(linhas) + "\n")


def _csv_icp(pasta: str, seed: int = 3) -> None:
    import numpy as np
    rng = np.random.default_rng(seed)
    els = ["Na 23", "Mg 24", "K 39", "Ca 44", "Mn 55", "Fe 56", "Cu 63", "Zn 66", "Sr 88", "Cd 111"]
    linhas = ["Amostra;Tipo;Origem;Lote;Sessao;" + ";".join(els)]
    for o in range(3):
        for a in range(8):
            base = rng.lognormal(0, 0.3, len(els))
            for j, e in enumerate(els):
                if e in ("Mn 55", "Sr 88", "Zn 66"):
                    base[j] *= 1 + 1.2 * o / 2
            for r in range(2):
                vals = base * rng.lognormal(0, 0.05, len(els))
                cel = [f"{v:.4f}".replace(".", ",") for v in vals]
                cel[els.index("Cd 111")] = "<0,01" if rng.random() < 0.3 else f"{0.02 * (1 + rng.random()):.4f}".replace(".", ",")
                linhas.append(f"O{o}A{a}R{r};amostra;origem{o};O{o}L{a};s{(a % 2) + 1};" + ";".join(cel))
    # Linhas de controle de qualidade (ficam FORA do modelo; alimentam o QC do relatorio):
    #   ld        = limite de deteccao de cada elemento (da' numero ao "<LD" e ao teste do branco)
    #   branco    = branco de reagente (esperado abaixo do LD)
    #   crm       = o material de referencia medido (duas determinacoes)
    #   crm_cert  = o valor CERTIFICADO desse material (recuperacao = medido / certificado)
    ld = {e: (0.01 if e == "Cd 111" else 0.005) for e in els}
    cert = dict(zip(els, (1.5, 0.8, 1.1, 0.7, 1.0, 0.9, 0.6, 1.0, 1.0, 0.03)))
    fmt = lambda x: f"{x:.4f}".replace(".", ",")  # noqa: E731
    linhas.append("LD;ld;;;;" + ";".join(fmt(ld[e]) for e in els))
    linhas.append("Branco1;branco;;;s1;" + ";".join(fmt(0.4 * ld[e]) for e in els))
    linhas.append("MRC1;crm;;;s1;" + ";".join(fmt(1.04 * cert[e]) for e in els))
    linhas.append("MRC2;crm;;;s2;" + ";".join(fmt(0.94 * cert[e]) for e in els))
    linhas.append("MRC_certificado;crm_cert;;;;" + ";".join(fmt(cert[e]) for e in els))
    _escrever(pasta, "dados.csv", "\n".join(linhas) + "\n")


def _csv_comparar(pasta: str, seed: int = 5) -> None:
    import numpy as np
    rng = np.random.default_rng(seed)
    linhas = ["amostra,grupo,replica,teor"]
    for gi, (g, media) in enumerate((("controle", 10.0), ("tratamento_1", 11.0), ("tratamento_2", 14.0))):
        for a in range(8):
            efeito = rng.normal(0, 0.8)
            for r in range(3):
                linhas.append(f"{g}_{a:02d},{g},{r + 1},{media + efeito + rng.normal(0, 0.3):.4f}")
    _escrever(pasta, "tabela.csv", "\n".join(linhas) + "\n")


def _csv_vip(pasta: str, seed: int = 7) -> None:
    """3 classes, 8 amostras fisicas por classe com 3 replicas cada; 12 variaveis, so' v00-v02 separam as classes."""
    import numpy as np
    rng = np.random.default_rng(seed)
    desloc = {"A": -2.0, "B": 0.0, "C": 2.0}
    linhas = ["amostra,classe," + ",".join(f"v{i:02d}" for i in range(12))]
    for classe, d in desloc.items():
        for a in range(8):
            base = rng.normal(0, 1, 12)
            base[:3] += d
            for _ in range(3):
                linhas.append(f"{classe}{a:02d},{classe}," + ",".join(f"{x:.4f}" for x in base + rng.normal(0, 0.1, 12)))
    _escrever(pasta, "tabela.csv", "\n".join(linhas) + "\n")


def _csv_monitorar(pasta: str, seed: int = 11) -> None:
    """Processo sintetico: 5 lotes de referencia em controle; 6 lotes novos, os 2 ultimos com deriva."""
    import numpy as np
    rng = np.random.default_rng(seed)
    eixo = np.linspace(0.0, 1.0, 60)
    base = np.exp(-((eixo - 0.45) / 0.08) ** 2) + 0.6 * np.exp(-((eixo - 0.75) / 0.06) ** 2)
    deriva = np.exp(-((eixo - 0.60) / 0.05) ** 2)
    cab = "amostra,lote," + ",".join(f"v{i:02d}" for i in range(eixo.size))

    def bloco(prefixo, n_lotes, n_por, delta_de):
        linhas = []
        for lt in range(n_lotes):
            desvio_lote = rng.normal(0, 0.01, eixo.size)
            for a in range(n_por):
                x = base * rng.normal(1.0, 0.02) + desvio_lote + delta_de(lt) * deriva + rng.normal(0, 0.01, eixo.size)
                linhas.append(f"{prefixo}{lt:02d}_{a:02d},{prefixo}{lt:02d}," + ",".join(f"{v:.5f}" for v in x))
        return linhas
    ref = bloco("R", 5, 12, lambda lt: 0.0)
    nov = bloco("N", 6, 20, lambda lt: 0.0 if lt < 4 else 0.08 * (lt - 3))
    _escrever(pasta, "referencia.csv", "\n".join([cab] + ref) + "\n")
    _escrever(pasta, "novos.csv", "\n".join([cab] + nov) + "\n")


def _escrever_envi(pasta: str, nome: str, cubo: Any) -> None:
    """Par .hdr + .bin ENVI (float32, interleave bip) a partir de um cubo (linhas, colunas, bandas)."""
    import numpy as np
    n_lin, n_col, n_b = cubo.shape
    with open(os.path.join(pasta, f"{nome}.bin"), "wb") as fh:
        fh.write(np.asarray(cubo, dtype="<f4").tobytes())
    onda = ", ".join(f"{v:.1f}" for v in np.linspace(450.0, 950.0, n_b))
    _escrever(pasta, f"{nome}.hdr", "\n".join([
        "ENVI", f"samples = {n_col}", f"lines = {n_lin}", f"bands = {n_b}", "header offset = 0",
        "file type = ENVI Standard", "data type = 4", "interleave = bip", "byte order = 0",
        "wavelength units = Nanometers", f"wavelength = {{{onda}}}"]) + "\n")


def _fotos(pasta: str, seed: int = 21) -> None:
    """Classe/amostra/foto.png: quadrados coloridos com ruido; as classes diferem na cor media."""
    import numpy as np
    from matplotlib import image as mpimg
    rng = np.random.default_rng(seed)
    cores = {"clara": (0.85, 0.75, 0.40), "media": (0.70, 0.50, 0.25), "escura": (0.45, 0.30, 0.15)}
    for classe, rgb in cores.items():
        for a in range(6):
            dest = os.path.join(pasta, classe, f"amostra{a}")
            os.makedirs(dest, exist_ok=True)
            base = np.clip(np.array(rgb) + rng.normal(0, 0.03, 3), 0, 1)
            for f in range(2):
                img = np.clip(base + rng.normal(0, 0.04, (48, 48, 3)), 0, 1)
                mpimg.imsave(os.path.join(dest, f"foto{f}.png"), img)


def _cubos_hsi(pasta: str, seed: int = 13) -> None:
    """Classe/objeto/vista: um objeto claro sobre fundo escuro; as classes diferem num pico."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n, n_b = 24, 30
    yy, xx = np.mgrid[0:n, 0:n]
    disco = (yy - n / 2) ** 2 + (xx - n / 2) ** 2 <= (n / 3) ** 2
    onda = np.linspace(0, 1, n_b)
    for k, classe in enumerate(("verde", "madura")):
        base = 0.55 + 0.25 * np.exp(-((onda - (0.35 + 0.3 * k)) / 0.12) ** 2)
        for obj in range(6):          # >= 5 objetos por classe: extras da tela X (reamostragem, N-PLS) avaliaveis
            amostra = os.path.join(pasta, classe, f"objeto{obj}")
            os.makedirs(amostra, exist_ok=True)
            fator = rng.normal(1.0, 0.03)
            for vista in range(2):
                cubo = 0.05 + rng.normal(0, 0.01, (n, n, n_b))
                cubo[disco] = base * fator + rng.normal(0, 0.02, (int(disco.sum()), n_b))
                _escrever_envi(amostra, f"vista{vista}", cubo)


def _csv_transferir(pasta: str, seed: int = 17) -> None:
    """Instrumento escravo = mestre deslocado de 2 canais, com ganho e linha de base proprios."""
    import numpy as np
    rng = np.random.default_rng(seed)
    eixo = np.linspace(0, 1, 80)
    def mestre(n):
        a = rng.uniform(0.5, 1.5, (n, 2))
        return a[:, :1] * np.exp(-((eixo - 0.35) / 0.06) ** 2) + a[:, 1:] * np.exp(-((eixo - 0.7) / 0.08) ** 2)
    def escravo(X):
        return 1.08 * np.roll(X, 2, axis=1) + 0.03 + rng.normal(0, 0.003, X.shape)
    cab = ",".join(f"{v:.4f}" for v in eixo)
    Xm = mestre(12)
    for nome, X in (("mestre.csv", Xm), ("escravo.csv", escravo(Xm)), ("novos_escravo.csv", escravo(mestre(20)))):
        _escrever(pasta, nome, cab + "\n" + "\n".join(",".join(f"{v:.5f}" for v in x) for x in X) + "\n")


def gerar(cenario: str, pasta: str) -> Dict[str, Any]:
    """Cria os arquivos do exemplo em `pasta` e devolve {pasta, arquivos, comando}."""
    if cenario not in EXEMPLOS:
        raise ValueError(f"cenario desconhecido: {cenario!r}; disponiveis: {', '.join(EXEMPLOS)}")
    os.makedirs(pasta, exist_ok=True)
    pasta = os.path.abspath(pasta)
    csv = os.path.join(pasta, "dados.csv")           # caminhos absolutos: o config roda de qualquer pasta
    base = dict(abrir_figuras_na_tela=False, pasta_saida=os.path.join(pasta, nome_pasta("resultados")), n_permutacoes=30, max_lvs=8)
    if cenario == "explorar":
        _csv_espectral(pasta, quantificar=False)
        cfg = dict(modo_entrada="csv", arquivo_csv=csv, coluna_classe="classe", coluna_grupo="lote",
                   nivel="N1", objetivo="exploratorio", **base)
    elif cenario == "classificar":
        _csv_espectral(pasta, quantificar=False)
        cfg = dict(modo_entrada="csv", arquivo_csv=csv, coluna_classe="classe", coluna_grupo="lote",
                   nivel="N1", objetivo="classificacao", **base)
    elif cenario == "quantificar":
        _csv_espectral(pasta, quantificar=True)
        cfg = dict(modo_entrada="csv", arquivo_csv=csv, coluna_classe="classe", coluna_concentracao="conc",
                   coluna_grupo="lote", nivel="N3", objetivo="quantificacao", **base)
    elif cenario == "multielementar":
        _csv_icp(pasta)
        cfg = dict(modo_entrada="multielementar", arquivo_csv=csv, coluna_classe="Origem", coluna_grupo="Lote",
                   icp_unidade="mg/kg", icp_coluna_amostra="Amostra", icp_coluna_tipo="Tipo", icp_coluna_sessao="Sessao",
                   pre_processamento="autoscaling", **base)
    elif cenario == "monitorar":
        _csv_monitorar(pasta)
        cfg = None
    elif cenario == "vip":
        _csv_vip(pasta)
        cfg = None
    elif cenario == "transferir":
        _csv_transferir(pasta)
        cfg = None
    elif cenario == "imagem":
        _fotos(os.path.join(pasta, "fotos"))
        cfg = dict(modo_entrada="imagem", pasta_dados=os.path.join(pasta, "fotos"), nivel="N1", objetivo="classificacao",
                   **base)
    elif cenario == "fluorescencia_eem":
        from guaraci.formatos_sinteticos import gerar_eem
        gerar_eem(os.path.join(pasta, "eem"))
        # mc = o preset da tecnica 'eem' em [2.4]; o padrao MSC+SG+MC derivaria o vetor achatado atravessando as linhas de emissao
        cfg = dict(modo_entrada="eem", pasta_dados=os.path.join(pasta, "eem"), nivel="N1", objetivo="classificacao",
                   pre_processamento="mc", **base)
    elif cenario == "cromatograma_gcms":
        from guaraci.formatos_sinteticos import gerar_cromatogramas_tic
        gerar_cromatogramas_tic(os.path.join(pasta, "cromatogramas"))
        cfg = dict(modo_entrada="gcms", pasta_dados=os.path.join(pasta, "cromatogramas"), nivel="N1", objetivo="classificacao", pre_processamento="mc", **base)
    elif cenario == "hsi":
        _cubos_hsi(os.path.join(pasta, "cubos"))
        # snv_mc e faixa aberta = o que a tela [2.4] grava para HSI (sem eles o fluxo avisa que a faixa padrao
        # 4000-10000 nao cobre o eixo e que o preset espectral nao se aplica ao HSI)
        cfg = dict(modo_entrada="hsi", hsi_pasta_dataset=os.path.join(pasta, "cubos"), pre_processamento="snv_mc",
                   faixa_min_cm=0, faixa_max_cm=999999, **base)
    else:  # comparar
        _csv_comparar(pasta)
        cfg = None
    arquivos = {"comparar": ["tabela.csv"], "vip": ["tabela.csv"], "monitorar": ["referencia.csv", "novos.csv"],
                "transferir": ["mestre.csv", "escravo.csv", "novos_escravo.csv"],
                "hsi": ["cubos"], "imagem": ["fotos"], "fluorescencia_eem": ["eem"],
                "cromatograma_gcms": ["cromatogramas"]}.get(cenario, ["dados.csv"])
    if cfg is not None:
        _escrever(pasta, "config.yaml", "# Gerado por `guaraci exemplo " + cenario + "` (dados sinteticos).\n" + _config(**cfg))
        arquivos.append("config.yaml")
    return dict(pasta=pasta, arquivos=arquivos, comando=comando_para_rodar(cenario), cenario=cenario)


def comando_para_rodar(cenario: str) -> str:
    return EXEMPLOS[cenario]["comando"]
