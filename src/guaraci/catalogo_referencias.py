"""catalogo_referencias.py -- referencias de cada tecnica e metodo, para a tela Sobre (6.3).

Regra do projeto: nenhum DOI entra aqui sem constar em
`docs/cientifico/referencias_verificadas.yaml` (conferido contra Crossref/DataCite).
`tests/test_catalogo_referencias.py` confere DOI, primeiro autor, ano e o inicio do
titulo contra esse registro, e exige que toda tecnica de `catalogo_tecnicas.TECNICAS`
apareca aqui -- com referencia ou, se for procedimento do proprio GUARACI, com `proprio=True`.

Item sem referencia so' existe para procedimento ou configuracao do proprio GUARACI
(`proprio=True`), com `nota` dizendo isso. Metodo de terceiros sem fonte conferida e'
lacuna: `lacunas()` a lista e o teste exige que nao haja nenhuma.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple


class Ref(NamedTuple):
    autores: str
    ano: int
    titulo: str
    doi: str


class Item(NamedTuple):
    metodo: str                   # nome curto do metodo (igual em PT e EN)
    refs: Tuple[str, ...]         # chaves de REFS
    tecnica: Optional[str] = None # id em catalogo_tecnicas.TECNICAS, se houver
    nota: str = ""                # lacuna ou observacao (PT)
    proprio: bool = False         # procedimento/configuracao do GUARACI: sem fonte externa por natureza


REFS: Dict[str, Ref] = {
    # -- PLS e regressao
    "geladi1986": Ref("Geladi, P.; Kowalski, B. R.", 1986, "Partial least-squares regression: a tutorial", "10.1016/0003-2670(86)80028-9"),
    "wold2001": Ref("Wold, S.; Sjostrom, M.; Eriksson, L.", 2001, "PLS-regression: a basic tool of chemometrics", "10.1016/S0169-7439(01)00155-1"),
    "andersson2009": Ref("Andersson, M.", 2009, "A comparison of nine PLS1 algorithms", "10.1002/cem.1248"),
    "alsouki2023": Ref("Alsouki, L. et al.", 2023, "Dual-sPLS: a family of Dual Sparse Partial Least Squares regressions for feature selection and prediction with tunable sparsity; evaluation on simulated and near-infrared (NIR) data", "10.1016/j.chemolab.2023.104813"),
    "smola2004": Ref("Smola, A. J.; Scholkopf, B.", 2004, "A tutorial on support vector regression", "10.1023/B:STCO.0000035301.49549.88"),
    # -- classificacao e autenticacao
    "brereton2014": Ref("Brereton, R. G.; Lloyd, G. R.", 2014, "Partial least squares discriminant analysis: taking the magic away", "10.1002/cem.2609"),
    "westerhuis2008": Ref("Westerhuis, J. A. et al.", 2008, "Assessment of PLSDA cross validation", "10.1007/s11306-007-0099-6"),
    "trygg2002": Ref("Trygg, J.; Wold, S.", 2002, "Orthogonal projections to latent structures (O-PLS)", "10.1002/cem.695"),
    "bylesjo2006": Ref("Bylesjo, M. et al.", 2006, "OPLS discriminant analysis: combining the strengths of PLS-DA and SIMCA classification", "10.1002/cem.1006"),
    "pomerantsev2008": Ref("Pomerantsev, A. L.", 2008, "Acceptance areas for multivariate classification derived by projection methods", "10.1002/cem.1147"),
    "pomerantsev2013": Ref("Pomerantsev, A. L.; Rodionova, O. Y.", 2013, "Concept and role of extreme objects in PCA/SIMCA", "10.1002/cem.2506"),
    "pomerantsev2020": Ref("Pomerantsev, A. L.; Rodionova, O. Y.", 2020, "Popular decision rules in SIMCA: critical review", "10.1002/cem.3250"),
    "rodionova2016": Ref("Rodionova, O. Y.; Titova, A. V.; Pomerantsev, A. L.", 2016, "Discriminant analysis is an inappropriate method of authentication", "10.1016/j.trac.2016.01.010"),
    "kucheryavskiy2020": Ref("Kucheryavskiy, S.", 2020, "mdatools - R package for chemometrics", "10.1016/j.chemolab.2020.103937"),
    # -- pre-processamento
    "savitzky1964": Ref("Savitzky, A.; Golay, M. J. E.", 1964, "Smoothing and differentiation of data by simplified least squares procedures", "10.1021/ac60214a047"),
    "geladi1985": Ref("Geladi, P.; MacDougall, D.; Martens, H.", 1985, "Linearization and scatter-correction for near-infrared reflectance spectra of meat", "10.1366/0003702854248656"),
    "barnes1989": Ref("Barnes, R. J.; Dhanoa, M. S.; Lister, S. J.", 1989, "Standard normal variate transformation and de-trending of near-infrared diffuse reflectance spectra", "10.1366/0003702894202201"),
    "martens1991": Ref("Martens, H.; Stark, E.", 1991, "Extended multiplicative signal correction and spectral interference subtraction: new preprocessing methods for near infrared spectroscopy", "10.1016/0731-7085(91)80188-F"),
    "afseth2012": Ref("Afseth, N. K.; Kohler, A.", 2012, "Extended multiplicative signal correction in vibrational spectroscopy, a tutorial", "10.1016/j.chemolab.2012.03.004"),
    "wold1998": Ref("Wold, S. et al.", 1998, "Orthogonal signal correction of near-infrared spectra", "10.1016/S0169-7439(98)00109-9"),
    "roger2003": Ref("Roger, J.-M.; Chauchard, F.; Bellon-Maurel, V.", 2003, "EPO-PLS external parameter orthogonalisation of PLS application to temperature-independent measurement of sugar content of intact fruits", "10.1016/S0169-7439(03)00051-0"),
    "martens2003": Ref("Martens, H. et al.", 2003, "Pre-whitening of data by covariance-weighted pre-processing", "10.1002/cem.780"),
    "whitaker2018": Ref("Whitaker, D. A.; Hayes, K.", 2018, "A simple algorithm for despiking Raman spectra", "10.1016/j.chemolab.2018.06.009"),
    "zhang2010": Ref("Zhang, Z.-M.; Chen, S.; Liang, Y.-Z.", 2010, "Baseline correction using adaptive iteratively reweighted penalized least squares", "10.1039/b922045c"),
    "dieterle2006": Ref("Dieterle, F. et al.", 2006, "Probabilistic quotient normalization as robust method to account for dilution of complex biological mixtures. Application in 1H NMR metabonomics", "10.1021/ac051632c"),
    "rinnan2009": Ref("Rinnan, A.; van den Berg, F.; Engelsen, S. B.", 2009, "Review of the most common pre-processing techniques for near-infrared spectra", "10.1016/j.trac.2009.07.007"),
    "engel2013": Ref("Engel, J. et al.", 2013, "Breaking with trends in pre-processing?", "10.1016/j.trac.2013.04.015"),
    # -- selecao de variaveis
    "chong2005": Ref("Chong, I.-G.; Jun, C.-H.", 2005, "Performance of some variable selection methods when multicollinearity is present", "10.1016/j.chemolab.2004.12.011"),
    "mehmood2012": Ref("Mehmood, T. et al.", 2012, "A review of variable selection methods in partial least squares regression", "10.1016/j.chemolab.2012.07.010"),
    "rajalahti2009a": Ref("Rajalahti, T. et al.", 2009, "Biomarker discovery in mass spectral profiles by means of selectivity ratio plot", "10.1016/j.chemolab.2008.08.004"),
    "rajalahti2009b": Ref("Rajalahti, T. et al.", 2009, "Discriminating variable test and selectivity ratio plot: quantitative tools for interpretation and variable (biomarker) selection in complex spectral or chromatographic profiles", "10.1021/ac802514y"),
    "norgaard2000": Ref("Norgaard, L. et al.", 2000, "Interval partial least-squares regression (iPLS): a comparative chemometric study with an example from near-infrared spectroscopy", "10.1366/0003702001949500"),
    "araujo2001": Ref("Araujo, M. C. U. et al.", 2001, "The successive projections algorithm for variable selection in spectroscopic multicomponent analysis", "10.1016/S0169-7439(01)00119-8"),
    "li2009": Ref("Li, H.; Liang, Y.; Xu, Q.; Cao, D.", 2009, "Key wavelengths screening using competitive adaptive reweighted sampling method for multivariate calibration", "10.1016/j.aca.2009.06.046"),
    "centner1996": Ref("Centner, V. et al.", 1996, "Elimination of uninformative variables for multivariate calibration", "10.1021/ac960321m"),
    # -- validacao
    "wold1978": Ref("Wold, S.", 1978, "Cross-validatory estimation of the number of components in factor and principal components models", "10.1080/00401706.1978.10489693"),
    "xu2001": Ref("Xu, Q.-S.; Liang, Y.-Z.", 2001, "Monte Carlo cross validation", "10.1016/S0169-7439(00)00122-2"),
    "filzmoser2009a": Ref("Filzmoser, P.; Liebmann, B.; Varmuza, K.", 2009, "Repeated double cross validation", "10.1002/cem.1225"),
    "eriksson2008": Ref("Eriksson, L.; Trygg, J.; Wold, S.", 2008, "CV-ANOVA for significance testing of PLS and OPLS models", "10.1002/cem.1187"),
    "martens2000": Ref("Martens, H.; Martens, M.", 2000, "Modified Jack-knife estimation of parameter uncertainty in bilinear modelling by partial least squares regression (PLSR)", "10.1016/S0950-3293(99)00039-7"),
    "nadeau2003": Ref("Nadeau, C.; Bengio, Y.", 2003, "Inference for the generalization error", "10.1023/A:1024068626366"),
    "esbensen2010": Ref("Esbensen, K. H.; Geladi, P.", 2010, "Principles of proper validation: use and abuse of re-sampling for validation", "10.1002/cem.1310"),
    "westad2015": Ref("Westad, F.; Marini, F.", 2015, "Validation of chemometric models - a tutorial", "10.1016/j.aca.2015.06.056"),
    "consonni2010": Ref("Consonni, V.; Ballabio, D.; Todeschini, R.", 2010, "Evaluation of model predictive ability by external validation techniques", "10.1002/cem.1290"),
    "brereton2006": Ref("Brereton, R. G.", 2006, "Consequences of sample size, variable selection, and model validation and optimisation, for predicting classification ability from analytical data", "10.1016/j.trac.2006.10.005"),
    "ambroise2002": Ref("Ambroise, C.; McLachlan, G. J.", 2002, "Selection bias in gene extraction on the basis of microarray gene-expression data", "10.1073/pnas.102102699"),
    "varma2006": Ref("Varma, S.; Simon, R.", 2006, "Bias in error estimation when using cross-validation for model selection", "10.1186/1471-2105-7-91"),
    "kapoor2023": Ref("Kapoor, S.; Narayanan, A.", 2023, "Leakage and the reproducibility crisis in machine-learning-based science", "10.1016/j.patter.2023.100804"),
    # -- divisao de amostras
    "galvao2005": Ref("Galvao, R. K. H. et al.", 2005, "A method for calibration and validation subset partitioning", "10.1016/j.talanta.2005.03.025"),
    "snee1977": Ref("Snee, R. D.", 1977, "Validation of regression models: methods and examples", "10.1080/00401706.1977.10489581"),
    # -- transferencia de calibracao
    "wang1991": Ref("Wang, Y.; Veltkamp, D. J.; Kowalski, B. R.", 1991, "Multivariate instrument standardization", "10.1021/ac00023a016"),
    "feudale2002": Ref("Feudale, R. N. et al.", 2002, "Transfer of multivariate calibration models: a review", "10.1016/S0169-7439(02)00085-0"),
    "workman2017": Ref("Workman, J. J.", 2017, "A review of calibration transfer practices and instrument differences in spectroscopy", "10.1177/0003702817736064"),
    # -- figuras de merito
    "olivieri2006": Ref("Olivieri, A. C. et al.", 2006, "Uncertainty estimation and figures of merit for multivariate calibration (IUPAC Technical Report)", "10.1351/pac200678030633"),
    "williams2014": Ref("Williams, P.", 2014, "The RPD statistic: a tutorial note", "10.1255/nirn.1419"),
    # -- predicao conformal
    "lei2018": Ref("Lei, J. et al.", 2018, "Distribution-free predictive inference for regression", "10.1080/01621459.2017.1307116"),
    "sadinle2018": Ref("Sadinle, M.; Lei, J.; Wasserman, L.", 2018, "Least ambiguous set-valued classifiers with bounded error levels", "10.1080/01621459.2017.1395341"),
    "romano2020": Ref("Romano, Y.; Sesia, M.; Candes, E. J.", 2020, "Classification with valid and adaptive coverage", "10.48550/arXiv.2006.02544"),
    "angelopoulos2021": Ref("Angelopoulos, A. N.; Bates, S.", 2021, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification", "10.48550/arXiv.2107.07511"),
    # -- comparacao entre grupos
    "tukey1949": Ref("Tukey, J. W.", 1949, "Comparing individual means in the analysis of variance", "10.2307/3001913"),
    "kramer1956": Ref("Kramer, C. Y.", 1956, "Extension of multiple range tests to group means with unequal numbers of replications", "10.2307/3001469"),
    "games1976": Ref("Games, P. A.; Howell, J. F.", 1976, "Pairwise multiple comparison procedures with unequal n's and/or variances: a Monte Carlo study", "10.3102/10769986001002113"),
    "dunn1964": Ref("Dunn, O. J.", 1964, "Multiple comparisons using rank sums", "10.1080/00401706.1964.10490181"),
    "rochon2012": Ref("Rochon, J.; Gondan, M.; Kieser, M.", 2012, "To test or not to test: preliminary assessment of normality when comparing two independent samples", "10.1186/1471-2288-12-81"),
    "smilde2005": Ref("Smilde, A. K. et al.", 2005, "ANOVA-simultaneous component analysis (ASCA): a new tool for analyzing designed metabolomics data", "10.1093/bioinformatics/bti476"),
    "zwanenburg2011": Ref("Zwanenburg, G. et al.", 2011, "ANOVA-principal component analysis and ANOVA-simultaneous component analysis: a comparison", "10.1002/cem.1400"),
    "thiel2017": Ref("Thiel, M.; Feraud, B.; Govaerts, B.", 2017, "ASCA+ and APCA+: extensions of ASCA and APCA in the analysis of unbalanced multifactorial designs", "10.1002/cem.2895"),
    # -- resolucao, multivias, imagem
    "tauler1995": Ref("Tauler, R.", 1995, "Multivariate curve resolution applied to second order data", "10.1016/0169-7439(95)00047-X"),
    "tauler2006": Ref("Tauler, R.; de Juan, A.", 2006, "Multivariate curve resolution", "10.1201/9781420018301.ch11"),
    "bayat2020": Ref("Bayat, M. et al.", 2020, "Application of the area correlation constraint in the MCR-ALS quantitative analysis of complex mixture samples", "10.1016/j.aca.2020.03.057"),
    "bro1997": Ref("Bro, R.", 1997, "PARAFAC. Tutorial and applications", "10.1016/S0169-7439(97)00032-4"),
    "bro1996": Ref("Bro, R.", 1996, "Multiway calibration. Multilinear PLS", "10.1002/(SICI)1099-128X(199601)10:1<47::AID-CEM400>3.0.CO;2-C"),
    "amigo2015": Ref("Amigo, J. M.; Babamoradi, H.; Elcoroaristizabal, S.", 2015, "Hyperspectral image analysis. A tutorial", "10.1016/j.aca.2015.09.030"),
    "nielsen1998": Ref("Nielsen, N.-P. V.; Carstensen, J. M.; Smedsgaard, J.", 1998, "Aligning of single and multiple wavelength chromatographic profiles for chemometric data analysis using correlation optimised warping", "10.1016/S0021-9673(98)00021-1"),
    # -- monitoramento e deriva
    "kourti1995": Ref("Kourti, T.; MacGregor, J. F.", 1995, "Process analysis, monitoring and diagnosis, using multivariate projection methods", "10.1016/0169-7439(95)80036-9"),
    "tracy1992": Ref("Tracy, N. D.; Young, J. C.; Mason, R. L.", 1992, "Multivariate control charts for individual observations", "10.1080/00224065.1992.12015232"),
    "jensen2006": Ref("Jensen, W. A. et al.", 2006, "Effects of parameter estimation on control chart properties: a literature review", "10.1080/00224065.2006.11918623"),
    "husermet2011": Ref("The Human Serum Metabolome (HUSERMET) Consortium; Dunn, W. B. et al.", 2011, "Procedures for large-scale metabolic profiling of serum and plasma using gas chromatography and liquid chromatography coupled to mass spectrometry", "10.1038/nprot.2011.335"),
    "wehrens2016": Ref("Wehrens, R. et al.", 2016, "Improved batch correction in untargeted MS-based metabolomics", "10.1007/s11306-016-1015-8"),
    "gama2014": Ref("Gama, J. et al.", 2014, "A survey on concept drift adaptation", "10.1145/2523813"),
    # -- dados composicionais (ICP)
    "filzmoser2009b": Ref("Filzmoser, P.; Hron, K.; Reimann, C.", 2009, "Principal component analysis for compositional data with outliers", "10.1002/env.966"),
    "palarea2013": Ref("Palarea-Albaladejo, J.; Martin-Fernandez, J. A.", 2013, "Values below detection limit in compositional chemical data", "10.1016/j.aca.2012.12.029"),
    # -- dados, formatos e relato
    "mcdonald1988": Ref("McDonald, R. S.; Wilks, P. A.", 1988, "JCAMP-DX: a standard form for exchange of infrared spectra in computer readable form", "10.1366/0003702884428734"),
    "mitchell2019": Ref("Mitchell, M. et al.", 2019, "Model cards for model reporting", "10.1145/3287560.3287596"),
    "rohman2017": Ref("Rohman, A.", 2017, "The use of infrared spectroscopy in combination with chemometrics for quality control and authentication of edible fats and oils: a review", "10.1080/05704928.2016.1266493"),
    # -- conferidas em 2026-10-15 (Crossref/DataCite) para fechar lacunas
    "lecao2011": Ref("Le Cao, K.-A.; Boitard, S.; Besse, P.", 2011, "Sparse PLS discriminant analysis: biologically relevant feature selection and graphical displays for multiclass problems", "10.1186/1471-2105-12-253"),
    "efron1987": Ref("Efron, B.", 1987, "Better bootstrap confidence intervals", "10.1080/01621459.1987.10478410"),
    "kennard1969": Ref("Kennard, R. W.; Stone, L. A.", 1969, "Computer aided design of experiments", "10.1080/00401706.1969.10490666"),
    "kruskal1952": Ref("Kruskal, W. H.; Wallis, W. A.", 1952, "Use of ranks in one-criterion variance analysis", "10.1080/01621459.1952.10483441"),
    "benjamini1995": Ref("Benjamini, Y.; Hochberg, Y.", 1995, "Controlling the false discovery rate: a practical and powerful approach to multiple testing", "10.1111/j.2517-6161.1995.tb02031.x"),
    "shapiro1965": Ref("Shapiro, S. S.; Wilk, M. B.", 1965, "An analysis of variance test for normality (complete samples)", "10.1093/biomet/52.3-4.591"),
    "brown1974": Ref("Brown, M. B.; Forsythe, A. B.", 1974, "Robust tests for the equality of variances", "10.1080/01621459.1974.10482955"),
    "lundberg2017": Ref("Lundberg, S. M.; Lee, S.-I.", 2017, "A unified approach to interpreting model predictions", "10.48550/arXiv.1705.07874"),
    "olivieri2015": Ref("Olivieri, A. C.", 2015, "Practical guidelines for reporting results in single- and multi-component analytical calibration: a tutorial", "10.1016/j.aca.2015.01.017"),
    "vanderheyden2001": Ref("Vander Heyden, Y. et al.", 2001, "Guidance for robustness/ruggedness tests in method validation", "10.1016/S0731-7085(00)00529-X"),
}


_SEM = "Procedimento proprio do GUARACI; as referencias sao das bases que ele usa."

# (id, nome PT, nome EN, itens)
TEMAS: List[Tuple[str, str, str, List[Item]]] = [
    ("classificacao", "Classificacao e autenticacao", "Classification and authentication", [
        Item("PLS-DA", ("brereton2014", "westerhuis2008"), "pls_da"),
        Item("OPLS-DA", ("trygg2002", "bylesjo2006")),
        Item("DD-SIMCA", ("pomerantsev2008", "pomerantsev2013", "pomerantsev2020", "kucheryavskiy2020"), "ddsimca"),
        Item("Autenticacao vs. discriminacao", ("rodionova2016",)),
        Item("Identificacao em conjunto aberto", ("pomerantsev2008", "sadinle2018"), "identificacao_conjunto_aberto", _SEM),
        Item("sPLS-DA", ("lecao2011",)),
    ]),
    ("regressao", "Regressao (teor)", "Regression (content)", [
        Item("PLS", ("geladi1986", "wold2001", "andersson2009")),
        Item("PLS unico (todas as especies)", ("geladi1986", "wold2001"), "pls_r_pooled"),
        Item("PLS por especie", ("geladi1986", "wold2001"), "pls_r_por_especie"),
        Item("Unico vs. por especie", ("nadeau2003",), "pooled_vs_local", _SEM),
        Item("Dual-sPLS", ("alsouki2023",)),
        Item("SVR", ("smola2004",)),
    ]),
    ("preproc", "Pre-processamento", "Pre-processing", [
        Item("Savitzky-Golay", ("savitzky1964",)),
        Item("MSC", ("geladi1985",)),
        Item("SNV", ("barnes1989",)),
        Item("EMSC", ("martens1991", "afseth2012")),
        Item("OSC", ("wold1998",)),
        Item("EPO", ("roger2003",)),
        Item("GLSW", ("martens2003",)),
        Item("airPLS (linha de base)", ("zhang2010",)),
        Item("Remocao de raios cosmicos", ("whitaker2018",)),
        Item("PQN", ("dieterle2006",)),
        Item("Revisoes", ("rinnan2009", "engel2013")),
        Item("Portao de correcao de sinal", ("martens1991", "wold1998", "engel2013"), "portao_correcao_sinal", _SEM),
    ]),
    ("selecao", "Selecao de variaveis", "Variable selection", [
        Item("VIP", ("wold2001", "chong2005", "mehmood2012")),
        Item("Estabilidade do VIP", ("wold2001", "chong2005"), "vip_estabilidade", _SEM),
        Item("Selectivity Ratio", ("rajalahti2009a", "rajalahti2009b")),
        Item("iPLS", ("norgaard2000",)),
        Item("SPA", ("araujo2001",)),
        Item("CARS", ("li2009",)),
        Item("UVE", ("centner1996",)),
    ]),
    ("validacao", "Validacao", "Validation", [
        Item("Validacao cruzada (n. de componentes)", ("wold1978",)),
        Item("Monte Carlo CV", ("xu2001",)),
        Item("Validacao cruzada dupla repetida", ("filzmoser2009a",)),
        Item("CV-ANOVA", ("eriksson2008",)),
        Item("Teste de permutacao do PLS-DA", ("westerhuis2008",)),
        Item("Jack-knife de Martens", ("martens2000",)),
        Item("Teste t corrigido entre modelos", ("nadeau2003",)),
        Item("Q2 externo", ("consonni2010",)),
        Item("Principios de validacao", ("esbensen2010", "westad2015", "brereton2006")),
        Item("Vazamento de informacao", ("ambroise2002", "varma2006", "kapoor2023")),
        Item("Intervalo bootstrap BCa", ("efron1987",)),
    ]),
    ("divisao", "Divisao de amostras", "Sample splitting", [
        Item("Kennard-Stone", ("kennard1969",), "kennard_stone"),
        Item("Duplex", ("snee1977",), "duplex"),
        Item("SPXY", ("galvao2005",), "spxy"),
        Item("Amostragem ativa", (), "amostragem_ativa", "Procedimento do GUARACI, descrito no manual.", proprio=True),
    ]),
    ("transferencia", "Transferencia de calibracao", "Calibration transfer", [
        Item("DS", ("wang1991",), "direct_standardization"),
        Item("PDS", ("wang1991",), "piecewise_direct_standardization"),
        Item("Revisoes", ("feudale2002", "workman2017")),
    ]),
    ("merito", "Figuras de merito", "Figures of merit", [
        Item("LOD, LOQ, sensibilidade", ("olivieri2006",), "figuras_de_merito_regressao"),
        Item("RPD e RER", ("williams2014",), "rpd_rer"),
        Item("Linearidade (falta de ajuste)", ("olivieri2015",), "linearidade"),
        Item("Robustez", ("vanderheyden2001",), "robustez", "O protocolo de perturbacao e' do GUARACI; a referencia e' a base do desenho."),
    ]),
    ("conformal", "Predicao conformal", "Conformal prediction", [
        Item("Introducao", ("angelopoulos2021",)),
        Item("Conformal de uma classe", ("angelopoulos2021",), "conformal_one_class"),
        Item("Conformal para regressao", ("lei2018",), "conformal_regressao"),
        Item("Conformal para classificacao", ("sadinle2018", "romano2020"), "conformal_classificacao"),
    ]),
    ("grupos", "Comparacao entre grupos", "Group comparison", [
        Item("ANOVA + Tukey-Kramer", ("tukey1949", "kramer1956"), "anova_tukey_unidade"),
        Item("Games-Howell", ("games1976",)),
        Item("Kruskal-Wallis + Dunn", ("kruskal1952", "dunn1964", "benjamini1995"), "kruskal_dunn_unidade",
             "Holm (1979, Scand. J. Stat. 6:65-70) nao tem DOI."),
        Item("Diagnostico de premissas", ("shapiro1965", "brown1974", "rochon2012"), "diagnostico_premissas",
             "Levene (1960) e' capitulo de livro, sem DOI. Rochon et al. (2012) e' o alerta contra decidir o teste so' por pre-teste de normalidade."),
        Item("ASCA / ASCA+", ("smilde2005", "zwanenburg2011", "thiel2017")),
    ]),
    ("multivias", "Resolucao, multivias e imagem", "Resolution, multiway and imaging", [
        Item("MCR-ALS", ("tauler1995", "tauler2006", "bayat2020"), "mcr_als"),
        Item("PARAFAC (EEM)", ("bro1997",), "parafac_eem"),
        Item("N-PLS", ("bro1996",)),
        Item("Imagem hiperespectral", ("amigo2015",)),
        Item("COW (alinhamento)", ("nielsen1998",), "cow_alinhamento"),
    ]),
    ("monitoramento", "Monitoramento e deriva", "Monitoring and drift", [
        Item("MSPC / T2 de Hotelling", ("kourti1995", "tracy1992", "jensen2006"), "monitoramento_mspc"),
        Item("Deriva por amostras de QC", ("husermet2011", "wehrens2016")),
        Item("Deriva de conceito", ("gama2014",)),
    ]),
    ("composicional", "Dados multielementares (ICP)", "Multi-element data (ICP)", [
        Item("Perfil multielementar", ("filzmoser2009b", "palarea2013"), "perfil_multielementar"),
    ]),
    ("dados", "Dados, perfis e relato", "Data, profiles and reporting", [
        Item("JCAMP-DX", ("mcdonald1988",)),
        Item("Ficha do modelo (model card)", ("mitchell2019",)),
        Item("Perfil de tecnica", (), "perfil_matriz", "Configuracao de partida do GUARACI (faixa e pre-processamento); as fontes estao nos itens de cada metodo.", proprio=True),
        Item("Perfil de tecnica de aquisicao", (), "perfil_tecnica_aquisicao", "Configuracao do GUARACI, nao metodo.", proprio=True),
        Item("Revisao de aplicacao (IV + quimiometria)", ("rohman2017",)),
        Item("Explicacao SHAP", ("lundberg2017",)),
    ]),
]


def citacao(chave: str) -> str:
    r = REFS[chave]
    return f"{r.autores} ({r.ano}). {r.titulo}. doi:{r.doi}"


def lacunas() -> List[Tuple[str, str]]:
    """(metodo, nota) dos itens sem nenhuma referencia conferida."""
    return [(it.metodo, it.nota) for _t, _p, _e, itens in TEMAS for it in itens if not it.refs and not it.proprio]


def chaves_usadas() -> List[str]:
    vistos: Dict[str, None] = {}
    for _t, _p, _e, itens in TEMAS:
        for it in itens:
            for k in it.refs:
                vistos.setdefault(k, None)
    return list(vistos)
