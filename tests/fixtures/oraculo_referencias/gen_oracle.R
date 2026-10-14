# gen_oracle.R -- gera os oraculos numericos de tests/test_oraculo_referencias.py
# rodando implementacoes de REFERENCIA EXTERNAS (nao escritas neste projeto):
#
#   * mdatools 0.16.0 (Kucheryavskiy, MIT) -- PLS (SIMPLS), vipscores(),
#     selratio(), ddsimca(), ddmoments.param(), dd.crit().
#     Fonte: mirror somente-leitura do CRAN no GitHub,
#     https://github.com/cran/mdatools @ 22aac0933bed4eb1121e40c7511fbccf950b95ec
#   * ropls 1.45.0 (Thevenot, CeCILL) -- nucleo NIPALS PLS/OPLS `.coreF`
#     (R/opls-functions.R, so' base R). Fonte: mirror do Bioconductor,
#     https://github.com/bioc/ropls @ a70ae7c2e27a7e029ce8a2520a133b082b90186c
#
# R 4.3.3 (Ubuntu, r-base-core). Os pacotes NAO sao instalados: os arquivos
# R/*.R de cada mirror sao carregados com source() -- mdatools so' precisa
# de base R para as funcoes usadas aqui; de ropls so' allClasses.R + opls-functions.R.
#
# Reproduzir (a partir DESTE diretorio):
#   git clone https://github.com/cran/mdatools /caminho/mdatools
#   git clone https://github.com/bioc/ropls   /caminho/ropls
#   MDATOOLS_R=/caminho/mdatools/R ROPLS_R=/caminho/ropls/R Rscript gen_oracle.R
mda_dir   <- Sys.getenv("MDATOOLS_R")
ropls_dir <- Sys.getenv("ROPLS_R")
for (f in list.files(mda_dir, pattern = "\\.R$", full.names = TRUE)) source(f)
source(file.path(ropls_dir, "allClasses.R"))     # so' setClass("opls")
source(file.path(ropls_dir, "opls-functions.R"))

wcsv <- function(x, nome) write.csv(x, nome, row.names = FALSE)
onehot <- function(cls) {
  lv <- sort(unique(cls))
  sapply(lv, function(k) as.numeric(cls == k))
}

# ---------------------------------------------------------------------
# Dados sinteticos com estrutura de espectro: 2-3 "bandas" latentes,
# uma variacao sistematica NAO correlacionada com a classe (o que o
# componente ortogonal do OPLS deve capturar) e offset != 0 (X NAO
# centrado, como chega do pre-processamento no pipeline).
# ---------------------------------------------------------------------
set.seed(20261014)
p  <- 30
ax <- seq(0, 1, length.out = p)
banda <- function(c, s) exp(-((ax - c) / s)^2)
s_cls  <- banda(0.30, 0.08)   # sinal de classe
s_orto <- banda(0.65, 0.15)   # variacao estruturada ortogonal a classe
s_ter  <- banda(0.85, 0.05)   # 3a classe

## (1) binario: 2 classes x 20
n2   <- 40
cls2 <- rep(c("A", "B"), each = n2 / 2)
X2 <- outer(ifelse(cls2 == "B", 1, 0) + rnorm(n2, sd = 0.3), s_cls) +
      outer(rnorm(n2, sd = 1.5), s_orto) +
      matrix(rnorm(n2 * p, sd = 0.05), n2) + 2.0
Y2 <- onehot(cls2)
wcsv(X2, "bin_X.csv"); wcsv(Y2, "bin_Y.csv")

m2 <- pls(X2, Y2, ncomp = 3, center = TRUE, scale = FALSE)
wcsv(m2$res$cal$y.pred[, 3, ], "bin_mda_ypred.csv")
wcsv(m2$coeffs$values[, 3, ], "bin_mda_coef.csv")
wcsv(vipscores(m2, ncomp = 3, type = "combined"), "bin_mda_vip.csv")
wcsv(selratio(m2, ncomp = 3), "bin_mda_sr.csv")
# PLS-DA do mdatools codifica classe como +1/-1 (nao 0/1)
d2 <- plsda(X2, factor(cls2), ncomp = 3, center = TRUE, scale = FALSE)
wcsv(d2$res$cal$y.pred[, 3, ], "bin_mda_plsda_ypred.csv")
r2 <- .coreF(X2, Y2, orthoI = 0, predI = 3, scaleC = "center",
             algoC = "nipals", crossvalI = 7, subsetL = FALSE,
             subsetVi = numeric(), .char2numF = NULL, info.txtC = "none")
wcsv(data.frame(vip = r2@vipVn), "bin_ropls_vip.csv")

## (2) PLS1 de regressao (y continuo) -- SIMPLS == NIPALS p/ 1 resposta
y1 <- as.vector(X2 %*% (s_cls - 0.5 * s_ter)) + rnorm(n2, sd = 0.05)
wcsv(data.frame(y = y1), "reg_y.csv")
m1 <- pls(X2, y1, ncomp = 4, center = TRUE, scale = FALSE)
wcsv(m1$res$cal$y.pred[, 4, ], "reg_mda_ypred.csv")
wcsv(m1$coeffs$values[, 4, ], "reg_mda_coef.csv")
wcsv(vipscores(m1, ncomp = 4, type = "combined"), "reg_mda_vip.csv")
wcsv(selratio(m1, ncomp = 4), "reg_mda_sr.csv")
# VIP de Wold (pesos W do NIPALS, espaco deflacionado) via ropls -- o VIP
# do mdatools usa os pesos do SIMPLS (R = W(P'W)^-1, espaco NAO
# deflacionado) e por isso NAO e' o mesmo numero; ver teste.
r1 <- .coreF(X2, matrix(y1), orthoI = 0, predI = 4, scaleC = "center",
             algoC = "nipals", crossvalI = 7, subsetL = FALSE,
             subsetVi = numeric(), .char2numF = NULL, info.txtC = "none")
wcsv(data.frame(vip = r1@vipVn), "reg_ropls_vip.csv")

## (3) multiclasse: 3 classes x 15 -- PLS2 NIPALS (ropls), pois SIMPLS
##     (mdatools) e NIPALS (sklearn) divergem para Y multi-coluna
n3   <- 45
cls3 <- rep(c("A", "B", "C"), each = n3 / 3)
X3 <- outer(ifelse(cls3 == "B", 1, 0) + rnorm(n3, sd = 0.3), s_cls) +
      outer(ifelse(cls3 == "C", 1, 0) + rnorm(n3, sd = 0.3), s_ter) +
      outer(rnorm(n3, sd = 1.5), s_orto) +
      matrix(rnorm(n3 * p, sd = 0.05), n3) + 2.0
Y3 <- onehot(cls3)
wcsv(X3, "multi_X.csv"); wcsv(Y3, "multi_Y.csv")
r3 <- .coreF(X3, Y3, orthoI = 0, predI = 3, scaleC = "center",
             algoC = "nipals", crossvalI = 7, subsetL = FALSE,
             subsetVi = numeric(), .char2numF = NULL, info.txtC = "none")
wcsv(r3@suppLs$yPreMN, "multi_ropls_ypred.csv")
wcsv(r3@coefficientMN, "multi_ropls_coef.csv")
wcsv(data.frame(vip = r3@vipVn), "multi_ropls_vip.csv")
wcsv(r3@scoreMN, "multi_ropls_scores.csv")

## (4) OPLS-DA binario (Trygg & Wold 2002) -- ropls, 1 pred + 2 orto
ro <- .coreF(X2, Y2[, 2, drop = FALSE], orthoI = 2, predI = 1,
             scaleC = "center", algoC = "nipals", crossvalI = 7,
             subsetL = FALSE, subsetVi = numeric(), .char2numF = NULL,
             info.txtC = "none")
wcsv(ro@scoreMN, "opls_ropls_tpred.csv")
wcsv(ro@orthoScoreMN, "opls_ropls_torth.csv")
wcsv(ro@suppLs$yPreMN, "opls_ropls_ypred.csv")
## PLS com 1 + 2 componentes: mesma predicao do OPLS (Trygg & Wold 2002)
m_eq <- pls(X2, Y2[, 2], ncomp = 3, center = TRUE, scale = FALSE)
wcsv(m_eq$res$cal$y.pred[, 3, ], "opls_mda_pls3_ypred.csv")

## (5) DD-SIMCA (Pomerantsev/Rodionova/Kucheryavskiy) -- mdatools
nt <- 30; nn <- 12
Xs  <- outer(rnorm(nt, sd = 1.0), s_cls) + outer(rnorm(nt, sd = 0.6), s_orto) +
       matrix(rnorm(nt * p, sd = 0.05), nt) + 1.0
Xsn <- outer(rnorm(nn, sd = 1.0), s_cls) + outer(rnorm(nn, sd = 0.6), s_orto) +
       outer(c(rep(0, nn / 2), rnorm(nn / 2, mean = 1.5, sd = 0.4)), s_ter) +
       matrix(rnorm(nn * p, sd = 0.05), nn) + 1.0
wcsv(Xs, "dds_Xtrain.csv"); wcsv(Xsn, "dds_Xnew.csv")
A <- 2
ds <- ddsimca(Xs, "alvo", ncomp = A, center = TRUE, scale = FALSE,
              pcv = NULL, alpha = 0.05, do.round = FALSE)
pr <- predict(ds, Xsn)
hq_tr <- data.frame(h = ds$calres$T2[, A], q = ds$calres$Q[, A])
hq_nw <- data.frame(h = pr$T2[, A], q = pr$Q[, A])
wcsv(hq_tr, "dds_mda_hq_train.csv"); wcsv(hq_nw, "dds_mda_hq_new.csv")
ph <- ddmoments.param(hq_tr$h); pq <- ddmoments.param(hq_tr$q)
crit_raw <- qchisq(1 - 0.05, ph$Nu + pq$Nu)
crit_rnd <- dd.crit(pq, ph, alpha = 0.05)[1, 1]   # Nh/Nq arredondados
f_nw <- hq_nw$h / ph$u0 * ph$Nu + hq_nw$q / pq$u0 * pq$Nu
wcsv(data.frame(h0 = ph$u0, Nh = ph$Nu, q0 = pq$u0, Nq = pq$Nu,
                fcrit = crit_raw, fcrit_round = crit_rnd),
     "dds_mda_params.csv")
wcsv(data.frame(f = f_nw), "dds_mda_f_new.csv")

cat("DONE\n")
