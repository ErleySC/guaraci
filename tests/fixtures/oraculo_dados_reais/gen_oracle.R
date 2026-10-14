# gen_oracle.R -- oraculos de tests/test_oraculo_dados_reais.py, com DADOS
# REAIS PUBLICADOS (nao sinteticos) e as mesmas referencias externas de
# tests/fixtures/oraculo_referencias/gen_oracle.R (mdatools 0.16.0 @
# cran/mdatools 22aac09, ropls 1.45.0 @ bioc/ropls a70ae7c; R 4.3.3).
#
# Datasets (NENHUM e' versionado aqui -- ver datasets/README.md; so' as
# SAIDAS dos modelos de referencia sao gravadas nesta pasta):
#   * sacurine -- metaboloma urinario de 183 adultos, LC-HRMS, 109
#     metabolitos, log10 (Thevenot E.A., Roux A., Xu Y., Ezan E., Junot C.
#     2015, J. Proteome Res. 14(8):3322-3335). Distribuido com o ropls
#     (inst/extdata/*.tsv). Baixar com
#     scripts/download_datasets/baixar_ropls_sacurine.py (SHA256 pinado).
#   * iris -- Fisher R.A. (1936), Ann. Eugenics 7:179-188. Embutido no R
#     (`datasets::iris`) e no scikit-learn (`load_iris`); conferido: os
#     150x4 valores sao identicos nas duas fontes.
#
# Reproduzir (a partir DESTE diretorio):
#   MDATOOLS_R=/caminho/mdatools/R ROPLS_R=/caminho/ropls/R \
#   GUARACI_DATASETS_DIR=/caminho/cache Rscript gen_oracle.R
for (f in list.files(Sys.getenv("MDATOOLS_R"), pattern = "\\.R$",
                     full.names = TRUE)) source(f)
source(file.path(Sys.getenv("ROPLS_R"), "allClasses.R"))
source(file.path(Sys.getenv("ROPLS_R"), "opls-functions.R"))

wcsv <- function(x, nome) write.csv(x, nome, row.names = FALSE)
onehot <- function(cls) {
  lv <- sort(unique(as.character(cls)))
  sapply(lv, function(k) as.numeric(as.character(cls) == k))
}
core <- function(x, y, ortho, pred, escala = "center")
  .coreF(x, y, orthoI = ortho, predI = pred, scaleC = escala,
         algoC = "nipals", crossvalI = 7, subsetL = FALSE,
         subsetVi = numeric(), .char2numF = NULL, info.txtC = "none")

# ---------------------------------------------------------------- sacurine
dir_s <- file.path(Sys.getenv("GUARACI_DATASETS_DIR"), "ropls_sacurine")
dm <- read.table(file.path(dir_s, "sacurine_dataMatrix.tsv"), header = TRUE,
                 sep = "\t", row.names = 1, check.names = FALSE)
sm <- read.table(file.path(dir_s, "sacurine_sampleMetadata.tsv"),
                 header = TRUE, sep = "\t", row.names = 1)
X <- t(as.matrix(dm))                      # 183 amostras x 109 variaveis
stopifnot(identical(rownames(X), rownames(sm)), dim(X) == c(183, 109))
Xa <- scale(X)                             # autoescalamento (sd ddof=1)
Y  <- onehot(sm$gender)                    # colunas F, M

m <- pls(Xa, Y, ncomp = 2, center = TRUE, scale = FALSE)
wcsv(m$res$cal$y.pred[, 2, ], "sac_mda_ypred.csv")
wcsv(m$coeffs$values[, 2, ], "sac_mda_coef.csv")
wcsv(selratio(m, ncomp = 2), "sac_mda_sr.csv")
r <- core(Xa, Y, 0, 2)
wcsv(data.frame(vip = r@vipVn), "sac_ropls_vip.csv")

# OPLS-DA como no vinheta do ropls: autoescalado, 1 preditivo, numero de
# ortogonais escolhido pelo proprio ropls (orthoI = NA, CV de 7 grupos)
o <- core(X, matrix(as.numeric(sm$gender == "M")), NA, 1, "standard")
wcsv(o@scoreMN, "sac_opls_tpred.csv")
wcsv(o@orthoScoreMN, "sac_opls_torth.csv")
wcsv(o@suppLs$yPreMN, "sac_opls_ypred.csv")

# -------------------------------------------------------------------- iris
Xi <- as.matrix(iris[, 1:4]); Yi <- onehot(iris$Species)
ri <- core(Xi, Yi, 0, 3)
wcsv(ri@suppLs$yPreMN, "iris_ropls_ypred.csv")
wcsv(data.frame(vip = ri@vipVn), "iris_ropls_vip.csv")

# DD-SIMCA da classe versicolor: treino = 40 primeiras versicolor; novos =
# 10 versicolor restantes + 50 virginica (iris[51:90,] / iris[c(91:100,101:150),])
tr <- 51:90; nv <- c(91:100, 101:150)
ds <- ddsimca(Xi[tr, ], "versicolor", ncomp = 2, center = TRUE,
              scale = FALSE, pcv = NULL, alpha = 0.05, do.round = FALSE)
pr <- predict(ds, Xi[nv, ])
hq_tr <- data.frame(h = ds$calres$T2[, 2], q = ds$calres$Q[, 2])
hq_nv <- data.frame(h = pr$T2[, 2], q = pr$Q[, 2])
wcsv(hq_tr, "iris_dds_hq_train.csv"); wcsv(hq_nv, "iris_dds_hq_new.csv")
ph <- ddmoments.param(hq_tr$h); pq <- ddmoments.param(hq_tr$q)
wcsv(data.frame(h0 = ph$u0, Nh = ph$Nu, q0 = pq$u0, Nq = pq$Nu,
                fcrit = qchisq(0.95, ph$Nu + pq$Nu)), "iris_dds_params.csv")
wcsv(data.frame(f = hq_nv$h / ph$u0 * ph$Nu + hq_nv$q / pq$u0 * pq$Nu),
     "iris_dds_f_new.csv")
cat("DONE\n")
