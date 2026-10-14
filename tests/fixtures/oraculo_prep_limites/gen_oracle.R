# gen_oracle.R -- oraculos de tests/test_oraculo_prep_limites.py a partir de
# implementacoes de REFERENCIA EXTERNAS (nao escritas neste projeto):
#
#   * mdatools 0.16.0 (Kucheryavskiy, MIT): prep.norm, prep.emsc,
#     prep.savgol, hotelling.crit. Fonte: https://github.com/cran/mdatools
#     @ 22aac0933bed4eb1121e40c7511fbccf950b95ec (carregado com source()).
#   * boot (Canty & Ripley, pacote recomendado que vem com o R): bca.ci.
#   * stats::qchisq.
#
# R 4.3.3. Reproduzir (a partir DESTE diretorio):
#   python gen_replicas.py
#   git clone https://github.com/cran/mdatools /caminho/mdatools
#   MDATOOLS_R=/caminho/mdatools/R Rscript gen_oracle.R
mda_dir <- Sys.getenv("MDATOOLS_R")
for (f in list.files(mda_dir, pattern = "\\.R$", full.names = TRUE)) source(f)
library(boot)

wcsv <- function(x, nome) write.csv(x, nome, row.names = FALSE)
semattr <- function(x) { x <- unclass(x); attributes(x) <- list(dim = dim(x)); x }

# ---------------------------------------------------------------------
# Espectros sinteticos: X_i = a_i + b_i*ref + linha de base quadratica +
# ruido, todos positivos (PQN divide por eles). 10 de treino + 5 novos.
# ---------------------------------------------------------------------
set.seed(20261015)
p  <- 101
ax <- seq(0, 1, length.out = p)
g  <- function(mu, s) exp(-0.5 * ((ax - mu) / s)^2)
ref <- 0.3 + 1.0 * g(0.25, 0.04) + 0.6 * g(0.55, 0.08) + 0.8 * g(0.8, 0.03)
n  <- 15
X  <- t(sapply(seq_len(n), function(i) {
  0.1 * runif(1) + runif(1, 0.6, 1.6) * ref +
    runif(1, -0.1, 0.1) * ax + runif(1, 0, 0.1) * ax^2 +
    rnorm(p, sd = 0.01)
}))
stopifnot(all(X > 0))
tr <- 1:10
wcsv(X, "X.csv")

wcsv(semattr(prep.norm(X, "snv")), "snv.csv")
wcsv(semattr(prep.emsc(X, degree = 0, mspectrum = colMeans(X[tr, ]))), "msc.csv")
wcsv(semattr(prep.emsc(X, degree = 2, mspectrum = colMeans(X[tr, ]))), "emsc2.csv")
ref_med <- apply(X[tr, ], 2, median)
wcsv(semattr(prep.norm(X, "pqn", ref.spectrum = ref_med)), "pqn.csv")

sg <- list(c(11, 2, 0), c(11, 2, 1), c(11, 3, 2), c(7, 3, 1), c(15, 4, 2))
for (cfg in sg) {
  wcsv(semattr(prep.savgol(X, width = cfg[1], porder = cfg[2], dorder = cfg[3])),
       sprintf("savgol_w%d_p%d_d%d.csv", cfg[1], cfg[2], cfg[3]))
}

# ------------------------------------------------ limite T2 (mdatools)
grade <- expand.grid(n = c(8, 20, 50, 300), k = c(1, 2, 5), alpha = c(0.05, 0.01))
grade <- grade[grade$n > grade$k, ]
grade$t2 <- mapply(function(n, k, a) hotelling.crit(n, k, alpha = a)[1, 1],
                   grade$n, grade$k, grade$alpha)
wcsv(grade, "hotelling.csv")

# ------------------------------------------------ limite Q (chi2 por momentos)
# Box (1954); Nomikos & MacGregor (1995): g*chi2_{h}(1-alpha), g = v/(2m),
# h = 2m^2/v, com m e v a media e a variancia (populacional) de Q.
q <- rchisq(40, df = 3) * 0.02
wcsv(data.frame(q = q), "q.csv")
m <- mean(q); v <- mean((q - m)^2)
wcsv(data.frame(alpha = c(0.05, 0.01),
                ucl = (v / (2 * m)) * qchisq(1 - c(0.05, 0.01), 2 * m^2 / v)),
     "q_limite.csv")

# ------------------------------------------------ IC BCa (boot)
for (nome in c("auc", "acc")) {
  t0   <- read.csv(sprintf("bca_%s_t0.csv", nome))$t0
  t    <- read.csv(sprintf("bca_%s_t.csv", nome))$t
  jack <- read.csv(sprintf("bca_%s_jack.csv", nome))$jack
  L    <- (length(jack) - 1) * (mean(jack) - jack)    # influencia jackknife
  ci   <- boot:::bca.ci(NULL, conf = 0.95, t0 = t0, t = t, L = L)
  # mesmos alfas ajustados do boot (linhas de bca.ci), lidos com o
  # quantil tipo 7 (interpolacao linear = np.percentile) em vez do
  # norm.inter do boot -- e' o que isola a unica escolha diferente.
  w  <- qnorm(sum(t < t0) / length(t))
  a  <- sum(L^3) / (6 * sum(L^2)^1.5)
  za <- qnorm(c(0.025, 0.975))
  adj <- pnorm(w + (w + za) / (1 - a * (w + za)))
  wcsv(data.frame(boot_lo = ci[1, 4], boot_hi = ci[1, 5],
                  adj_lo = adj[1], adj_hi = adj[2],
                  q7_lo = quantile(t, adj[1], type = 7, names = FALSE),
                  q7_hi = quantile(t, adj[2], type = 7, names = FALSE)),
       sprintf("bca_%s_ci.csv", nome))
}
cat("ok\n")
