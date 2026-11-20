# Gera tests/oraculos_r/oraculo_pls_r.json: referencias INDEPENDENTES em R para PLS1 (pacote `pls`,
# Mevik & Wehrens, NIPALS "oscorespls"), VIP (formula de Chong & Jun 2005 sobre o modelo do `pls`),
# MSC (`pls::msc`), SNV, PCA (prcomp) com T2 e Q, e os limites T2 (F, fase II) e Q (Box/Jackson-Mudholkar,
# qchisq). So' R base + `pls` + `stats`.
#
# Uso (fora da CI; o JSON gerado e' o que entra no repositorio):
#   apt-get install r-base-core r-cran-pls      (ou install.packages("pls"))
#   Rscript tests/oraculos_r/gen_oraculo_pls_r.R
num <- function(x) paste0("[", paste(sprintf("%.17g", as.numeric(x)), collapse = ","), "]")
mat <- function(m) paste0("[", paste(apply(as.matrix(m), 1, num), collapse = ","), "]")
suppressMessages(library(pls))

espectros <- function(seed, n, p) {
  set.seed(seed)
  eixo <- seq(0, 1, length.out = p)
  picos <- cbind(exp(-((eixo - 0.3) / 0.05)^2), exp(-((eixo - 0.6) / 0.08)^2), exp(-((eixo - 0.8) / 0.03)^2))
  C <- matrix(runif(n * 3), n, 3)
  esc <- runif(n, 0.8, 1.2); off <- runif(n, -0.05, 0.05)
  X <- (C %*% t(picos)) * esc + off + matrix(rnorm(n * p, 0, 0.005), n, p)
  list(X = X, y = C[, 1] * 10)
}

vip_chong_jun <- function(m, a) {
  W <- m$loading.weights[, 1:a, drop = FALSE]
  Tt <- m$scores[, 1:a, drop = FALSE]
  q <- m$Yloadings[1, 1:a]
  ss <- colSums(Tt^2) * q^2
  Wn <- sweep(W, 2, sqrt(colSums(W^2)), "/")
  sqrt(nrow(W) * (Wn^2 %*% ss) / sum(ss))
}

casos <- character()
for (cs in list(list(id = "A", seed = 11, n = 30, p = 60, a = 3),
                list(id = "B", seed = 12, n = 45, p = 120, a = 4),
                list(id = "C", seed = 13, n = 20, p = 40, a = 2))) {
  d <- espectros(cs$seed, cs$n, cs$p)
  X <- d$X; y <- d$y
  snv <- t(apply(X, 1, function(v) (v - mean(v)) / sd(v)))           # sd do R: divisor n-1
  mscX <- msc(X)
  m <- plsr(y ~ X, ncomp = cs$a, method = "oscorespls", scale = FALSE, validation = "none")
  b <- coef(m, ncomp = cs$a, intercept = FALSE)[, 1, 1]
  b0 <- coef(m, ncomp = cs$a, intercept = TRUE)[1, 1, 1]
  yhat <- fitted(m)[, 1, cs$a]
  vip <- vip_chong_jun(m, cs$a)
  pc <- prcomp(X, center = TRUE, scale. = FALSE, rank. = cs$a)
  Tp <- pc$x[, 1:cs$a, drop = FALSE]
  T2 <- rowSums(sweep(Tp^2, 2, apply(Tp, 2, var), "/"))
  Xc <- scale(X, center = TRUE, scale = FALSE)
  Q <- rowSums((Xc - Tp %*% t(pc$rotation[, 1:cs$a]))^2)
  n <- cs$n; k <- cs$a
  t2lim <- k * (n - 1) * (n + 1) / (n * (n - k)) * qf(0.95, k, n - k)
  vq <- var(Q) * (n - 1) / n                                           # variancia populacional
  g <- vq / (2 * mean(Q)); h <- 2 * mean(Q)^2 / vq
  qlim <- g * qchisq(0.95, h)
  casos <- c(casos, sprintf(paste0('{"id":"%s","ncomp":%d,"X":%s,"y":%s,"snv_sd_n1":%s,"msc":%s,',
                                   '"pls_coef":%s,"pls_intercepto":%.17g,"pls_yhat":%s,"vip":%s,',
                                   '"pca_T2":%s,"pca_Q":%s,"t2_lim_95":%.17g,"q_lim_95":%.17g}'),
    cs$id, cs$a, mat(X), num(y), mat(snv), mat(mscX), num(b), b0, num(yhat), num(vip), num(T2), num(Q),
    t2lim, qlim))
}
vers <- sprintf('{"R":"%s","pls":"%s"}', getRversion(), as.character(packageVersion("pls")))
arq <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "oraculo_pls_r.json")
writeLines(sprintf('{"versoes":%s,"casos":[%s]}', vers, paste(casos, collapse = ",\n")), arq)
cat("gravado", arq, "\n")
