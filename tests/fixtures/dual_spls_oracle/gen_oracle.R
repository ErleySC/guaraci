# gen_oracle.R -- gera o oraculo numerico para tests/test_dual_spls.py
# (test_dual_spls_matches_r_oracle), rodando a implementacao REAL de
# referencia do metodo Dual-sPLS.
#
# Codigo-fonte usado (d.spls.norm.R, d.spls.lasso.R): pacote R
# `dual.spls` 0.1.4 (Alsouki & Wahl, MIT + file LICENSE), removido do
# CRAN em 2024-04-20 mas preservado no mirror somente-leitura do CRAN no
# GitHub: https://github.com/cran/dual.spls (commit da branch `master`
# no momento da geracao -- ver DESCRIPTION dentro do mesmo repo p/
# Version: 0.1.4). Rodado com R 4.3.3 (Ubuntu, `apt-get install
# r-base-core`), sem pacotes adicionais (a funcao `d.spls.lasso` so'
# usa base R).
#
# Reproduzir: baixar R/d.spls.norm.R e R/d.spls.lasso.R desse repo para
# o mesmo diretorio deste script, ajustar os `source()` abaixo, e rodar
#     Rscript gen_oracle.R
# a partir DESTE diretorio (escreve os .csv aqui do lado).
source("d.spls.norm.R")
source("d.spls.lasso.R")

set.seed(42)
n <- 30
p <- 12
X <- matrix(rnorm(n * p), nrow = n, ncol = p)
beta_true <- c(2, -1.5, 0, 0, 3, 0, 0, 0, 1, 0, 0, -2)
y <- as.vector(X %*% beta_true) + rnorm(n, sd = 0.1)

write.csv(X, "X.csv", row.names = FALSE)
write.csv(data.frame(y = y), "y.csv", row.names = FALSE)

mod <- d.spls.lasso(X = X, y = y, ncp = 4, ppnu = 0.7, verbose = TRUE)

write.csv(mod$Bhat, "Bhat.csv", row.names = FALSE)
write.csv(mod$loadings, "loadings.csv", row.names = FALSE)
write.csv(mod$scores, "scores.csv", row.names = FALSE)
write.csv(data.frame(intercept = mod$intercept), "intercept.csv", row.names = FALSE)
write.csv(mod$fitted.values, "fitted.csv", row.names = FALSE)
write.csv(data.frame(zerovar = mod$zerovar), "zerovar.csv", row.names = FALSE)
write.csv(data.frame(lambda = mod$lambda), "lambda.csv", row.names = FALSE)
write.csv(data.frame(Xmean = mod$Xmean), "Xmean.csv", row.names = FALSE)

cat("DONE\n")
