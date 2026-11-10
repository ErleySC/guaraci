# gen_oracle.R -- segundo oraculo (caso de borda sparsity=0), usado por
# tests/test_dual_spls.py::test_dual_spls_matches_r_oracle_ppnu_zero.
# Mesma fonte/versao/metodologia que ../gen_oracle.R (ver cabecalho la').
source("../d.spls.norm.R")
source("../d.spls.lasso.R")

set.seed(1)
n <- 50
p <- 10
X <- matrix(rnorm(n * p), n, p)
beta <- rnorm(p)
y <- as.vector(X %*% beta) + rnorm(n, sd = 0.01)
mod <- d.spls.lasso(X = X, y = y, ncp = 3, ppnu = 0.0, verbose = TRUE)

write.csv(X, "X.csv", row.names = FALSE)
write.csv(data.frame(y = y), "y.csv", row.names = FALSE)
write.csv(mod$Bhat, "Bhat.csv", row.names = FALSE)
write.csv(data.frame(intercept = mod$intercept), "intercept.csv", row.names = FALSE)
write.csv(data.frame(zerovar = mod$zerovar), "zerovar.csv", row.names = FALSE)
write.csv(data.frame(lambda = mod$lambda), "lambda.csv", row.names = FALSE)
