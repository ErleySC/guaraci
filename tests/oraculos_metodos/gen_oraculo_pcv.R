# Gera os fixtures de tests/test_pcv_regressao.py a partir do pacote R `pcv` 1.2.0 (S. Kucheryavskiy;
# espelho https://github.com/cran/pcv), a referencia independente do metodo.
# Uso:  Rscript gen_oraculo_pcv.R /caminho/do/clone/cran/pcv   (grava oraculo_pcv.json ao lado do script)
# Os arquivos R do pacote sao lidos com source(), sem instalar. A parte ORTOGONAL do conjunto de pseudo-validacao
# usa numeros aleatorios (runif) e por isso e' desligada aqui: guarda-se a parte explicada (X.pv.hat), as matrizes D,
# as distancias ortogonais q_k, R, R_k e o RMSEP-PV, que nao depende da parte ortogonal (ela e' ortogonal ao
# espaco de R, entao nao muda as predicoes; isto e' conferido numericamente abaixo com duas sementes).
args <- commandArgs(trailingOnly = TRUE)
fonte <- if (length(args)) args[1] else "pcv"
env <- new.env()
for (f in list.files(file.path(fonte, "R"), pattern = "\\.R$", full.names = TRUE)) sys.source(f, envir = env)
qreg <- list()
getx_real <- env$getxpvorth
env$getxpvorth <- function(q.k, X.k, PRM) { qreg[[length(qreg) + 1]] <<- q.k; matrix(0, length(q.k), ncol(X.k)) }

num <- function(x) paste0("[", paste(sprintf("%.17g", as.numeric(x)), collapse = ","), "]")
mat <- function(m) paste0("[", paste(apply(as.matrix(m), 1, num), collapse = ","), "]")

gen <- function(seed, n, p, nr, ruido = 0.5) {
  set.seed(seed)
  Tt <- matrix(rnorm(n * 4), n, 4) %*% diag(c(4, 3, 2, 1))
  P <- qr.Q(qr(matrix(rnorm(p * 4), p, 4)))
  X <- Tt %*% t(P) + matrix(rnorm(n * p), n, p) * 0.2
  B <- matrix(rnorm(4 * nr), 4, nr)
  Y <- Tt %*% B + matrix(rnorm(n * nr), n, nr) * ruido
  list(X = X, Y = Y)
}

casos <- list(
  list(id = "P1_ven5_n30_p12", d = gen(11, 30, 12, 1), ncomp = 4, cv = list("ven", 5), center = TRUE, scale = FALSE, scope = "global"),
  list(id = "P2_loo_n20_p8", d = gen(12, 20, 8, 1), ncomp = 3, cv = list("loo"), center = TRUE, scale = FALSE, scope = "global"),
  list(id = "P3_ven4_local_scale_pls2", d = gen(13, 36, 10, 2), ncomp = 3, cv = list("ven", 4), center = TRUE, scale = TRUE, scope = "local"),
  list(id = "P4_ven6_n24_p40", d = gen(14, 24, 40, 1), ncomp = 5, cv = list("ven", 6), center = TRUE, scale = FALSE, scope = "global"),
  list(id = "P5_segmentos_explicitos", d = gen(15, 30, 9, 1), ncomp = 3, cv = rep(1:5, each = 6), center = TRUE, scale = FALSE, scope = "global"),
  list(id = "P6_ncomp_truncado", d = gen(16, 15, 10, 1), ncomp = 12, cv = list("ven", 5), center = TRUE, scale = FALSE, scope = "global"),
  list(id = "P7_sem_centrar", d = gen(17, 28, 7, 1), ncomp = 3, cv = list("ven", 4), center = FALSE, scale = FALSE, scope = "global")
)

saida <- character()
for (cs in casos) {
  X <- cs$d$X; Y <- cs$d$Y; n <- nrow(X)
  qreg <- list()
  Xhat <- env$pcvpls(X, Y, ncomp = cs$ncomp, cv = cs$cv, center = cs$center, scale = cs$scale, cv.scope = cs$scope)
  ind <- as.integer(env$pcvcrossval(cs$cv, n, Y[, 1]))
  nseg <- max(ind)
  q <- numeric(n)
  for (k in seq_len(nseg)) q[ind == k] <- qreg[[k]]
  ncomp_ef <- ncol(attr(Xhat, "D"))
  # modelo global (mesmo centramento/escala do pacote) e RMSEP-PV de 1..ncomp_ef componentes
  mX <- if (cs$center) colMeans(X) else rep(0, ncol(X)); mY <- if (cs$center) colMeans(Y) else rep(0, ncol(Y))
  sX <- if (cs$scale) apply(X, 2, sd) else rep(1, ncol(X)); sY <- if (cs$scale) apply(Y, 2, sd) else rep(1, ncol(Y))
  m <- env$simpls(scale(X, mX, sX), scale(Y, mY, sY), ncomp_ef)
  rmsep_de <- function(Xp) sapply(seq_len(ncomp_ef), function(a) {
    Yh <- sweep(sweep(scale(Xp, mX, sX) %*% m$R[, 1:a, drop = FALSE] %*% t(m$C[, 1:a, drop = FALSE]), 2, sY, "*"), 2, mY, "+")
    sqrt(colMeans((Y - Yh)^2))
  })
  # conferencia: com a parte ortogonal real (duas sementes) o RMSEP-PV nao muda
  env$getxpvorth <- getx_real
  set.seed(1); Xf1 <- env$pcvpls(X, Y, ncomp = cs$ncomp, cv = cs$cv, center = cs$center, scale = cs$scale, cv.scope = cs$scope)
  set.seed(2); Xf2 <- env$pcvpls(X, Y, ncomp = cs$ncomp, cv = cs$cv, center = cs$center, scale = cs$scale, cv.scope = cs$scope)
  env$getxpvorth <- function(q.k, X.k, PRM) { qreg[[length(qreg) + 1]] <<- q.k; matrix(0, length(q.k), ncol(X.k)) }
  dif <- max(abs(rmsep_de(Xf1) - rmsep_de(Xf2)), abs(rmsep_de(Xf1) - rmsep_de(Xhat)))
  stopifnot(dif < 1e-8)
  saida <- c(saida, sprintf(paste0('{"id":"%s","cv":%s,"center":%s,"scale":%s,"scope":"%s","ncomp_in":%d,"ncomp":%d,',
      '"X":%s,"Y":%s,"ind":%s,"Xhat":%s,"D":%s,"q":%s,"R":%s,"Rk":%s,"rmsep_pv":%s,"dif_semente":%.3g}'),
    cs$id, if (is.list(cs$cv)) sprintf('["%s"%s]', cs$cv[[1]], if (length(cs$cv) > 1) paste0(",", cs$cv[[2]]) else "") else num(cs$cv),
    tolower(cs$center), tolower(cs$scale), cs$scope, cs$ncomp, ncomp_ef,
    mat(X), mat(Y), num(ind), mat(Xhat), mat(attr(Xhat, "D")), num(q), mat(attr(Xhat, "R")),
    paste0("[", paste(sapply(seq_len(nseg), function(k) mat(attr(Xhat, "Rk")[k, , ])), collapse = ","), "]"),
    mat(rmsep_de(Xhat)), dif))
}
vers <- sprintf('{"R":"%s","pcv":"%s"}', getRversion(), read.dcf(file.path(fonte, "DESCRIPTION"))[1, "Version"])
arq <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "oraculo_pcv.json")
writeLines(sprintf('{"versoes":%s,"casos":[%s]}', vers, paste(saida, collapse = ",\n")), arq)
cat("gravado", arq, "\n")
