# Gera os fixtures de tests/test_robpca.py a partir de rrcov::PcaHubert (referencia independente).
# Uso:  Rscript gen_oraculo_robpca.R   (grava oraculo_robpca.json ao lado do script)
# Versoes usadas na geracao: ver o campo "versoes" do JSON.
suppressMessages(library(rrcov))

num <- function(x) paste0("[", paste(sprintf("%.17g", as.numeric(x)), collapse = ","), "]")
mat <- function(m) paste0("[", paste(apply(as.matrix(m), 1, num), collapse = ","), "]")
lgl <- function(x) paste0("[", paste(ifelse(x, "true", "false"), collapse = ","), "]")

make <- function(seed, n, p, r, nout, shift, noise = 0.3, spread = 3) {
  set.seed(seed)
  L <- matrix(rnorm(p * r), p, r)
  X <- matrix(rnorm(n * r), n, r) %*% t(L) * spread + matrix(rnorm(n * p), n, p) * noise
  if (nout > 0) X[seq_len(nout), ] <- X[seq_len(nout), ] + matrix(rnorm(nout * p, shift), nout, p)
  X
}

casos <- list(
  list(id = "A_n40_p30_k3",      X = make(1, 40, 30, 3, 5, 3),   args = list(k = 3)),
  list(id = "B_n20_p8_auto",     X = make(2, 20, 8, 2, 3, 6, 0.5), args = list(k = 0)),
  list(id = "C_n100_p3_mcd",     X = make(3, 100, 3, 3, 10, 5, 1, 1), args = list(k = 0)),
  list(id = "D_n60_p50_a06",     X = make(4, 60, 50, 3, 8, 4),   args = list(k = 2, alpha = 0.6)),
  list(id = "E_n30_p5_scale",    X = make(5, 30, 5, 3, 3, 6, 1, 1), args = list(k = 2, scale = TRUE)),
  list(id = "F_n50_p40_auto",    X = make(6, 50, 40, 4, 6, 4),   args = list(k = 0)),
  list(id = "G_n40_p30_k1",      X = make(7, 40, 30, 1, 5, 3),   args = list(k = 1)),
  list(id = "H_n25_p60_kernel",  X = make(8, 25, 60, 2, 4, 4),   args = list(k = 2)),
  list(id = "I_n80_p6_nomcd",    X = make(9, 80, 6, 3, 8, 5, 0.6, 1), args = list(k = 3, mcd = FALSE))
)

saida <- character()
for (cs in casos) {
  r <- do.call(PcaHubert, c(list(x = cs$X), cs$args))
  args <- cs$args
  js <- sprintf(paste0('{"id":"%s","args":{%s},"X":%s,"k":%d,"quan":%d,"eigenvalues":%s,"center":%s,',
                       '"loadings":%s,"scores":%s,"sd":%s,"od":%s,"cutoff_sd":%.17g,"cutoff_od":%.17g,"flag":%s}'),
    cs$id,
    paste(vapply(names(args), function(a) {
      v <- args[[a]]
      sprintf('"%s":%s', a, if (is.logical(v)) tolower(as.character(v)) else format(v))
    }, ""), collapse = ","),
    mat(cs$X), r@k, as.integer(r@quan), num(r@eigenvalues), num(r@center),
    mat(r@loadings), mat(r@scores), num(r@sd), num(r@od), slot(r, "cutoff.sd"), slot(r, "cutoff.od"), lgl(r@flag))
  saida <- c(saida, js)
}
vers <- sprintf('{"R":"%s","rrcov":"%s","robustbase":"%s"}', getRversion(),
                as.character(packageVersion("rrcov")), as.character(packageVersion("robustbase")))
arq <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "oraculo_robpca.json")
writeLines(sprintf('{"versoes":%s,"casos":[%s]}', vers, paste(saida, collapse = ",\n")), arq)
cat("gravado", arq, "\n")
