# R entry point to compiled code wl1tf.c
wl1tf <- function(x, lambda, yl=NULL, yr=NULL, w=NULL, eps=1e-6, eta=0.96, maxiters=50, verbose=0) {
  if (length(w) > 0) {
    stopifnot(all(w > 0))
  }
  .Call(wl1tf_R, x, lambda, yl, yr, w, eps, eta, as.integer(maxiters), as.integer(verbose))
}
