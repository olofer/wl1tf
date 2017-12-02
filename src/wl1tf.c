/*
  Intended interface:

  rep <- wl1tf(
    x, lambda, yl, yr, w,     # problem data
    eps, eta, maxiters,       # algorithm settings
    verbose)                  # enable printf's

  Here x can be a vector or a matrix.
  R uses column-major matrix storage.

  (1) R CMD build wl1tf && R CMD INSTALL --build [tar-gz-name-from-previous]
  (2) R CMD INSTALL (--preclean) (--clean) --build wl1tf
  (3) WITHIN R: remove.packages("wl1tf")

*/

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>

#define MEMALLOC malloc
#define MEMFREE free
#define THEPRINTF Rprintf

#undef __INCLUDE_BANDED_CHOLESKY_TEST__
#define __COMPILE_WITH_INTERNAL_TICTOC__
#undef __SCATTERED_ASSERTIONS__

#include "ell12utils.h"
#include "ell11.h"

SEXP wl1tf_R(
  SEXP x,
  SEXP lambda,
  SEXP yl,
  SEXP yr,
  SEXP w,
  SEXP sx_eps,
  SEXP sx_eta,
  SEXP sx_maxiters,
  SEXP sx_verbose)
{
  int verbose = 0;
  int initopt = 1;
  int maxiters = __ELL11_DEFAULT_MAXITERS;
  double eta = __ELL11_DEFAULT_ETA;
  double eps = __ELL11_DEFAULT_EPS;
  int nw = 0;
  double *weights = NULL;
  int nyl = 0;
  int nyr = 0;
  double *pyl = NULL;
  double *pyr = NULL;

  #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
  fclk_timespec _tic1, _toc1;
  #endif

  if (TYPEOF(sx_verbose) != INTSXP || length(sx_verbose) != 1) {
    error("verbose argument is misspecified");
  }

  if (INTEGER(sx_verbose)[0] > 0) {
    verbose = 1;
  }

  if (TYPEOF(x) != REALSXP) {
    error("input data x must be numeric");
  }

  int n = length(x);
  int r = n;
  int c = 1;
  SEXP dims = getAttrib(x, R_DimSymbol);
  if (!isNull(dims)) {
    if (length(dims) != 2) {
      error("input data x cannot be a multi-dim array");
    }
    r = INTEGER(dims)[0];
    c = INTEGER(dims)[1];
  }

  if (n != r * c) {
    error("dimension and length of x are inconsistent");
  }

  if (r < __ELL11_MINIMUM_SAMPLES) {
    error("signal is shorter than minimum allowed = %i", __ELL11_MINIMUM_SAMPLES);
  }

  /* Conclude the signal length = # rows (r) and the number of signals = # columns (c) */

  if (verbose > 0) {
    THEPRINTF("length(x) = %i\n", n);
    THEPRINTF("dim(x) = (%i, %i)\n", r, c);
  }

  if (TYPEOF(lambda) != REALSXP) {
    error("input lambda must be numeric");
  }

  if (!(length(lambda) == 1 || length(lambda) == c)) {
    error("wrong number of lambda elements"); 
  }

  for (int l = 0; l < length(lambda); l++) {
    if (REAL(lambda)[l] <= 0.0) {
      error("lambda[%i] is nonpositive", l);
    }
  }

  /* Determine the lhs and rhs boundary condition specifications */

  /* LHS b.c. */
  if (length(yl) != 0) {
    if (TYPEOF(yl) != REALSXP) {
      error("non-empty yl must be numeric");
    }
    if (!(length(yl) == 1 || length(yl) == 2)) {
      error("non-empty yl must have 1 or 2 elements");
    }
    nyl = length(yl);
    pyl = REAL(yl);
  }

  /* RHS b.c. */
  if (length(yr) != 0) {
    if (TYPEOF(yr) != REALSXP) {
      error("non-empty yr must be numeric");
    }
    if (!(length(yr) == 1 || length(yr) == 2)) {
      error("non-empty yr must have 1 or 2 elements");
    }
    nyr = length(yr);
    pyr = REAL(yr);
  }

  /* Determine weight vector input (OK if c elements or r*c elements or 0 elements) */
  if (length(w) != 0) {
    if (TYPEOF(w) != REALSXP) {
      error("non-empty w must be numeric");
    }
    nw = length(w);
    if (nw != c && nw != r*c) {
      error("w required to have element count compatible with x");
    }
    weights = REAL(w);
  }

  /* Read eps argument (0, 1] */
  if (length(sx_eps) != 0) {
    if (length(sx_eps) != 1 || TYPEOF(sx_eps) != REALSXP) {
      error("eps must be numeric scalar");
    }
    if (REAL(sx_eps)[0] > 0.0 && REAL(sx_eps)[0] < 1.0) {
      eps = REAL(sx_eps)[0];
    } else {
      warning("provided eps was ignored (out of bounds)");
    }
  }

  /* Read eta argument (0, 1] */
  if (length(sx_eta) != 0) {
    if (length(sx_eta) != 1 || TYPEOF(sx_eta) != REALSXP) {
      error("eta must be numeric scalar");
    }
    if (REAL(sx_eta)[0] > 0.0 && REAL(sx_eta)[0] <= 1.0) {
      eta = REAL(sx_eta)[0];
    } else {
      warning("provided eta was ignored (out of bounds)");
    }
  }

  if (TYPEOF(sx_maxiters) != INTSXP || length(sx_maxiters) != 1) {
    error("maxiters argument is misspecified");
  }

  if (INTEGER(sx_maxiters)[0] >= 1 && INTEGER(sx_maxiters)[0] <= __ELL11_MAXIMUM_ITERATIONS) {
    maxiters = INTEGER(sx_maxiters)[0];
  } else {
    warning("provided maxiters was ignored (out of bounds)");
  }

  if (verbose > 0) {
    THEPRINTF("[%s]: will use b.c. points; nyl=%i, nyr=%i\n", __func__, nyl, nyr);
    THEPRINTF("[%s]: will use eps=%e, eta=%f, and maxiters=%i\n", __func__, eps, eta, maxiters);
  }

  ell11ProgramData dat11;
  int retcode = setupEll11ProgramBuffers(&dat11, r, nyl, nyr, verbose);

  if (retcode != 0) {
    error("Failed to initialize ell11 data buffers (retcode = %i)", retcode);
  }

  #ifdef __INCLUDE_BANDED_CHOLESKY_TEST__
  /* Run test suite: 10 factorizations; 2 RHSs for each eq. */
  randomFactorizeSolveTest(&dat11, 10, 2);
  #endif

  double *px = REAL(x);

  SEXP y = PROTECT(allocMatrix(REALSXP, r, c));
  double *py = REAL(y);

  SEXP fy = PROTECT(allocVector(REALSXP, c));
  double *pfy = REAL(fy);

  SEXP fxi = PROTECT(allocVector(REALSXP, c));
  double *pfxi = REAL(fxi);

  SEXP inf = PROTECT(allocMatrix(REALSXP, 3, c));
  double *pinf = REAL(inf);

  SEXP itr = PROTECT(allocVector(INTSXP, c));
  int *pitr = INTEGER(itr);

  SEXP chr = PROTECT(allocVector(INTSXP, c));
  int *pchr = INTEGER(chr);

  SEXP cvg = PROTECT(allocVector(INTSXP, c));
  int *pcvg = INTEGER(cvg);

  #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
  SEXP clk = PROTECT(allocMatrix(REALSXP, 3, c));
  double *pclk = REAL(clk);
  #endif

  /* Call solver once for each column of input data */

  int kk = 0;
  while (kk < c)
  {
    int iters = -1;
    int cholerr = -1;
    double objy = 0.0;
    double objxi = 0.0;
    double lambda_kk = (length(lambda) == c ? REAL(lambda)[kk] : REAL(lambda)[0]);

    double *wkk = weights; // This is NULL if nw == 0; and it is fixed for all kk if nw == c
    if (nw != 0 && nw != c) {
      wkk = &(weights[kk * r]);
    }

    #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_tic1);
    #endif
  
    retcode = ell11ProgramSolve(
      &dat11,
      &(px[kk * r]),
      lambda_kk,
      wkk,           /* NOTE: if NULL, default unity weights are used */
      pyl,
      pyr,
      eta,
      eps,
      maxiters,
      initopt,
      &(py[kk * r]), /* y */
      &objy,         /* fy */
      NULL,          /* ignore return of xi vector */
      &objxi,        /* fxi */
      &iters,
      &cholerr,
      &(pinf[kk * 3])
      );
      
    #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_toc1);
    pclk[kk * 3 + 0] = fclk_delta_timestamps(&_tic1, &_toc1); // total time of call kk
    pclk[kk * 3 + 1] = __11_global_clock_0; // time spent on banded Cholesky factorization
    pclk[kk * 3 + 2] = __11_global_clock_1; // time spent on solve using the factorization
    #endif

    pfy[kk] = objy;     /* loss term */
    pfxi[kk] = objxi;   /* regularization term */
    pitr[kk] = iters;   /* Num. iterations done */
    pchr[kk] = cholerr; /* Cholesky error code */
    pcvg[kk] = retcode; /* Convergence flag */

    if (verbose > 0) {
      THEPRINTF("converged(%i) = %i (iterations = %i)\n", kk, retcode, iters);
    }
    
    kk++;
  }

  releaseEll11ProgramBuffers(&dat11, verbose);

  #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
  int list_elems = 8;
  #else
  int list_elems = 7;
  #endif

  int unprotect_elems = list_elems + 2;

  SEXP OS = PROTECT(allocVector(VECSXP, list_elems));
  SET_VECTOR_ELT(OS, 0, fy);
  SET_VECTOR_ELT(OS, 1, fxi);
  SET_VECTOR_ELT(OS, 2, inf);
  SET_VECTOR_ELT(OS, 3, y);
  SET_VECTOR_ELT(OS, 4, itr);
  SET_VECTOR_ELT(OS, 5, chr);
  SET_VECTOR_ELT(OS, 6, cvg);

  SEXP nms = PROTECT(allocVector(STRSXP, list_elems));
  SET_STRING_ELT(nms, 0, mkChar("fy"));
  SET_STRING_ELT(nms, 1, mkChar("fxi"));
  SET_STRING_ELT(nms, 2, mkChar("stop.eps"));
  SET_STRING_ELT(nms, 3, mkChar("y"));
  SET_STRING_ELT(nms, 4, mkChar("iterations"));
  SET_STRING_ELT(nms, 5, mkChar("chol.err"));
  SET_STRING_ELT(nms, 6, mkChar("converged"));

  #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
  // Set eighth element and name it
  SET_VECTOR_ELT(OS, 7, clk);
  SET_STRING_ELT(nms, 7, mkChar("clock"));
  #endif
  
  setAttrib(OS, R_NamesSymbol, nms);
  UNPROTECT(unprotect_elems);

  return OS;
}
