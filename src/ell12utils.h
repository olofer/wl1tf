/*
 * ell12utils.h
 *
 * Common code for l1 and l2 fitting l1 trend filter solvers.
 * For use in R package code.
 *
 */
 
#ifndef __ELL12UTILS_H_
#define __ELL12UTILS_H_

#ifndef MEMALLOC
#define MEMALLOC malloc
#endif

#ifndef MEMFREE
#define MEMFREE free
#endif

#ifndef THEPRINTF
#define THEPRINTF printf
#endif

static inline double __dmaxFromPair(double a, double b) {
  if (a>=b) return a;
    else return b;
}

static inline double norminf(double *x, int n) {
  int i; double s=0.0; double axi;
  for (i=0;i<n;i++) {
    axi = fabs(x[i]);
    if (axi>s) s=axi;
  }
  return s;
}

/* daxpy: y <- y + a*x, so a=1.0 or -1.0 for add or sub */
/* static inline void blas_addvec(double *y,double *x,int n,double a) {
  cblas_daxpy(n, a, x, 1, y, 1);
} */

/*
F77_NAME(daxpy)(const int *n, const double *alpha,
    const double *dx, const int *incx,
    double *dy, const int *incy);
 */

/* daxpy: y <- y + a*x, so a=1.0 or -1.0 for add or sub */
static inline void blas_addvec(double *y,double *x,int n,double a) {
  int incx = 1;
  int incy = 1;
  F77_NAME(daxpy)(&n, &a, x, &incx, y, &incy);
}

static inline void flipsign(double *y,int n) {
  int i; for (i=0;i<n;i++) y[i]=-y[i];
}

static inline double vecmean(double *x,int n) {
  int i; double s=0.0; for (i=0;i<n;i++) s+=x[i]; return (s/n);
}

static inline double alpha1(double alpha0,double *x,double *dx,int n) {
  int i; double a1=alpha0; double tmp;
  for (i=0;i<n;i++) {
    if (dx[i]<0.0) {
      tmp=-x[i]/dx[i];
      if (tmp<a1) a1=tmp;
    }
  }
  return a1;
}

static inline double mu1(double a,double *z,double *dz,double *s,double *ds,int n) {
  int i; double aa=0.0;
  for (i=0;i<n;i++)
    aa+=(z[i]+a*dz[i])*(s[i]+a*ds[i]);
  return (aa/n);
}

static inline void ewprodxy(double *z,double *x,double *y,int n) {
  int i; for (i=0;i<n;i++) z[i]=x[i]*y[i];
}

static inline void ewdivxy(double *z,double *x,double *y,int n) {
  int i; for (i=0;i<n;i++) z[i]=x[i]/y[i];
}

static inline void ewmaccxyw(double *z,double *x,double *y,double *w,int n) {
  int i; for (i=0;i<n;i++) z[i]=x[i]*y[i]+w[i];
}

static inline void ewmaccnxyw(double *z,double *x,double *y,double *w,int n) {
  int i; for (i=0;i<n;i++) z[i]=-x[i]*y[i]+w[i];
}

static inline void scmaccxyw(double *z,double x,double *y,double *w,int n) {
  int i; for (i=0;i<n;i++) z[i]=x*y[i]+w[i];
}

#endif

