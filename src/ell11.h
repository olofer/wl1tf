/*
 * ell11.h
 *
 * ell-1 regularized weighted sum-of-absolute-errors
 * scalar signal reconstruction/filtering.
 *
 * Can use observation weights.
 * Allows specification of boundary conditions.
 *
 * Solves the following convex optimization problem:
 *
 * (1)  min_{y(i)} sum_i w(i)*|y(i)-x(i)| + lambda*sum_j |y(j)-2*y(j+1)+y(j+2)|
 *
 * where i=1..n, and the range of j depends on whether boundary points are
 * provided with the arguments yl, yr (can be empty = []). See details below.
 *
 * If input x is a matrix, each column of x will be processed and the
 * output y will also be a matrix of the same size as x.
 *
 * Observation weights w(i)=1 for all i; if w is empty. 
 *
 * ALGORITHM:
 *
 * Program (1) is recast as a linear program (LP) which is solved
 * using a primal-dual interior point method with special structure
 * in ~O(n) time and memory.
 *
 * (LP) min_z { h'*z }, s.t. E*z<=f
 *
 */

#ifndef __ELL11_H__
#define __ELL11_H__

#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
#include <time.h>
#include "fastclock.h"
#endif

#ifdef __SCATTERED_ASSERTIONS__
#warning "Scattered assertions are turned ON: is this really intended?"
#endif

/* Hard-coded "restrictions" */
#define __ELL11_MINIMUM_SAMPLES 5
#define __ELL11_MAXIMUM_ITERATIONS 100

/* Defaults */
#define __ELL11_DEFAULT_MAXITERS 50
#define __ELL11_DEFAULT_EPS 1.0e-6
#define __ELL11_DEFAULT_ETA 0.96

/*
 
 The LP has the following form:
 
 z = [y; xi1; xi2] is a vector with length n+n+nxi elements.
 
 (n)   y is the solution of interest
 (n)   xi1 are constraints for the absolute value inequalities for fitting error.
 (nxi) xi2 are constraints for the absolute value inequalities for penalty on 2nd diff.
 
 (LP) minimize h'*z, s.t. E*z<=f
 
 with the cost term:
 
 h = [0; w; lambda*ones(nxi,1)]
 
 where w is an n-vector of positive observation weights;
 and lambda>0 is the regularization parameter.
 
 The constraint data {E,f} are structured as follows:
 
 E = [Ey1, Exi1, 0;
      Ey2, 0, Exi2]
      
 f = [f1; f2]
 
 with f1 = kron(x, [1;-1]); and f2 = 0; except for optional boundary terms.
 
 Ey1  = kron(eye(n), [1;-1])
 Exi1 = kron(eye(n), [-1;-1])
 Ey2 = "shifted blocks" of [1, -2, 1; -1, 2, -1] (1 column shifts)
 Exi2 = kron(eye(nxi), [-1;-1])
 
 The details of handling of Ey2 depends on nyr, nyl (boundary costs).
 
 ***
 
 The main step (at each iteration) in the algorithm is to solve the 
 equation:
 
 (E'*Q*E)*x = b, for x, for two different RHSs b, where Q is a diagonal matrix.
 Let Q = diag([q1;q2]) where q1 is of length 2*n and q2 is of length 2*nxi.
 
 Let M(Q) = E'*Q*E; M then has the following structure
 
 M = [M11, M12, M13;
      M21, M22, 0  ;
      M31, 0, M33  ]
      
 with M12 = M21', M13 = M31' due to symmetry.
 
 M11 = Ey1'*diag(q1)*Ey1 + Ey2'*diag(q2)*Ey2   : sym. pentadiagonal (n)
 M21 = Exi1'*diag(q1)*Ey1                      : diagonal (n)
 M22 = Exi1'*diag(q1)*Exi1                     : diagonal (n)
 M31 = Exi2'*diag(q2)*Ey2                      : banded tri-diagonal (nxi-by-n)
 M33 = Exi2'*diag(q2)*Exi2                     : diagonal (nxi)
 
 Using Schur complements (block elimination) it is enough
 to factorize (banded Cholesky) the following matrix:
 
 M11p = M11 - M31'*inv(M33)*M31 - M21'*inv(M22)*M21 = L*L'
 
 Solving for x=[x1;x2;x3] with RHS b=[b1;b2;b3] then reduces to these steps:
 
 1) (L*L')*x1 = b1 - M31'*inv(M33)*b3 - M21'*inv(M22)*b2
 2) x2 = inv(M22)*(-M21*x1 + b2)
 3) x3 = inv(M33)*(-M31*x1 + b3)
 
 The banded structure of M31 depends on whether boundary cost terms are included.
 M11p will always be symmetric pentadiagonal and it is enough to store only the
 two lower diagonal and the diagonal (3*n buffer storage).
 
 NOTE: the code contains a test-suite for the above linear equation
       with random b and Q that verifies the solution by resubstitution
       and residual evaluation.
 
 */

/* ell11 : first "1" means L1 objective, second "1" means L1 penalty */
typedef struct ell11ProgramData {
  int n;
  int nxi; /* nxi = n-2 if no/free b.c. */
  int nyl;
  int nyr; /* in general: nxi = n-2 + nyl + nyr */
  
  double *w;  /* n-vector weights for observation; if NULL, all are 1.0 implied */
  double *uw;
  double *h;
  double *f;
  
  int ld; /* set to 3 */
  int kd; /* set to 2 */
  char uplo; /* set to 'L' */
  
  double *M11; /* banded/diagonal matrix buffers: all are O(n) memory */
  double *M11p;
  double *M21;
  double *M22;
  double *iM22;
  double *M31;
  double *M33;
  double *iM33;
  
  double *scratch;
  int scratchsize;
  
  double *buf;
  int bufsize;    /* # doubles allocated */
  
} ell11ProgramData;

/* Deallocate buffers */
void releaseEll11ProgramBuffers(
  ell11ProgramData *dat,
  int verbose)
{
  if (dat->buf!=NULL) {
    MEMFREE(dat->buf);
    dat->buf = NULL;
    if (verbose>0)
      THEPRINTF("[%s]: bufsize=%i (%.2f kB)\n",
        __func__, dat->bufsize, (sizeof(double)*dat->bufsize)/1024.0);
  }
}

/* Allocate buffers needed for length-n optimization problems and setup the
 * structurally fixed Ey representation which is a function of the 
 * boundary conditions provided by (yl,nyl) [left] and (yr,nyr) [right].
 */
int setupEll11ProgramBuffers(
  ell11ProgramData *dat,
  int n, int nyl, int nyr,
  int verbose)
{
  if (dat==NULL) return -1;  
  if (n<3 || nyl<0 || nyr<0) return -1;
  if (nyl>2 || nyr>2) return -1;
  
  memset(dat, 0, sizeof(ell11ProgramData));
  
  /* The boundary conditions affect the required number of xi variables */
  dat->n = n;
  dat->nxi = (n-2) + nyl + nyr;
  dat->nyl = nyl;
  dat->nyr = nyr;
  
  int nxi = dat->nxi; /* (save typing power below) */
  
  /* Determine how much memory is required for double-buffers */
  int nz = 2*dat->n + dat->nxi;
  int nq = 2*dat->n + 2*dat->nxi;
  
  dat->uplo = 'L'; /* do not change ! */
  dat->ld = 3;   /* do not change ! */
  dat->kd = 2;   /* do not change ! */
  
  /* Order is: w, uw, h, f, M11, M11p, M21, M22, iM22, M31, M33, iM33 */
  
  int bufsize = 0;
  
  bufsize += n;        /* w */
  bufsize += n;        /* uw */
  bufsize += nz;       /* h */
  bufsize += nq;       /* f */
  bufsize += 3*n;      /* M11 */
  bufsize += 3*n;      /* M11p */
  bufsize += n;        /* M21 */
  bufsize += n;        /* M22 */
  bufsize += n;        /* iM22 */
  bufsize += 3*n;      /* M31 */
  bufsize += nxi;      /* M33 */
  bufsize += nxi;      /* iM33 */
       
  dat->scratchsize = 10*nq + 6*nz;
  
  bufsize += dat->scratchsize;
  
  dat->bufsize = bufsize;
  dat->buf = MEMALLOC(sizeof(double)*(dat->bufsize));
  
  if (dat->buf==NULL) return -2;
  
  if (verbose>0)
    THEPRINTF("[%s]: bufsize=%i\n", __func__, dat->bufsize);
    
  /* Order is: w, uw, h, f, M11, M11p, M21, M22, iM22, M31, M33, iM33 */
  
  int ofs = 0;
  
  dat->w    = &(dat->buf[ofs]); ofs += n;
  dat->uw   = &(dat->buf[ofs]); ofs += n;
  dat->h    = &(dat->buf[ofs]); ofs += nz;
  dat->f    = &(dat->buf[ofs]); ofs += nq;
  dat->M11  = &(dat->buf[ofs]); ofs += 3*n;
  dat->M11p = &(dat->buf[ofs]); ofs += 3*n;
  dat->M21  = &(dat->buf[ofs]); ofs += n;
  dat->M22  = &(dat->buf[ofs]); ofs += n;
  dat->iM22 = &(dat->buf[ofs]); ofs += n;
  dat->M31  = &(dat->buf[ofs]); ofs += 3*n;
  dat->M33  = &(dat->buf[ofs]); ofs += nxi;
  dat->iM33 = &(dat->buf[ofs]); ofs += nxi;
  
  dat->scratch = &(dat->buf[ofs]);
  ofs += dat->scratchsize;
  
  #ifdef __SCATTERED_ASSERTIONS__
  if (ofs!=bufsize) {
    THEPRINTF("(ASSERTION FAIL): [%s]: bufsize=%i != ofs=%i\n",
      __func__, dat->bufsize, ofs);
    releaseEll11ProgramBuffers(dat, verbose);
    return -7;
  }
  #endif
  
  /* Setup buffer of default unity sample weights */
  for (ofs=0;ofs<dat->n;ofs++) dat->uw[ofs] = 1.0;
  
  /* Zero the RHS vector f; it is only nonzero for specific b.c:s (managed per solver instance) */
  memset(dat->f, 0, sizeof(double)*nq);
  
  /* Zero cost term; it will be managed by the solver routine per instance */
  memset(dat->h, 0, sizeof(double)*nz);
  
  return 0;
}

/*
 * Specially structured matrix-vector operations
 * needed in PDIPM algorithm loop
 */

/* [y1;y2] <- E*[x1;x2;x3]+[z1;z2]
 * E blocked as above:
 *   y1 <- Ey1*x1 + Exi1*x2
 *   y2 <- Ey2*x1 + Exi2*x3
 * where y1 has length 2*n and y2 has length 2*nxi
 * and x1,x2,x3 have lengths n,n,nxi
 */
static inline void __11_datEmultxplusz(
  const ell11ProgramData *dat,
  double *y, double *x, double *z)
{
  int n = dat->n;
  int nxi = dat->nxi;
  /* set y <- z */
  memcpy(y, z, 2*(n+nxi)*sizeof(double));
  double *x1 = &x[0];
  double *x2 = &x[n];
  double *x3 = &x[2*n];
  double *y1 = &y[0];
  double *y2 = &y[2*n];
  int rr, cc;
  /* y1 <- y1 + Ey1*x1 + Exi1*x2 */
  for (cc=0,rr=0;cc<n;cc++,rr+=2) {
    y1[rr] += x1[cc] - x2[cc];
    y1[rr+1] += -x1[cc] - x2[cc];
  }
  /* y2 <- y2 + Exi2*x3 */
  for (cc=0,rr=0;cc<nxi;cc++,rr+=2) {
    y2[rr] += -x3[cc];
    y2[rr+1] += -x3[cc];
  }
  /* y2 <- y2 + Ey2*x1 */
  rr = 0; cc = 0;
  if (dat->nyl==2) {
    y2[rr] += x1[cc];
    y2[rr+1] += -x1[cc];
    rr += 2;
    y2[rr] += -2*x1[cc]+x1[cc+1];
    y2[rr+1] += 2*x1[cc]-x1[cc+1];
    rr += 2;
  } else if (dat->nyl==1) {
    y2[rr] += -2*x1[cc]+x1[cc+1];
    y2[rr+1] += 2*x1[cc]-x1[cc+1];
    rr += 2;
  }
  for (cc=0;cc<n-2;cc++) {
    y2[rr] += x1[cc]-2*x1[cc+1]+x1[cc+2];
    y2[rr+1] += -x1[cc]+2*x1[cc+1]-x1[cc+2];
    rr += 2;
  }
  if (dat->nyr==2) {
    y2[rr] += x1[cc]-2*x1[cc+1];
    y2[rr+1] += -x1[cc]+2*x1[cc+1];
    rr += 2;
    y2[rr] += x1[cc+1];
    y2[rr+1] += -x1[cc+1];
    rr += 2;
  } else if (dat->nyr==1) {
    y2[rr] += x1[cc]-2*x1[cc+1];
    y2[rr+1] += -x1[cc]+2*x1[cc+1];
    rr += 2;
  }

  #ifdef __SCATTERED_ASSERTIONS__
  if (rr!=2*nxi || cc!=n-2) {
    THEPRINTF("[%s]: ERROR rr=%i, 2*nxi=%i, cc=%i, n=%i\n",
      __func__, rr, 2*nxi, cc, n);
  }
  #endif

}

/* y <- E'*x, E structured as above.
 * y = [y1;y2;y3] and x=[x1;x2] implies the following
 *
 * y1 <- Ey1'*x1 + Ey2'*x2
 * y2 <- Exi1'*x1
 * y3 <- Exi2'*x2
 *
 */
static inline void __11_datEtmultx(
  const ell11ProgramData *dat,
  double *y, double *x)
{
  int n = dat->n;
  int nxi = dat->nxi;
  double *y1 = &y[0];
  double *y2 = &y[n];
  double *y3 = &y[2*n];
  double *x1 = &x[0];
  double *x2 = &x[2*n];
  int rr, cc;
  for (rr=0,cc=0;rr<n;rr++,cc+=2) {
    y1[rr] = x1[cc]-x1[cc+1];
  }
  /* y1<-y1+Ey2'*x2 */
  rr = 0; cc = 0;
  if (dat->nyl==2) {
    /* OK as is */
  } else if (dat->nyl==1) {
    y1[cc] += -2*x2[rr]+2*x2[rr+1]+x2[rr+2]-x2[rr+3];
    cc++;
  } else {
    y1[cc] += x2[rr]-x2[rr+1];
    cc++;
    y1[cc] += -2*x2[rr]+2*x2[rr+1]+x2[rr+2]-x2[rr+3];
    cc++;
  }
  while (rr+6<=2*nxi) {
    y1[cc] += x2[rr]-x2[rr+1]-2*x2[rr+2]+2*x2[rr+3]+x2[rr+4]-x2[rr+5];
    rr+=2;
    cc++;
  }
  if (dat->nyr==2) {
    /* OK here as is! */
  } else if (dat->nyr==1) {
    y1[cc] += x2[rr]-x2[rr+1]-2*x2[rr+2]+2*x2[rr+3];
    cc++;
  } else {
    y1[cc] += x2[rr]-x2[rr+1]-2*x2[rr+2]+2*x2[rr+3];
    cc++;
    y1[cc] += x2[rr+2]-x2[rr+3];
    cc++;
  }

  #ifdef __SCATTERED_ASSERTIONS__
  if (cc!=n || rr+4!=2*nxi) {
    THEPRINTF("[%s]: cc=%i (n=%i); rr+4=%i, 2*nxi=%i\n", __func__, cc, n, rr+4, 2*nxi);
  }
  #endif

  for (rr=0,cc=0;rr<n;rr++,cc+=2) {
    y2[rr] = -x1[cc]-x1[cc+1];
  }
  for (rr=0,cc=0;rr<nxi;rr++,cc+=2) {
    y3[rr] = -x2[cc]-x2[cc+1];
  }
}

/* 
 * Main linear equation factorization step.
 *
 * Create  pos. def. factorization of the Schur complement
 * matrix M11p (see comments/documentation above).
 *
 */

int __11_factorizeEtDE(ell11ProgramData *dat, double *d)
{
  if (dat==NULL) return -1;
  if (d==NULL) return -2;
  
  int n = dat->n;
  int nxi = dat->nxi;
  
  double *d1 = &d[0];
  double *d2 = &d[2*n];
  
  int ii, jj, kk;
  
  /* Create diagonals for M21 and M22 and iM22 */
  for (ii=0,jj=0;ii<n;ii++,jj+=2) {
    double e1 = d1[jj] + d1[jj+1];
    double e2 = -d1[jj] + d1[jj+1];
    dat->M22[ii] = e1; /* NOTE: Ey1'*diag(d1)*Ey1 is equivalent to Exi1'*diag(d1)*Exi1 */
    dat->iM22[ii] = 1.0/e1;
    dat->M21[ii] = e2;
  }
  
  /* Create diagonal for M33 and iM33 */
  for (ii=0,jj=0;ii<nxi;ii++,jj+=2) {
    double e1 = d2[jj] + d2[jj+1];
    dat->M33[ii] = e1;
    dat->iM33[ii] = 1.0/e1;
  }
  
  /* Create banded matrix buffer M31 */
  int nyl = dat->nyl;
  int nyr = dat->nyr;
  
  /*memset(dat->M31, 0, 3*n*sizeof(double));*/
  int colmax = nxi;
  jj = 0;
  
  if (nyl==2) { /* M31 is tri-subdiagonal */
    colmax = nxi-2;
  } else if (nyl==1) { /* M31 is tri-diagonal */
    dat->M31[0] = 0.0;
    dat->M31[1] = 2.0*d2[0] - 2.0*d2[1];
    dat->M31[2] = -d2[2] + d2[3];
    colmax = nxi-1;
    jj += 1;
  } else { /* M31 is tri-superdiagonal */
    dat->M31[0] = 0.0;
    dat->M31[1] = 0.0;
    dat->M31[2] = -d2[0] + d2[1];
    dat->M31[3] = 0.0;
    dat->M31[4] = 2.0*d2[0] - 2.0*d2[1] ;
    dat->M31[5] = -d2[2] + d2[3];
    jj += 2;
  }
  
  ii = 0;
  int ofs = 3*jj;
  while (jj<colmax) {
    /* common pattern for three-column elements; using implied matrices */
    dat->M31[ofs] = -d2[ii] + d2[ii+1];
    dat->M31[ofs+1] = 2.0*d2[ii+2] -2.0*d2[ii+3];
    dat->M31[ofs+2] = -d2[ii+4] + d2[ii+5];
    jj++;
    ofs += 3;
    ii += 2;
  }

  #ifdef __SCATTERED_ASSERTIONS__
  if (ii+4!=2*nxi) {
    THEPRINTF("[%s]: ERROR ii=%i, 2*nxi=%i\n", __func__, ii, 2*nxi);
  }
  #endif

  if (nyr==2) {
    /* OK as is */
  } else if (nyr==1) {
    /* pad final column */
    dat->M31[ofs] = -d2[ii] + d2[ii+1];
    dat->M31[ofs+1] = 2.0*d2[ii+2] -2.0*d2[ii+3];
    dat->M31[ofs+2] = 0.0;
    jj++;
  } else {
    /* pad final two columns */
    dat->M31[ofs] = -d2[ii] + d2[ii+1];
    dat->M31[ofs+1] = 2.0*d2[ii+2] -2.0*d2[ii+3];
    dat->M31[ofs+2] = 0.0;
    dat->M31[ofs+3] = -d2[ii+2] + d2[ii+3];
    dat->M31[ofs+4] = 0.0;
    dat->M31[ofs+5] = 0.0;
    jj += 2;
  }

  #ifdef __SCATTERED_ASSERTIONS__
  if (jj!=n) {
    THEPRINTF("[%s]: WARNING: jj=%i but n=%i\n", __func__, jj, n);
  }
  #endif

  /* Create symmetric banded matrix buffer M11 <- Ey1'*diag(d1)*Ey1 + Ey2'*diag(d2)*Ey2 */
  
  /*memset(dat->M11, 0, 3*n*sizeof(double));*/
  
  jj = 0;
  /*colmax = nxi-nyl;*/ /* equal to n+nyr-2 */
  if (nyl==2) {
    /* OK as is */
  } else if (nyl==1) {
    dat->M11[0] = 4*(d2[0]+d2[1])+d2[2]+d2[3];
    dat->M11[1] = -2*(d2[0]+d2[1]+d2[2]+d2[3]);
    dat->M11[2] = d2[2]+d2[3];
    jj++;
  } else {
    dat->M11[0] = d2[0]+d2[1];
    dat->M11[1] = -2*(d2[0]+d2[1]);
    dat->M11[2] = d2[0]+d2[1];
    dat->M11[3] = 4*(d2[0]+d2[1])+d2[2]+d2[3];
    dat->M11[4] = -2*(d2[0]+d2[1]+d2[2]+d2[3]);
    dat->M11[5] = d2[2]+d2[3];
    jj += 2;
  }
  
  ii = 0;
  ofs = 3*jj;
  while (jj<n-2) {
    /* common pattern for sub-diagonal elements */
    dat->M11[ofs] = d2[ii]+d2[ii+1]+4*(d2[ii+2]+d2[ii+3])+d2[ii+4]+d2[ii+5];
    dat->M11[ofs+1] = -2*(d2[ii+2]+d2[ii+3]+d2[ii+4]+d2[ii+5]);
    dat->M11[ofs+2] = d2[ii+4]+d2[ii+5];
    jj++;
    ofs += 3;
    ii += 2;
  }
  
  if (nyr==2) {
    dat->M11[ofs] = d2[ii]+d2[ii+1]+4*(d2[ii+2]+d2[ii+3])+d2[ii+4]+d2[ii+5];
    dat->M11[ofs+1] = -2*(d2[ii+2]+d2[ii+3]+d2[ii+4]+d2[ii+5]);
    dat->M11[ofs+3] = d2[ii+2]+d2[ii+3]+4*(d2[ii+4]+d2[ii+5])+d2[ii+6]+d2[ii+7];
    ii += 8;
  } else if (nyr==1) {
    dat->M11[ofs] = d2[ii]+d2[ii+1]+4*(d2[ii+2]+d2[ii+3])+d2[ii+4]+d2[ii+5];
    dat->M11[ofs+1] = -2*(d2[ii+2]+d2[ii+3]+d2[ii+4]+d2[ii+5]);
    dat->M11[ofs+3] = d2[ii+2]+d2[ii+3]+4*(d2[ii+4]+d2[ii+5]);
    ii += 6;
  } else {
    dat->M11[ofs] = d2[ii]+d2[ii+1]+4*(d2[ii+2]+d2[ii+3]);
    dat->M11[ofs+1] = -2*(d2[ii+2]+d2[ii+3]);
    dat->M11[ofs+3] = d2[ii+2]+d2[ii+3];
    ii += 4;
  }
  dat->M11[ofs+2] = 0.0;
  dat->M11[ofs+4] = 0.0;
  dat->M11[ofs+5] = 0.0;

  #ifdef __SCATTERED_ASSERTIONS__
  if (ii!=2*nxi) {
    THEPRINTF("[%s]: ERROR ii=%i, 2*nxi=%i\n", __func__, ii, 2*nxi);
  }
  #endif

  /* update diagonal of M11 buffer with Ey1'*diag(d1)*Ey1 */
  for (ii=0,jj=0,ofs=0;ii<n;ii++,jj+=2,ofs+=3) {
    dat->M11[ofs] += d1[jj]+d1[jj+1];
  }
  
  /* Create "donwdated" M11p <- M11 - M21'*iM22*M21 - M31'*iM33*M31 */
/*  memset(dat->M11p, 0, 3*n*sizeof(double)); */
  memcpy(dat->M11p, dat->M11, 3*n*sizeof(double));
  for (ii=0,ofs=0;ii<n;ii++,ofs+=3) {
    double m21i = dat->M21[ii];
    dat->M11p[ofs] -= m21i*(dat->iM22[ii])*m21i;
  }
  
  /* M31'*iM33*M31 downdate loop (inner unrolled) */
  double dv[3];
  double *c0, *c1, *c2;
  if (nyl==2) {
    dv[0] = dat->iM33[0];
    dv[1] = dat->iM33[1];
    dv[2] = dat->iM33[2];
    kk = 2;
  } else if (nyl==1) {
    dv[0] = 0.0;
    dv[1] = dat->iM33[0];
    dv[2] = dat->iM33[1];
    kk = 1;
  } else {
    dv[0] = 0.0;
    dv[1] = 0.0;
    dv[2] = dat->iM33[0];
    kk = 0;
  }
  for (jj=0,ofs=0;jj<n-2;jj++) {
    c0 = &dat->M31[ofs];
    dat->M11p[ofs] -= c0[0]*dv[0]*c0[0]+c0[1]*dv[1]*c0[1]+c0[2]*dv[2]*c0[2];
    c1 = &dat->M31[ofs+3];
    dat->M11p[ofs+1] -= c0[1]*dv[1]*c1[0]+c0[2]*dv[2]*c1[1];
    c2 = &dat->M31[ofs+6];
    dat->M11p[ofs+2] -= c0[2]*dv[2]*c2[0];
    kk++;
    dv[0] = dv[1];
    dv[1] = dv[2];
    dv[2] = (kk>=nxi ? 0.0 : dat->iM33[kk]); /* should be enough here with == */
    ofs += 3;
  }
/*  THEPRINTF("[%s]: kk=%i, nxi=%i\n", __func__, kk, nxi);*/
  c0 = &dat->M31[ofs];
  dat->M11p[ofs] -= c0[0]*dv[0]*c0[0]+c0[1]*dv[1]*c0[1]+c0[2]*dv[2]*c0[2];
  c1 = &dat->M31[ofs+3];
  dat->M11p[ofs+1] -= c0[1]*dv[1]*c1[0]+c0[2]*dv[2]*c1[1];
  kk++;
  dv[0] = dv[1];
  dv[1] = dv[2];
  dv[2] = (kk>=nxi ? 0.0 : dat->iM33[kk]);
  ofs += 3;
  c0 = &dat->M31[ofs];
  dat->M11p[ofs] -= c0[0]*dv[0]*c0[0]+c0[1]*dv[1]*c0[1]+c0[2]*dv[2]*c0[2];
    
  /* in-place factorize M11p (banded Cholesky) */  
  int info = -1;
 
  /* F77_NAME(dpbtrf)(const char* uplo, const int* n, const int* kd,double* ab, const int* ldab, int* info); */
  F77_NAME(dpbtrf)(&(dat->uplo), &n, &(dat->kd), dat->M11p, &(dat->ld), &info);
  
  /* info=0 if OK, -i if param i had illegal value, i if not pos.def matrix */
  return info;
}

/* 
 * Solve the sym. block equation for a new RHS b (n+n+nxi)-vector.
 *
 * M11*x1 + M21'*x2 + M31'*x3 = b1
 * M21*x1 + M22 *x2 + 0       = b2
 * M31*x1 + 0       + M33 *x3 = b3
 *
 * b = [b1;b2;b3], and x=[x1;x2;x3], x1,b1,x2,b2 lengths n; x3,b3 lengths nxi
 *
 * Solution x is overwritten to b.
 * Exploits the prefactorized banded Schur complement
 * from the companion routine above.
 *
 */
int __11_solveFactorizedEq(ell11ProgramData *dat, double *b) {

  /* 1. replace b1: b1 <- b1 - M21'*inv(M22)*b2 - M31'*inv(M33)*b3
   * 2. in-place solve M11p*x1 = b1 (so that b1 <- x1 = solution )
   * 3. replace b2 <- inv(M22)*(b2-M21*x1), so that b2 <- x2
   * 4. replace b3 <- inv(M33)*(b3-M31*x1), so that b3 <- x3
   *
   * done!
   */
   
  int n = dat->n;
  int nxi = dat->nxi;
  double *u1 = &b[0];
  double *u2 = &b[n];
  double *u3 = &b[2*n];
  int ii;
  
  /* step 1 */
  for (ii=0;ii<n;ii++)
    u1[ii] -= (dat->M21[ii])*(dat->iM22[ii])*u2[ii];
  
  for (ii=0;ii<nxi;ii++)
    u3[ii] *= dat->iM33[ii];

/*
  cblas_dgbmv(
    CblasColMajor, CblasTrans,
    nxi, n, dat->nyl, 2-dat->nyl, // m,n,kl,ku;
    -1.0, dat->M31, 3,
    u3, 1,
    1.0, u1, 1);
*/ 

  char trans = 'T';
  char notrans = 'N';
  int kl = dat->nyl;
  int ku = 2 - dat->nyl;
  double alpha = -1.0;
  int lda = 3;
  int incx = 1;
  double beta = 1.0;
  int incy = 1;

  F77_NAME(dgbmv)(
    &trans, &nxi, &n, &kl, &ku,
    &alpha, dat->M31, &lda,
    u3, &incx,
    &beta, u1, &incy);

  /*
  F77_NAME(dgbmv)(const char *trans, const int *m, const int *n,
    const int *kl,const int *ku,
    const double *alpha, const double *a, const int *lda,
    const double *x, const int *incx,
    const double *beta, double *y, const int *incy);
  */
  
  for (ii=0;ii<nxi;ii++)
    u3[ii] *= dat->M33[ii];
  
  /* step 2: solve (L*L')*x1 = u1, inplace: u1 <- x1 */
  int nhrs = 1;
  int info = -1;

  /* F77_NAME(dpbtrs)(const char* uplo, const int* n,
     const int* kd, const int* nrhs,
     const double* ab, const int* ldab,
     double* b, const int* ldb, int* info);
  */
  
  F77_NAME(dpbtrs)(
    &(dat->uplo), &n, &(dat->kd),
    &nhrs, dat->M11p, &(dat->ld), u1, &n, &info);
  
  if (info!=0) return info;
  
  /* step 3: do b2 <- b2-M21*x1, b2 <- inv(M22)*b2 (merged since diagonal) */
  for (ii=0;ii<n;ii++)
    u2[ii] = dat->iM22[ii]*(u2[ii]-dat->M21[ii]*u1[ii]);
  
  /* step 4: do b3 <- b3-M31*x1, b3 <- inv(M33)*b3 */

  F77_NAME(dgbmv)(
    &notrans, &nxi, &n, &kl, &ku,
    &alpha, dat->M31, &lda,
    u1, &incx,
    &beta, u3, &incy);

/* cblas_dgbmv(
     CblasColMajor, CblasNoTrans,
     nxi, n, dat->nyl, 2-dat->nyl, // m,n,kl,ku;
     -1.0, dat->M31, 3,
     u1, 1,
     1.0, u3, 1); */

  for (ii=0;ii<nxi;ii++)
    u3[ii] *= dat->iM33[ii];

  return 0;
}


#if 0
/*
 * Basic utility functions needed in main PDIPM loop
 */

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

static inline void blas_addvec(double *y,double *x,int n,double a) {
  /* cblas_daxpy(n, a, x, 1, y, 1); */
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

/* Global variable for additional diagnostics timer (ugly) */
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
static double __11_global_clock_0 = 0.0;
static double __11_global_clock_1 = 0.0;
#endif

/*
 * Solve a new instance of the problem given data xsig and (if applicable boundary terms yl,yr).
 * Observation weights are specified by w (dat->w are default unity weights).
 * NOTE: the program size is fixed by the structure dat (pre-allocated buffers).
 *
 */
int ell11ProgramSolve(
  ell11ProgramData *dat,
  double *xsig,
  double lambda,
  double *w,
  double *yl,
  double *yr,
  double eta,
  double eps,
  int maxiters,
  int initopt,
  double *y,
  double *fy,
  double *xi,
  double *fxi,
  int *iters,
  int *cholerr,
  double *inftuple)
{
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
  fclk_timespec _tic1, _toc1;
  __11_global_clock_0 = 0.0;
  /* the above clock value should contain
   * the total time spent in "factorizeHpEtDE"
   * at exit of this routine
   */
  __11_global_clock_1 = 0.0; /* and this one is to time all triangular solves */
#endif

  /* check for basic misuse */
  if (dat==NULL) return -1;
  if (xsig==NULL) return -1;
  if (dat->nyl>0 && yl==NULL) return -1;
  if (dat->nyr>0 && yr==NULL) return -1;
  if (maxiters<=0) return -1;
  if (eta<=0.0 || eps<=0.0) return -1;
  
  double y2d[6] = { 1.0, -2.0, 1.0, -1.0, 2.0, -1.0};
  
  /* conform with notation for general inequality QP:
   *   min h'*x, s.t. E*x<=f
   * with x an n-vector and E has q rows.
   * So here: n = size of augmented decision variable x=[y;xi1;xi2]
   * NOTE: n != dat->n; instead n = 2*dat->n+dat->nxi.
   */

  int n = 2*dat->n + dat->nxi;
  int q = 2*dat->n + 2*dat->nxi;
  
  double
    *vx, *vz, *vs, *ve,
    *vrL, *vrs, *vrsz, *vrbar, *vhbar,
    *vdx, *vds, *vdz, *vtmp1, *vtmp2, *vtmp3, *vtmp4;
  
  int idx = 0;
  double *mbuf = dat->scratch;
  
  vx=&mbuf[idx]; idx+=n;
  vz=&mbuf[idx]; idx+=q;
  vs=&mbuf[idx]; idx+=q;
  ve=&mbuf[idx]; idx+=q;
  vrL=&mbuf[idx]; idx+=n;
  vrs=&mbuf[idx]; idx+=q;
  vrsz=&mbuf[idx]; idx+=q;
  vrbar=&mbuf[idx]; idx+=n;
  vhbar=&mbuf[idx]; idx+=n;
  vdx=&mbuf[idx]; idx+=n;
  vds=&mbuf[idx]; idx+=q;
  vdz=&mbuf[idx]; idx+=q;
  vtmp1=&mbuf[idx]; idx+=n;
  vtmp2=&mbuf[idx]; idx+=q;
  vtmp3=&mbuf[idx]; idx+=q;
  vtmp4=&mbuf[idx]; idx+=q;
  
  if (idx!=dat->scratchsize) {
    THEPRINTF("[%s]: ERROR %i=idx!=scratchsize=%i\n",
      __func__, idx, dat->scratchsize);
    return -1;
  }
  
  /* Initialize f1 and f2 separately (partitions of f) */
  
  double *f = dat->f;
  double *f1 = &(f[0]);
  double *f2 = &(f[2*dat->n]);
  
  for (idx=0;idx<dat->n;idx++) {
    int ofs = 2*idx;
    f1[ofs] = xsig[idx];
    f1[ofs+1] = -xsig[idx];
  }
  
  /* ASSUMES the "body" of f was already zeroed during initialization */
  int fq = 2*dat->nxi;
  f2[0] = 0.0; f2[1] = 0.0; f2[2] = 0.0; f2[3] = 0.0;
  f2[fq-4] = 0.0; f2[fq-3] = 0.0; f2[fq-2] = 0.0; f2[fq-1] = 0.0;
  
  if (dat->nyl>0) {
    /* set boundaries of f correctly given yl and Ey */
    if (dat->nyl==1) {
      f2[0] = -y2d[0]*yl[0];
      f2[1] = -y2d[3]*yl[0];
    } else {
      f2[0] = -y2d[0]*yl[0] - y2d[1]*yl[1];
      f2[1] = -y2d[3]*yl[0] - y2d[4]*yl[1];
      f2[2] = -y2d[0]*yl[1];
      f2[3] = -y2d[3]*yl[1];
    }
  }
  
  if (dat->nyr>0) {
    /* set boundaries of f correctly given yr and Ey */
    if (dat->nyr==1) {
      f2[fq-2] = -y2d[2]*yr[0];
      f2[fq-1] = -y2d[5]*yr[0];
    } else {
      f2[fq-4] = -y2d[2]*yr[0];
      f2[fq-3] = -y2d[5]*yr[0];
      f2[fq-2] = -y2d[1]*yr[0] - y2d[2]*yr[1];
      f2[fq-1] = -y2d[4]*yr[0] - y2d[5]*yr[1];
    }
  }
  
  /* If no weights are provided; point to unit weights */
  if (w==NULL) {
/*    memcpy(dat->w, dat-uw, (dat->n)*sizeof(double)); */
    w = dat->uw;
  } /* otherwise use the provided w pointer as is */
  
  /* Buffer pointers are initialized: proceed with PDIPM initialization */
  double *h = dat->h;
  /*for (idx=0;idx<dat->n;idx++) h[(dat->n)+idx] = w[idx];*/
  memcpy(&h[dat->n], w, (dat->n)*sizeof(double));
  for (idx=2*dat->n;idx<n;idx++) h[idx] = lambda;

  int numiters = 0;
  int cholretval = 0;
  int solvretval = 0;
  int oktostop = 0;
  
  double thrL, thrs, thrmu;
  double mu, alphaa, mua, sigma, xia;
  double etainf, beta;
  double infL, infs;
  
  double hinf = norminf(h, n);
  double finf = norminf(dat->f, q);
  
  thrL = (1.0+hinf)*eps;
  thrs = (1.0+finf)*eps;
  thrmu = eps;
  
  for (idx=0;idx<n;idx++) vx[idx] = 0.0;
  for (idx=0;idx<q;idx++) ve[idx] = 1.0;
  
  /* Determine inf-norms of input data (h,f,E) to heuristically set an initial point */
  if (initopt>0) {
    etainf = __dmaxFromPair(hinf, finf);
    etainf = __dmaxFromPair(etainf, 2.0); /* E matrix inf norm = 2 */
    beta = sqrt(etainf);
    for (idx=0;idx<q;idx++) {
      vz[idx] = beta;
      vs[idx] = beta;
    }
    if (initopt>1) {
      double mxsig = vecmean(xsig, dat->n);
      for (idx=0;idx<dat->n;idx++) {
        vx[idx] = mxsig;
        vx[idx+dat->n] = 2.0*fabs(xsig[idx]-mxsig);
      }
    }
  } else {
    memcpy(vz, ve, q*sizeof(double));
    memcpy(vs, ve, q*sizeof(double));
  }
  
  /* Calculate first residual */
      
  /* rL=h+E'*z; note that h is the objective gradient vector */
  __11_datEtmultx(dat, vrL, vz);
  blas_addvec(vrL, h, n, 1.0);
  /* rs=s+E*x-f; */
  __11_datEmultxplusz(dat, vrs, vx, vs);
  blas_addvec(vrs, f, q, -1.0);
  /* rsz=s.*z; */
  ewprodxy(vrsz, vs, vz, q);
  /* mu = sum(rsz)/q; */
  mu = vecmean(vrsz, q);
  
  infL = norminf(vrL, n);
  infs = norminf(vrs, q);
  oktostop = (infL<thrL && infs<thrs && mu<thrmu);
    
  /* Jump into algorithm main loop */
  while ( numiters<=maxiters && !oktostop  ) {
    ewdivxy(vtmp2, vz, vs, q); /* vtmp2=vz./vs; */
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_tic1);
#endif
    cholretval = __11_factorizeEtDE(dat, vtmp2);
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_toc1);
    __11_global_clock_0 += fclk_delta_timestamps(&_tic1, &_toc1);
#endif
    if (cholretval!=0) break;
    ewmaccnxyw(vtmp4, vtmp2, vrs, vz, q); /* vtmp4=-vtmp2.*vrs+vz */
    __11_datEtmultx(dat, vdx, vtmp4);
    blas_addvec(vdx, vrL, n, -1.0); /* vdx = E'*vtmp4 - vrL */
    /* Backsubstitution #1 using factorization above; store in vdx, rhs in vdx also */
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_tic1);
#endif
    solvretval = __11_solveFactorizedEq(dat, vdx);
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_toc1);
    __11_global_clock_1 += fclk_delta_timestamps(&_tic1, &_toc1);
#endif
    if (solvretval!=0) break;
    __11_datEmultxplusz(dat, vds, vdx, vrs);
    flipsign(vds, q);  /* vds=-(E*vdx+vrs); */ 
    ewmaccxyw(vdz, vtmp2, vds, vz, q);
    flipsign(vdz, q); /* vdz=-(z+vtmp2.*vds); */
    alphaa = 1.0;
    alphaa = alpha1(alphaa, vz, vdz, q);
    alphaa = alpha1(alphaa, vs, vds, q);
    mua = mu1(alphaa, vz, vdz, vs, vds, q);
    sigma = mua/mu;
    sigma *= sigma*sigma; /* sigma=(mua/mu)^3; */
    /* Update rhs for backsubstitution #2 */
    ewmaccxyw(vrsz, vds, vdz, vrsz, q);
    scmaccxyw(vrsz, -sigma*mu, ve, vrsz, q); /* vrsz=vrsz+ds.*dz-sigma*mu*ve; */
    ewdivxy(vtmp4, vrsz, vs, q);
    ewmaccnxyw(vtmp4, vtmp2, vrs, vtmp4, q); /* vtmp4=vrsz./vs-vtmp2.*vrs; */
    __11_datEtmultx(dat, vdx, vtmp4);
    blas_addvec(vdx, vrL, n, -1.0); /* vdx = E'*vtmp4 - vrL */
    /* Backsubstitution #2 */
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_tic1);
#endif
    solvretval = __11_solveFactorizedEq(dat, vdx);
#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_toc1);
    __11_global_clock_1 += fclk_delta_timestamps(&_tic1, &_toc1);
#endif
    if (solvretval!=0) break;
    __11_datEmultxplusz(dat, vds, vdx, vrs);
    flipsign(vds, q);  /* vds=-(E*vdx+vrs); */ 
    ewdivxy(vtmp4, vrsz, vs, q);
    ewmaccxyw(vdz, vtmp2, vds, vtmp4, q);
    flipsign(vdz, q); /* vdz=-(vrsz./vs+vtmp2.*vds); */
    /* Adjust search direction and take a step */
    alphaa = 1.0;
    alphaa = alpha1(alphaa, vz, vdz, q);
    alphaa = alpha1(alphaa, vs, vds, q);
    xia = alphaa*eta;
    blas_addvec(vx, vdx, n, xia);
    blas_addvec(vz, vdz, q, xia);
    blas_addvec(vs, vds, q, xia);
    /* Recalculate residuals */
    __11_datEtmultx(dat, vrL, vz);
    blas_addvec(vrL, h, n, 1.0);  /* rL=h+E'*z; */
    __11_datEmultxplusz(dat, vrs, vx, vs);
    blas_addvec(vrs, f, q, -1.0); /* rs=s+E*x-f; */
    ewprodxy(vrsz, vs, vz, q);
    mu = vecmean(vrsz, q); /* rsz=s.*z; mu = sum(rsz)/q; */
    /* check if converged */
    infL = norminf(vrL, n);
    infs = norminf(vrs, q);
    oktostop = (infL<thrL && infs<thrs && mu<thrmu);
    numiters++;  /* Log the iteration and iterate again */
  }
    
  /* Return final vector, objective value, iteration count, and clean up */
  /* NOTE: xi1 is not returned at all (only used for fitting cost evaluation) */
  if (y!=NULL) memcpy(y, vx, (dat->n)*sizeof(double));
  if (xi!=NULL) memcpy(xi, &vx[2*dat->n], (dat->nxi)*sizeof(double));
  
  if (fy!=NULL) {
    /* return the dot(w,xi1) part of the cost (fitting cost) */
    double su = 0.0;
    for (idx=(dat->n);idx<(2*dat->n);idx++) su += h[idx]*vx[idx];
    *fy = su;
  }
  
  if (fxi!=NULL) {
    /* return the lambda*sum(xi2) part of the cost (penalty cost) */
    double su = 0.0;
    for (idx=2*dat->n;idx<n;idx++) su += h[idx]*vx[idx];
    *fxi = su;
  }
  
  if (iters!=NULL) *iters = numiters;
  if (cholerr!=NULL) *cholerr = (cholretval==0 ? 0 : 1);
  
  /* return relative accuracy at final iteration */
  if (inftuple!=NULL) {
    inftuple[0] = infL/(1.0+hinf);
    inftuple[1] = infs/(1.0+finf);
    inftuple[2] = mu;
    /* all of the above are below eps, if converged */
  }
  
  return (oktostop ? 1 : 0);
}

#ifdef __INCLUDE_BANDED_CHOLESKY_TEST__

void make_runif(double *x, int n, double a, double b) {
  int ii;
  for (ii=0;ii<n;ii++)
    x[ii] = a + (b-a) * unif_rand(); //genrand_real3();
}

double maxInfDiff(double *x, double *y, int n) {
  double dii, s=0.0;
  int ii;
  for (ii=0;ii<n;ii++) {
    dii = fabs(x[ii]-y[ii]);
    if (dii>s) s=dii;
  }
  return s;
}

void AugmentedMultTest(ell11ProgramData *dat, double *x, double *y)
{
  /* y <- M*x, with M = [M11,M12,M13;M21,M22,0;M31,0,M33], M symmetric */
  
  if (dat==NULL || x==NULL || y==NULL) return;
  
  int ii;
  int n = dat->n;
  int nxi = dat->nxi;
  double *x1 = x;
  double *x2 = &x[n];
  double *x3 = &x[2*n];
  double *y1 = y;
  double *y2 = &y[n];
  double *y3 = &y[2*n];

  /*
  F77_NAME(dsbmv)(const char *uplo, const int *n, const int *k,
    const double *alpha, const double *a, const int *lda,
    const double *x, const int *incx,
    const double *beta, double *y, const int *incy);
  */

  char clo = 'L';
  double done = 1.0;
  double dzero = 0.0;
  int ithree = 3;
  int ione = 1;
  
  /* y1 <- M11*x1, y1 <- y1+M21'*x2, y1 <- y1+M31'*x3 */
  F77_NAME(dsbmv)(
    &clo, &n, &(dat->kd),
    &done, dat->M11, &ithree,
    x1, &ione,
    &dzero, y1, &ione);

  /*
  cblas_dsbmv(
    CblasColMajor, CblasLower,
    n, dat->kd,
    1.0, dat->M11, 3,
    x1, 1,
    0.0, y1, 1);
  */

  for (ii=0;ii<n;ii++)
    y1[ii] += dat->M21[ii]*x2[ii];

  char ctrans = 'T';
  char cnotrans = 'N';
  int ku = 2 - dat->nyl;

  /*
  F77_NAME(dgbmv)(const char *trans, const int *m, const int *n,
    const int *kl,const int *ku,
    const double *alpha, const double *a, const int *lda,
    const double *x, const int *incx,
    const double *beta, double *y, const int *incy);
  */
  
  /*
  cblas_dgbmv(
    CblasColMajor, CblasTrans,
    nxi, n, dat->nyl, 2-dat->nyl,
    1.0, dat->M31, 3,
    x3, 1,
    1.0, y1, 1);
  */

  F77_NAME(dgbmv)(
    &ctrans, &nxi, &n, &(dat->nyl), &ku,
    &done, dat->M31, &ithree,
    x3, &ione,
    &done, y1, &ione);
  
  /* y2 <- M22*x2, y2 <- y2 + M21*x1 */
  for (ii=0;ii<n;ii++)
    y2[ii] = dat->M22[ii]*x2[ii] + dat->M21[ii]*x1[ii];

  /* y3 <- M33*x3, y3 <- y3 + M31*x1 */
  for (ii=0;ii<nxi;ii++)
    y3[ii] = dat->M33[ii]*x3[ii];
  
  /*
  cblas_dgbmv(
    CblasColMajor, CblasNoTrans,
    nxi, n, dat->nyl, 2-dat->nyl,
    1.0, dat->M31, 3,
    x1, 1,
    1.0, y3, 1);
    */

  F77_NAME(dgbmv)(
    &cnotrans, &nxi, &n, &(dat->nyl), &ku,
    &done, dat->M31, &ithree,
    x1, &ione,
    &done, y3, &ione);
}

int randomFactorizeSolveTest(
  ell11ProgramData *dat, int numf, int rhsperf)
{
  THEPRINTF("[%s]: numf=%i, nrhs=%i\n", __func__, numf, rhsperf);

  GetRNGstate();
  
  int ff, bb;  
  double *d = dat->scratch;
  
  int nx = 2*dat->n+dat->nxi;
  int nq = 2*dat->n+2*dat->nxi;
  
  int info;
  
  double *x1 = &(dat->scratch[nq]);
  double *x2 = &(dat->scratch[nq+nx]);
  double *er = &(dat->scratch[nq+nx+nx]);
  
  for (ff=0;ff<numf;ff++){
    make_runif(d, nq, 0.5, 1.5);
    info = __11_factorizeEtDE(dat, d);
    if (info==0) {
      for (bb=0;bb<rhsperf;bb++) {
        /* random rhs; then solve, then check residual */
        make_runif(x1, nx, -1.0, 1.0);
        memcpy(x2, x1, sizeof(double)*nx);
        /* RHS x1 -> solution x1 (in-place) */
        info = __11_solveFactorizedEq(dat, x1);
        if (info==0) {
          AugmentedMultTest(dat, x1, er); /* er <- M*x1 */
          THEPRINTF("[%s]: max-abs-diff (%i:%i) = %e\n", __func__, ff, bb, maxInfDiff(er, x2, nx));
        } else {
          THEPRINTF("[%s]: failed to factorize (ff=%i:bb=%i).\n", __func__, ff, bb);
        }
      }
    } else {
      THEPRINTF("[%s]: failed to factorize (ff=%i).\n", __func__, ff);
    }
  }

  PutRNGstate();
      
  return 0;
}
#endif

#endif
