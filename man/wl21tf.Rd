\name{wl21tf}
\alias{wl21tf}
\title{Quadratic programming l1 trend filtering}
\usage{
wl21tf(x, lambda, yl=NULL, yr=NULL, w=NULL, eps=1e-6, eta=0.96, maxiters=50, verbose=0)
}
\arguments{
\item{x}{Numeric input vector or matrix. If matrix, each column is a signal.}
\item{lambda}{Regularization parameter. Must be positive. Can be a single scalar or a vector with one element per signal/column of \code{x}.}
\item{yl}{0, 1, or 2 elements specifying the left hand side boundary condition (implied cost).}
\item{yr}{0, 1, or 2 elements specifying the right hand side boundary condition (implied cost).}
\item{w}{Numeric vector/matrix compatible with x providing (positive) observation weights.}
\item{eps}{Convergence epsilon.}
\item{eta}{Interior point algorithm damping parameter.}
\item{maxiters}{Maximum interior point iterations.}
\item{verbose}{Show some information messages.}
}
\value{
List containing the quadratic programming solution vector/matrix field \code{y} and an assortment of metadata.
}
\description{
Solves the following convex denoising optimization program in time and memory O(n). Minimize for vector \eqn{y} (n elements): \deqn{(1/2)\sum_{i=1}^{n} w_i (x_i - y_i)^2 + \lambda\sum_{j}|y_j-2y_{j+1}+y_{j+2}|} given data vector \eqn{x} and observation weights \eqn{w}. This typically yields a piecewise linear approximation to the sequence \eqn{x} where the break points are automatically allocated and controlled by adjusting \eqn{\lambda > 0}.
}
\examples{

z <- (1:3000-1)/1000
z <- z - floor(z)
x <- 2.0 * z + rnorm(length(z))
X.none <- wl21tf(x, 100.0, yl=c(), yr=c())
plot(x, col = "black")
lines(X.none$y, col = "blue", lwd = 4)

}
