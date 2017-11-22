\name{wl1tf-package}
\alias{wl1tf-package}
\docType{package}
\title{
\packageTitle{wl1tf}
}
\description{
\packageDescription{wl1tf}
}
\details{

The DESCRIPTION file:
\packageDESCRIPTION{wl1tf}
\packageIndices{wl1tf}

See \code{\link{wl1tf}} and the example below.
}
\author{
\packageAuthor{wl1tf}

Maintainer: \packageMaintainer{wl1tf}
}

\examples{

require(wl1tf)
z <- (1:3000-1)/1000
z <- z - floor(z)
x <- 2.0 * z + rnorm(length(z))
X <- wl1tf(x, 100.0)
stopifnot(all(X$converged==1))
plot(x, col = "black")
lines(X$y, col = "blue", lwd = 4)

}

\seealso{
Classical l2-fitting l1 trend filtering: \url{http://web.stanford.edu/~boyd/papers/l1_trend_filter.html}
}
