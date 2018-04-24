#
# Test simulation of the Telegraph process (2 levels).
#   - https://en.wikipedia.org/wiki/Telegraph_process
#
# Jump rates defined by 2 positive numbers lambda_a and _b.
# The process is integrated and hidden in Gaussian noise.
#
# The task is to do "online" batch l1 trend filtering using 
# a left hand side boundary condition to link up the next
# trend solution to the previous.
#
# RUN: Rscript --vanilla wl1tf-lhsbc.R
#

library(wl1tf)

set.seed(0)

value_a <- 1.0
lambda_a <- 0.10  # jump rate towards state a
value_b <- -1.0
lambda_b <- 0.10  # jump rate towards state b

deltaTime <- 1.0 / 100.0
maxTime <- 1000.0
maxSteps <- maxTime / deltaTime
batchTime <- 25.0

print(sprintf('samples/batch = %f', batchTime / deltaTime))

T <- array(NA, maxSteps)
X <- array(NA, maxSteps)  # Telegraph process X(t)
Y <- array(NA, maxSteps)  # Y(t) = integral of X(t)
Z <- array(NA, maxSteps)  # Y(t) + noise
W <- array(NA, maxSteps)  # recovered signal ~ Y(t), estimated from Z(t)
yl <- NULL
the_state <- 0
t <- 0
y <- 0
kk <- 0
kk0 <- 0
tbatch <- 0
batchSplits <- 0
while (kk < maxSteps) {
  if (the_state == 0) {
    v <- value_a
    if (runif(1) < 1 - exp(-deltaTime * lambda_b)) {
      the_state <- 1
    }
  } else {
    v <- value_b
    if (runif(1) < 1 - exp(-deltaTime * lambda_a)) {
      the_state <- 0
    }
  }
  kk <- kk + 1
  T[kk] <- t
  X[kk] <- v
  Y[kk] <- y
  Z[kk] <- Y[kk] + rnorm(1, sd = 10.0)
  y <- y + v * deltaTime
  t <- t + deltaTime
  tbatch <- tbatch + deltaTime
  if (tbatch >= batchTime) {
    # l1 trend filtering of new batch; using LHS boundary condition (penalty cost)
    r <- wl21tf(as.vector(Z[kk0:kk]), lambda = 10000.0,
      yl = yl, yr = NULL, w = NULL,
      eps = 1e-6, eta = 0.96, maxiters = 50,
      verbose = 0)
    stopifnot(r$converged == 1)
    W[kk0:kk] <- r$y
    ny <- length(r$y)
    yl <- c(r$y[ny - 1], r$y[ny])
    kk0 <- kk + 1
    tbatch <- 0
    batchSplits <- c(batchSplits, t)
  }
}

#print(names(r))
print(sprintf('num time steps = %i', maxSteps))

# Next check mean and variance of process X against analytical results
ref_mean <- (value_a * lambda_a + value_b * lambda_b) / (lambda_a + lambda_b)
sample_mean <- mean(X)
print(sprintf('ref mean = %f, sample mean = %f', ref_mean, sample_mean))

ref_var <- ((value_a - value_b) ^ 2) * lambda_a * lambda_b / (lambda_a + lambda_b) ^ 2
sample_var <- var(X)
print(sprintf('ref var. = %f, sample var. = %f', ref_var, sample_var))

# Visualize integrated Telegraph process
png("batch-1.png", type = "cairo-png", bg = "transparent", width = 750, height = 500)
plot(T, Z,
  type = 'l',
  xlab = 'time t',
  ylab = 'Y(t)',
  main = '"Online" recovery of an integrated Telegraph process')
lines(T, Y, col = 'blue', lwd = 2)
for (ii in 1:length(batchSplits)) {
  abline(v = batchSplits[ii], lty = 2, lwd = 1, col = 'blue')
}
lines(T, W, col = 'green', lwd = 2, lty = 1)
lines(T, Y - W, col = 'red', lwd = 2, lty = 1)
legend(
  "bottomleft",
  bg = "white",
  legend = c("Noisy data", "True signal", "Estimate", "Residual"),
  lty = c(1, 1, 1, 1),
  lwd = c(2, 2, 2, 2),
  col = c("black", "blue", "green", "red"),
  text.col = c("black", "blue", "green", "red"))
dev.off()
