#
# Check glmgen::trendfilter(.) versus wl1tf::wl21tf(.)
# using default settings, on a simple test case
#

library(glmgen)  # obtain from: https://github.com/statsmaths/glmgen
library(wl1tf)

set.seed(0)

n <- 1000
x <- seq(from = -2*pi, to = 2*pi, len = n)
ytrue <- 1.5*sin(x) + sin(2*x)
y <- ytrue + rnorm(n, sd = 0.5)
lambda.tf <- 50.0
w21 <- wl1tf::wl21tf(y, lambda = lambda.tf)
out <- glmgen::trendfilter(y, k=1, lambda = lambda.tf)
yy <- predict(out, x.new = 1:n)
err <- max(abs(w21$y - yy))

png("check-1.png", type="cairo-png", bg="transparent", width=500, height=500)
plot(x, y,
  xlab = "x", ylab = "y", 
  main = sprintf("l1 trend filter check: diff = %e", err))
lines(x, ytrue, col = "black", lwd = 4, lty = 2)
lines(x, w21$y, col = "blue", lwd = 4)
lines(x, yy, col = "red", lwd = 4, lty = 3)
legend(
  "bottomleft",
  legend = c(
    sprintf("wl1tf;  lambda = %.1f", lambda.tf),
    sprintf("glmgen; lambda = %.1f", lambda.tf),
    "Truth"),
  lty = c(1, 3, 2),
  lwd = c(4, 4, 4),
  col = c("blue", "red", "black"),
  text.col = c("blue", "red", "black"))
dev.off()
