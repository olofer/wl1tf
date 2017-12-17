#
# Checking the ~O(n) solve
#

library(wl1tf)
lambda <- 10.0
num.cols.per.block <- 1000 #500
size.vector <- seq(from=100, to=5000, by=100)
m <- length(size.vector)
time.11 <- array(NA, c(m, 1))
iter.11 <- array(NA, c(m, 1))
time.21 <- array(NA, c(m, 1))
iter.21 <- array(NA, c(m, 1))
for (k in 1:m) {
  nc <- size.vector[k]
  print(nc) # show heart-beat
  Xc <- array(rnorm(nc * num.cols.per.block), c(nc, num.cols.per.block))
  rep.11 <- wl11tf(Xc, lambda)
  stopifnot(all(rep.11$converged == 1))
  iter.11[k] <- mean(rep.11$iterations)
  time.11[k] <- mean(rep.11$clock[1, ]) * 1.0e3 # convert to milliseconds
  rep.21 <- wl21tf(Xc, lambda)
  stopifnot(all(rep.21$converged == 1))
  iter.21[k] <- mean(rep.21$iterations)
  time.21[k] <- mean(rep.21$clock[1, ]) * 1.0e3
}
 
#dev.new()
png("speed-1.png", type="cairo-png", bg="transparent", width=500, height=500);
plot(x=NULL, y=NULL, xlim=c(0, size.vector[m]), ylim=c(0, time.11[m]),
  xlab="signal length n",
  ylab=sprintf("average over %i random signals per signal length", num.cols.per.block))
points(size.vector, time.11, lwd=4, col="blue")
lines(size.vector, time.11, lwd=4, col="blue", lty=1)
lines(size.vector, iter.11, lwd=2, col="blue", lty=3)
points(size.vector, time.21, lwd=4, col="red")
lines(size.vector, time.21, lwd=4, col="red", lty=1)
lines(size.vector, iter.21, lwd=2, col="red", lty=3)
legend(
  "topleft",
  legend = c("wl11tf (LP)", "wl21tf (QP)"),
  lty = c(1, 1),
  lwd = c(4, 4),
  col = c("blue", "red"),
  text.col = c("blue", "red"))
dev.off()
