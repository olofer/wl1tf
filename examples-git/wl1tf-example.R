# Basic example plot for wl1tf package
library(wl1tf)
data(Seatbelts)
col.name <- "kms"
tt <- 1:nrow(Seatbelts)
xx <- as.vector(Seatbelts[ , col.name])
lambda.11 <- 50.0
lambda.21 <- 500.0
w1 <- wl11tf(xx, lambda.11)
w2 <- wl21tf(xx, lambda.21)
png("example-1.png", type="cairo-png", bg="transparent", width=500, height=500);
plot(tt, xx, xlab = "month", ylab = col.name)
lines(tt, w1$y, col = "blue", lwd = 2)
lines(tt, w2$y, col = "red", lwd = 2)
legend(
  "topleft",
  legend = c(
    sprintf("l1-fit lambda = %.1f", lambda.11),
    sprintf("l2-fit lambda = %.1f", lambda.21)),
  lty = c(1, 1),
  lwd = c(2, 2),
  col = c("blue", "red"),
  text.col = c("blue", "red"))
dev.off()