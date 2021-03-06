# wl1tf
Linear and quadratic programming l1 trend filtering with `R` wrapper.
Supports observation weights and boundary conditions.

## Quick install
Start `R`. Load `devtools` library. Then run:
```{r}
install_github("olofer/wl1tf")
```

## Basic use-case
Fits piecewise linear approximations to a batch of data with automatic break-points: `R` code: `examples-git/wl1tf-check.R`

![Basic use-case solution compared to `glmgen::trendfilter`](/examples-git/check-1.png)

Basic use-case solution compared to `glmgen::trendfilter`.

## More examples
Work required (and space) is linear in signal length: `R` code: `examples-git/wl1tf-speed.R`

![Work required (and space) is linear in signal length](/examples-git/speed-1.png)

Solid lines are mean solve time in milliseconds. Dashed lines are mean number of iterations required to converge.

The l1 fitting option is more median-like and the l2-fitting is more mean-like. Significantly different values of `lambda` are typically required to obtain comparable solutions for l1- and l2-fitting.

Basic trend filtering: `R` code: `examples-git/wl1tf-example.R` 

![Basic trend filtering](/examples-git/example-1.png)

## Advanced example
Piece-by-piece solution linked by LHS boundary condition/cost: `R` code: `examples-git/wl1tf-lhsbc.R`

![Patched up solution](/examples-git/batch-1.png)

The blue vertical dashed lines indicate the boundaries between batches. The solution is constructed from left to right. As soon as the next batch of data is acquired, the next short segment of the total solution is found by calling `wl21tf` with a left-hand-side boundary condition. This ensures that the total signal is "continuous".

## Documentation
Further see e.g.
```{r}
library(wl1tf)
help(wl1tf)
```
