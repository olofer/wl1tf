# wl1tf
Linear and quadratic programming l1 trend filtering with R wrapper.
Supports observation weights and boundary conditions.

## Quick install
Start R. Load devtools library. Then run:
```{r}
install_github("olofer/wl1tf")
```

## Examples
The l1 fitting option is more median-like and the l2-fitting is more mean-like.

Basic trend filtering: `R` code: `examples-git/wl1tf-example.R` 

![Basic trend filtering](/examples-git/example-1.png)

Work required (and space) is linear in signal length: `R` code: `examples-git/wl1tf-speed.R`

![Work required (and space) is linear in signal length](/examples-git/speed-1.png)

Solid lines are mean solve time in milliseconds. Dashed lines are mean number of iterations required to converge.

Further see e.g.
```{r}
library(wl1tf)
help(wl1tf)
```
