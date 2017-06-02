#!/bin/Rscript
args <- commandArgs(TRUE)

library(compiler)
library(nleqslv)
options(digits = 10)

ns <- as.numeric(args[1])
numberProcesses <- as.numeric(args[2])

## Set Parameters
rho <- 2
Xs <- runif(ns, 1000000, 2000000) / 10000
betas <- rbeta(ns, 3, 1)
betas <- betas / sum(betas)
gamma <- runif(1, 1, 1.3)
drts <- 0.8

prodSum <- gamma * sum(betas * Xs^rho)^(drts / rho - 1)

## p <- numeric(ns)
## for(i in 1:ns) {
##     p[i] <- -1 * betas[i] * Xs[i]^(rho - 1) * prodSum
## }
p <- -1 * betas * Xs^(rho - 1) * prodSum
Y <- -1 * gamma * sum(betas * Xs^rho)^(drts / rho)

ssfun <- function(vars, betas, rho, drts) {

    Xs <- vars[1:ns]
    gamma <- vars[(ns + 1)]
    
    prodSum <- gamma * sum(betas * Xs^rho)^(drts / rho - 1)

    ## for-loop version
    ## r <- numeric(ns + 1)
    ## for(i in 1:ns) {
    ##     r[i] <- -1*p[i] + betas[i] * Xs[i]^(rho - 1) * prodSum
    ## }

    ## parallel mcmapply-version
    ## r <- c(mcmapply(function(i) p[i] + betas[i] * Xs[i]^(rho - 1) * prodSum,
    ##                 1:ns, mc.cores = numberProcesses),
    ##        Y + gamma * sum(betas * Xs^rho)^(drts / rho)
    ##        )

    r <- c(p + betas * Xs^(rho - 1) * prodSum,
           Y + gamma * sum(betas * Xs^rho)^(drts / rho))
    r
}

ssfun.bc <- cmpfun(ssfun)


start <- Sys.time()
sol <- nleqslv(x = c(rep(150, ns), 1.15),
               fn = ssfun, jac = NULL,
               betas = betas, rho = rho, drts = drts)

residual <- sum(abs(sol$x - c(Xs, gamma)))

end <- Sys.time()
diff.time <- difftime(end, start, units = "sec")

cat("R, ",
    ifelse(grepl("criterion near zero", sol$message),
           "CONVERGED", "DIVERGED"), ", ",
    residual, ", ",
    ns, ", ",
    sol$iter, ", ",
    numberProcesses, ", ",
    diff.time, "\n",
    sep = "")
