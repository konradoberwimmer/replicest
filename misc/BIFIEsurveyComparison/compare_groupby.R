library(BIFIEsurvey)

rm(list=ls())
source("./quantiles.R")
options(scipen = 99)

# artificial data ----

df_wgts = data.frame(
  wgt = c(1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 1.75, 1.75, 2.0, 2.0),
  repwgt1 = c(2.0, 0.0, 1.25, 1.25, 1.5, 1.5, 1.75, 1.75, 2.0, 2.0),
  repwgt2 = c(1.0, 1.0, 2.5, 0.0, 1.5, 1.5, 1.75, 1.75, 2.0, 2.0),
  repwgt3 = c(1.0, 1.0, 1.25, 1.25, 3.0, 0.0, 1.75, 1.75, 2.0, 2.0),
  repwgt4 = c(1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 3.5, 0.0, 2.0, 2.0),
  repwgt5 = c(1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 1.75, 1.75, 4.0, 0.0)
)

df_imp1 = cbind(data.frame(
  x = 1:10,
  y = c(1,0,1,0,1,0,1,0,1,0)
), df_wgts)
df_imp2 = cbind(data.frame(
  x = 1:10,
  y = c(1,0,0,0,0,0,1,0,1,0)
), df_wgts)
df_imp3 = cbind(data.frame(
  x = 1:10,
  y = c(1,0,1,1,1,0,1,0,1,0)
), df_wgts)
df_imp4 = cbind(data.frame(
  x = 1:10,
  y = c(1,0,0,1,0,0,1,0,1,0)
), df_wgts)

dat.BO <- BIFIE.data(
  data.list = list(df_imp1, df_imp2, df_imp3, df_imp4),
  wgt = "wgt",
  wgtrep = df_wgts[, paste0("repwgt", 1:5)],
)

summary(dat.BO)
res <- BIFIE.univar(dat.BO, "x", group = "y")
res$stat_M
res$stat_SD

# PIRLS 2021 AUT data ----

df_pirls_2021_aut <- read.csv("./data/asgautr5.csv", header = TRUE, sep = ";")
df_pirls_2021_aut_BO <- BIFIE.data.jack(df_pirls_2021_aut, wgt = "TOTWGT", jktype = "JK_TIMSS2", pv_vars = c("ASRREA", "ASRLIT", "ASRINF", "ASRIIE", "ASRRSI", "ASRIBM"), cdata = TRUE)
summary(df_pirls_2021_aut_BO)

## correlations ----
#no groups
res <- BIFIE.correl(df_pirls_2021_aut_BO, vars = c("ITSEX", "ASRREA"))
res$stat.cor
res$stat.cov
#groups
res <- BIFIE.correl(df_pirls_2021_aut_BO, vars = c("ITSEX", "ASRREA"), group = "ASRIBM")
res$stat.cor
res$stat.cov

## quantiles ----
#lower
res <- lsanalyzer_func_quantile(df_pirls_2021_aut_BO, vars = c("ASRREA", "ASBG11F"), breaks = c(0.1, 0.25, 0.5, 0.75, 0.9), useInterpolation = FALSE, group = "ITSEX")
res$stat
#interpolation
res <- lsanalyzer_func_quantile(df_pirls_2021_aut_BO, vars = c("ASRREA", "ASBG11F"), breaks = c(0.1, 0.25, 0.5, 0.75, 0.9), group = "ITSEX")
res$stat
#upper
res <- lsanalyzer_func_quantile(df_pirls_2021_aut_BO, vars = c("ASRREA", "ASBG11F"), breaks = c(0.1, 0.25, 0.5, 0.75, 0.9), useInterpolation = FALSE, mimicIdbAnalyzer = TRUE, group = "ITSEX")
res$stat

## linear regression ----
res <- BIFIE.linreg(df_pirls_2021_aut_BO, formula = ASRREA ~ ASBG03 + ASBG04, group = "ITSEX")
res$stat
