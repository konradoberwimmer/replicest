library(BIFIEsurvey)
rm(list=ls())

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
