lsanalyzer_func_quantile <- function(BIFIEobj, vars, breaks, useInterpolation = TRUE, mimicIdbAnalyzer = FALSE, group=NULL, group_values=NULL)
{
  userfct <- function(X,w)
  {
    params <- numeric()
    for (cc in 1:ncol(X))
    {
      vx <- X[,cc]
      vw <- w
      ord <- order(vx,na.last=TRUE)
      vx <- vx[ord]
      vw <- vw[ord]
      if (any(is.na(vx)))
      {
        first_na <- min(which(is.na(vx)))
        vx <- vx[1:(first_na-1)]
        vw <- vw[1:(first_na-1)]
      }
      if (length(vx)>0)
      {
        relw <- cumsum(vw)/sum(vw)
        agg <- data.frame(x=vx,w=relw)
        for (bb in breaks)
        {
          if (any(agg$w<bb) && !all(agg$w<bb))
          {
            pos <- max(which(agg$w<bb))
            lowx <- agg$x[pos]
            loww <- agg$w[pos]
            uppx <- agg$x[pos+1]
            uppw <- agg$w[pos+1]
            if (useInterpolation) param <- lowx + ((uppx-lowx) * (bb-loww) / (uppw - loww + 10^-20))
            if (!useInterpolation && !mimicIdbAnalyzer) param <- lowx
            if (!useInterpolation && mimicIdbAnalyzer) param <- uppx
            params <- c(params,param)
          } else
          {
            params <- c(params,NaN)
          }
        }
      } else
      {
        params <- c(params,rep(NaN,length(breaks)))
      }
    }
    return(params)
  }
  
  userparnames <- character()
  for (vv in vars) userparnames <- c(userparnames,paste0(vv,"_yval_",breaks))
  res <- BIFIEsurvey::BIFIE.by(BIFIEobj = BIFIEobj,
                               vars = vars,
                               userfct = userfct,
                               userparnames = userparnames,
                               group = group,
                               group_values = group_values)
  
  res$stat$var <- sub("\\_yval\\_([0-9]|\\.)*$", "", res$stat$parm)
  res$stat$yval <- as.numeric(sub("^.*\\_yval\\_", "", res$stat$parm))
  res$stat$quant <- res$stat$est
  
  return(res)
}
