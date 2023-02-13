#STAT534 Homework 4
#Author: Joseph David

#Problem 1
#computes laplace approximation of the log marginal likelihood
getLaplaceApprox <- function(response,explanatory,data,betaMode){
  source("helper_hw4.R")
  
  y = data[,response]
  x = data[,explanatory]
  
  logLik = logisticLoglik(y,x,betaMode)
  
  return (-(1/2)*(betaMode[1]^2 + betaMode[2]^2) + logLik - (1/2)*logdet(-getHessian(y,x,betaMode)))
}

#Problem 2
#Implements the metropolis-hastings algorithm
getPosteriorMeans <- function(response,explanatory,data,betaMode,niter){
  source("helper_hw4.R")
  
  y = data[,response];
  x = data[,explanatory];
  
  current_beta = betaMode;
  current_logLik = logisticLoglik(y,x,current_beta);
  betas = matrix(0,2,niter);
  
  for (j in 1:niter){
    new_beta = mvrnorm(1,current_beta,-solve(getHessian(y,x,betaMode)));
    new_logLik = logisticLoglik(y,x,new_beta);
    
    if (new_logLik >= current_logLik){
      current_beta = new_beta;
      current_logLik = new_logLik;
    }
    else{
      u = runif(1);
      if (log(u) <= new_logLik - current_logLik){
        current_beta = new_beta;
        current_logLik = new_logLik;
      }
    }
    betas[,j] = current_beta
  }
  return(rowMeans(betas))
}

#Problem 3
bayesLogistic = function(apredictor,response,data,NumberOfIterations)
{
  source("helper_hw4.R")
  
  y = data[,response];
  x = data[,apredictor];
  
  betaMode = getcoefNR(response,apredictor,data);
  beta = getPosteriorMeans(response,apredictor,data,betaMode,NumberOfIterations);
  
  return(list(apredictor=apredictor,
              logmarglik=getLaplaceApprox(response,apredictor,data,betaMode),
              beta0bayes=beta[1],
              beta1bayes=beta[2]))
}

#PARALLEL VERSION
#datafile = the name of the file with the data
#NumberOfIterations = number of iterations of the Metropolis-Hastings algorithm
#clusterSize = number of separate processes; each process performs one or more
#univariate regressions
main <- function(datafile,NumberOfIterations,clusterSize)
{
  #read the data
  data = read.table(datafile,header=FALSE);
  
  #the sample size is 148 (number of rows)
  #the explanatory variables are the first 60 columns for '534binarydata.txt'
  #the last column is the binary response
  response = ncol(data);
  lastPredictor = ncol(data)-1;
  
  #initialize a cluster for parallel computing
  cluster <- makeCluster(clusterSize, type = "SOCK")
  clusterExport(cluster, c("getPosteriorMeans","getLaplaceApprox"))
  clusterEvalQ(cluster,library(MASS))
  
  #run the MC3 algorithm from several times
  results = clusterApply(cluster, 1:lastPredictor, bayesLogistic,
                         response,data,NumberOfIterations);
  
  #print out the results
  for(i in 1:lastPredictor)
  {
    cat('Regression of Y on explanatory variable ',results[[i]]$apredictor,
        ' has log marginal likelihood ',results[[i]]$logmarglik,
        ' with beta0 = ', results[[i]]$beta0bayes,
        ' and beta1 = ', results[[i]]$beta1bayes,
        '\n');    
  }
  
  #destroy the cluster
  stopCluster(cluster);  
}

#NOTE: YOU NEED THE PACKAGE 'SNOW' FOR PARALLEL COMPUTING
require(snow);
source("helper_hw4.R")
library(MASS)

#this is where the program starts
#main('/Users/joseph/Courses/stat534/hw4/534binarydata.txt',10000,10);

#testing Problem 1
#response=61
#explanatory=2
#data = as.matrix(read.table("/Users/joseph/Courses/stat534/hw4/534binarydata.txt",header=FALSE,sep="\t"))
#betaMode = getcoefNR(response,explanatory,data)
#print(getLaplaceApprox(response,explanatory,data,betaMode))

#explanatory = 1;
#niter = 10000;
#betaMode = getcoefNR(response,explanatory,data);
#print(getPosteriorMeans(response,explanatory,data,betaMode,niter))
