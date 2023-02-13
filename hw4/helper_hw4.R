logdet <- function(A) {
  ev <- eigen(A)    
  eigenvalues <- ev$values 
  sum(log(eigenvalues))    
}

inverseLogit <- function(x){
  return(exp(x)/(1+exp(x))); 
}
#function for the computation of the Hessian
inverseLogit2 <- function(x){
  return(exp(x)/(1+exp(x))^2); 
}

getPi <- function(x,beta){
  x0 = cbind(rep(1,nrow(as.matrix(x))),x);
  return(inverseLogit(x0%*%beta));
}

#another function for the computation of the Hessian
getPi2 <- function(x,beta){
  x0 = cbind(rep(1,nrow(as.matrix(x))),x);
  return(inverseLogit2(x0%*%beta));
}

logisticLoglik <- function(y,x,beta){
  Pi = getPi(x,beta);
  return(sum(y*log(Pi))+sum((1-y)*log(1-Pi)));
}

#obtain the gradient for Newton-Raphson
getGradient <- function(y,x,beta){
  gradient = matrix(0,2,1);
  Pi = getPi(x,beta);
  
  gradient[1,1] = sum(y-Pi);
  gradient[2,1] = sum((y-Pi)*x);
  
  return(gradient);
}

getHessian <- function(y,x,beta){
  hessian = matrix(0,2,2);
  Pi2 = getPi2(x,beta);
  
  hessian[1,1] = sum(Pi2);
  hessian[1,2] = sum(Pi2*x);
  hessian[2,1] = hessian[1,2];
  hessian[2,2] = sum(Pi2*x^2);
  
  return(-hessian);
}

#this function implements our own Newton-Raphson procedure
getcoefNR <- function(response,explanatory,data)
{
  #2x1 matrix of coefficients`
  beta = matrix(0,2,1);
  y = data[,response];
  x = data[,explanatory];
  
  #current value of log-likelihood
  currentLoglik = logisticLoglik(y,x,beta);
  
  #infinite loop unless we stop it someplace inside
  while(1)
  {
    newBeta = beta - solve(getHessian(y,x,beta))%*%getGradient(y,x,beta);
    newLoglik = logisticLoglik(y,x,newBeta);
    
    #at each iteration the log-likelihood must increase
    if(newLoglik<currentLoglik)
    {
      cat("CODING ERROR!!\n");
      break;
    }
    beta = newBeta;
    #stop if the log-likelihood does not improve by too much
    if(newLoglik-currentLoglik<1e-6)
    {
      break; 
    }
    currentLoglik = newLoglik;
  }
  
  return(beta);
}

#NOTE: YOU NEED THE PACKAGE 'RCDD' TO PROPERLY RUN THIS CODE
#load the 'RCDD' package
library(rcdd);

#this is the version of the 'isValidLogistic' function
#based on Charles Geyers RCDD package
#returns TRUE if the calculated MLEs can be trusted
#returns FALSE otherwise
isValidLogisticRCDD <- function(response,explanatory,data)
{
  if(0==length(explanatory))
  {
    #we assume that the empty logistic regresion is valid
    return(TRUE);
  }
  logisticreg = suppressWarnings(glm(data[,response] ~ as.matrix(data[,as.numeric(explanatory)]),family=binomial(link=logit),x=TRUE));
  tanv = logisticreg$x;
  tanv[data[,response] == 1, ] <- (-tanv[data[,response] == 1, ]);
  vrep = cbind(0, 0, tanv);
  #with exact arithmetic; takes a long time
  #lout = linearity(d2q(vrep), rep = "V");
  
  lout = linearity(vrep, rep = "V");
  return(length(lout)==nrow(data));
}

#this function uses ’glm’ to fit a logistic regression
#and returns the BIC = deviance + log(SampleSize)*NumberOfRegressionCoefficients
getLogisticAIC <- function(response,explanatory,data)
{
  #check if the regression has no explanatory variables
  if(0==length(explanatory))
  {
    #regression with no explanatory variables
    deviance = glm(data[,response] ~ 1,family=binomial(link=logit))$deviance;
  }
  else
  {
    #regression with at least one explanatory variable
    deviance = glm(data[,response] ~ as.matrix(data[,as.numeric(explanatory)]),family=binomial(link=logit))$deviance;
  }
  return(deviance+2*(1+length(explanatory)));
}


#Homework 3, Problem 1
#Implementation of the MC3 algorithm.
#Inputs:
#response is an integer column number of data which corresponds to the response variable.
#data is the matrix of data, including explanatory and response variables.
#n_iter is the number of iterations for the algorithm to run
#Outputs:
#Returns the best logistic regression model (as a sorted vector of variables) as well as the 
#AIC of that model. 
MC3search <- function(response,data,n_iter){
  #Iteration 0
  p <- 60;
  response <- 61;
  vars <- seq(1,p)
  k <- sample(vars,1);
  
  isValid <- FALSE;
  while (!isValid){
    currentModel <- sample(vars,k,replace=FALSE);
    isValid <- isValidLogisticRCDD(response, currentModel, data);
  }
  bestRegression <- currentModel;
  regAIC <- vector('numeric',n_iter);
  regAIC[1] <- getLogisticAIC(response, currentModel, data);
  regAIC_min <- regAIC[1]
  
  #Iteration r
  #Steps 1 and 2
  for (r in (2:n_iter) ){
    nbhdValid <- list();
    
    counter <- 1;
    for (var in currentModel){
      if ( isValidLogisticRCDD( response, setdiff( currentModel, var ), data ) ){
        nbhdValid[[counter]] <- setdiff( currentModel, var )
        counter <- counter + 1;
      }
    }
    for (var in setdiff( vars, currentModel )){
      if ( isValidLogisticRCDD( response, union( currentModel, var ), data ) ){
        nbhdValid[[counter]] <- union( currentModel, var )
        counter <- counter + 1;
      }
    }
    
    #Step 3
    newModel <- nbhdValid[[sample( 1 : (length(nbhdValid)), 1 )]];
    
    #Step 4
    nbhdValid_new <- list();
    
    counter <- 1;
    for (var in newModel){
      if ( isValidLogisticRCDD( response, setdiff( newModel, var ), data ) ){
        nbhdValid_new[[counter]] <- setdiff( newModel, var )
        counter <- counter + 1;
      }
    }
    for (var in setdiff( vars, newModel )){
      if ( isValidLogisticRCDD( response, union( newModel, var ), data ) ){
        nbhdValid_new[[counter]] <- union( newModel, var )
        counter <- counter + 1;
      }
    }
    
    #Steps 5 and 6
    aic <- getLogisticAIC(response, currentModel, data);
    p <- -aic - log(length(nbhdValid));
    aic_new <- getLogisticAIC(response, newModel, data);
    p_new <- -aic_new - log(length(nbhdValid_new));
    
    #Steps 7 and 8
    if (p_new > p){
      currentModel <- newModel;
      if (aic_new < regAIC_min){
        bestRegression <- currentModel;
        regAIC_min <- aic_new;
      }
      regAIC[r] <- aic_new;
    } else{
      u <- runif(1);
      if (log(u) < (p_new - p)){
        currentModel <- newModel;
        if (aic_new < regAIC_min){
          bestRegression <- currentModel;
          regAIC_min <- aic_new;
        }
        regAIC[r] <- aic_new;
      } else{
        regAIC[r] <- aic;
      }
    }
  }
  results <- list("Best Regression" = sort(bestRegression), "Best AIC" = regAIC_min)
  return(results)
}

