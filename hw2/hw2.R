#Problem 1
#this function uses ’glm’ to fit a logistic regression
#and returns the BIC = deviance + log(SampleSize)*NumberOfRegressionCoefficients
getLogisticBIC <- function(response,explanatory,data)
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

#HOMEWORK 2, PROBLEM 2: Forward greedy search
forwardSearchAIC <- function(response,data,lastPredictor)
{
  #start with the empty regression with no predictors
  bestRegression = NULL;
  #calculate the AIC of the empty regression
  bestRegressionAIC = getLogisticAIC(response,bestRegression,data);
  cat('\n\n\n\nforwardSearch :: The empty logistic regression has AIC = ',bestRegressionAIC,'\n');
  
  #vector that keeps track of all the variables
  #that are currently NOT in the model
  VariablesNotInModel = 1:lastPredictor;
  
  #add variables one at a time to the current regression
  #and retain that variable that gives the smallest values of AIC associated
  #Make the model that includes that variable to be the current model
  #if it gives a smaller AIC than the AIC of the current regression
  
  #stop when there are no variables that can be included in the model
  stepNumber = 0;
  while(length(VariablesNotInModel)>=1)
  {
    #record the number of steps performed
    stepNumber = stepNumber + 1;
    
    #create a vector that records the AIC values of the regressions
    #we are examining; the number of these regressions is equal
    #with the number of variables that are not in the model
    regAIC = vector('numeric',length(VariablesNotInModel));
    
    #take each variable that is not in the model
    #and include it in the model
    k=1;
    for (var in VariablesNotInModel)
    {
      regAIC[k] <- getLogisticAIC(response,union(bestRegression,var),data);
      k <- k+1;
    }
    if(min(regAIC) >= bestRegressionAIC)
    {
      break;
    }
    else
    {
      bestRegressionAIC <- min(regAIC);
      varToRemove <- VariablesNotInModel[which.min(regAIC)]
      bestRegression <- union(bestRegression, varToRemove);
      VariablesNotInModel <- setdiff(VariablesNotInModel,varToRemove);
    }
  }
  
  return(list(aic=bestRegressionAIC,reg=bestRegression));
}

#HOMEWORK 2, PROBLEM 3: Backward greedy search
backwardSearchAIC <- function(response,data,lastPredictor)
{
  #start with the empty regression with no predictors
  bestRegression = 1:lastPredictor;
  #calculate the AIC of the empty regression
  bestRegressionAIC = getLogisticAIC(response,bestRegression,data);
  cat('\n\n\n\nbackwardSearch :: The full logistic regression has AIC = ',bestRegressionAIC,'\n');
  
  #sequentially delete one variable from the current regression
  #and retain that variable that gives the smallest AIC; make the model
  #in which that variable is deleted to be the current model if
  #this leads to a current model with a smaller AIC
  stepNumber = 0;
  while(length(bestRegression)>=1)
  {
    #record the number of steps performed
    stepNumber = stepNumber + 1;
    
    #create a vector that records the AIC values of the regressions
    #we are examining; the number of these regressions is equal
    #with the number of variables that are not in the model
    regAIC = vector('numeric',length(bestRegression));
    
    #take each variable that is not in the model
    #and include it in the model
    k=1;
    for (var in bestRegression)
    {
      regAIC[k] <- getLogisticAIC(response,setdiff(bestRegression,var),data);
      k <- k+1;
    }
    if(min(regAIC) >= bestRegressionAIC)
    {
      break;
    }
    else
    {
      bestRegressionAIC <- min(regAIC);
      varToRemove <- bestRegression[which.min(regAIC)]
      bestRegression <- setdiff(bestRegression, varToRemove);
    }
  }
  
  return(list(aic=bestRegressionAIC,reg=bestRegression));
}

data <- read.table("/Users/joseph/Courses/stat534/hw2/534binarydata.txt")
lastPredictor=60
response=61
print(forwardSearchAIC(response,data,lastPredictor));
print(backwardSearchAIC(response,data,lastPredictor));



#HW2, Problem 4: Forward and Backward Greedy BIC
forwardSearchBIC <- function(response,data,lastPredictor)
{
  #start with the empty regression with no predictors
  bestRegression = NULL;
  #calculate the BIC of the empty regression
  bestRegressionBIC = getLogisticBIC(response,bestRegression,data);
  cat('\n\n\n\nforwardSearch :: The empty logistic regression has BIC = ',bestRegressionBIC,'\n');
  
  #vector that keeps track of all the variables
  #that are currently NOT in the model
  VariablesNotInModel = 1:lastPredictor;
  
  #add variables one at a time to the current regression
  #and retain that variable that gives the smallest values of BIC associated
  #Make the model that includes that variable to be the current model
  #if it gives a smaller BIC than the BIC of the current regression
  
  #stop when there are no variables that can be included in the model
  stepNumber = 0;
  while(length(VariablesNotInModel)>=1)
  {
    #record the number of steps performed
    stepNumber = stepNumber + 1;
    
    #create a vector that records the BIC values of the regressions
    #we are examining; the number of these regressions is equal
    #with the number of variables that are not in the model
    regBIC = vector('numeric',length(VariablesNotInModel));
    
    #take each variable that is not in the model
    #and include it in the model
    k=1;
    for (var in VariablesNotInModel)
    {
      regBIC[k] <- getLogisticBIC(response,union(bestRegression,var),data);
      k <- k+1;
    }
    if(min(regBIC) >= bestRegressionBIC)
    {
      break;
    }
    else
    {
      bestRegressionBIC <- min(regBIC);
      varToRemove <- VariablesNotInModel[which.min(regBIC)]
      bestRegression <- union(bestRegression, varToRemove);
      VariablesNotInModel <- setdiff(VariablesNotInModel,varToRemove);
    }
  }
  
  return(list(bic=bestRegressionBIC,reg=bestRegression));
}

backwardSearchBIC <- function(response,data,lastPredictor)
{
  #start with the empty regression with no predictors
  bestRegression = 1:lastPredictor;
  #calculate the BIC of the empty regression
  bestRegressionBIC = getLogisticBIC(response,bestRegression,data);
  cat('\n\n\n\nbackwardSearch :: The full logistic regression has BIC = ',bestRegressionBIC,'\n');
  
  #sequentially delete one variable from the current regression
  #and retain that variable that gives the smallest BIC; make the model
  #in which that variable is deleted to be the current model if
  #this leads to a current model with a smaller BIC
  stepNumber = 0;
  while(length(bestRegression)>=1)
  {
    #record the number of steps performed
    stepNumber = stepNumber + 1;
    
    #create a vector that records the BIC values of the regressions
    #we are examining; the number of these regressions is equal
    #with the number of variables that are not in the model
    regBIC = vector('numeric',length(bestRegression));
    
    #take each variable that is not in the model
    #and include it in the model
    k=1;
    for (var in bestRegression)
    {
      regBIC[k] <- getLogisticBIC(response,setdiff(bestRegression,var),data);
      k <- k+1;
    }
    if(min(regBIC) >= bestRegressionBIC)
    {
      break;
    }
    else
    {
      bestRegressionBIC <- min(regBIC);
      varToRemove <- bestRegression[which.min(regBIC)]
      bestRegression <- setdiff(bestRegression, varToRemove);
    }
  }
  
  return(list(BIC=bestRegressionBIC,reg=bestRegression));
}

