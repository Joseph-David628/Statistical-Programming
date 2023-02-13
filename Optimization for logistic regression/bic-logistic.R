#this function uses 'glm' to fit a logistic regression
#and returns the BIC = deviance + log(SampleSize)*NumberOfCoefficients 
#please note that I changed the function a bit to account for the case
#when there are no predictors
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
  return(deviance+log(nrow(data))*(1+length(explanatory)));
}


#this function generates all the logistic regressions with
#M explanatory variables and returns the regression with
#the minimum BIC
findBestLogisticBIC <- function(response,data,M,lastPredictor)
{
  #index of all the predictors
  allpred = 1:lastPredictor;
  #generate all combinations of allpred taken M at a time
  allPredictorSets = combn(allpred,M);
  
  #fit the first regression
  bestBIC = getLogisticBIC(response,allPredictorSets[,1],data);
  bestReg = allPredictorSets[,1];
  
  if(ncol(allPredictorSets)>=2)
  {
    for(j in 2:ncol(allPredictorSets))
    {
      currentBIC = getLogisticBIC(response,allPredictorSets[,j],data);
      #cat('Processing regression ',j+1,' out of ',ncol(allPredictorSets),'\n');
      if(currentBIC<bestBIC)
      {
        bestBIC = currentBIC;
        bestReg = allPredictorSets[,j];
      }
    }
  }
  
  return(list(bic=bestBIC,reg=bestReg));
}

#we structure the R code as a C program; this is a choice, not a must
main <- function(datafile,nMaxNumberExplanatory)
{
  #read the data
  data = read.table(datafile,header=FALSE);

  #the sample size is 148 (number of rows)
  #the explanatory variables are the first 10 columns
  #the last column is the binary response
  response = ncol(data);
  lastPredictor = ncol(data)-1;

  #we cannot ask for more predictors than the total
  #number of explanatory variables
  nMaxPred = min(nMaxNumberExplanatory,lastPredictor);

  #create a vector to record the smallest bic of
  #regressions with the same number of predictors
  minBIC = vector('numeric',nMaxPred);
  #create a list to record the explanatory variables
  #of the regression for which the smallest BIC is attained
  minRegression = vector('list',nMaxPred);
  
  #determine the best regression of each size up to nMaxPred
  for(M in 1:nMaxPred)
  {
    res =  findBestLogisticBIC(response,data,M,lastPredictor);
    minBIC[M] = res$bic;
    minRegression[M][[1]] = res$reg;
    cat('Best regression with ',M,' predictors has BIC = ',minBIC[M],' and predictors [');
    for(i in 1:length(minRegression[M][[1]])) cat(' ',as.numeric(minRegression[M][[1]][i]));
    cat(']\n');
  }
  
  j = which.min(minBIC);
  cat('\n\n Best regression has ',j, 'predictors [');
  for(i in 1:length(minRegression[j][[1]])) cat(' ',as.numeric(minRegression[j][[1]][i]));
  cat(']\n');
}

#this is equivalent to running your executable file at the command prompt
#if you would have been writing C code
main('534binarydatasmall.txt',10);