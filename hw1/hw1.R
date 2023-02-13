logdet <- function(A) {
	ev <- eigen(A)    
	eigenvalues <- ev$values 
	sum(log(eigenvalues))    
}

logmarglik <- function(D,A) {
  D1 <- D[,1]              
  DA <- D[,A]
  MA <- diag(length(A)) + (t(DA)%*%DA)
  n <- length(D1)
  lgamma((n+length(A)+2)/2) - lgamma((length(A)+2)/2) + (-.5)*(logdet(MA)) + ((-n-length(A)-2)/2)*log(1+t(D1)%*%D1-t(D1)%*%DA%*%solve(MA)%*%t(DA)%*%D1)  
}

D <- as.matrix(read.table("/Users/joseph/Courses/stat534/erdata.txt"))
A <- c(2,5,10)
print(logmarglik(D,A))