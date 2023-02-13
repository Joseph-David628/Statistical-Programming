//STAT534 Final Project
//Author: Joseph David

#include "matrices.h"
#include <mpi.h>

#define GETRESULTS	1
#define SHUTDOWNTAG	0

static int myrank;

int n = 148;
int p = 60;

void primary();
void replica(int primaryname);
gsl_matrix* getCoefNR(int, int, gsl_matrix*);
gsl_matrix* inverseLogit(gsl_matrix* x);
gsl_matrix* inverseLogit2(gsl_matrix* x);
gsl_matrix* getPi(gsl_matrix* x, gsl_matrix* beta);
gsl_matrix* getPi2(gsl_matrix* x, gsl_matrix* beta);
double logisticLoglik(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta);
double logstar(gsl_matrix* y, gsl_matrix*, gsl_matrix*);
gsl_matrix* getGradient(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta);
gsl_matrix* getHessian(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta);
double getLaplaceApprox(int, int, gsl_matrix*, gsl_matrix*);
double getMonteCarlo(int, int, gsl_matrix*, int niter, gsl_rng*);
gsl_matrix* makeCholesky(gsl_matrix*);
gsl_matrix* randomMVN(gsl_rng*, gsl_matrix*);
gsl_matrix* getPosteriorMeans(int, int, gsl_matrix*, gsl_matrix*, int, gsl_rng* r);


int main(int argc, char* argv[])
{
   MPI_Init(&argc, &argv);

   //Get ID
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   if(myrank==0)
   {
      primary();
   }
   else
   {
      replica(myrank);
   }

   MPI_Finalize();

   return 0;
}

void primary()
{
   double results[5][5];   //list to store results
   double temp[5];
   double temp2[5];
   int i,j,k;
   int var;
   int rank;
   int ntasks;
   int jobsRunning;
   int work[1];
   double workresults[5];
   FILE* fout;
   MPI_Status status;

   fout = fopen("results.txt","w");

   MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

   fprintf(stdout, "Total Number of processors = %d\n",ntasks);

   jobsRunning = 1;

   for(i=0;i<5;i++){results[i][2]=-1000;}

   for(var=0; var<p; var++)
   {
      work[0] = var;
      
      if(jobsRunning < ntasks)
      {
         MPI_Send(&work, 1, MPI_INT, jobsRunning, GETRESULTS, MPI_COMM_WORLD);
         printf("Primary sends out work request [%d] to replica [%d]\n",
                work[0],jobsRunning);
         jobsRunning++;
      }
      else
      {
         MPI_Recv(workresults, 5, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
         
         printf("Primary has received the result of work request [%d] from replica [%d]\n",
                (int) workresults[0],status.MPI_SOURCE);

         //deteremine whether to store in results
         for(i=0;i<5;i++)
         {
            //shift list if smaller logmarglik is found
            if(workresults[2] > results[i][2]){
               for(k=0;k<5;k++){
                  temp[k] = results[i][k];   	//copy results
                  results[i][k] = workresults[k];	//write in new results
               }
               for(j=i;j<4;j++){
                  for(k=0;k<5;k++){
                     temp2[k] = results[j+1][k];
                     results[j+1][k] = temp[k];
                     temp[k] = temp2[k];
                  }
               }
               break;  //if we added to results, break out of for loop
            }
         }

         printf("Primary sends out work request [%d] to replica [%d]\n",
                work[0],status.MPI_SOURCE);

         MPI_Send(&work,1,MPI_INT,status.MPI_SOURCE,GETRESULTS,MPI_COMM_WORLD); 
      }
   }

   //Collect extra results now

   for(rank=1; rank<ntasks; rank++)
   {
      MPI_Recv(workresults,5,MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
      printf("Primary has received the result of work request [%d]\n",
                (int) workresults[0]);
      
      //deteremine whether to store in results
      for(i=0;i<5;i++)
      {
         //shift list if larger logmarglik is found
         if(workresults[2] > results[i][2]){
            for(k=0;k<5;k++){
               temp[k] = results[i][k];   	//copy results
               results[i][k] = workresults[k];	//write in new results
         }
         for(j=i;j<4;j++){
               for(k=0;k<5;k++){
                  temp2[k] = results[j+1][k];
                  results[j+1][k] = temp[k];
                  temp[k] = temp2[k];
               }
            }
            break;  //if we added to results, break out of for loop
         }
      }
   }

   for(i=0;i<5;i++)
   {
      fprintf(fout, "%d\t%f\t%f\t%f\t%f\n", int(results[i][0]), results[i][1], results[i][2], results[i][3], results[i][4]);
   }
   printf("Print results to file.\n");

   printf("Tell the replicas to shutdown.\n");
   // Shut down the replica processes
   for(rank=1; rank<ntasks; rank++)
   {
      printf("Primary is shutting down replica [%d]\n",rank);
      MPI_Send(0,
	            0,
               MPI_INT,
               rank,		// shutdown this particular node
               SHUTDOWNTAG,		// tell it to shutdown
	       MPI_COMM_WORLD);
   }

   printf("Got to the end of Primary code\n");

   fclose(fout);

   // return to the main function
   return;
}

void replica(int replicaname)
{
   //random seeding
   const gsl_rng_type* T;
   gsl_rng* r;

   //initialize rng object
   gsl_rng_env_setup();
   T = gsl_rng_default;
   r = gsl_rng_alloc(T);

   int i,j,k;
   int work[1];   //input from primary
   double workresults[5];   //output for primary
   MPI_Status status;
   //file where data is stored
   char datafilename[] = "534finalprojectdata.txt";

   //allocate and initialize data matrix
   gsl_matrix* data = gsl_matrix_alloc(n,p+1);
   //read the data
   FILE* datafile = fopen(datafilename,"r");
   if(NULL==datafile){
	fprintf(stderr,"Cannot open data file [%s]\n",datafilename);
        return;
   }
   if(0!=gsl_matrix_fscanf(datafile,data)){
   fprintf(stderr,"File [%s] does not have the required format.\n",datafilename);
   return;
   }
   fclose(datafile);

   int notDone = 1;
   while(notDone)
   {
      printf("Replica %d is waiting\n",replicaname);
      MPI_Recv(&work, // the input from primary
	       1,		// the size of the input
	       MPI_INT,		// the type of the input
               0,		// from the PRIMARY node (rank=0)
               MPI_ANY_TAG,	// any type of order is fine
               MPI_COMM_WORLD,
               &status);
      printf("Replica %d just received smth\n",replicaname);
   

   //switch on type of work request
   switch(status.MPI_TAG)
   {
      case GETRESULTS:{
         workresults[0] = (double)work[0];   //first result is variable number
	 gsl_matrix* betaMode = gsl_matrix_alloc(2,1); //beta mode using getCoefNR

         //find beta mode,
         betaMode = getCoefNR(p,work[0],data);
         //printf("Calculated a beta mode in replica %d \n",replicaname);
         //compute Laplace approximation and store in workresults
         workresults[1] = getLaplaceApprox(p,work[0],data,betaMode);
	 //printf("Calculated Laplace approx in replica %d = %f\n",replicaname,workresults[1]);         
         //compute Monte Carlo Integration Approx with 10000 samples
         workresults[2] = getMonteCarlo(p, work[0], data, 10000, r);
	 //printf("Calculated Monte carlo for replica %d\n",replicaname);

         //generate betaMode approx using Metropolis Hastings, 10000 iterations
         gsl_matrix* temp = gsl_matrix_alloc(2,1);
         temp = getPosteriorMeans(p, work[0], data, betaMode, 10000, r);
	 workresults[3] = gsl_matrix_get(temp,0,0);
         workresults[4] = gsl_matrix_get(temp,1,0);
         //printf("Calculated posterior means in replica %d = %f, %f\n",replicaname,workresults[3],workresults[4]);

         gsl_matrix_free(betaMode);
         gsl_matrix_free(temp);

	 MPI_Send(&workresults,5,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
	 printf("replica %d finished processing work request [%d]\n",replicaname,work[0]);

	 break;
       }
       case SHUTDOWNTAG:{
          printf("Replica %d was told to shutdown.\n", replicaname);
          return;
       }
       default:{
          notDone=0;
          printf("The replica code shouldn't be here.\n");
          return;
       }   
   }
   }
   gsl_rng_free(r);
   return;
}


//function definitions

gsl_matrix* getCoefNR(int response, int explanatory, gsl_matrix* data)
{
   int i,j;
   //int counter = 0;

   //declare and initialize beta as (0,0), get log likelihood
   gsl_matrix* beta = gsl_matrix_alloc(2,1);
   gsl_matrix_set(beta,0,0,0);
   gsl_matrix_set(beta,1,0,0);

   //declare explanatory and response variables
   gsl_matrix* y = gsl_matrix_alloc(n,1);
   gsl_matrix* x = gsl_matrix_alloc(n,1);

   //initialize
   for(i=0;i<n;i++)
   {
      gsl_matrix_set(y,i,0,gsl_matrix_get(data,i,response));
      gsl_matrix_set(x,i,0,gsl_matrix_get(data,i,explanatory));
   }
   
   double currentLoglik = logisticLoglik(y,x,beta);
   double newLoglik;

   //declare new iteration beta
   gsl_matrix* beta_new = gsl_matrix_alloc(2,1);

   //declare matrices for calculating Hessian
   gsl_matrix* hessian = gsl_matrix_alloc(2,2);
   gsl_matrix* inv_Hess = gsl_matrix_alloc(2,2);

   //run newton-raphson until change is small
   while(1)
   {   
      //calculate new beta
      hessian = getHessian(y,x,beta);
      inv_Hess = inverse(hessian);
      //printf("inv Hess = %f,%f,%f,%f\n",gsl_matrix_get(inv_Hess,0,0),gsl_matrix_get(inv_Hess,1,0),gsl_matrix_get(inv_Hess,0,1),gsl_matrix_get(inv_Hess,1,1));
      matrixproduct(inv_Hess,getGradient(y,x,beta),beta_new);
      gsl_matrix_set(beta_new,0,0,gsl_matrix_get(beta,0,0)-gsl_matrix_get(beta_new,0,0));
      gsl_matrix_set(beta_new,1,0,gsl_matrix_get(beta,1,0)-gsl_matrix_get(beta_new,1,0));
         
      newLoglik = logisticLoglik(y,x,beta_new);
      //if(counter%2==0){printf("The newLoglik = %f\n",newLoglik);}
      //counter++;

      if(newLoglik<currentLoglik)
      {
         printf("CODING ERROR!!\n");
         break;
      }

      gsl_matrix_set(beta,0,0,gsl_matrix_get(beta_new,0,0));
      gsl_matrix_set(beta,1,0,gsl_matrix_get(beta_new,1,0));
      //exit loop if new beta does not change much
      if (newLoglik - currentLoglik < .000001)
      {
         break;
      }
      //if not, run another iteration
      currentLoglik = newLoglik;
   }

   //free matrices
   gsl_matrix_free(y);
   gsl_matrix_free(x);
   gsl_matrix_free(hessian);
   gsl_matrix_free(inv_Hess);
   gsl_matrix_free(beta_new);
   //printf("Calculated betaMode in Newton-Raphson, beta0 = %f and beta1 = %f\n",gsl_matrix_get(beta,0,0),gsl_matrix_get(beta,1,0));
   //printf("NR ran for %d iteration\n",counter);
   return(beta);
}

gsl_matrix* inverseLogit(gsl_matrix* x)
{
   int i;
   gsl_matrix* result = gsl_matrix_alloc(n,1);
   for(i=0;i<n;i++)
   {
      gsl_matrix_set(result,i,0,exp(gsl_matrix_get(x,i,0))/(1+exp(gsl_matrix_get(x,i,0))));
   }
   return(result);
}

gsl_matrix* inverseLogit2(gsl_matrix* x)
{
   int i;
   gsl_matrix* result = gsl_matrix_alloc(n,1);
   for(i=0;i<n;i++)
   {   
      gsl_matrix_set(result,i,0,exp(gsl_matrix_get(x,i,0))/pow(1+exp(gsl_matrix_get(x,i,0)),2));
   }
   return(result);
}

gsl_matrix* getPi(gsl_matrix* x, gsl_matrix* beta)
{
   int i,j;
   gsl_matrix* x0 = gsl_matrix_alloc(n,2);
   gsl_matrix* result = gsl_matrix_alloc(n,1);

   //initialize x0 as column of 1s then x column
   for(i=0;i<n;i++)
   {
      gsl_matrix_set(x0,i,0,1);
      gsl_matrix_set(x0,i,1,gsl_matrix_get(x,i,0));
   }
   matrixproduct(x0, beta, result);
   result = inverseLogit(result);

   gsl_matrix_free(x0);

   return(result);
}

gsl_matrix* getPi2(gsl_matrix* x, gsl_matrix* beta)
{
   int i,j;
   double temp;
   gsl_matrix* x0 = gsl_matrix_alloc(n,2);
   gsl_matrix* result = gsl_matrix_alloc(n,1);

   //initialize x0 as column of 1s then x column
   for(i=0;i<n;i++)
  {
      gsl_matrix_set(x0,i,0,1);
      temp = gsl_matrix_get(x,i,0);
      gsl_matrix_set(x0,i,1,temp);
   }
   matrixproduct(x0, beta, result);
   result = inverseLogit2(result);

   gsl_matrix_free(x0);

   return(result);
}

double logisticLoglik(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta)
{ 
   int i;
   double answer = 0;
   gsl_matrix* Pi = gsl_matrix_alloc(n,1);
   Pi = getPi(x, beta);
   for(i=0;i<n;i++)
   {
      answer += (gsl_matrix_get(y,i,0)*log(gsl_matrix_get(Pi,i,0))) + ((1-gsl_matrix_get(y,i,0))*log(1-gsl_matrix_get(Pi,i,0)));
   }

   gsl_matrix_free(Pi);

   return(answer);
}

double logstar(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta)
{
  return(-log(6.28)-0.5*(pow(gsl_matrix_get(beta,0,0),2)+pow(gsl_matrix_get(beta,1,0),2))+logisticLoglik(y,x,beta));
}

gsl_matrix* getGradient(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta)
{
   int i;
   gsl_matrix* gradient = gsl_matrix_alloc(2,1);
   gsl_matrix* Pi = gsl_matrix_alloc(n,1);

   Pi = getPi(x,beta);
   gsl_matrix_set(gradient,0,0,0);
   gsl_matrix_set(gradient,1,0,0);
   
   for(i=0;i<n;i++)
   {
      gsl_matrix_set(gradient,0,0,gsl_matrix_get(gradient,0,0)+gsl_matrix_get(y,i,0) - gsl_matrix_get(Pi,i,0));
      gsl_matrix_set(gradient,1,0,gsl_matrix_get(gradient,1,0)+(gsl_matrix_get(y,i,0) - gsl_matrix_get(Pi,i,0))*gsl_matrix_get(x,i,0));
   }

   gsl_matrix_free(Pi);

   return(gradient);
}

gsl_matrix* getHessian(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta)
{
   int i;
   gsl_matrix* hessian = gsl_matrix_alloc(2,2);
   gsl_matrix_set(hessian,0,0,0);
   gsl_matrix_set(hessian,0,1,0);
   gsl_matrix_set(hessian,1,0,0);
   gsl_matrix_set(hessian,1,1,0);
   gsl_matrix* Pi2 = gsl_matrix_alloc(n,1);

   Pi2 = getPi2(x,beta);

   for(i=0;i<n;i++)
   {
      gsl_matrix_set(hessian,0,0,gsl_matrix_get(hessian,0,0)-gsl_matrix_get(Pi2,i,0));
      gsl_matrix_set(hessian,0,1,gsl_matrix_get(hessian,0,1)-(gsl_matrix_get(Pi2,i,0)*gsl_matrix_get(x,i,0)));
      gsl_matrix_set(hessian,1,1,gsl_matrix_get(hessian,1,1)-(gsl_matrix_get(Pi2,i,0)*pow(gsl_matrix_get(x,i,0),2)));
   }
   gsl_matrix_set(hessian,1,0,gsl_matrix_get(hessian,0,1));

   gsl_matrix_free(Pi2);

   return(hessian);
}

double getLaplaceApprox(int response,int explanatory,gsl_matrix* data,gsl_matrix* betaMode)
{
   int i;
   gsl_matrix* y = gsl_matrix_alloc(n,1);
   gsl_matrix* x = gsl_matrix_alloc(n,1);

   for(i=0;i<n;i++)
   {
      gsl_matrix_set(y,i,0,gsl_matrix_get(data,i,response));
      gsl_matrix_set(x,i,0,gsl_matrix_get(data,i,explanatory));
   }

   double logLik = logisticLoglik(y,x,betaMode);

   gsl_matrix* hessian = getHessian(y,x,betaMode);
   gsl_matrix_set(hessian,0,0,-gsl_matrix_get(hessian,0,0));
   gsl_matrix_set(hessian,0,1,-gsl_matrix_get(hessian,0,1));
   gsl_matrix_set(hessian,1,0,-gsl_matrix_get(hessian,1,0));
   gsl_matrix_set(hessian,1,1,-gsl_matrix_get(hessian,1,1));

   double answer = (-0.5)*(pow(gsl_matrix_get(betaMode,0,0),2)+pow(gsl_matrix_get(betaMode,1,0),2))+logLik-(0.5)*logdet(hessian);

   gsl_matrix_free(x);
   gsl_matrix_free(y);

   return(answer);
}

double getMonteCarlo(int response, int explanatory, gsl_matrix* data, int niter, gsl_rng* r)
{
   int i;
   double answer = 0;
   gsl_matrix* sample = gsl_matrix_alloc(2,1);

   gsl_matrix* y = gsl_matrix_alloc(n,1);
   gsl_matrix* x = gsl_matrix_alloc(n,1);
   for(i=0;i<n;i++)
   {
      gsl_matrix_set(y,i,0,gsl_matrix_get(data,i,response));
      gsl_matrix_set(x,i,0,gsl_matrix_get(data,i,explanatory));
   }

   //make identity (which equals covariance) matrix
   gsl_matrix* eye = gsl_matrix_alloc(2,2);
   gsl_matrix_set_identity(eye);

   for(i=0;i<niter;i++)
   {
      answer += exp(logisticLoglik(y,x,randomMVN(r,eye)));
   }
   return(log(answer/niter));   
}

//inputs: r = initialized GSL_RNG object
//samples = sample from MVN distribution
//covariance = covariance matrix
//writes (in samples) a single sample from MVN with given covariance 
gsl_matrix* randomMVN(gsl_rng* r, gsl_matrix* covariance)
{
  int i,j;
  int p = covariance->size1;
  gsl_matrix* result = gsl_matrix_alloc(p,1);
  gsl_matrix* samples = gsl_matrix_alloc(p,1);

  //find cholesky matrix
  gsl_matrix* cholesky = gsl_matrix_alloc(p,p);
  cholesky = makeCholesky(covariance);

  //declare random guassian number matrix Z
  gsl_matrix* Z = gsl_matrix_alloc(p,1);

  //initialize Z with samples from N(0,1)
  for(i=0;i<p;i++){
    gsl_matrix_set(Z,i,0,gsl_ran_ugaussian(r));
  }

  matrixproduct(cholesky,Z,samples);

  //convert sample to regular matrix
  for(i=0;i<p;i++)
  {
      gsl_matrix_set(result,i,0,gsl_matrix_get(samples,i,0));
  }

  //deallocate
  gsl_matrix_free(samples);
  gsl_matrix_free(cholesky);
  gsl_matrix_free(Z);

  return(result);
}

//returns the lower triangular Cholesky matrix of K
gsl_matrix* makeCholesky(gsl_matrix* K)
{
  int i,j;
  int p = K->size1;
  gsl_matrix* cholesky = gsl_matrix_alloc(p,p);
  gsl_matrix_memcpy(cholesky,K);
  gsl_linalg_cholesky_decomp1(cholesky);
  for(i=0;i<p-1;i++){
    for(j=i+1;j<p;j++){
      gsl_matrix_set(cholesky,i,j,0);
    }
  }
  return(cholesky); 
}

gsl_matrix* getPosteriorMeans(int response, int explanatory, gsl_matrix* data, gsl_matrix* betaMode, int niter, gsl_rng* r)
{
   int i,j;
   double u;  

   gsl_matrix* y = gsl_matrix_alloc(n,1);
   gsl_matrix* x = gsl_matrix_alloc(n,1);

   for(i=0;i<n;i++)
   {
      gsl_matrix_set(y,i,0,gsl_matrix_get(data,i,response));
      gsl_matrix_set(x,i,0,gsl_matrix_get(data,i,explanatory));
   }

   gsl_matrix* current_beta = gsl_matrix_alloc(2,1);
   current_beta = betaMode;
   //printf("betaMode0 = %f and betaMode1 = %f\n",gsl_matrix_get(betaMode,0,0),gsl_matrix_get(betaMode,1,0));
   double current_logLik = logstar(y,x,current_beta);
 
   //store mean of beta samples
   gsl_matrix* result = gsl_matrix_alloc(2,1);
   gsl_matrix_set(result,0,0,0);
   gsl_matrix_set(result,1,0,0);

   gsl_matrix* new_beta = gsl_matrix_alloc(2,1);
   double new_logLik;

   //covariance matrix
   //printf("Calculating covariance mat in PosteriorMeans.\n");
   gsl_matrix* cov = gsl_matrix_alloc(2,2);
   cov = getHessian(y,x,betaMode);
   cov = inverse(cov);
   gsl_matrix_set(cov,0,0,-gsl_matrix_get(cov,0,0));
   gsl_matrix_set(cov,1,0,-gsl_matrix_get(cov,1,0));
   gsl_matrix_set(cov,0,1,-gsl_matrix_get(cov,0,1));
   gsl_matrix_set(cov,1,1,-gsl_matrix_get(cov,1,1));

   for(i=0;i<niter;i++)
   {
      //sample from N(0,cov), then add current_beta to sample from N(current_beta,cov)
      new_beta = randomMVN(r, cov);
      gsl_matrix_set(new_beta,0,0, gsl_matrix_get(new_beta,0,0) + gsl_matrix_get(current_beta,0,0) );
      gsl_matrix_set(new_beta,1,0, gsl_matrix_get(new_beta,1,0) + gsl_matrix_get(current_beta,1,0) );
      new_logLik = logstar(y,x,new_beta);
      //if(i % 500 == 0){printf("currentLoglik = %f and newLoglik = %f\n",current_logLik,new_logLik);}
      //determine whether to accept proposed state
      if(new_logLik >= current_logLik)
      {
         current_beta = new_beta;
         current_logLik = new_logLik;
	
      }
      else
      { 
         //generate sample for Unif(0,1)
         u = gsl_ran_flat(r,0,1);

         if(log(u) <= new_logLik - current_logLik )
         {
            current_beta = new_beta;
            current_logLik = new_logLik;
         }
      }
      //add current_beta to result
      gsl_matrix_set(result,0,0, gsl_matrix_get(result,0,0)+gsl_matrix_get(current_beta,0,0) );
      gsl_matrix_set(result,1,0, gsl_matrix_get(result,1,0)+gsl_matrix_get(current_beta,1,0) );
   }
   //printf("finished sampling in PosteriorMeans.\n");   
   //gsl_matrix_free(y);
   //gsl_matrix_free(x);
   //gsl_matrix_free(current_beta);
   //gsl_matrix_free(new_beta);
   //gsl_matrix_free(cov);
   
   gsl_matrix_set(result,0,0,gsl_matrix_get(result,0,0)/niter);
   gsl_matrix_set(result,1,0,gsl_matrix_get(result,1,0)/niter);

   return(result);
}








