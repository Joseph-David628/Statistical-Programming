#include "matrices.h"
#include <gsl/gsl_randist.h>
#include<gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_linalg.h>

void makeCovariance(gsl_matrix*, gsl_matrix*);
gsl_matrix* makeCholesky(gsl_matrix*);
void randomMVN(gsl_rng*, gsl_matrix*, gsl_matrix*);

//generate 10000 samples from MVN distribution with covariance equal to
//our data's covariance, and writes the covariance of our samples to file
//also prints covariance of data to file for comparison
int main()
{
  const gsl_rng_type* T;
  gsl_rng* r;
  int i, j;
  double temp;

  int n = 158; //sample size
  int p = 51; //number of variables
  char datafilename[] = "erdata.txt"; //name of the data file

  //initialize RNG object
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc(T);
 
  //allocate matrices
  gsl_matrix* data = gsl_matrix_alloc(n,p);
  gsl_matrix* data_mean_sub = gsl_matrix_alloc(n,p);
  gsl_matrix* samples = gsl_matrix_alloc(10000,p);
  gsl_matrix* temp_sample = gsl_matrix_alloc(p,1);

  //read the data
  FILE* datafile = fopen(datafilename,"r");
  if(NULL==datafile){
  	  fprintf(stderr,"Cannot open data file [%s]\n",datafilename);
	  return(0);
  }
  if(0!=gsl_matrix_fscanf(datafile,data)){
  fprintf(stderr,"File [%s] does not have the required format.\n",datafilename);
  return(0);
  }
  fclose(datafile);

  //find covariance of data
  gsl_matrix* covariance = gsl_matrix_alloc(p,p);
  makeCovariance(covariance,data);

  //generate 10000 samples
  for(i=0;i<10000;i++){
    randomMVN(r,temp_sample,covariance);
    for(j=0;j<p;j++){	    
      gsl_matrix_set(samples,i,j,gsl_matrix_get(temp_sample,j,0));
    }
  }

  //calculate covariance of samples
  gsl_matrix* cov2 = gsl_matrix_alloc(p,p);
  makeCovariance(cov2,samples);
  
  //print covariance to compare results
  char filename2[] = "covariance_matrix.txt";
  printmatrix(filename2,covariance);

  //write results to file
  char filename[] = "results_matrix.txt";
  printmatrix(filename, cov2);

  gsl_rng_free(r);
  gsl_matrix_free(data);
  gsl_matrix_free(covariance);
  gsl_matrix_free(cov2);
  gsl_matrix_free(samples);

  return(1);

}

//inputs: r = initialized GSL_RNG object
//samples = sample from MVN distribution
//covariance = covariance matrix
//writes in samples a single sample from MVN with given covariance 
void randomMVN(gsl_rng* r, gsl_matrix* samples, gsl_matrix* covariance)
{
  int i,j;
  int p = covariance->size1;

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

  //deallocate
  gsl_matrix_free(cholesky);
  gsl_matrix_free(Z);
}

//computes covariance of X and stores in covX
void makeCovariance(gsl_matrix* covX, gsl_matrix* X)
{
  int i,j,temp;
  int n = X -> size1;
  int p = X -> size2;

  gsl_matrix* data_normal = gsl_matrix_alloc(n,p);

  double means[p];

  for(j=0;j<p;j++){
    temp=0;
    for(i=0;i<n;i++){
      temp += gsl_matrix_get(X,i,j);
    }
    means[j] = temp / n;
  }

  //subtract column mean from data
  for(i=0;i<n;i++){
    for(j=0;j<p;j++){
      gsl_matrix_set(data_normal,i,j,gsl_matrix_get(X,i,j)-means[j]);
    }
  }

  matrixproduct(transposematrix(data_normal),data_normal,covX);
  for(i=0;i<p;i++){
    for(j=0;j<p;j++){
      gsl_matrix_set(covX,i,j,gsl_matrix_get(covX,i,j)/n);
    }
  }

  gsl_matrix_free(data_normal);
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


