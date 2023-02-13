/*
 FILE: MAIN.CPP

 This program creates a linked list with all the regressions with 1 or 2 predictors.
 This list keeps the regressions in decreasing order of their marginal likelihoods.
*/

#include<stdio.h>
#include "matrices.h"
#include "regmodels.h"
#include<gsl/gsl_matrix.h>

double marglik(gsl_matrix*, int, int*);

int main()
{
  int i,j;

  int nMaxRegs = 10; //number of regressions to keep
  int n = 158; //sample size
  int p = 51; //number of variables
  char datafilename[] = "erdata.txt"; //name of the data file
  char outputfilename[] = "regressions1-2.txt";

  //allocate the data matrix
  gsl_matrix* data = gsl_matrix_alloc(n,p);

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

  //create the head of the list of regressions
  LPRegression regressions = new Regression;
  //properly mark the end of the list
  regressions->Next = NULL;

  int A[p-1]; //indices of the variables present in the regression
  int lenA = -1; //number of indices
  
  //add the regressions with one predictor
  lenA = 1;
  for(i=1;i<p;i++)
  {
    A[0] = i+1;
    AddRegression(nMaxRegs, regressions,
                  lenA, A,
                  marglik(data,lenA,(int*)A));
  }

  //add regressions with two predictors
  lenA=2;
  for(i=1;i<p-1;i++)
  {
    A[0]=i+1;
    for(j=i+1;j<p;j++){
      A[1]=j+1;
      AddRegression(nMaxRegs, regressions,
      lenA, A,
      marglik(data,lenA,(int*)A));
    }
  }

  //save the list in a file
  SaveRegressions(outputfilename,regressions);

  //delete all regressions
  DeleteAllRegressions(regressions);

  //free memory
  gsl_matrix_free(data);
  delete regressions; regressions = NULL;

  return(1);
}

double marglik(gsl_matrix* data, int lenA, int* A){
	int i,j;
	double answer;

	int n = data->size1;
	int p = data->size2;

	gsl_matrix* D1 = gsl_matrix_alloc(n,1);
	for(i=0; i<n; i++){ gsl_matrix_set(D1,i,0,gsl_matrix_get(data,i,0)); }

	gsl_matrix* DA = gsl_matrix_alloc(n,lenA);
	for(j=0;j<lenA;j++){
		for(i=0;i<n;i++){
			gsl_matrix_set(DA,i,j,gsl_matrix_get(data,i,A[j]-1));
		}
	}

	gsl_matrix* MA = gsl_matrix_alloc(lenA,lenA);
	gsl_matrix* MA_inv = gsl_matrix_alloc(lenA,lenA);

	matrixproduct(transposematrix(DA),DA,MA);
	for(i=0;i<lenA;i++){ gsl_matrix_set(MA,i,i,gsl_matrix_get(MA,i,i)+1); }
	MA_inv = inverse(MA);

	double one = lgamma((n+lenA+2)/2);
	double two = lgamma((lenA+2)/2);

	double three = (-0.5)*(logdet(MA));

	gsl_matrix* temp1 = gsl_matrix_alloc(1,1);
	matrixproduct(transposematrix(D1),D1,temp1); 

	gsl_matrix* temp2 = gsl_matrix_alloc(1,lenA);
	matrixproduct(transposematrix(D1),DA,temp2);

	gsl_matrix* temp3 = gsl_matrix_alloc(1,lenA);
	matrixproduct(temp2,MA_inv,temp3);

	gsl_matrix* temp4 = gsl_matrix_alloc(1,n);
	matrixproduct(temp3,transposematrix(DA),temp4);

	gsl_matrix* four = gsl_matrix_alloc(1,1);
	matrixproduct(temp4,D1,four);

	answer = one - two + three + ((-n-lenA-2)/2)*log(1 + gsl_matrix_get(temp1,0,0) - gsl_matrix_get(four,0,0));

	gsl_matrix_free(D1);
	gsl_matrix_free(DA);
	gsl_matrix_free(MA);
	gsl_matrix_free(MA_inv);
	gsl_matrix_free(temp1);
	gsl_matrix_free(temp2);
	gsl_matrix_free(temp3);
	gsl_matrix_free(temp4);
	gsl_matrix_free(four);

	return(answer);

}
