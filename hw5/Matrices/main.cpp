#include "matrices.h"

double marglik(int, int, double**, int, int*);

int main(){
int n = 158; //sample size
int p = 51; //number of variables
int i;

int A[] = {2,5,10};//indices of the variables present in the regression
int lenA = 3; //number of indices
char datafilename[] = "erdata.txt";

//allocate the data matrix
double** data = allocmatrix(n,p);

//read the data
readmatrix(datafilename,n,p,data);
printf("Marginal likelihood of regression [1|%d",A[0]);
for(i=1;i<lenA;i++){
        printf(",%d",A[i]);
	}
printf("] = %.5lf\n",marglik(n,p,data,lenA,A));

//free memory
freematrix(n,data);

return(1);
}

double marglik(int n, int p, double** data, int lenA, int* A){
	
	int i,j;
	double answer;

	double** D1 = allocmatrix(n,1);
	for(i=0; i<n; i++){ D1[i][0] = data[i][0]; }
	
	double** DA = allocmatrix(n,lenA);
	for(i=0; i<n; i++){
		for(j=0;j<lenA;j++){ DA[i][j]= data[i][A[j]-1]; }
	}

	double** MA = allocmatrix(lenA,lenA);
	double** MA_inv = allocmatrix(lenA,lenA);
	matrixproduct(lenA, n, lenA, transposematrix(n,lenA,DA), DA, MA);
	for(i=0;i<lenA;i++){ MA[i][i]+=1; } /*add identity by adding 1's on diag*/
	copymatrix(lenA,lenA,MA,MA_inv); 	/*copy MA into MA_inv*/
	inverse(lenA,MA_inv);  		/*now MA_inv stores inverse*/
	
	double one = lgamma((n+lenA+2)/2);
	double two = lgamma((lenA+2)/2);

	double three = (-0.5)*(logdet(lenA,MA));

	double** temp1 = allocmatrix(1,1);
	matrixproduct(1,n,1,transposematrix(n,1,D1),D1,temp1);  /*store D1^T*D1 value in temp1*/

	double** temp2 = allocmatrix(1,lenA);
	matrixproduct(1,n,lenA,transposematrix(n,1,D1),DA,temp2);

	double** temp3 = allocmatrix(1,lenA);
	matrixproduct(1,lenA,lenA,temp2,MA_inv,temp3);

	double** temp4 = allocmatrix(1,n);
	matrixproduct(1,lenA,n,temp3,transposematrix(n,lenA,DA),temp4);

	double** four = allocmatrix(1,1);
	matrixproduct(1,n,1,temp4,D1,four);

	answer = one - two + three + ((-n - lenA-2)/2)*log(1+ temp1[0][0] - four[0][0]);

	freematrix(n,D1);
	freematrix(n,DA);
	freematrix(lenA,MA);
	freematrix(lenA,MA_inv);
	freematrix(1,temp1);
	freematrix(1,temp2);
	freematrix(1,temp3);
	freematrix(1,temp4);
	freematrix(1,four);

	return(answer);
}

