#include "matrices.h"

double getDeterminant(double**, int);

int main(){
	int n = 10;
	int j;
	double determinant = 0;
	char filename[] = "mybandedmatrix.txt";
	double** A = allocmatrix(n,n);
	readmatrix(filename,n,n,A);

	determinant = getDeterminant(A,n);

	freematrix(n,A);

	printf("Determinant of Matrix = %.4lf \n",determinant);
	
}

/* recursively calculates the determinant of square n matrix A using minors */
double getDeterminant(double** A, int n){
	if(n==1){ return A[0][0]; }
	if(n==2){ return (A[0][0]*A[1][1] - A[0][1]*A[1][0]); }

	int i, k, j;
	double det=0;
	double** minor = allocmatrix(n-1,n-1);
	for(j=0;j<n;j++){
		for(k=0;k<n-1;k++){
			for(i=0;i<j;i++){ minor[k][i] = A[k+1][i]; }
			for(i=j;i<n-1;i++){ minor[k][i] = A[k+1][i+1]; }
		}
		det += A[0][j] * (pow(-1,j)) * getDeterminant(minor,n-1);
	}
	return det;
}
