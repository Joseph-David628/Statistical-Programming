/*
 Compile with

 mpic++ firstmpi.cpp -o firstmpi

 Run on 5 nodes with

 mpirun -np 5 -hostfile machines.txt firstmpi

*/

#include "mpi.h"
#include <stdio.h>

int main(int argc,char **argv)
{
   int rank, size;

   //START MPI SESSION
   MPI_Init(&argc,&argv );

   //WHAT IS THE ID OF THE PROCESS?
   MPI_Comm_rank( MPI_COMM_WORLD, &rank);

   //FIND OUT HOW MANY PROCESSES ARE AVAILABLE
   MPI_Comm_size( MPI_COMM_WORLD, &size);

   //PRINT OUT THE INFO TEN TIMES
   while(1)
   {
     printf( "Hello world! I'm %d of %d\n",rank, size );
   }

   //FINALIZE THE MPI SESSION
   MPI_Finalize();

   //end the program
   return 0;
}


