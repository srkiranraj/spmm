/**
 * @author : Hardhik Mallipeddi (mallipeddi.hardhik@research.iiit.ac.in)
 * @author : Kiran Raj (kiran.raj@research.iiit.ac.in)
 *
 * COMPILE : sh compile.sh 
 *
 * RUN : ./a.out
 * 
 **/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "sparse_matrix.h"

int main()
{

	sparse_matrix A = sparse_matrix(10, 10, 42);
	sparse_matrix B = sparse_matrix(10, 10, 80);

	return 0;
}