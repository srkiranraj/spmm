/**
 * @author : Hardhik Mallipeddi (mallipeddi.hardhik@research.iiit.ac.in)
 * @author : Kiran Raj (kiran.raj@research.iiit.ac.in)
 *
 * Implementation of sparse_matrix Class.
 * 
 **/

#include "sparse_matrix.h"

#include <stdio.h>
#include <stdlib.h>

sparse_matrix::sparse_matrix(int rows, int cols, int nnz){
	this->rows = rows;
	this->cols = cols;
	this->nnz = nnz;

	this->ir = (int *)malloc((rows + 1) * sizeof(int));
	this->jc = (int *)malloc((nnz) * sizeof(int));
	this->val = (int *)malloc((nnz) * sizeof(int));

	for (int i = 0; i < (rows+1); ++i)
		this->ir[i] = 0;

	for (int i = 0; i < nnz; ++i)
	{
		this->jc[i] = 0;
		this->val[i] = 0;
	}
}

sparse_matrix::~sparse_matrix(){
	printf("Destructor :: Deleting object\n");
}