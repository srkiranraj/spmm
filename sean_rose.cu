/**
 * @author : Sean Rose (http://www.cs.fsu.edu/research/projects/rose_report.pdf)
 * 
 * Implementation of SPMM CUDA Kernels
 *
 */

__global__ void GetNNZ(sparse_matrix A, sparse_matrix B, sparse_matrix C, int* workingSet)
{
	const int laneId = threadIdx.x;
	const int warpId = blockIdx.x;
	
	int* nonzeros;
	int rowAStart, rowAEnd, rowBStart, rowBEnd;
	int nnz;
	int colC;
	
	extern __shared__ int nzCount[];
	
	nonzeros = &workingSet[warpId * B.cols];
	
	// Iterate through each assigned row in A.
	for(int rowA = warpId; rowA < A.rows; rowA += gridDim.x)
	{
		rowAStart = A.ir[rowA];
		rowAEnd = A.ir[rowA + 1];
		// There are no non-zeros in this row so continue
		if(rowAStart == rowAEnd)
		{
			if (laneId == 0)
				C.ir[rowA] = 0;
			__syncthreads();
			continue;
		}

		// Reset the nz counts
		nzCount[laneId] = 0;
		
		// reset the nonzeros table
		for (int i=laneId; i<B.cols; i+= warpSize)
		{
			nonzeros[i] = 0;
		}
		__syncthreads();
		
		for(int i = rowAStart; i < rowAEnd; ++i)
		{
			rowBStart = B.ir[A.jc[i]];
			rowBEnd = B.ir[A.jc[i]+1];

			for (int j = rowBStart + laneId; j < rowBEnd; j += warpSize)
			{
				colC = B.jc[j];
				nzCount[laneId] += nonzeros[colC] == 0;
				nonzeros[colC] = 1;
			}
			__syncthreads();
		}

		if(laneId == 0)
		{
			nnz = nzCount[0];
			for(int i = 1; i < warpSize; ++i)
			{
				nnz += nzCount[i];
			}
			C.ir[rowA] = nnz;

		}
		
		__syncthreads();
	}
}

__global__ void GetVals(sparse_matrix A, sparse_matrix B, sparse_matrix C, int* indexTable)
{
	const int laneId = threadIdx.x;
	const int bloackId = blockIdx.x;
	
	__shared__ unsigned int back;
	
	int rowAStart; // The index into A.jc and A.val
	int rowAEnd; // The boundary index for A
	float valA; // The value of the current A nonzero
	int rowBStart; // The index into B.jc and B.val
	int rowBEnd; // The boundary index for B
	int colB; // The current column in B being used
	int rowCStart; // The index into C.jc and C.val
	int rowCEnd; // The boundary index for C
	int hash; // The calculated hash value
	int i, j; // Loop iterators

	// Set the global hash table to point to the space
	// used by this warp
	int* gColHashTable;
	float* gValHashTable;
	int globalEntries;
	
	indexTable = &indexTable[C.cols * blockId];
	
	if(laneId == 0)
		back = 0;
	
	for(int rowA = blockId; rowA < A.rows; rowA += gridDim.x)
	{
		rowAStart = A.ir[rowA];
		rowAEnd = A.ir[rowA + 1];
		for(i = laneId; i < C.cols; ++i)
		{
			indexTable[i] = -1;
		}
		__syncthreads();

		// Set the location of the global hash table
		rowCStart = C.ir[rowA];
		rowCEnd = C.ir[rowA + 1];
		globalEntries = rowCEnd - rowCStart;
		gColHashTable = &C.jc[rowCStart];
		gValHashTable = &C.val[rowCStart];
		for(i = rowAStart; i < rowAEnd; ++i)
		{
			valA = A.val[i];
			rowBStart = B.ir[A.jc[i]];
			rowBEnd = B.ir[A.jc[i] + 1];
			int curIdx;
			int* storeInt;
			float* storeFloat;
			float valB;
			for(j = rowBStart + laneId; __any(j < rowBEnd); j += warpSize)
			{
				colB = j < rowBEnd ? B.jc[j] : -1;
				curIdx = colB == -1 ? -1 : indexTable[colB];
				hash = colB != -1 && curIdx == -1 ? atomicInc(&back, globalEntries - 1) : curIdx;
				storeInt = hash == -1 ? &hash : &indexTable[colB];
				*storeInt = hash;
				storeInt = hash == -1 ? &colB : &gColHashTable[hash];
				*storeInt = colB;
				valB = colB == -1 ? 1 : B.val[j];
				storeFloat = hash == -1 ? &valA : &gValHashTable[hash];
				*storeFloat += valB * valA;
			}
		} // For each nonzero in the A row
	} // For each assigned row in A
}

__global__ void SortCols(sparse_matrix C, int maxRowNNZ, int* workQueue)
{
	const int laneId = threadIdx.x;
	const int blockId = blockIdx.x;
	
	// Dynamic shared memory
	extern __shared__ int sharedMem[];
	
	// The maximum size of the queue
	const int queueSize = (maxRowNNZ / 2) + 1;
	
	// The maximum number of passes needed
	int maxShift = __log2f(C.cols) / RADIX_BITS;
	
	// The number of passes for the work in the queue
	int* workPasses = &workQueue[blockId * queueSize];
	
	// The front of the bucket for the work in the queue
	int* workFronts = &workQueue[gridDim.x * queueSize];
	workFronts = &workFronts[blockId * queueSize];
	
	// The back of the bucket for the work in the queue
	int* workBacks = &workQueue[gridDim.x * queueSize * 2];
	workBacks = &workBacks[blockId * queueSize];
	int front; // The front of the work queue.
	__shared__ unsigned int back; // The back of the work queue.

	// Holds the sizes for the buckets being sorted by the threads
	int* bucketSizes = &sharedMem[laneId * RADIX_BASE];

	// The ending index of the buckets being sorted
	int* bucketBounds = &sharedMem[blockDim.x * RADIX_BASE];
	bucketBounds = &bucketBounds[laneId * RADIX_BASE];
	int pass; // The pass number of the current bucket
	int bucketFront; // The index of the front of the bucket
	int bucketBack; // The index of the back of the bucket
	int bucketIdx; // The index of an item in the bucket
	int shiftCount; // The number of bits to shift to get the index
	int iTmp; // A temporary variable for swapping
	float fTmp;
	int swapIdx; // The index to swap with
	int queueIdx; // An index into the work queue
	int prev; // The previous bucket offset
	int subIdx;

	for(int rowC = blockId; rowC < C.rows; rowC += gridDim.x)
	{
		// Skip if there are not non-zeros to sort
		if(C.ir[rowC] == C.ir[rowC + 1])
			continue;
		
		// Clear the work queue
		for(int i = laneId + 1; i < queueSize; i += blockDim.x)
		{
			workPasses[i] = -1;
		}
		workPasses[0] = 0;
		workFronts[0] = C.ir[rowC];
		workBacks[0] = C.ir[rowC + 1];
		front = 0;
		back = 1;
		__syncthreads();
		
		// While there is more work in the queue
		while(front != back)
		{
			queueIdx = (front + laneId) % queueSize;
			
			// Get the work
			pass = workPasses[queueIdx];
			bucketFront = workFronts[queueIdx];
			bucketBack = workBacks[queueIdx];
			
			// Clear this work
			workPasses[queueIdx] = -1;
			
			// Move the front forward
			if((back > front && back - front <= blockDim.x) || (back < front && (back + queueSize) - front <= blockDim.x))
			{
				front = back;
			}
			else
			{
				front = (front + blockDim.x) % queueSize;
			}
			
			// There is work to do
			if(pass >= 0)
			{
				// Clear the bucket sizes
				for(int i = 0; i < RADIX_BASE; ++i)
				{
					bucketSizes[i] = 0;
				}
				shiftCount = (maxShift - pass) * RADIX_BITS;
				
				// First, determine the size of the buckets
				for(int i = bucketFront; i < bucketBack; ++i)
				{
					++bucketSizes[(C.jc[i] >> shiftCount) & RADIX_MASK];
				}
				
				// Determine the indexes of the buckets and put
				// them into the work queue
				prev = bucketFront;
				for(int i = 0; i < RADIX_BASE; ++i)
				{
					// Determine the bucket end
					bucketIdx = bucketSizes[i] + prev;

					// Place the bucket into the work queue only
					// if it has items to be sorted
					if(bucketSizes[i] > 1)
					{
						queueIdx = atomicInc(&back, queueSize - 1);
						workPasses[queueIdx] = pass + 1;
						workFronts[queueIdx] = prev;
						workBacks[queueIdx] = bucketIdx;
					}
					
					// Store the bucket end
					bucketSizes[i] = bucketIdx;
					bucketBounds[i] = bucketIdx;
					prev = bucketIdx;
				}

				// Place the items into the buckets
				bucketIdx = bucketFront;
				while(bucketIdx != bucketBack)
				{
					subIdx = (C.jc[bucketIdx] >> shiftCount) & RADIX_MASK;
					swapIdx = --bucketSizes[subIdx];;
					
					// Done sorting this bucket, move to the next open one
					if(swapIdx == bucketIdx)
					{
						do 
						{
							bucketIdx = bucketBounds[subIdx++];
						} while(bucketIdx != bucketBack && bucketSizes[subIdx] == bucketIdx);
					}
					else
					{
						// Swap swapIdx and bucketIdx
						iTmp = C.jc[swapIdx];
						C.jc[swapIdx] = C.jc[bucketIdx];
						C.jc[bucketIdx] = iTmp;
						fTmp = C.val[swapIdx];
						C.val[swapIdx] = C.val[bucketIdx];
						C.val[bucketIdx] = fTmp;
					}
				}
			} // If this thread has work
			__syncthreads();
		} // While there is work to do
	} // For all rows in C
}