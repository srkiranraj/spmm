/**
 * @author : Hardhik Mallipeddi (mallipeddi.hardhik@research.iiit.ac.in)
 * @author : Kiran Raj (kiran.raj@research.iiit.ac.in)
 *
 * Class Structure for representing sparse matrices in CSR format
 * 
 *	Init :
 *			sparse_matrix A = sparse_matrix(int rows, int cols, int nnz);
 *
 **/

class sparse_matrix
{
	public:
		/* data */
		int rows;
		int cols;
		int nnz;

		int *ir;
		int *jc;
		int *val;

		/* constructors */
		sparse_matrix(int rows, int cols, int nnz);

		/* destructor */
		~sparse_matrix();

		/* methods */
};