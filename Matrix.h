#ifndef MATRIX_H
#define MATRIX_H

typedef struct //Matrix
{
	int columns;
	int length;
	int lines;
	double list[];
}Matrix;

/* 
 * Prints a matrix m
 */
void print_matrix(Matrix *m);

/* 
 * Prints a list m of matrices of same length
 */
void print_matrices(Matrix **m, size_t length);

/* 
 * Returns a matrix col*lines 
 * with 0 if !(random) and 
 * random double values in [-2,5;2,5] else
 */
Matrix *init_matrix(int col, int lines, int random);

/* 
 * Returns a matrix col*lines 
 * with 1 only
 */
Matrix *init_ones(int col, int lines);

/*
 * Returns a matrix 1*1 with the a value (double)
 */
Matrix *onebyone_matrix(double a);

/*
 * Returns the transpose of m
 */
Matrix *transpose(Matrix *m);

/* 
 * Returns m+n matrix
 */
Matrix *add_matrix(Matrix *m, Matrix *n);

/* 
 * Returns m - n matrix
 */
Matrix *sub_matrix(Matrix *m, Matrix *n);

/*
 * Returns m*n matrix
 */
Matrix *multiply_matrix(Matrix *m, Matrix *n);

/*
 * Returns a*m where type(a) = double
 */
Matrix *mult_matrix(double a, Matrix *m);

/*
 * Returns sigmoid function on x
 */
double sigmoid(double x);

/*
 * Returns derivated sigmoid function on x
 */
double d_sigmoid(double x);

/*
 * Returns sigmoid function on matrix m = on each element of m
 */
Matrix *sigmoid_matrix(Matrix *m);

/*
 * Returns derivated sigmoid function on m = on each elmt of m
 */
Matrix *d_sigmoid_matrix(Matrix *m);

/*
 * Returns Hadamart product between m and n = m ⊙ n
 */
Matrix *hada_product(Matrix *m, Matrix *n);

/*
 * Returns list of a - b where
 * a,b are the matrices in m,n
 */
Matrix **sub_listofmatrices(Matrix **m, Matrix **n, size_t length);

/*
 * Returns list of a + b where
 * a,b are the matrices in m,n
 */
Matrix **add_listofmatrices(Matrix **m, Matrix **n, size_t length);

/*
 * Returns list of a*x where
 * type(a) = double and x are the matrices in m
 */
Matrix **mult_listofmatrices(double a, Matrix **m, size_t length);

#endif