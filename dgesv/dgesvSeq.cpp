///////////////////////////////////////////////////
//
// Created by Thomas Alauzet on Sept 14, 2018
//
///////////////////////////////////////////////////

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <mkl_lapacke.h>

double* generate_matrix(const int n)
{
	double* matrix = (double* )malloc(sizeof(double)* n * n);

	for (int i = 0; i < n * n; i++) matrix[i] = rand() % 100;

	return matrix;
}

void print_matrix(const int n, const double* matrix, const char* name)
{
	printf("matrix: %s \n", name);

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			printf("%f ", matrix[i * n + j]);
		}
		printf("\n");
	}
}

double check_error(const int n, const double* A, const double* B)
{
	double error = 0.0;

	for (int i = 0; i < n * n; i++) error += fabs(B[i] - A[i]);

	return error;
}

void my_dgesv(const int n, const double* A, const double* B, double* X)
{
	clock_t tStart = clock();
	double  sum    = 0.0;
	double* LU     = new double[n * n];

	///////////////////////////////////////////////////
	//
	//  P0 : LU Decomposition for AX = B / LU = A
	//
	///////////////////////////////////////////////////

	for (int j = 0; j < n * n; j++)
	{
		LU[j] = (j % (n + 1) == 0) ? 1 : 0;
	}

	for (int j = 0; j < n; j++)
	{
		for (int i = 0; i < j + 1; i++)
		{
			sum = 0.0;
			for (int k = 0; k < i; k++)
			{
				sum += LU[n * i + k] * LU[n * k + j];
			}
			LU[n * i + j] = A[n * i + j] - sum;
		}
		for (int i = j + 1; i < n; i++)
		{
			sum = 0.0;
			for (int k = 0; k < j; k++)
			{
				sum += LU[n * i + k] * LU[n * k + j];
			}
			LU[n * i + j] = (A[n * i + j] - sum) / LU[n * j + j];
		}
	}

	printf("Time spent on LU Factoring : %f \n", (double)(clock() - tStart) / CLOCKS_PER_SEC);


	///////////////////////////////////////////////////
	//
	//  Finding the solution
	//
	///////////////////////////////////////////////////

	tStart = clock();

	double* z = new double[n];

	for (int column_idx = 0; column_idx < n; column_idx++)
	{
		for (int i = 0; i < n; i++)
		{
			sum = 0.0;
			for (int p = 0; p < i; p++)
			{
				sum += LU[n * i + p] * z[p];
			}
			z[i] = B[n * i + column_idx] - sum;
		}

		for (int i = n - 1; i >= 0; i--)
		{
			sum = 0.0;
			for (int p = i + 1; p < n; p++)
			{
				sum += LU[n * i + p] * X[n * p + column_idx];
			}
			X[n * i + column_idx] = (z[i] - sum) / LU[n * i + i];
		}
	}

	printf("Time spent on Reduction : %f \n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

	delete[] z;
	delete[] LU;
}

int main(int argc, char *argv[])
{
	///////////////////////////////////////////////////
	//
	// Init Matices
	//
	///////////////////////////////////////////////////

	srand(rand());						// Init random seed
	
	int size = 10;						// Default size of the square matrix

	if (argc > 1) size = atoi(argv[1]);

	double* A = generate_matrix(size);  // Left side matrix    
	double* B = generate_matrix(size);  // Right side matrix 
	double* X = generate_matrix(size);  // Solution 
	double* P = generate_matrix(size);  // P = AX for later checking


	///////////////////////////////////////////////////
	//
	//  Solving for AX = B with MKL and LAPACKE
	//
	///////////////////////////////////////////////////

	clock_t tStart = clock();

	MKL_INT  n    = size, nrhs = n, lda = n, ldb = n, info;
	MKL_INT* ipiv = (MKL_INT*)malloc(sizeof(MKL_INT)*n);
	info          = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, A, lda, ipiv, B, ldb);
	
	printf("Time spent on MKL : %f \n", (double)(clock() - tStart) / CLOCKS_PER_SEC);


	///////////////////////////////////////////////////
	//
	//  Solving for AX = B with my implementation
	//
	///////////////////////////////////////////////////

	tStart = clock();

	my_dgesv(size, A, B, X);

	printf("Time spent on My Implementation : %f \n \n", (double)(clock() - tStart) / CLOCKS_PER_SEC);


	///////////////////////////////////////////////////
	//
	//  Checking results with matrice norm
	//
	///////////////////////////////////////////////////

	tStart = clock();

	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			double s = 0.0;
			for (int r = 0; r < size; r++)
			{
				s += A[size * i + r] * X[size * r + j];
			}
			P[size * i + j] = s;
		}
	}

	double error = check_error(size, P, B);

	printf("Time spent on Checking The Result : %f \n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

	printf("Error of My Implementation : %f \n", error);

	return(0);
}