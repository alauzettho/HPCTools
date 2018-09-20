///////////////////////////////////////////////////
//
// Created by Thomas Alauzet on Sept 14, 2018
//
// Compilation : mpi++ dgesv.cpp -fopenmp
// Execution   : mpirun -np 1 ./a.out 1500 4
//
///////////////////////////////////////////////////


//#include <omp.h>
#include <mpi.h>
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

	//#pragma omp parallel for reduction(+:error)
	for (int i = 0; i < n * n; i++) error += fabs(B[i] - A[i]);

	return error;
}

void my_dgesv(const int n, const int nproc, const int nrank, const double* A, const double* B, double* X)
{
	///////////////////////////////////////////////////
	//
	//  Build MPI arrays for gathering
	//	The fastest when nproc | n
	//
	///////////////////////////////////////////////////
	
	int* posinit       = new int[nproc];
	int* number_gather = new int[nproc];

	for (int i = 0; i < nproc; i++)
	{
		number_gather[i] = n * (n / nproc + ((i < n % nproc) ? 1 : 0));
	}
	
	posinit[0] = 0;

	for (int i = 1; i < nproc; i++)
	{
		posinit[i] = posinit[i - 1] + number_gather[i - 1];
	}

	int idx_start_col = posinit[nrank] / n;
	int n_column_proc = n / nproc + ((nrank < n % nproc) ? 1 : 0);

	double* LU     = new double[n * n];
	double* X_temp = new double[n * n_column_proc];


	///////////////////////////////////////////////////
	//
	//  P0 : LU Decomposition for AX = B / LU = A
	//
	///////////////////////////////////////////////////

	double  sum    = 0.0;
	clock_t tStart = clock();

	if (nrank == 0)
	{
		for (int j = 0; j < n * n; j++) LU[j] = (j % (n + 1) == 0) ? 1 : 0;

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
	}

	if (nrank == 0) printf("Time spent on LU Factoring : %f \n", (double)(clock() - tStart) / CLOCKS_PER_SEC);


	///////////////////////////////////////////////////
	//
	//  Gather values in LU
	//
	///////////////////////////////////////////////////

	tStart = clock();

	MPI_Bcast(LU, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (nrank == 0) printf("Time spent on Communication 1: %f \n", (double)(clock() - tStart) / CLOCKS_PER_SEC);


	///////////////////////////////////////////////////
	//
	//  Finding the solution
	//	Every proc solves for a column of the system
	//
	///////////////////////////////////////////////////

	tStart = clock();

	int idx = 0;

	for (int column_idx = idx_start_col; column_idx < idx_start_col + n_column_proc; column_idx++)
	{
		double* z = new double[n];

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
				sum += LU[n * i + p] * X_temp[n * idx + p];
			}
			X_temp[n * idx + i] = (z[i] - sum) / LU[n * i + i];
		}

		delete[] z;
		idx++;
	}

	if (nrank == 0) printf("Time spent on Reduction : %f \n", (double)(clock() - tStart) / CLOCKS_PER_SEC);


	///////////////////////////////////////////////////
	//
	//  Gather values in X
	//
	///////////////////////////////////////////////////

	tStart = clock();

	MPI_Allgatherv(&X_temp[0], number_gather[nrank], MPI_DOUBLE, &X[0], &number_gather[0], &posinit[0], MPI_DOUBLE, MPI_COMM_WORLD);

	if (nrank == 0) printf("Time spent on Communication 2: %f \n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

	///////////////////////////////////////////////////
	//
	//  Transpose X (Not optimised but we have to)
	//
	///////////////////////////////////////////////////
	
	tStart = clock();

	double temp = 0.0;
	
	for (int i = 1; i < n * n - 1; i++)
	{
		if (i / n > i % n)
		{
			temp = X[i];
			X[i] = X[n * (i % n) + (i / n)];
			X[n * (i % n) + (i / n)] = temp;
		}
	}

	if (nrank == 0) printf("Time spent on Transposing : %f \n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

	delete[] LU;
	delete[] X_temp;
	delete[] posinit;
	delete[] number_gather;
}

int main(int argc, char *argv[])
{
	///////////////////////////////////////////////////
	//
	// Init MPI, OMP and Matices
	//
	///////////////////////////////////////////////////

	srand(rand());						// Init random seed
	
	int size  = 10;						// Default size of the square matrix
	int nproc = 0;						// Number of Procs
	int nrank = 0;						// Rank of the Proc

	//if (argc > 2) omp_set_num_threads(atoi(argv[2]));
	if (argc > 1) size = atoi(argv[1]);

	double* A = generate_matrix(size);  // Left side matrix    
	double* B = generate_matrix(size);  // Right side matrix 
	double* X = generate_matrix(size);  // Solution 
	double* P = generate_matrix(size);  // P = AX for later checking

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
	
	assert(size >= nproc);				// Make shure we don't overuse procs


	///////////////////////////////////////////////////
	//
	//  Solving for AX = B with MKL and LAPACKE
	//
	///////////////////////////////////////////////////

	clock_t tStart = clock();

	MKL_INT  n    = size, nrhs = n, lda = n, ldb = n, info;
	MKL_INT* ipiv = (MKL_INT*)malloc(sizeof(MKL_INT)*n);
	info          = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, A, lda, ipiv, B, ldb);
	
	if (nrank == 0) printf("Time spent on MKL : %f \n", (double)(clock() - tStart) / CLOCKS_PER_SEC);


	///////////////////////////////////////////////////
	//
	//  Solving for AX = B with my implementation
	//
	///////////////////////////////////////////////////

	tStart = clock();

	my_dgesv(size, nproc, nrank, A, B, X);

	if (nrank == 0) printf("Time spent on My Implementation : %f \n", (double)(clock() - tStart) / CLOCKS_PER_SEC);


	///////////////////////////////////////////////////
	//
	//  Checking results with matrice norm
	//
	///////////////////////////////////////////////////

	if (nrank == 0)
	{
		tStart = clock();

		//#pragma omp parallel for
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
		printf("Time spent on Checking The Result : %f \n \n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
		printf("Error of My Implementation : %f \n", error);
	}

	MPI_Finalize();

	return(0);
}