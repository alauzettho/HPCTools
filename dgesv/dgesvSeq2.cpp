///////////////////////////////////////////////////
//
// Created by Thomas Alauzet on Sept 28, 2018
//
///////////////////////////////////////////////////

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

void my_dgesv(const int n, double** A, double** B, double** X)
{
	///////////////////////////////////////////////////
	//
	// Initialization
	//
	///////////////////////////////////////////////////

	int i;
	int j;
	int k;

	double 	sum = 0.0;
	double*   z = new double[n];
	double** LU = new double*[n];

	for (i = 0; i < n; i++)
	{
		LU[i] = new double[n];
		for (j = 0; j < n; j++) LU[i][j] = 0.0;
	}


	///////////////////////////////////////////////////
	//
	// Finding LU
	//
	///////////////////////////////////////////////////


	for (j = 0; j < n; j++)
	{
		for (i = 0; i <= j; i++)
		{
			sum = 0.0;
			for (k = 0; k < i; k++)
			{
				sum += LU[i][k] * LU[k][j];
			}
			LU[i][j] = A[i][j] - sum;
		}

		for (i = j + 1; i < n; i++)
		{
			sum = 0.0;
			for (k = 0; k < j; k++)
			{
				sum += LU[i][k] * LU[k][j];
			}
			LU[i][j] = (A[i][j] - sum) / LU[j][j];
		}
	}


	///////////////////////////////////////////////////
	//
	// Finding X
	//
	///////////////////////////////////////////////////


	for (j = 0; j < n; j++)
	{
		for (i = 0; i < n; i++)
		{
			sum = 0.0;
			for (k = 0; k < i; k++)
			{
				sum += LU[i][k] * z[k];
			}
			z[i] = B[i][j] - sum;
		}

		for (i = n - 1; i >= 0; i--)
		{
			sum = 0.0;
			for (k = i + 1; k < n; k++)
			{
				sum += LU[i][k] * X[k][j];
			}
			X[i][j] = (z[i] - sum) / LU[i][i];
		}
	}
}


int main(int argc, char *argv[])
{
	///////////////////////////////////////////////////
	//
	// Init Matices
	//
	///////////////////////////////////////////////////

	srand(rand());
	
	int n = (argc > 1) ? atoi(argv[1]) : 100;

	double** A = new double*[n];
	double** B = new double*[n];
	double** X = new double*[n];

	for (int i = 0; i < n; i++)
	{
		A[i] = new double[n];
		B[i] = new double[n];
		X[i] = new double[n];

		for (int j = 0; j < n; j++)
		{
			A[i][j] = rand() % 100;
			B[i][j] = rand() % 100;
			X[i][j] = 0.0;
		}
	}

	///////////////////////////////////////////////////
	//
	//  Solving for AX = B with my implementation
	//
	///////////////////////////////////////////////////

	clock_t t = clock();


	my_dgesv(n, A, B, X);


	t = clock() - t;

	printf ("Time : %f \n", (float)t / CLOCKS_PER_SEC);

	return(0);
}