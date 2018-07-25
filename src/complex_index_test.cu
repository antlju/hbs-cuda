#include <cufft.h>
#include "errcheck.h"
#include "cuffterr.h"
#include <iostream>
#include <cmath>

#define NN 8
#define NX NN
#define NY NN
#define NZ NN
#define NZH NZ/2+1
#define NX_TILE 4
#define NY_TILE 4
#define NZ_TILE NN

typedef int Int;
typedef double Real;
typedef double2 Complex;

__global__
void complex_kernel(Complex *f)
{
	int j = threadIdx.y+blockIdx.y*blockDim.y;
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	//if (i < NX && j < NY && k < ((NZ >> 1) + 1))
	if (i < NX && j < NY)
	{
		for (Int k=0;k<((NZ >> 1)+1);k++)
		{
			f[k+((NZ >> 1)+1)*(j+NY*i)].x = k+((NZ >> 1)+1)*(j+NY*i);
			f[k+((NZ >> 1)+1)*(j+NY*i)].y = k;
		}
	}
}

void printTest(Complex *f)
{
	for (Int i=0;i<NX;i++)
	{
		for (Int j=0;j<NY;j++)
		{
			for (Int k=0;k<(NZ/2+1);k++)
			{
				std::cout << f[k+((NZ >> 1)+1)*(j+NY*i)].x << "," <<
					f[k+((NZ >> 1)+1)*(j+NY*i)].y << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "----- " << std::endl;
	}
}
Int main()
{

	Complex *d_test,*h_test;

	cudaCheck(cudaMallocHost((void**)&h_test,sizeof(Complex)*NX*NY*(NZ/2+1)));
	cudaCheck(cudaMalloc((void**)&d_test,sizeof(Complex)*NX*NY*(NZ/2+1)));
	
	dim3 blx(NX/NX_TILE,NY/NY_TILE);
	dim3 tpb(NX_TILE,NY_TILE);
	
	complex_kernel<<<blx,tpb>>>(d_test);

	cudaCheck(cudaMemcpy(h_test,d_test,sizeof(Complex)*NX*NY*(NZ/2+1),cudaMemcpyDeviceToHost));

	printTest(h_test);

	       
	return 0;
}
