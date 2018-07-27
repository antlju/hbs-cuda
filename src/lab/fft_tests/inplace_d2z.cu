#include <cufft.h>
#include "errcheck.h"
#include "cuffterr.h"
#include <iostream>
#include <cmath>

#define NN 4
#define NX NN
#define NY NN
#define NZ NN
#define NZH NZ/2+1

__host__ __device__
inline size_t oindx(int i, int j, int k)
{
	return k+NZH*(j+NY*i);
}

__host__ __device__
inline size_t iindx(int i, int j, int k)
{
	return k+(NZ+2)*(j+(NY)*i);
}

void inithostmem(double *h_mem)
{
	for (int i=0;i<NX;i++)
	{
		for (int j=0;j<NY;j++)
		{
			for (int k=0;k<NZ;k++)
			{
				h_mem[k+NY*(j+(NX+2)*i)] = 1.0;
			}
		}
	}
}

void printHost(cufftDoubleComplex *f)
{
	for (int i=0;i<NX;i++)
	{
		for (int j=0;j<NY;j++)
		{
			for (int k=0;k<NZH;k++)
			{
				double re = f[k+NZH*(j+NY*i)].x;
				double im = f[k+NZH*(j+NY*i)].y;
				std::cout << re << "," << im << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "--- " << std::endl;
	}
       
}

int main()
{
	double *h_mem;
	cufftDoubleComplex *xform_h_mem;
	cudaCheck(cudaMallocHost(&h_mem,NX*NY*NZ*sizeof(double)));
	cudaCheck(cudaMallocHost(&xform_h_mem,NX*NY*NZH*sizeof(cufftDoubleComplex)));
	inithostmem(h_mem);
	
	cufftDoubleComplex *d_mem;

	cudaCheck(cudaMalloc(&d_mem,NX*NY*NZH*sizeof(cufftDoubleComplex)));
	cudaCheck(cudaMemcpy(d_mem,h_mem,NX*NY*NZH*sizeof(cufftDoubleComplex),cudaMemcpyHostToDevice));

	cufftHandle pland2z;

	CUFFT_CHECK(cufftPlan3d(&pland2z,NX,NY,NZ,CUFFT_D2Z));
	CUFFT_CHECK(cufftExecD2Z(pland2z,(cufftDoubleReal*)d_mem,d_mem));

	cudaCheck(cudaMemcpy(xform_h_mem,d_mem,
			     NX*NY*NZH*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToHost));

	printHost(xform_h_mem);
	return 0;
}
