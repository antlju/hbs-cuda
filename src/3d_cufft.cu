#include <cufft.h>
#include "errcheck.h"
#include "cuffterr.h"
#include <iostream>
#include <cmath>

#define NN 8
#define NX 8
#define NY 8
#define NZ 8

__host__ __device__
inline size_t iindx(size_t i, size_t j,size_t k)
{
	return k+NZ*(j+NY*i);
}

void init_host(cufftDoubleComplex *f, double *x)
{
	//f[0].x = 1.0;
	//f[0].y = 0.0;
	
	
	for (int i=0;i<NX;i++)
	{
		for (int j=0;j<NY;j++)
		{
			for (int k=0;k<NZ;k++)
			{
				f[iindx(i,j,k)].x = 1.0;
				f[iindx(i,j,k)].y = 0.0;
			}
		}
	}
	
}

void printComplex(cufftDoubleComplex *f)
{
	std::cout << "Printing complex array: \n";
	double re,im;
	
	for (int i=0;i<NX;i++)
	{
		for (int j=0;j<NY;j++)
		{
			for (int k=0;k<NZ;k++)
			{
				re = f[iindx(i,j,k)].x;
				im = f[iindx(i,j,k)].y;
				
				if (fabs(re) < 1e-14)
					re = 0.0;
				if (fabs(im) < 1e-14)
					im = 0.0;
		
		
				std::cout << "(" << re << "," << im << ") ";
			}
			std::cout << "\n";
		}
		std::cout << "---- \n";
	}
}

int main()
{
	cufftHandle plan3d;
	CUFFT_CHECK(cufftPlan3d(&plan3d,NX,NY,NZ,CUFFT_Z2Z));
		    
	cufftDoubleComplex *h_mem;
	cufftDoubleComplex *d_in;
	cufftDoubleComplex *d_out;
	
	cudaCheck(cudaMallocHost((void**)&h_mem,sizeof(cufftDoubleComplex)*NY*NX*NZ));
	cudaCheck(cudaMalloc((void**)&d_in,sizeof(cufftDoubleComplex)*NX*NY*NZ));
	cudaCheck(cudaMalloc((void**)&d_out,sizeof(cufftDoubleComplex)*NX*NY*NZ));

	double linspace[NX];
	double L0=0.0,L1=2*M_PI;
	double dx = (L1-L0)/NX;
	
	for (int i=0;i<NX;i++)
		linspace[i] = i*dx;
	
	init_host(h_mem,linspace);
	printComplex(h_mem);
	
	cudaCheck(cudaMemcpy(d_in,h_mem,sizeof(cufftDoubleComplex)*NX*NY*NZ,cudaMemcpyHostToDevice));

	std::cout << "\n Executing forward C2C transform. \n\n";
	if (cufftExecZ2Z(plan3d,d_in,d_out,CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
		return 0;	
	}
	
	cudaCheck(cudaMemcpy(h_mem,d_out,sizeof(cufftDoubleComplex)*NX*NY*NZ,cudaMemcpyDeviceToHost));

	printComplex(h_mem);

	
	/// Free mem
	cudaCheck(cudaFreeHost(h_mem));
	cudaCheck(cudaFree(d_in));
	cudaCheck(cudaFree(d_out));
	CUFFT_CHECK(cufftDestroy(plan3d));
	
	return 0;
}
		
