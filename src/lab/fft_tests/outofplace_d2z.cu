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
inline size_t indx(size_t i, size_t j, size_t k)
{
	return k+(NZ)*(j+NY*i);
}

void setComplex(cufftDoubleComplex *f)
{
	for (int i=0;i<NX;i++)
	{
		for (int j=0;j<NY;j++)
		{
			for (int k=0;k<NZH;k++)
			{
				f[k+NZH*(j+NY*i)].x = 0.0;
				f[k+NZH*(j+NY*i)].y = 0.0;
				
				/*
				if (re < 1e-14)
					re = 0.0;

				if (im < 1e-14)
					im = 0.0;
				*/
				
			}
			
		}
		
	}

	f[0].x = sqrt(2*M_PI);
	f[0].y = 0.0;
       
}


void printComplex(cufftDoubleComplex *f)
{
	std::cout << "\n Printing complex array \n" << std::endl;
	for (int i=0;i<NX;i++)
	{
		for (int j=0;j<NY;j++)
		{
			for (int k=0;k<NZH;k++)
			{
				double re = f[k+NZH*(j+NY*i)].x;
				double im = f[k+NZH*(j+NY*i)].y;
				
				/*
				if (re < 1e-14)
					re = 0.0;

				if (im < 1e-14)
					im = 0.0;
				*/
				std::cout << re << "," << im << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "--- " << std::endl;
	}
       
}

void printReal(cufftDoubleReal *f)
{
	for (int i=0;i<NX;i++)
	{
		for (int j=0;j<NY;j++)
		{
			for (int k=0;k<NZ;k++)
			{
				double re = f[k+(NZ+2)*(j+NY*i)];
				if (re < 1e-14)
					re = 0.0;
				
				std::cout << re << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "--- " << std::endl;
	}
       
}

void setReal(cufftDoubleReal *f)
{
	for (int i=0;i<NX;i++)
	{
		for (int j=0;j<NY;j++)
		{
			for (int k=0;k<NZ;k++)
			{
				f[indx(i,j,k)] = 0.0;
			}

		}

	}
	f[indx(0,0,0)] = sqrt(2*M_PI)*1e10;
	
}

int main()
{

	cufftHandle pland2z, planz2z;
	CUFFT_CHECK(cufftPlan3d(&pland2z,NX,NY,NZ,CUFFT_D2Z));
	CUFFT_CHECK(cufftPlan3d(&planz2z,NX,NY,NZH,CUFFT_Z2Z));
	
	cufftDoubleReal *hreal;
	cufftDoubleReal *dreal;
	cufftDoubleComplex *dcomplex;
	cufftDoubleComplex *hcomplex;
	
	cudaCheck(cudaMallocHost((void**)&hreal,NX*NY*NZ*sizeof(cufftDoubleReal)));
	cudaCheck(cudaMallocHost((void**)&hcomplex,NX*NY*NZH*sizeof(cufftDoubleComplex)));
	cudaCheck(cudaMalloc((void**)&dreal,NX*NY*NZ*sizeof(cufftDoubleReal)));
	cudaCheck(cudaMalloc((void**)&dcomplex,NX*NY*NZH*sizeof(cufftDoubleComplex)));
	
	//setReal(hreal);
	//cudaCheck(cudaMemcpy(dreal,hreal,NX*NY*NZ*sizeof(cufftDoubleReal),cudaMemcpyHostToDevice));

	setComplex(hcomplex);
	printComplex(hcomplex);
	
	/// Exec real-to-complex transform
	/*
	if (cufftExecD2Z(pland2z,dreal,dcomplex) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: ExecD2Z Forward failed");
		return 0;	
	}
	*/
	
	/// Exec complex-to-complex transform
	if (cufftExecZ2Z(planz2z,dcomplex,dcomplex,CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
		return 0;	
	}

	cudaCheck(cudaMemcpy(hcomplex,dcomplex,
			     NX*NY*NZH*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToHost));
	
	printComplex(hcomplex);

	
	cudaCheck(cudaFreeHost(hreal));
	cudaCheck(cudaFreeHost(hcomplex));
	cudaCheck(cudaFree(dreal));
	cudaCheck(cudaFree(dcomplex));
	CUFFT_CHECK(cufftDestroy(pland2z));
	
	return 0;
}
