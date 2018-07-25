#include <cufft.h>
#include "errcheck.h"
#include "cuffterr.h"
#include <iostream>
#include <cmath>

#define NN 8


void init_host(cufftDoubleComplex *f, double *x)
{
	//f[0].x = 1.0;
	//f[0].y = 0.0;
	
	
	for (int i=0;i<NN;i++)
	{
		f[i].x = cos(x[i]);
		f[i].y = 0.0;
	}
	
}

void printComplex(cufftDoubleComplex *f)
{
	std::cout << "Printing complex array: \n";
	double re,im;
	
	for (int i=0;i<NN;i++)
	{
		re = f[i].x;
		im = f[i].y;
		
		if (fabs(re) < 1e-14)
			re = 0.0;
		if (fabs(im) < 1e-14)
			im = 0.0;
		
		
		std::cout << "(" << re << "," << im << ") \n";
	}
}

int main()
{
	cufftHandle plan1d;
	CUFFT_CHECK(cufftPlan1d(&plan1d,NN,CUFFT_Z2Z,1));
		    
	cufftDoubleComplex *h_mem;
	cufftDoubleComplex *d_in;
	cufftDoubleComplex *d_out;
	
	cudaCheck(cudaMallocHost((void**)&h_mem,sizeof(cufftDoubleComplex)*NN));
	cudaCheck(cudaMalloc((void**)&d_in,sizeof(cufftDoubleComplex)*NN));
	cudaCheck(cudaMalloc((void**)&d_out,sizeof(cufftDoubleComplex)*NN));

	double linspace[NN];
	double L0=0.0,L1=2*M_PI;
	double dx = (L1-L0)/NN;
	
	for (int i=0;i<NN;i++)
		linspace[i] = i*dx;
	
	init_host(h_mem,linspace);
	printComplex(h_mem);
	
	cudaCheck(cudaMemcpy(d_in,h_mem,sizeof(cufftDoubleComplex)*NN,cudaMemcpyHostToDevice));

	std::cout << "\n Executing forward C2C transform. \n\n";
	if (cufftExecZ2Z(plan1d,d_in,d_out,CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
		return 0;	
	}
	
	cudaCheck(cudaMemcpy(h_mem,d_out,sizeof(cufftDoubleComplex)*NN,cudaMemcpyDeviceToHost));

	printComplex(h_mem);

	std::cout << "\n" << NN << "       " << (NN >> 1) +1 << "\n";
	/// Free mem
	cudaCheck(cudaFreeHost(h_mem));
	cudaCheck(cudaFree(d_in));
	cudaCheck(cudaFree(d_out));
	CUFFT_CHECK(cufftDestroy(plan1d));
	
	return 0;
}
		
