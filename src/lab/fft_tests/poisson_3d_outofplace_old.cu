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
#define NX_TILE NN
#define NY_TILE NN
#define NZ_TILE NN

__host__ __device__
inline size_t iindx(size_t i, size_t j,size_t k)
{
	return k+(NZ)*(j+NY*i);
}

__host__ __device__
inline size_t oindx(size_t i, size_t j,size_t k)
{
	return k+((NZ >> 1) + 1)*(j+NY*i);
}

__global__
void freqDiv_kernel(cufftDoubleComplex *f, const double xlen)
{
	int k = threadIdx.z;
	int j = threadIdx.y;
	int i = threadIdx.x;

	if (i < NX && j < NY && k < NZ/2+1)
	{
		f[oindx(i,j,k)].x = oindx(i,j,k);
		f[oindx(i,j,k)].y = k;
	}

}

__global__
void freqDiv_kernel2(double *f)
{
	int k = threadIdx.z;
	int j = threadIdx.y;
	int i = threadIdx.x;

	if (i < NX && j < NY && k < NZ)
	{
		f[iindx(i,j,k)] = iindx(i,j,k);
	}

}

void init_real(cufftDoubleReal *f, double *x)
{

	
	
	for (int i=0;i<NX;i++)
	{
		for (int j=0;j<NY;j++)
		{
			for (int k=0;k<NZ;k++)
			{
				f[iindx(i,j,k)] = 0.0;
				//f[iindx(i,j,k)] = 0.0;
			}
		}
	}
	//f[0] = 1.0;
	
}


void printComplex(cufftDoubleComplex *f)
{
	std::cout << "Printing complex array: \n";
	double re,im;
	
	for (int i=0;i<NX;i++)
	{
		for (int j=0;j<NY;j++)
		{
			for (int k=0;k<(NZ/2+1);k++)
			{
				re = f[oindx(i,j,k)].x;
				im = f[oindx(i,j,k)].y;
		
		
				std::cout << "(" << re << "," << im << ") ";
			}
			std::cout << "\n";
		}
		std::cout << "---- \n";
	}
}

void printReal(cufftDoubleReal *f)
{
	std::cout << "Printing real array: \n";
	
	for (int i=0;i<NX;i++)
	{
		for (int j=0;j<NY;j++)
		{
			for (int k=0;k<NZ;k++)
			{
				
				double re = f[iindx(i,j,k)];
				if (fabs(re) < 1e-14)
					re = 0.0;
				std::cout << re <<" ";
			}
			std::cout << "\n";
		}
		std::cout << "---- \n";
	}
}

void printTest(double *f)
{
	std::cout << "Printing real array: \n";
	
	for (int i=0;i<NX;i++)
	{
		for (int j=0;j<NY;j++)
		{
			for (int k=0;k<NZ;k++)
			{
				std::cout << f[iindx(i,j,k)] <<" ";
			}
			std::cout << "\n";
		}
		std::cout << "---- \n";
	}
}


void normalise(cufftDoubleReal *f)
{
	for (int i=0;i<NX;i++)
	{
		for (int j=0;j<NY;j++)
		{
			for (int k=0;k<NZ;k++)
			{
				f[iindx(i,j,k)] = f[iindx(i,j,k)]/(NX*NY*NZ);


			}

		}

	}
}

int main()
{
	cufftHandle plan3d_d2z,plan3d_z2d;
	CUFFT_CHECK(cufftPlan3d(&plan3d_d2z,NX,NY,NZ,CUFFT_D2Z));
	CUFFT_CHECK(cufftPlan3d(&plan3d_z2d,NX,NY,NZ,CUFFT_Z2D));
	
	cufftDoubleComplex *h_mem;
	cufftDoubleComplex *h_out;
	cufftDoubleReal *h_rout;
	cufftDoubleComplex *d_in;
	cufftDoubleComplex *d_out;
	cufftDoubleReal *d_rout;
	double *d_test,*h_test;
	cudaCheck(cudaMallocHost((void**)&h_test,sizeof(double)*NX*NY*NZ));
	cudaCheck(cudaMalloc((void**)&d_test,sizeof(double)*NX*NY*NZ));
	cudaCheck(cudaMallocHost((void**)&h_mem,sizeof(cufftDoubleComplex)*NY*NX*NZH));
	cudaCheck(cudaMallocHost((void**)&h_out,sizeof(cufftDoubleComplex)*NY*NX*NZH));
	cudaCheck(cudaMallocHost((void**)&h_rout,sizeof(cufftDoubleReal)*NY*NX*NZ));
	cudaCheck(cudaMalloc((void**)&d_in,sizeof(cufftDoubleComplex)*NX*NY*NZH));
	cudaCheck(cudaMalloc((void**)&d_out,sizeof(cufftDoubleComplex)*NX*NY*NZH));
	cudaCheck(cudaMalloc((void**)&d_rout,sizeof(cufftDoubleReal)*NX*NY*NZ));
	cufftDoubleReal *h_real = (cufftDoubleReal *)h_mem;
	
	double linspace[NX];
	double L0=0.0,L1=2*M_PI;
	double dx = (L1-L0)/NX;
	
	for (int i=0;i<NX;i++)
		linspace[i] = i*dx;
	
	//init_host(h_mem,linspace);
	init_real(h_real,linspace);
	printReal(h_real);
	
	//cudaCheck(cudaMemcpy(d_in,h_mem,sizeof(cufftDoubleComplex)*NX*NY*NZH,cudaMemcpyHostToDevice));
	
	/*
	///--------------
	std::cout << "\n Executing forward R2C transform... \n\n";
	if (cufftExecD2Z(plan3d_d2z,(cufftDoubleReal *)d_in,d_out) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: ExecD2Z Forward failed");
		return 0;	
	}
	cudaCheck(cudaDeviceSynchronize());
	*/
	
	///--------------
	std::cout << "\n Running freq div kernel... \n\n";
	dim3 blx(NX/NX_TILE,NY/NY_TILE,NZ/NZ_TILE);
	dim3 tpb(NX_TILE,NY_TILE,NZ_TILE);
	//freqDiv_kernel<<<blx,tpb>>>(d_out,L1-L0);
	freqDiv_kernel2<<<blx,tpb>>>(d_test);
	
	cudaCheck(cudaDeviceSynchronize());

	/*	
	///--------------
	std::cout << "\n Executing forward C2R transform... \n\n";
	if (cufftExecZ2D(plan3d_z2d,d_out,d_rout) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: ExecZ2D Backward failed");
		return 0;	
	}
	*/
	cudaCheck(cudaDeviceSynchronize());

	cudaCheck(cudaMemcpy(h_test,d_test,sizeof(double)*NX*NY*NZ,cudaMemcpyDeviceToHost));
	//cudaCheck(cudaMemcpy(h_out,d_out,sizeof(cufftDoubleComplex)*NX*NY*NZH,cudaMemcpyDeviceToHost));
	//cudaCheck(cudaMemcpy(h_rout,d_rout,sizeof(cufftDoubleReal)*NX*NY*NZ,cudaMemcpyDeviceToHost));
	//cufftDoubleReal *h_outReal = (cufftDoubleReal*)h_out;
	//normalise(h_rout);
	//printComplex(h_out);
	//printReal(h_rout);
	printTest(h_test);
	/// Free mem
	cudaCheck(cudaFreeHost(h_mem));
	cudaCheck(cudaFreeHost(h_out));
	cudaCheck(cudaFreeHost(h_rout));
	cudaCheck(cudaFree(d_in));
	cudaCheck(cudaFree(d_out));
	CUFFT_CHECK(cufftDestroy(plan3d_d2z));
	CUFFT_CHECK(cufftDestroy(plan3d_z2d));
	return 0;
}
		
/*
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
				f[oindx(i,j,k)].x = 1.0;
				f[oindx(i,j,k)].y = 0.0;
			}
		}
	}
	
}

 */
