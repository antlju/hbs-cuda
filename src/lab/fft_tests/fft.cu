#include "common.h"
#include <complex>
#include "cufft.h"
#include "fftmesh.h"

#include <iostream>

#define NZH NZ/2+1

/// Instantiate global objects
cufftHandle d2zplan;
Mesh u(NX,NY,NZ,1);
cufftDoubleComplex *h_out;
cufftDoubleComplex *d_out;
Grid grid(NX,NY,NZ,0.0,2*M_PI);
Timer timer;

__host__ __device__
inline size_t oindx(Int i, Int j, Int k)
{
	return k+NZH*(j+NY*i);
}

__host__ __device__
inline size_t iindx(Int i, Int j, Int k)
{
	return k+(NZ+2)*(j+NY*i);
}

__host__ void initHost(Mesh &f, const Grid &grid)
{
	Real *x = grid.h_linspace;
	for (Int i=0;i<f.nx_;i++)
	{
		for (Int j=0;j<f.ny_;j++)
		{
			for (Int k=0;k<f.nz_;k++)
			{
				f.h_data[f.indx(i,j,k,0)] = sin(x[k]);
			}
		}
	}
}

__global__
void setInput(cufftDoubleReal *f, const Grid grid)
{
	Real *x = grid.h_linspace;
	const Int i = threadIdx.x + blockIdx.x*blockDim.x;
	const Int j = threadIdx.y + blockIdx.y*blockDim.y;

	if (i < NX && j < NY)
	{
		for (Int k=0;k<NZ;k++)
		{
			f[iindx(i,j,k)] = sqrt(2*M_PI)*sin(2.0*x[k]);
			/*
			if (k == 0 && i == 0 && j == 0)
				f[iindx(i,j,k)] = 3.0;
			else
				f[iindx(i,j,k)] = 0.0;
			*/
		}
	}
}




void printHost(cufftDoubleComplex *hdat)
{
	for (Int i=0;i<NX;i++)
	{
		for (Int j=0;j<NY;j++)
		{
			for (Int k=0;k<NZH;k++)
			{
				Real re = (Real)hdat[oindx(i,j,k)].x;
				Real im = (Real)hdat[oindx(i,j,k)].y;
				std::cout << "(" << re << "," << im << ")" << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "--------" << std::endl;
	}
}

Int main()
{
	std::cout << "Executing FFT test w/ size: (N=" << NN << ")^3" << std::endl;
	const Int Nx=NX;
	const Int Ny=NY;
	const Int Nz=NZ;
	const Int Nzh=NZH;

	CUFFT_CHECK(cufftPlan3d(&d2zplan,Nx,Ny,Nz,CUFFT_D2Z));
	
	u.allocateHost(); u.allocateDevice();

	cudaCheck(cudaMallocHost((void**)&h_out, sizeof(cufftDoubleComplex) * Nx * Ny * Nzh));
	cudaCheck(cudaMalloc((void**)&d_out, sizeof(cufftDoubleComplex) * Nx * Ny * Nzh));

			  
	grid.setHostLinspace();
	initHost(u,grid);
	//u.print();
	u.copyToDevice();

	dim3 tpb(NY_TILE,NZ_TILE); 
	dim3 blx(NN/NY_TILE,NN/NZ_TILE);
	cufftDoubleReal *d_outReal = &d_out[0].x;
	setInput<<<blx,tpb>>>(d_outReal,grid);
	CUFFT_CHECK(cufftExecD2Z(d2zplan,d_outReal,d_out));
	cudaCheck(cudaMemcpy(h_out,d_out,sizeof(cufftDoubleComplex)*Nx*Ny*Nzh,cudaMemcpyDeviceToHost));

	
	printHost(h_out);
	
	CUFFT_CHECK(cufftDestroy(d2zplan));
	return 0;
}


/*

__global__
void setComplex(cufftDoubleComplex *f)
{
	const Int i = threadIdx.x + blockIdx.x*blockDim.x;
	const Int j = threadIdx.y + blockIdx.y*blockDim.y;

	if (i < NX && j < NY)
	{
		for (Int k=0;k<NZH;k++)
		{
			f[oindx(i,j,k)].x = i;
			f[oindx(i,j,k)].y = 0;
		}
	}
}

*/
