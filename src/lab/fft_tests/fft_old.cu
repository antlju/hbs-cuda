#include "common.h"
#include "cufft.h"
#include "fftmesh.h"

#include <iostream>

/// Instantiate global objects
cufftHandle r2cplanfw;
cufftHandle c2rplanbw;
fftMesh u_fft(NX,NY,NZ);
Mesh u(NX,NY,NZ,1);
Mesh output(NX,NY,NZ,1);
Grid grid(NX,NY,NZ,0.0,2*M_PI);
Timer timer;

__device__
size_t oindx(size_t i, size_t j, size_t k)
{
	return k+(NZ/2+1)*(j+NY*i);
}

__global__
void pssnFreqDiv_kernel(fftMesh fft, const Real xlen)
{
	Int II,JJ;
	Int Nx=fft.nx_,Ny=fft.ny_,Nzh=fft.nzh_;
	Real k1,k2,k3,Pi=M_PI;
	Real fac = 0.0;

	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;

	
	if (j < Ny && k < Nzh)
	{
		for (Int i=0;i<Nx;i++)
		{
			fft.df_data[oindx(i,j,k)].x = 1.0;
		}
	}
	
}


__global__ void mesh2fft_kernel(Mesh f, fftMesh fft, const Int vi)
{
	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;

	if (j < f.ny_ && k < f.nz_)
	{
		for (Int i=0;i<f.nx_;i++)
			fft.ds_data[k+(f.ny_+2)*(j+f.nx_*i)] = (cufftDoubleReal)f(i,j,k,vi);
	}
}

__global__ void fft_normalise_kernel(fftMesh fft)
{
	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;

	if (j < fft.ny_ && k < fft.nz_)
	{
		for (Int i=0;i<fft.nx_;i++)
			fft.ds_data[k+(fft.ny_+2)*(j+fft.nx_*i)] = fft.ds_data[k+(fft.ny_+2)*(j+fft.nx_*i)]/(fft.nx_*fft.ny_*fft.nz_);
	}
}


__global__ void fft2mesh_kernel(fftMesh fft, Mesh f, const Int vi)
{
	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;

	if (j < f.ny_ && k < f.nz_)
	{
		for (Int i=0;i<f.nx_;i++)
			f(i,j,k,vi) = (Real)fft.ds_data[k+(f.ny_+2)*(j+f.nx_*i)];
	}
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



Int main()
{
	std::cout << "Executing FFT test w/ size: (N=" << NN << ")^3" << std::endl;
	const Int Nx=NX;
	const Int Ny=NY;
	const Int Nz=NZ;

	CUFFT_CHECK(cufftPlan3d(&r2cplanfw,Nx,Ny,Nz,CUFFT_D2Z));
	
	CUFFT_CHECK(cufftPlan3d(&c2rplanbw,Nx,Ny,Nz,CUFFT_Z2D));
	
	u_fft.allocateDevice();
	u.allocateHost(); u.allocateDevice();
	output.allocateHost(); output.allocateDevice();


	grid.setHostLinspace();
	initHost(u,grid);
	//u.print();
	u.copyToDevice();

	dim3 tpb(NY_TILE,NZ_TILE); 
	dim3 blx(NN/NY_TILE,NN/NZ_TILE);

	/// Copy from host to device fft.
	mesh2fft_kernel<<<blx,tpb>>>(u,u_fft,0);
	CUFFT_CHECK(cufftExecD2Z(r2cplanfw,u_fft.ds_data,(cufftDoubleComplex*)u_fft.ds_data));
	pssnFreqDiv_kernel<<<blx,tpb>>>(u_fft,grid.xlen);
	CUFFT_CHECK(cufftExecZ2D(c2rplanbw,(cufftDoubleComplex*)u_fft.ds_data,u_fft.ds_data));
	fft_normalise_kernel<<<blx,tpb>>>(u_fft);
	fft2mesh_kernel<<<blx,tpb>>>(u_fft,output,0);
	output.copyFromDevice();

	//Real meandiff = 0.0;
	//Real diff = 0.0;
	//for (Int k=0;k<u.nz_;k++)
	//{
	//	diff = fabs(u.h_data[u.indx(0,0,k,0)]-2*output.h_data[output.indx(0,0,k,0)]);
	//	meandiff += diff;
	//}
	//meandiff = meandiff/(NX*NY*NZ);
	//std::cout << meandiff << std::endl;
	//Int kk = NZ/2;
	//std::cout << output.h_data[output.indx(3,2,kk,0)] << "\t" << sin(grid.h_linspace[kk]) << std::endl;
	
	CUFFT_CHECK(cufftDestroy(r2cplanfw));
	CUFFT_CHECK(cufftDestroy(c2rplanbw));
	cudaCheck(cudaFree(u_fft.ds_data));
	return 0;
}

