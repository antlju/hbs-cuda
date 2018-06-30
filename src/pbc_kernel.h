#pragma once
#include "typedefs.h"
#include "indexing.h"

__global__
void pbc_x_kernel(Real *u, const Int Nx, const Int Ng, const Int Nvars)
{
	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;

	for (size_t vi=0;vi<Nvars;vi++)
	{
		for (size_t l=0;l<Ng;l++)
		{
			//set pbc along x
			u[fIdx(l-Ng,j,k,vi)] = u[fIdx(Nx-(Ng-l),j,k,vi)];
			u[fIdx(Nx+l,j,k,vi)] = u[fIdx(l,j,k,vi)];
		}
	}
}

__global__
void pbc_y_kernel(Real *u, const Int Ny, const Int Ng, const Int Nvars)
{
	/// Global indices
	const Int i = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;

	for (size_t vi=0;vi<Nvars;vi++)
	{
		for (size_t l=0;l<Ng;l++)
		{
			//set pbc along y
			u[fIdx(i,l-Ng,k,vi)] = u[fIdx(i,Ny-(Ng-l),k,vi)];
			u[fIdx(i,Ny+l,k,vi)] = u[fIdx(i,l,k,vi)];
		}
	}
}

__global__
void pbc_z_kernel(Real *u, const Int Nz, const Int Ng, const Int Nvars)
{
	/// Global indices
	const Int i = threadIdx.x + blockIdx.x*blockDim.x;
	const Int j = threadIdx.y + blockIdx.y*blockDim.y;

	for (size_t vi=0;vi<Nvars;vi++)
	{
		for (size_t l=0;l<Ng;l++)
		{
			//set pbc along z
			u[fIdx(i,j,l-Ng,vi)] = u[fIdx(i,j,Nz-(Ng-l),vi)];
			u[fIdx(i,j,Nz+l,vi)] = u[fIdx(i,j,l,vi)];
		}
	}
}
