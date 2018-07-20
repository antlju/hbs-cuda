#pragma once

#include "typedefs.h"
#include <cassert>

class Grid
{
public:
	size_t nx_,ny_,nz_;
	Real L0_,L1_,dx_,invdx_;
	Real *h_linspace;
	Real xlen;
	__host__ Grid(Real Nx, Real Ny, Real Nz, Real L0, Real L1) :
		nx_(Nx),ny_(Ny),nz_(Nz),L0_(L0),L1_(L1)
	{
		dx_ = (L1-L0)/Nx;
		invdx_ = 1.0/dx_;
		xlen = L1-L0;
	}

	__host__ void setHostLinspace()
	{
		assert(nx_ > 0);
		cudaCheck(cudaMallocHost(&h_linspace,sizeof(Real)*nx_));
		for (size_t i=0;i<nx_;i++)
		{
			h_linspace[i] = L0_+dx_*i;
		}
	}

	
}; /// End class Grid
