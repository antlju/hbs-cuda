#include <iostream>
#include "common.h"

typedef double Real;

/// Instantiate global objects
Timer timer;
Mesh u(NX,NY,NZ,1);
Mesh du(1,NY,NZ,1);
Grid grid(NX,NY,NZ,0.0,2*M_PI);

__global__ void reduce1(Mesh f, Mesh out)
{
	/// Global indices
	const Int j = blockIdx.x;
	const Int k = threadIdx.x;

	if (j < f.ny_ && k < f.nz_)
	{
		out(0,j,0) += f(0,j,k); 
	}
}

__global__
void reduce0_noshared(Mesh f, Mesh out)
{
	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;
	

	if (j < f.ny_ && k < f.nz_)
	{
		for (Int i=0;i<f.nx_;i++)
		{
			out(0,j,k,0) += f(i,j,k,0);
		}
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
				f.h_data[f.indx(i,j,k,0)] = 1.0;
			}
		}
	}
}

Int main()
{
	timer.createEvents();
	std::cout << "Executing w/ size: (N=" << NN << ")^3" << std::endl;
	u.allocateHost(); u.allocateDevice();
	du.allocateHost(); du.allocateDevice();
	
	grid.setHostLinspace();
	initHost(u,grid);
	u.copyToDevice();

	dim3 tpb0(NY_TILE,NZ_TILE); 
	dim3 blx0(NN/NY_TILE,NN/NZ_TILE);
	dim3 tpb1(NZ_TILE*NZ_TILE);
	dim3 blx1(NZ_TILE*NZ_TILE);
	timer.recordStart();
	reduce0_noshared<<<blx0,tpb0>>>(u,du);
	reduce1<<<blx1,tpb1>>>(du,du);
	timer.recordStop();
	timer.sync();
	timer.print();
	du.copyFromDevice();
	std::cout << du.h_data[du.indx(0,0,0)] << " " << NY*NX*NZ << std::endl;
	du.print();
	
	return 0;
}

     
