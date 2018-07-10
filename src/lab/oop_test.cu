#define __ROLLING_CACHE__ 1 /// Enables the rolling cache NZ/NY_TILE defs

#include "common.h"
#include "fmesh.h"
#include "grid.h"
#include "timer.h"
#include <iostream>

/// Typedefs mesh template classes with number of ghostpoints NG
/// (from fd_params.h)
typedef fMesh<Real,NG> Mesh;

/// This kernel demonstrates that I can access obj member variables such as
/// nx_,ny_ and indexing member methods.
__global__ void testKernel(Mesh u)
{
	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;

	if (j < u.ny_ && k < u.nz_)
	{
		for (size_t i=0;i<u.nx_;i++)
		{
			u(i,j,k) = u.indx(i,j,k);
		}
	}
}

/// Instantiate global objects

Mesh u(NX,NY,NZ,1);
Grid grid(NX,NY,NZ,0.0,2*M_PI);
Timer timer;

Int main()
{
	std::cout << "Executing w/ size: (N =" << NN << ")^3" << std::endl;
	u.allocateHost(); u.allocateDevice();

	timer.createEvents();
	//globalMesh.copyToDevice();
	dim3 tpb(NY_TILE,NZ_TILE); 
	dim3 blx(NN/NY_TILE,NN/NZ_TILE);
	timer.recordStart();
	testKernel<<<blx,tpb>>>(u);
	timer.recordStop();
	timer.synch();
	
	timer.print();
	
	u.copyFromDevice();
	u.print();

	
	return 0;
};

     
