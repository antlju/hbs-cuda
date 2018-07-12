#include "common.h"
#include "grid.h"
#include "timer.h"

#include <iostream>

#include "bundle.h"

/// Instantiate global objects
Mesh u(NX,NY,NZ,3);
Grid grid(NX,NY,NZ,0.0,2*M_PI);
Timer timer;

__global__ void bundleClassKernel(Mesh f, Grid grid)
{
	__shared__ Real fs[3][NY_TILE+2*NG][NZ_TILE+2*NG];
	
	const Int ng = f.ng_;
	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;
	
	/// Local indices
	const Int lj = threadIdx.x + ng;
	const Int lk = threadIdx.y + ng;
	const Int li = 0; /// the "center" of the bundle (fd stencil) in any "roll step".
	                  /// This will always be zero for any
	                  /// global index i along the array.
	
	Real vB[3*(4*NG+1)*(1+2*NG)];
	Bundle Bndl(&vB[0],4*NG+1,3);

	for (Int vi=0;vi<f.nvars_;vi++)
	{
		bundleInit(Bndl,f,j,k,vi);
	}

	if (j < f.ny_ && k < f.nz_)
	{
		for (Int i=0;i<f.nx_;i++)
		{
			for (Int vi=0;vi<Bndl.nvars_;vi++)
			{
				Int qi = (i+lj+lk) % 9;
				f(i,j,k,vi) = qi;
			}
		}
	}
	
	
}


Int main()
{
	std::cout << "Executing w/ size: (N=" << NN << ")^3" << std::endl;
	u.allocateHost(); u.allocateDevice();

	grid.setHostLinspace();
//	initHost(u,grid);
	//u.print();
	
	timer.createEvents();
	u.copyToDevice();
	dim3 tpb(NY_TILE,NZ_TILE); 
	dim3 blx(NN/NY_TILE,NN/NZ_TILE);
	timer.recordStart();
 
	pbc_x_kernel<<<blx,tpb>>>(u);
	pbc_y_kernel<<<blx,tpb>>>(u);
	pbc_z_kernel<<<blx,tpb>>>(u);

	bundleClassKernel<<<blx,tpb>>>(u,grid);
//curlKernel<<<blx,tpb>>>(u,du,grid);
	
	timer.recordStop();
	timer.synch();

	u.copyFromDevice();

	u.print();
	
//testCurl(du);
	
	timer.print();
	

	

	return 0;
};

     
