#include "common.h"
#include "grid.h"
#include "timer.h"

#include <iostream>

#include "bundle.h"
#include "derivatives.h"
#include "shared.h"

/// Instantiate global objects
Mesh u(NX,NY,NZ,3);
Grid grid(NX,NY,NZ,0.0,2*M_PI);
Timer timer;

__global__ void vecBundleKernel(Mesh f, Grid grid)
{
	__shared__ Real smem[3*(NY_TILE+2*NG)*(NZ_TILE+2*NG)];

	Shared fs(smem,NY_TILE,NZ_TILE,3,NG); /// Shared memory object for indexing
	
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

	/// Bundle memory and Bundle pointer to that memory
	Real vB[3*(4*NG+1)*(1+2*NG)];
	Bundle Bndl(&vB[0],4*NG+1,3);

	/// Initialise for rolling cache
	for (Int vi=0;vi<f.nvars_;vi++)
	{
		bundleInit(Bndl,f,j,k,vi);
	}

	if (j < f.ny_ && k < f.nz_)
	{
		for (Int i=0;i<f.nx_;i++)
		{
			for (Int vi=0;vi<f.nvars_;vi++)
			{
				/// Load the shared memory tile 
				fs[vi][lj][lk] = f(i+2,j,k,vi);
				
				/// If at yz-tile edges assign ghost points
				if (lj == NG)
				{
					fs[vi][lj-1][lk] = f(i+2,j-1,k,vi);
					fs[vi][lj-2][lk] = f(i+2,j-2,k,vi);
					fs[vi][lj+NY_TILE][lk] = f(i+2,j+NY_TILE,k,vi);
					fs[vi][lj+NY_TILE+1][lk] = f(i+2,j+NY_TILE+1,k,vi);
				}
				if (lk == NG)
				{
					fs[vi][lj][lk-1] = f(i+2,j,k-1,vi);
					fs[vi][lj][lk-2] = f(i+2,j,k-2,vi);
					fs[vi][lj][lk+NZ_TILE] = f(i+2,j,k+NZ_TILE,vi);
					fs[vi][lj][lk+NZ_TILE+1] = f(i+2,j,k+NZ_TILE+1,vi);
				}
			}
			__syncthreads();

			
			/// *** ___ Roll the cache ! ___ ***
			/// Load shared tile into local bundle
			for (Int vi=0;vi<Bndl.nvars_;vi++)
			{
				for (Int q=0;q<4*NG+1;q++)
				{
					Bndl(-2,q,vi) = Bndl(-1,q,vi);
					Bndl(-1,q,vi) = Bndl(0,q,vi);
					Bndl(0,q,vi) = Bndl(1,q,vi);
					Bndl(1,q,vi) = Bndl(2,q,vi);
				}
			

				/// Add last element from shared tile
				Bndl(NG,0,vi) = fs[vi][lj][lk];
				Bndl(NG,1,vi) = fs[vi][lj+1][lk];
				Bndl(NG,2,vi) = fs[vi][lj][lk-1];
				Bndl(NG,3,vi) = fs[vi][lj-1][lk];
				Bndl(NG,4,vi) = fs[vi][lj][lk+1];
				Bndl(NG,5,vi) = fs[vi][lj+2][lk];
				Bndl(NG,6,vi) = fs[vi][lj][lk-2];
				Bndl(NG,7,vi) = fs[vi][lj-2][lk];
				Bndl(NG,8,vi) = fs[vi][lj][lk+2];
			}

			/// Do operations on bundle:
			f(i,j,k,0) = delx(Bndl,1.0/grid.dx_,li,0);
			       
		}//End for loop over i.
		
	} //End j,k if statement
	
	
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
				f.h_data[f.indx(i,j,k,0)] = x[i];
				f.h_data[f.indx(i,j,k,1)] = 2;
				f.h_data[f.indx(i,j,k,2)] = 3;
			}
		}
	}
}


Int main()
{
	std::cout << "Executing w/ size: (N=" << NN << ")^3" << std::endl;
	u.allocateHost(); u.allocateDevice();

	grid.setHostLinspace();
	initHost(u,grid);
	//u.print();
	
	timer.createEvents();
	u.copyToDevice();
	dim3 tpb(NY_TILE,NZ_TILE); 
	dim3 blx(NN/NY_TILE,NN/NZ_TILE);
	timer.recordStart();
 
	//pbc_x_kernel<<<blx,tpb>>>(u);
	//pbc_y_kernel<<<blx,tpb>>>(u);
	//pbc_z_kernel<<<blx,tpb>>>(u);

	vecBundleKernel<<<blx,tpb>>>(u,grid);
//curlKernel<<<blx,tpb>>>(u,du,grid);
	
	timer.recordStop();
	timer.synch();

	u.copyFromDevice();

	u.print();
	
//testCurl(du);
	
	timer.print();
	

	

	return 0;
};

     
