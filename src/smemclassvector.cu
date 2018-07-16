#include "common.h"
#include "grid.h"
#include "timer.h"

#include <iostream>


/// Instantiate global objects
Mesh u(NX,NY,NZ,3);
Mesh du(NX,NY,NZ,3);
Grid grid(NX,NY,NZ,0.0,2*M_PI);
Timer timer;

__device__ void rollBundleCacheNoShared(Bundle Bndl, Mesh f, const Int i, const Int j, const Int k)
{
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
		Bndl(NG,0,vi) = f(i,j,k,vi);
		Bndl(NG,1,vi) = f(i,j+1,k,vi);
		Bndl(NG,2,vi) = f(i,j,k-1,vi);
		Bndl(NG,3,vi) = f(i,j-1,k,vi);
		Bndl(NG,4,vi) = f(i,j,k+1,vi);
		Bndl(NG,5,vi) = f(i,j+2,k,vi);
		Bndl(NG,6,vi) = f(i,j,k-2,vi);
		Bndl(NG,7,vi) = f(i,j-2,k,vi);
		Bndl(NG,8,vi) = f(i,j,k+2,vi);
	}
}


__device__ void rollBundleCache(Bundle Bndl, Shared fs, const Int lj, const Int lk)
{
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
		Bndl(2,0,vi) = fs(lj,lk,vi);
		Bndl(2,1,vi) = fs(lj+1,lk,vi);
		Bndl(2,2,vi) = fs(lj,lk-1,vi);
		Bndl(2,3,vi) = fs(lj-1,lk,vi);
		Bndl(2,4,vi) = fs(lj,lk+1,vi);
		Bndl(2,5,vi) = fs(lj+2,lk,vi);
		Bndl(2,6,vi) = fs(lj,lk-2,vi);
		Bndl(2,7,vi) = fs(lj-2,lk,vi);
		Bndl(2,8,vi) = fs(lj,lk+2,vi);
	}
}

__global__ void smemClassKernel(Mesh f, Mesh df, Grid grid)
{
	__shared__ Real smem[3*(NY_TILE+2*NG)*(NZ_TILE+2*NG)];

	Shared fs(smem,NY_TILE,NZ_TILE,3,NG); /// Shared memory object for indexing
	
	const Int ng = f.ng_;
	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;
	
	/// Local indices	
	const Int lj = threadIdx.x;
	const Int lk = threadIdx.y;
	const Int li = 0; /// the "center" of the bundle (fd stencil) in any "roll step".
	                  /// This will always be zero for any
	                  /// global index i along the array.

	/// Bundle memory and Bundle pointer to that memory
	Real vB[3*(4*NG+1)*(1+2*NG)];
	//Real sB[(4*NG+1)*(1+2*NG)];
	Bundle Bndl(&vB[0],4*NG+1,3);

	/// Initialise for rolling cache
	for (Int vi=0;vi<f.nvars_;vi++)
	{
		bundleInit(Bndl,f,j,k,vi);
	}
	__syncthreads();

	const Int vi = 0;
	 
	if (j < f.ny_ && k < f.nz_)
	{
		for (Int i=0;i<f.nx_;i++)
		{
			///Load shared memory and ghostpts
			loadShared(fs,f,
				   i,j,k,
				   lj,lk); //loadShared() def'd in shared.h
			//fs(lk,lj,vi) = f(i+2,j,k);
			__syncthreads();
			
			/// *** ___ Roll the cache ! ___ ***
			/// Load shared tile into local bundle
			//rollBundleCacheNoShared(Bndl,f,i+2,j,k); //Time taken: 0.0583 ms for N=8
			rollBundleCache(Bndl,fs,lj,lk);

			/// Do operations on bundle:	
			df(i,j,k,0) = delz(Bndl,1.0/grid.dx_,li,2);
			       
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
				f.h_data[f.indx(i,j,k,2)] = sin(x[k]);
				//f.h_data[f.indx(i,j,k,0)] = f.indx(i,j,k,0);//sin(x[k]);
				//f.h_data[f.indx(i,j,k,1)] = 2*(x[j]+1);
				//f.h_data[f.indx(i,j,k,2)] = 3*(x[k]+1);
			}
		}
	}
}


Int main()
{
	std::cout << "Executing w/ size: (N=" << NN << ")^3" << std::endl;
	u.allocateHost(); u.allocateDevice();
	du.allocateHost(); du.allocateDevice();
	
	grid.setHostLinspace();
	initHost(u,grid);
	//u.print();
	
	timer.createEvents();
	u.copyToDevice();
	dim3 tpb(NY_TILE,NZ_TILE); 
	dim3 blx(NN/NY_TILE,NN/NZ_TILE);
	timer.recordStart();
 
	pbc_x_kernel<<<blx,tpb>>>(u);
	pbc_y_kernel<<<blx,tpb>>>(u);
	pbc_z_kernel<<<blx,tpb>>>(u);

	smemClassKernel<<<blx,tpb>>>(u,du,grid);
	//zderivKernel<<<blx,tpb>>>(u,du,grid.dx_);
//curlKernel<<<blx,tpb>>>(u,du,grid);
	
	timer.recordStop();
	timer.sync();

	du.copyFromDevice();
	printf("%.6f \t %.6f \n",du.h_data[du.indx(0,0,1,0)],du.h_data[du.indx(du.nx_-1,du.ny_-1,du.nz_-1,0)]);
	//du.print();
	
//testCurl(du);
	
	timer.print();
	
	return 0;
};

     
