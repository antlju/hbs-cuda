#include "common.h"
#include "grid.h"
#include "timer.h"

#include <iostream>


/// Instantiate global objects
Mesh u(NX,NY,NZ,3);
Mesh du(NX,NY,NZ,1);
Grid grid(NX,NY,NZ,0.0,2*M_PI);
Timer timer;



__global__ void laplKernel(Mesh f, Mesh df, Grid grid)
{
	__shared__ Real smem[3*(NY_TILE+2*NG)*(NZ_TILE+2*NG)];

	Shared fs(smem,NY_TILE,NZ_TILE,3,NG); /// Shared memory object for indexing

	const Real invdx = 1.0/grid.dx_;
	const Real invdx2 = invdx*invdx;
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
	Real P[1]; /// Local scalar "pencil"

	
	/// Initialise for rolling cache
	for (Int vi=0;vi<f.nvars_;vi++)
	{
		bundleInit(Bndl,f,j,k,vi);
	}
	__syncthreads();
       
	
	//Bndl(-1,0,0) = f(-2,j,k,0);
	//Bndl(0,0,0) = f(-1,j,k,0);
	//Bndl(1,0,0) = f(0,j,k,0);
	//Bndl(2,0,0) = f(1,j,k,0);

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
			rollBundleCache(Bndl,fs,lj,lk);
			//Bndl(-2,0,0) = Bndl(-1,0,0);
			//Bndl(-1,0,0) = Bndl(0,0,0);
			//Bndl(0,0,0) = Bndl(1,0,0);
			//Bndl(1,0,0) = Bndl(2,0,0);
			//Bndl(2,0,0) = f(i+2,j,k);
			
			/// Do operations on bundle:
			//curl(Bndl,P,li,invdx,invdx,invdx);
			lapl(Bndl,P,li,invdx2,invdx2,invdx2);
			// Set pencil
			//df(i,j,k,0) = del2z(Bndl,invdx2,li,0);
			//df(i,j,k,0) = dely(Bndl,invdx,li,1);
			//df(i,j,k,0) = fabs(Bndl(0,0,0)-f(i,j,k,0));
			df(i,j,k,0) = P[0];
			//df(i,j,k,0) = 1; df(i,j,k,1) = 2; df(i,j,k,2) = 3;
			//df(i,j,k,0) = delz(Bndl,invdx,li,0);
			       
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
				//f.h_data[f.indx(i,j,k,0)] = 1.0*sin(x[i])+2.0*sin(x[j])+3.0*sin(x[k]);
				//f.h_data[f.indx(i,j,k,2)] = sin(x[k]);
				/// Initialises f = (1z,2x,3y) -> curl(f) = (3,1,2)
				f.h_data[f.indx(i,j,k,0)] = sin(x[j]);
				f.h_data[f.indx(i,j,k,1)] = sin(x[j]);
				f.h_data[f.indx(i,j,k,2)] = 0;
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
	
	
	laplKernel<<<blx,tpb>>>(u,du,grid);
	//zderivKernel<<<blx,tpb>>>(u,du,grid.dx_);
//curlKernel<<<blx,tpb>>>(u,du,grid);
	
	timer.recordStop();
	timer.sync();

	du.copyFromDevice();
	//printf("%.6f \t %.6f \n",du.h_data[du.indx(0,0,1,0)],du.h_data[du.indx(du.nx_-1,du.ny_-1,du.nz_-1,0)]);
	//du.print();
	
//testCurl(du);
	du.print();
	timer.print();
	
	return 0;
};

     
