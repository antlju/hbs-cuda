#pragma once

__device__
void set_force(Real *P, const Int i, Grid grid, const Real kf)
{
	Real Amp = 1.0;
	Real *x = grid.d_linspace;
	P[0] = Amp*sin(kf*x[i]);
	P[1] = 0.0;
	P[2] = 0.0;
}

__global__
void RHSk_copy_kernel(Mesh rhsk, Mesh rhsk_1)
{
	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;

	if (j < rhsk.ny_ && k < rhsk.nz_)
	{
		for (Int i=0;i<rhsk.nx_;i++)
		{
			rhsk_1(i,j,k,0) = rhsk(i,j,k,0);
			rhsk_1(i,j,k,1) = rhsk(i,j,k,1);
			rhsk_1(i,j,k,2) = rhsk(i,j,k,2);
		}
	}
}

__global__
void calculate_RHSk_kernel(Mesh u, Mesh rhsk, Grid grid, SolverParams params)
{
	/// u and RHSk are vector fields
	__shared__ Real smem[3*(NY_TILE+2*NG)*(NZ_TILE+2*NG)];

	Shared fs(smem,NY_TILE,NZ_TILE,3,NG); /// Shared memory object for indexing

	/// Finite difference coefficients.
	/// A cubic grid is assumed such that dx = dy = dz.
	const Real invdx = 1.0/grid.dx_;
	const Real invdx2 = invdx*invdx;
	
	/// Global indices (for whole array)
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;
	
	/// Local indices (local to block)	
	const Int lj = threadIdx.x;
	const Int lk = threadIdx.y;
	const Int li = 0; /// the "center" of the bundle (fd stencil) in any "roll step".
	                  /// This will always be zero for any
	                  /// global index i along the array.

	/// Bundle memory and Bundle pointer to that memory
	Real vB[3*(4*NG+1)*(1+2*NG)];
	Bundle Bndl(&vB[0],4*NG+1,3);
	
	/// Local vector "pencil"
	Real P[3];
	
	/// Initialise for rolling cache
	for (Int vi=0;vi<u.nvars_;vi++)
	{
		bundleInit(Bndl,u,j,k,vi);
	}
	__syncthreads();

	//const Int vi = 0;
	 
	if (j < u.ny_ && k < u.nz_)
	{
		for (Int i=0;i<u.nx_;i++)
		{
			///Load shared memory and ghostpts
			loadShared(fs,u,
				   i,j,k,
				   lj,lk); //loadShared() def'd in shared.h
			__syncthreads();
			
			/// *** ___ Roll the cache ! ___ ***
			rollBundleCache(Bndl,fs,lj,lk);

			/// *** ____ Here comes the operations ! ___ ***
			/// Compute (u dot grad)u and store in P.
			udotgradu(Bndl,P,li,invdx,invdx,invdx);
			/// Add P to RHSk.
			rhsk(i,j,k,0) += P[0]; rhsk(i,j,k,1) += P[1]; rhsk(i,j,k,2) += P[2];

			/// Compute vector laplacian and store in P.
			vlapl(Bndl,P,li,invdx2,invdx2,invdx2);
			/// Add P to RHSk.
			rhsk(i,j,k,0) += P[0]; rhsk(i,j,k,1) += P[1]; rhsk(i,j,k,2) += P[2];

			/// Compute force and store in P.
			set_force(P,k,grid,params.kf);
			/// Add P to RHSk.
			rhsk(i,j,k,0) += P[0]; rhsk(i,j,k,1) += P[1]; rhsk(i,j,k,2) += P[2];

		}//End for loop over i.
		
	} //End j,k if statement
	
	
} //End calculate_RHSk_kernel()
