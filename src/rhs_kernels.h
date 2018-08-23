#pragma once

__device__
void set_force(Real *P, const Int i, Grid grid, const Real kf,const Real f0)
{
	Real *x = grid.d_linspace;
	//Real Amp = 1.0/40;
	P[0] = f0*sin(kf*x[i]);
	//P[0] = 1.0;
	P[1] = 0.0;
	P[2] = 0.0;
}

/*
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
*/

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
	Real P1[3];
	Real P2[3];
	Real P3[3];
	
	const Real nu = params.viscosity;
	//printf("nu : %f, 1/20: %f \n",nu,1.0/20);
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
			udotgradu(Bndl,P1,li,invdx,invdx,invdx);
			/// _Set_ P to RHSk. (So that we dont add to RHSk from prev step) 
			rhsk(i,j,k,0) = P1[0]; rhsk(i,j,k,1) = P1[1]; rhsk(i,j,k,2) = P1[2];
			
			/// Compute vector laplacian and store in P.
			vlapl(Bndl,P2,li,invdx*invdx,invdx*invdx,invdx*invdx);
			/// Add P to RHSk.
			rhsk(i,j,k,0) += P2[0]*nu; rhsk(i,j,k,1) += P2[1]*nu; rhsk(i,j,k,2) += P2[2]*nu;

			/// Compute force and store in P.
			set_force(P3,j,grid,params.kf,params.f0);
			/// Add P to RHSk.
			rhsk(i,j,k,0) += P3[0]; rhsk(i,j,k,1) += P3[1]; rhsk(i,j,k,2) += P3[2];
	

		}//End for loop over i.
		
	} //End j,k if statement
	
	
} //End calculate_RHSk_kernel()
