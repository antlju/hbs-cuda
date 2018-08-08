#pragma once
#include "rk3_coeffs.h"
/// Kernels for updating pressure with the result of the Poisson solver (stored in psi mesh),
/// and enforcing solenoidal condition on u. Calculating the gradient of psi is done
__global__
void update_pressure_kernel(Mesh p, Mesh psi)
{
	const Int j = threadIdx.x + blockIdx.x * blockDim.x;
	const Int k = threadIdx.y + blockIdx.y * blockDim.y;

	if ( j < p.ny_ && k < p.nz_)
	{
		for (Int i=0;i<p.nx_;i++)
		{
			p(i,j,k,0) += psi(i,j,k,0);
		}
	}
}

/// Computes the gradient of psi (scalar field -> vector field)
__global__
void calc_gradpsi_kernel(Mesh psi, Mesh gradpsi, const Real dx)
{
	/// The only diff op will happen on the pressure (grad(p).
	/// Need to investigate if there's any speedup by using shared memory
	/// for only accessing u,rhsk,rhsk_1,
	__shared__ Real psi_smem[1*(NY_TILE+2*NG)*(NZ_TILE+2*NG)];
	
	Shared psi_s(psi_smem,NY_TILE,NZ_TILE,1,NG);

	/// Finite difference coefficients.
	/// A cubic grid is assumed such that dx = dy = dz.
	const Real invdx = 1.0/dx;
	//const Real invdx2 = invdx*invdx;
	
	/// Global indices (for whole array)
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;
	
	/// Local indices (local to block)	
	const Int lj = threadIdx.x;
	const Int lk = threadIdx.y;
	const Int li = 0; /// the "center" of the bundle (finite difference stencil) in any "roll step".
	                  /// This will always be zero for any
	                  /// global index i along the array.

	/// Bundle memory and Bundle pointer to that memory
	Real sB[1*(4*NG+1)*(1+2*NG)];
	Bundle psiBndl(&sB[0],4*NG+1,1); /// Scalar bundle
	
	/// Thread local vector "pencil"
	Real P[3];
	
	/// Initialise for rolling cache
	for (Int vi=0;vi<psi.nvars_;vi++)
	{
		bundleInit(psiBndl,psi,j,k,vi);
	}
	__syncthreads();

	/// Loop over mesh
	if (j < psi.ny_ && k < psi.nz_)
	{
		for (Int i=0;i<psi.nx_;i++)
		{
			///Load shared memory and ghostpts
			loadShared(psi_s,psi,i,j,k,lj,lk); //loadShared() def'd in shared.h
			__syncthreads();
			
			/// *** ___ Roll the cache ! ___ ***
			rollBundleCache(psiBndl,psi_s,lj,lk);

			/// *** ____ Here comes the operations ! ___ ***
			//Compute gradient of the pressure (scalar bundle -> vector pencil)
			sgrad(psiBndl,P,li,invdx,invdx,invdx);

			///Set array values from pencil
			gradpsi(i,j,k,0) = P[0];
			gradpsi(i,j,k,1) = P[1];
			gradpsi(i,j,k,2) = P[2];
	
		}//End for loop over i.
		
	} //End j,k if statement
	
	
} //End calc_gradpsi_kernel()

__global__
void enforce_solenoidal_kernel(Mesh u, Mesh ustar, Mesh gradpsi, SolverParams params, const Int k_rk)
{
	/// Parameters
	const Real rho = 1.0;//params.rho;
	const Real dt = params.h_dt[0];

	/// RK3 coeffs and factor
	const Real alphak = rk_alpha(k_rk);
	const Real gradfac = 2.0*alphak*dt/rho;
	
	/// Global indices (for whole array)
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;

	if ( j < u.ny_ && k < u.nz_)
	{
		for (Int i=0;i<u.nx_;i++)
		{
			for (Int vi=0;vi<u.nvars_;vi++)
			{
				u(i,j,k,vi) = ustar(i,j,k,vi)-gradpsi(i,j,k,vi)*gradfac;
			}
			
		}
		
	}
}
