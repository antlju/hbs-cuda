#pragma once
#include "rk3_coeffs.h"

/// *******************************************************************************************************


/// Calculate u* = u+(2*dt*(alpha(k)/rho))*grad(p)+(dt*beta(k))*RHSk+(dt*gamma(k))*RHSk_1
__global__
void calculate_uStar_kernel(Mesh u, Mesh rhsk, Mesh rhsk_1, Mesh p, Mesh ustar,
			    SolverParams params, const Real dx, const Int k_rk)
{
	/// The only diff op will happen on the pressure (grad(p).
	/// Need to investigate if there's any speedup by using shared memory
	/// for only accessing u,rhsk,rhsk_1,
	__shared__ Real p_smem[1*(NY_TILE+2*NG)*(NZ_TILE+2*NG)];
	
	Shared p_s(p_smem,NY_TILE,NZ_TILE,1,NG);
	
	/// Parameters
	const Real rho = params.rho;
	const Real dt = params.h_dt[0];

	/// Set RK3 coefficients
	const Real alphak = rk_alpha(k_rk);
        const Real betak = rk_beta(k_rk);
        const Real gammak = rk_gamma(k_rk);
	
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

	/* DEBUGGING PRINT
	if (j == 5 && k == 5)
	{
		printf("ustar dt: %f, rho: %f, alphak: %f, betak: %f, gammak: %f \n",dt, rho, alphak, betak, gammak);
	}
	*/
	
	/// Bundle memory and Bundle pointer to that memory
	Real sB[1*(4*NG+1)*(1+2*NG)];
	Bundle pBndl(&sB[0],4*NG+1,1); /// Scalar bundle
	
	/// Local vector "pencil"
	Real P[3];
	
	/// Initialise for rolling cache
	for (Int vi=0;vi<p.nvars_;vi++)
	{
		bundleInit(pBndl,p,j,k,vi);
	}
	__syncthreads();

	const Real rhskfac = (dt*betak),pnclfac = 2*alphak*dt/rho,rhsk_1fac = dt*gammak;
	
	/// Loop over mesh
	if (j < u.ny_ && k < u.nz_)
	{
		for (Int i=0;i<u.nx_;i++)
		{
			///Load shared memory and ghostpts
			loadShared(p_s,p,i,j,k,lj,lk); //loadShared() def'd in shared.h
			__syncthreads();
			
			/// *** ___ Roll the cache ! ___ ***
			rollBundleCache(pBndl,p_s,lj,lk);

			/// *** ____ Here comes the operations ! ___ ***
			//Compute gradient of the pressure (scalar bundle -> vector pencil)
			sgrad(pBndl,P,li,invdx,invdx,invdx);

			ustar(i,j,k,0) = u(i,j,k,0)+(rhsk(i,j,k,0)*rhskfac)+(rhsk_1(i,j,k,0)*rhsk_1fac)-(P[0]*pnclfac);
			ustar(i,j,k,1) = u(i,j,k,1)+(rhsk(i,j,k,1)*rhskfac)+(rhsk_1(i,j,k,1)*rhsk_1fac)-(P[1]*pnclfac);
			ustar(i,j,k,2) = u(i,j,k,2)+(rhsk(i,j,k,2)*rhskfac)+(rhsk_1(i,j,k,2)*rhsk_1fac)-(P[2]*pnclfac);
			
			/*
			for (Int vi=0;vi<ustar.nvars_;vi++)
			{
				if (k_rk == 1) /// Optimise away access to rhsk_1 when gammak == 0.0.
				{
					ustar(i,j,k,vi) = u(i,j,k,vi) + rhsk(i,j,k,vi)*(dt*betak)
						- P[vi]*(2*alphak*dt/rho);
				}
				else
				{
					ustar(i,j,k,vi) = u(i,j,k,vi) + rhsk(i,j,k,vi)*(dt*betak)
						+ rhsk_1(i,j,k,vi)*(dt*gammak) - P[vi]*(2*alphak*dt/rho);
				}
			}
			*/
	
		}//End for loop over i.
		
	} //End j,k if statement
	
	
} //End calculate_uStar_kernel()


/// *******************************************************************************************************


/// Kernel to calculate the source term for Poisson eq: div(u*)*rho/(2*alpha^k*dt)
__global__
void calc_divergence_uStar_kernel(Mesh ustar, Mesh psi, SolverParams params, const Real dx, const Int k_rk)
{
	__shared__ Real ustar_smem[3*(NY_TILE+2*NG)*(NZ_TILE+2*NG)];
	
	Shared ustar_s(ustar_smem,NY_TILE,NZ_TILE,3,NG);
	
	/// Parameters
	const Real rho = params.rho;
	const Real dt = params.h_dt[0];

	/// Set RK3 coefficients
	const Real alphak = rk_alpha(k_rk);

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

	/* DEBUGGING PRINT
	if (j == 5 && k == 5)
	{
		printf("div ustar dt: %f, rho: %f, alphak: %f \n",dt, rho, alphak);
	}
	*/
	
	/// Factor for multiplying div(u*)
	const Real divu_fac = rho/(2*alphak*dt);

	/// Bundle memory and Bundle pointer to that memory
	Real vB[3*(4*NG+1)*(1+2*NG)];
	Bundle ustarBndl(&vB[0],4*NG+1,3); /// Vector bundle
	
	/// Thread local scalar "pencil"
	Real P[1];
	
	/// Initialise for rolling cache
	for (Int vi=0;vi<ustar.nvars_;vi++)
	{
		bundleInit(ustarBndl,ustar,j,k,vi);
	}
	__syncthreads();

	/// Loop over mesh
	if (j < ustar.ny_ && k < ustar.nz_)
	{
		for (Int i=0;i<ustar.nx_;i++)
		{
			///Load shared memory and ghostpts
			loadShared(ustar_s,ustar,i,j,k,lj,lk); //loadShared() def'd in shared.h
			__syncthreads();
			
			/// *** ___ Roll the cache ! ___ ***
			rollBundleCache(ustarBndl,ustar_s,lj,lk);

			/// *** ____ Here comes the operations ! ___ ***
			//Compute divergence of ustar (vector bundle -> scalar pencil)
			divergence(ustarBndl,P,li,invdx,invdx,invdx);
			/// Add to poisson solver input mesh (psi) with proper factor
			psi(i,j,k,0) = P[0]*divu_fac;
			
		}//End for loop over i.
		
	} //End j,k if statement
	
	
} //End calc_divergence_uStar_kernel()


/// *******************************************************************************************************
