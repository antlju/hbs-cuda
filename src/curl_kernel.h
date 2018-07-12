#pragma once
#include "derivatives.h" //includes delx(),dely(),delz()

__device__ void curl(const Real *B, Real *P, const Real xfac, const Real yfac, const Real zfac)
{
        Real dx,dy,dz;
	/// Compute omega_1 = d_2u_3-d_3u_2
	dy = dely(B,yfac,0,2);
	dz = delz(B,zfac,0,1); 
	P[0] = dy-dz;
        
	/// Compute omega_2 = d_3u_1-d_1u_3
	dz = delz(B,zfac,0,0);
	dx = delx(B,xfac,0,2);
	P[1] = dz-dx;
        
	/// Compute omega_3 = d_1u_2-d_2u_1
	dx = delx(B,xfac,0,1);
	dy = dely(B,yfac,0,0);
	P[3] = dx-dy;
       
}

__global__ void curlKernel(Mesh f, Mesh df, Grid grid)
{
	__shared__ Real fs[3][NY_TILE+2*NG][NZ_TILE+2*NG];


	const Int ng = f.ng_;
	const Real dx = grid.dx_;
	
	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;
	
	/// Local indices
	const Int bj = threadIdx.x + ng;
	const Int bk = threadIdx.y + ng;
	const Int bi = 0; /// the "center" of the bundle (fd stencil) in any "roll step".
	                  /// This will always be zero for any
	                  /// global index i along the array.
	
	/// Thread local vector bundle, stencil size in every direction:
	Real vB[3*(4*NG+1)*(2*NG+1)];

	Real vP[3]; //vector pencil
	
	/// Initialisation
	for (Int vi=0;vi<f.nvars_;vi++)
		bundleInit(vB,f,j,k,vi);
	
	__syncthreads();

	if (j < f.ny_ && k < f.nz_)
	{
		for (Int i=0;i<f.nx_;i++)
		{

			for (Int vi=0;vi<f.nvars_;vi++)
			{
				/// Load shared memory 
				fs[vi][bj][bk] = f(i+2,j,k,vi);
				//fs[1][bj][bk] = f(i+2,j,k,1);
				//fs[2][bj][bk] = f(i+2,j,k,2);

				/// If at yz-tile edges assign ghost points
				if (bj == NG)
				{
					fs[vi][bj-1][bk] = f(i+2,j-1,k,vi);
					fs[vi][bj-2][bk] = f(i+2,j-2,k,vi);
					fs[vi][bj+NY_TILE][bk] = f(i+2,j+NY_TILE,k,vi);
					fs[vi][bj+NY_TILE+1][bk] = f(i+2,j+NY_TILE+1,k,vi);
				}
				if (bk == NG)
				{
					fs[vi][bj][bk-1] = f(i+2,j,k-1,vi);
					fs[vi][bj][bk-2] = f(i+2,j,k-2,vi);
					fs[vi][bj][bk+NZ_TILE] = f(i+2,j,k+NZ_TILE,vi);
					fs[vi][bj][bk+NZ_TILE+1] = f(i+2,j,k+NZ_TILE+1,vi);
				}
			}
			__syncthreads();

			/// *** ___ Roll the cache ! ___ ***
			/// Load shared tile into local bundle
			for (Int q=0;q<4*NG+1;q++)
			{
				for (Int vi=0;vi<f.nvars_;vi++)
				{
					vB[bIdx(-2,q,vi)] = vB[bIdx(-1,q,vi)];
					vB[bIdx(-1,q,vi)] = vB[bIdx(0,q,vi)];
					vB[bIdx(0,q,vi)] = vB[bIdx(1,q,vi)];
					vB[bIdx(1,q,vi)] = vB[bIdx(2,q,vi)];
				}
			}

			/// Add last element from shared tile
			for (Int vi=0;vi<f.nvars_;vi++)
			{
				vB[bIdx(NG,0,vi)] = fs[vi][bj][bk];
				vB[bIdx(NG,1,vi)] = fs[vi][bj+1][bk];
				vB[bIdx(NG,2,vi)] = fs[vi][bj][bk-1];
				vB[bIdx(NG,3,vi)] = fs[vi][bj-1][bk];
				vB[bIdx(NG,4,vi)] = fs[vi][bj][bk+1];
				vB[bIdx(NG,5,vi)] = fs[vi][bj+2][bk];
				vB[bIdx(NG,6,vi)] = fs[vi][bj][bk-2];
				vB[bIdx(NG,7,vi)] = fs[vi][bj-2][bk];
				vB[bIdx(NG,8,vi)] = fs[vi][bj][bk+2];
			}

			/// *** ___ Perform bundle -> pencil operations  ___ ***
			
			
			//df[fIdx(i,j,k)] = B[bIdx(0,3,0)]; //DEBUG
			//df(i,j,k,vi) = delz(B,1.0/dx,bi,vi); /// bi should be 0 always!
			curl(vB,vP,1.0/dx,1.0/dx,1.0/dx);

			df(i,j,k,0) = vP[0];
			df(i,j,k,1) = vP[1];
			df(i,j,k,2) = vP[2];
		}
		
		
	}

}
