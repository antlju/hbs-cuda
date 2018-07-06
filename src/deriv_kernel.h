#pragma once
#include "derivatives.h"

__device__ void bundleInit(Real *B, Mesh f, const Int j, const Int k, const Int vi)
{
	for (Int bi=-(NG-1);bi<NG+1;bi++)
	{
		B[bIdx(bi,0,vi)] = f(bi-1,j,k,vi);
		B[bIdx(bi,1,vi)] = f(bi-1,j+1,k,vi);
		B[bIdx(bi,2,vi)] = f(bi-1,j,k-1,vi);
		B[bIdx(bi,3,vi)] = f(bi-1,j-1,k,vi);
		B[bIdx(bi,4,vi)] = f(bi-1,j,k+1,vi);
		B[bIdx(bi,5,vi)] = f(bi-1,j+2,k,vi);
		B[bIdx(bi,6,vi)] = f(bi-1,j,k-2,vi);
		B[bIdx(bi,7,vi)] = f(bi-1,j-2,k,vi);
		B[bIdx(bi,8,vi)] = f(bi-1,j,k+2,vi);
	}
}

__global__ void zderivKernel(Mesh f, Mesh df, const Real dx)
{
	const Int ng = f.ng_;
	
	__shared__ Real fs[NY_TILE+2*NG][NZ_TILE+2*NG];

	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;
	
	/// Local indices
	const Int bj = threadIdx.x + ng;
	const Int bk = threadIdx.y + ng;
	const Int bi = 0; /// the "center" of the bundle. This will always be zero
	                  /// for any global index i along the array.
	
	const Int vi=0; /// Just for testing scalar function
		
	/// Thread local bundle, stencil size in every direction:
	Real B[(4*NG+1)*(2*NG+1)];

	/// Initialisation
	bundleInit(B,f,j,k,vi);
	__syncthreads();

	if (j < f.ny_ && k < f.nz_)
	{
		for (Int i=0;i<f.nx_;i++)
		{
			
			/// Load shared memory 
			fs[bj][bk] = f(i+2,j,k);

			/// If at yz-tile edges assign ghost points
			if (bj == NG)
			{
				fs[bj-1][bk] = f(i+2,j-1,k);
				fs[bj-2][bk] = f(i+2,j-2,k);
				fs[bj+NY_TILE][bk] = f(i+2,j+NY_TILE,k);
				fs[bj+NY_TILE+1][bk] = f(i+2,j+NY_TILE+1,k);
			}
			if (bk == NG)
			{
				fs[bj][bk-1] = f(i+2,j,k-1);
				fs[bj][bk-2] = f(i+2,j,k-2);
				fs[bj][bk+NZ_TILE] = f(i+2,j,k+NZ_TILE);
				fs[bj][bk+NZ_TILE+1] = f(i+2,j,k+NZ_TILE+1);
			}
			__syncthreads();

			/// *** ___ Roll the cache ! ___ ***
			/// Load shared tile into local bundle
			for (Int q=0;q<4*NG+1;q++)
			{
				B[bIdx(-2,q,vi)] = B[bIdx(-1,q,vi)];
				B[bIdx(-1,q,vi)] = B[bIdx(0,q,vi)];
				B[bIdx(0,q,vi)] = B[bIdx(1,q,vi)];
				B[bIdx(1,q,vi)] = B[bIdx(2,q,vi)];
			}

			/// Add last element from shared tile
			B[bIdx(NG,0,vi)] = fs[bj][bk];
			B[bIdx(NG,1,vi)] = fs[bj+1][bk];
			B[bIdx(NG,2,vi)] = fs[bj][bk-1];
			B[bIdx(NG,3,vi)] = fs[bj-1][bk];
			B[bIdx(NG,4,vi)] = fs[bj][bk+1];
			B[bIdx(NG,5,vi)] = fs[bj+2][bk];
			B[bIdx(NG,6,vi)] = fs[bj][bk-2];
			B[bIdx(NG,7,vi)] = fs[bj-2][bk];
			B[bIdx(NG,8,vi)] = fs[bj][bk+2];

			/// *** ___ Perform bundle -> pencil operations  ___ ***
			
			
			//df[fIdx(i,j,k)] = B[bIdx(0,3,0)]; //DEBUG
			df(i,j,k,vi) = delz(B,1.0/dx,bi,vi); /// bi should be 0 always!
	
		}
		
		
	}

}
