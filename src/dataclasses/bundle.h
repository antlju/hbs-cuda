#pragma once

/// Initialise bundle by loading from global memory array
__device__
void bundleInit(Bundle Bndl, Mesh f, const Int j, const Int k, const Int vi)
{
		Bndl(-1,0,vi) = f(-2,j,k,vi);
		Bndl(0,0,vi) = f(-1,j,k,vi);
		Bndl(1,0,vi) = f(0,j,k,vi);
		Bndl(2,0,vi) = f(1,j,k,vi);

		Bndl(-1,1,vi) = f(-2,j+1,k,vi);
		Bndl(0,1,vi) = f(-1,j+1,k,vi);
		Bndl(1,1,vi) = f(0,j+1,k,vi);
		Bndl(2,1,vi) = f(1,j+1,k,vi);

		Bndl(-1,2,vi) = f(-2,j,k-1,vi);
		Bndl(0,2,vi) = f(-1,j,k-1,vi);
		Bndl(1,2,vi) = f(0,j,k-1,vi);
		Bndl(2,2,vi) = f(1,j,k-1,vi);

		Bndl(-1,3,vi) = f(-2,j-1,k,vi);
		Bndl(0,3,vi) = f(-1,j-1,k,vi);
		Bndl(1,3,vi) = f(0,j-1,k,vi);
		Bndl(2,3,vi) = f(1,j-1,k,vi);

		Bndl(-1,4,vi) = f(-2,j,k+1,vi);
		Bndl(0,4,vi) = f(-1,j,k+1,vi);
		Bndl(1,4,vi) = f(0,j,k+1,vi);
		Bndl(2,4,vi) = f(1,j,k+1,vi);

		Bndl(-1,5,vi) = f(-2,j+2,k,vi);
		Bndl(0,5,vi) = f(-1,j+2,k,vi);
		Bndl(1,5,vi) = f(0,j+2,k,vi);
		Bndl(2,5,vi) = f(1,j+2,k,vi);

		Bndl(-1,6,vi) = f(-2,j,k-2,vi);
		Bndl(0,6,vi) = f(-1,j,k-2,vi);
		Bndl(1,6,vi) = f(0,j,k-2,vi);
		Bndl(2,6,vi) = f(1,j,k-2,vi);

		Bndl(-1,7,vi) = f(-2,j-2,k,vi);
		Bndl(0,7,vi) = f(-1,j-2,k,vi);
		Bndl(1,7,vi) = f(0,j-2,k,vi);
		Bndl(2,7,vi) = f(1,j-2,k,vi);

		Bndl(-1,8,vi) = f(-2,j,k+2,vi);
		Bndl(0,8,vi) = f(-1,j,k+2,vi);
		Bndl(1,8,vi) = f(0,j,k+2,vi);
		Bndl(2,8,vi) = f(1,j,k+2,vi);
}


/// Rolling cache bundle
__device__
void rollBundleCache(Bundle Bndl, Shared fs, const Int lj, const Int lk)
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

/// Rolling cache read directly from array in global memory
/*
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
*/

/// Old non-bundle class version
/* 
/// This initialises the bundle to prepare for rolling x-direction cache
__device__
void bundleInit(Real *B, Mesh f, const Int j, const Int k, const Int vi)
{
	for (Int bi=-(f.ng_-1);bi<f.ng_+1;bi++)
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
*/
