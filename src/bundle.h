#pragma once
__device__
void bundleInit(Bundle B, Mesh f, const Int j, const Int k, const Int vi)
{
	for (Int bi=-(f.ng_-1);bi<f.ng_+1;bi++)
	{
		B(bi,0,vi) = f(bi-1,j,k,vi);
		B(bi,1,vi) = f(bi-1,j+1,k,vi);
		B(bi,2,vi) = f(bi-1,j,k-1,vi);
		B(bi,3,vi) = f(bi-1,j-1,k,vi);
		B(bi,4,vi) = f(bi-1,j,k+1,vi);
		B(bi,5,vi) = f(bi-1,j+2,k,vi);
		B(bi,6,vi) = f(bi-1,j,k-2,vi);
		B(bi,7,vi) = f(bi-1,j-2,k,vi);
		B(bi,8,vi) = f(bi-1,j,k+2,vi);
	}
}

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
