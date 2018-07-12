#pragma once
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
