#pragma once
#include "typedefs.h"
#include "indexing.h"

__device__
void loadBundle(Real *B, const Real *f, const Int bi,
		const Int i, const Int j, const Int k,
		const Int vi)
{
	B[bIdx(bi,0,vi)] = f[fIdx(i,j,k,vi)];
	B[bIdx(bi,1,vi)] = f[fIdx(i,j+1,k,vi)];
	B[bIdx(bi,2,vi)] = f[fIdx(i,j,k-1,vi)];
	B[bIdx(bi,3,vi)] = f[fIdx(i,j-1,k,vi)];
	B[bIdx(bi,4,vi)] = f[fIdx(i,j,k+1,vi)];
	B[bIdx(bi,5,vi)] = f[fIdx(i,j+2,k,vi)];
	B[bIdx(bi,6,vi)] = f[fIdx(i,j,k-2,vi)];
	B[bIdx(bi,7,vi)] = f[fIdx(i,j-2,k,vi)];
	B[bIdx(bi,8,vi)] = f[fIdx(i,j,k+2,vi)];
}

__device__
void loadBundlexGhosts(Real *B, const Real *f, const Int j, const Int k, const Int vi)
{
	B[bIdx(-2,0,vi)] = f[fIdx(-2,j,k,vi)];
	B[bIdx(-2,1,vi)] = f[fIdx(-2,j+1,k,vi)];
	B[bIdx(-2,2,vi)] = f[fIdx(-2,j,k-1,vi)];
	B[bIdx(-2,3,vi)] = f[fIdx(-2,j-1,k,vi)];
	B[bIdx(-2,4,vi)] = f[fIdx(-2,j,k+1,vi)];
	B[bIdx(-2,5,vi)] = f[fIdx(-2,j+2,k,vi)];
	B[bIdx(-2,6,vi)] = f[fIdx(-2,j,k-2,vi)];
	B[bIdx(-2,7,vi)] = f[fIdx(-2,j-2,k,vi)];
	B[bIdx(-2,8,vi)] = f[fIdx(-2,j,k+2,vi)];

	B[bIdx(-1,0,vi)] = f[fIdx(-1,j,k,vi)];
	B[bIdx(-1,1,vi)] = f[fIdx(-1,j+1,k,vi)];
	B[bIdx(-1,2,vi)] = f[fIdx(-1,j,k-1,vi)];
	B[bIdx(-1,3,vi)] = f[fIdx(-1,j-1,k,vi)];
	B[bIdx(-1,4,vi)] = f[fIdx(-1,j,k+1,vi)];
	B[bIdx(-1,5,vi)] = f[fIdx(-1,j+2,k,vi)];
	B[bIdx(-1,6,vi)] = f[fIdx(-1,j,k-2,vi)];
	B[bIdx(-1,7,vi)] = f[fIdx(-1,j-2,k,vi)];
	B[bIdx(-1,8,vi)] = f[fIdx(-1,j,k+2,vi)];

	B[bIdx(NX_TILE,0,vi)] = f[fIdx(NN,j,k,vi)];
	B[bIdx(NX_TILE,1,vi)] = f[fIdx(NN,j+1,k,vi)];
	B[bIdx(NX_TILE,2,vi)] = f[fIdx(NN,j,k-1,vi)];
	B[bIdx(NX_TILE,3,vi)] = f[fIdx(NN,j-1,k,vi)];
	B[bIdx(NX_TILE,4,vi)] = f[fIdx(NN,j,k+1,vi)];
	B[bIdx(NX_TILE,5,vi)] = f[fIdx(NN,j+2,k,vi)];
	B[bIdx(NX_TILE,6,vi)] = f[fIdx(NN,j,k-2,vi)];
	B[bIdx(NX_TILE,7,vi)] = f[fIdx(NN,j-2,k,vi)];
	B[bIdx(NX_TILE,8,vi)] = f[fIdx(NN,j,k+2,vi)];

	B[bIdx(NX_TILE+1,0,vi)] = f[fIdx(NN+1,j,k,vi)];
	B[bIdx(NX_TILE+1,1,vi)] = f[fIdx(NN+1,j+1,k,vi)];
	B[bIdx(NX_TILE+1,2,vi)] = f[fIdx(NN+1,j,k-1,vi)];
	B[bIdx(NX_TILE+1,3,vi)] = f[fIdx(NN+1,j-1,k,vi)];
	B[bIdx(NX_TILE+1,4,vi)] = f[fIdx(NN+1,j,k+1,vi)];
	B[bIdx(NX_TILE+1,5,vi)] = f[fIdx(NN+1,j+2,k,vi)];
	B[bIdx(NX_TILE+1,6,vi)] = f[fIdx(NN+1,j,k-2,vi)];
	B[bIdx(NX_TILE+1,7,vi)] = f[fIdx(NN+1,j-2,k,vi)];
	B[bIdx(NX_TILE+1,8,vi)] = f[fIdx(NN+1,j,k+2,vi)];
}
