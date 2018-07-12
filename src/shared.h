#pragma once
#include "typedefs.h"
#include <cassert>
#include <iostream>

class Shared
{
public:
	Real *smPtr_; //Pointer to sharedMemory

	size_t ny_tile_;
	size_t nz_tile_;
	size_t nvars_;
	size_t ng_;

	__host__ __device__
		Shared(Real *fs, const Int NyTile, const Int NzTile, const Int Nvars, const Int Ng) :
	smPtr_(fs), ny_tile_(NyTile), nz_tile_(NzTile), nvars_(Nvars), ng_(Ng)
	{
		
	}
	
	__host__ __device__
	inline size_t indx(const Int lj, const Int lk, const Int vi=0)
	{
		return vi*(ny_tile_+2*ng_)*(nz_tile_+2*ng_)+
			(lk+NG)+(nz_tile_+2*ng_)*(lj+ng_);
	}
	
	__device__
		Real& operator()(const Int lj, const Int lk, const Int vi=0)
	{
		return smPtr_[ indx(lj,lk,vi) ];
	}
	
	//__device__
//		const Real& operator()(const Int lj, const Int lk, const Int vi=0) const
	//{
	//return smPtr_[ indx(lj,lk,vi) ];
	//}
	
};

__device__ void loadShared(Shared fs, Mesh f, const Int i, const Int j, const Int k, const Int lj, const Int lk)
{
	for (Int vi=0;vi<f.nvars_;vi++)
	{
		/// Load the shared memory tile
		/// fs for "f shared"
		fs(vi,lj,lk) = f(i+2,j,k,vi);
		
		/// If at yz-tile edges assign ghost points
				if (lj == 0)
				{
					fs(lj-1,lk,vi) = f(i+2,j-1,k,vi);
					fs(lj-2,lk,vi) = f(i+2,j-2,k,vi);
					fs(lj+fs.ny_tile_,lk,vi) = f(i+2,j+fs.ny_tile_,k,vi);
					fs(lj+fs.ny_tile_+1,lk,vi) = f(i+2,j+fs.ny_tile_+1,k,vi);
				}
				if (lk == 0)
				{
					fs(lj,lk-1,vi) = f(i+2,j,k-1,vi);
					fs(lj,lk-2,vi) = f(i+2,j,k-2,vi);
					fs(lj,lk+fs.nz_tile_,vi) = f(i+2,j,k+fs.nz_tile_,vi);
					fs(lj,lk+fs.nz_tile_+1,vi) = f(i+2,j,k+fs.nz_tile_+1,vi);
				}
	}
	//__syncthreads(); //Sync is done outside this function (inside the proper kernel)
}
