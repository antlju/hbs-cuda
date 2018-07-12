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
		Shared(T *fs, const Int NyTile, const Int NzTile, const Int Nvars, const Int Ng) :
	smPtr_(fs), ny_tile_(NyTile), nz_tile_(NzTile), nvars_(Nvars), ng_(Ng)
	{
		
	}
	
	__host__ __device__
		inline size_t indx(const Int i, const Int q, const Int vi=0)
	{
		return ;
	}
	
	__device__
		T& operator()(const Int i, const Int q, const Int vi=0)
	{
		return b_data[ indx(i,q,vi) ];
	}
	
	__device__
		const T& operator()(const Int i, const Int q, const Int vi=0) const
	{
		return b_data[ indx(i,q,vi) ];
	}
	
};
