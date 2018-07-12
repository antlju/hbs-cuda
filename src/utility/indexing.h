#pragma once
#include "typedefs.h"
#include "fd_params.h"
#include "input_params.h"

/// This gives the (i,j,k)-coordinate for component vi of a discretised 3D vector field
/// stored as a linear array in memory.
/// ROW MAJOR ORDER! (k fastest index, i slowest). This is C-style.
__host__ __device__ inline size_t fIdx(const Int i, const Int j, const Int k, const Int vi=0)
{
	return vi*(NZ+2*NGHOSTS)*(NY+2*NGHOSTS)*(NX+2*NGHOSTS)
		+(k+NGHOSTS)+(NX+2*NGHOSTS)*((j+NGHOSTS)+(NY+2*NGHOSTS)*(i+NGHOSTS));
}



__host__ __device__ inline size_t bIdx(const Int i, const Int q, const Int vi=0)
{
	
	//assert(Nvars == 1 || Nvars == 3);
	return q*(1+2*NG)+vi*(1+2*NG)+(i+NG);
}

__host__ __device__ inline size_t pIdx(const Int i, const Int vi=0)
{
	//return q*NVARS*(1+2*NG)+vi*(1+2*NG)+(i+NG);
	return vi*(1+2*NG)+(i+NG);
}


/*
__host__ __device__ inline size_t bIdx(const Int i, const Int q, const Int vi=0)
{
	return q*NVARS*(NX+2*NG)+vi*(NX+2*NG)+(i+NG);
}

*/
