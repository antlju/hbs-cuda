#pragma once

#include "typedefs.h"
#include "fd_params.h"

#define NG NGHOSTS

/// This gives the (i,j,k)-coordinate for component vi of a discretised 3D vector field
/// stored as a linear array in memory. It
__host__ __device__ inline size_t fIdx(const Int i, const Int j, const Int k, const Int vi=0)
{
	return vi*(NZ+2*NG)*(NY+2*NG)*(NX+2*NG)
		+(i+NG)+(NY+2*NG)*((j+NG)+(NZ+2*NG)*(k+NG));
}
