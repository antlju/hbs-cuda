#pragma once

#include "typedefs.h"
#include "fd_params.h"


/// This gives the (i,j,k)-coordinate for component vi of a discretised 3D vector field
/// stored as a linear array in memory. It
__host__ __device__ inline size_t fIdx(const Int i, const Int j, const Int k, const Int vi=0)
{
	return vi*(NZ+2*NGHOSTS)*(NY+2*NGHOSTS)*(NX+2*NGHOSTS)
		+(i+NGHOSTS)+(NY+2*NGHOSTS)*((j+NGHOSTS)+(NZ+2*NGHOSTS)*(k+NGHOSTS));
}
