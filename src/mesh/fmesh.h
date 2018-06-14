/*!
 * Classes for the "big" mesh memory structure for use on host (CPU)
 * and device (GPU).
 */

#pragma once

/// Put these typedefs into another header later
typedef int Int;
typedef double Real;

template<class T, Int NG=0>
class fMeshDevice
{
public:
	T *mem_;
	
	/// Internal indexing function taking ghostpoints into account. I.e 4D->1D map
	__host__ __device__ size_t indx(Int i, Int j, Int k, Int vi=0) const
	{
		return vi*(nz_+2*NG)*(ny_+2*NG)*(nx_+2*NG)
                        +(i+NG)+(ny_+2*NG)*((j+NG)+(nz_+2*NG)*(k+NG));
	}

	fMeshDevice(size_t Nx, size_t Ny, size_t Nz, size_t Nvars) :
	{
		cudaMalloc(&mem_, sizeof(T)*Nvars*(Nx+2*NG)*(Ny+2*NG)*(Nz+2*NG));
	}
	
};
