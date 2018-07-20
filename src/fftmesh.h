#pragma once
#include "typedefs.h"
#include <cassert>
#include "errcheck.h"

/// A class for device fft memory.
/// Based on the equivalent in the hbs-code.
/// Specifically for in-place real-to-complex 3D transforms to solve Poisson's equation.

class fftMesh
{
public:
	cufftDoubleComplex *df_data; //Frequency space data, includes padding for real space.
	cufftDoubleReal *ds_data;
	
	size_t nx_,ny_,nz_,nzh_;
	
	__host__ fftMesh(const Int Nx, const Int Ny, const Int Nz) :
		nx_(Nx), ny_(Ny), nz_(Nz), nzh_(Nz/2+1)
	{
		df_data = (cufftDoubleComplex *) ds_data;
	}
	
	__host__
	void allocateDevice()	
	{
		assert(nx_*ny_*nz_ > 0);
		cudaCheck(
			cudaMalloc((void**)&ds_data, sizeof(cufftDoubleComplex)*nx_*ny_*nzh_)
			);
		       
	}
 
	/// Internal indexing for "input" (real data) array
	__device__
        size_t iindx(size_t i, size_t j, size_t k) const
        {
                assert(i<nx_ && j < ny_ && k < nz_);
                return k+(nz_+2)*(j+ny_*i);
        }
	
	__device__
        size_t oindx(size_t i, size_t j, size_t k) const
        {
                assert(i<nx_ && j < ny_ && k < nzh_);
                return k+(nzh_)*(j+ny_*i);
        }

	__device__
	Real getFreqReal(const size_t i, const size_t j, const size_t k)
	{
		return (Real)df_data[oindx(i,j,k)].x;
	}

	__device__
	Real getFreqImag(const size_t i, const size_t j, const size_t k)
	{
		return (Real)df_data[oindx(i,j,k)].y;
	}
	
	__device__
	void setFreqReal(const size_t i, const size_t j, const size_t k, const Real val)
	{
		df_data[oindx(i,j,k)].x = val;
	}

		__device__
	void setFreqImag(const size_t i, const size_t j, const size_t k, const Real val)
	{
		df_data[oindx(i,j,k)].y = val;
	}
	
};
