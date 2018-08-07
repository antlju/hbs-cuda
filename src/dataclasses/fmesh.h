#pragma once
#include "typedefs.h"
#include <cassert>
#include <iostream>

template <class T, Int H>
class fMesh
{
public:
	T *d_data;
	T *h_data;
	size_t nx_,ny_,nz_,nvars_,ng_ = H;
	size_t totsize_;

	/// Constructor
	__host__ fMesh(const Int Nx,const Int Ny,const Int Nz,const Int Nvars) :
		nx_(Nx),ny_(Ny),nz_(Nz),nvars_(Nvars),
		totsize_((Nx+2*H)*(Ny+2*H)*(Nz+2*H)*Nvars)
	{
		
	}

	/// Memory methods
	__host__ void allocateHost()
	{
		assert(totsize_ > 0);
		cudaCheck(cudaMallocHost(&h_data,sizeof(T)*totsize_));	
	}

	__host__ void allocateDevice()
	{
		assert(totsize_ > 0);
		cudaCheck(cudaMalloc((void**)&d_data,sizeof(T)*totsize_));
	}

	__host__ void copyToDevice()
	{
		cudaCheck(
			cudaMemcpy(d_data,h_data,sizeof(T)*totsize_,cudaMemcpyHostToDevice)
			);
	}

	__host__ void copyFromDevice()
	{
		cudaCheck(
			cudaMemcpy(h_data,d_data,sizeof(T)*totsize_,cudaMemcpyDeviceToHost)
			);
	}

	/// Operator overloading
	/// This gives the (i,j,k)-coordinate for component vi of a discretised 3D vector field
/// stored as a linear array in memory.
/// ROW MAJOR ORDER! (k fastest index, i slowest). This is C-style.
	__host__ __device__ inline size_t indx(const Int i, const Int j, const Int k, const Int vi=0)
	{
		return vi*(nx_+2*ng_)*(ny_+2*ng_)*(nz_+2*ng_)
			+(k+ng_)+(nx_+2*ng_)*((j+ng_)+(ny_+2*ng_)*(i+ng_));
	}
	
        __device__ T& operator()(Int i, Int j, Int k, Int vi=0)
        {
                return d_data[ indx(i,j,k,vi) ];
        }

        __device__ const T& operator()(Int i, Int j, Int k, Int vi=0) const
        {
                return d_data[ indx(i,j,k,vi) ];
        }

	/// utilities
	__host__ void print()
	{

                for (size_t vi=0;vi<nvars_;vi++)
                {
                        std::cout << "========================" << std::endl;
                        std::cout << "fMesh print. Component: " << vi << std::endl;
                        std::cout << "========================" << std::endl;
                        for (size_t i=0;i<nx_;i++)
                        {
                                for (size_t j=0;j<ny_;j++)
                                {
                                        for (size_t k=0;k<nz_;k++)
                                        {
                                                std::cout << h_data[ indx(i,j,k,vi)] << "\t";
                                        }
                                        std::cout << std::endl;
                                }
                                std::cout << "----------------------------" << std::endl;
                        }
                        std::cout << std::endl;
                }
        }

	__host__ Real max()
	{
		Real max_new,max_old;
		
                for (size_t vi=0;vi<nvars_;vi++)
                {
                        for (size_t i=0;i<nx_;i++)
                        {
                                for (size_t j=0;j<ny_;j++)
                                {
                                        for (size_t k=0;k<nz_;k++)
                                        {
                                                max_new = fabs(h_data[ indx(i,j,k,vi)]);
						if (max_new > max_old)
							max_old = max_new;
                                        }
                                }

                        }
                }
		return max_old;
        }


}; /// End class Mesh
