#define __ROLLING_CACHE__ 1 /// Enables the rolling cache NZ/NY_TILE defs

#include "common.h"
#include <iostream>

template <class T, Int H>
class Mesh
{
public:
	size_t nx_,ny_,nz_,nvars_,ng_ = H;
	size_t totsize_;
	/// CUDA API methods, outside const/dest methods b/c reasons:
	/// https://stackoverflow.com/questions/24869167/trouble-launching-cuda-kernels-from-static-initialization-code
	__host__ Mesh(const Int Nx,const Int Ny,const Int Nz,const Int Nvars) :
		nx_(Nx),ny_(Ny),nz_(Nz),nvars_(Nvars),
		totsize_((Nx+2*H)*(Ny+2*H)*(Nz+2*H)*Nvars)
	{
		
	}

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
		
	}
	
private:
	T *d_data;
	T *h_data;

}; /// End class Mesh




Mesh<Real,2> globalMesh(NX,NY,NZ,3);

Int main()
{
	globalMesh.allocateHost(); globalMesh.allocateDevice();
	std::cout << globalMesh.nx_ << std::endl;



	return 0;
};

     
