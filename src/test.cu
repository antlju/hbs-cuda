#include <iostream>

#define NGHOSTS 4
#define NN 8
#define NV 1

typedef int Int;
typedef double Real;

template<class T, Int NG=4>
class fMeshDevice
{
public:
	T *mem_;
	size_t nx_, ny_, nz_, nvars_;
	
	/// Internal indexing function taking ghostpoints into account. I.e 4D->1D map
	__host__ __device__ size_t indx(Int i, Int j, Int k, Int vi=0) const
	{
		return vi*(nz_+2*NG)*(ny_+2*NG)*(nx_+2*NG)
                        +(i+NG)+(ny_+2*NG)*((j+NG)+(nz_+2*NG)*(k+NG));
	}

	fMeshDevice(size_t Nx, size_t Ny, size_t Nz, size_t Nvars) :
		nx_(Nx), ny_(Ny), nz_(Nz), nvars_(Nvars)
	{
		cudaMalloc(&mem_, sizeof(T)*Nvars*(Nx+2*NG)*(Ny+2*NG)*(Nz+2*NG));
	}
	
};


Int main()
{
	size_t Nx=NN,Ny=NN,Nz=NN,Nv=NV;
	fMeshDevice<Real,NGHOSTS> deviceMesh(Nx,Ny,Nz,Nv);
	std::cout << "OK COMPILED IT MAYBE COOL" << std::endl;

}
