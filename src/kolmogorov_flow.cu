#include "common.h"

//#include "kflow_allocation.h" /// Includes the data class instantion and memory allocation.

/// Global instantiation of data classes.
Mesh uu(NX,NY,NZ,3); /// Velocity vector field.
Mesh uStar(NX,NY,NZ,3); /// u*, step velocity vector field.
Mesh RHSk(NX,NY,NZ,3); /// RHS^k Runge-Kutta substep vector field
Mesh RHSk_1(NX,NY,NZ,3); /// RHS^(k-1) Runge-Kutta substep vector field
Mesh pp(NX,NY,NZ,1); /// Pressure scalar field
Mesh psi(NX,NY,NZ,1); /// \Psi scalar field
Mesh gradPsi(NX,NY,NZ,3); /// \grad{\Psi} vector field.
Mesh verify(NX,NY,NZ,3); /// Vector field to store analytic solution for verification.
Grid grid(NX,NY,NZ,0,2*M_PI); ///
	
Complex *fftComplex;
Real *fftReal;

/// Forward declarations
__host__ void free_device_mem();



/// main()
Int main() 
{
	/// Create cuFFT plans.
	cufftHandle planD2Z,planZ2D;
	CUFFT_CHECK(cufftPlan3d(&planD2Z,NX,NY,NZ,CUFFT_D2Z));
	CUFFT_CHECK(cufftPlan3d(&planZ2D,NX,NY,NZ,CUFFT_Z2D));
	
	/// Device memory allocation.
	/// -------------------------------------
	uu.allocateDevice();
	uStar.allocateDevice();
	RHSk.allocateDevice();
	RHSk_1.allocateDevice();
	pp.allocateDevice();
	psi.allocateDevice();
	gradPsi.allocateDevice();
	verify.allocateDevice();

	cudaCheck(cudaMalloc((void**)&fftReal,sizeof(Real)*NX*NY*NZ));
	cudaCheck(cudaMalloc((void**)&fftComplex,sizeof(Complex)*NX*NY*(NZ/2+1)));
	/// -------------------------------------


	/// Free device memory.
	free_device_mem();
	return 0;
}

__host__
void free_device_mem()
{
	cudaCheck(cudaFree(uu.d_data));
	cudaCheck(cudaFree(uStar.d_data));
	cudaCheck(cudaFree(RHSk.d_data));
	cudaCheck(cudaFree(RHSk_1.d_data));
	cudaCheck(cudaFree(pp.d_data));
	cudaCheck(cudaFree(psi.d_data));
	cudaCheck(cudaFree(gradPsi.d_data));
	cudaCheck(cudaFree(verify.d_data));
	cudaCheck(cudaFree(fftComplex));
	cudaCheck(cudaFree(fftReal));
}
