#include "common.h"
/* 27 jul 2018, 16:25.
I've only sketched the outline of the solver. I need to implement all the kernels and ways to save
statistics and verification.
 */


//#include "kflow_allocation.h" /// Includes the data class instantion and memory allocation.

/// Global instantiation of data classes.
/// These are accessible for kernels.
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
	
	/// -------------------------------------
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
	/// Set up solver parameters. ::: This should probably be read from a file.
	///---------------------------------------
	SolverParams params;
	params.maxTimesteps = 100;
	params.currentTimestep = 0;
	params.Uchar = 1.0/2;
	params.viscosity = 1.0/10;
	params.kf = 1.0; /// Kolmogorov frequency.
	
	/// -------------------------------------
	/// Run solver for the set maximum no. of timesteps.
	///---------------------------------------
	for (Int timestep = 0;timestep<params.maxTimesteps;timestep++)
	{
		params.currentTimestep = timestep;
		RungeKuttaStepping();
	}

       	/// Free device memory.
	free_device_mem();
	return 0;
}

/// Runge-Kutta stepping. Computes RK3 substeps from k=1 to k=3
__host__
void RungeKuttaStepping()
{  
        /// From the previous step we have k=0 (time step n) data.
        /// We want to arrive at data for k=3 (time step n+1).
        /// (compare with Rosti & Brandt 2017)

	for (Int k_rk = 1;k_rk<=3;k_rk++)
	{
		/// First calculate RHSk = -D_j u_i u_j+(nu/rho)*Lapl(u))+force
                /// Then calc. u* = u+(2*dt*(alpha(k)/rho))*grad(p)
                ///                        +(dt*beta(k))*RHSk+(dt*gamma(k))*RHSk_1
                /// This is all done within the bundle/pencil framework.

		/// First set RHSk_1 to be RHSk from previous step.
		RHSk_copy_kernel<<<,>>>(RHSk, RHSk_1);

		/// Apply PBCS to u and calculate RHS^k
		apply_pbc(u);
		calculate_RHSk_kernel<<<,>>>();

		/// Calculate ustar
		calculate_uStar_kernel<<<,>>>();

		/// Solve the Poisson equation for Psi.
		apply_pbc(uStar);
		calc_divergence_uStar_kernel<<<,>>>(); ///Result is stored in the Psi array.
		Poisson_FFT_solver();

		/// Update pressure and enforce solenoidal condition
		update_pressure(pp);
		enforce_solenoidal();	
	}

	
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
