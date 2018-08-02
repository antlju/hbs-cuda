#include "common.h"

/// Include solver specific kernels
#include "rhs_kernels.h"
#include "ustar_kernels.h"

/* 2 aug 2018, 15:00.
Implementation of RHSk copy, RHSk calc, uStar calc kernels. Don't know if they give the right numbers of course 
but they don't give completely unreasonable numbers and everything compiles fine. cuda-memcheck gives no errors.
 */

/// Global instantiation of data classes.
/// These can be passed to device kernels. They contain pointers to device memory.
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

dim3 ThreadsPerBlock(NY_TILE,NZ_TILE); 
dim3 NoOfBlocks(NN/NY_TILE,NN/NZ_TILE);

/// Forward declarations
__host__ void launch_output();
__host__ void free_device_mem();
__host__ void apply_pbc(Mesh f);
__host__ void RungeKuttaStepping(Mesh u, Mesh uStar, Mesh RHSk, Mesh RHSk_1,
			Mesh pp, Mesh psi, Mesh gradPsi, Complex *fftComplex, Real *fftReal,
			Grid grid, SolverParams params);




/// main()
Int main() 
{
	launch_output();
	
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
	verify.allocateHost();
	
	grid.setHostLinspace(); /// Allocates and sets host linspace (in this case equivalent to NumPy's np.linspace(0,2*Pi,NX))
	grid.copyLinspaceToDevice(); /// Allocates device memory and copies from host
	
	cudaCheck(cudaMalloc((void**)&fftReal,sizeof(Real)*NX*NY*NZ));
	cudaCheck(cudaMalloc((void**)&fftComplex,sizeof(Complex)*NX*NY*(NZ/2+1)));

	
	
	/// -------------------------------------
	/// Set up solver parameters. ::: This should probably be read from a file.
	///---------------------------------------
	SolverParams params;
	params.maxTimesteps = 1;
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
		RungeKuttaStepping(uu,uStar,RHSk,RHSk_1,
				   pp,psi,gradPsi,fftComplex,fftReal,
				   grid,params);
	}

	uStar.allocateHost();
	uStar.copyFromDevice();
	uStar.print();

       	/// Free device memory.
	free_device_mem();
	CUFFT_CHECK(cufftDestroy(planD2Z));
	CUFFT_CHECK(cufftDestroy(planZ2D));
	return 0;
}

/// Runge-Kutta stepping. Computes RK3 substeps from k=1 to k=3
__host__
void RungeKuttaStepping(Mesh u, Mesh uStar, Mesh RHSk, Mesh RHSk_1,
			Mesh p, Mesh psi, Mesh gradPsi, Complex *fftComplex, Real *fftReal,
			Grid grid, SolverParams params)
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
		RHSk_copy_kernel<<<NoOfBlocks,ThreadsPerBlock>>>(RHSk, RHSk_1);

		/// Apply PBCS to u and calculate RHS^k
		apply_pbc(u);
		calculate_RHSk_kernel<<<NoOfBlocks,ThreadsPerBlock>>>(u,RHSk,grid,params);

		/// Calculate ustar
		calculate_uStar_kernel<<<NoOfBlocks,ThreadsPerBlock>>>(u,RHSk,RHSk_1,p,uStar,
					      params,grid.dx_,k_rk);

		/// Solve the Poisson equation for Psi.
		apply_pbc(uStar);
		//calc_divergence_uStar_kernel<<<,>>>(); ///Result is stored in the Psi array.
		//Poisson_FFT_solver();

		/// Update pressure and enforce solenoidal condition
		//update_pressure(pp);
		//enforce_solenoidal();	
	}

	
}

__host__
void apply_pbc(Mesh f)
{
	pbc_x_kernel<<<NoOfBlocks,ThreadsPerBlock>>>(f);
	pbc_y_kernel<<<NoOfBlocks,ThreadsPerBlock>>>(f);
	pbc_z_kernel<<<NoOfBlocks,ThreadsPerBlock>>>(f);
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

__host__
void launch_output()
{
	std::cout << "Launching kolmogorov flow with cubic space size N^3 = NX*NY*NZ = " << NN << "^3." << std::endl;
	std::cout << "Tile sizes are \n X_TILE: " << NX_TILE << ",\n Y_TILE: " << NY_TILE << ",\n Z_TILE: " << NZ_TILE << "." << std::endl;
}
