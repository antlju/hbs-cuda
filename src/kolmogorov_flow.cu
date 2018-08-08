/* 7 aug 2018, 13:09. 
I have completed the solver and it seems to give a reasonable value for max(u) up to NN=32, but for larger sizes it diverges! Need to fix
*/
/* 2 aug 2018, 15:45.
Implemented divergence of ustar, this is hard to test w/o full solver implementation since divergence is zero out of the box. But it seems to give back numbers in a proper way (not sure if "correct" numbers).
 */

/* 2 aug 2018, 15:00.
Implementation of RHSk copy, RHSk calc, uStar calc kernels. Don't know if they give the right numbers of course 
but they don't give completely unreasonable numbers and everything compiles fine. cuda-memcheck gives no errors.
 */

#include "common.h"

/// Include kernels
#include "rhs_kernels.h"
#include "ustar_kernels.h"
#include "pressure_gradpsi_solenoidal.h"
#include "poisson_fft.h"
#include "reductions.h"

/// Global instantiation of data classes.
/// These can be passed to device kernels. They contain pointers to device memory.
Mesh uu(NX,NY,NZ,3); /// Velocity vector field.
Mesh uStar(NX,NY,NZ,3); /// u*, step velocity vector field.
Mesh RHSk(NX,NY,NZ,3); /// RHS^k Runge-Kutta substep vector field
Mesh RHSk_1(NX,NY,NZ,3); /// RHS^(k-1) Runge-Kutta substep vector field
Mesh Pp(NX,NY,NZ,1); /// Pressure scalar field
Mesh Psi(NX,NY,NZ,1); /// \Psi scalar field
Mesh gradPsi(NX,NY,NZ,3); /// \grad{\Psi} vector field.
Mesh verify(NX,NY,NZ,3); /// Vector field to store analytic solution for verification.
Mesh uu_stats(NX,NY,NZ,3); //// Vector field to store statistics by reduction computations!
Grid grid(NX,NY,NZ,0,2*M_PI); ///
	
Complex *fftComplex;
Real *fftReal;

Timer timer;

SolverParams params;
Real *d_umax;
Real *h_umax;

/// GPU kernel call layout.
dim3 ThreadsPerBlock(NY_TILE,NZ_TILE); 
dim3 NoOfBlocks(NN/NY_TILE,NN/NZ_TILE);

/// Forward declarations
__host__ void copyMeshOnDevice(Mesh in, Mesh out);
__host__ void launch_output();
__host__ void free_device_mem();
__host__ void apply_pbc(Mesh f);
__host__ void update_timestep(SolverParams params, const Real dx, const Real umax);
__host__ void RungeKuttaStepping(Mesh u, Mesh ustar, Mesh rhsk, Mesh rhsk_1,
				 Mesh p, Mesh psi, Mesh gradpsi, Mesh stats,
				 Complex *fftcomplex, Real *fftreal,
				 Grid grid, SolverParams params,
				 cufftHandle pland2z, cufftHandle planz2d);

/// main()
Int main() 
{
	timer.createEvents();

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
	Pp.allocateDevice();
	Psi.allocateDevice();
	gradPsi.allocateDevice();
	verify.allocateDevice();
	verify.allocateHost();
	uu_stats.allocateDevice();
	
	grid.setHostLinspace(); /// Allocates and sets host linspace (in this case equivalent to NumPy's np.linspace(0,2*Pi,NX))
	grid.copyLinspaceToDevice(); /// Allocates device memory and copies from host
	
	cudaCheck(cudaMalloc((void**)&fftReal,sizeof(Real)*NX*NY*NZ));
	cudaCheck(cudaMalloc((void**)&fftComplex,sizeof(Complex)*NX*NY*(NZ/2+1)));

	/// Allocate single value variables on host and device
	cudaCheck(cudaMallocHost(&params.h_dt,sizeof(Real)));
	cudaCheck(cudaMalloc((void**)&params.d_dt,sizeof(Real)));
	cudaCheck(cudaMalloc((void**)&d_umax,sizeof(Real)));
	cudaCheck(cudaMallocHost(&h_umax,sizeof(Real)));

	
	/// -------------------------------------
	/// Set up solver parameters. ::: This should probably be read from a file.
	///---------------------------------------
	params.maxTimesteps = 10;
	std::cout << "Max timesteps: " << params.maxTimesteps << std::endl;
	params.currentTimestep = 0;
	params.Uchar = 1.0/2;
	params.viscosity = 1.0/10;
	params.kf = 1.0; /// Kolmogorov frequency.

	/// Set up initial timestep size based on forcing
	Real forceabs,fmax = 0.0;
	for (size_t i=0;i<NX;i++)
	{
		forceabs = fabs(sin(params.kf*grid.h_linspace[i]));
		if (forceabs > fmax)
			fmax = forceabs;
	}
	update_timestep(params,grid.dx_,fmax);
	
	
	/// -------------------------------------
	/// Run solver for the set maximum no. of timesteps.
	///---------------------------------------
	timer.recordStart();
	for (Int timestep = 0;timestep<params.maxTimesteps;timestep++)
	{
		params.currentTimestep = timestep;
		RungeKuttaStepping(uu,uStar,RHSk,RHSk_1,
				   Pp,Psi,gradPsi,uu_stats,
				   fftComplex,fftReal,
				   grid,params,
				   planD2Z,planZ2D);
	}
	timer.recordStop();
	timer.sync();
	timer.print();
	
	uu.allocateHost();
	std::cout << "Finished timestepping after " << params.maxTimesteps << " steps." << std::endl;
	uu.copyFromDevice();
	std::cout << "Maximum value of velocity field: " << uu.max() << "\n";
	


	
       	/// Free device memory.
	free_device_mem();
	CUFFT_CHECK(cufftDestroy(planD2Z));
	CUFFT_CHECK(cufftDestroy(planZ2D));
	return 0;
}

/// Runge-Kutta stepping. Computes RK3 substeps from k=1 to k=3
__host__
void RungeKuttaStepping(Mesh u, Mesh ustar, Mesh rhsk, Mesh rhsk_1,
			Mesh p, Mesh psi, Mesh gradpsi, Mesh stats,
			Complex *fftcomplex, Real *fftreal,
			Grid grid, SolverParams params,
			cufftHandle pland2z, cufftHandle planz2d)
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
		copyMeshOnDevice(rhsk, rhsk_1);
		
		/// Apply PBCS to u and calculate RHS^k
		apply_pbc(u);
		calculate_RHSk_kernel<<<NoOfBlocks,ThreadsPerBlock>>>(u,rhsk,grid,params);

		// If k_rk == 1 update the timestep dt
                if (k_rk == 1)
		{
			calc_max(u,stats);
			cudaCheck(cudaMemcpy(h_umax,&stats.d_data[0],sizeof(Real),cudaMemcpyDeviceToHost));
                        update_timestep(params,grid.dx_,h_umax[0]);
		}

		/// Calculate ustar
		calculate_uStar_kernel<<<NoOfBlocks,ThreadsPerBlock>>>(u,rhsk,rhsk_1,p,ustar,
					      params,grid.dx_,k_rk);

		/// Solve the Poisson equation for Psi.
		apply_pbc(ustar);
		///Result from kernel below is stored in the Psi array for input to Poisson solver.
		calc_divergence_uStar_kernel<<<NoOfBlocks,ThreadsPerBlock>>>(ustar,psi,
									     params,grid.dx_,k_rk);

		Poisson_FFT_solver(psi,fftcomplex,fftreal,grid.xlen,k_rk,pland2z,planz2d);
		
                /// Update pressure, calculate gradient of psi to enforce solenoidal condition
		update_pressure_kernel<<<NoOfBlocks,ThreadsPerBlock>>>(p,psi);
		apply_pbc(psi);
		calc_gradpsi_kernel<<<NoOfBlocks,ThreadsPerBlock>>>(psi,gradpsi,grid.dx_);
		enforce_solenoidal_kernel<<<NoOfBlocks,ThreadsPerBlock>>>(u,ustar,gradpsi,
									  params,k_rk);	
	}
	//uu.copyFromDevice();
	//std::cout << params.currentTimestep << ": " << uu.max() << std::endl;
	
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
	cudaCheck(cudaFree(Pp.d_data));
	cudaCheck(cudaFree(Psi.d_data));
	cudaCheck(cudaFree(gradPsi.d_data));
	cudaCheck(cudaFree(verify.d_data));
	cudaCheck(cudaFree(fftComplex));
	cudaCheck(cudaFree(fftReal));
	cudaCheck(cudaFree(params.d_dt));
	cudaCheck(cudaFree(d_umax));
	cudaCheck(cudaFree(uu_stats.d_data));
}

__host__
void launch_output()
{
	std::cout << "Launching kolmogorov flow with cubic space size N^3 = NX*NY*NZ = " << NN << "^3." << std::endl;
	std::cout << "Tile sizes are \n X_TILE: " << NX_TILE << ",\n Y_TILE: " << NY_TILE << ",\n Z_TILE: " << NZ_TILE << "." << std::endl;
}

__host__
void
copyMeshOnDevice(Mesh in, Mesh out)
{
	cudaCheck(cudaMemcpy(out.d_data,in.d_data,sizeof(Real)*in.totsize_,cudaMemcpyDeviceToDevice));
}

__host__
void update_timestep(SolverParams params, const Real dx, const Real umax)
{
	//std::cout << umax << std::endl;
	
	if (params.currentTimestep !=1 )
	{
		Real c1=0.1;
		Real c2=c1; //Courant numbers for advection and diffusion respectively

		Real nu = params.viscosity,L=dx;
        
		//Factor of 1/3 since dx=dy=dz
	
		Real adv = (1.0/3)*c1*dx/umax;
		Real diff = (1.0/3)*c2*pow(L,2)/nu;
		//std::cout << "adv : " << adv << "\t diff: " << diff << std::endl;
		//Set new time step size according to CFL condition
		if (adv < diff)
			params.h_dt[0] = adv;
		else
			params.h_dt[0] = diff;

		//std::cout << adv << " " << diff << std::endl; /// Print for debug
	}
	
	
	//params.dt_copyToDevice();
}
