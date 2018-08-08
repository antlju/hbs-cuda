#pragma once
#include "typedefs.h"

/// A parameter container class for use in various solvers.
class SolverParams {
public:
        /// Time step size (to be updated according to CFL before each Runge-Kutta ingegration step).
        Real *h_dt; 
        Real dt_old; /// Previous step size
        Real dt_init;
	Real *d_dt; /// Pointer to dt variable stored on device
        
        /// Max time steps.
        Int maxTimesteps;

	///
	Int saveintrvl;
	
        ///current time step
        Int currentTimestep;
        
        /// Reynolds number, viscosity
        /// Might require updates in time steps.
        Real Re;
        Real viscosity;

        /// Density
        Real rho;
	//Kinematic viscosity
	Real mu;
        /// Kolmogorov frequency
        Real kf;

	/// Characteristic velocity
	Real Uchar;

	__host__ void dt_copyFromDevice()
	{
		cudaCheck(
			cudaMemcpy(h_dt,d_dt,sizeof(Real),cudaMemcpyDeviceToHost)
			);
	}
	
	__host__ void dt_copyToDevice()
	{
		cudaCheck(
			cudaMemcpy(d_dt,h_dt,sizeof(Real),cudaMemcpyHostToDevice)
			);
	}
	
}; // End class SolverParams
