#pragma once
#include "typedefs.h"

/// A parameter container class for use in various solvers.
class SolverParams {
public:
        /// Time step size (to be updated according to CFL before each Runge-Kutta ingegration step).
        Real dt; 
        Real dt_old; /// Previous step size
        Real dt_init;
        
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

}; // End class SolverParams
