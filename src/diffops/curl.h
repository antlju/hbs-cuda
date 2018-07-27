#pragma once
#include "typedefs.h"
#include "derivatives.h"

/// Computes omega = curl(u).
/// Where u is a 3-vector bundle and omega is a 3-vector pencil.
__device__
void curl(Bundle u, Real *omega, const Int i, const Real xfac, const Real yfac, const Real zfac)
{
        Real dx,dy,dz;  
	/// Compute omega_1 = d_2u_3-d_3u_2
	dy = dely(u,yfac,i,2);
	dz = delz(u,zfac,i,1); 
	omega[0] = dy-dz;

	/// Compute omega_2 = d_3u_1-d_1u_3
	dz = delz(u,zfac,i,0);
	dx = delx(u,xfac,i,2);
	omega[1] = dz-dx;
        
	/// Compute omega_3 = d_1u_2-d_2u_1
	dx = delx(u,xfac,i,1);
	dy = dely(u,yfac,i,0);
	omega[2] = dx-dy;
	
}


