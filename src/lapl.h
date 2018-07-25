#pragma once
#include "typedefs.h"
#include "derivatives.h"

/// Computes P = lapl(B).
/// P is a scalar and B is a vector.
__device__
void lapl(Bundle B, Real *P, const Int i, const Real xfac, const Real yfac, const Real zfac)
{

	P[0] = del2x(B,xfac,i,0)+del2y(B,yfac,i,0)+del2z(B,zfac,i,0);

}

/// Computes vector laplacian P = lapl(B) = (lapl(B_x),lapl(B_y),lapl(B_z))
/// P is a vector "pencil" and B a vector Bundle.
__device__
void vlapl(Bundle B, Real *P, const Int i, const Real xfac, const Real yfac, const Real zfac)
{
	for (Int vi=0;vi<B.nvars_;vi++)
	{
		P[vi] = del2x(B,xfac,i,vi)+del2y(B,yfac,i,vi)+del2z(B,zfac,i,vi);
	}
	
}
