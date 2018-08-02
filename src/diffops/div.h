#pragma once
#include "typedefs.h"
#include "derivatives.h"

/// Divergence of a vector field P = div(B).
/// P is a scalar "pencil", and B is a 3-vector bundle.
__device__
void divergence(Bundle B, Real *P, const Int li, const Real xfac, const Real yfac, const Real zfac)
{
	P[0] = delx(B,xfac,li,0)+dely(B,yfac,li,1)+delz(B,zfac,li,2);
}
