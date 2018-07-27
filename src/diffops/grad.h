#pragma once
#include "typedefs.h"
#include "derivatives.h"

/// Computes P = grad(B).
/// Where B is a scalar bundle and P is a 3-vector pencil.
__device__
void sgrad(Bundle B, Real *P, const Int i, const Real xfac, const Real yfac, const Real zfac)
{
        P[0] = delx(B,xfac,i,0);
	P[1] = dely(B,yfac,i,0);
	P[2] = delz(B,zfac,i,0);
	
}

/// Takes a vector bundle and computes P = (B dot grad)B, a vector Pencil.
__device__
void udotgradu(Bundle B, Real *P,const Int i, const Real xfac, const Real yfac, const Real zfac)
{
	for (Int vi=0;vi<B.nvars_;vi++)
	{
		P[vi] = B(i,0,0)*delx(B,xfac,i,vi)+
			B(i,0,1)*dely(B,yfac,i,vi)+
			B(i,0,2)*delz(B,zfac,i,vi);
	}
}
