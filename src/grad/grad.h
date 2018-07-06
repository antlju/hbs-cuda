#pragma once

/* Gradient implementation kernel 3-bundle -> kernel 3-pencil */

/// Computes the vector pencil gradient grad(f) of a scalar function bundle f(x,y,z).
void sgrad(const Real *B, Real *P, const Int i, const Real xfac, const Real yfac, const Real zfac)
{
	P[pIdx(i,0)] = delx(B,xfac,i,0);
	P[pIdx(i,1)] = dely(B,yfac,i,0);
	P[pIdx(i,2)] = delz(B,zfac,i,0);
}


