#pragma once
#include "constants.h"

/// Functions for getting the proper RK3 coefficients.
__device__
Real rk_alpha(const Int k)
{
	switch(k)
	{
	case 1: return alpha1;
	case 2: return alpha2;
	case 3: return alpha3;
	default: return 0.0;
	}
}

__device__
Real rk_beta(const Int k)
{
	switch(k)
	{
	case 1: return beta1;
	case 2: return beta2;
	case 3: return beta3;
	default: return 0.0;
	}
}

__device__
Real rk_gamma(const Int k)
{
	switch(k)
	{
	case 1: return gamma1;
	case 2: return gamma2;
	case 3: return gamma3;
	default: return 0.0;
	}
}
