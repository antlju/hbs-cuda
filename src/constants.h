#pragma once

/// Central FD coefficients for 1st derivative 4th accuracy order. Antisymmetric coefficients.
__constant__ const Real d1_4_2C = -1.0/12; //Coeff for +-2h, where h is the stepsize.
__constant__ const Real d1_4_1C = 2.0/3; //+-1h


/// Central FD coefficients for 2nd derivative 4th accuracty order.
/// symmetric coefficients.
__constant__ const Real d2_4_2C = -1.0/12;
__constant__ const Real d2_4_1C = 4.0/3;
__constant__ const Real d2_4_0C = -5.0/2;

/// RK3 constants
/// Coefficient values taken from Rosti & Brandt 2017
__constant__ const Real alpha1 = 4.0/15;
__constant__ const Real alpha2 = 1.0/15;
__constant__ const Real alpha3 = 1.0/6;

__constant__ const Real beta1 = 8.0/15;
__constant__ const Real beta2 = 5.0/12;
__constant__ const Real beta3 = 3.0/4;

__constant__ const Real gamma1 = 0.0;
__constant__ const Real gamma2 = -17.0/60;
__constant__ const Real gamma3 = -5.0/12;
