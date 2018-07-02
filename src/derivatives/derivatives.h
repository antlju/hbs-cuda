#pragma once
/// Derivative forward declaration

/// 1st and 2nd order finite differences
__device__
Real fd4d1(const Real m2h,const Real m1h,const Real p1h,const Real p2h);
__device__
Real fd4d2(Real m2h, Real m1h, Real mid, Real p1h, Real p2h);

/// 1st and 2nd order partial derivatives
/// First:
__device__
Real delx(const Bundle *B, const Real dxfactor, const Int i, const Int vi);
__device__
Real dely(const Bundle *B, const Real dyfactor, const Int i, const Int vi);
/// Second:
__device__
Real del2x(const Bundle *B, const Real dxfactor, const Int i, const Int vi);
__device__
Real del2y(const Bundle *B, const Real dyfactor, const Int i, const Int vi);
__device__
Real del2z(const Bundle *B, const Real dzfactor, const Int i, const Int vi);

