#pragma once
/// Derivative forward declaration

/// 1st and 2nd order finite differences
extern __device__
Real fd4d1(const Real m2h,const Real m1h,const Real p1h,const Real p2h);
extern __device__
Real fd4d2(Real m2h, Real m1h, Real mid, Real p1h, Real p2h);

/// 1st and 2nd order partial derivatives
/// First:
extern __device__
Real delx(const Real *B, const Real dxfactor, const Int i, const Int vi);
extern __device__
Real dely(const Real *B, const Real dyfactor, const Int i, const Int vi);
extern __device__
Real delz(const Real *B, const Real dzfactor, const Int i, const Int vi);

/// Second:
extern __device__
Real del2x(const Real *B, const Real dxfactor, const Int i, const Int vi);
extern __device__
Real del2y(const Real *B, const Real dyfactor, const Int i, const Int vi);
extern __device__
Real del2z(const Real *B, const Real dzfactor, const Int i, const Int vi);
