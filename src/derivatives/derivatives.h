#pragma once
#include "typedefs.h" //int,double/float->Int,Real
#include "indexing.h" //for bIdx()
#include "../constants.h" //fnite difference constants


/// 1st and 2nd order finite differences
__device__
Real fd4d1(const Real m2h,const Real m1h,const Real p1h,const Real p2h)
{
        return d1_4_2C*(p2h-m2h)+d1_4_1C*(p1h-m1h);
}

/// Second derivative stencil for 4th order finite difference.
__device__
Real fd4d2(Real m2h, Real m1h, Real mid, Real p1h, Real p2h)
{

        return d2_4_2C*(m2h+p2h)+d2_4_1C*(m1h+p1h)+d2_4_0C*mid;
}

/// 1st and 2nd order partial derivatives
/// First:
__device__
Real delx(Bundle B, const Real dxfactor, const Int i, const Int vi)
{
        return dxfactor * fd4d1(B(i-2,0,vi),B(i-1,0,vi),
                                B(i+1,0,vi),B(i+2,0,vi));
}

__device__
Real dely(Bundle B, const Real dyfactor, const Int i, const Int vi)
{
        return dyfactor * fd4d1(B(i,7,vi),B(i,3,vi), //refer to qjkmap.h for q->j vals.
                                B(i,1,vi),B(i,5,vi));
}

__device__
Real delz(Bundle B, const Real dzfactor, const Int i, const Int vi)
{
        return dzfactor * fd4d1(B(i,6,vi),B(i,2,vi), //refer to qjkmap.h for q->k vals.
                                B(i,4,vi),B(i,8,vi));
}


/* Old no bundle class
/// 1st and 2nd order partial derivatives
/// First:
__device__
Real delx(const Real *B, const Real dxfactor, const Int i, const Int vi)
{
        return dxfactor * fd4d1(B[bIdx(i-2,0,vi)],B[bIdx(i-1,0,vi)],
                                B[bIdx(i+1,0,vi)],B[bIdx(i+2,0,vi)]);
}

__device__
Real dely(const Real *B, const Real dyfactor, const Int i, const Int vi)
{
        return dyfactor * fd4d1(B[bIdx(i,7,vi)],B[bIdx(i,3,vi)], //refer to qjkmap.h for q->j vals.
                                B[bIdx(i,1,vi)],B[bIdx(i,5,vi)]);
}

__device__
Real delz(const Real *B, const Real dzfactor, const Int i, const Int vi)
{
        return dzfactor * fd4d1(B[bIdx(i,6,vi)],B[bIdx(i,2,vi)], //refer to qjkmap.h for q->k vals.
                                B[bIdx(i,4,vi)],B[bIdx(i,8,vi)]);
}

/// Second:
__device__
Real del2x(const Real *B, const Real dxfactor, const Int i, const Int vi)
{
        return dxfactor * fd4d2(B[bIdx(i-2,0,vi)],B[bIdx(i-1,0,vi)],
				B[bIdx(i,0,vi)],
                                B[bIdx(i+1,0,vi)],B[bIdx(i+2,0,vi)]);
}

__device__
Real del2y(const Real *B, const Real dyfactor, const Int i, const Int vi)
{
        return dyfactor * fd4d2(B[bIdx(i,7,vi)],B[bIdx(i,3,vi)],
				B[bIdx(i,0,vi)],
                                B[bIdx(i,1,vi)],B[bIdx(i,5,vi)]);
}

__device__
Real del2z(const Real *B, const Real dzfactor, const Int i, const Int vi)
{
        return dzfactor * fd4d2(B[bIdx(i,6,vi)],B[bIdx(i,2,vi)],
				B[bIdx(i,0,vi)],
                                B[bIdx(i,4,vi)],B[bIdx(i,8,vi)]);
}
*/
