#pragma once
#include "typedefs.h"
#include "fd_params.h"
#include "input_params.h"

__constant__ const Real d1_4_2C = -1.0/12;
__constant__ const Real d1_4_1C = 2.0/3;

__device__
Real fd4d1(const Real m2h,const Real m1h,const Real p1h,const Real p2h)
{
        return d1_4_2C*(p2h-m2h)+d1_4_1C*(p1h-m1h);
}

__device__ Real delz(const Real *B, const Real dzfactor, const Int i, const Int vi)
{
        return dzfactor * fd4d1(B[bIdx(i,6,vi)],B[bIdx(i,2,vi)], //refer to qjkmap.h for q->k vals.
                                B[bIdx(i,4,vi)],B[bIdx(i,8,vi)]);
}


__device__ Real dely_rc(const Real B[NY_TILE+2*NG][NZ_TILE+2*NG], const Real dyfactor, const Int bj, const Int bk, const Int vi)
{
        return dyfactor * fd4d1(B[bj-2][bk],B[bj-1][bk],
				B[bj+1][bk],B[bj+2][bk]);
}

__device__ Real delz_rc(const Real B[NY_TILE+2*NG][NZ_TILE+2*NG], const Real dzfactor, const Int bj, const Int bk, const Int vi)
{
        return dzfactor * fd4d1(B[bj][bk-2],B[bj][bk-1],
				B[bj][bk+1],B[bj][bk+2]);
}

