#pragma once
#include "typedefs.h"
#include <cassert>
#include <iostream>

template <class T, Int H>
class bundleMesh
{
public:
	T *b_data; //Pointer to bundle

	size_t bundleSize_; //Eg 4*NG+1
	size_t nvars_;

	__host__ __device__
		bundleMesh(T *Bdata, const Int BundleSize, const Int Nvars) :
	b_data(Bdata), bundleSize_(BundleSize), nvars_(Nvars)
	{
		
	}
	
	__host__ __device__
		inline size_t indx(const Int i, const Int q, const Int vi=0)
	{
		return q*nvars_*(1+2*H)+vi*(1+2*H)+(i+H);
	}
	
	__device__
		T& operator()(const Int i, const Int q, const Int vi=0)
	{
		return b_data[ indx(i,q,vi) ];
	}
	
	__device__
		const T& operator()(const Int i, const Int q, const Int vi=0) const
	{
		return b_data[ indx(i,q,vi) ];
	}
	
};
