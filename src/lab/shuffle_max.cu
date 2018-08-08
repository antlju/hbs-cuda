#include <iostream>
#include "common.h"

typedef double Real;

/// Instantiate global objects
Timer timer;
Mesh u(NY,NY,NZ,1);
Mesh du(NX,NY,NZ,1);
Grid grid(NX,NY,NZ,0.0,2*M_PI);

__inline__ __device__
Real warpReduceMax(Real val)
{
	for (Int offset = warpSize/2; offset > 0; offset /=2)
	{
		val = max(val,__shfl_down(val,offset));
	}
	return val;
}

__inline__ __device__
Real blockReduceMax(Real val)
{
	static __shared__ Real shared[32]; // Shared memory for 32 partial sums (warpSize is 32)
	Int lane = threadIdx.x % warpSize;
	Int wid = threadIdx.x / warpSize;

	val = warpReduceMax(val); /// Each warp performs partial reduction

	if (lane == 0)
		shared[wid] = val; // Write reduced value to shared memory

	__syncthreads(); /// Wait for all partial reductions.

	/// Read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid == 0)
		val = warpReduceMax(val); /// Final reduce within first warp

	return val;
}

__global__ void deviceReduceMax_kernel(Real *in, Real *out, Int N)
{
	Real maxval = 0.0;

	/// Reduce multiple elements per thread via grid striding
	for (Int i= blockIdx.x *blockDim.x + threadIdx.x;
	     i<N;
	     i+=blockDim.x * gridDim.x)
	{
		maxval = max(in[i],maxval);
	}
	maxval = blockReduceMax(maxval);
	if (threadIdx.x == 0)
		out[blockIdx.x] = maxval;
}


__host__ void initHost(Mesh &f, const Grid &grid)
{
	Real *x = grid.h_linspace;
	for (Int i=0;i<f.nx_;i++)
	{
		for (Int j=0;j<f.ny_;j++)
		{
			for (Int k=0;k<f.nz_;k++)
			{
				f.h_data[f.indx(i,j,k,0)] = sin(x[k])/2;
			}
		}
	}

	//f.h_data[f.indx(0,5,6,0)] = 64.0;
	//f.h_data[f.indx(0,0,0,0)] = 3.0;
	//f.h_data[f.indx(0,8,133,0)] = 1024;
}

Int main()
{
	timer.createEvents();
	std::cout << "Executing w/ size: (N=" << NN << ")^3" << std::endl;
	u.allocateHost(); u.allocateDevice();
	du.allocateHost(); du.allocateDevice();
	
	grid.setHostLinspace();
	initHost(u,grid);
	u.copyToDevice();

	const Int threads = 1024;
	const Int NN3 = (NX+2*NGHOSTS)*(NY+2*NGHOSTS)*(NZ+2*NGHOSTS);
	const Int blocks = min((NN3+(threads-1))/threads,1024);
	

	timer.recordStart();
	deviceReduceMax_kernel<<<blocks,threads>>>(u.d_data,du.d_data,NN3);
	deviceReduceMax_kernel<<<1,1024>>>(du.d_data,du.d_data,NN3);
	timer.recordStop();
	timer.sync();
	timer.print();
	du.copyFromDevice();
	std::cout << du.h_data[0] << std::endl;
	//du.print();
	
	return 0;
}

     
/*

 */
