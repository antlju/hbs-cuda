#include <iostream>
#include "common.h"

typedef double Real;

/// Instantiate global objects
Timer timer;
Mesh u(NY,NY,NZ,1);
Mesh du(NX,NY,NZ,1);
Grid grid(NX,NY,NZ,0.0,2*M_PI);

__global__
void reduce0(Mesh f, Mesh out)
{
	__shared__ Real smem[NY_TILE][NZ_TILE];
	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;

	const Int lj = threadIdx.x;
	const Int lk = threadIdx.y;

	if (j < f.ny_ && k < f.nz_)
	{
		for (Int i=0;i<f.nx_;i++)
		{
			/// Load shared mem
			smem[lj][lk] = f(i,j,k);

			for (unsigned int s=blockDim.y/2;s>0;s>>=1)
			{
				if (lk < s)
				{
					smem[lj][lk] += smem[lj][lk +s];
				}
				__syncthreads();
			}

			
			for (unsigned int s=blockDim.x/2;s>0;s>>=1)
			{
				if (lj < s)
				{
					smem[lj][lk] += smem[lj + s][lk];
				}
				__syncthreads();
			}
			

			if (lk == 0)
				out(0,blockIdx.x*blockDim.x,blockDim.y*blockIdx.y) += smem[0][0];
		}
	}
	
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
				f.h_data[f.indx(i,j,k,0)] = 1.0;
			}
		}
	}

	//f.h_data[f.indx(0,5,6,0)] = 3.0;
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

	dim3 tpb(NY_TILE,NZ_TILE); 
	dim3 blx(NN/NY_TILE,NN/NZ_TILE);

	timer.recordStart();
	reduce0<<<blx,tpb>>>(u,du);
	timer.recordStop();
	timer.sync();
	timer.print();
	du.copyFromDevice();
	std::cout << du.h_data[du.indx(0,0,0)] << " " << NY*NX*NZ << std::endl;
	//du.print();
	
	return 0;
}

     
/*

void reduce0(Mesh f, Mesh out)
{
	__shared__ Real smem[NY_TILE][NZ_TILE];
	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;

	const Int lj = threadIdx.x;
	const Int lk = threadIdx.y;

	if (j < f.ny_ && k < f.nz_)
	{
		for (Int i=0;i<1;i++)
		{
			/// Load shared mem
			smem[lj][lk] = f(i,j,k);

			for (unsigned int s=blockDim.y/2;s>0;s>>=1)
			{
				if (lk < s)
				{
					smem[lj][lk] += smem[lj][lk +s];
				}
				__syncthreads();
			}
			

			if (lk == 0)
				out(i,j,blockDim.y*blockIdx.y) += smem[lj][0];
		}
	}
	
}


__global__
void reduce0(Mesh f, Mesh out)
{
	__shared__ Real smem[NY_TILE][NZ_TILE];
	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;

	const Int lj = threadIdx.x;
	const Int lk = threadIdx.y;

	if (j < f.ny_ && k < f.nz_)
	{
		/// Load shared mem
		smem[lj][lk] = f(0,j,k);

		/// Do reduction in shared mem
		for (size_t s=1;s<blockDim.y; s*= 2)
		{
			Int index = 2 * s * lk;
			if (index < blockDim.y)
			{
				smem[lj][index] += smem[lj][index +s];
			}
			__syncthreads();
		}

		//write result to global mem
		if (lk == 0) out(0,j,blockIdx.y) = smem[lj][0];
	}
	
}




__global__
void reduce0(Mesh f, Mesh out)
{
	__shared__ Real smem[NY_TILE][NZ_TILE];
	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;

	const Int lj = threadIdx.x;
	const Int lk = threadIdx.y;

	if (j < f.ny_ && k < f.nz_)
	{
		/// Load shared mem
		smem[lj][lk] = f(0,j,k);

		/// Do reduction in shared mem
		for (size_t s=1;s<blockDim.y; s*= 2)
		{
			if (lk % (2*s) == 0)
			{
				smem[lj][lk] += smem[lj][lk +s];
			}
			__syncthreads();
		}

		//write result to global mem
		if (lk == 0) out(0,j,blockIdx.y) = smem[lj][0];
	}
	
}














 */
