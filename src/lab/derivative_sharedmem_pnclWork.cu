/// Code for testing the bundle/pencil framework on
/// __shared__ memory with a simple derivative example of
/// a scalar field.

/// Library includes
#include <cmath>
#include <string>
#include <iostream>

/// Project local includes
#include "typedefs.h"
#include "fd_params.h"
#include "errcheck.cuh"
#include "indexing.h"
#include "printing.h"

#define NN NSIZE ///From fd_params.h
#define BUNDLESIZE 4*NGHOSTS+1 // Number of pencils in a bundle, ie the FD stencil


__device__ void qtojk(Int *jk, const Int q)
{
	
	Int qtoj[9],qtok[9];
	qtoj[0] = 0;
        qtoj[1] = 1;
        qtoj[2] = 0;
        qtoj[3] = -1;
        qtoj[4] = 0;
        qtoj[5] = 2;
        qtoj[6] = 0;
        qtoj[7] = -2;
        qtoj[8] = 0;

        qtok[0] = 0;
        qtok[1] = 0;
        qtok[2] = -1;
        qtok[3] = 0;
        qtok[4] = 1;
        qtok[5] = 0;
        qtok[6] = -2;
        qtok[7] = 0;
        qtok[8] = 2;

	jk[0] = qtoj[q];
	jk[1] = qtok[q];
	
}



/// Bundle indexing. Takes care of ghost points.
__host__ __device__ size_t bIdx(Int i, Int q=0, Int vi=0)
{
	//std::cout << q << std::endl;
	//assert(q>=0 && q<4*H+1);
	//assert(vi>=0 && vi<(Int)nvars_);
	Int indx = q*(NN+2*NGHOSTS)+vi*(NN+2*NGHOSTS)+(i+NGHOSTS);
	return indx; 
}

__global__ void deriv_kernel(const Int *f, Int *df)
{
	/// Let's test with NN bundles per block so that we launch NN*NN threads per block.
	/// We also have NN blocks so that in total we have NN*NN*NN threads == size of array.
	/// So for every fixed j we have NN bundles (one for each k), and each bundle has NN
	/// elements along each "pencil".

	/// Declare block-specific shared bundle memory.
	__shared__ Int Pncl[NN][NN];
	//const Int gidx = blockIdx.x * blockDim.x * blockDim.y
	//	+ threadIdx.y * blockDim.x + threadIdx.x;

	const Int j = threadIdx.y;
	const Int k = threadIdx.x;
	const Int i = blockIdx.x;

	if (i < NN && j < NN && k < NN)
	{
		Pncl[j][k] = f[fIdx(i,j,k)];
		__syncthreads();
		df[fIdx(i,j,k)] = Pncl[j][k];

		
	}
}

__host__ void init_hostmem(Int *h_mem, const Int Nsize)
{
	
	for (Int i=0;i<Nsize;i++)
	{
		for (Int j=0;j<Nsize;j++)
		{
			for (Int k=0;k<Nsize;k++)
			{
				h_mem[fIdx(i,j,k,0)] = fIdx(i,j,k,0);
			}
		}
	}
}

Int main()
{
	/// FD constants
	Real host_d1_4_2C = -1.0/12;
	Real host_d1_4_1C = 2.0/3;
	//checkCuda(cudaMemcpyToSymbol("d1_4_2C",&host_d1_4_2C,sizeof(Real),0,cudaMemcpyHostToDevice));
	//checkCuda(cudaMemcpyToSymbol("d1_4_1C",&host_d1_4_1C,sizeof(Real),0,cudaMemcpyHostToDevice));
	
	Int N3 = NN*NN*NN; //NN cubed
	Int Nsize = NN;
	/// Set up host memory
	//Real *hostMem,*linspace,*hostMem_d;
	Int *hostMem,*hostMem_d;
	checkCuda(cudaMallocHost(&hostMem,sizeof(Int)*N3));
	checkCuda(cudaMallocHost(&hostMem_d,sizeof(Int)*N3));

	/// Initialise hostmemory
	init_hostmem(hostMem,NN);

	printfield(hostMem,NN,1);
	
	/// Set up device memory
	Int *d_f,*d_df;
	checkCuda(cudaMalloc((void**)&d_f,sizeof(Int)*N3));
	checkCuda(cudaMalloc((void**)&d_df,sizeof(Int)*N3));

	/// Copy from host to device
	checkCuda(cudaMemcpy(d_f,hostMem,sizeof(Int)*N3,cudaMemcpyHostToDevice));

	/// Call kernel
	const Int threadsPerBlock = 1024;
	const Int blocks = (NN+threadsPerBlock-1)/threadsPerBlock;
	dim3 tpB(NN,NN);
	//deriv_kernel<<<blocks,threadsPerBlock>>>(d_f,d_df);
	deriv_kernel<<<NN,tpB>>>(d_f,d_df);

	checkCuda(cudaMemcpy(hostMem_d,d_df,sizeof(Int)*N3,cudaMemcpyDeviceToHost));

	printfield(hostMem_d,NN,1);

	
	/// Free memory
	checkCuda(cudaFreeHost(hostMem));
	checkCuda(cudaFree(d_f));
	checkCuda(cudaFree(d_df));
	return 0;
}
