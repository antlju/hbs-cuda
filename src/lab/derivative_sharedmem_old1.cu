/// Code for testing the bundle/pencil framework on
/// __shared__ memory with a simple derivative example of
/// a scalar field.

/// Library includes
#include <cmath>

/// Project local includes
#include "typedefs.h"
#include "fd_params.h"
#include "errcheck.cuh"
#include "indexing.h"
#include "printing.h"

#define NN NSIZE ///From fd_params.h
#define BUNDLESIZE 4*NGHOSTS+1

/// Central FD coefficients for 1st derivative 4th accuracy order. Antisymmetric coefficients.
//__constant__ Real d1_4_2C; //Coeff for +-2h, where h is the stepsize.
//__constant__ Real d1_4_1C; //+-1h

__device__ Real fd4d1(const Real m2h,const Real m1h,const Real p1h,const Real p2h)
{
	Real d1_4_2C = -1.0/12;
	Real d1_4_1C = 2.0/3;
        return d1_4_2C*(p2h-m2h)+d1_4_1C*(p1h-m1h);
}

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
	Int indx = q*(NX+2*NGHOSTS)+vi*(NX+2*NGHOSTS)+(i+NGHOSTS);
	return indx; 
}

__device__ Real delz(const Real *B, const Real invdz, const Int i, const Int vi)
{
	return invdz * fd4d1(B[bIdx(i-2,0,vi)],B[bIdx(i-1,0,vi)],
                                B[bIdx(i+1,0,vi)],B[bIdx(i+2,0,vi)]);
}

__global__ void deriv_kernel(const Real *f, Real *df, const Real dx)
{


	/// Declare block-specific shared bundle memory.
	__shared__ Real Bndl[NN][(NN+2*NGHOSTS)*BUNDLESIZE];
	__shared__ Int jk[2];
	
	const Int j = blockIdx.x;
	const Int k = threadIdx.y;
	const Int i = threadIdx.x;

	if (i < NN && j < NN && k < NN)
	{
		Bndl[k][bIdx(i,0,0)] = f[fIdx(i,j,k)];
		Bndl[k][bIdx(i-2,0,0)] = f[fIdx(i-2,j,k)];
		Bndl[k][bIdx(i-1,0,0)] = f[fIdx(i-1,j,k)];
		Bndl[k][bIdx(i+1,0,0)] = f[fIdx(i+1,j,k)];
		Bndl[k][bIdx(i+2,0,0)] = f[fIdx(i+2,j,k)];
		__syncthreads();

		
		/// Compute diff op
		df[fIdx(i,j,k,0)] = (1.0/dx) * fd4d1(Bndl[k][bIdx(i-2,0,0)],Bndl[k][bIdx(i-1,0,0)],
                                Bndl[k][bIdx(i+1,0,0)],Bndl[k][bIdx(i+2,0,0)]);
	}
		
}


__host__ void set_linspace(Real *linspace, const Real dx, const Int Nsize)
{
	for (Int i=0;i<Nsize;i++)
	{
		linspace[i] = i*dx;
	}
	
}

__host__ void init_hostmem(Real *h_mem, const Real *x, const Int Nsize)
{
	
	for (Int i=0;i<Nsize;i++)
	{
		for (Int j=0;j<Nsize;j++)
		{
			for (Int k=0;k<Nsize;k++)
			{
				h_mem[fIdx(i,j,k,0)] = sin(x[k]);
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
	
	Int N3 = (NN+2*NGHOSTS)*(NN+2*NGHOSTS)*(NN+2*NGHOSTS); //NN cubed
	Int Nsize = NN;
	/// Set up host memory
	Real *hostMem,*linspace,*hostMem_d;
	checkCuda(cudaMallocHost(&linspace,sizeof(Real)*NN));
	checkCuda(cudaMallocHost(&hostMem,sizeof(Real)*N3));
	memset(hostMem,0.0,sizeof(Real)*N3);
	checkCuda(cudaMallocHost(&hostMem_d,sizeof(Real)*N3));

	/// Set up linspace
	Real L0 = 0;
	Real L1 = 2*M_PI;
	Real dx = (L1-L0)/(Nsize); 
	set_linspace(linspace,dx,NN);
	printlin(linspace,NN);
	/// Initialise hostmemory
	init_hostmem(hostMem,linspace,NN);

	printfield(hostMem,NN,1);

	
	/// Set up device memory
	Real *d_f,*d_df;
	checkCuda(cudaMalloc((void**)&d_f,sizeof(Real)*N3));
	checkCuda(cudaMemset(d_f,0.0,sizeof(Real)*N3));
	checkCuda(cudaMalloc((void**)&d_df,sizeof(Real)*N3));

	/// Copy from host to device
	checkCuda(cudaMemcpy(d_f,hostMem,sizeof(Real)*N3,cudaMemcpyHostToDevice));

	/// Call kernel
	const Int threadsPerBlock = 1024;
	const Int blocks = (NN+threadsPerBlock-1)/threadsPerBlock;
	dim3 tpB(NN,NN);
	//deriv_kernel<<<blocks,threadsPerBlock>>>(d_f,d_df);
	deriv_kernel<<<NN,tpB>>>(d_f,d_df,dx);

	checkCuda(cudaMemcpy(hostMem_d,d_df,sizeof(Real)*N3,cudaMemcpyDeviceToHost));

	printfield(hostMem_d,NN,1);
	
	/// Free memory
	checkCuda(cudaFreeHost(hostMem));
	checkCuda(cudaFreeHost(linspace));
	checkCuda(cudaFree(d_f));
	checkCuda(cudaFree(d_df));
	
	return 0;
}
