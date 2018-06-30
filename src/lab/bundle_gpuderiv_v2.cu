#include <cmath>
#include "fd_params.h"
#include "typedefs.h"
#include "errcheck.h"
#include "indexing.h"
#include "pbc_kernel.h"


__host__ void printfield(Real *h_mem,const Int Nsize, const Int Nvars, const Int onecomp);
__host__ Real fieldsum(Real *h_mem,const Int Nsize, const Int Nvars);
__host__ Real cosdiffAbs(Real *h_mem,const Int Nsize,const Real *x, const Int Nvars);
__host__ Real cosdiffAbsMean(Real *h_mem,const Int Nsize, const Real *x, const Int Nvars);

__device__ Real fd4d1(const Real m2h,const Real m1h,const Real p1h,const Real p2h)
{
	Real d1_4_2C = -1.0/12;
	Real d1_4_1C = 2.0/3;
        return d1_4_2C*(p2h-m2h)+d1_4_1C*(p1h-m1h);
}

__device__ Real delz(const Real *B, const Real dzfactor, const Int i, const Int vi)
{
        return dzfactor * fd4d1(B[bIdx(i,6,vi)],B[bIdx(i,2,vi)], //refer to qjkmap.h for q->k vals.
                                B[bIdx(i,4,vi)],B[bIdx(i,8,vi)]);
}

__device__ void loadBundlexGhosts(Real *B, const Real *f, const Int j, const Int k, const Int vi)
{
	B[bIdx(-2,0,vi)] = f[fIdx(-2,j,k,vi)];
	B[bIdx(-2,1,vi)] = f[fIdx(-2,j+1,k,vi)];
	B[bIdx(-2,2,vi)] = f[fIdx(-2,j,k-1,vi)];
	B[bIdx(-2,3,vi)] = f[fIdx(-2,j-1,k,vi)];
	B[bIdx(-2,4,vi)] = f[fIdx(-2,j,k+1,vi)];
	B[bIdx(-2,5,vi)] = f[fIdx(-2,j+2,k,vi)];
	B[bIdx(-2,6,vi)] = f[fIdx(-2,j,k-2,vi)];
	B[bIdx(-2,7,vi)] = f[fIdx(-2,j-2,k,vi)];
	B[bIdx(-2,8,vi)] = f[fIdx(-2,j,k+2,vi)];

	B[bIdx(-1,0,vi)] = f[fIdx(-1,j,k,vi)];
	B[bIdx(-1,1,vi)] = f[fIdx(-1,j+1,k,vi)];
	B[bIdx(-1,2,vi)] = f[fIdx(-1,j,k-1,vi)];
	B[bIdx(-1,3,vi)] = f[fIdx(-1,j-1,k,vi)];
	B[bIdx(-1,4,vi)] = f[fIdx(-1,j,k+1,vi)];
	B[bIdx(-1,5,vi)] = f[fIdx(-1,j+2,k,vi)];
	B[bIdx(-1,6,vi)] = f[fIdx(-1,j,k-2,vi)];
	B[bIdx(-1,7,vi)] = f[fIdx(-1,j-2,k,vi)];
	B[bIdx(-1,8,vi)] = f[fIdx(-1,j,k+2,vi)];

	B[bIdx(NX_TILE,0,vi)] = f[fIdx(NN,j,k,vi)];
	B[bIdx(NX_TILE,1,vi)] = f[fIdx(NN,j+1,k,vi)];
	B[bIdx(NX_TILE,2,vi)] = f[fIdx(NN,j,k-1,vi)];
	B[bIdx(NX_TILE,3,vi)] = f[fIdx(NN,j-1,k,vi)];
	B[bIdx(NX_TILE,4,vi)] = f[fIdx(NN,j,k+1,vi)];
	B[bIdx(NX_TILE,5,vi)] = f[fIdx(NN,j+2,k,vi)];
	B[bIdx(NX_TILE,6,vi)] = f[fIdx(NN,j,k-2,vi)];
	B[bIdx(NX_TILE,7,vi)] = f[fIdx(NN,j-2,k,vi)];
	B[bIdx(NX_TILE,8,vi)] = f[fIdx(NN,j,k+2,vi)];

	B[bIdx(NX_TILE+1,0,vi)] = f[fIdx(NN+1,j,k,vi)];
	B[bIdx(NX_TILE+1,1,vi)] = f[fIdx(NN+1,j+1,k,vi)];
	B[bIdx(NX_TILE+1,2,vi)] = f[fIdx(NN+1,j,k-1,vi)];
	B[bIdx(NX_TILE+1,3,vi)] = f[fIdx(NN+1,j-1,k,vi)];
	B[bIdx(NX_TILE+1,4,vi)] = f[fIdx(NN+1,j,k+1,vi)];
	B[bIdx(NX_TILE+1,5,vi)] = f[fIdx(NN+1,j+2,k,vi)];
	B[bIdx(NX_TILE+1,6,vi)] = f[fIdx(NN+1,j,k-2,vi)];
	B[bIdx(NX_TILE+1,7,vi)] = f[fIdx(NN+1,j-2,k,vi)];
	B[bIdx(NX_TILE+1,8,vi)] = f[fIdx(NN+1,j,k+2,vi)];
}

__global__ void kernel(const Real *f, Real *df, const Real dx)
{

	__shared__ Real B[BUNDLESIZE];
	__shared__ Real P[NX_TILE];
	
	Real *Bptr = &B[0];

	/// Global indices
	const Int i = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.z + blockIdx.z*blockDim.z;
	const Int j = threadIdx.y + blockIdx.y*blockDim.y;

	/// Local indices
	const Int bi = threadIdx.x + NG;
	//const Int bk = threadIdx.z + NG;
	//const Int bj = threadIdx.y + NG;

	const Int vi=0;
	if (i < NX && j < NY && k < NZ)
	{
		B[bIdx(bi,0,vi)] = f[fIdx(i,j,k,vi)];
		B[bIdx(bi,1,vi)] = f[fIdx(i,j+1,k,vi)];
		B[bIdx(bi,2,vi)] = f[fIdx(i,j,k-1,vi)];
		B[bIdx(bi,3,vi)] = f[fIdx(i,j-1,k,vi)];
		B[bIdx(bi,4,vi)] = f[fIdx(i,j,k+1,vi)];
		B[bIdx(bi,5,vi)] = f[fIdx(i,j+2,k,vi)];
		B[bIdx(bi,6,vi)] = f[fIdx(i,j,k-2,vi)];
		B[bIdx(bi,7,vi)] = f[fIdx(i,j-2,k,vi)];
		B[bIdx(bi,8,vi)] = f[fIdx(i,j,k+2,vi)];
		__syncthreads();
		
		if (i == 0)
		{
			loadBundlexGhosts(B,f,j,k,vi);
			__syncthreads();
		}



		P[bi] = delz(B,1.0/dx,bi,vi);

		df[fIdx(i,j,k)] = P[bi];
		
	}		

}

__host__ void set_linspace(Real *linspace, const Real dx, const Int Nsize)
{
	for (Int i=0;i<Nsize;i++)
	{
		linspace[i] = i*dx;
	}
	
}

__host__ void initHost(Real *h, const Real *x, const Int Nsize)
{
	for (Int i=0;i<NN;i++)
	{
		for (Int j=0;j<NN;j++)
		{
			for (Int k=0;k<NN;k++)
			{
				//h[fIdx(i,j,k)] = i+NX*(j+NY*k);
				h[fIdx(i,j,k)] = sin(x[k]);
			}
		}
	}
}

__host__ void initcosx(Real *h, const Real *x, const Int Nsize)
{
	for (Int i=0;i<NN;i++)
	{
		for (Int j=0;j<NN;j++)
		{
			for (Int k=0;k<NN;k++)
			{
				//h[fIdx(i,j,k)] = i+NX*(j+NY*k);
				h[fIdx(i,j,k)] = cos(x[k]);
			}
		}
	}
}

Int main()
{
	
	Int NNG3 = (NN+2*NG)*(NN+2*NG)*(NN+2*NG);
	Real *h_in,*h_out,*d_in,*d_out,*linspace;
	Real *h_cosx;
	cudaCheck(cudaMallocHost(&h_cosx,sizeof(Real)*NNG3));
	cudaCheck(cudaMallocHost(&h_in,sizeof(Real)*NNG3));
	cudaCheck(cudaMallocHost(&h_out,sizeof(Real)*NNG3));
	cudaCheck(cudaMalloc((void**)&d_in,sizeof(Real)*NNG3));
	cudaCheck(cudaMalloc((void**)&d_out,sizeof(Real)*NNG3));

	cudaCheck(cudaMallocHost(&linspace,sizeof(Real)*NN));
	
	/// Set up linspace
	Real L0 = 0;
	Real L1 = 2*M_PI;
	Real dx = (L1-L0)/(NN); 
	set_linspace(linspace,dx,NN);
	//printlin(linspace,NN);
	/// Initialise hostmemory
	initHost(h_in,linspace,NN);
	initcosx(h_cosx,linspace,NN);
	
	//printfield(h_in,NN,1);
	//printf("host init sum: %f\n", fieldsum(h_in,NN,1));

	cudaEvent_t start, stop;
	cudaCheck(cudaEventCreate(&start));
	cudaCheck(cudaEventCreate(&stop));
	cudaCheck(cudaMemcpy(d_in,h_in,sizeof(Real)*NNG3,cudaMemcpyHostToDevice));
	//const Int tpb = 256;
	//dim3 threadsPerBlock(tpb);
	//const Int blcks = (NN+tpb-1)/tpb;
	//dim3 blocks((NN+tpb+1)/tpb);

	//dim3 tpb(NY_TILE,NZ_TILE); //1024 max threads per block on Quadro P4000
	dim3 tpb(NX_TILE,NY_TILE,NZ_TILE); 
	dim3 blx(NN/NX_TILE,NN/NY_TILE,NN/NZ_TILE);

	cudaCheck(cudaEventRecord(start));
	pbc_x_kernel<<<blx,tpb>>>(d_in,NX,NG,1);
	pbc_y_kernel<<<blx,tpb>>>(d_in,NY,NG,1);
	pbc_z_kernel<<<blx,tpb>>>(d_in,NZ,NG,1);
	
	//cudaCheck(cudaDeviceSynchronize()); /// Needed if several streams
	
	kernel<<<blx,tpb>>>(d_in,d_out,dx);
	cudaCheck(cudaEventRecord(stop));
	
	cudaCheck(cudaMemcpy(h_out,d_out,sizeof(Real)*NNG3,cudaMemcpyDeviceToHost));

	cudaCheck(cudaEventSynchronize(stop));
	float ms = 0;
	
	const Int printOneComp = 1;
	//printfield(h_out,NN,1,printOneComp);
	//printfield(h_in,NN,1,printOneComp);
	//printfield(h_cosx,NN,1,printOneComp);
	//printf("host out sum: %f\n", fieldsum(h_out,NN,1));
	printf("Max error: %.10f\n", cosdiffAbsMean(h_out,NN,linspace,1));

	cudaCheck(cudaEventElapsedTime(&ms, start, stop));

	printf("Time taken for GPU bundle/pencil z-derivative: %.4f ms\n", ms);
	printf("Size (N = %i)^3 \n", NN);
	
	cudaCheck(cudaFreeHost(h_in));
	cudaCheck(cudaFreeHost(h_out));
	cudaCheck(cudaFree(d_in));
	cudaCheck(cudaFree(d_out));

	
	return 0;
}

__host__ void printfield(Real *h_mem,const Int Nsize, const Int Nvars, const Int onecomp)
{
	if (onecomp == 1)
	{
		printf("\n");
		for (Int vi=0;vi<Nvars;vi++)
		{
			printf("---------------- COMPONENT %i --------- \n", vi);
			for (Int i=0;i<1;i++)
			{
				for (Int j=0;j<Nsize;j++)
				{
					for (Int k=0;k<Nsize;k++)
					{
						printf("%.3f ", h_mem[fIdx(i,j,k,vi)]);
					}
					printf("\n");
				}
				printf("----------------\n");
			}

		}
		printf("\n");
	}
	else
	{
		printf("\n");
		for (Int vi=0;vi<Nvars;vi++)
		{
			printf("---------------- COMPONENT %i --------- \n", vi);
			for (Int i=0;i<Nsize;i++)
			{
				for (Int j=0;j<Nsize;j++)
				{
					for (Int k=0;k<Nsize;k++)
					{
						printf("%.3f ", h_mem[fIdx(i,j,k,vi)]);
					}
					printf("\n");
				}
				printf("----------------\n");
			}

		}
		printf("\n");
	}
}

__host__ Real fieldsum(Real *h_mem,const Int Nsize, const Int Nvars)
{

	Real sum = 0;
	for (Int vi=0;vi<Nvars;vi++)
	{
		for (Int i=0;i<Nsize;i++)
		{
			for (Int j=0;j<Nsize;j++)
			{
				for (Int k=0;k<Nsize;k++)
				{
					sum += h_mem[fIdx(i,j,k,vi)];
				}
			
			}
		}

	}
	return sum/(NN*NN);
}

__host__ Real cosdiffAbs(Real *h_mem,const Int Nsize, const Real *x, const Int Nvars)
{

	Real sum = 0;
	for (Int vi=0;vi<Nvars;vi++)
	{
		for (Int i=0;i<Nsize;i++)
		{
			for (Int j=0;j<Nsize;j++)
			{
				for (Int k=0;k<Nsize;k++)
				{
					printf("%.3f ", fabs(h_mem[fIdx(i,j,k,vi)]-cos(x[k])));
				}
				printf("\n");
			}
			printf("\n\n");
		}

	}
	return sum/(NN*NN*NN);
}

__host__ Real cosdiffAbsMean(Real *h_mem,const Int Nsize, const Real *x, const Int Nvars)
{

	Real maxE = 0;
	for (Int vi=0;vi<Nvars;vi++)
	{
		for (Int i=0;i<Nsize;i++)
		{
			for (Int j=0;j<Nsize;j++)
			{
				for (Int k=0;k<Nsize;k++)
				{
					//printf("%.3f ", fabs(h_mem[fIdx(i,j,k,vi)]-cos(x[k])));
					//sum += fabs(h_mem[fIdx(i,j,k,vi)]-cos(x[k]));
					Real val = fabs(h_mem[fIdx(i,j,k,vi)]-cos(x[k]));
					if ( val > maxE)
						maxE = val;
				}
				//printf("\n");
			}
			//printf("\n\n");
		}

	}
	//return sum/(Nsize*Nsize*Nsize);
	return maxE;
}

/*
	const Int threadsPerDim = 32; //
	dim3 blockSize(threadsPerDim,threadsPerDim); //Number of threads per block
	                                            //(max for gtx 850M: 1024 = 32*32)
	const Int blocksPerDim = ((NN+NG)+threadsPerDim-1)/threadsPerDim;
	dim3 gridSize(blocksPerDim, blocksPerDim); //Number of blocks per grid

	const Int bundleSize = threadsPerDim*BUNDLESIZE;
	const Int pencilLength = NN;
	const Int pencilsPerBlock = 
	kernel<<<gridSize,blockSize,bundleSize>>>(d_in,d_out);
*/
