
#include "typedefs.h"
#include "errcheck.h"
#define NN 128
#define NX NN
#define NY NN
#define NZ NN
#define NG 0
#define NGHOSTS NG
#define BUNDLESIZE (NN+2*NG)*(4*NG+1)
#define NX_TILE 4
#define NY_TILE 4

__host__ void printfield(Real *h_mem,const Int Nsize, const Int Nvars);
__host__ Real fieldsum(Real *h_mem,const Int Nsize, const Int Nvars);

/// This gives the (i,j,k)-coordinate for component vi of a discretised 3D vector field
/// stored as a linear array in memory. It
__host__ __device__ inline size_t fIdx(const Int i, const Int j, const Int k, const Int vi=0)
{
	return vi*(NZ+2*NGHOSTS)*(NY+2*NGHOSTS)*(NX+2*NGHOSTS)
		+(i+NGHOSTS)+(NX+2*NGHOSTS)*((j+NGHOSTS)+(NY+2*NGHOSTS)*(k+NGHOSTS));
}

__host__ __device__ inline size_t bIdx(const Int i, const Int q,const Int b)
{
	return (i+NG)+q*(NN+2*NG)+b*BUNDLESIZE;
}

__device__ int getGlobalIdx_2D_1D()
{
	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;				
	int threadId = blockId * blockDim.x + threadIdx.x; 
	return threadId;
}

__device__ Int getGlobalIdx_2D_3D()
{
	Int blockId = blockIdx.x 
			 + blockIdx.y * gridDim.x; 
	Int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
			   + (threadIdx.z * (blockDim.x * blockDim.y))
			   + (threadIdx.y * blockDim.x)
			   + threadIdx.x;
	return threadId;
}

__device__ int getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

__global__ void kernel(Real *f, Real *df)
{

	
	Int k = threadIdx.x + blockIdx.x*blockDim.x;
	Int j = threadIdx.y + blockIdx.y*blockDim.y;

	Int p = 0;
	if (j < NN & k < NN)
	{
		for (Int i=0;i<NN;i++)
		{
			df[fIdx(i,j,k)] = 1;
		}
	}
		

}


__host__ void initHost(Real *h)
{
	for (Int i=0;i<NN;i++)
	{
		for (Int j=0;j<NN;j++)
		{
			for (Int k=0;k<NN;k++)
			{
				//h[fIdx(i,j,k)] = i+NX*(j+NY*k);
				h[fIdx(i,j,k)] = 1;
			}
		}
	}
}

Int main()
{
	Int NNG3 = (NN+2*NG)*(NN+2*NG)*(NN+2*NG);
	Real *h_in,*h_out,*d_in,*d_out;
	cudaCheck(cudaMallocHost(&h_in,sizeof(Real)*NNG3));
	cudaCheck(cudaMallocHost(&h_out,sizeof(Real)*NNG3));
	cudaCheck(cudaMalloc((void**)&d_in,sizeof(Real)*NNG3));
	cudaCheck(cudaMalloc((void**)&d_out,sizeof(Real)*NNG3));

	initHost(h_in);
	//printfield(h_in,NN,1);
	printf("host init sum: %f\n", fieldsum(h_in,NN,1));
	
	cudaCheck(cudaMemcpy(d_in,h_in,sizeof(Real)*NNG3,cudaMemcpyHostToDevice));
	//const Int tpb = 256;
	//dim3 threadsPerBlock(tpb);
	//const Int blcks = (NN+tpb-1)/tpb;
	//dim3 blocks((NN+tpb+1)/tpb);

	dim3 tpb(32,32);
	dim3 blx(8,8);
	kernel<<<blx,tpb>>>(d_in,d_out);
	
	cudaCheck(cudaMemcpy(h_out,d_out,sizeof(Real)*NNG3,cudaMemcpyDeviceToHost));

	//printfield(h_out,NN,1);
	printf("host out sum: %f\n", fieldsum(h_out,NN,1));
	
	cudaCheck(cudaFreeHost(h_in));
	cudaCheck(cudaFreeHost(h_out));
	cudaCheck(cudaFree(d_in));
	cudaCheck(cudaFree(d_out));

	
	return 0;
}

__host__ void printfield(Real *h_mem,const Int Nsize, const Int Nvars)
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
					printf("%.f ", h_mem[fIdx(i,j,k,vi)]);
				}
				printf("\n");
			}
			printf("----------------\n");
		}

	}
	printf("\n");
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
	return sum/((NN+2*NG)*(NN+2*NG)*(NN+2*NG));
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
