#define __ROLLING_CACHE__ 1 /// Enables the rolling cache NZ/NY_TILE defs

#include "common.h"

__host__ void printfield(Real *h_mem,const Int Nsize, const Int Nvars, const Int onecomp);
__host__ Real fieldsum(Real *h_mem,const Int Nsize, const Int Nvars);
__host__ Real cosdiffAbs(Real *h_mem,const Int Nsize,const Real *x, const Int Nvars);
__host__ Real cosdiffAbsMean(Real *h_mem,const Int Nsize, const Real *x, const Int Nvars);

__device__ void bundleRollCache(Real *B, const Int vi)
{
	/// Roll the cache along bundle
	for (Int bi=-NG;bi<NG;bi++)
	{
		for (Int q=0;q<4*NG+1;q++)
		{
			B[bIdx(bi,q,vi)] = B[bIdx(bi+1,q,vi)];
		}
	}
	
}

__device__ void bundleInit(Real *B, const Real *f, const Int j, const Int k, const Int vi)
{
	for (Int bi=-(NG-1);bi<NG+1;bi++)
	{
		B[bIdx(bi,0,vi)] = f[fIdx(bi-1,j,k,vi)];
		B[bIdx(bi,1,vi)] = f[fIdx(bi-1,j+1,k,vi)];
		B[bIdx(bi,2,vi)] = f[fIdx(bi-1,j,k-1,vi)];
		B[bIdx(bi,3,vi)] = f[fIdx(bi-1,j-1,k,vi)];
		B[bIdx(bi,4,vi)] = f[fIdx(bi-1,j,k+1,vi)];
		B[bIdx(bi,5,vi)] = f[fIdx(bi-1,j+2,k,vi)];
		B[bIdx(bi,6,vi)] = f[fIdx(bi-1,j,k-2,vi)];
		B[bIdx(bi,7,vi)] = f[fIdx(bi-1,j-2,k,vi)];
		B[bIdx(bi,8,vi)] = f[fIdx(bi-1,j,k+2,vi)];
	}
}

__global__ void grad_kernel(const Real *f, Real *df, const Real dx)
{

	__shared__ Real fs[3][NY_TILE+2*NG][NZ_TILE+2*NG];

	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;
	
	/// Local indices
	const Int bj = threadIdx.x + NG;
	const Int bk = threadIdx.y + NG;
	const Int bi = 0; /// the "center" of the bundle. This will always be zero
	                  /// for any global index i along the array.

	//const Int nvars = 3;
	const Int vi=0; /// Just for testing scalar function
		
	/// Thread local bundle, stencil size in every direction:
	Real B[(4*NG+1)*(2*NG+1)];

	/// Initialisation
	for (Int vi=0;vi<nvars;vi++)
	{
		bundleInit(B,f,j,k,vi);
		__syncthreads();
	}

	if (j < NY && k < NZ)
	{
		for (Int i=0;i<NX;i++)
		{
			/// For every component
			for (Int vi=0;vi<nvars;vi++)
			{
				/// Load shared memory 
				fs[vi][bj][bk] = f[fIdx(i+2,j,k,vi)];

				/// If at yz-tile edges assign ghost points
				if (bj == NG)
				{
					fs[vi][bj-1][bk] = f[fIdx(i+2,j-1,k,vi)];
					fs[vi][bj-2][bk] = f[fIdx(i+2,j-2,k,vi)];
					fs[vi][bj+NY_TILE][bk] = f[fIdx(i+2,j+NY_TILE,k,vi)];
					fs[vi][bj+NY_TILE+1][bk] = f[fIdx(i+2,j+NY_TILE+1,k,vi)];
				}
				if (bk == NG)
				{
					fs[vi][bj][bk-1] = f[fIdx(i+2,j,k-1,vi)];
					fs[vi][bj][bk-2] = f[fIdx(i+2,j,k-2,vi)];
					fs[vi][bj][bk+NZ_TILE] = f[fIdx(i+2,j,k+NZ_TILE,vi)];
					fs[vi][bj][bk+NZ_TILE+1] = f[fIdx(i+2,j,k+NZ_TILE+1,vi)];
				}
				__syncthreads();

				/// *** ___ Roll the cache ! ___ ***
				/// Load shared tile into local bundle
				for (Int q=0;q<4*NG+1;q++)
				{
					B[bIdx(-2,q,vi)] = B[bIdx(-1,q,vi)];
					B[bIdx(-1,q,vi)] = B[bIdx(0,q,vi)];
					B[bIdx(0,q,vi)] = B[bIdx(1,q,vi)];
					B[bIdx(1,q,vi)] = B[bIdx(2,q,vi)];
				}

				/// Add last element from shared tile
				B[bIdx(NG,0,vi)] = fs[vi][bj][bk];
				B[bIdx(NG,1,vi)] = fs[vi][bj+1][bk];
				B[bIdx(NG,2,vi)] = fs[vi][bj][bk-1];
				B[bIdx(NG,3,vi)] = fs[vi][bj-1][bk];
				B[bIdx(NG,4,vi)] = fs[vi][bj][bk+1];
				B[bIdx(NG,5,vi)] = fs[vi][bj+2][bk];
				B[bIdx(NG,6,vi)] = fs[vi][bj][bk-2];
				B[bIdx(NG,7,vi)] = fs[vi][bj-2][bk];
				B[bIdx(NG,8,vi)] = fs[vi][bj][bk+2];
			}
			/// *** ___ Perform bundle -> pencil operations  ___ ***
			
			sgrad(B,P,i,1.0/dx,1.0/dx,1.0/dx);
			
			/// *** ___ Copy pencil to 
			df[fIdx(i,j,k,vi)] =
				sgrad(B,P,i,1.0/dx,1.0/dx,1.0/dx); /// bi should be 0 always!
	
		}
		
		
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
	dim3 tpb(NY_TILE,NZ_TILE); 
	dim3 blx(NN/NY_TILE,NN/NZ_TILE);

	cudaCheck(cudaEventRecord(start));
	pbc_x_kernel<<<blx,tpb>>>(d_in,NX,NG,1);
	pbc_y_kernel<<<blx,tpb>>>(d_in,NY,NG,1);
	pbc_z_kernel<<<blx,tpb>>>(d_in,NZ,NG,1);
	
	//cudaCheck(cudaDeviceSynchronize()); /// Needed if several streams
	
	grad_kernel<<<blx,tpb>>>(d_in,d_out,dx);
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

	printf("Time taken for GPU rolling cache bundle z-derivative: %.4f ms\n", ms);
	printf("Including pbc_xyz kernels!\n");
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
