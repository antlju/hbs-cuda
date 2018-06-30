#include <cmath>
#include "typedefs.h"
#include "errcheck.h"
#define NN 8
#define NX NN
#define NY NN
#define NZ NN
#define NG 2
#define NGHOSTS NG
#define NY_TILE 8
#define NZ_TILE 8
#define NX_TILE 1

__host__ void printfield(Real *h_mem,const Int Nsize, const Int Nvars, const Int onecomp);
__host__ Real fieldsum(Real *h_mem,const Int Nsize, const Int Nvars);
__host__ Real cosdiffAbs(Real *h_mem,const Int Nsize,const Real *x, const Int Nvars);
__host__ Real cosdiffAbsMean(Real *h_mem,const Int Nsize, const Real *x, const Int Nvars);

/// This gives the (i,j,k)-coordinate for component vi of a discretised 3D vector field
/// stored as a linear array in memory.
/// ROW MAJOR ORDER! (k fastest index, i slowest). This is C-style.
__host__ __device__ inline size_t fIdx(const Int i, const Int j, const Int k, const Int vi=0)
{
	return vi*(NZ+2*NGHOSTS)*(NY+2*NGHOSTS)*(NX+2*NGHOSTS)
		+(k+NGHOSTS)+(NX+2*NGHOSTS)*((j+NGHOSTS)+(NY+2*NGHOSTS)*(i+NGHOSTS));
}


__host__ __device__ void apply_pbc(Real *u)
{
	size_t Ng = NGHOSTS,Nvars=1;
        size_t Nx=NN,Ny=NN,Nz=NN;
        
        for (size_t vi=0;vi<Nvars;vi++)
        {
                
                for (size_t i=0;i<Nx;i++)
                {
                        for (size_t j=0;j<Ny;j++)
                        {
                                for (size_t k=0;k<Nz;k++)
                                {
                                        for (size_t l=0;l<Ng;l++)
                                        {
                                                //set pbc along x
                                                u[fIdx(l-Ng,j,k,vi)] = u[fIdx(Nx-(Ng-l),j,k,vi)];
                                                u[fIdx(Nx+l,j,k,vi)] = u[fIdx(l,j,k,vi)];

                                                //set pbc along y
                                                u[fIdx(i,l-Ng,k,vi)] = u[fIdx(i,Ny-(Ng-l),k,vi)];
                                                u[fIdx(i,Ny+l,k,vi)] = u[fIdx(i,l,k,vi)];

                                                //set pbc along z
                                                u[fIdx(i,j,l-Ng,vi)] = u[fIdx(i,j,Nz-(Ng-l),vi)];
                                                u[fIdx(i,j,Nz+l,vi)] = u[fIdx(i,j,l,vi)];
                                        }
                                }
                        }
                }
               
        }
}

__device__ Real fd4d1(const Real m2h,const Real m1h,const Real p1h,const Real p2h)
{
	Real d1_4_2C = -1.0/12;
	Real d1_4_1C = 2.0/3;
        return d1_4_2C*(p2h-m2h)+d1_4_1C*(p1h-m1h);
}


__device__ Real delz(const Real B[4*NG+1][NN+2*NG], const Real dzfactor, const Int i, const Int vi)
{
        return dzfactor * fd4d1(B[6][i],B[2][i],
		     B[4][i],B[8][i]);
}

__global__ void kernel(Real *f, Real *df, const Real dx)
{

	__shared__ Real B[2*NG+1][NY_TILE+2*NG][NZ_TILE+2*NG];

	/// Global indices
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;

	/// Local indices
	const Int bk = threadIdx.y + NG;
	const Int bj = threadIdx.x + NG;
	const Int bim2 = 0,bim1 = 1, bi0 = 2, bip1 = 3, bip2 = 4;

	if (j < NY && k < NZ)
	{
		//B[bim2][bk][bj] = 0.0;
		B[bim1][bk][bj] = f[fIdx(-2,j,k)];
		__syncthreads();
		B[bi0][bk][bj] = f[fIdx(-1,j,k)];
		__syncthreads();
		B[bip1][bk][bj] = f[fIdx(0,j,k)];
		__syncthreads();
		B[bip2][bk][bj] = f[fIdx(1,j,k)];
		__syncthreads();

		for (Int i=0;i<NN;i++)
		{
			B[bim2][bk][bj] = B[bim1][bk][bj];
			__syncthreads();
			B[bim1][bk][bj] = B[bi0][bk][bj];
			__syncthreads();
			B[bi0][bk][bj] = B[bip1][bk][bj];
			__syncthreads();
			B[bip1][bk][bj] = B[bip2][bk][bj];
			__syncthreads();
			B[bip2][bk][bj] = f[fIdx(i+NG,j,k)];
			__syncthreads();
			
			df[fIdx(i,j,k)] = 1.0/dx * fd4d1(B[bi0][bj][bk-2],B[bi0][bj][bk-1],
							 B[bi0][bj][bk+1],B[bi0][bj][bk+2]);
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
				h[fIdx(i,j,k)] = fIdx(i,j,k);//sin(x[k]);
			}
		}
	}
}

Int main()
{
	Int NNG3 = (NN+2*NG)*(NN+2*NG)*(NN+2*NG);
	Real *h_in,*h_out,*d_in,*d_out,*linspace;
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
	apply_pbc(h_in);
	
	//printfield(h_in,NN,1);
	//printf("host init sum: %f\n", fieldsum(h_in,NN,1));
	
	cudaCheck(cudaMemcpy(d_in,h_in,sizeof(Real)*NNG3,cudaMemcpyHostToDevice));
	//const Int tpb = 256;
	//dim3 threadsPerBlock(tpb);
	//const Int blcks = (NN+tpb-1)/tpb;
	//dim3 blocks((NN+tpb+1)/tpb);

	//dim3 tpb(NY_TILE,NZ_TILE); //1024 max threads per block on Quadro P4000
	dim3 tpb(NY_TILE,NZ_TILE); 
	dim3 blx(NN/NY_TILE,NN/NZ_TILE);
	kernel<<<blx,tpb>>>(d_in,d_out,dx);
	
	cudaCheck(cudaMemcpy(h_out,d_out,sizeof(Real)*NNG3,cudaMemcpyDeviceToHost));

	const Int printOneComp = 1;
	printfield(h_out,NN,1,printOneComp);
	printfield(h_in,NN,1,printOneComp);
	//printf("host out sum: %f\n", fieldsum(h_out,NN,1));
	//printf("%.3f", cosdiffAbsMean(h_out,NN,linspace,1));
	
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
	return sum/(NN*NN*NN);
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

	Real sum = 0;
	for (Int vi=0;vi<Nvars;vi++)
	{
		for (Int i=0;i<Nsize;i++)
		{
			for (Int j=0;j<Nsize;j++)
			{
				for (Int k=0;k<Nsize;k++)
				{
					//printf("%.3f ", fabs(h_mem[fIdx(i,j,k,vi)]-cos(x[k])));
					sum += fabs(h_mem[fIdx(i,j,k,vi)]-cos(x[k]));
				}
				//printf("\n");
			}
			//printf("\n\n");
		}

	}
	return sum/(Nsize*Nsize*Nsize);
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
