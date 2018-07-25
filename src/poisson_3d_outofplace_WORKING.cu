#include <cufft.h>
#include "errcheck.h"
#include "cuffterr.h"
#include <iostream>
#include <iomanip>
#include <cmath>

#define NN 512
#define NX NN
#define NY NN
#define NZ NN
#define NZH NZ/2+1
#define NX_TILE 16
#define NY_TILE 16
#define NZ_TILE NN

typedef int Int;
typedef double Real;
typedef double2 Complex;

__host__ __device__
inline size_t cindx(size_t i, size_t j, size_t k)
{
	return k+((NZ >> 1)+1)*(j+NY*i); /// (NZ >> 1) is a bitshift expression equiv to NZ/2
}

__host__ __device__
inline size_t rindx(size_t i, size_t j, size_t k)
{
	return k+NZ*(j+NY*i); /// (NZ >> 1) is a bitshift expression equiv to NZ/2
}


__global__
void freqDiv_kernel(Complex *f, const Real xlen)
{
	Int II,JJ; 
	Int j = threadIdx.y+blockIdx.y*blockDim.y;
	Int i = threadIdx.x+blockIdx.x*blockDim.x;
        Real k1,k2,k3,Pi=M_PI,fac;
	
	//if (i < NX && j < NY && k < ((NZ >> 1) + 1))
	if (i < NX && j < NY)
	{
		if (2*i<NX)
			II = i;
		else
                        II = NX-i;

		if (2*j<NY)
			JJ = j;
		else
			JJ = NX-j;
		
		k2 = 2*Pi*JJ/xlen;
                k1 = 2*Pi*II/xlen;
		for (Int k=0;k<((NZ >> 1)+1);k++)
		{
			k3 = 2*Pi*k/xlen;
			fac = -1.0*(k1*k1+k2*k2+k3*k3);

			if (fabs(fac) < 1e-14)
			{
				f[cindx(i,j,k)].x = 0.0;
				f[cindx(i,j,k)].y = 0.0;	
			}
			else
			{
				f[cindx(i,j,k)].x = f[cindx(i,j,k)].x/fac;
				f[cindx(i,j,k)].y = f[cindx(i,j,k)].y/fac;
			}
		}
	}
}

void printComplex(Complex *f)
{
	for (Int i=0;i<NX;i++)
	{
		for (Int j=0;j<NY;j++)
		{
			for (Int k=0;k<(NZ/2+1);k++)
			{
				std::cout << std::setprecision(3) << f[cindx(i,j,k)].x << "," <<
					f[cindx(i,j,k)].y << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "----- " << std::endl;
	}
}

void printReal(Real *f)
{
	Real norm = NX*NY*NZ;
	for (Int i=0;i<NX;i++)
	{
		for (Int j=0;j<NY;j++)
		{
			for (Int k=0;k<NZ;k++)
			{
				std::cout << std::setprecision(3) << f[rindx(i,j,k)]/norm << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "----- " << std::endl;
	}
}

void initReal(Real *f, Real *x, Real q1, Real q2)
{
	for (Int i=0;i<NX;i++)
	{
		for (Int j=0;j<NY;j++)
		{
			for (Int k=0;k<NZ;k++)
			{
				f[rindx(i,j,k)] = sin(q1*x[k])+sin(q2*x[j]);
				//f[rindx(i,j,k)] = sin(x[k]);
			}
		}
	}
}

void normalise(Real *f)
{
	Real norm = NX*NY*NZ;
	for (Int i=0;i<NX;i++)
	{
		for (Int j=0;j<NY;j++)
		{
			for (Int k=0;k<NZ;k++)
			{
				f[rindx(i,j,k)] /= norm;
			}
		}
	}
}

void testResult(Real *f, Real *x,Real q1,Real q2)
{
	Real maxErr = 0.0,diff=0.0,anltc=0.0;
	for (Int i=0;i<NX;i++)
	{
		for (Int j=0;j<NY;j++)
		{
			for (Int k=0;k<NZ;k++)
			{
				anltc = -1.0*((sin(q1*x[k])/(q1*q1))+(sin(q2*x[j])/(q2*q2)));
				//anltc = -1.0*sin(x[k]);
				diff = fabs(anltc-f[rindx(i,j,k)]);
				if (diff > maxErr)
					maxErr = diff;
			}
		}
	}
	std::cout << "Max error:" << maxErr << std::endl;
}


Int main()
{

	cufftHandle pland2z,planz2d;
	CUFFT_CHECK(cufftPlan3d(&pland2z,NX,NY,NZ,CUFFT_D2Z));
	CUFFT_CHECK(cufftPlan3d(&planz2d,NX,NY,NZ,CUFFT_Z2D));
	
	Real *d_real,*h_real;
	cudaCheck(cudaMallocHost((void**)&h_real,sizeof(Real)*NX*NY*NZ));
	cudaCheck(cudaMalloc((void**)&d_real,sizeof(Real)*NX*NY*NZ));
	
	Complex *d_xform,*h_xform;
	cudaCheck(cudaMallocHost((void**)&h_xform,sizeof(Complex)*NX*NY*(NZ/2+1)));
	cudaCheck(cudaMalloc((void**)&d_xform,sizeof(Complex)*NX*NY*(NZ/2+1)));

	///-----------------
	double linspace[NX];
	double L0=0.0,L1=2*M_PI;
	double dx = (L1-L0)/NX;
	
	for (int i=0;i<NX;i++)
		linspace[i] = i*dx;

	Real q1=1.0;
	Real q2=q1;
	initReal(h_real,linspace,q1,q2);
	cudaCheck(cudaMemcpy(d_real,h_real,sizeof(Real)*NX*NY*NZ,cudaMemcpyHostToDevice));
	
	dim3 blx(NX/NX_TILE,NY/NY_TILE);
	dim3 tpb(NX_TILE,NY_TILE);
	
	///--------------
	std::cout << "\n Executing forward R2C transform... \n\n";
	if (cufftExecD2Z(pland2z,d_real,d_xform) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: ExecD2Z Forward failed");
		return 0;	
	}

	///--------------
	freqDiv_kernel<<<blx,tpb>>>(d_xform,L1-L0);

	///--------------
	std::cout << "\n Executing backward C2R transform... \n\n";
	if (cufftExecZ2D(planz2d,d_xform,d_real) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: ExecZ2D backward failed");
		return 0;	
	}

	cudaCheck(cudaMemcpy(h_real,d_real,sizeof(Real)*NX*NY*NZ,cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(h_xform,d_xform,sizeof(Complex)*NX*NY*(NZ/2+1),cudaMemcpyDeviceToHost));

	//printReal(h_real);
	normalise(h_real);
	testResult(h_real,linspace,q1,q2);
	       
	return 0;
}
