#include <cufft.h>
#include "errcheck.h"
#include "cuffterr.h"
#include <iostream>
#include <cmath>

#define NN 4
#define NX NN
#define NY NN
#define NZ NN
#define NZH NZ/2+1


void printComplex(cufftDoubleComplex *f)
{
	for (int i=0;i<NX;i++)
	{
		for (int j=0;j<NY;j++)
		{
			for (int k=0;k<NZH;k++)
			{
				double re = f[k+NZH*(j+NY*i)].x;
				double im = f[k+NZH*(j+NY*i)].y;
				std::cout << re << "," << im << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "--- " << std::endl;
	}
       
}

void printReal(cufftDoubleReal *f)
{
	for (int i=0;i<NX;i++)
	{
		for (int j=0;j<NY;j++)
		{
			for (int k=0;k<NZ;k++)
			{
				double re = f[k+(NZ+2)*(j+NY*i)];
				std::cout << re << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "--- " << std::endl;
	}
       
}

void setReal(cufftDoubleReal *f)
{
	for (int i=0;i<NX;i++)
	{
		for (int j=0;j<NY;j++)
		{
			for (int k=0;k<NZ;k++)
			{
				f[k+(NZ)*(j+NY*i)] = k;
			}

		}

	}
}

int main()
{
	cufftDoubleReal *hreal;
	cudaCheck(cudaMallocHost(&hreal,NX*NY*NZH*sizeof(cufftDoubleComplex)));
	setReal(hreal);
	//printReal(hreal);
	cufftDoubleComplex *d_out = (cufftDoubleComplex *)hreal;
	printComplex(d_out);
	return 0;
}
