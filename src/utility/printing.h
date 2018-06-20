#pragma once

#include <stdio.h>
#include "typedefs.h"
#include "indexing.h"

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
					printf("%f ", h_mem[fIdx(i,j,k,vi)]);
				}
				printf("\n");
			}
			printf("----------------\n");
		}

	}
	printf("\n");
}

__host__ void printlin(Real *linarr,const Int N)
{
	printf("\n");
	printf("Linear array:\n");
	for (Int i=0;i<N;i++)
		printf("%f\n", linarr[i]);

	printf("\n");
}
