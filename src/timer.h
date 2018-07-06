#pragma once
#include <stdio.h>

class Timer
{
public:
	cudaEvent_t start, stop;
	float ms; //Needs to be float for CUDA API.
	__host__ void createEvents()
	{
		cudaCheck(cudaEventCreate(&start));
		cudaCheck(cudaEventCreate(&stop));
	}

	__host__ void recordStart()
	{
		cudaCheck(cudaEventRecord(start));
	}
	__host__ void recordStop()
	{
		cudaCheck(cudaEventRecord(stop));
	}
	__host__ void synch()
	{
		cudaCheck(cudaEventSynchronize(stop));
	}

	__host__ void print()
	{
		cudaCheck(cudaEventElapsedTime(&ms, start, stop));
		printf("Time taken: %.4f ms\n", ms);
	}
	
}; /// End class Timer
