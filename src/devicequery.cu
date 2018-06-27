#include <stdio.h>

#include "errcheck.h"
#include "typedefs.h" //int->Int, double->Real



Int main()
{
	Int nDevices;

	cudaCheck(cudaGetDeviceCount(&nDevices));

	//Loop over devices.
	for (Int i=0;i<nDevices;i++)
	{
		/// Comments are for gtx 850M on laptop
		cudaDeviceProp prop; //Properties struct
		cudaCheck(cudaGetDeviceProperties(&prop, i));
		printf("Device number: %i \n", i);
		printf("    Device name: %s \n", prop.name);
		printf("    Peak memory bandwidth (GB/s): %f\n",
		       2.0*prop.memoryClockRate*(prop.memoryBusWidth/8.0)/1.0e6);
		printf("    Compute capability: %i.%i \n", prop.major, prop.minor); //5.0
		printf("    Total amount of shared memory per block: %lu Real (double) values \n",
		       prop.sharedMemPerBlock); // 49k doubles => ~10 (512+4)*9 sized bundles.
		printf("    Max threads per block: %d\n", prop.maxThreadsPerBlock);//1024
		printf("    Warp size: %d\n", prop.warpSize); //32

	}

	

	return 0;
}
