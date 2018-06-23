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
		cudaDeviceProp prop; //Properties struct
		cudaCheck(cudaGetDeviceProperties(&prop, i));
		printf("Device number: %i \n", i);
		printf("    Device name: %s \n", prop.name);
		printf("    Peak memory bandwidth (GB/s): %f\n",
		       2.0*prop.memoryClockRate*(prop.memoryBusWidth/8.0)/1.0e6);
	}

	

	return 0;
}
