#include <iostream>

#define NN 32

typedef int Int;
typedef double Real;


__global__ void add_kernel(Int N, Real *x, Real *y)
{
	Int ti = threadIdx.x;
	if (ti < N)
		x[ti] += y[ti];
}

Int main()
{
	Int N = NN;
	Real *x, *y;
	cudaMallocManaged(&x,N*sizeof(Real));
	cudaMallocManaged(&y,N*sizeof(Real));

	for(Int i=0;i<N;i++)
	{
		x[i] = 1.0; y[i] = 2.0;
	}
	
	add_kernel<<<1,N>>>(N,x,y);
	cudaDeviceSynchronize();

	for (Int i=0;i<N;i++)
	{
		std::cout << x[i] << std::endl;
	}
	
	return 0;
}
