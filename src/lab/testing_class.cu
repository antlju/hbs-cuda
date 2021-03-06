/// Example of instantiating a class on the GPU and passing it 

#include <iostream>

typedef double Real;
typedef int Int;

#define NN 32

class GPUClass
{
public:
	Int dim_;
	Real *data_;
	
	__host__ void printData()
	{
		std::cout << "--- printing unified memory data: ----" << std::endl;
		for (Int i=0;i<dim_;i++)
		{
			std::cout << data_[i] << std::endl;
		}
		std::cout << std::endl;
	}
		
};


__global__ void kernel(GPUClass &in)
{
	const Int indx = threadIdx.x;
	in.data_[indx] = 2.0;
}

Int main()
{
	GPUClass instance;
	instance.dim_ = NN;
	cudaMallocManaged((void**)&instance.data_, sizeof(Real)*instance.dim_); //Unified memory allocation

	///Pass to kernel w/o copying
	kernel<<<1,NN>>>(instance);

	instance.printData();
	
	cudaFree(instance.data_);
	return 0;
}



////----------------------------------------------------------------
////----------------------------------------------------------------
////----------------------------------------------------------------
////----------------------------------------------------------------

/// Example that works with a class containing pointers and just handles "regular" memory copy"
/*
class GPUClass
{
public:
	Int dim_;
	Real *d_data_;
	Real *h_data_;

	__host__ void printHostData()
	{
		std::cout << "--- printing host data: ----" << std::endl;
		for (Int i=0;i<dim_;i++)
		{
			std::cout << h_data_[i] << std::endl;
		}
		std::cout << std::endl;
	}
		
};

__global__ void kernel(Real *data)
{
	const Int index = threadIdx.x;
	data[index] = 2.0;
}



Int main()
{
	GPUClass instance;
	instance.dim_ = NN;
	instance.h_data_ = new Real[NN];
	instance.printHostData();
	
	cudaMalloc((void**)&instance.d_data_,sizeof(Real)*instance.dim_);
	cudaMemcpy(instance.d_data_,instance.h_data_,sizeof(Real)*instance.dim_,cudaMemcpyHostToDevice);

	kernel<<<1,instance.dim_>>>(instance.d_data_); //This DOES work!

	cudaMemcpy(instance.h_data_,instance.d_data_,sizeof(Real)*instance.dim_,cudaMemcpyDeviceToHost);

	instance.printHostData();
	
	return 0;
}
*/

////----------------------------------------------------------------
////----------------------------------------------------------------
////----------------------------------------------------------------
////----------------------------------------------------------------

/// Something on the way... can't set actual data like I want and get
/// "Program hit cudaErrorLaunchFailure (error 4) due to "unspecified launch failure" on CUDA API call to cudaFree. "
/// and
/// "Program hit cudaErrorLaunchFailure (error 4) due to "unspecified launch failure" on CUDA API call to cudaFreeHost. "

/*
class GPUClass
{
public:
	Int dimension_;
	Real *data_;
	Real *host_data_;

	__host__ GPUClass(Int dim) :
		dimension_(dim)
	{
		cudaMallocHost((void**)&host_data_, sizeof(Real)*NN);
		cudaMalloc((void**)&data_, sizeof(Real)*NN);
	}

	__host__ ~GPUClass()
	{

	}

	__host__ void printData()
	{
		for (Int i=0;i<dimension_;i++)
		{
			printf("%i : %f \n", i, host_data_[i]);
		}
	}

	__host__ void copyToHost()
	{
		cudaMemcpy(host_data_,data_,sizeof(Real)*NN,cudaMemcpyDeviceToHost);
		for (Int i=0;i<dimension_;i++)
			host_data_[i] += 1.0;
	}

	__device__ void setData(Int index)
	{
		data_[index] = 2;
	}

};

__global__ void kernel(GPUClass &inst)
{
	const Int idx = threadIdx.x;
	if (idx < inst.dimension_)
		inst.setData(idx);

}


Int main()
{
	Int dim = NN;
	GPUClass instance(dim);

	kernel<<<1,1>>>(instance);
	instance.copyToHost();
	instance.printData();
	
	return 0;
}
*/

////----------------------------------------------------------------
////----------------------------------------------------------------
////----------------------------------------------------------------
////----------------------------------------------------------------


/// This is a working example of unified memory
/*
class GPUClass
{
public:
	Int dimension_;
	Real *data_;

	__host__ __device__ GPUClass(Int dim, Real *indataptr) :
		dimension_(dim)
	{
		data_ = indataptr;
	}

	__host__ __device__ ~GPUClass()
	{
		cudaFree(data_);
	}
};

__global__ void kernel(Int N, Real *data)
{
	Int idx = threadIdx.x;
	if (idx < N)
		data[idx] = idx;
}

Int main()
{
	Real *unified;
	cudaMallocManaged(&unified, NN*sizeof(Real));
	Int dim = NN;
	
	//GPUClass instance(dim,unified);

	/// Call the kernel in a single block and w the minimum of SIZE=32 threads
	kernel<<<1,NN>>>(dim, unified);
	cudaDeviceSynchronize();

	for (Int i=0;i<NN;i++)
		std::cout << i << ": " <<  unified[i] << std::endl;

	return 0;
}
*/  

////----------------------------------------------------------------
////----------------------------------------------------------------
////----------------------------------------------------------------
////----------------------------------------------------------------



/// Some old test
/*
__host__ __device__ Class GPUClass
{
public:
	Real *vector_;

	__host__ __device__ GPUClass()
	{
		cudaMalloc((void**)&vector_, sizeof(Real) * SIZE);
	}

	__host__ __device__ ~GPUClass();
	{
		cudaFree(vector_);
	}

};

__global__ kernel(Real *outptr)
{
	/// Make some initial data as shared memory
	__shared__ Real sharedvec[SIZE];

	sharedvec[threadIdx.x] = threadIdx.x;
	__syncthreads();
	
	/// Create class on the GPU
	GPUClass instance();
	instance.vector_[threadIdx.x] = sharedvec[threadIdx.x];

	cudaMemcpy(instance.vector_, outptr, sizeof(Real) * SIZE, cudaMemcpyDeviceToHost);

	
}

Int main()
{
	Real *output;
	cudaMallocHost((void**)&output, sizeof(Real)*SIZE);
	
	/// Call the kernel in a single block and w the minimum of 32 threads
	kernel<<<1,SIZE>>>(outptr);

	for (i=0;i<SIZE;i++)
		std::cout << output[i] << std::endl;

	
}
*/
