PARAMS=

#This arch and sm settings should be compatible with my notebook Geforce GTX 850M card.
NVCCFLAGS= #-std=c++14 --gpu-architecture=compute_50 --gpu-code=sm_61

NVCC=nvcc

CUDACOMPILE=${NVCC} -c ${NVCCFLAGS} ${PARAMS}
CUDALINK=${NVCC} ${NVCCFLAGS} ${PARAMS}

HEADERS=mesh/fmesh.h


test:
	${CUDALINK} testing_class.cu -o test.out

clean:
	rm *.out
