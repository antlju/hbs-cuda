PARAMS=
NVCCFLAGS=-std=c++14
NVCC=nvcc

CUDACOMPILE=${NVCC} -c ${NVCCFLAGS} ${PARAMS}
CUDALINK=${NVCC} ${NVCCFLAGS} ${PARAMS}

HEADERS=mesh/fmesh.h


test:
	${CUDALINK} test.cu -o test.out
