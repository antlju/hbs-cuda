PARAMS=
INCLUDEPATH= -I errorcheck,mesh,utility,parameters

#This arch and sm settings should be compatible with my notebook Geforce GTX 850M card.
NVCCFLAGS= --ptxas-options=-v -std=c++14 --gpu-architecture=compute_50 --gpu-code=sm_50

NVCC=nvcc

CUDACOMPILE=${NVCC} -c ${NVCCFLAGS} ${PARAMS} ${INCLUDEPATH}
CUDALINK=${NVCC} ${NVCCFLAGS} ${PARAMS} ${INCLUDEPATH}


smem_test:
	${CUDALINK}  shared_mem_test_v2.cu -o smem_test.out

dquery:
	${CUDALINK} devicequery.cu -o dquery.out

clean:
	rm *.out
