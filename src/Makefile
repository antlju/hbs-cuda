PARAMS=
INCLUDEPATH= -I errorcheck,mesh,utility,parameters

#This arch and sm settings should be compatible with my notebook Geforce GTX 850M card.
NVCCFLAGS= --ptxas-options=-v -std=c++14 --gpu-architecture=compute_50 --gpu-code=sm_50

NVCC=nvcc

CUDACOMPILE=${NVCC} -c ${NVCCFLAGS} ${PARAMS} ${INCLUDEPATH}
CUDALINK=${NVCC} ${NVCCFLAGS} ${PARAMS} ${INCLUDEPATH}

rollcachederiv:
	${CUDALINK} rollcache_gpuderiv_v1.cu -o rcache_gpuderiv.out

bundlegpu:
	${CUDALINK}  bundle_gpuderiv_v2.cu -o bundlegpu.out 

smem_test:
	${CUDALINK}  rolling_cache_v1.cu -o smem_test.out

dquery:
	${CUDALINK} devicequery.cu -o dquery.out

clean:
	rm *.out
