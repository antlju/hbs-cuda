PARAMS=
INCLUDEPATH= -I .,errorcheck,mesh,utility,parameters,derivatives,bundlepencil

#This arch and sm settings should be compatible with my notebook Geforce GTX 850M card.
NVCCFLAGS= --ptxas-options=-v -std=c++14 --gpu-architecture=compute_${CC} --gpu-code=sm_${CC}

NVCC=nvcc

CUDACOMPILE=${NVCC} -c ${NVCCFLAGS} ${PARAMS} ${INCLUDEPATH}
CUDALINK=${NVCC} ${NVCCFLAGS} ${PARAMS} ${INCLUDEPATH}

oop:
	${CUDALINK} oop_deriv.cu -o oop_deriv.out

bundlercache:
	${CUDALINK} bprcache_deriv.cu -o bprcache_deriv.out

bundle:
	${CUDALINK} bpframework_deriv.cu -o bundle_deriv.out

rcache:
	${CUDALINK} rollcache_gpuderiv_v1.cu -o rcache.out

clean:
	rm *.out *.o
