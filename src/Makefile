PARAMS=
INCLUDEPATH= -I .,errorcheck,mesh,utility,parameters,derivatives,curl

#This arch and sm settings should be compatible with my notebook Geforce GTX 850M card.
NVCCFLAGS= --ptxas-options=-v -std=c++14 --gpu-architecture=compute_${CC} --gpu-code=sm_${CC}

NVCC=nvcc

CUDACOMPILE=${NVCC} -c ${NVCCFLAGS} ${PARAMS} ${INCLUDEPATH}
CUDALINK=${NVCC} ${NVCCFLAGS} ${PARAMS} ${INCLUDEPATH}

curltest:
	${CUDALINK} curl_test.cu -o curl.exe

oop:
	${CUDALINK} oop_deriv.cu -o oop_deriv.exe

bundlercache:
	${CUDALINK} bprcache_deriv.cu -o bprcache_deriv.exe

bundle:
	${CUDALINK} bpframework_deriv.cu -o bundle_deriv.exe

rcache:
	${CUDALINK} rollcache_gpuderiv_v1.cu -o rcache.exe

clean:
	rm *.exe
