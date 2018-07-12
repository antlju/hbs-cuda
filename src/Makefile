PARAMS=
INCLUDEPATH= -I .,errorcheck,mesh,utility,parameters,derivatives,curl

#This arch and sm settings should be compatible with my notebook Geforce GTX 850M card.
NVCCFLAGS= --ptxas-options=-v -std=c++14 --gpu-architecture=compute_${CC} --gpu-code=sm_${CC}

NVCC=nvcc

CUDACOMPILE=${NVCC} -c ${NVCCFLAGS} ${PARAMS} ${INCLUDEPATH}
CUDALINK=${NVCC} ${NVCCFLAGS} ${PARAMS} ${INCLUDEPATH}

smemclass:
	${CUDALINK} smemclass.cu -o outsmemclass.exe
dquery:
	${CUDALINK} devicequery.cu -o devquery.exe

clean:
	rm *.exe
