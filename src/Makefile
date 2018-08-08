INCLUDEPATH= -I .,errorcheck,dataclasses,utility,parameters,derivatives,diffops, -I/usr/local/cuda-9.2/inc -L/usr/local/cuda-9.2/lib

#This arch and sm settings should be compatible with my notebook Geforce GTX 850M card.
NVCCFLAGS= --ptxas-options=-v -std=c++14 --gpu-architecture=compute_${CC} --gpu-code=sm_${CC} -lcufft

shuff:
	nvcc shuffle_max.cu ${INCLUDEPATH} ${NVCCFLAGS} -o shuff.exe

reduct:
	nvcc reductiontest.cu ${INCLUDEPATH} ${NVCCFLAGS} -o reduct.exe

kflow:
	nvcc kolmogorov_flow.cu ${INCLUDEPATH} ${NVCCFLAGS} -o kflow.exe
