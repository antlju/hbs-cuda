PARAMS=
INCLUDEPATH= -I .,errorcheck,mesh,utility,parameters,derivatives,bundlepencil

#This arch and sm settings should be compatible with my notebook Geforce GTX 850M card.
CFLAGS=-std=c++14 -O3
NVCCFLAGS= --ptxas-options=-v ${CFLAGS} --gpu-architecture=compute_${CC} --gpu-code=sm_${CC}

HEADERS=*.h
CUSRC=*.cu derivatives/*.cu
CUOBJ=$(CUSRC:.cu=.o)
NVCC=nvcc

CUDACOMPILE=$(NVCC) $(NVCCFLAGS) $(PARAMS) $(INCLUDEPATH) -c
CUDALINK=$(NVCC) $(NVCCFLAGS) $(PARAMS) $(INCLUDEPATH)

EXEC=test.exe

all: $(CUSRC) $(EXEC)

$(EXEC): $(CUOBJ) $(HEADERS)
	$(CUDALINK) $(CUOBJ) -o $@

%.o: $(CUSRC)
	$(CUDACOMPILE) $< -o $@
