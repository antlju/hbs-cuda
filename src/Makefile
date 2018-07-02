objects = bprcache_deriv.o derivatives/derivatives.o

all: $(objects)
	nvcc -arch=sm_${CC} $(objects) -o test.exe

%.o: %.cu
	nvcc -arch=sm_${CC} -I.,errorcheck,mesh,utility,parameters,derivatives,bundlepencil -dc $< -o $@

clean:
	rm -f *.o *.exe
