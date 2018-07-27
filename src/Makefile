INCLUDEPATH= -I .,errorcheck,dataclasses,utility,parameters,derivatives,diffops, -I/usr/local/cuda-9.2/inc -L/usr/local/cuda-9.2/lib

#This arch and sm settings should be compatible with my notebook Geforce GTX 850M card.
NVCCFLAGS= --ptxas-options=-v -std=c++14 --gpu-architecture=compute_${CC} --gpu-code=sm_${CC} -lcufft


bundletest:
	nvcc bundle_test.cu ${INCLUDEPATH} ${NVCCFLAGS} -o test.exe

vlapltest:
	nvcc vlapltest.cu ${INCLUDEPATH} ${NVCCFLAGS} -o vlaplexec.exe

lapltest:
	nvcc lapltest.cu ${INCLUDEPATH} ${NVCCFLAGS} -o laplexec.exe

gradtest:
	nvcc gradtest.cu ${INCLUDEPATH} ${NVCCFLAGS} -o exegrad.exe

poisson3doutofplace:
	nvcc poisson_3d_outofplace.cu ${INCLUDEPATH} ${NVCCFLAGS} -o pssn.exe

threed_r2c:
	nvcc 3d_r2c.cu ${INCLUDEPATH} ${NVCCFLAGS} -o r2c.exe

threedim:
	nvcc 3d_cufft.cu ${INCLUDEPATH} ${NVCCFLAGS} -o threedim.exe

twodim:
	nvcc 2d_cufft.cu ${INCLUDEPATH} ${NVCCFLAGS} -o twodim.exe
onedim:
	nvcc 1d_cufft.cu ${INCLUDEPATH} ${NVCCFLAGS} -o onedim.exe
outofplace:
	nvcc outofplace_d2z.cu ${INCLUDEPATH} ${NVCCFLAGS} -o exec_outofplace.exe
typetest:
	nvcc cufftrealcomplex_test.cu ${INCLUDEPATH} ${NVCCFLAGS} -o typetest.exe
inplace:
	nvcc inplace_d2z.cu ${INCLUDEPATH} ${NVCCFLAGS} -o d2z_inplace.exe
fft:
	nvcc fft.cu ${INCLUDEPATH} ${NVCCFLAGS} -o cufft.exe
