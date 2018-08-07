#pragma once

/* -----------------------------
These functions copy the finite difference mesh psi (no ghost points) to fftReal array and
solves the Poisson equation and then copy it back to Psi.

 --------------------------------------- */

/// ---------------------------------------
/// Utility functions
__host__ __device__
inline size_t cindx(size_t i, size_t j, size_t k)
{
	return k+((NZ >> 1)+1)*(j+NY*i); /// (NZ >> 1) is a bitshift expression equiv to NZ/2
}

__host__ __device__
inline size_t rindx(size_t i, size_t j, size_t k)
{
	return k+NZ*(j+NY*i); /// (NZ >> 1) is a bitshift expression equiv to NZ/2
}

/// This copy function assumes mesh to be scalar field
__global__
void copy_mesh2fft_kernel(Real *fft_array, Mesh mesh)
{
	const Int k = threadIdx.y+blockIdx.y*blockDim.y;
	const Int j = threadIdx.x+blockIdx.x*blockDim.x;
	if (j < mesh.ny_ && k < mesh.nz_)
	{
		for (Int i=0;i<mesh.nx_;i++)
		{
			fft_array[rindx(i,j,k)] = mesh(i,j,k,0);
		}
	}
}

__global__
void copy_fft2mesh_and_normalise_kernel(Real *fft_array, Mesh mesh)
{
	const Int k = threadIdx.y+blockIdx.y*blockDim.y;
	const Int j = threadIdx.x+blockIdx.x*blockDim.x;
	const Real size = mesh.nx_*mesh.ny_*mesh.nz_;
	if (j < mesh.ny_ && k < mesh.nz_)
	{
		for (Int i=0;i<mesh.nx_;i++)
		{
			mesh(i,j,k,0) = fft_array[rindx(i,j,k)]/size;
		}
	}
}

/// ---------------------------------------

__global__
void freqDiv_kernel(Complex *f, const Real xlen)
{
	Int II,JJ; 
	Int j = threadIdx.y+blockIdx.y*blockDim.y;
	Int i = threadIdx.x+blockIdx.x*blockDim.x;
        Real k1,k2,k3,Pi=M_PI,fac;
	
	//if (i < NX && j < NY && k < ((NZ >> 1) + 1))
	if (i < NX && j < NY)
	{
		if (2*i<NX)
			II = i;
		else
                        II = NX-i;

		if (2*j<NY)
			JJ = j;
		else
			JJ = NX-j;
		
		k2 = 2*Pi*JJ/xlen;
                k1 = 2*Pi*II/xlen;
		for (Int k=0;k<((NZ >> 1)+1);k++)
		{
			k3 = 2*Pi*k/xlen;
			fac = -1.0*(k1*k1+k2*k2+k3*k3);

			if (fabs(fac) < 1e-14)
			{
				f[cindx(i,j,k)].x = 0.0;
				f[cindx(i,j,k)].y = 0.0;	
			}
			else
			{
				f[cindx(i,j,k)].x = f[cindx(i,j,k)].x/fac;
				f[cindx(i,j,k)].y = f[cindx(i,j,k)].y/fac;
			}
		}
	}
}

__host__
void Poisson_FFT_solver(Mesh psi, Complex *fftcomplex, Real *fftreal, const Real xlen, const Int k_rk,cufftHandle pland2z, cufftHandle planz2d)
{
	dim3 tpb(NY_TILE,NZ_TILE); 
	dim3 blx(NN/NY_TILE,NN/NZ_TILE);
	
	copy_mesh2fft_kernel<<<blx,tpb>>>(fftreal,psi);

	///--------------
	///std::cout << "\n Executing forward R2C transform... \n\n";
	if (cufftExecD2Z(pland2z,fftreal,fftcomplex) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: ExecD2Z Forward failed");	
	}

	///--------------
	freqDiv_kernel<<<blx,tpb>>>(fftcomplex,xlen);

	///--------------
	///std::cout << "\n Executing backward C2R transform... \n\n";
	if (cufftExecZ2D(planz2d,fftcomplex,fftreal) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: ExecZ2D backward failed");	
	}

	copy_fft2mesh_and_normalise_kernel<<<blx,tpb>>>(fftreal,psi);
}


