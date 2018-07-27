__global__ void curlKernel(Mesh f, Mesh df, Grid grid)
{
	__shared__ Real smem[3*(NY_TILE+2*NG)*(NZ_TILE+2*NG)];

	Shared fs(smem,NY_TILE,NZ_TILE,3,NG); /// Shared memory object for indexing

	const Real invdx = 1.0/grid.dx_;
	const Int ng = f.ng_;
	/// Global indices
	const Int j = threadIdx.x + blockIdx.x*blockDim.x;
	const Int k = threadIdx.y + blockIdx.y*blockDim.y;
	
	/// Local indices	
	const Int lj = threadIdx.x;
	const Int lk = threadIdx.y;
	const Int li = 0; /// the "center" of the bundle (fd stencil) in any "roll step".
	                  /// This will always be zero for any
	                  /// global index i along the array.

	/// Bundle memory and Bundle pointer to that memory
	Real vB[3*(4*NG+1)*(1+2*NG)];
	//Real sB[(4*NG+1)*(1+2*NG)];
	Bundle Bndl(&vB[0],4*NG+1,3);
	Real P[3]; /// Local vector "pencil"
	
	/// Initialise for rolling cache
	for (Int vi=0;vi<f.nvars_;vi++)
	{
		bundleInit(Bndl,f,j,k,vi);
	}
	__syncthreads();

	const Int vi = 0;
	 
	if (j < f.ny_ && k < f.nz_)
	{
		for (Int i=0;i<f.nx_;i++)
		{
			///Load shared memory and ghostpts
			loadShared(fs,f,
				   i,j,k,
				   lj,lk); //loadShared() def'd in shared.h
			//fs(lk,lj,vi) = f(i+2,j,k);
			__syncthreads();
			
			/// *** ___ Roll the cache ! ___ ***
			rollBundleCache(Bndl,fs,lj,lk);

			/// Do operations on bundle:
			curl(Bndl,P,li,invdx,invdx,invdx);

			// Set pencil
			df(i,j,k,0) = P[0]; df(i,j,k,1) = P[1]; df(i,j,k,2) = P[2];
			//df(i,j,k,0) = delz(Bndl,invdx,li,0);
			       
		}//End for loop over i.
		
	} //End j,k if statement
	
	
}
