#include "common.h"
#include "grid.h"
#include "timer.h"

#include <iostream>


#include "deriv_kernel.h"

__host__
void initHost(Mesh &u, const Grid &grid)
{
	const Real *x = grid.h_linspace;
	
	for (Int i=0;i<u.nx_;i++)
	{
		for (Int j=0;j<u.ny_;j++)
		{
			for (Int k=0;k<u.nz_;k++)
			{
				//h[fIdx(i,j,k)] = i+NX*(j+NY*k);
				u.h_data[u.indx(i,j,k)] = sin(x[k]);
			}
		}
	}
}

__host__
Real maxErrCosz(Mesh &u, const Grid &grid)
{
	const Real *z = grid.h_linspace;
	Real maxE = 0;
	for (size_t vi=0;vi<u.nvars_;vi++)
	{
		for (size_t i=0;i<u.nx_;i++)
		{
			for (size_t j=0;j<u.ny_;j++)
			{
				for (size_t k=0;k<u.nz_;k++)
				{
					Real val = fabs(u.h_data[u.indx(i,j,k,vi)]-cos(z[k]));
					if (val > maxE)
						maxE = val;
				}
			}
		}
	}
	return maxE;
}

/// Instantiate global objects

Mesh u(NX,NY,NZ,1);
Mesh du(NX,NY,NZ,1);
Grid grid(NX,NY,NZ,0.0,2*M_PI);
Timer timer;



Int main()
{
	std::cout << "Executing w/ size: (N=" << NN << ")^3" << std::endl;
	u.allocateHost(); u.allocateDevice();
	du.allocateHost(); du.allocateDevice();

	grid.setHostLinspace();
	initHost(u,grid);
	//u.print();
	
	timer.createEvents();
	u.copyToDevice();
	dim3 tpb(NY_TILE,NZ_TILE); 
	dim3 blx(NN/NY_TILE,NN/NZ_TILE);
	timer.recordStart();
 
	pbc_x_kernel<<<blx,tpb>>>(u);
	pbc_y_kernel<<<blx,tpb>>>(u);
	pbc_z_kernel<<<blx,tpb>>>(u);
	
	zderivKernel<<<blx,tpb>>>(u,du,grid.dx_);
	
	timer.recordStop();
	timer.synch();

	du.copyFromDevice();
	printf("Max error: %.10f\n", maxErrCosz(du,grid));
	timer.print();
	

	//du.print();

	return 0;
};

     
