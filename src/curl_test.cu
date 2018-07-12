#include "common.h"
#include "grid.h"
#include "timer.h"

#include <iostream>

#include "bundle.h"
#include "curl_kernel.h"

__host__
void initHost(Mesh &u, const Grid &grid)
{
	const Real *x = grid.h_linspace;
	const Real *y = grid.h_linspace;
	const Real *z = grid.h_linspace;
	
	for (Int i=0;i<u.nx_;i++)
	{
		for (Int j=0;j<u.ny_;j++)
		{
			for (Int k=0;k<u.nz_;k++)
			{
				//h[fIdx(i,j,k)] = i+NX*(j+NY*k);
				u.h_data[u.indx(i,j,k,0)] = 1.0*x[i]+1.0*y[j]+1.0*z[k];
				u.h_data[u.indx(i,j,k,1)] = 2.0*x[i]+2.0*y[j]+2.0*z[k];
				u.h_data[u.indx(i,j,k,2)] = 3.0*x[i]+3.0*y[j]+3.0*z[k];
			}
		}
	}
	
}

/// Instantiate global objects
Mesh u(NX,NY,NZ,3);
Mesh du(NX,NY,NZ,3);
Grid grid(NX,NY,NZ,0.0,2*M_PI);
Timer timer;

__host__ void testCurl(Mesh &du)
{
	Real ref[3];
	ref[0] = 1;
	ref[1] = -2;
	ref[2] = 1;

	Real maxE[3];
	
	for (Int i=0;i<du.nx_;i++)
	{
		for (Int j=0;j<du.ny_;j++)
		{
			for (Int k=0;k<du.nz_;k++)
			{
				for (Int vi=0;vi<du.nvars_;vi++)
				{
					Real val = du.h_data[du.indx(i,j,k,vi)];
					Real err = fabs(val-ref[vi]);
					if (err > maxE[vi])
						maxE[vi] = err;
				}
			}
		}
	}

	std::cout << "x max error: " << maxE[0] << std::endl;
	std::cout << "y max error: " << maxE[1] << std::endl;
	std::cout << "z max error: " << maxE[2] << std::endl;
}


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
	
	curlKernel<<<blx,tpb>>>(u,du,grid);
	
	timer.recordStop();
	timer.synch();

	du.copyFromDevice();

	du.print();
	
	testCurl(du);
	
	timer.print();
	

	

	return 0;
};

     
