#pragma once

/// Global instantiation of data classes.
Mesh uu(NX,NY,NZ,3); /// Velocity vector field.
Mesh uStar(NX,NY,NZ,3); /// u*, step velocity vector field.
Mesh RHSk(NX,NY,NZ,3); /// RHS^k Runge-Kutta substep vector field
Mesh RHSk_1(NX,NY,NZ,3); /// RHS^(k-1) Runge-Kutta substep vector field
Mesh pp(NX,NY,NZ,1); /// Pressure scalar field
Mesh psi(NX,NY,NZ,1); /// \Psi scalar field
Mesh gradPsi(NX,NY,NZ,3); /// \grad{\Psi} vector field.
Mesh verify(NX,NY,NZ,3); /// Vector field to store analytic solution for verification.

Complex *fftComplex;
Real *fftReal;

/// Allocate device memory.
uu.allocateDevice();
uStar.allocateDevice();
RHSk.allocateDevice();
RHSk_1.allocateDevice();
pp.allocateDevice();
psi.allocateDevice();
gradPsi.allocateDevice();
verify.allocateDevice();

cudaCheck(cudaMalloc((void**)&fftReal,sizeof(Real)*NX*NY*NZ));
cudaCheck(cudaMalloc((void**)&fftComplex,sizeof(Complex)*NX*NY*(NZ/2+1)));
