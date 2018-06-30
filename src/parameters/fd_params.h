#pragma once

#define NN 256
#define NX NN
#define NY NN
#define NZ NN
#define NG 2
#define NGHOSTS NG
#define NY_TILE 1
#define NZ_TILE 1
#define NX_TILE NN
#define NVARS 1
#define BUNDLESIZE NVARS*(4*NG+1)*(NX_TILE+2*NG)
