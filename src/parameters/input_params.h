#pragma once


#define NN 32
#define NX NN
#define NY NN
#define NZ NN


#if __ROLLING_CACHE__ == 1

#define NY_TILE 32
#define NZ_TILE 32
#define NX_TILE 1

#else

#define NY_TILE 1
#define NZ_TILE 1
#define NX_TILE NN

#endif
