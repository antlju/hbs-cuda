#pragma once

#define NG 2
#define NGHOSTS NG
#define NVARS 1
#define BUNDLESIZE3 NVARS*(4*NG+1)*(NX_TILE+2*NG)
#define BUNDLESIZE1 (4*NG+1)*(NX_TILE+2*NG)
