#pragma once

#include <cmath>
#include <cassert>

#include "typedefs.h"
#include "errcheck.h"
#include "cuffterr.h"
#include "fd_params.h"
#include "input_params.h"
#include "indexing.h"
#include "fmesh.h"
#include "pbmesh.h"
#include "grid.h"
#include "timer.h"

typedef fMesh<Real,2> Mesh;
typedef bundleMesh<Real, 2> Bundle;

#include "shared.h"
#include "bundle.h"

#include "pbc_kernel.h"
#include "derivatives.h"
#include "curl.h"
