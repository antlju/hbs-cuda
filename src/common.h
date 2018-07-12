#pragma once

#include <cmath>
#include <cassert>

#include "typedefs.h"
#include "errcheck.h"
#include "fd_params.h"
#include "input_params.h"
#include "indexing.h"
#include "fmesh.h"
#include "pbmesh.h"

typedef fMesh<Real,NG> Mesh;
typedef bundleMesh<Real, NG> Bundle;
#include "pbc_kernel.h"

/* Includes before OOP
#include "typedefs.h"
#include "errcheck.h"
#include "fd_params.h"
#include "input_params.h"
#include "indexing.h"
#include "pbc_kernel.h"
#include "derivatives.h"
#include "bundleload.h"
*/
