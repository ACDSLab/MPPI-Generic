#pragma once
// HIP build shim for <cooperative_groups.h>. This directory is only on the
// include path for the HIP build (USE_HIP), so this is reached only under HIP.
// The library merely includes the header; its cg:: alias is otherwise unused.
#include <hip/hip_cooperative_groups.h>
