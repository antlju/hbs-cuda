/// This code is taken from an example of nVIDIA's Mark Harris.
/// stdio.h and assert.h are from 'regular C' and could well be replaced.

#pragma once
#include <stdio.h>
#include <assert.h>

/// Wrapper function for error checking of CUDA API calls.
/// Any CUDA API call returns a value of type cudaError_t.
inline cudaError_t cudaCheck(cudaError_t result)
{
	if (result != cudaSuccess) /// Check if call return the 'cudaSucces' value
	{
		/// Print error string of result to stderr stream
		fprintf(stderr, "CUDA Runtime error: %s\n", cudaGetErrorString(result));
		/// cudaGetErrorString() should set result to cudaSuccess
		assert(result == cudaSuccess);
	}
	return result;
}
