#pragma once

#include <iostream>
#include <cmath>
#include <limits>
#include <cuda_runtime.h>

#define PT_INFINITY 1e30f
#define PT_PI 3.1415926535897932385f

__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * PT_PI / 180.0f;
}

__host__ __device__ inline float fclamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

#define CUDA_CHECK(val) check_cuda((val), #val, __FILE__, __LINE__)
inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}
