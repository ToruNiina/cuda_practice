#ifndef CUDA_PRACTICE_ISING_RENDER_CUH
#define CUDA_PRACTICE_ISING_RENDER_CUH
#include <cstdlib>
#include <thrust/device_ptr.h>

void render_field(const dim3 blocks, const dim3 threads, cudaStream_t stream,
                  thrust::device_ptr<bool> spins,
                  cudaArray_const_t array,
                  const std::size_t x_size,
                  const std::size_t y_size);

#endif // CUDA_PRACTICE_ISING_RENDER_CUH
