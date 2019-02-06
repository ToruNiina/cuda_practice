#ifndef CUDA_PRACTICE_ISING_CUH
#define CUDA_PRACTICE_ISING_CUH
#include <cstdlib>
#include <thrust/device_ptr.h>

void set_expdE(const float beta, const float J, const float H,
               const bool verbose = true) noexcept;

void update_field(const dim3 blocks, const dim3 threads,
                  const thrust::device_ptr<bool>        spins,
                  const thrust::device_ptr<const float> random,
                  const std::size_t x_size, const std::size_t y_size);

void initialize_field(const dim3 blocks, const dim3 threads,
                      const thrust::device_ptr<bool>        spins,
                      const thrust::device_ptr<const float> random,
                      const std::size_t x_size, const std::size_t y_size,
                      const float threshold);


#endif // CUDA_PRACTICE_ISING_CUH
