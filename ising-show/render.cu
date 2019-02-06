#include "render.cuh"

// surface reference object that should be in global (file) scope
surface<void, cudaSurfaceType2D> surf_ref;

__global__
void render_kernel(thrust::device_ptr<bool> spins,
                   const std::size_t x_size, const std::size_t y_size)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const std::size_t xdim = blockDim.x * gridDim.x;
    const std::size_t offset = x + y * xdim;
    if(offset >= x_size * y_size)
    {
        return;
    }

    const unsigned char color = spins[offset] ? 0xFF : 0x00;

    uchar4 pixel;
    pixel.x = color;
    pixel.y = color;
    pixel.z = color;
    pixel.w = 0xFF;

    surf2Dwrite(pixel, surf_ref, x * sizeof(uchar4), y, cudaBoundaryModeZero);
    return;
}

void render_field(const dim3 blocks, const dim3 threads, cudaStream_t stream,
                  thrust::device_ptr<bool> spins,
                  cudaArray_const_t array,
                  const std::size_t x_size,
                  const std::size_t y_size)
{
    const auto bindsurf = cudaBindSurfaceToArray(surf_ref, array);
    assert(bindsurf == 0);

    render_kernel<<<blocks, threads, 0, stream>>>(spins, x_size, y_size);
    return;
}
