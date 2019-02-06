#include "ising.cuh"
#include <cmath>

// pre-calculated exp(dE / (kB * T))
__constant__ float exp_dE_beta[10];

void set_expdE(const float beta, const float J, const float H, const bool verbose) noexcept
{
    // up == true, down == false;
    // cash exp(dE) to constant memory
    const float exp_dE[10] = {         // case {neighbors}, center
        std::exp(beta * ( 4*J + 2*H)), // {up, up, up, up}, down
        std::exp(beta * ( 4*J - 2*H)), // {dn, dn, dn, dn}, up
        std::exp(beta * ( 2*J + 2*H)), // {up, up, up, dn}, down
        std::exp(beta * ( 2*J - 2*H)), // {dn, dn, dn, up}, up
        std::exp(beta * ( 0*J + 2*H)), // {up, up, dn, dn}, down
        std::exp(beta * ( 0*J - 2*H)), // {dn, dn, up, up}, up
        std::exp(beta * (-2*J + 2*H)), // {up, dn, dn, dn}, down
        std::exp(beta * (-2*J - 2*H)), // {dn, up, up, up}, up
        std::exp(beta * (-4*J + 2*H)), // {dn, dn, dn, dn}, down
        std::exp(beta * (-4*J - 2*H))  // {up, up, up, up}, up
    };
    const cudaError_t err_dE =
        cudaMemcpyToSymbol(exp_dE_beta, exp_dE, sizeof(float) * 10);
    assert(err_dE == 0);

    if(verbose)
    {
        std::cerr << "Info: precalculated exp(dE) are copied to constant memory"
                  << std::endl;
    }
    return;
}

__global__
void update_field_kernel(
        const thrust::device_ptr<bool>        spins,
        const thrust::device_ptr<const float> random,
        const std::size_t x_size, const std::size_t y_size,
        bool turn)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    // checkerboard pattern
    if     ( turn && ((x+y)%2 == 1)) {return;}
    else if(!turn && ((x+y)%2 == 0)) {return;}

    const std::size_t xdim = blockDim.x * gridDim.x;
    const std::size_t offset = x + y * xdim;
    if(offset >= x_size * y_size)
    {
        return;
    }

    const std::size_t n_offset = (y+1 < y_size) ? x + (y+1) * xdim : x;
    const std::size_t e_offset = (x+1 < x_size) ? (x+1) + y * xdim : y * xdim;
    const std::size_t s_offset = (y-1 >= 0) ? x + (y-1) * xdim : x + (y_size-1) * xdim;
    const std::size_t w_offset = (x-1 >= 0) ? (x-1) + y * xdim : x_size - 1 + y * xdim;

    const bool c = spins[  offset]; // center
    const bool n = spins[n_offset]; // north
    const bool e = spins[e_offset]; // east
    const bool s = spins[s_offset]; // south
    const bool w = spins[w_offset]; // west

    std::size_t dJ = 0;
    if(c == n) {++dJ;}
    if(c == e) {++dJ;}
    if(c == s) {++dJ;}
    if(c == w) {++dJ;}
    const std::size_t dH = c ? 1 : 0;

    if(exp_dE_beta[dH + dJ * 2] > random[offset])
    {
        spins[offset] = (!c);
    }
    return;
}

void update_field(const dim3 blocks, const dim3 threads,
                  const thrust::device_ptr<bool>        spins,
                  const thrust::device_ptr<const float> random,
                  const std::size_t x_size, const std::size_t y_size)
{
    update_field_kernel<<<blocks, threads>>>(
            spins, random, x_size, y_size, true);
    update_field_kernel<<<blocks, threads>>>(
            spins, random, x_size, y_size, false);
}

__global__
void initialize_field_kernel(
        const thrust::device_ptr<bool>        spins,
        const thrust::device_ptr<const float> random,
        const std::size_t x_size, const std::size_t y_size,
        const float threshold)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    const std::size_t xdim = blockDim.x * gridDim.x;
    const std::size_t offset = x + y * xdim;
    if(offset >= x_size * y_size)
    {
        return;
    }

    spins.get()[offset] = random.get()[offset] > threshold;

    return;
}

void initialize_field(const dim3 blocks, const dim3 threads,
                      const thrust::device_ptr<bool>        spins,
                      const thrust::device_ptr<const float> random,
                      const std::size_t x_size, const std::size_t y_size,
                      const float threshold)
{
    initialize_field_kernel<<<blocks, threads>>>(
            spins, random, x_size, y_size, threshold);
    return;
}



