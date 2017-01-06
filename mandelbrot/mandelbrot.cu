#include <png++/png.hpp>

template<typename realT>
struct Complex
{
    realT r, i;
};

template<typename realT>
__device__ __forceinline__
Complex<realT> add(const Complex<realT> a, const Complex<realT> b)
{
    return Complex<realT>{a.r + b.r, a.i + b.i};
}

template<typename realT>
__device__ __forceinline__
Complex<realT> mul(const Complex<realT> a, const Complex<realT> b)
{
    return Complex<realT>{a.r * b.r - a.i * b.i, a.r * b.i + b.r * a.i};
}

template<typename realT>
__device__ __forceinline__ 
Complex<realT> mandelbrot_step(const Complex<realT> z, const Complex<realT> c)
{
    return add(mul(z, z), c);
}

template<typename realT>
__device__ __forceinline__
realT abs(const Complex<realT> z)
{
    return z.r * z.r + z.i * z.i;
}

template<typename realT>
__device__ bool
is_mandelbrot(const Complex<realT> z, const std::size_t threshold)
{
    Complex<realT> x{0., 0.};
    for(std::size_t i=0; i<threshold; ++i)
    {
        x = mandelbrot_step(x, z);
        if(abs(x) > 2.)
            return false;
    }
    return true;
}

template<typename realT>
__global__ void
write_mandelbrot(
        bool* result,
        const realT rmin, const realT rmax, const realT imin, const realT imax,
        const std::size_t th, const std::size_t x_dim, const std::size_t y_dim)
{
    const std::size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const std::size_t y = threadIdx.y + blockIdx.y * blockDim.y;

    const std::size_t offset = x + gridDim.x * blockDim.x * y;
    if(offset > x_dim * y_dim) return;

    const realT r = rmin + (rmax - rmin) / x_dim * x;
    const realT i = imin + (imax - imin) / y_dim * y;
    const Complex<realT> z{r, i};

    result[offset] = is_mandelbrot(z, th);
    return;
}


int main()
{
    const double rmin = -2.0;
    const double rmax =  1.0;
    const double imin = -1.5;
    const double imax =  1.5;
    const std::size_t threshold = 400;
    const std::size_t x_dim = 1024;
    const std::size_t y_dim = 1024;

    bool* mandelbrot;
    cudaError_t ermalloc = cudaMalloc((void**)&mandelbrot, x_dim * y_dim);
    assert(ermalloc == 0);

    dim3 blocks(x_dim/32, y_dim/32);
    dim3 threads(32, 32);

    write_mandelbrot<double><<<blocks, threads>>>(
            mandelbrot, rmin, rmax, imin, imax, threshold, x_dim, y_dim);

    bool *result = new bool[x_dim * y_dim];
    cudaError_t ercpy = cudaMemcpy(
            result, mandelbrot, x_dim * y_dim, cudaMemcpyDeviceToHost);
    assert(ercpy == 0);

    cudaFree(mandelbrot);

    png::image<png::rgb_pixel> image(x_dim, y_dim);
    for(std::size_t i=0; i<x_dim; ++i)
    {
        for(std::size_t j=0; j<y_dim; ++j)
        {
            std::size_t offset = i + x_dim * j;
            if(result[offset])
                image[i][j] = png::rgb_pixel(255, 0, 0);
            else
                image[i][j] = png::rgb_pixel(0,0,0);
        }
    }

    image.write("mandelbrot.png");
    return 0;
}
