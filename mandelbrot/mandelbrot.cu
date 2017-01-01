#include <png++/png.hpp>

template<typename realT>
struct Complex
{
    realT r, i;
};

template<typename realT>
__device__ Complex<realT> add(const Complex<realT> a, const Complex<realT> b)
{
    return Complex<realT>{a.r + b.r, a.i + b.i};
}

template<typename realT>
__device__ Complex<realT> mul(const Complex<realT> a, const Complex<realT> b)
{
    return Complex<realT>{a.r * b.r - a.i * b.i, a.r * b.i + b.r * a.i};
}

template<typename realT>
__device__ Complex<realT>
mandelbrot_step(const Complex<realT> z, const Complex<realT> c)
{
    return add(mul(z, z), c);
}

template<typename realT>
__device__ realT abs(const Complex<realT> z)
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
        const std::size_t th)
{
    const int x = blockIdx.x;
    const int y = blockIdx.y;
    const int offset = x + gridDim.x * y;

    const realT r = rmin + (rmax - rmin) / 1000 * x;
    const realT i = imin + (imax - imin) / 1000 * y;
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
    const std::size_t threshold = 200;

    bool* mandelbrot;
    cudaError_t ermalloc = cudaMalloc((void**)&mandelbrot, 1000 * 1000);
    assert(ermalloc == 0);

    dim3 grid(1000, 1000);
    write_mandelbrot<double><<<grid, 1>>>(
            mandelbrot, rmin, rmax, imin, imax, threshold);

    bool result[1000 * 1000];
    cudaError_t ercpy = cudaMemcpy(
            result, mandelbrot, 1000 * 1000, cudaMemcpyDeviceToHost);
    assert(ercpy == 0);

    cudaFree(mandelbrot);

    png::image<png::rgb_pixel> image(1000, 1000);
    for(std::size_t i=0; i<1000; ++i)
    {
        for(std::size_t j=0; j<1000; ++j)
        {
            std::size_t offset = i + 1000 * j;
            if(result[offset])
                image[i][j] = png::rgb_pixel(255, 0, 0);
            else
                image[i][j] = png::rgb_pixel(0,0,0);
        }
    }

    image.write("mandelbrot.png");
    return 0;
}
