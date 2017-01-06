#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <png++/png.hpp>

template<typename realT>
struct judge_mandelbrot
{
    const std::size_t threshold;

    judge_mandelbrot(const std::size_t th): threshold(th){}

    __host__ __device__
    bool operator()(const thrust::complex<realT>& z) const
    {
        thrust::complex<realT> x(0., 0.);
        for(std::size_t i=0; i<threshold; ++i)
        {
            x = x * x + z;
            if(thrust::abs(x) > 2.)
                return false;
        }
        return true;
    }
};

int main()
{
    const thrust::complex<double> min(-2.0, -1.5);
    const thrust::complex<double> max( 1.0,  1.5);
    const std::size_t threshold    =  200;
    const std::size_t r_resolution = 1024;
    const std::size_t i_resolution = 1024;
    const std::size_t size = r_resolution * i_resolution;
    const double dr = (max.real() - min.real()) / r_resolution;
    const double di = (max.imag() - min.imag()) / i_resolution;
    std::cerr << "dr = " << dr << std::endl;
    std::cerr << "di = " << di << std::endl;

    // generate complex plane
    thrust::host_vector<thrust::complex<double>> host_plane(size);
    for(std::size_t i=0; i<r_resolution; ++i)
    {
        for(std::size_t j=0; j<i_resolution; ++j)
        {
            const thrust::complex<double> val =
                min + thrust::complex<double>(dr * i, di * j);
            const std::size_t offset = i + r_resolution * j;
            host_plane[offset] = val;
        }
    }
    std::cerr << "complex plane generated" << std::endl;

    thrust::device_vector<thrust::complex<double>> complex_plane = host_plane;
    std::cerr << "complex plane copied" << std::endl;

    judge_mandelbrot<double> judge(threshold);
    thrust::device_vector<bool> is_mandelbrot(size);
    thrust::transform(complex_plane.begin(), complex_plane.end(),
            is_mandelbrot.begin(), judge);
    std::cerr << "calculation end" << std::endl;

    thrust::host_vector<bool> host_is_mandelbrot = is_mandelbrot;

    png::image<png::rgb_pixel> image(r_resolution, i_resolution);
    for(std::size_t i=0; i<r_resolution; ++i)
    {
        for(std::size_t j=0; j<i_resolution; ++j)
        {
            const std::size_t offset = i + r_resolution * j;

            if(host_is_mandelbrot[offset])
                image[i][j] = png::rgb_pixel(255, 0, 0);
            else
                image[i][j] = png::rgb_pixel(0,0,0);
        }
    }
    image.write("mandelbrot.png");
    return 0;
}
