#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <png++/png.hpp>

template<typename realT>
struct mandelbrot
{
    typedef realT real_type;
    typedef thrust::complex<real_type> complex_type;

    mandelbrot(const std::size_t th,
               const std::size_t Nr, const std::size_t Ni,
               const real_type dr, const real_type di,
               const complex_type& min)
        : threshold_(th), num_r_(Nr), num_i_(Ni), dr_(dr), di_(di), min_(min)
    {}

    __host__ __device__
    bool operator()(const std::size_t& idx) const
    {
        const std::size_t r_idx = idx / num_i_;
        const std::size_t i_idx = idx % num_r_;
        const complex_type z(min_ + complex_type(r_idx * dr_, i_idx * di_));

        thrust::complex<realT> x(0., 0.);
        for(std::size_t i=0; i<threshold_; ++i)
        {
            x = x * x + z;
            if(thrust::abs(x) > 2.)
                return false;
        }
        return true;
    }

    const std::size_t threshold_, num_r_, num_i_;
    const real_type dr_, di_;
    const complex_type min_;
};

int main()
{
    const thrust::complex<double> min(-2.0, -1.5);
    const thrust::complex<double> max( 1.0,  1.5);
    const std::size_t threshold =  200;
    const std::size_t num_r     = 1024;
    const std::size_t num_i     = 1024;
    const std::size_t size = num_r * num_i;
    const double dr = (max.real() - min.real()) / num_r;
    const double di = (max.imag() - min.imag()) / num_i;

    thrust::counting_iterator<std::size_t> begin(0);
    thrust::counting_iterator<std::size_t> end(size);
    thrust::device_vector<bool> is_mandelbrot(size);

    const mandelbrot<double> judge(threshold, num_r, num_i, dr, di, min);

    thrust::transform(begin, end, is_mandelbrot.begin(), judge);

    thrust::host_vector<bool> host_is_mandelbrot = is_mandelbrot;

    png::image<png::rgb_pixel> image(num_r, num_i);
    for(std::size_t i=0; i<num_r; ++i)
    {
        for(std::size_t j=0; j<num_i; ++j)
        {
            const std::size_t offset = i + num_r * j;

            if(host_is_mandelbrot[offset])
                image[i][j] = png::rgb_pixel(255, 0, 0);
            else
                image[i][j] = png::rgb_pixel(0,0,0);
        }
    }
    image.write("mandelbrot.png");
    return 0;
}
