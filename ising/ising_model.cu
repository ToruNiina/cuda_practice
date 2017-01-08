#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <random>
#include <iomanip>
#include <cmath>

#include <png++/png.hpp>
#include <curand.h>

template<typename charT>
std::map<std::string, std::string>
read_file(std::basic_istream<charT>& is)
{
    if(!is.good())
        throw std::invalid_argument("file open error");
    std::map<std::string, std::string> contents;

    while(!is.eof())
    {
        std::string line;
        std::getline(is, line);
        if(line.empty()) continue;
        std::istringstream iss(line);
        iss >> std::ws;
        if(iss.peek() == '#') continue;
        std::string key, value;
        char eq;
        iss >> key >> eq >> value;
        if(eq != '=') throw std::runtime_error("file format error");

        contents[key] = value;
    }
    return contents;
}

std::size_t digit(std::size_t n)
{
    std::size_t dig = 0;
    while(n > 0)
    {
        ++dig;
        n /= 10;
    }
    return dig;
}

// pre-calculated exp(dE / (kB * T))
__constant__ float exp_dE_beta[10];

// use texture memory as spins
// texture<bool, 2, cudaReadModeElementType> field1;
// texture<bool, 2, cudaReadModeElementType> field2;

__global__
void update_field(bool* spins, const float* random,
        const std::size_t x_size, const std::size_t y_size, bool turn)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(turn)
    {
        if((x+y)%2 == 1) return;
    }
    else
    {
        if((x+y)%2 == 0) return;
    }

    const std::size_t xdim = blockDim.x * gridDim.x;
    const std::size_t offset = x + y * xdim;
    if(offset >= x_size * y_size) return;

    const std::size_t n_offset = (y+1 < y_size) ? x + (y+1) * xdim : x;
    const std::size_t e_offset = (x+1 < x_size) ? (x+1) + y * xdim : y * xdim;
    const std::size_t s_offset = (y-1 >= 0) ? x + (y-1) * xdim : x + (y_size-1) * xdim;
    const std::size_t w_offset = (x-1 >= 0) ? (x-1) + y * xdim : x_size - 1 + y * xdim;

    const bool c = spins[offset];   // center
    const bool n = spins[n_offset]; // north
    const bool e = spins[e_offset]; // east
    const bool s = spins[s_offset]; // south
    const bool w = spins[w_offset]; // west

    std::size_t dJ = 0;
    if(c == n) ++dJ;
    if(c == e) ++dJ;
    if(c == s) ++dJ;
    if(c == w) ++dJ;
    const std::size_t dH = c ? 1 : 0;

    if(exp_dE_beta[dH + dJ * 2] > random[offset])
        spins[offset] = (!c);
    return;
}


int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: ./ising <input.dat>" << std::endl;
        std::cerr << "input: width  = <int>"      << std::endl;
        std::cerr << "     : height = <int>"      << std::endl;
        std::cerr << "     : steps  = <int>"      << std::endl;
        std::cerr << "     : seed   = <int>"      << std::endl;
        std::cerr << "     : J      = <float>"     << std::endl;
        std::cerr << "     : H      = <float>"     << std::endl;
        std::cerr << "     : T      = <float>"     << std::endl;
        std::cerr << "     : kB     = <float>"     << std::endl;
        return 1;
    }

    std::ifstream ifs(argv[1]);
    if(!ifs.good())
    {
        std::cerr << "file output error: " << argv[1] << std::endl;
        return 1;
    }
    const std::map<std::string, std::string> contents = read_file(ifs);

    const std::size_t   w    = std::stoul(contents.at("width"));
    const std::size_t   h    = std::stoul(contents.at("height"));
    const std::size_t   step = std::stoul(contents.at("steps"));
    const std::uint64_t seed = std::stoul(contents.at("seed"));
    const float         J    = std::stof(contents.at("J"));
    const float         H    = std::stof(contents.at("H"));
    const float         T    = std::stof(contents.at("T"));
    const float         kB   = std::stof(contents.at("kB"));
    const float         beta = 1. / (kB * T);
    std::cerr << "input file read" << std::endl;

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
    std::cerr << "precalculated exp(dE) are copied to constant memory" << std::endl;

    // allocate global memory to store spins and random numbers
    bool *spins;
    cudaError_t err_spins = cudaMalloc((void**)&spins, sizeof(bool) * w * h);
    assert(err_spins == 0);

    float *random;
    cudaError_t err_random = cudaMalloc((void**)&random, sizeof(float) * w * h);
    assert(err_random == 0);
    std::cerr << "device memory for spins and randoms are allocated" << std::endl;

    // prepair cuRAND generators
    curandGenerator_t rng;
    curandStatus_t st_gen = curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    assert(st_gen == CURAND_STATUS_SUCCESS);
    curandStatus_t st_seed = curandSetPseudoRandomGeneratorSeed(rng, seed);
    assert(st_seed == CURAND_STATUS_SUCCESS);
    std::cerr << "cuRAND generator created" << std::endl;

    // set initial configuration as random boolean
    bool *snapshot = new bool[w * h];
    std::cerr << "host memory for snapshot allocated" << std::endl;

    std::mt19937 mt(seed);
    std::bernoulli_distribution distro(0.5);
    for(std::size_t i=0; i < w * h; ++i)
        snapshot[i] = distro(mt);
    std::cerr << "initial snapshot created" << std::endl;

    cudaError_t cpy_init =
        cudaMemcpy(spins, snapshot, sizeof(bool)*w*h, cudaMemcpyHostToDevice);
    assert(cpy_init == 0);
    std::cerr << "initial state copied" << std::endl;

#if defined(OUTPUT_TEXT)
    std::ofstream ofs("ising_traj.dat");
    char *traj = new char[w * h];
#endif

    for(std::size_t i=0; i<step; ++i)
    {
#ifdef OUTPUT_PNG
        // copy snapshot
        cudaError_t ercpy = cudaMemcpy(
                snapshot, spins, sizeof(bool) * w * h, cudaMemcpyDeviceToHost);
        assert(ercpy == 0);
 
        // write out
        std::ostringstream filename;
        filename << "ising" << std::setfill('0') << std::setw(digit(step))
                 << i << ".png";
        png::image<png::rgb_pixel> image(w, h);
        for(std::size_t i=0; i<w; ++i)
        {
            for(std::size_t j=0; j<h; ++j)
            {
                std::size_t offset = i + w * j;
                if(snapshot[offset])
                    image[i][j] = png::rgb_pixel(255, 255, 255);
                else
                    image[i][j] = png::rgb_pixel(0,0,0);
            }
        }
        image.write(filename.str().c_str());

#elif defined(OUTPUT_TEXT)

        cudaError_t ercpy = cudaMemcpy(
                snapshot, spins, sizeof(bool) * w * h, cudaMemcpyDeviceToHost);
        assert(ercpy == 0);

        for(std::size_t i=0; i<w*h; ++i)
            traj[i] = static_cast<char>(snapshot[i]) + 48;
        ofs << traj << std::endl;

#endif //OUTPUT

        // generate random numbers
        curandStatus_t st_genrnd = curandGenerateUniform(rng, random, w * h);
        assert(st_genrnd == CURAND_STATUS_SUCCESS);

        // update spins
        dim3 blocks(w/32, h/32);
        dim3 threads(32, 32);
        update_field<<<blocks, threads>>>(spins, random, w, h, true);
        update_field<<<blocks, threads>>>(spins, random, w, h, false);
    }

    curandStatus_t destroy = curandDestroyGenerator(rng);
    assert(destroy == CURAND_STATUS_SUCCESS);
    cudaError_t free_spin = cudaFree(spins);
    assert(free_spin == 0);
    cudaError_t free_random = cudaFree(random);
    assert(free_random == 0);

    return 0;
}
