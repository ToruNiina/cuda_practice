#include "ising.cuh"
#include "render.cuh"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <cuda_gl_interop.h>
#include <curand.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <iomanip>
#include <cmath>

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

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: ./ising <input.dat>"  << std::endl;
        std::cerr << "input: width  = <int>"       << std::endl;
        std::cerr << "     : height = <int>"       << std::endl;
        std::cerr << "     : steps  = <int>"       << std::endl;
        std::cerr << "     : seed   = <int>"       << std::endl;
        std::cerr << "     : J      = <float>"     << std::endl;
        std::cerr << "     : H      = <float>"     << std::endl;
        std::cerr << "     : T      = <float>"     << std::endl;
        std::cerr << "     : kB     = <float>"     << std::endl;
        return 1;
    }

    // ------------------------------------------------------------------------
    // read input file
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

    // ------------------------------------------------------------------------
    // calculate energy difference and acceptance probability
    set_expdE(beta, J, H);

    // ------------------------------------------------------------------------
    // allocate global memory to store spins and random numbers
    thrust::device_vector<bool>  spins (w * h);
    thrust::device_vector<float> random(w * h);

    std::cerr << "device memory for spins and randoms are allocated" << std::endl;

    // ------------------------------------------------------------------------
    // prepair cuRAND generators
    curandGenerator_t rng;

    curandStatus_t st_gen = curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    assert(st_gen == CURAND_STATUS_SUCCESS);

    curandStatus_t st_seed = curandSetPseudoRandomGeneratorSeed(rng, seed);
    assert(st_seed == CURAND_STATUS_SUCCESS);

    std::cerr << "cuRAND generator created" << std::endl;

    // ------------------------------------------------------------------------
    // set initial configuration as random boolean
    {
        curandStatus_t st_genrnd = curandGenerateUniform(rng,
                thrust::device_pointer_cast(random.data()).get(), w * h);
        assert(st_genrnd == CURAND_STATUS_SUCCESS);
    }
    const dim3 blocks (w/32, h/32);
    const dim3 threads(32, 32);
    initialize_field(blocks, threads,
                     thrust::device_pointer_cast(spins.data()),
                     thrust::device_pointer_cast(random.data()),
                     w, h, 0.5f);

    // ------------------------------------------------------------------------
    // initialize GLFW/OpenGL

    // if lambda has no capture, it can be converted into function pointer.
    glfwSetErrorCallback([](int error, const char* desc) -> void {
        std::printf("Error(%d): %s", error, desc);
    });
    if (!glfwInit())
    {
        curandDestroyGenerator(rng);
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_DEPTH_BITS,            0);
    glfwWindowHint(GLFW_STENCIL_BITS,          0);

    glfwWindowHint(GLFW_SRGB_CAPABLE,          GL_TRUE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

    glfwWindowHint(GLFW_OPENGL_PROFILE,        GLFW_OPENGL_CORE_PROFILE);

    std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> window(
        glfwCreateWindow(w, h, "ising model", NULL, NULL), &glfwDestroyWindow
    );

    if(window.get() == nullptr)
    {
        curandDestroyGenerator(rng);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glfwMakeContextCurrent(window.get());

    // from OpenGL Wiki
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
    {
        std::cout << "Failed to initialize OpenGL context" << std::endl;
        curandDestroyGenerator(rng);
        glfwTerminate();
        return -1;
    }

    glfwSwapInterval(1);

    // ------------------------------------------------------------------------
    // create cuda Stream

    cudaStream_t stream;
    {
        const auto err = cudaStreamCreateWithFlags(&stream, cudaStreamDefault);
        assert(err == 0);
    }

    // ------------------------------------------------------------------------
    // create FrameBuffer/RenderBuffer/cudaGraphicsResource/cudaArray_t

    GLuint frame_buffer;
    glCreateFramebuffers(1, &frame_buffer);

    GLuint render_buffer;
    glCreateRenderbuffers(1, &render_buffer);

    // attach render_buffer -> frame_buffer
    glNamedFramebufferRenderbuffer(
            frame_buffer, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, render_buffer);

    // get current frame buffer size
    int width, height;
    glfwGetFramebufferSize(window.get(), &width, &height);

    // allocate renderbuffer storage
    glNamedRenderbufferStorage(render_buffer, GL_RGBA8, width, height);

    // register render_buffer as cuda graphics resource
    cudaGraphicsResource_t cuda_graphics_resource;
    const auto gl_register_image_err = cudaGraphicsGLRegisterImage(
            &cuda_graphics_resource, render_buffer,
            GL_RENDERBUFFER,
            cudaGraphicsRegisterFlagsSurfaceLoadStore |
            cudaGraphicsRegisterFlagsWriteDiscard
            );
    assert(gl_register_image_err == 0);

    // map graphics resource for access by cuda
    {
        const auto err =
            cudaGraphicsMapResources(1, &cuda_graphics_resource, /* default stream = */ 0);
        assert(err == 0);
    }

    // map cuda_graphics_resource to cuda_array to enable to access by cuda_array
    cudaArray_t cuda_array;
    {
        const auto err = cudaGraphicsSubResourceGetMappedArray(
                &cuda_array, cuda_graphics_resource, 0, 0);
        assert(err == 0);
    }

    // unmap graphics resource after access
    {
        const auto err = cudaGraphicsUnmapResources(
                1, &cuda_graphics_resource, /* default stream = */ 0);
        assert(err == 0);
    }

    // ------------------------------------------------------------------------
    // simulate for some steps
    std::cout << "start simulation..." << std::endl;
    for(std::size_t i=0; i<step; ++i)
    {
        glfwPollEvents();
        if(glfwWindowShouldClose(window.get()))
        {
            break;
        }

        // --------------------------------------------------------------------
        // render current state into cuda array


        if(i % 2 == 0)
        {
            render_field(blocks, threads, stream,
                         thrust::device_pointer_cast(spins.data()),
                         cuda_array, w, h);
        }

        // --------------------------------------------------------------------
        // update window with current rendering status

        glBlitNamedFramebuffer(
                frame_buffer, /* default = */ 0,
                0, 0, w, h,
                0, h, w, 0,
                GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glfwSwapBuffers(window.get());

        // --------------------------------------------------------------------
        // generate random numbers
        curandStatus_t st_genrnd = curandGenerateUniform(rng,
                thrust::device_pointer_cast(random.data()).get(), w * h);
        assert(st_genrnd == CURAND_STATUS_SUCCESS);

        // --------------------------------------------------------------------
        // update spins
        update_field(blocks, threads,
                     thrust::device_pointer_cast(spins.data()),
                     thrust::device_pointer_cast(random.data()),
                     w, h);
    }
    std::cout << "simulation done." << std::endl;

    // ------------------------------------------------------------------------
    // destroy buffers

    std::cout << "deleting OpenGL buffers..." << std::endl;
    glDeleteRenderbuffers(1, &render_buffer);
    glDeleteFramebuffers (1, &frame_buffer);

    cudaGraphicsUnregisterResource(cuda_graphics_resource);
    std::cout << "done." << std::endl;

    // ------------------------------------------------------------------------
    // destroy cuRAND generators
    std::cout << "destroying cuRAND Generator..." << std::endl;
    curandStatus_t destroy = curandDestroyGenerator(rng);
    assert(destroy == CURAND_STATUS_SUCCESS);
    std::cout << "done." << std::endl;

    // ------------------------------------------------------------------------
    // destroy glfw window struct

    window.reset();
    glfwTerminate();

    return 0;
}
