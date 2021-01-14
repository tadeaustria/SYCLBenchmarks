#include "CL/sycl.hpp"
#include "../../Headers/Stencil.h"
#include <chrono>

void exceptionHandler(sycl::exception_list exceptions)
{
  for (std::exception_ptr const &e : exceptions)
  {
    try
    {
      std::rethrow_exception(e);
    }
    catch (sycl::exception const &e)
    {
      std::cout << "Caught asynchronous SYCL exception:\n"
                << e.what() << std::endl;
    }
  }
}

int main(int argc, char *argv[])
{
  int timesteps = 2000;
  int size = 1024;
  int wgsize = 16;
  bool detailedOutput = true;
  bool mismatchFound = false;
  std::string outputFile;

  if (argc > 1)
  {
    if (strcmp(argv[1], "-h") == 0)
    {
      std::cout << "Usage of Matmul: main.exe [timesteps=" << timesteps << "] [size=" << size << "] [wgsize=" << wgsize << "] [-v]\n";
      std::cout << "Size must be smaller than " << (1024 * 16) << "\n";
      std::cout << "If -v is preset only the time will be in the output" << std::endl;
      return 0;
    }
    timesteps = atoi(argv[1]);
    if (argc > 2)
      size = atoi(argv[2]);
    if (argc > 3)
      wgsize = atoi(argv[3]);
    if (argc > 4){
      if(strcmp(argv[4], "-v") == 0){
        detailedOutput = false;
      }else{
        outputFile = argv[4];
      }
    }
  }

  if (size <= 0 || size > 1024 * 16)
    size = 1024;

  auto vecAHost = static_cast<cl_float *>(_aligned_malloc(size * size * sizeof(cl_float), 4096));
  auto vecBHost = static_cast<cl_float *>(_aligned_malloc(size * size * sizeof(cl_float), 4096));

  int xSource = size / 4;
  int ySource = size / 4;

#pragma omp parallel for collapse(2)
  for (int i = 0; i < size; i++)
  {
    for (int j = 0; j < size; j++)
    {
      if (i == ySource && j == xSource)
        vecAHost[i * size + j] = max;
      else
        vecAHost[i * size + j] = min;
      vecBHost[i * size + j] = -42.0f;
    }
  }

  std::vector<cl::sycl::event> events;
  events.reserve(timesteps);

  // Start measuring time before buffer is copied to device
  std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();

  std::chrono::nanoseconds profilingStart, profilingEnd;

  try
  {
    // Creating buffer of 4 ints to be used inside the kernel code
    cl::sycl::buffer<cl::sycl::cl_float, 2> vecA(vecAHost, cl::sycl::range<2>(size, size));
    cl::sycl::buffer<cl::sycl::cl_float, 2> vecB(vecBHost, cl::sycl::range<2>(size, size));

    // Creating SYCL queue
    cl::sycl::queue queue{cl::sycl::default_selector(), exceptionHandler, cl::sycl::property::queue::enable_profiling()};

    if (detailedOutput)
      std::cout << "Running Heat Stencil with " << size << "x" << size << " on workgroups " << wgsize << "x" << wgsize << " for " << timesteps << " timesteps\n";

    // Size of index space for kernel
    cl::sycl::nd_range<2> NumOfWorkItems{vecA.get_range(), cl::sycl::range<2>(wgsize, wgsize)};

    // cl::sycl::event startEvent;
    // cl::sycl::event endEvent;
    for (int i = 0; i < timesteps; i++)
    {
      // Submitting command group(work) to queue
      auto event = queue.submit([&](cl::sycl::handler &cgh) {
        // Getting write only access to the vecA on a device
        auto Source = vecA.get_access<cl::sycl::access::mode::read>(cgh);
        auto Target = vecB.get_access<cl::sycl::access::mode::discard_write>(cgh);
        if (i % 2 == 1)
        {
          Source = vecB.get_access<cl::sycl::access::mode::read>(cgh);
          Target = vecA.get_access<cl::sycl::access::mode::discard_write>(cgh);
        }
        // Executing kernel
        cgh.parallel_for<class VecAdd>(
            NumOfWorkItems, [=](cl::sycl::nd_item<2> item) {
              auto x = item.get_global_id()[1];
              auto y = item.get_global_id()[0];
              float temp = 4.0f * Source[item.get_global_id()];
              temp += (x == 0) ? Source[item.get_global_id()] : Source[y][x - 1];
              temp += (x == size - 1) ? Source[item.get_global_id()] : Source[y][x + 1];
              temp += (y == 0) ? Source[item.get_global_id()] : Source[y - 1][x];
              temp += (y == size - 1) ? Source[item.get_global_id()] : Source[y + 1][x];
              if (x == xSource && y == ySource)
              {
                Target[item.get_global_id()] = Source[item.get_global_id()];
              }
              else
              {
                Target[item.get_global_id()] = temp / 8.0f;
              }
            });
      });
      events.push_back(event);
      // if (i == 0)
      //   startEvent = endEvent;
      if (detailedOutput && ((i + 1) % 25 == 0))
        std::cout << i + 1 << " of " << timesteps << "\r";
    }

    {
      // Getting read only access to the buffer on the host.
      // Implicit barrier waiting for queue to complete the work.
      auto HostAccessorA = vecA.get_access<cl::sycl::access::mode::read>();

      auto end = std::chrono::high_resolution_clock::now();

      std::chrono::nanoseconds kernelTime{0};
      for(auto event : events){
        auto profilingStart = std::chrono::nanoseconds(event.get_profiling_info<cl::sycl::info::event_profiling::command_start>());
        auto profilingEnd = std::chrono::nanoseconds(event.get_profiling_info<cl::sycl::info::event_profiling::command_end>());
        kernelTime += (profilingEnd - profilingStart);
      }

      if (detailedOutput)
      {
        std::cout << "Host Time taken " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms \n";
        std::cout << "Kernel Time " << std::chrono::duration_cast<std::chrono::milliseconds>(kernelTime).count() << " ms \n";
      }
      else
      {
        std::cout << timesteps << ";" << size << ";" << wgsize << ";SYCL " << queue.get_context().get_platform().get_info<cl::sycl::info::platform::version>() << " "
                  << queue.get_device().get_info<cl::sycl::info::device::name>() << ";"
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ";"
                  << std::chrono::duration_cast<std::chrono::milliseconds>(kernelTime).count() << "\n";
      }

      if (detailedOutput)
      {
        if(outputFile.empty()){
          printStencil(HostAccessorA.get_pointer(), size, size, std::cout);
        }else{
          auto outputStream = std::ofstream(outputFile);
          printStencil(HostAccessorA.get_pointer(), size, size, outputStream, size, size);
        }
        std::cout << "0 out of " << vecA.get_range()[0] << "\r";
      }

      if (HostAccessorA[size / 4][size / 4] != max)
        mismatchFound = true;

        // Sanity check
#ifdef WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for collapse(2) schedule(static)
#endif
      for (unsigned long long i = 0; i < vecA.get_range()[0]; i++)
      {
        for (unsigned long long j = 0; j < vecA.get_range()[1]; j++)
        {
          if (HostAccessorA[i][j] < min || HostAccessorA[i][j] > max)
          {
#pragma omp critical
            {
              std::cout << "The result is incorrect for element: " << i << ", " << j
                        << " , " << HostAccessorA[i][j] << std::endl;
              mismatchFound = true;
              //break;
            }
          }
        }
        if (detailedOutput)
          std::cout << (i + 1) << " out of " << vecA.get_range()[0] << "\r";
      }
    }
    if (!mismatchFound && detailedOutput)
    {
      std::cout << "The results are correct!" << std::endl;
    }
  }
  catch (sycl::exception const &e)
  {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
    return 1;
  }

  _aligned_free(vecBHost);
  _aligned_free(vecAHost);

  return mismatchFound;
}