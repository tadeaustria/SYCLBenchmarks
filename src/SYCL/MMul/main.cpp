#include "CL/sycl.hpp"
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
  int size = 1024;
  int wgsize = 16;
  bool detailedOutput = true;
  bool MismatchFound = false;
  if (argc > 1)
  {
    if (strcmp(argv[1], "-h") == 0)
    {
      std::cout << "Usage of Matmul: main.exe [size=" << size << "] [wgsize=" << wgsize << "] [-v]\n";
      std::cout << "Size must be smaller than " << (1024 * 14) << "\n";
      std::cout << "If -v is preset only the time will be in the output" << std::endl;
      return 0;
    }
    size = atoi(argv[1]);
    if (argc > 2)
      wgsize = atoi(argv[2]);
    if (argc > 3 && strcmp(argv[3], "-v") == 0)
      detailedOutput = false;
  }

  if (size <= 0 || size >= 1024 * 14)
    size = 1024;

#ifdef _WIN32
  auto vecAHost = static_cast<cl_float *>(_aligned_malloc(size * size * sizeof(cl_float), 4096));
  auto vecBHost = static_cast<cl_float *>(_aligned_malloc(size * size * sizeof(cl_float), 4096));
  auto vecCHost = static_cast<cl_float *>(_aligned_malloc(size * size * sizeof(cl_float), 4096));
#else
  auto vecAHost = static_cast<cl_float *>(aligned_alloc(4096, size * size * sizeof(cl_float)));
  auto vecBHost = static_cast<cl_float *>(aligned_alloc(4096, size * size * sizeof(cl_float)));
  auto vecCHost = static_cast<cl_float *>(aligned_alloc(4096, size * size * sizeof(cl_float)));
#endif

#pragma omp parallel for collapse(2)
  for (int i = 0; i < size; i++)
  {
    for (int j = 0; j < size; j++)
    {
      vecAHost[i * size + j] = 10000.0f * i + j;
      vecBHost[i * size + j] = (i == j) ? 1.0f : 0.0f;
      vecCHost[i * size + j] = -42.0f;
    }
  }

  try
  {
    auto start = std::chrono::high_resolution_clock::now();

    // Creating buffer of 4 ints to be used inside the kernel code
    cl::sycl::buffer<cl::sycl::cl_float, 2> vecA(vecAHost, cl::sycl::range<2>(size, size));
    cl::sycl::buffer<cl::sycl::cl_float, 2> vecB(vecBHost, cl::sycl::range<2>(size, size));
    cl::sycl::buffer<cl::sycl::cl_float, 2> vecC(vecCHost, cl::sycl::range<2>(size, size));

    // Creating SYCL queue
    cl::sycl::queue queue{cl::sycl::default_selector(), exceptionHandler, cl::sycl::property::queue::enable_profiling()};

    if (detailedOutput)
      std::cout << "Running Matrix multiplication with " << size << "x" << size << " on workgroups " << wgsize << "x" << wgsize << "\n";

    // Size of index space for kernel
    cl::sycl::nd_range<2> NumOfWorkItems{vecA.get_range(), cl::sycl::range<2>(wgsize, wgsize)};

    // Submitting command group(work) to queue
    auto event = queue.submit([&](cl::sycl::handler &cgh) {
      // Getting write only access to the vecA on a device
      auto AccessorB = vecB.get_access<cl::sycl::access::mode::read>(cgh);
      auto AccessorA = vecA.get_access<cl::sycl::access::mode::read>(cgh);
      auto AccessorC = vecC.get_access<cl::sycl::access::mode::discard_write>(cgh);
      // Executing kernel
      cgh.parallel_for<class VecAdd>(
          NumOfWorkItems, [=](cl::sycl::nd_item<2> item) {
            float temp = 0;
            uint32_t idy = item.get_global_id()[0];
            uint32_t idx = item.get_global_id()[1];
            uint32_t sizey = AccessorA.get_range()[0];

            for (uint32_t i = 0; i < sizey; i++)
            {
              temp += AccessorA[idy][i] * AccessorB[i][idx];
            }
            AccessorC[idy][idx] = temp;
          });
    });

    auto profilingStart = std::chrono::nanoseconds(event.get_profiling_info<cl::sycl::info::event_profiling::command_start>());
    auto profilingEnd = std::chrono::nanoseconds(event.get_profiling_info<cl::sycl::info::event_profiling::command_end>());

    {
      // Getting read only access to the buffer on the host.
      // Implicit barrier waiting for queue to complete the work.
      auto HostAccessorC = vecC.get_access<cl::sycl::access::mode::read>();

      auto end = std::chrono::high_resolution_clock::now();
      if (detailedOutput)
      {
        std::cout << "Host Time taken " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms \n";
        std::cout << "Kernel Time " << profilingEnd.count() << " " << profilingStart.count() << " " << std::chrono::duration_cast<std::chrono::milliseconds>(profilingEnd - profilingStart).count() << " ms \n";
      }
      else
      {
        std::cout << size << ";" << wgsize << ";SYCL " << queue.get_context().get_platform().get_info<cl::sycl::info::platform::version>() << " "
                  << queue.get_device().get_info<cl::sycl::info::device::name>() << ";"
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ";"
                  << std::chrono::duration_cast<std::chrono::milliseconds>(profilingEnd - profilingStart).count() << "\n";
      }
      auto HostAccessorA = vecA.get_access<cl::sycl::access::mode::read>();
      auto HostAccessorB = vecB.get_access<cl::sycl::access::mode::read>();

      if (detailedOutput)
        std::cout << "0 out of " << vecC.get_range()[0] << "\r";

        // // Check the results
#pragma omp parallel for collapse(2)
      for (unsigned long long i = 0; i < vecC.get_range()[0]; i++)
      {
        for (unsigned long long j = 0; j < vecC.get_range()[1]; j++)
        {
          if (HostAccessorC[i][j] != HostAccessorA[i][j])
          {
#pragma omp critical
            {
              std::cout << "The result is incorrect for element: " << i << ", " << j
                        << " , " << HostAccessorC[i][j] << " expected: " << HostAccessorA[i][j]
                        << std::endl;
              MismatchFound = true;
              // break;
            }
          }
          if (HostAccessorB[i][j] != (i == j) ? 1 : 0)
          {
#pragma omp critical
            {
              std::cout << "The B result is incorrect for element: " << i << ", " << j
                        << " , " << HostAccessorB[i][j] << " expected: " << (i == j)
                        << std::endl;
              MismatchFound = true;
            }
          }
        }
        if (detailedOutput)
          std::cout << (i + 1) << " out of " << vecC.get_range()[0] << "\r";
      }
    }
    if (!MismatchFound && detailedOutput)
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

#if _WIN32
  _aligned_free(vecCHost);
  _aligned_free(vecBHost);
  _aligned_free(vecAHost);
#else
  free(vecCHost);
  free(vecBHost);
  free(vecAHost);
#endif

  return MismatchFound;
}