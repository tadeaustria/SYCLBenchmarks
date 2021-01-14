
#include "../../Headers/Prefix.h"
#include "CL/sycl.hpp"
#include <chrono>
#include <thread>

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

void executePrefix(cl::sycl::handler &cgh, cl::sycl::buffer<cl::sycl::cl_float, 1> &from, cl::sycl::buffer<cl::sycl::cl_float, 1> &to,
                   cl::sycl::buffer<cl::sycl::cl_float, 1> &temp, cl::sycl::buffer<cl::sycl::cl_float, 1> &preSum, size_t totalSize, size_t totalItems, size_t workGroup)
{

  auto AccessorA = from.get_access<cl::sycl::access::mode::read>(cgh);
  auto AccessorB = to.get_access<cl::sycl::access::mode::discard_write>(cgh);
  auto AccessorTemp = temp.get_access<cl::sycl::access::mode::discard_write>(cgh);
  auto AccessorPreSum = preSum.get_access<cl::sycl::access::mode::discard_write>(cgh);

  // Size of index space for kernel
  cl::sycl::nd_range<1> NumOfWorkGroups{totalItems, cl::sycl::range<1>(workGroup)};

  // Executing kernel
  cgh.parallel_for<class PrefixSum>(
      NumOfWorkGroups, [=](cl::sycl::nd_item<1> item) {
        auto x = item.get_global_id()[0];
        auto xi = item.get_local_id()[0];
        unsigned int offset = 1;
        if (2 * x < totalSize)
        {
          AccessorTemp[2 * x] = AccessorA[2 * item.get_global_id()[0]];
          AccessorTemp[2 * x + 1] = AccessorA[2 * item.get_global_id()[0] + 1];
        }
        else
        {
          AccessorTemp[2 * x] = 0.0f;
          AccessorTemp[2 * x + 1] = 0.0f;
        }

        for (unsigned int d = (item.get_local_range()[0] * 2) >> 1; d > 0; d >>= 1) // build sum in place up the tree
        {
          item.mem_fence(cl::sycl::access::fence_space::global_space);
          item.barrier(cl::sycl::access::fence_space::global_space);
          if (xi < d)
          {
            unsigned int ai = offset * (2 * xi + 1) - 1 + (item.get_group()[0] * item.get_local_range()[0] * 2);
            unsigned int bi = offset * (2 * xi + 2) - 1 + (item.get_group()[0] * item.get_local_range()[0] * 2);
            // unsigned int ai = (2 * x + offset - 1);
            // unsigned int bi = (2 * x + 2 * offset - 1);
            AccessorTemp[bi] += AccessorTemp[ai];
          }
          offset *= 2;
        }

        if (xi == 0)
        {
          AccessorPreSum[item.get_group()[0]] = AccessorTemp[(item.get_group()[0] + 1) * item.get_local_range()[0] * 2 - 1];
          AccessorTemp[(item.get_group()[0] + 1) * item.get_local_range()[0] * 2 - 1] = 0.0f; // clear the last element
        }

        for (unsigned int d = 1; d < (item.get_local_range()[0] * 2); d *= 2) // traverse down tree & build scan
        {
          offset >>= 1;
          item.mem_fence(cl::sycl::access::fence_space::global_space);
          item.barrier(cl::sycl::access::fence_space::global_space);
          if (xi < d)
          {
            unsigned int ai = offset * (2 * xi + 1) - 1 + (item.get_group()[0] * item.get_local_range()[0] * 2);
            unsigned int bi = offset * (2 * xi + 2) - 1 + (item.get_group()[0] * item.get_local_range()[0] * 2);
            // unsigned int ai = (2 * x + offset - 1);
            // unsigned int bi = (2 * x + 2 * offset - 1);
            float t = AccessorTemp[ai];
            AccessorTemp[ai] = AccessorTemp[bi];
            AccessorTemp[bi] += t;
          }
        }

        item.mem_fence(cl::sycl::access::fence_space::global_space);
        item.barrier(cl::sycl::access::fence_space::global_space);
        if (2 * x < totalSize)
        {
          AccessorB[2 * item.get_global_id()[0]] = AccessorTemp[2 * x];
          AccessorB[2 * item.get_global_id()[0] + 1] = AccessorTemp[2 * x + 1];
        }
      });
}

int main(int argc, char *argv[])
{

  int32_t size = 1024;
  bool detailedOutput = true;
  bool MismatchFound = false;

  if (argc > 1)
  {
    if (strcmp(argv[1], "-h") == 0)
    {
      std::cout << "Usage of Prefix Sum: main.exe [size=" << size << "] [-v]\n";
      std::cout << "Size must be smaller than " << (4194304) << " due to algorithm 1024*1024*4\n";
      std::cout << "If -v is preset only the time will be in the output" << std::endl;
      return 0;
    }
    size = atoi(argv[1]);
    if (argc > 2 && strcmp(argv[2], "-v") == 0)
      detailedOutput = false;
  }

  if (size <= 0 || size > 4194304)
    size = 1024;

  int32_t wgsize = size / 2;

  auto vecAHost = static_cast<cl_float *>(_aligned_malloc(size*sizeof(cl_float), 4096));
  auto vecBHost = static_cast<cl_float *>(_aligned_malloc(size*sizeof(cl_float), 4096));

#pragma omp parallel for
  for (int i = 0; i < size; i++)
  {
    // hostAccessorA[i] = size - i - 1;
    vecAHost[i] = 1.0f;
    vecBHost[i] = -1.0f;
  }

  try
  {
    auto start = std::chrono::high_resolution_clock::now();

    cl::sycl::buffer<cl::sycl::cl_float, 1> vecA{vecAHost, cl::sycl::range<1>(size)};
    cl::sycl::buffer<cl::sycl::cl_float, 1> vecB{vecBHost, cl::sycl::range<1>(size)};

    // Creating SYCL queue
    cl::sycl::queue queue{cl::sycl::default_selector(), exceptionHandler, cl::sycl::property::queue::enable_profiling()};

    auto deviceLimit = queue.get_device().get_info<cl::sycl::info::device::max_work_item_sizes>();

    int wgsize2 = 1;

    // Workgroup size must be a power of 2 due to the algorithm
    // try use upper bound of power of 2
    wgsize = static_cast<int>(pow(2.0f, ceil(log2((float)wgsize))));
    // deviceLimit[2] = 2;
    // check upper limit of workgroup size with device limits
    if (wgsize > deviceLimit[2])
    {
      int temp = wgsize;
      // take lower bound of power of 2 of device limit
      wgsize = static_cast<int>(pow(2.0f, floor(log2((float)deviceLimit[2]))));
      // caluclate workgroup size for inner presum task
      wgsize2 = static_cast<int>(pow(2.0f, ceil(log2((float)size / wgsize / 2))));
    }

    cl::sycl::buffer<cl::sycl::cl_float, 1> vecTemp{cl::sycl::range<1>(wgsize2 * wgsize * 2)};
    cl::sycl::buffer<cl::sycl::cl_float, 1> preSum{cl::sycl::range<1>(wgsize2)};

    if (detailedOutput)
      std::cout << "Running Prefix Sum on " << size << " using workgroupsize of " << wgsize << " and " << wgsize2 << " steps\n";

    // Submitting command group(work) to queue
    auto event1 = queue.submit([&](cl::sycl::handler &cgh) {
      executePrefix(cgh, vecA, vecB, vecTemp, preSum, size, wgsize2 * wgsize, wgsize);
    });

    auto event2 = queue.submit([&](cl::sycl::handler &cgh) {
      size_t groups = wgsize2 < 2 ? 1 : wgsize2 / 2;
      executePrefix(cgh, preSum, preSum, vecTemp, vecA, wgsize2, groups, groups);
    });

    auto event3 = queue.submit([&](cl::sycl::handler &cgh) {
      auto AccessorB = vecB.get_access<cl::sycl::access::mode::read_write>(cgh);
      auto AccessorPreSum = preSum.get_access<cl::sycl::access::mode::read>(cgh);

      // Size of index space for kernel
      cl::sycl::nd_range<1> NumOfWorkGroups{wgsize2 * wgsize, cl::sycl::range<1>(wgsize)};

      // Executing kernel
      cgh.parallel_for<class PartSumAdd>(
          NumOfWorkGroups, [=](cl::sycl::nd_item<1> item) {
            auto x = item.get_global_id()[0];
            if (2 * x < size)
            {
              AccessorB[2 * x] += AccessorPreSum[item.get_group()[0]];
              AccessorB[2 * x + 1] += AccessorPreSum[item.get_group()[0]];
            }
          });
    });

    {
      // Getting read only access to the buffer on the host.
      // Implicit barrier waiting for queue to complete the work.
      auto HostAccessorB = vecB.get_access<cl::sycl::access::mode::read>();

      auto end = std::chrono::high_resolution_clock::now();

      // auto profilingStart = std::chrono::nanoseconds(event1.get_profiling_info<cl::sycl::info::event_profiling::command_start>());
      auto profilingStart = std::chrono::nanoseconds(event1.get_profiling_info<cl::sycl::info::event_profiling::command_start>() + event2.get_profiling_info<cl::sycl::info::event_profiling::command_start>() + event3.get_profiling_info<cl::sycl::info::event_profiling::command_start>());
      auto profilingEnd = std::chrono::nanoseconds(event1.get_profiling_info<cl::sycl::info::event_profiling::command_end>() + event2.get_profiling_info<cl::sycl::info::event_profiling::command_end>() + event3.get_profiling_info<cl::sycl::info::event_profiling::command_end>());
      // auto profilingEnd = std::chrono::nanoseconds(event3.get_profiling_info<cl::sycl::info::event_profiling::command_end>());

      if (detailedOutput)
      {
        std::cout << "Host Time taken " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us \n";
        std::cout << "Kernel Time " << profilingEnd.count() << " " << profilingStart.count() << " " << std::chrono::duration_cast<std::chrono::microseconds>(profilingEnd - profilingStart).count() << " us \n";
      }
      else
      {
        std::cout << size << ";SYCL " << queue.get_context().get_platform().get_info<cl::sycl::info::platform::version>() << " "
                  << queue.get_device().get_info<cl::sycl::info::device::name>() << ";"
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << ";"
                  << std::chrono::duration_cast<std::chrono::microseconds>(profilingEnd - profilingStart).count() << "\n";
      }

      if (detailedOutput)
        std::cout << "0 out of " << vecA.get_range()[0] << "\r";

      float temp = 0.0f;
      for (int32_t k = size - 1, i = 0; k >= 0; k--, i++)
      {
        // if (!floatCompare(HostAccessorB[i], temp))
        if(HostAccessorB[i] != temp)
        {
          printf("%d: expected: %f\t\tactual: %f\n", i, temp, HostAccessorB[i]);
          MismatchFound = true;
        }
        // temp += k;
        temp += 1.0f;
        if (detailedOutput && i % 100 == 0)
          std::cout << (i + 1) << " out of " << size << "\r";
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
  
  _aligned_free(vecBHost);
  _aligned_free(vecAHost);

  return MismatchFound;
}