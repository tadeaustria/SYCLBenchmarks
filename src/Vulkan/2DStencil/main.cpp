// This is free and unencumbered software released into the public domain.
//
// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.
//
// In jurisdictions that recognize copyright laws, the author or authors
// of this software dedicate any and all copyright interest in the
// software to the public domain. We make this dedication for the benefit
// of the public at large and to the detriment of our heirs and
// successors. We intend this dedication to be an overt act of
// relinquishment in perpetuity of all present and future rights to this
// software under copyright law.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// For more information, please refer to <http://unlicense.org/>

#include "vulkan/vulkan.hpp"
#include "../../Headers/Stencil.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#define BAIL_ON_BAD_RESULT(result)                             \
  if (VK_SUCCESS != (result))                                  \
  {                                                            \
    fprintf(stderr, "Failure at %u %s\n", __LINE__, __FILE__); \
    exit(-1);                                                  \
  }

VkResult vkGetBestComputeQueueNPH(vk::PhysicalDevice &physicalDevice, uint32_t &queueFamilyIndex)
{

  auto properties = physicalDevice.getQueueFamilyProperties();
  int i = 0;
  for (auto prop : properties)
  {
    vk::QueueFlags maskedFlags = (~(vk::QueueFlagBits::eTransfer | vk::QueueFlagBits::eSparseBinding) & prop.queueFlags);
    if (!(vk::QueueFlagBits::eGraphics & maskedFlags) && (vk::QueueFlagBits::eCompute & maskedFlags))
    {
      queueFamilyIndex = i;
      return VK_SUCCESS;
    }
    i++;
  }
  i = 0;
  for (auto prop : properties)
  {
    vk::QueueFlags maskedFlags = (~(vk::QueueFlagBits::eTransfer | vk::QueueFlagBits::eSparseBinding) & prop.queueFlags);
    if (vk::QueueFlagBits::eCompute & maskedFlags)
    {
      queueFamilyIndex = i;
      return VK_SUCCESS;
    }
    i++;
  }
  return VK_ERROR_INITIALIZATION_FAILED;
}

int main(int argc, const char *const argv[])
{

  int32_t timesteps = 2000;
  int32_t size = 1024;
  int32_t wgsize = 16;
  int32_t device = 0;
  bool detailedOutput = true;
  std::string outputFile;

  if (argc > 1)
  {
    if (strcmp(argv[1], "-h") == 0)
    {
      std::cout << "Usage of Matmul: main.exe [timesteps=" << timesteps << "] [size=" << size << "] [wgsize=" << wgsize << "] [device=" << device << "][-v]\n";
      std::cout << "Size must be smaller than " << (1024 * 16) << "\n";
      std::cout << "If -v is preset only the time will be in the output" << std::endl;
      return 0;
    }
    timesteps = atoi(argv[1]);
    if (argc > 2)
      size = atoi(argv[2]);
    if (argc > 3)
      wgsize = atoi(argv[3]);
    if (argc > 4)
      device = atoi(argv[4]);
    if (argc > 5){
      if(strcmp(argv[5], "-v") == 0){
        detailedOutput = false;
      }else{
        outputFile = argv[5];
      }
    }
  }

  if (size <= 0 || size > 1024 * 16)
    size = 1024;

  const int32_t bufferLength = static_cast<int32_t>(size * size);
  const uint32_t bufferSize = sizeof(float_t) * bufferLength;
  // we are going to need two buffers from this one memory
  const vk::DeviceSize memorySize = bufferSize;

  auto matAHost = static_cast<float_t *>(_aligned_malloc(size * size * sizeof(float_t), 4096));

#ifdef WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int32_t k = 0; k < size; k++)
  {
    for (int32_t j = 0; j < size; j++)
    {
      if (k == size / 4 && j == size / 4)
        matAHost[k * size + j] = max;
      else
        matAHost[k * size + j] = min;
    }
  }

  if (detailedOutput)
    printStencil(matAHost, size, size, std::cout);

  try
  {
    auto start = std::chrono::high_resolution_clock::now();

    // initialize the vk::ApplicationInfo structure
    vk::ApplicationInfo applicationInfo("Stencil", 1, "Vulkan.hpp", 1, VK_API_VERSION_1_1);

    // initialize the vk::InstanceCreateInfo
    std::vector<char *> layers = {
        // "VK_LAYER_LUNARG_api_dump",
        // "VK_LAYER_KHRONOS_validation"
    };
    vk::InstanceCreateInfo instanceCreateInfo({}, &applicationInfo, static_cast<uint32_t>(layers.size()), layers.data());
    // create a UniqueInstance
    vk::UniqueInstance instance = vk::createInstanceUnique(instanceCreateInfo);
    auto physicalDevices = instance->enumeratePhysicalDevices();

    auto physicalDevice = physicalDevices[device];
    // for (auto &physicalDevice : physicalDevices)
    {
      auto props = physicalDevice.getProperties();

      // get the QueueFamilyProperties of the first PhysicalDevice
      std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
      uint32_t computeQueueFamilyIndex = 0;

      // get the best index into queueFamiliyProperties which supports compute and stuff
      BAIL_ON_BAD_RESULT(vkGetBestComputeQueueNPH(physicalDevice, computeQueueFamilyIndex));

      std::vector<char *> extensions = {
          "VK_EXT_external_memory_host",
          //"VK_KHR_shader_float16_int8",
          "VK_KHR_timeline_semaphore"};
      // create a UniqueDevice
      float queuePriority = 0.0f;

      vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), static_cast<uint32_t>(computeQueueFamilyIndex), 1, &queuePriority);
      vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR> deviceCreateInfo = {
          vk::DeviceCreateInfo(vk::DeviceCreateFlags(), 1, &deviceQueueCreateInfo, 0, nullptr, static_cast<uint32_t>(extensions.size()), extensions.data()),
          vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR(true)};
      vk::UniqueDevice device = physicalDevice.createDeviceUnique(deviceCreateInfo.get<vk::DeviceCreateInfo>());

      if (detailedOutput)
        std::cout << "Device found: " << props.deviceName << std::endl;

      vk::DispatchLoaderDynamic dll(instance.get(), vkGetInstanceProcAddr, device.get(), vkGetDeviceProcAddr);

      auto memoryProperties2 = physicalDevice.getMemoryProperties2();
      vk::PhysicalDeviceMemoryProperties const &memoryProperties = memoryProperties2.memoryProperties;

      // set memoryTypeIndex to an invalid entry in the properties.memoryTypes array
      uint32_t memoryTypeIndexStaging = VK_MAX_MEMORY_TYPES;
      uint32_t memoryTypeIndexDevice = VK_MAX_MEMORY_TYPES;

      auto extMemProperties = device->getMemoryHostPointerPropertiesEXT(
          vk::ExternalMemoryHandleTypeFlagBitsKHR::eHostAllocationEXT, matAHost,
          dll);

      for (uint32_t k = 0; k < memoryProperties.memoryTypeCount; k++)
      {
        if ((vk::MemoryPropertyFlagBits::eDeviceLocal & memoryProperties.memoryTypes[k].propertyFlags) &&
            (memorySize * 2 < memoryProperties.memoryHeaps[memoryProperties.memoryTypes[k].heapIndex].size))
        {
          memoryTypeIndexDevice = k;
        }
        if (vk::MemoryPropertyFlagBits::eHostVisible & memoryProperties.memoryTypes[k].propertyFlags && vk::MemoryPropertyFlagBits::eHostCoherent & memoryProperties.memoryTypes[k].propertyFlags &&
            (memorySize < memoryProperties.memoryHeaps[memoryProperties.memoryTypes[k].heapIndex].size) && (extMemProperties.memoryTypeBits & (1 << k)) != 0)
          {
            memoryTypeIndexStaging = k;
          }
      }

      BAIL_ON_BAD_RESULT(memoryTypeIndexStaging == VK_MAX_MEMORY_TYPES ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS);
      BAIL_ON_BAD_RESULT(memoryTypeIndexDevice == VK_MAX_MEMORY_TYPES ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS);

      vk::StructureChain<vk::MemoryAllocateInfo, vk::ImportMemoryHostPointerInfoEXT>
          allocInfo = {
              vk::MemoryAllocateInfo(memorySize, memoryTypeIndexStaging),
              vk::ImportMemoryHostPointerInfoEXT(vk::ExternalMemoryHandleTypeFlagBitsKHR::eHostAllocationEXT, matAHost)};

      auto memoryA = device->allocateMemoryUnique(allocInfo.get<vk::MemoryAllocateInfo>());

      auto memoryADevice = device->allocateMemoryUnique(vk::MemoryAllocateInfo(memorySize, memoryTypeIndexDevice));
      auto memoryBDevice = device->allocateMemoryUnique(vk::MemoryAllocateInfo(memorySize, memoryTypeIndexDevice));

      auto inA_buffer = device->createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), bufferSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive));
      device->bindBufferMemory(inA_buffer.get(), memoryA.get(), 0);
      auto inA_bufferDevice = device->createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), bufferSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive));
      device->bindBufferMemory(inA_bufferDevice.get(), memoryADevice.get(), 0);
      auto inB_bufferDevice = device->createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), bufferSize, vk::BufferUsageFlagBits::eStorageBuffer, vk::SharingMode::eExclusive));
      device->bindBufferMemory(inB_bufferDevice.get(), memoryBDevice.get(), 0);

      // create a DescriptorSetLayout
      std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBinding{
          {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
          {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}};

      vk::UniqueDescriptorSetLayout descriptorSetLayout = device->createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), static_cast<uint32_t>(descriptorSetLayoutBinding.size()), descriptorSetLayoutBinding.data()));

      std::ifstream myfile;
      myfile.open("shaders/Stencil.comp.spv", std::ios::ate | std::ios::binary);
      if (!myfile.is_open())
      {
        myfile.open("../../shaders/Stencil.comp.spv", std::ios::ate | std::ios::binary);
        if (!myfile.is_open())
        {
          std::cout << "File Stencil.comp.spv not found" << std::endl;
          return EXIT_FAILURE;
        }
      }

      if (detailedOutput)
        std::cout << "Running Heat Stencil with " << size << "x" << size << " on workgroups " << wgsize << "x" << wgsize << " for " << timesteps << " timesteps\n";

      auto fileSize = myfile.tellg();
      std::vector<unsigned int> shader_spv(fileSize / sizeof(unsigned int));
      myfile.seekg(0);
      myfile.read(reinterpret_cast<char *>(shader_spv.data()), fileSize);
      myfile.close();

      auto shaderModule = device->createShaderModuleUnique(vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), shader_spv.size() * sizeof(unsigned int), shader_spv.data()));

      // create a PipelineLayout using that DescriptorSetLayout
      std::vector<vk::PushConstantRange> pushConstants = {
          {vk::ShaderStageFlagBits::eCompute, 0u, 4u * sizeof(uint32_t)},
      };
      vk::UniquePipelineLayout pipelineLayout = device->createPipelineLayoutUnique(
          vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(),
                                       1,
                                       &descriptorSetLayout.get(),
                                       static_cast<uint32_t>(pushConstants.size()),
                                       pushConstants.data()));

      std::vector<uint32_t> Values = {static_cast<uint32_t>(wgsize), static_cast<uint32_t>(wgsize), 1u};
      std::vector<vk::SpecializationMapEntry> Entries;

      for (uint32_t i = 0; i < static_cast<uint32_t>(Values.size()); i++)
      {
        Entries.emplace_back(100u + i, static_cast<uint32_t>(sizeof(uint32_t)) * i, sizeof(uint32_t));
      }

      vk::SpecializationInfo SpecializationInfo(
          static_cast<uint32_t>(Entries.size()), Entries.data(),
          Values.size() * sizeof(uint32_t), Values.data());

      vk::ComputePipelineCreateInfo computePipelineInfo(
          vk::PipelineCreateFlags(),
          vk::PipelineShaderStageCreateInfo(
              vk::PipelineShaderStageCreateFlags(),
              vk::ShaderStageFlagBits::eCompute,
              shaderModule.get(),
              "main", &SpecializationInfo),
          pipelineLayout.get());

      auto pipeline = device->createComputePipelineUnique(nullptr, computePipelineInfo);

      std::array<vk::DescriptorSetLayout, 2> descrSetLayouts = {
          descriptorSetLayout.get(),
          descriptorSetLayout.get()};

      auto descriptorPoolSize = vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, static_cast<uint32_t>(descrSetLayouts.size() * descriptorSetLayoutBinding.size()));
      auto descriptorPool = device->createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet), 2, 1, &descriptorPoolSize));

      auto descriptorSets = device->allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(descriptorPool.get(), static_cast<uint32_t>(descrSetLayouts.size()), descrSetLayouts.data()));

      vk::DescriptorBufferInfo inA_descriptorBufferInfo(inA_bufferDevice.get(), 0, VK_WHOLE_SIZE);
      vk::DescriptorBufferInfo inB_descriptorBufferInfo(inB_bufferDevice.get(), 0, VK_WHOLE_SIZE);
      // vk::DescriptorBufferInfo outC_descriptorBufferInfo(outC_bufferDevice.get(), 0, VK_WHOLE_SIZE);

      std::vector<vk::WriteDescriptorSet> writeSets = {
          vk::WriteDescriptorSet{
              descriptorSets[0].get(),
              0,
              0,
              1,
              vk::DescriptorType::eStorageBuffer,
              nullptr,
              &inA_descriptorBufferInfo,
              nullptr},
          vk::WriteDescriptorSet{
              descriptorSets[0].get(),
              1,
              0,
              1,
              vk::DescriptorType::eStorageBuffer,
              nullptr,
              &inB_descriptorBufferInfo,
              nullptr},
          vk::WriteDescriptorSet{
              descriptorSets[1].get(),
              1,
              0,
              1,
              vk::DescriptorType::eStorageBuffer,
              nullptr,
              &inA_descriptorBufferInfo,
              nullptr},
          vk::WriteDescriptorSet{
              descriptorSets[1].get(),
              0,
              0,
              1,
              vk::DescriptorType::eStorageBuffer,
              nullptr,
              &inB_descriptorBufferInfo,
              nullptr}};

      // device->updateDescriptorSets(static_cast<uint32_t>(writeSets.size()), writeSets.data(), 0, nullptr);
      device->updateDescriptorSets(writeSets, /*copies*/ nullptr);

      vk::StructureChain<vk::SemaphoreCreateInfo, vk::SemaphoreTypeCreateInfoKHR> semaphoeCreateInfo = {
          vk::SemaphoreCreateInfo(vk::SemaphoreCreateFlags()),
          vk::SemaphoreTypeCreateInfoKHR(vk::SemaphoreType::eTimeline, 0)};

      auto timedSemaphore = device->createSemaphoreUnique(semaphoeCreateInfo.get<vk::SemaphoreCreateInfo>());

      auto commandPool = device->createCommandPoolUnique(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, computeQueueFamilyIndex));

      auto queryPool =
          device->createQueryPoolUnique(vk::QueryPoolCreateInfo(
              vk::QueryPoolCreateFlags(), vk::QueryType::eTimestamp, 2));

      auto commandBuffers = device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(commandPool.get(), vk::CommandBufferLevel::ePrimary, 4));
      std::array<vk::BufferCopy, 1> region = {vk::BufferCopy(0, 0, memorySize)};
      uint32_t arr[] = {static_cast<uint32_t>(size), static_cast<uint32_t>(size), static_cast<uint32_t>(size / 4), static_cast<uint32_t>(size / 4)};

      commandBuffers[0]->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse)));
      commandBuffers[0]->bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.value.get());
      commandBuffers[0]->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, 1, &descriptorSets[0].get(), 0, nullptr);
      commandBuffers[0]->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(arr), arr);
      commandBuffers[0]->dispatch(size / wgsize, size / wgsize, 1);
      commandBuffers[0]->end();

      commandBuffers[1]->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse)));
      commandBuffers[1]->bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.value.get());
      commandBuffers[1]->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, 1, &descriptorSets[1].get(), 0, nullptr);
      commandBuffers[1]->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(arr), arr);
      commandBuffers[1]->dispatch(size / wgsize, size / wgsize, 1);
      commandBuffers[1]->end();

      commandBuffers[2]->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)));
      commandBuffers[2]->resetQueryPool(queryPool.get(), 0, 2);
      commandBuffers[2]->copyBuffer(inA_buffer.get(), inA_bufferDevice.get(), region);
      commandBuffers[2]->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                                        queryPool.get(), 0);
      commandBuffers[2]->end();

      commandBuffers[3]->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)));
      commandBuffers[3]->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                                        queryPool.get(), 1);
      commandBuffers[3]->copyBuffer(inA_bufferDevice.get(), inA_buffer.get(), region);
      commandBuffers[3]->end();

      auto queue = device->getQueue(computeQueueFamilyIndex, 0);

      uint64_t semaphoreValueOld = 0;
      uint64_t semaphoreValueNew = 1;
      const vk::PipelineStageFlags semaphoreFlags = vk::PipelineStageFlagBits::eTopOfPipe;
      vk::StructureChain<vk::SubmitInfo, vk::TimelineSemaphoreSubmitInfoKHR> submitInfo = {
          vk::SubmitInfo(0, nullptr, nullptr, 1, &commandBuffers[2].get(), 1, &timedSemaphore.get()),
          vk::TimelineSemaphoreSubmitInfoKHR(0, nullptr, 1, &semaphoreValueNew)};
      queue.submit(1, &submitInfo.get<vk::SubmitInfo>(), vk::Fence());
      semaphoreValueOld = semaphoreValueNew++;

      for (int i = 0; i < timesteps; i++)
      {
        submitInfo = {
            vk::SubmitInfo(1, &timedSemaphore.get(), &semaphoreFlags, 1, &commandBuffers[i % 2].get(), 1, &timedSemaphore.get()),
            vk::TimelineSemaphoreSubmitInfoKHR(1, &semaphoreValueOld, 1, &semaphoreValueNew)};
        queue.submit(1, &submitInfo.get<vk::SubmitInfo>(), vk::Fence());
        semaphoreValueOld = semaphoreValueNew++;
      }
      submitInfo = {
          vk::SubmitInfo(1, &timedSemaphore.get(), &semaphoreFlags, 1, &commandBuffers[3].get(), 1, &timedSemaphore.get()),
          vk::TimelineSemaphoreSubmitInfoKHR(1, &semaphoreValueOld, 1, &semaphoreValueNew)};
      queue.submit(1, &submitInfo.get<vk::SubmitInfo>(), vk::Fence());
      device->waitSemaphoresKHR(vk::SemaphoreWaitInfoKHR(vk::SemaphoreWaitFlagsKHR(), 1, &timedSemaphore.get(), &semaphoreValueNew), UINT64_MAX, dll);
      auto end = std::chrono::high_resolution_clock::now();

      uint64_t timestamps[2];
      device->getQueryPoolResults(
          queryPool.get(), 0, /*queryCount=*/2, sizeof(timestamps),
          timestamps, /*Stride=*/sizeof(uint64_t), vk::QueryResultFlagBits::eWait | vk::QueryResultFlagBits::e64);

      auto nanoseconds = std::chrono::nanoseconds((timestamps[1] - timestamps[0]) * static_cast<uint64_t>(props.limits.timestampPeriod));

      if (detailedOutput)
      {
        std::cout << "Host Time taken " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms \n";
        std::cout << "Pure Kernel Time taken " << std::chrono::duration_cast<std::chrono::milliseconds>(nanoseconds).count() << " ms \n";
      }
      else
      {
        std::cout << timesteps << ";" << size << ";" << wgsize << ";Vulkan native " << props.deviceName << ";"
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ";"
                  << std::chrono::duration_cast<std::chrono::milliseconds>(nanoseconds).count() << "\n";
      }

      if (detailedOutput)
      {
        if(outputFile.empty()){
          printStencil(matAHost, size, size, std::cout);
        }else{
          auto outputStream = std::ofstream(outputFile);
          printStencil(matAHost, size, size, outputStream, size, size);
        }
        std::cout << "0 out of " << bufferLength << "\r";
      }

      BAIL_ON_BAD_RESULT(matAHost[(size / 4) * size + (size / 4)] == max ? VK_SUCCESS : VK_ERROR_UNKNOWN);

#pragma omp parallel for schedule(static)
      for (int32_t k = 0; k < bufferLength; k++)
      {
        if (matAHost[k] < min || matAHost[k] > max)
        {
#pragma omp critical
          printf("%d: MatA: %f\n", k, matAHost[k]);
        }
        BAIL_ON_BAD_RESULT(matAHost[k] >= min && matAHost[k] <= max ? VK_SUCCESS : VK_ERROR_UNKNOWN);
        if (detailedOutput && (k % size == 0))
          std::cout << (k / size + 1) << " out of " << bufferLength / size << "\r";
      }
      // device->unmapMemory(memoryA.get());
      if (detailedOutput)
        std::cout << "The results are correct!" << std::endl;
    }
  }
  catch (vk::SystemError &err)
  {
    std::cout << "vk::SystemError: " << err.what() << std::endl;
    exit(-1);
  }
  catch (std::runtime_error &err)
  {
    std::cout << "std::runtime_error: " << err.what() << std::endl;
    exit(-1);
  }
  catch (...)
  {
    std::cout << "unknown error\n";
    exit(-1);
  }

  _aligned_free(matAHost);

  return EXIT_SUCCESS;
}