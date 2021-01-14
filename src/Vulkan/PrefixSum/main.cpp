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

#include "../../Headers/Prefix.h"
#include "vulkan/vulkan.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <math.h>

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

template <typename T>
uint32_t conv(T value) { return static_cast<uint32_t>(value); }

int main(int argc, const char *const argv[])
{

  int32_t size = 1024;
  int32_t device = 0;
  bool detailedOutput = true;
  if (argc > 1)
  {
    if (strcmp(argv[1], "-h") == 0)
    {
      std::cout << "Usage of Prefix Sum: main.exe [size=" << size << "] [device=" << device << "] [-v]\n";
      std::cout << "Size must be smaller than " << (4194304) << " due to algorithm 1024*1024*4\n";
      std::cout << "If -v is preset only the time will be in the output" << std::endl;
      return 0;
    }
    size = atoi(argv[1]);
    if (argc > 2)
      device = atoi(argv[2]);
    if (argc > 3 && strcmp(argv[3], "-v") == 0)
      detailedOutput = false;
  }

  if (size <= 0 || size > 4194304)
    size = 1024;

  int32_t wgsize = (size + 1) / 2;

  const int64_t bufferLength = static_cast<int64_t>(size);
  const uint64_t bufferSize = sizeof(float_t) * bufferLength;
  // we are going to need two buffers from this one memory
  const vk::DeviceSize memorySize = bufferSize;

  auto vecAHost = static_cast<float_t *>(_aligned_malloc(size * sizeof(float_t), 4096));
  auto vecBHost = static_cast<float_t *>(_aligned_malloc(size * sizeof(float_t), 4096));

#pragma omp parallel for schedule(static)
  for (int32_t k = 0; k < size; k++)
  {
    // vecAHost[size - k - 1] = static_cast<float>(k);
    vecAHost[k] = 1.0f;
    vecBHost[k] = -1.0f;
  }

  try
  {
    auto start = std::chrono::high_resolution_clock::now();

    // initialize the vk::ApplicationInfo structure
    vk::ApplicationInfo applicationInfo("VecAdd", 1, "Vulkan.hpp", 1, VK_API_VERSION_1_1);

    // initialize the vk::InstanceCreateInfo
    std::vector<char *> layers = {
        // "VK_LAYER_LUNARG_api_dump",
        // "VK_LAYER_KHRONOS_validation"
    };
    vk::InstanceCreateInfo instanceCreateInfo({}, &applicationInfo, conv(layers.size()), layers.data());

    // create a UniqueInstance
    vk::UniqueInstance instance = vk::createInstanceUnique(instanceCreateInfo);

    auto physicalDevices = instance->enumeratePhysicalDevices();

    auto physicalDevice = physicalDevices[device];
    // for (auto &physicalDevice : physicalDevices)
    {

      auto props = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>();

      // get the QueueFamilyProperties of the first PhysicalDevice
      std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
      uint32_t computeQueueFamilyIndex = 0;

      // get the best index into queueFamiliyProperties which supports compute and stuff
      BAIL_ON_BAD_RESULT(vkGetBestComputeQueueNPH(physicalDevice, computeQueueFamilyIndex));

      std::vector<char *> extensions = {
          "VK_EXT_external_memory_host"
          //,"VK_KHR_shader_float16_int8"
      };
      // create a UniqueDevice
      float queuePriority = 0.0f;

      vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), conv(computeQueueFamilyIndex), 1, &queuePriority);
      vk::DeviceCreateInfo deviceCreateInfo(vk::DeviceCreateFlags(), 1, &deviceQueueCreateInfo, 0, nullptr, conv(extensions.size()), extensions.data());
      vk::UniqueDevice device = physicalDevice.createDeviceUnique(deviceCreateInfo);

      if (detailedOutput)
        std::cout << "Device found: " << props.get<vk::PhysicalDeviceProperties2>().properties.deviceName << std::endl;

      vk::DispatchLoaderDynamic dll(instance.get(), vkGetInstanceProcAddr, device.get(), vkGetDeviceProcAddr);

      auto memoryProperties2 = physicalDevice.getMemoryProperties2();

      vk::PhysicalDeviceMemoryProperties const &memoryProperties = memoryProperties2.memoryProperties;

      int wgsize2 = 1;

      // Workgroup size must be a power of 2 due to the algorithm
      // try use upper bound of power of 2
      wgsize = static_cast<int>(pow(2.0f, ceil(log2((float)wgsize))));

      // check upper limit of workgroup size with device limits
      const int deviceLimit = props.get<vk::PhysicalDeviceProperties2>().properties.limits.maxComputeWorkGroupSize[0];
      if (wgsize > deviceLimit)
      {
        // int temp = wgsize;
        // take lower bound of power of 2 of device limit
        wgsize = static_cast<int>(pow(2.0f, floor(log2((float)deviceLimit))));
        // caluclate workgroup size for inner presum task
        // wgsize2 = (temp + deviceLimit - 1) / wgsize;
        wgsize2 = static_cast<int>(pow(2.0f, ceil(log2(((float)size / wgsize / 2)))));
      }

      uint64_t partSumBufferSize = sizeof(float_t) * wgsize2;

      // set memoryTypeIndex to an invalid entry in the properties.memoryTypes array
      uint32_t memoryTypeIndexStaging = VK_MAX_MEMORY_TYPES;
      uint32_t memoryTypeIndexDevice = VK_MAX_MEMORY_TYPES;

      auto extMemProperties = device->getMemoryHostPointerPropertiesEXT(
          vk::ExternalMemoryHandleTypeFlagBitsKHR::eHostAllocationEXT, vecAHost,
          dll);

      for (uint32_t k = 0; k < memoryProperties.memoryTypeCount; k++)
      {
        if ((vk::MemoryPropertyFlagBits::eDeviceLocal & memoryProperties.memoryTypes[k].propertyFlags) &&
            (memorySize * 3ull < memoryProperties.memoryHeaps[memoryProperties.memoryTypes[k].heapIndex].size))
        {
          memoryTypeIndexDevice = k;
        }
        if (vk::MemoryPropertyFlagBits::eHostVisible & memoryProperties.memoryTypes[k].propertyFlags && vk::MemoryPropertyFlagBits::eHostCoherent & memoryProperties.memoryTypes[k].propertyFlags &&
            (memorySize * 2ull < memoryProperties.memoryHeaps[memoryProperties.memoryTypes[k].heapIndex].size) && (extMemProperties.memoryTypeBits & (1 << k)) != 0)
        {
          memoryTypeIndexStaging = k;
        }
      }

      BAIL_ON_BAD_RESULT(memoryTypeIndexStaging == VK_MAX_MEMORY_TYPES ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS);
      BAIL_ON_BAD_RESULT(memoryTypeIndexDevice == VK_MAX_MEMORY_TYPES ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS);

      auto inA_buffer = device->createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), memorySize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive));
      auto outB_buffer = device->createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), memorySize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive));

      auto inA_bufferDevice = device->createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), memorySize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive));
      auto outB_bufferDevice = device->createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), memorySize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive));
      auto partSum_bufferDevice = device->createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), partSumBufferSize, vk::BufferUsageFlagBits::eStorageBuffer, vk::SharingMode::eExclusive));
      auto temp_bufferDevice = device->createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), wgsize2 * wgsize * 2 * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer, vk::SharingMode::eExclusive));

      auto partSumBufferSizeMem = std::max(partSumBufferSize, device->getBufferMemoryRequirements(partSum_bufferDevice.get()).size);

      auto memoryCDevice = device->allocateMemoryUnique(vk::MemoryAllocateInfo(partSumBufferSizeMem, memoryTypeIndexDevice));

      vk::StructureChain<vk::MemoryAllocateInfo, vk::ImportMemoryHostPointerInfoEXT>
          allocInfoA = {
              vk::MemoryAllocateInfo(device->getBufferMemoryRequirements(inA_buffer.get()).size, memoryTypeIndexStaging),
              vk::ImportMemoryHostPointerInfoEXT(vk::ExternalMemoryHandleTypeFlagBitsKHR::eHostAllocationEXT, vecAHost)};
      vk::StructureChain<vk::MemoryAllocateInfo, vk::ImportMemoryHostPointerInfoEXT>
          allocInfoB = {
              vk::MemoryAllocateInfo(device->getBufferMemoryRequirements(outB_buffer.get()).size, memoryTypeIndexStaging),
              vk::ImportMemoryHostPointerInfoEXT(vk::ExternalMemoryHandleTypeFlagBitsKHR::eHostAllocationEXT, vecBHost)};

      auto memoryA = device->allocateMemoryUnique(allocInfoA.get<vk::MemoryAllocateInfo>());
      auto memoryB = device->allocateMemoryUnique(allocInfoB.get<vk::MemoryAllocateInfo>());

      auto memoryADevice = device->allocateMemoryUnique(vk::MemoryAllocateInfo(device->getBufferMemoryRequirements(inA_bufferDevice.get()).size, memoryTypeIndexDevice));
      auto memoryBDevice = device->allocateMemoryUnique(vk::MemoryAllocateInfo(device->getBufferMemoryRequirements(outB_bufferDevice.get()).size, memoryTypeIndexDevice));
      auto memoryTempDevice = device->allocateMemoryUnique(vk::MemoryAllocateInfo(device->getBufferMemoryRequirements(temp_bufferDevice.get()).size, memoryTypeIndexDevice));

      device->bindBufferMemory(inA_buffer.get(), memoryA.get(), 0);
      device->bindBufferMemory(outB_buffer.get(), memoryB.get(), 0);
      device->bindBufferMemory(inA_bufferDevice.get(), memoryADevice.get(), 0);
      device->bindBufferMemory(outB_bufferDevice.get(), memoryBDevice.get(), 0);
      device->bindBufferMemory(partSum_bufferDevice.get(), memoryCDevice.get(), 0);
      device->bindBufferMemory(temp_bufferDevice.get(), memoryTempDevice.get(), 0);

      // create a DescriptorSetLayout
      std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBinding{
          {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
          {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
          {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
          {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}};
      vk::UniqueDescriptorSetLayout descriptorSetLayout = device->createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), conv(descriptorSetLayoutBinding.size()), descriptorSetLayoutBinding.data()));
      vk::UniqueDescriptorSetLayout descriptorSetLayout2 = device->createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), conv(descriptorSetLayoutBinding.size() - 2), descriptorSetLayoutBinding.data()));

      std::ifstream myfile;
      myfile.open("shaders/Prefix.comp.spv", std::ios::ate | std::ios::binary);
      if (!myfile.is_open())
      {
        myfile.open("../../shaders/Prefix.comp.spv", std::ios::ate | std::ios::binary);
        if (!myfile.is_open())
        {
          std::cout << "File Prefix.comp.spv not found" << std::endl;
          return EXIT_FAILURE;
        }
      }

      if (detailedOutput)
        std::cout << "Running Prefix Sum on " << size << " using workgroupsize of " << wgsize << " and " << wgsize2 << " steps\n";

      auto fileSize = myfile.tellg();
      std::vector<unsigned int> shader_spv(fileSize / sizeof(unsigned int));
      myfile.seekg(0);
      myfile.read(reinterpret_cast<char *>(shader_spv.data()), fileSize);
      myfile.close();

      auto shaderModule = device->createShaderModuleUnique(vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), shader_spv.size() * sizeof(unsigned int), shader_spv.data()));

      myfile.open("shaders/PartSumAdd.comp.spv", std::ios::ate | std::ios::binary);
      if (!myfile.is_open())
      {
        myfile.open("../../shaders/PartSumAdd.comp.spv", std::ios::ate | std::ios::binary);
        if (!myfile.is_open())
        {
          std::cout << "File PartSumAdd.comp.spv not found" << std::endl;
          return EXIT_FAILURE;
        }
      }

      fileSize = myfile.tellg();
      shader_spv.resize(fileSize / sizeof(unsigned int));
      myfile.seekg(0);
      myfile.read(reinterpret_cast<char *>(shader_spv.data()), fileSize);
      myfile.close();

      auto shaderModule2 = device->createShaderModuleUnique(vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), shader_spv.size() * sizeof(unsigned int), shader_spv.data()));

      std::vector<vk::PushConstantRange> pushConstants = {
          {vk::ShaderStageFlagBits::eCompute, 0, conv(1 * sizeof(uint32_t))},
      };
      // create a PipelineLayout using that DescriptorSetLayout
      vk::UniquePipelineLayout pipelineLayout = device->createPipelineLayoutUnique(
          vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(),
                                       1,
                                       &descriptorSetLayout.get(),
                                       conv(pushConstants.size()), pushConstants.data()));
      vk::UniquePipelineLayout pipelineLayout2 = device->createPipelineLayoutUnique(
          vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(),
                                       1,
                                       &descriptorSetLayout2.get(),
                                       conv(pushConstants.size()), pushConstants.data()));

      // set Workgroupsize
      std::vector<uint32_t> Values = {conv(wgsize)};
      std::vector<vk::SpecializationMapEntry> Entries;

      for (size_t i = 0; i < Values.size(); i++)
      {
        Entries.emplace_back(100u + uint32_t(i), uint32_t(sizeof(uint32_t) * i), sizeof(uint32_t));
      }
      // set Workgroup size again as Constant ID to be able to allocate local memory in shader
      // (seems to be not possible by directly using gl_WorkGroupSize.x)
      Values.emplace_back(static_cast<uint32_t>(wgsize * 2));
      Entries.emplace_back(2, /*offset=*/uint32_t(sizeof(uint32_t)), sizeof(uint32_t));

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

      computePipelineInfo = vk::ComputePipelineCreateInfo(
          vk::PipelineCreateFlags(),
          vk::PipelineShaderStageCreateInfo(
              vk::PipelineShaderStageCreateFlags(),
              vk::ShaderStageFlagBits::eCompute,
              shaderModule2.get(),
              "main", &SpecializationInfo),
          pipelineLayout2.get());

      auto pipeline3 = device->createComputePipelineUnique(nullptr, computePipelineInfo);

      Values[1] = static_cast<uint32_t>(wgsize2);
      Values[0] = Values[1] < 2 ? 1 : Values[1] / 2;

      computePipelineInfo = vk::ComputePipelineCreateInfo(
          vk::PipelineCreateFlags(),
          vk::PipelineShaderStageCreateInfo(
              vk::PipelineShaderStageCreateFlags(),
              vk::ShaderStageFlagBits::eCompute,
              shaderModule.get(),
              "main", &SpecializationInfo),
          pipelineLayout.get());

      auto pipeline2 = device->createComputePipelineUnique(nullptr, computePipelineInfo);

      std::vector<vk::DescriptorPoolSize> descriptorPoolSizes = {
          vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, conv(descriptorSetLayoutBinding.size())),
          vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, conv(descriptorSetLayoutBinding.size())),
          vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, conv(descriptorSetLayoutBinding.size() - 2))};

      auto descriptorPool = device->createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet), 3, conv(descriptorPoolSizes.size()), descriptorPoolSizes.data()));

      std::vector<vk::DescriptorSetLayout> DSLayouts = {
          descriptorSetLayout.get(),
          descriptorSetLayout.get(),
          descriptorSetLayout2.get()};
      auto descriptorSets = device->allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(descriptorPool.get(), conv(DSLayouts.size()), DSLayouts.data()));

      vk::DescriptorBufferInfo inA_descriptorBufferInfo(inA_bufferDevice.get(), 0, VK_WHOLE_SIZE);
      vk::DescriptorBufferInfo outB_descriptorBufferInfo(outB_bufferDevice.get(), 0, VK_WHOLE_SIZE);
      vk::DescriptorBufferInfo partSum_descriptorBufferInfo(partSum_bufferDevice.get(), 0, VK_WHOLE_SIZE);
      vk::DescriptorBufferInfo temp_descriptorBufferInfo(temp_bufferDevice.get(), 0, VK_WHOLE_SIZE);

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
              &outB_descriptorBufferInfo,
              nullptr},
          vk::WriteDescriptorSet{
              descriptorSets[0].get(),
              2,
              0,
              1,
              vk::DescriptorType::eStorageBuffer,
              nullptr,
              &partSum_descriptorBufferInfo,
              nullptr},
          vk::WriteDescriptorSet{
              descriptorSets[0].get(),
              3,
              0,
              1,
              vk::DescriptorType::eStorageBuffer,
              nullptr,
              &temp_descriptorBufferInfo,
              nullptr},
          vk::WriteDescriptorSet{
              descriptorSets[1].get(),
              0,
              0,
              1,
              vk::DescriptorType::eStorageBuffer,
              nullptr,
              &partSum_descriptorBufferInfo,
              nullptr},
          vk::WriteDescriptorSet{
              descriptorSets[1].get(),
              1,
              0,
              1,
              vk::DescriptorType::eStorageBuffer,
              nullptr,
              &partSum_descriptorBufferInfo,
              nullptr},
          vk::WriteDescriptorSet{
              descriptorSets[1].get(),
              2,
              0,
              1,
              vk::DescriptorType::eStorageBuffer,
              nullptr,
              &inA_descriptorBufferInfo,
              nullptr},
          vk::WriteDescriptorSet{
              descriptorSets[1].get(),
              3,
              0,
              1,
              vk::DescriptorType::eStorageBuffer,
              nullptr,
              &temp_descriptorBufferInfo,
              nullptr},
          vk::WriteDescriptorSet{
              descriptorSets[2].get(),
              0,
              0,
              1,
              vk::DescriptorType::eStorageBuffer,
              nullptr,
              &outB_descriptorBufferInfo,
              nullptr},
          vk::WriteDescriptorSet{
              descriptorSets[2].get(),
              1,
              0,
              1,
              vk::DescriptorType::eStorageBuffer,
              nullptr,
              &partSum_descriptorBufferInfo,
              nullptr}};

      device->updateDescriptorSets(writeSets, /*copies*/ nullptr);

      auto commandPool = device->createCommandPoolUnique(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlags(), computeQueueFamilyIndex));

      auto commandBuffers = device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(commandPool.get(), vk::CommandBufferLevel::ePrimary, 3));

      std::array<vk::BufferCopy, 1> region = {vk::BufferCopy(0, 0, memorySize)};
      std::array<vk::BufferCopy, 1> region2 = {vk::BufferCopy(0, 0, partSumBufferSize)};

      std::array<const uint32_t, 1> pushConstants1 = {conv(size)};
      std::array<const uint32_t, 1> pushConstants2 = {conv(wgsize2)};

      auto queryPool =
          device->createQueryPoolUnique(vk::QueryPoolCreateInfo(
              vk::QueryPoolCreateFlags(), vk::QueryType::eTimestamp, 6));

      commandBuffers[0]->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)));
      commandBuffers[0]->resetQueryPool(queryPool.get(), 0, 6);
      commandBuffers[0]->copyBuffer(inA_buffer.get(), inA_bufferDevice.get(), region);
      commandBuffers[0]->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                                        queryPool.get(), 0);
      commandBuffers[0]->bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.value.get());
      commandBuffers[0]->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, 1, &descriptorSets[0].get(), 0, nullptr);
      commandBuffers[0]->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, conv(pushConstants1.size() * sizeof(uint32_t)), pushConstants1.data());
      commandBuffers[0]->dispatch(wgsize2, 1, 1);
      commandBuffers[0]->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                                        queryPool.get(), 3);
      commandBuffers[0]->end();

      commandBuffers[1]->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)));
      commandBuffers[1]->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                                        queryPool.get(), 1);
      commandBuffers[1]->bindPipeline(vk::PipelineBindPoint::eCompute, pipeline2.value.get());
      commandBuffers[1]->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, 1, &descriptorSets[1].get(), 0, nullptr);
      commandBuffers[1]->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, conv(pushConstants2.size() * sizeof(uint32_t)), pushConstants2.data());
      commandBuffers[1]->dispatch(1, 1, 1);
      commandBuffers[1]->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                                        queryPool.get(), 4);
      commandBuffers[1]->end();

      commandBuffers[2]->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)));
      commandBuffers[2]->bindPipeline(vk::PipelineBindPoint::eCompute, pipeline3.value.get());
      commandBuffers[2]->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout2.get(), 0, 1, &descriptorSets[2].get(), 0, nullptr);
      commandBuffers[2]->pushConstants(pipelineLayout2.get(), vk::ShaderStageFlagBits::eCompute, 0, conv(pushConstants1.size() * sizeof(uint32_t)), pushConstants1.data());
      commandBuffers[2]->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                                        queryPool.get(), 2);
      commandBuffers[2]->dispatch(wgsize2, 1, 1);
      commandBuffers[2]->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                                        queryPool.get(), 5);
      commandBuffers[2]->copyBuffer(outB_bufferDevice.get(), outB_buffer.get(), region);
      commandBuffers[2]->end();

      auto queue = device->getQueue(computeQueueFamilyIndex, 0);

      auto Sem1 = device->createSemaphoreUnique(vk::SemaphoreCreateInfo(vk::SemaphoreCreateFlags()));
      auto Sem2 = device->createSemaphoreUnique(vk::SemaphoreCreateInfo(vk::SemaphoreCreateFlags()));

      const vk::PipelineStageFlags wait = vk::PipelineStageFlagBits::eTopOfPipe;

      std::array<vk::SubmitInfo, 3> submitInfos = {
          vk::SubmitInfo(0, nullptr, nullptr, 1, &commandBuffers[0].get(), 1, &Sem1.get()),
          vk::SubmitInfo(1, &Sem1.get(), &wait, 1, &commandBuffers[1].get(), 1, &Sem2.get()),
          vk::SubmitInfo(1, &Sem2.get(), &wait, 1, &commandBuffers[2].get(), 0, nullptr)};

      // vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer.get(), 0, nullptr);
      auto Fence = device->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlags()));
      queue.submit(submitInfos, Fence.get());

      auto result = device->waitForFences(1, &Fence.get(), true, UINT64_MAX);
      if (result != vk::Result::eSuccess)
        vk::throwResultException(result, "Fence error");

      auto end = std::chrono::high_resolution_clock::now();

      uint64_t timestamps[6];
      device->getQueryPoolResults(
          queryPool.get(), 0, /*queryCount=*/6, sizeof(timestamps),
          timestamps, /*Stride=*/sizeof(uint64_t), vk::QueryResultFlagBits::eWait | vk::QueryResultFlagBits::e64);

      auto nanoseconds = std::chrono::nanoseconds((timestamps[3] + timestamps[4] + timestamps[5] - timestamps[0] - timestamps[1] - timestamps[2]) * static_cast<uint64_t>(props.get<vk::PhysicalDeviceProperties2>().properties.limits.timestampPeriod));
      if (detailedOutput)
      {
        std::cout << "Host Time taken " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us \n";
        std::cout << "Pure Kernel Time taken " << std::chrono::duration_cast<std::chrono::microseconds>(nanoseconds).count() << " us \n";
      }
      else
      {
        std::cout << size << ";Vulkan native " << props.get<vk::PhysicalDeviceProperties2>().properties.deviceName << ";"
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << ";"
                  << std::chrono::duration_cast<std::chrono::microseconds>(nanoseconds).count() << "\n";
      }

      if (detailedOutput)
        std::cout << "0 out of " << bufferLength << "\r";

      float temp = 0.0f;
      for (int32_t k = size - 1, i = 0; k >= 0; k--, i++)
      {
        // if (!floatCompare(vecBHost[i],temp))
        if (vecBHost[i] != temp)
        {
          printf("%d: expected: %f\t\tactual: %f\n", i, temp, vecBHost[i]);
        }
        BAIL_ON_BAD_RESULT(floatCompare(vecBHost[i], temp) ? VK_SUCCESS : VK_ERROR_OUT_OF_HOST_MEMORY);
        // temp += k;
        temp += 1.0f;
        if (detailedOutput && i % 1000 == 0)
          std::cout << (i + 1) << " out of " << size << "\r";
      }
      // device->unmapMemory(memoryB.get());
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

  _aligned_free(vecBHost);
  _aligned_free(vecAHost);

  return EXIT_SUCCESS;
}