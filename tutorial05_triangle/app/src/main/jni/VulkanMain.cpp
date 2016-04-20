// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include <cassert>
#include <cstdio>
#include <android/log.h>
#include <android_native_app_glue.h>
#include "vulkan_wrapper.h"
#include "VulkanMain.hpp"

// Android log function wrappers
static const char* kTAG = "Vulkan-Tutorial04";
#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, kTAG, __VA_ARGS__))
#define LOGW(...) \
  ((void)__android_log_print(ANDROID_LOG_WARN, kTAG, __VA_ARGS__))
#define LOGE(...) \
  ((void)__android_log_print(ANDROID_LOG_ERROR, kTAG, __VA_ARGS__))

// Vulkan call wrapper
#define CALL_VK(func)                                                 \
  if (VK_SUCCESS != (func)) {                                         \
    __android_log_print(ANDROID_LOG_ERROR, "Tutorial ",               \
                        "Vulkan error. File[%s], line[%d]", __FILE__, \
                        __LINE__);                                    \
    assert(false);                                                    \
  }

// Global Variables ...
struct VulkanDeviceInfo {
    bool initialized_;

    VkInstance          instance_;
    VkPhysicalDevice    gpuDevice_;
    VkDevice            device_;

    VkSurfaceKHR        surface_;
    VkQueue             queue_;
};
VulkanDeviceInfo  device;

struct VulkanSwapchainInfo {
    VkSwapchainKHR swapchain_;
    uint32_t swapchainLength_;

    VkExtent2D displaySize_;
    VkFormat displayFormat_;

    // array of frame buffers and views
    VkFramebuffer* framebuffers_;
    VkImageView* displayViews_;
};
VulkanSwapchainInfo  swapchain;

struct VulkanBufferInfo {
    VkBuffer vertexBuf;
};
VulkanBufferInfo buffers;

struct VulkanGfxPipelineInfo {
    VkPipelineLayout  layout;
    VkPipelineCache   cache;
    VkPipeline        pipeline;
};
VulkanGfxPipelineInfo gfxPipeline;

struct VulkanRenderInfo {
    VkRenderPass renderPass_;
    VkCommandPool cmdPool_;
    VkCommandBuffer* cmdBuffer_;
    uint32_t         cmdBufferLen_;
    VkSemaphore   semaphore_;
    VkFence       fence_;
};
VulkanRenderInfo render;

// Android Native App pointer...
android_app* androidAppCtx = nullptr;

// Create vulkan device
void CreateVulkanDevice(ANativeWindow* platformWindow,
                        VkApplicationInfo* appInfo) {
  std::vector<const char *> instance_extensions;
  std::vector<const char *> device_extensions;

  instance_extensions.push_back("VK_KHR_surface");
  instance_extensions.push_back("VK_KHR_android_surface");

  device_extensions.push_back("VK_KHR_swapchain");

  // **********************************************************
  // Create the Vulkan instance
  VkInstanceCreateInfo instanceCreateInfo;
  memset(&instanceCreateInfo, 0, sizeof(instanceCreateInfo));
  instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceCreateInfo.pApplicationInfo = appInfo;
  instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(instance_extensions.size());
  instanceCreateInfo.ppEnabledExtensionNames = instance_extensions.data();
  CALL_VK(vkCreateInstance(&instanceCreateInfo, nullptr, &device.instance_));

  VkAndroidSurfaceCreateInfoKHR createInfo;
  memset(&createInfo, 0, sizeof(createInfo));
  createInfo.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
  createInfo.window = platformWindow;
  CALL_VK(vkCreateAndroidSurfaceKHR(device.instance_, &createInfo, nullptr,
                                    &device.surface_));
  // Find one GPU to use:
  // On Android, every GPU device is equal -- supporting graphics/compute/present
  // for this sample, we use the very first GPU device found on the system
  uint32_t  gpuCount = 0;
  CALL_VK(vkEnumeratePhysicalDevices(device.instance_, &gpuCount, nullptr));
  VkPhysicalDevice tmpGpus[gpuCount];
  CALL_VK(vkEnumeratePhysicalDevices(device.instance_, &gpuCount, tmpGpus));
  device.gpuDevice_ = tmpGpus[0];     // Pick up the first GPU Device

  // Create a logical device (vulkan device)
  float priorities[] = { 1.0f, };
  VkDeviceQueueCreateInfo queueCreateInfo;
  memset(&queueCreateInfo, 0, sizeof(queueCreateInfo));
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueCount = 1;
  queueCreateInfo.queueFamilyIndex = 0;
  queueCreateInfo.pQueuePriorities = priorities;

  VkDeviceCreateInfo deviceCreateInfo;
  memset(&deviceCreateInfo, 0, sizeof(deviceCreateInfo));
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.queueCreateInfoCount = 1;
  deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
  deviceCreateInfo.enabledExtensionCount =
          static_cast<uint32_t>(device_extensions.size());
  deviceCreateInfo.ppEnabledExtensionNames = device_extensions.data();
  deviceCreateInfo.pEnabledFeatures = nullptr;

  CALL_VK(vkCreateDevice(device.gpuDevice_, &deviceCreateInfo, nullptr,
                         &device.device_));
  vkGetDeviceQueue(device.device_, 0, 0, &device.queue_);
}

void CreateSwapChain() {
  LOGI("->createSwapChain");
  memset(&swapchain, 0, sizeof(swapchain));

  // **********************************************************
  // Get the surface capabilities because:
  //   - It contains the minimal and max length of the chain, we will need it
  //   - It's necessary to query the supported surface format (R8G8B8A8 for
  //   instance ...)
  VkSurfaceCapabilitiesKHR surfaceCapabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.gpuDevice_, device.surface_,
                                            &surfaceCapabilities);
  // Query the list of supported surface format and choose one we like
  uint32_t formatCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device.gpuDevice_, device.surface_,
                                       &formatCount, nullptr);
  VkSurfaceFormatKHR *formats = new VkSurfaceFormatKHR [formatCount];
  vkGetPhysicalDeviceSurfaceFormatsKHR(device.gpuDevice_, device.surface_,
                                       &formatCount, formats);
  LOGI("Got %d formats", formatCount);

  uint32_t chosenFormat;
  for (chosenFormat = 0; chosenFormat < formatCount; chosenFormat++) {
    if (formats[chosenFormat].format == VK_FORMAT_R8G8B8A8_UNORM)
      break;
  }
  assert(chosenFormat < formatCount);

  swapchain.displaySize_ = surfaceCapabilities.currentExtent;
  swapchain.displayFormat_ = formats[chosenFormat].format;

  // **********************************************************
  // Create a swap chain (here we choose the minimum available number of surface
  // in the chain)
  uint32_t queueFamily = 0;
  VkSwapchainCreateInfoKHR swapchainCreateInfo;
  memset(&swapchainCreateInfo, 0, sizeof(swapchainCreateInfo));
  swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swapchainCreateInfo.surface = device.surface_;
  swapchainCreateInfo.minImageCount = surfaceCapabilities.minImageCount;
  swapchainCreateInfo.imageFormat = formats[chosenFormat].format;
  swapchainCreateInfo.imageColorSpace = formats[chosenFormat].colorSpace;
  swapchainCreateInfo.imageExtent = surfaceCapabilities.currentExtent;
  swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  swapchainCreateInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  swapchainCreateInfo.imageArrayLayers = 1;
  swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  swapchainCreateInfo.queueFamilyIndexCount = 1;
  swapchainCreateInfo.pQueueFamilyIndices = &queueFamily;
  swapchainCreateInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  swapchainCreateInfo.oldSwapchain = VK_NULL_HANDLE;
  swapchainCreateInfo.clipped = VK_FALSE;
  CALL_VK(vkCreateSwapchainKHR(device.device_, &swapchainCreateInfo,
                               nullptr, &swapchain.swapchain_));

  // Get the length of the created swap chain
  CALL_VK(vkGetSwapchainImagesKHR(device.device_, swapchain.swapchain_,
                                  &swapchain.swapchainLength_, nullptr));
  delete [] formats;
  LOGI("<-createSwapChain");
}

void CreateFrameBuffers(VkRenderPass& renderPass,
                        VkImageView depthView = VK_NULL_HANDLE) {

  // query display attachment to swapchain
  uint32_t SwapchainImagesCount = 0;
  CALL_VK(vkGetSwapchainImagesKHR(device.device_, swapchain.swapchain_,
                                  &SwapchainImagesCount, nullptr));
  VkImage* displayImages = new VkImage[SwapchainImagesCount];
  CALL_VK(vkGetSwapchainImagesKHR(device.device_, swapchain.swapchain_,
                                  &SwapchainImagesCount, displayImages));

  // create image view for each swapchain image
  swapchain.displayViews_ = new VkImageView[SwapchainImagesCount];
  for (uint32_t i = 0; i < SwapchainImagesCount; i++) {
    VkImageViewCreateInfo viewCreateInfo;
    memset(&viewCreateInfo, 0, sizeof(viewCreateInfo));
    viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCreateInfo.image = displayImages[i];
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreateInfo.format = swapchain.displayFormat_;
    viewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_R;
    viewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_G;
    viewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_B;
    viewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_A;
    viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCreateInfo.subresourceRange.baseMipLevel = 0;
    viewCreateInfo.subresourceRange.levelCount = 1;
    viewCreateInfo.subresourceRange.baseArrayLayer = 0;
    viewCreateInfo.subresourceRange.layerCount = 1;
    viewCreateInfo.flags = 0;
    CALL_VK(vkCreateImageView(device.device_, &viewCreateInfo, nullptr,
                              &swapchain.displayViews_[i]));
  }
  delete[] displayImages;

  // create a framebuffer from each swapchain image
  swapchain.framebuffers_ = new VkFramebuffer[swapchain.swapchainLength_];
  for (uint32_t i = 0; i < swapchain.swapchainLength_; i++) {
    VkImageView attachments[2] = {
            swapchain.displayViews_[i], depthView,
    };
    VkFramebufferCreateInfo fbCreateInfo;
    fbCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbCreateInfo.renderPass = renderPass;
    fbCreateInfo.layers = 1;
    fbCreateInfo.attachmentCount = 1;  // 2 if using depth
    fbCreateInfo.pAttachments = attachments;
    fbCreateInfo.width = static_cast<uint32_t>(swapchain.displaySize_.width);
    fbCreateInfo.height = static_cast<uint32_t>(swapchain.displaySize_.height);
    fbCreateInfo.attachmentCount = (depthView == VK_NULL_HANDLE ? 1 : 2);
    CALL_VK(vkCreateFramebuffer(device.device_, &fbCreateInfo, nullptr,
                                &swapchain.framebuffers_[i]));
  }
}

// A helper function
bool MapMemoryTypeToIndex(uint32_t typeBits,
                          VkFlags requirements_mask,
                          uint32_t *typeIndex) {
  VkPhysicalDeviceMemoryProperties memoryProperties;
  vkGetPhysicalDeviceMemoryProperties(device.gpuDevice_, &memoryProperties);
  // Search memtypes to find first index with those properties
  for (uint32_t i = 0; i < 32; i++) {
    if ((typeBits & 1) == 1) {
      // Type is available, does it match user properties?
      if ((memoryProperties.memoryTypes[i].propertyFlags &
           requirements_mask) == requirements_mask) {
        *typeIndex = i;
        return true;
      }
    }
    typeBits >>= 1;
  }
  return false;
}

// Create our vertex buffer
bool CreateBuffers(void) {
  // -----------------------------------------------
  // Create the triangle vertex buffer

  // Vertex positions
  const float vertexData[] = {
          -1.0f, -1.0f, 0.0f,
          1.0f, -1.0f, 0.0f,
          0.0f,  1.0f, 0.0f,
  };

  // Create a vertex buffer
  uint32_t queueIdx = 0;
  VkBufferCreateInfo createBufferInfo;
  memset(&createBufferInfo, 0, sizeof(createBufferInfo));
  createBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  createBufferInfo.size = sizeof(vertexData);
  createBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  createBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  createBufferInfo.pQueueFamilyIndices = &queueIdx;
  createBufferInfo.queueFamilyIndexCount = 1;
  CALL_VK(vkCreateBuffer(device.device_, &createBufferInfo, nullptr,
                         &buffers.vertexBuf));

  VkMemoryRequirements memReq;
  vkGetBufferMemoryRequirements(device.device_, buffers.vertexBuf, &memReq);

  VkMemoryAllocateInfo allocInfo;
  memset(&allocInfo, 0,sizeof(allocInfo));
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = sizeof(vertexData);
  allocInfo.memoryTypeIndex = 0;  // Memory type assigned in the next step

  // Assign the proper memory type for that buffer
  allocInfo.allocationSize = memReq.size;
  MapMemoryTypeToIndex(memReq.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                       &allocInfo.memoryTypeIndex);

  // Allocate memory for the buffer
  VkDeviceMemory deviceMemory;
  CALL_VK(vkAllocateMemory(device.device_, &allocInfo, nullptr, &deviceMemory));

  void* data;
  CALL_VK(vkMapMemory(device.device_, deviceMemory, 0, sizeof(vertexData), 0,
                      &data));
  memcpy(data, vertexData, sizeof(vertexData));
  vkUnmapMemory(device.device_, deviceMemory);

  CALL_VK(vkBindBufferMemory(device.device_, buffers.vertexBuf, deviceMemory, 0));
  return true;
}

void DeleteBuffers(void) {
        vkDestroyBuffer(device.device_, buffers.vertexBuf, nullptr);
}

enum ShaderType { VERTEX_SHADER, FRAGMENT_SHADER };
VkResult loadShaderFromFile(const char* filePath, VkShaderModule* shaderOut,
                            ShaderType type) {
  // Read the file
  assert(androidAppCtx);
  AAsset* file = AAssetManager_open(androidAppCtx->activity->assetManager,
                             filePath, AASSET_MODE_BUFFER);
  size_t fileLength = AAsset_getLength(file);

  char* fileContent = new char[fileLength];

  AAsset_read(file, fileContent, fileLength);

  VkShaderModuleCreateInfo shaderModuleCreateInfo;
  memset(&shaderModuleCreateInfo, 0, sizeof(shaderModuleCreateInfo));
  shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shaderModuleCreateInfo.pNext = nullptr;
  shaderModuleCreateInfo.codeSize = fileLength;
  shaderModuleCreateInfo.pCode = (const uint32_t*)fileContent;
  VkResult result = vkCreateShaderModule(
          device.device_, &shaderModuleCreateInfo, nullptr, shaderOut);
  assert(result == VK_SUCCESS);

  delete[] fileContent;

  return result;
}

// Create Graphics Pipeline
VkResult CreateGraphicsPipeline() {
  memset(&gfxPipeline, 0, sizeof(gfxPipeline));
  // Create pipeline layout (empty)
  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  memset(&pipelineLayoutCreateInfo, 0,sizeof(pipelineLayoutCreateInfo));
  pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  CALL_VK(vkCreatePipelineLayout(device.device_, &pipelineLayoutCreateInfo,
                                 nullptr, &gfxPipeline.layout));

  // No dynamic state in that tutorial
  VkPipelineDynamicStateCreateInfo dynamicStateInfo;
  memset(&dynamicStateInfo, 0, sizeof(dynamicStateInfo));
  pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;

  VkShaderModule vertexShader,fragmentShader;
  loadShaderFromFile("shaders/tri.vert.spv", &vertexShader, VERTEX_SHADER);
  loadShaderFromFile("shaders/tri.frag.spv", &fragmentShader, FRAGMENT_SHADER);

  // Specify vertex and fragment shader stages
  VkPipelineShaderStageCreateInfo shaderStages[2] = {
      {
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          nullptr,      // pNext
          0,            // flags
          VK_SHADER_STAGE_VERTEX_BIT, //stage
          vertexShader,               //module
          "main",                     //pName
          nullptr,                    //pSpecializationInfo
      },
      {
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          nullptr,
          0,
          VK_SHADER_STAGE_FRAGMENT_BIT,
          fragmentShader,
          "main",
          nullptr,
      }
  };

  VkViewport viewports {
          0.0f, 0.0f,     // x, y
          (float)swapchain.displaySize_.width,
          (float)swapchain.displaySize_.height,
           0.0f, 1.0f,    // min/max Depth
  };

  VkRect2D scissor {
          {0, 0,},
          swapchain.displaySize_,
  };
  // Specify viewport info
  VkPipelineViewportStateCreateInfo viewportInfo;
  memset(&viewportInfo, 0, sizeof(viewportInfo));
  viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportInfo.viewportCount = 1;
  viewportInfo.pViewports = &viewports;
  viewportInfo.scissorCount = 1;
  viewportInfo.pScissors = &scissor;

  // Specify multisample info
  VkSampleMask sampleMask = ~0u;
  VkPipelineMultisampleStateCreateInfo multisampleInfo;
  memset(&multisampleInfo, 0, sizeof(multisampleInfo));
  multisampleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampleInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multisampleInfo.minSampleShading = 0;
  multisampleInfo.pSampleMask = &sampleMask;

  // Specify color blend state
  VkPipelineColorBlendAttachmentState attachmentStates;
  memset(&attachmentStates, 0, sizeof(attachmentStates));
  attachmentStates.colorWriteMask =
          VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  attachmentStates.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlendInfo;
  memset(&colorBlendInfo, 0,sizeof(colorBlendInfo));
  colorBlendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlendInfo.logicOpEnable = VK_FALSE;
  colorBlendInfo.logicOp = VK_LOGIC_OP_COPY;
  colorBlendInfo.attachmentCount = 1;
  colorBlendInfo.pAttachments = &attachmentStates;

  // Specify rasterizer info
  VkPipelineRasterizationStateCreateInfo rasterInfo;
  memset(&rasterInfo, 0, sizeof(rasterInfo));
  rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
  rasterInfo.cullMode = VK_CULL_MODE_NONE;
  rasterInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterInfo.lineWidth = 1;

  // Specify input assembler state
  VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo;
  memset(&inputAssemblyInfo, 0, sizeof(inputAssemblyInfo));
  inputAssemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  // Specify vertex input state
  VkVertexInputBindingDescription vertex_input_bindings;
  memset(&vertex_input_bindings, 0, sizeof(vertex_input_bindings));
  vertex_input_bindings.binding = 0;
  vertex_input_bindings.stride = 3 * sizeof(float);
  vertex_input_bindings.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  VkVertexInputAttributeDescription vertex_input_attributes[1] = {
      {
          0, // location
          0, // binding
          VK_FORMAT_R32G32B32_SFLOAT,
          0, // offset
      }
  };
  VkPipelineVertexInputStateCreateInfo vertexInputInfo;
  memset(&vertexInputInfo, 0,sizeof(vertexInputInfo));
  vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount = 1;
  vertexInputInfo.pVertexBindingDescriptions = &vertex_input_bindings;
  vertexInputInfo.vertexAttributeDescriptionCount = 1;
  vertexInputInfo.pVertexAttributeDescriptions = vertex_input_attributes;

  // Create the pipeline cache
  VkPipelineCacheCreateInfo pipelineCacheInfo;
  memset(&pipelineCacheInfo, 0, sizeof(pipelineCacheInfo));
  pipelineCacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;

  CALL_VK(vkCreatePipelineCache(device.device_, &pipelineCacheInfo, nullptr,
                                &gfxPipeline.cache));

  // Create the pipeline
  VkGraphicsPipelineCreateInfo pipelineCreateInfo;
  memset(&pipelineCreateInfo, 0, sizeof(pipelineCreateInfo));
  pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineCreateInfo.stageCount = 2;
  pipelineCreateInfo.pStages = shaderStages;
  pipelineCreateInfo.pVertexInputState = &vertexInputInfo;
  pipelineCreateInfo.pInputAssemblyState = &inputAssemblyInfo;
  pipelineCreateInfo.pTessellationState = nullptr;
  pipelineCreateInfo.pViewportState = &viewportInfo;
  pipelineCreateInfo.pRasterizationState = &rasterInfo;
  pipelineCreateInfo.pMultisampleState = &multisampleInfo;
  pipelineCreateInfo.pDepthStencilState = nullptr;
  pipelineCreateInfo.pColorBlendState = &colorBlendInfo;
  pipelineCreateInfo.pDynamicState = &dynamicStateInfo;
  pipelineCreateInfo.layout = gfxPipeline.layout;
  pipelineCreateInfo.renderPass = render.renderPass_;
  pipelineCreateInfo.subpass = 0;
  pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
  pipelineCreateInfo.basePipelineIndex = 0;

  VkResult pipelineResult =
          vkCreateGraphicsPipelines(device.device_, gfxPipeline.cache, 1,
                                    &pipelineCreateInfo, nullptr, &gfxPipeline.pipeline);

  // We don't need the shaders anymore, we can release their memory
  vkDestroyShaderModule(device.device_, vertexShader, nullptr);
  vkDestroyShaderModule(device.device_, fragmentShader, nullptr);

  return pipelineResult;
}

void DeleteGraphicsPipeline(void) {
  if (gfxPipeline.pipeline == VK_NULL_HANDLE)
     return;
  vkDestroyPipeline(device.device_, gfxPipeline.pipeline, nullptr);
  vkDestroyPipelineCache(device.device_,gfxPipeline.cache, nullptr);
  vkDestroyPipelineLayout(device.device_, gfxPipeline.layout, nullptr);
}
// InitVulkan:
//   Initialize Vulkan Context when android application window is created
//   upon return, vulkan is ready to draw frames
bool InitVulkan(android_app* app) {
  androidAppCtx = app;

  if (!InitVulkan()) {
    LOGW("Vulkan is unavailable, install vulkan and re-start");
    return false;
  }

  VkApplicationInfo appInfo;
  memset(&appInfo, 0, sizeof(appInfo));
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pApplicationName = "tutorial05_triangle_window";
  appInfo.pEngineName = "tutorial";

  // create a device
  CreateVulkanDevice(app->window, &appInfo);

  CreateSwapChain();

  // -----------------------------------------------------------------
  // Create render pass
  VkAttachmentDescription attachmentDescriptions;
  memset(&attachmentDescriptions, 0, sizeof(attachmentDescriptions));
  attachmentDescriptions.format = swapchain.displayFormat_;
  attachmentDescriptions.samples = VK_SAMPLE_COUNT_1_BIT;
  attachmentDescriptions.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachmentDescriptions.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachmentDescriptions.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachmentDescriptions.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachmentDescriptions.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  attachmentDescriptions.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference colourReference { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

  VkSubpassDescription subpassDescription;
  memset(&subpassDescription, 0, sizeof(subpassDescription));
  subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpassDescription.flags = 0;
  subpassDescription.colorAttachmentCount = 1;
  subpassDescription.pColorAttachments = &colourReference;

  VkRenderPassCreateInfo renderPassCreateInfo;
  memset(&renderPassCreateInfo, 0, sizeof(renderPassCreateInfo));
  renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassCreateInfo.attachmentCount = 1;
  renderPassCreateInfo.pAttachments = &attachmentDescriptions;
  renderPassCreateInfo.subpassCount = 1;
  renderPassCreateInfo.pSubpasses = &subpassDescription;
  CALL_VK(vkCreateRenderPass(device.device_, &renderPassCreateInfo,
                             nullptr, &render.renderPass_));

  // -----------------------------------------------------------------
  // Create 2 frame buffers.
  CreateFrameBuffers(render.renderPass_);

  CreateBuffers();    // create vertex buffers

  // Create graphics pipeline
  CreateGraphicsPipeline();

  // -----------------------------------------------
  // Create a pool of command buffers to allocate command buffer from
  VkCommandPoolCreateInfo cmdPoolCreateInfo {
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      nullptr,
      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      0
  };
  CALL_VK(vkCreateCommandPool(device.device_, &cmdPoolCreateInfo,
                              nullptr, &render.cmdPool_));

  // Record a command buffer that just clear the screen
  // 1 command buffer draw in 1 framebuffer
  // In our case we need 2 command as we have 2 framebuffer
  render.cmdBufferLen_ = swapchain.swapchainLength_;
  render.cmdBuffer_ = new VkCommandBuffer[swapchain.swapchainLength_];
  for (int bufferIndex = 0; bufferIndex < swapchain.swapchainLength_;
       bufferIndex++) {
    // We start by creating and declare the "beginning" our command buffer
    VkCommandBufferAllocateInfo cmdBufferCreateInfo {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr,
        render.cmdPool_,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        render.cmdBufferLen_,
    };
    CALL_VK(vkAllocateCommandBuffers(device.device_, &cmdBufferCreateInfo,
                                   &render.cmdBuffer_[bufferIndex]));

    VkCommandBufferBeginInfo cmdBufferBeginInfo;
    memset(&cmdBufferBeginInfo, 0, sizeof(cmdBufferBeginInfo));
    cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    CALL_VK(vkBeginCommandBuffer(render.cmdBuffer_[bufferIndex],
                                 &cmdBufferBeginInfo));

    // Now we start a renderpass. Any draw command has to be recorded in a
    // renderpass
    VkClearValue clearVals;
    memset(&clearVals, 0, sizeof(clearVals));
    clearVals.color = {0.0f, 0.34f, 0.90f, 1.0f };
    VkRenderPassBeginInfo renderPassBeginInfo;
    memset(&renderPassBeginInfo, 0, sizeof(renderPassBeginInfo));
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = render.renderPass_;
    renderPassBeginInfo.framebuffer = swapchain.framebuffers_[bufferIndex];
    // renderPassBeginInfo.renderArea.offset = {0, 0};
    renderPassBeginInfo.renderArea.extent = swapchain.displaySize_;
    renderPassBeginInfo.clearValueCount = 1;
    renderPassBeginInfo.pClearValues = &clearVals;

    vkCmdBeginRenderPass(render.cmdBuffer_[bufferIndex], &renderPassBeginInfo,
                         VK_SUBPASS_CONTENTS_INLINE);
    // Bind what is necessary to the command buffer
    vkCmdBindPipeline(render.cmdBuffer_[bufferIndex], VK_PIPELINE_BIND_POINT_GRAPHICS,
                      gfxPipeline.pipeline);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(render.cmdBuffer_[bufferIndex], 0, 1, &buffers.vertexBuf,
                           &offset);

    // Draw Triangle
    vkCmdDraw(render.cmdBuffer_[bufferIndex], 3, 1, 0, 0);

    vkCmdEndRenderPass(render.cmdBuffer_[bufferIndex]);
    CALL_VK(vkEndCommandBuffer(render.cmdBuffer_[bufferIndex]));
  }

  // We need to create a fence to be able, in the main loop, to wait for our
  // draw command(s) to finish before swapping the framebuffers
  VkFenceCreateInfo fenceCreateInfo { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                      nullptr, 0, };
  CALL_VK(vkCreateFence(device.device_, &fenceCreateInfo,
                        nullptr, &render.fence_));

  // We need to create a semaphore to be able to wait, in the main loop, for our
  // framebuffer to be available for us before drawing.
  VkSemaphoreCreateInfo semaphoreCreateInfo {
      VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      nullptr,
      0,
  };
  CALL_VK(vkCreateSemaphore(device.device_, &semaphoreCreateInfo,
                            nullptr, &render.semaphore_));

  device.initialized_ = true;
  return true;
}

// IsVulkanReady():
//    native app poll to see if we are ready to draw...
bool IsVulkanReady(void) {
  return device.initialized_;
}

void DeleteSwapChain() {
  for (int i = 0; i < swapchain.swapchainLength_; i++) {
    vkDestroyFramebuffer(device.device_, swapchain.framebuffers_[i], nullptr);
    vkDestroyImageView(device.device_, swapchain.displayViews_[i], nullptr);
  }
  delete[] swapchain.framebuffers_;
  delete[] swapchain.displayViews_;

  vkDestroySwapchainKHR(device.device_, swapchain.swapchain_, nullptr);
}

void DeleteVulkan() {
  vkFreeCommandBuffers(device.device_, render.cmdPool_,
          render.cmdBufferLen_, render.cmdBuffer_);
  delete[] render.cmdBuffer_;

  vkDestroyCommandPool(device.device_, render.cmdPool_, nullptr);
  vkDestroyRenderPass(device.device_, render.renderPass_, nullptr);
  DeleteSwapChain();
  DeleteGraphicsPipeline();
  DeleteBuffers();

  vkDestroyDevice(device.device_, nullptr);
  vkDestroyInstance(device.instance_, nullptr);

  device.initialized_ = false;
}

// Draw one frame
bool VulkanDrawFrame(void) {
  uint32_t nextIndex;
  // Get the framebuffer index we should draw in
  CALL_VK(vkAcquireNextImageKHR(device.device_, swapchain.swapchain_,
                              UINT64_MAX, render.semaphore_,
                              VK_NULL_HANDLE, &nextIndex));
  CALL_VK(vkResetFences(device.device_, 1, &render.fence_));
  VkSubmitInfo submit_info {
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        1,                   // waitSemaphoreCount
        &render.semaphore_,  // pWaitSemaphores
        nullptr,             // pWaitDstStageMask
        1,                   // commandBufferCount
        .pCommandBuffers = &render.cmdBuffer_[nextIndex],
        0,                   // signalSemaphoreCount
        nullptr              // pSignalSemaphores
  };
  CALL_VK(vkQueueSubmit(device.queue_, 1, &submit_info, render.fence_));
  CALL_VK(vkWaitForFences(device.device_, 1, &render.fence_, VK_TRUE, 100000000));

  LOGI("Drawing frames......");

  VkResult result;
  VkPresentInfoKHR presentInfo;
  memset(&presentInfo, 0, sizeof(presentInfo));
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &swapchain.swapchain_;
  presentInfo.pImageIndices = &nextIndex;
  presentInfo.pResults = &result;

  vkQueuePresentKHR(device.queue_, &presentInfo);
  return true;
}
