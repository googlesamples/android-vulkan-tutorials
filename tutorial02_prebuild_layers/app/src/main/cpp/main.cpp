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

#include <android/log.h>

#include <cassert>
#include <vector>

#include <game-activity/native_app_glue/android_native_app_glue.h>
#include "vulkan_wrapper.h"
#include "TutorialValLayer.hpp"

// Android log function wrappers
static const char* kTAG = "Vulkan-Tutorial02";
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

// Global variables
VkInstance tutorialInstance;
VkPhysicalDevice tutorialGpu;
VkDevice tutorialDevice;
VkSurfaceKHR tutorialSurface;

// We will call this function the window is opened.
// This is where we will initialise everything
bool initialized_ = false;
bool initialize(android_app* app);

// Functions interacting with Android native activity
void android_main(struct android_app* state);
void terminate(void);
void handle_cmd(android_app* app, int32_t cmd);

// typical Android NativeActivity entry function
void android_main(struct android_app* app) {
  app->onAppCmd = handle_cmd;

  int events;
  android_poll_source* source;
  do {
    if (ALooper_pollAll(initialized_ ? 1 : 0, nullptr, &events,
                        (void**)&source) >= 0) {
      if (source != NULL) source->process(app, source);
    }
  } while (app->destroyRequested == 0);
}

bool initialize(android_app* app) {
  // Load Android vulkan and retrieve vulkan API function pointers
  if (!InitVulkan()) {
    LOGE("Vulkan is unavailable, install vulkan and re-start");
    return false;
  }

  VkApplicationInfo appInfo = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pNext = nullptr,
      .pApplicationName = "tutorial02_prebuilt_layers",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "tutorial",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_MAKE_VERSION(1, 1, 0),
  };

  // Enable validation and debug layer/extensions, together with other necessary
  // extensions
  LayerAndExtensions layerUtil;
  std::vector<const char *> layers = {"VK_LAYER_KHRONOS_validation"};
  std::vector<const char*> extensions = {"VK_KHR_surface", "VK_KHR_android_surface"};
  for (auto layerName : layers) {
    // check vulkan sees the layers packed inside this app's APK. layers could also be
    // be pushed to Android with adb on command line, this sample does test that approach
    // in the sense: if you want to use the adb way, you want to use it to enable/disable too
    //               if you want to use the source code way, pack the layer into apk
    //    blending different ways might work, feel free to use and experiment if you prefer.
    assert(layerUtil.isLayerSupported(layerName));
  }
  for (auto extName : extensions) {
    assert(layerUtil.isExtensionSupported(
        extName, VK_NULL_HANDLE, nullptr));
  }

  // Find the supported debug callback extensions and its layers.
  std::pair<const char*, const char*> dbgExt = layerUtil.getDbgReportExtInfo();
  if(dbgExt.second) {
    extensions.push_back(dbgExt.second);
    if (strcmp(dbgExt.first, VULKAN_DRIVER)) {
       bool alreadyIn = false;
       for(auto& name : layers) {
         if (!strcmp(name, dbgExt.first)) {
           alreadyIn = true;
           break;
         }
       }
       if(!alreadyIn)
         layers.push_back(dbgExt.first);
    }
  }

  // Create Vulkan instance, requesting all enabled layers / extensions
  // available on the system
  VkInstanceCreateInfo instanceCreateInfo{
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pNext = nullptr,
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = static_cast<uint32_t>(layers.size()),
      .ppEnabledLayerNames = layers.data(),
      .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
      .ppEnabledExtensionNames = extensions.data(),
  };
  CALL_VK(vkCreateInstance(&instanceCreateInfo, nullptr, &tutorialInstance));

  // Create debug callback obj and connect to Vulkan instance
  layerUtil.hookDbgReportExt(tutorialInstance);

  // Find one GPU to use:
  // On Android, every GPU device is equal -- supporting
  // graphics/compute/present
  // for this sample, we use the very first GPU device found on the system
  uint32_t gpuCount = 0;
  CALL_VK(vkEnumeratePhysicalDevices(tutorialInstance, &gpuCount, nullptr));
  VkPhysicalDevice tmpGpus[gpuCount];
  CALL_VK(vkEnumeratePhysicalDevices(tutorialInstance, &gpuCount, tmpGpus));
  tutorialGpu = tmpGpus[0];  // Pick up the first GPU Device
  
  // check for vulkan info on this GPU device
  VkPhysicalDeviceProperties gpuProperties;
  vkGetPhysicalDeviceProperties(tutorialGpu, &gpuProperties);
  LOGI("Vulkan Physical Device Name: %s", gpuProperties.deviceName);
  LOGI("Vulkan Physical Device Info: apiVersion: %x \n\t driverVersion: %x",
       gpuProperties.apiVersion, gpuProperties.driverVersion);
  LOGI("API Version Supported: %d.%d.%d",
       VK_VERSION_MAJOR(gpuProperties.apiVersion),
       VK_VERSION_MINOR(gpuProperties.apiVersion),
       VK_VERSION_PATCH(gpuProperties.apiVersion));

  VkAndroidSurfaceCreateInfoKHR createInfo{
      .sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
      .pNext = nullptr,
      .flags = 0,
      .window = app->window};
  CALL_VK(vkCreateAndroidSurfaceKHR(tutorialInstance, &createInfo, nullptr,
                                    &tutorialSurface));

  VkSurfaceCapabilitiesKHR surfaceCapabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(tutorialGpu, tutorialSurface,
                                            &surfaceCapabilities);

  LOGI("Vulkan Surface Capabilities:\n");
  LOGI("\timage count: %u - %u\n", surfaceCapabilities.minImageCount,
       surfaceCapabilities.maxImageCount);
  LOGI("\tarray layers: %u\n", surfaceCapabilities.maxImageArrayLayers);
  LOGI("\timage size (now): %dx%d\n", surfaceCapabilities.currentExtent.width,
       surfaceCapabilities.currentExtent.height);
  LOGI("\timage size (extent): %dx%d - %dx%d\n",
       surfaceCapabilities.minImageExtent.width,
       surfaceCapabilities.minImageExtent.height,
       surfaceCapabilities.maxImageExtent.width,
       surfaceCapabilities.maxImageExtent.height);
  LOGI("\tusage: %x\n", surfaceCapabilities.supportedUsageFlags);
  LOGI("\tcurrent transform: %u\n", surfaceCapabilities.currentTransform);
  LOGI("\tallowed transforms: %x\n", surfaceCapabilities.supportedTransforms);
  LOGI("\tcomposite alpha flags: %u\n", surfaceCapabilities.currentTransform);

  // Find a GFX queue family
  uint32_t queueFamilyCount;
  vkGetPhysicalDeviceQueueFamilyProperties(tutorialGpu, &queueFamilyCount,
                                           nullptr);
  assert(queueFamilyCount);
  std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(tutorialGpu, &queueFamilyCount,
                                           queueFamilyProperties.data());

  uint32_t queueFamilyIndex;
  for (queueFamilyIndex = 0; queueFamilyIndex < queueFamilyCount;
       queueFamilyIndex++) {
    if (queueFamilyProperties[queueFamilyIndex].queueFlags &
        VK_QUEUE_GRAPHICS_BIT) {
      break;
    }
  }
  assert(queueFamilyIndex < queueFamilyCount);

  // Create a logical device from GPU we picked
  float priorities[] = {
      1.0f,
  };
  VkDeviceQueueCreateInfo queueCreateInfo{
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .queueFamilyIndex = queueFamilyIndex,
      .queueCount = 1,
      // Send nullptr for queue priority, instead of the priority array
      // so debug extension could catch the bug and call back app's debug
      // function
      .pQueuePriorities = nullptr,  // priorities,
  };

  VkDeviceCreateInfo deviceCreateInfo{
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = nullptr,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queueCreateInfo,
      .enabledExtensionCount = 0,
      .ppEnabledExtensionNames = nullptr,
      .pEnabledFeatures = nullptr,
  };

  CALL_VK(
      vkCreateDevice(tutorialGpu, &deviceCreateInfo, nullptr, &tutorialDevice));
  initialized_ = true;
  return true;
}

void terminate(void) {
  vkDestroySurfaceKHR(tutorialInstance, tutorialSurface, nullptr);
  vkDestroyDevice(tutorialDevice, nullptr);
  vkDestroyInstance(tutorialInstance, nullptr);

  initialized_ = false;
}

// Process the next main command.
void handle_cmd(android_app* app, int32_t cmd) {
  switch (cmd) {
    case APP_CMD_INIT_WINDOW:
      // The window is being shown, get it ready.
      initialize(app);
      break;
    case APP_CMD_TERM_WINDOW:
      // The window is being hidden or closed, clean it up.
      terminate();
      break;
    default:
      LOGI("event not handled: %d", cmd);
  }
}
