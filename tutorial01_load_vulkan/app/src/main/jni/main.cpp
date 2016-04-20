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
#include <android/log.h>
#include <android_native_app_glue.h>
#include "vulkan_wrapper.h"

// Android log function wrappers
static const char* kTAG = "Vulkan-Tutorial01";
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
void terminate();
void handle_cmd(android_app* app, int32_t cmd);



// typical Android NativeActivity entry function
void android_main(struct android_app* app) {
  app_dummy();

  app->onAppCmd = handle_cmd;

  int events;
  android_poll_source* source;
  do {
    if (ALooper_pollAll(initialized_? 1: 0, nullptr, &events, (void**)&source) >= 0) {
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

  VkApplicationInfo appInfo;
  memset(&appInfo, 0, sizeof(appInfo));
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pApplicationName = "tutorial01_load_vulkan";
  appInfo.pEngineName = "tutorial";

  // prepare necessary extensions: Vulkan on Android need these to function
  std::vector<const char *> instanceExt, deviceExt;
  instanceExt.push_back("VK_KHR_surface");
  instanceExt.push_back("VK_KHR_android_surface");
  deviceExt.push_back("VK_KHR_swapchain");

  // Create the Vulkan instance
  VkInstanceCreateInfo instanceCreateInfo;
  memset(&instanceCreateInfo, 0, sizeof(instanceCreateInfo));
  instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceCreateInfo.pApplicationInfo = &appInfo;
  instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExt.size());
  instanceCreateInfo.ppEnabledExtensionNames = instanceExt.data();
  CALL_VK(vkCreateInstance(&instanceCreateInfo, nullptr, &tutorialInstance));

  // if we create a surface, we need the surface extension
  VkAndroidSurfaceCreateInfoKHR createInfo;
  memset(&createInfo, 0, sizeof(createInfo));
  createInfo.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
  createInfo.window = app->window;
  CALL_VK(vkCreateAndroidSurfaceKHR(tutorialInstance, &createInfo, nullptr,
                                    &tutorialSurface));

  // Find one GPU to use:
  // On Android, every GPU device is equal -- supporting graphics/compute/present
  // for this sample, we use the very first GPU device found on the system
  uint32_t  gpuCount = 0;
  CALL_VK(vkEnumeratePhysicalDevices(tutorialInstance, &gpuCount, nullptr));
  VkPhysicalDevice tmpGpus[gpuCount];
  CALL_VK(vkEnumeratePhysicalDevices(tutorialInstance, &gpuCount, tmpGpus));
  tutorialGpu = tmpGpus[0];     // Pick up the first GPU Device

  // check for vulkan info on this GPU device
  VkPhysicalDeviceProperties  gpuProperties;
  vkGetPhysicalDeviceProperties(tutorialGpu, &gpuProperties);
  LOGI("Vulkan Physical Device Name: %s", gpuProperties.deviceName);
  LOGI("Vulkan Physical Device Info: apiVersion: %x \n\t driverVersion: %x",
       gpuProperties.apiVersion, gpuProperties.driverVersion);
  LOGI("API Version Supported: %d.%d.%d",
           VK_VERSION_MAJOR(gpuProperties.apiVersion),
           VK_VERSION_MINOR(gpuProperties.apiVersion),
           VK_VERSION_PATCH(gpuProperties.apiVersion));

  VkSurfaceCapabilitiesKHR surfaceCapabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(tutorialGpu,
                                            tutorialSurface,
                                            &surfaceCapabilities);

  LOGI("Vulkan Surface Capabilities:\n");
  LOGI("\timage count: %u - %u\n", surfaceCapabilities.minImageCount,
       surfaceCapabilities.maxImageCount);
  LOGI("\tarray layers: %u\n", surfaceCapabilities.maxImageArrayLayers);
  LOGI("\timage size (now): %dx%d\n",
       surfaceCapabilities.currentExtent.width,
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

  // Create a logical device from GPU we picked
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
  deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExt.size());
  deviceCreateInfo.ppEnabledExtensionNames = deviceExt.data();
  deviceCreateInfo.pEnabledFeatures = nullptr;

  CALL_VK(vkCreateDevice(tutorialGpu, &deviceCreateInfo, nullptr,
                         &tutorialDevice));
  initialized_ = true;
  return 0;
}

void terminate() {

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


