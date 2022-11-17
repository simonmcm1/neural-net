#include "Context.h"
#include "../Logging.h"

#define DEBUG (!NDEBUG)
//#define DEBUG 1

static VKAPI_ATTR VkBool32 VKAPI_CALL debugMessageCallback(
	VkDebugReportFlagsEXT flags,
	VkDebugReportObjectTypeEXT objectType,
	uint64_t object,
	size_t location,
	int32_t messageCode,
	const char* pLayerPrefix,
	const char* pMessage,
	void* pUserData)
{
	LOG_DEBUG("[VALIDATION]: {} - {}\n", pLayerPrefix, pMessage);
	return VK_FALSE;
}

void Context::open() 
{
	const char* app_name = "Vulkan headless example";
	vk::ApplicationInfo appInfo("Compute", 0, "Example", 0, VK_API_VERSION_1_2);

	vk::InstanceCreateInfo instanceCreateInfo({}, &appInfo);

	uint32_t layerCount = 0;
	const char* validationLayers[] = { "VK_LAYER_KHRONOS_validation" };
	layerCount = 1;
	std::vector<const char*> instanceExtensions = { "VK_KHR_get_physical_device_properties2"};
#if DEBUG
	// Check if layers are available
	//uint32_t instanceLayerCount;
	//vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr);
	//std::vector<VkLayerProperties> instanceLayers(instanceLayerCount);
	//vkEnumerateInstanceLayerProperties(&instanceLayerCount, instanceLayers.data());
	std::vector<vk::LayerProperties> instanceLayers = vk::enumerateInstanceLayerProperties();

	bool layersAvailable = true;
	for (auto layerName : validationLayers) {
		bool layerAvailable = false;
		for (auto instanceLayer : instanceLayers) {
			if (strcmp(instanceLayer.layerName, layerName) == 0) {
				layerAvailable = true;
				break;
			}
		}
		if (!layerAvailable) {
			layersAvailable = false;
			break;
		}
	}

	if (layersAvailable) {
		instanceExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
		instanceCreateInfo.ppEnabledLayerNames = validationLayers;
		instanceCreateInfo.enabledLayerCount = layerCount;
	}
#endif
	instanceCreateInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
	instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
	instance = vk::createInstance(instanceCreateInfo);

#if DEBUG
	if (layersAvailable) {
		const vk::DebugReportCallbackCreateInfoEXT debugReportCreateInfo(
			vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning,
			debugMessageCallback);

		PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT = reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(instance.getProcAddr("vkCreateDebugReportCallbackEXT"));

		// We have to explicitly load this function.
		assert(vkCreateDebugReportCallbackEXT);
		VkDebugReportCallbackEXT tmp;
		VkDebugReportCallbackCreateInfoEXT tmpinfo = debugReportCreateInfo;
		vkCreateDebugReportCallbackEXT(instance, &tmpinfo, nullptr, &tmp);
		debugReportCallback = tmp;
	}
#endif

	// Pick a discrete physical device if we can
	std::vector<vk::PhysicalDevice> physical_devices = instance.enumeratePhysicalDevices();
	physical_device = physical_devices[0];
	for (const auto &dev : physical_devices) {
		vk::PhysicalDeviceProperties properties = dev.getProperties();
		LOG_DEBUG("GPU: {}", properties.deviceName);
		LOG_DEBUG("  type={}", vk::to_string(properties.deviceType));
		if( properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
			physical_device = dev;
		}
	}
	vk::PhysicalDeviceProperties deviceProperties = physical_device.getProperties();
	LOG_DEBUG("using {}", deviceProperties.deviceName);

	// Request a single compute queue
	const float defaultQueuePriority(0.0f);

	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physical_device.getQueueFamilyProperties();
	vk::DeviceQueueCreateInfo queueCreateInfo({}, 0, 1, &defaultQueuePriority);
	for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++) {
		if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute) {
			queueFamilyIndex = i;
			queueCreateInfo.queueFamilyIndex = i;
			break;
		}
	}

	// Create logical device
	vk::DeviceCreateInfo deviceCreateInfo({}, 1, &queueCreateInfo);

	std::vector<const char*> deviceExtensions = {};
	deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
	deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
	device = physical_device.createDevice(deviceCreateInfo);

	
	//auto& requested_synchronisation2_features = device.request_extension_features<VkPhysicalDeviceSynchronization2FeaturesKHR>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR);
	//requested_synchronisation2_features.synchronization2 = VK_TRUE;

	// Get a compute queue
	queue = device.getQueue(queueFamilyIndex, 0);

	// Compute command pool
	vk::CommandPoolCreateInfo cmdPoolInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndex);
	command_pool = device.createCommandPool(cmdPoolInfo);
}

void Context::close()
{
	device.destroyCommandPool(command_pool);
	device.destroy();
#if DEBUG
	if (debugReportCallback) {
		PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallback = reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT"));
		assert(vkDestroyDebugReportCallback);
		vkDestroyDebugReportCallback(instance, debugReportCallback, nullptr);
	}
#endif
	instance.destroy();
}