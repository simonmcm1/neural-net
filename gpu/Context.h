#pragma once
#include <vulkan/vulkan.hpp>

class Context {
public:
	vk::Instance instance;
	vk::Device device;
	vk::PhysicalDevice physical_device;

	vk::DebugReportCallbackEXT debugReportCallback{};

	uint32_t queueFamilyIndex;
	vk::Queue queue;
	vk::CommandPool command_pool;

	void open();
	void close();
};