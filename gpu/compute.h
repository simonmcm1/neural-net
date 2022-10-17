#pragma once
#include <vulkan/vulkan.hpp>
#include "Buffer.h"

int go();

class Compute {
private:

	vk::PipelineCache pipelineCache;
	vk::CommandBuffer commandBuffer;
	vk::Fence fence;
	vk::DescriptorPool descriptorPool;
	vk::DescriptorSetLayout descriptorSetLayout;;
	vk::DescriptorSet descriptorSet;
	vk::PipelineLayout pipelineLayout;
	vk::Pipeline pipeline;
	vk::ShaderModule shaderModule;

	Context _context;
	HostDeviceBufferPair _input_buffer;

	static constexpr uint32_t kBufferElements = 32;
	static constexpr vk::DeviceSize kBufferSize = kBufferElements * sizeof(uint32_t);

public:
	Compute();
	void run();
	virtual ~Compute();
};