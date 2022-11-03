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

	vk::ShaderModule shaderModule;

	vk::Pipeline collect_inputs_pipeline;
	vk::Pipeline activate_pipeline;
	vk::Pipeline reset_pipeline;

	Context _context;
	HostDeviceBufferPair _input_buffer;
	HostDeviceBufferPair _output_buffer;
	HostDeviceBufferPair _activated_buffer;
	HostDeviceBufferPair _data_buffer;

	static constexpr uint32_t kBufferElements = 32;
	static constexpr vk::DeviceSize kBufferSize = kBufferElements * sizeof(float_t);

public:
	Compute();
	void run();
	virtual ~Compute();
};