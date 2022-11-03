#pragma once
#include <vulkan/vulkan.hpp>
#include "Buffer.h"
#include "Pipeline.h"

int go();

class Compute {
private:

	vk::PipelineCache pipelineCache;
	
	vk::Fence fence;
	vk::DescriptorPool descriptorPool;
	vk::DescriptorSetLayout descriptorSetLayout;;
	
	vk::PipelineLayout pipelineLayout;

	vk::ShaderModule shaderModule;

	Context _context;

	std::unordered_map<std::string, std::unique_ptr<ComputePass>> _passes;

public:
	vk::DescriptorSet descriptor_set;
	vk::CommandBuffer command_buffer;

	Compute() = delete;
	Compute(Context& context, std::vector<HostDeviceBufferPair*> &buffers, std::vector<std::string> &pass_names);
	virtual ~Compute();

	ComputePass &pass(const std::string& name);
	void run();

};