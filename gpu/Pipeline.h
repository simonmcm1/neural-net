#pragma once

#include "Context.h"
#include <memory>

struct PushConstants {
	uint32_t layer_weights_offset;
	uint32_t input_size;
	uint32_t layer_size;
};

class ComputePass {
private:
	Context& _context;
	vk::PipelineLayout *_pipeline_layout;

public:
	std::string name;
	vk::ShaderModule shader_module;
	vk::Pipeline pipeline;

	ComputePass() = delete;
	ComputePass(Context& context, vk::PipelineLayout *pipeline_layout) : _context(context), _pipeline_layout(pipeline_layout) {}

	void bind_and_dispatch(
		vk::CommandBuffer& command_buffer, 
		vk::DescriptorSet& descriptor_set, 
		uint32_t x, 
		uint32_t y, 
		uint32_t z, 
		PushConstants push_constants);
	void destroy();
};

std::unique_ptr<ComputePass> create_pipeline(Context& context, const std::string& shader_name, vk::PipelineLayout *pipeline_layout, vk::PipelineCache& pipeline_cache);