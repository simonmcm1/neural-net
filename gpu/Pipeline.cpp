#include "Pipeline.h"
#include "../util.h"
#include <exception>

vk::ShaderModule load_shader(vk::Device& device, const std::vector<char>& bytes) {
	vk::ShaderModuleCreateInfo create_info{};
	create_info.codeSize = bytes.size();
	create_info.pCode = reinterpret_cast<const uint32_t*>(bytes.data());

	vk::ShaderModule shader = device.createShaderModule(create_info);
	return shader;
}


std::unique_ptr<ComputePass> create_pipeline(Context &context, const std::string& shader_name, vk::PipelineLayout *pipeline_layout, vk::PipelineCache &pipeline_cache)
{
	std::unique_ptr<ComputePass> result = std::make_unique<ComputePass>(context, pipeline_layout);
	result->name = shader_name;

	//specialization
	struct SpecializationData {
		uint32_t LAYER_SIZE = 2;
		uint32_t INPUT_SIZE = 2;
	} specializationData;

	std::vector<vk::SpecializationMapEntry> specializationMapEntries{
		vk::SpecializationMapEntry(0, 0, sizeof(uint32_t)),
		vk::SpecializationMapEntry(1, sizeof(uint32_t), sizeof(uint32_t))
	};
	vk::SpecializationInfo specializationinfo(specializationMapEntries.size(), specializationMapEntries.data(), sizeof(SpecializationData), &specializationData);


	//shader
	auto shader_code = read_file("../../../gpu/assets/" + shader_name + ".spv");
	result->shader_module = load_shader(context.device, shader_code);
	vk::PipelineShaderStageCreateInfo shaderStage({}, vk::ShaderStageFlagBits::eCompute, result->shader_module, "main", &specializationinfo);
	if ((VkShaderModule)shaderStage.module == VK_NULL_HANDLE) {
		throw std::runtime_error("failed to create shader stage");
	}

	vk::ComputePipelineCreateInfo computePipelineCreateInfo;
	computePipelineCreateInfo.layout = *pipeline_layout;
	computePipelineCreateInfo.stage = shaderStage;

	auto pipeline = context.device.createComputePipeline(pipeline_cache, computePipelineCreateInfo);
	if (pipeline.result != vk::Result::eSuccess) {
		throw std::runtime_error("Failed to create pipeline:" + vk::to_string(pipeline.result));
	}
	
	result->pipeline = pipeline.value;
	return result;
}

void ComputePass::bind_and_dispatch(
	vk::CommandBuffer& command_buffer, 
	vk::DescriptorSet &descriptor_set, 
	uint32_t x, 
	uint32_t y, 
	uint32_t z, 
	PushConstants push_constants)
{
	command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *_pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
	command_buffer.pushConstants(*_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants), &push_constants);
	command_buffer.dispatch(x, y, z);
}

void ComputePass::destroy() {
	_context.device.destroyPipeline(pipeline);
	_context.device.destroyShaderModule(shader_module);
}
