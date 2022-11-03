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


vk::Pipeline create_pipeline(Context &context, const std::string& shader_name, vk::PipelineLayout &pipeline_layout, vk::PipelineCache &pipeline_cache)
{
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
	auto shaderModule = load_shader(context.device, shader_code);
	vk::PipelineShaderStageCreateInfo shaderStage({}, vk::ShaderStageFlagBits::eCompute, shaderModule, "main", &specializationinfo);
	if ((VkShaderModule)shaderStage.module == VK_NULL_HANDLE) {
		throw std::runtime_error("failed to create shader stage");
	}

	vk::ComputePipelineCreateInfo computePipelineCreateInfo;
	computePipelineCreateInfo.layout = pipeline_layout;
	computePipelineCreateInfo.stage = shaderStage;

	auto pipelineRes = context.device.createComputePipeline(pipeline_cache, computePipelineCreateInfo);
	if (pipelineRes.result != vk::Result::eSuccess) {
		throw std::runtime_error("Failed to create pipeline:" + vk::to_string(pipelineRes.result));
	}
	return pipelineRes.value;
}

