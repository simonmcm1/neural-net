#pragma once

#include "Context.h"

struct PushConstants {
	uint32_t layer_weights_offset;
};

vk::Pipeline create_pipeline(Context& context, const std::string& shader_name, vk::PipelineLayout& pipeline_layout, vk::PipelineCache& pipeline_cache);