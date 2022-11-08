#version 450
#extension GL_EXT_shader_atomic_float : enable

#include "shared.glsl"

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void weighted_add(uint input_index, uint node_index) {
	uint index = node_index * PushConstants.input_size + input_index;
	index += PushConstants.layer_weights_offset;
	float weighted_input = input_buf[input_index] * data_buf[index];
	atomicAdd(out_buf[node_index], weighted_input); 
}

void main() 
{
	uint node_index = gl_GlobalInvocationID.x;
	uint input_index = gl_GlobalInvocationID.y;

	if (node_index >= PushConstants.layer_size || input_index >= PushConstants.input_size) 
		return;	

	weighted_add(input_index, node_index);
	//activated_buf[node_index] = PushConstants.layer_weights_offset;
}
