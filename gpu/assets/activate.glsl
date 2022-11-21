#version 450

#include "shared.glsl"

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


void main() 
{
	uint node_index = gl_GlobalInvocationID.x;
	if (node_index >= PushConstants.layer_size) 
		return;	

	node_index += PushConstants.layer_output_offset;

	activated_buf[node_index] = sigmoid(out_buf[node_index]);
	//activated_buf[node_index] = PushConstants.layer_size;
}
