#version 450

#include "shared.glsl"

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() 
{
	uint node_index = gl_GlobalInvocationID.x;

	if (node_index >= PushConstants.layer_size) 
		return;	

	float o = activated_buf[node_index + PushConstants.layer_output_offset];
	float cd = cost_derivative(o, expected_buf[node_index]);
	float ad = sigmoid_derivative(out_buf[node_index + PushConstants.layer_output_offset]);
	float delta = cd * ad;
	delta_buf[node_index + PushConstants.layer_output_offset] = delta;
}
