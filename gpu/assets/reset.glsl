#version 450

#include "shared.glsl"

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() 
{
	uint node_index = gl_GlobalInvocationID.x;
	if (node_index >= LAYER_SIZE) 
		return;	

	//reset out_buf 
	out_buf[node_index] = 0.0;

	//copy last layer activations to input
	input_buf[node_index] = activated_buf[node_index];
}