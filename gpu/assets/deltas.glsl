#version 450

#include "shared.glsl"

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() 
{
	uint node_index = gl_GlobalInvocationID.x;
	uint input_index = gl_GlobalInvocationID.y;

	if (node_index >= PushConstants.layer_size) 
		return;	

	double o = output[node_index + PushConstants.layer_output_offset];
	double cd = cost_derivative(o, expected[node_index]);
	double ad = sigmoid_derivative(layer_data.get_activation_input(nlayers - 1, node_index));
	double delta = cd * ad;
	layer_data.set_delta(nlayers - 1, node_index, delta);

	weighted_add(input_index, node_index);
	//activated_buf[node_index] = PushConstants.layer_weights_offset;
}
