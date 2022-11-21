layout(binding = 0) buffer InBuffer {
   float input_buf[ ];
};

layout(binding = 1) buffer OutBuffer {
   float out_buf[ ];
};

layout(binding = 2) buffer DataBuffer {
   float data_buf[ ];
};

layout(binding = 3) buffer ActivatedBuffer {
   float activated_buf[ ];
};

layout(binding = 4) buffer DeltaBuffer {
   float delta_buf[ ];
};

layout(binding = 5) buffer ExpectedBuffer {
   float expected_buf[ ];
};

//layout (constant_id = 0) const uint LAYER_SIZE = 300;

//push constants block
layout( push_constant ) uniform constants
{
	uint layer_weights_offset;
	uint input_size;
	uint layer_size;
	uint layer_output_offset;
} PushConstants;

float sigmoid(float x)
{
	return 1.0 / (1.0 + exp(-x));
}

float sigmoid_derivative(float x) {
	float fx = sigmoid(x);
	return fx * (1 - fx);
}

float cost_derivative(float res, float expected) {
	return res - expected;
}