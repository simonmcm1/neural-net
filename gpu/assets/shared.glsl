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

layout (constant_id = 0) const uint LAYER_SIZE = 300;
layout (constant_id = 1) const uint INPUT_SIZE = 300;
