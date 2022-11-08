#pragma once
#include "../Network.h"
#include "compute.h"

class GPUNetwork {
private:
	std::vector<float> weights;
	std::vector<float> input;
	std::vector<float> activated;

	HostDeviceBufferPair _input_buffer;
	HostDeviceBufferPair _output_buffer;
	HostDeviceBufferPair _data_buffer;
	HostDeviceBufferPair _activated_buffer;

	std::unique_ptr<Compute> _compute;
	Context _context;

	uint32_t _output_size;
public:
	void init(Network& network);
	void destroy();
	void calculate(const std::vector<double>& input_data, std::vector<float> &output);
};