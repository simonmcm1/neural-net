#pragma once
#include "../Network.h"
#include "compute.h"
#include <cstdint>

#ifdef __linux__
	typedef float float_t;
#endif

struct Buffers {
	std::vector<double> *input;
	std::vector<float> *expected;
	std::vector<float> *activated;
	std::vector<float> *output;
	std::vector<float> *deltas;
};

class GPUNetwork {
private:
	std::vector<float> weights;

	HostDeviceBufferPair _input_buffer;
	HostDeviceBufferPair _output_buffer;
	HostDeviceBufferPair _data_buffer;
	HostDeviceBufferPair _activated_buffer;
	HostDeviceBufferPair _deltas_buffer;
	HostDeviceBufferPair _expected_buffer;

	std::unique_ptr<Compute> _compute;
	Context _context;

	uint32_t _output_size;
	uint32_t _network_size;

	
	
	vk::CommandBuffer& start_commands();
	void end_commands(vk::CommandBuffer& cmd);
	void init_commands(vk::CommandBuffer& command_buffer, const Network& network);

	void calculate_commands(vk::CommandBuffer& command_buffer, const Network& network);
	void readback_commands(vk::CommandBuffer& command_buffer, const Network& network);
	void gradient_commands(vk::CommandBuffer& command_buffer, const Network& network);
	

public:
	void init(Network& network);
	void destroy();
	void setup_calculate_only_pipeline(const Network& network);
	void calculate(const Buffers &buffers);
	void training_step(const Buffers &buffers);


};