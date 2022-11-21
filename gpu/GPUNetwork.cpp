#include "GPUNetwork.h"
#include "compute.h"
#include "../Logging.h"

void GPUNetwork::init(Network& network) {
	_context.open();

	std::vector<float_t> weights;
	std::vector<uint32_t> layer_weight_sizes;
	uint32_t max_layer_size = 0;
	_network_size = 0;

	for (const auto layer : network.layers) {
		//flatten biases into weights
		for (int node_index = 0; node_index < layer.size; node_index++) {
			for (int input_index = 0; input_index < layer.input_size; input_index++) {
				weights.push_back(layer.weights[node_index * layer.input_size + input_index]);
			}
			weights.push_back(layer.biases[node_index]);
		}

		layer_weight_sizes.push_back(layer.weights.size() + layer.biases.size());
		if (layer.size > max_layer_size) {
			max_layer_size = layer.size;
		}
		_network_size += layer.size;
	}

	_output_size = network.layers[network.layers.size() - 1].size;
	uint32_t input_size = network.layers[0].input_size;

	_input_buffer = HostDeviceBufferPair(&_context, std::max(max_layer_size + 1, input_size + 1) * sizeof(float_t));
	_output_buffer = HostDeviceBufferPair(&_context, _network_size * sizeof(float_t));
	_data_buffer = HostDeviceBufferPair(&_context, weights.size() * sizeof(float_t));
	_activated_buffer = HostDeviceBufferPair(&_context, _network_size * sizeof(float_t));
	_deltas_buffer = HostDeviceBufferPair(&_context, weights.size() * sizeof(float_t));

	_data_buffer.store(weights.data(), weights.size() * sizeof(float_t));

	std::vector<HostDeviceBufferPair*> buffers = { &_input_buffer, &_output_buffer, &_data_buffer, &_activated_buffer, &_deltas_buffer };

	for (const auto buffer : buffers) {
		if (buffer->device.buffer == vk::Buffer(nullptr)) {
			throw std::runtime_error("buffer was null");
		}
	}
	std::vector<std::string> pipelines = { "compute", "activate", "reset", "clear" };

	_compute = std::make_unique<Compute>(_context, buffers, pipelines);
}

void GPUNetwork::setup_calculate_only_pipeline(const Network& network)
{
	auto& cmd = start_commands();
	init_commands(cmd, network);
	calculate_commands(cmd, network);
	readback_commands(cmd, network);
	end_commands(cmd);
}

vk::CommandBuffer& GPUNetwork::start_commands()
{
	//write command buffer
	auto& command_buffer = _compute->command_buffer;
	vk::CommandBufferBeginInfo cmdBufInfo;
	command_buffer.begin(&cmdBufInfo);
	return command_buffer;
}

void GPUNetwork::init_commands(vk::CommandBuffer& command_buffer, const Network& network)
{
	// Barrier to ensure that input buffer transfer is finished before compute shader reads from it
	_input_buffer.transfer_in_barrier(command_buffer);
	_data_buffer.transfer_in_barrier(command_buffer);

	auto& descriptor_set = _compute->descriptor_set;
	PushConstants constants{ 0, 0, 0 };
	constants.layer_weights_offset = 0;
	constants.layer_output_offset = 0;

	//clear out buffer to all zeroes since it uses an atomic add instead of assigning
	constants.layer_size = _network_size;
	_compute->pass("clear").bind_and_dispatch(command_buffer, descriptor_set, _network_size, 1, 1, constants);
	_output_buffer.compute_write_readwrite_barrier(command_buffer);

}

void GPUNetwork::end_commands(vk::CommandBuffer& command_buffer)
{
	command_buffer.end();
}

void GPUNetwork::readback_commands(vk::CommandBuffer &command_buffer, const Network &network)
{
	// Barrier to ensure that shader writes are finished before buffer is read back from GPU
	_activated_buffer.shader_write_barrier(command_buffer);
	_output_buffer.shader_write_barrier(command_buffer);

	// Read back to host visible buffer
	vk::BufferCopy copyRegion(0, 0, _network_size * sizeof(float_t));
	command_buffer.copyBuffer(_activated_buffer.device.buffer, _activated_buffer.host.buffer, 1, &copyRegion);
	command_buffer.copyBuffer(_output_buffer.device.buffer, _output_buffer.host.buffer, 1, &copyRegion);
	_activated_buffer.transfer_out_barrier(command_buffer);
	_output_buffer.transfer_out_barrier(command_buffer);
}

void GPUNetwork::calculate_commands(vk::CommandBuffer& command_buffer, const Network &network)
{
	auto& descriptor_set = _compute->descriptor_set;
	PushConstants constants{ 0, 0, 0 };
	constants.layer_weights_offset = 0;
	constants.layer_output_offset = 0;

	//clear out buffer to all zeroes since it uses an atomic add instead of assigning
	constants.layer_size = _network_size;
	_compute->pass("clear").bind_and_dispatch(command_buffer, descriptor_set, _network_size, 1, 1, constants);
	_output_buffer.compute_write_readwrite_barrier(command_buffer);

	for (uint32_t layer_index = 0; layer_index < network.layers.size(); layer_index++)
	{
		auto& layer = network.layers[layer_index];

		constants.input_size = layer.input_size + 1; //+1 for bias
		constants.layer_size = layer.size;

		// collect weighted inputs
		// y +1 is bias
		_compute->pass("compute").bind_and_dispatch(command_buffer, descriptor_set, layer.size, layer.input_size + 1, 1, constants);

		//barrier on write to output_buffer before it can be read
		_output_buffer.compute_write_read_barrier(command_buffer);

		//calculate activations
		_compute->pass("activate").bind_and_dispatch(command_buffer, descriptor_set, layer.size, 1, 1, constants);

		//barrier on write to activations_buffer before it can be read
		_activated_buffer.compute_write_read_barrier(command_buffer);

		//reset output buffer and copy activated to new input
		//+1 on x inserts bias in shader. TODO: better way to do this
		_compute->pass("reset").bind_and_dispatch(command_buffer, descriptor_set, layer.size + 1, 1, 1, constants);

		//barrier on write to activations_buffer before it can be read
		_activated_buffer.compute_write_read_barrier(command_buffer);
		
		constants.layer_weights_offset += (layer.weights.size() + layer.biases.size());
		constants.layer_output_offset += layer.size;
	}
}

void GPUNetwork::gradient_commands(vk::CommandBuffer& command_buffer, const Network& network)
{
	_expected_buffer.transfer_in_barrier(command_buffer);

}

void GPUNetwork::training_step(const Buffers &buffers)
{

}

void GPUNetwork::calculate(const Buffers &buffers) {
	std::vector<float_t> inputs;
	for (const auto val : *buffers.input) {
		inputs.push_back(static_cast<float_t>(val));
	}
	//bias
	inputs.push_back(1.0);

	_input_buffer.store(inputs.data(), inputs.size() * sizeof(float_t));

	_compute->run();

	// Copy to output
	buffers.output->resize(_network_size);
	buffers.activated->resize(_network_size);
	_activated_buffer.host.read_back(buffers.activated->data(), _network_size * sizeof(float_t));
	_output_buffer.host.read_back(buffers.output->data(), _network_size * sizeof(float_t));

	_context.queue.waitIdle();
}

void GPUNetwork::destroy() {

	_input_buffer.destroy();
	_output_buffer.destroy();
	_data_buffer.destroy();
	_activated_buffer.destroy();
	_deltas_buffer.destroy();
	_expected_buffer.destroy();

	_compute.reset();
	_context.close();
}