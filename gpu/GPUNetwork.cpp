#include "GPUNetwork.h"
#include "compute.h"
#include "../Logging.h"

void GPUNetwork::init(Network& network) {
	Context context;
	context.open();

	std::vector<float_t> weights;
	std::vector<uint32_t> layer_weight_sizes;
	uint32_t max_layer_size = 0;
	for (const auto layer : network.layers) {
		for (const auto weight : layer.weights) {
			weights.push_back(weight);
		}
		layer_weight_sizes.push_back(layer.weights.size());
		if (layer.size > max_layer_size) {
			max_layer_size = layer.size;
		}
	}

	std::vector<float_t> inputs;
	for (const auto val : network.training_data[0].data) {
		inputs.push_back(val);
	}

	uint32_t output_layer_size = network.layers[network.layers.size() - 1].size;

	// Copy input data to VRAM using a staging buffer
	auto input_buffer = HostDeviceBufferPair(&context, inputs.size() * sizeof(float_t));
	auto output_buffer = HostDeviceBufferPair(&context, max_layer_size * sizeof(float_t));
	auto data_buffer = HostDeviceBufferPair(&context, weights.size() * sizeof(float_t));
	auto activated_buffer = HostDeviceBufferPair(&context, max_layer_size * sizeof(float_t));

	input_buffer.store(inputs.data(), inputs.size() * sizeof(float_t));
	data_buffer.store(weights.data(), weights.size() * sizeof(float_t));

	std::vector<HostDeviceBufferPair*> buffers = { &input_buffer, &output_buffer, &data_buffer, &activated_buffer };
	std::vector<std::string> pipelines = { "compute", "activate", "reset" };
	Compute* compute = new Compute(context, buffers, pipelines);

	//write command buffer
	auto& command_buffer = compute->command_buffer;
	auto& descriptor_set = compute->descriptor_set;
	vk::CommandBufferBeginInfo cmdBufInfo;
	PushConstants constants{ 0 };
	command_buffer.begin(&cmdBufInfo);

	// Barrier to ensure that input buffer transfer is finished before compute shader reads from it
	input_buffer.transfer_in_barrier(command_buffer);
	data_buffer.transfer_in_barrier(command_buffer);

	constants.layer_weights_offset = 0;
	for (uint32_t layer_index = 0; layer_index < network.layers.size(); layer_index++)
	{
		auto& layer = network.layers[layer_index];
		//collect weighted inputs
		compute->pass("compute").bind_and_dispatch(command_buffer, descriptor_set, layer.size, layer.input_size, 1, constants);

		//barrier on write to output_buffer before it can be read
		output_buffer.compute_write_read_barrier(command_buffer);

		//calculate activations
		compute->pass("activate").bind_and_dispatch(command_buffer, descriptor_set, layer.size, 1, 1, constants);

		//barrier on write to activations_buffer before it can be read
		activated_buffer.compute_write_read_barrier(command_buffer);

		//reset output buffer and copy activated to new input
		compute->pass("reset").bind_and_dispatch(command_buffer, descriptor_set, layer.size, 1, 1, constants);

		constants.layer_weights_offset += layer_weight_sizes[layer_index];
	}

	// Barrier to ensure that shader writes are finished before buffer is read back from GPU
	activated_buffer.shader_write_barrier(command_buffer);

	// Read back to host visible buffer
	vk::BufferCopy copyRegion(0, 0, output_layer_size * sizeof(float_t));
	command_buffer.copyBuffer(activated_buffer.device.buffer, activated_buffer.host.buffer, 1, &copyRegion);
	activated_buffer.transfer_out_barrier(command_buffer);

	command_buffer.end();

	compute->run();

	// Copy to output
	std::vector<float> computeOutput(output_layer_size);
	//_input_buffer.host.read_back(computeOutput.data(), kBufferSize);
	//_output_buffer.host.read_back(computeOutput.data(), kBufferSize);
	activated_buffer.host.read_back(computeOutput.data(), output_layer_size * sizeof(float_t));

	context.queue.waitIdle();

	LOG_DEBUG("Compute output:\n");
	for (auto v : computeOutput) {
		LOG_DEBUG("{}\t", v);
	}

	std::cout << std::endl;



	input_buffer.destroy();
	output_buffer.destroy();
	data_buffer.destroy();
	activated_buffer.destroy();


	delete(compute);
}