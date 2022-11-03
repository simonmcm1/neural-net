#if defined(_WIN32)
#pragma comment(linker, "/subsystem:console")
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <thread>
#include <chrono>

#include "compute.h"
#include "../util.h"
#include "Pipeline.h"

#define LOG(...) printf(__VA_ARGS__)

Compute::Compute(Context &context, std::vector<HostDeviceBufferPair *> &buffers, std::vector<std::string> &pass_names) :
	_context(context)
{
	/*
		Prepare compute pipeline
	*/
	{
		std::vector<vk::DescriptorPoolSize> poolSizes = {
			vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1)
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo({}, 1, static_cast<uint32_t>(poolSizes.size()), poolSizes.data());
		descriptorPool = _context.device.createDescriptorPool(descriptorPoolInfo);
		
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings;
		for (size_t i = 0; i < buffers.size(); i++) {
			setLayoutBindings.push_back(vk::DescriptorSetLayoutBinding(i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute));
		}
		
		vk::DescriptorSetLayoutCreateInfo descriptorLayout({}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data());
		descriptorSetLayout = _context.device.createDescriptorSetLayout(descriptorLayout);

		vk::PushConstantRange push_constants(vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants));
		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo({}, 1, & descriptorSetLayout, 1, &push_constants);
		pipelineLayout = _context.device.createPipelineLayout(pipelineLayoutCreateInfo);

		vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, 1, &descriptorSetLayout);
		auto sets = _context.device.allocateDescriptorSets(allocInfo);
		assert(sets.size() > 0);
		descriptor_set = sets[0];

		std::vector<vk::DescriptorBufferInfo> buffer_descriptors;
		std::vector<vk::WriteDescriptorSet> computeWriteDescriptorSets;;
		for (size_t i = 0; i < buffers.size(); i++) {
			buffer_descriptors.push_back(vk::DescriptorBufferInfo(buffers[i]->device.buffer, 0, VK_WHOLE_SIZE));
			computeWriteDescriptorSets.push_back(
				vk::WriteDescriptorSet(
					descriptor_set, i, {}, 1,
					vk::DescriptorType::eStorageBuffer, {}, &buffer_descriptors[i]));
		}

		_context.device.updateDescriptorSets(static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, nullptr);
	
		vk::PipelineCacheCreateInfo pipelineCacheCreateInfo;
		pipelineCache = _context.device.createPipelineCache(pipelineCacheCreateInfo);

		for (auto &pass_name : pass_names) {
			_passes[pass_name] = create_pipeline(_context, pass_name, &pipelineLayout, pipelineCache);
		}

		// Fence for compute CB sync
		vk::FenceCreateInfo fenceCreateInfo(vk::FenceCreateFlagBits::eSignaled);
		fence = _context.device.createFence(fenceCreateInfo);
	
		// Create a command buffer for compute operations
		vk::CommandBufferAllocateInfo cmdBufAllocateInfo(_context.command_pool, vk::CommandBufferLevel::ePrimary, 1);
		command_buffer = _context.device.allocateCommandBuffers(cmdBufAllocateInfo)[0];
	}
}

ComputePass &Compute::pass(const std::string& name) {
	if (_passes.find(name) == _passes.end()) {
		throw std::runtime_error("tried to get pipeline that doesn't exist");
	}
	return *(_passes[name]);
}

void Compute::run() {

	// Submit compute work
	_context.device.resetFences(1, &fence);
	const vk::PipelineStageFlags waitStageMask = vk::PipelineStageFlagBits::eTransfer;
	vk::SubmitInfo computeSubmitInfo(0, nullptr, &waitStageMask, 1, &command_buffer, 0, nullptr);
	_context.queue.submit(1, &computeSubmitInfo, fence);
	_context.device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX);
}


Compute::~Compute()
{
	// Clean up
	_context.device.destroyPipelineLayout(pipelineLayout);
	_context.device.destroyDescriptorSetLayout(descriptorSetLayout);
	_context.device.destroyDescriptorPool(descriptorPool);
	for (auto& pass : _passes) {
		pass.second->destroy();
	}
	_context.device.destroyPipelineCache(pipelineCache);
	_context.device.destroyFence(fence);
	
	_context.device.destroyShaderModule(shaderModule);

	_context.close();
}


int go() {
	Context context;
	context.open();

	/*
	Prepare storage buffers
*/
	std::vector<float_t> weights(8);
	weights[0] = 0.15;
	weights[1] = 0.2;
	weights[2] = 0.25;
	weights[3] = 0.3;
	weights[4] = 0.4;
	weights[5] = 0.45;
	weights[6] = 0.5;
	weights[7] = 0.55;

	std::vector<float_t> inputs(2);
	inputs[0] = 0.05;
	inputs[1] = 0.1;


	// Copy input data to VRAM using a staging buffer
	auto input_buffer = HostDeviceBufferPair(&context, inputs.size() * sizeof(float_t));
	auto output_buffer = HostDeviceBufferPair(&context, 200);
	auto data_buffer = HostDeviceBufferPair(&context, weights.size() * sizeof(float_t));
	auto activated_buffer = HostDeviceBufferPair(&context, 200);

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

	auto* _input_buffer = buffers[0];
	auto* _output_buffer = buffers[1];
	auto* _data_buffer = buffers[2];
	auto* _activated_buffer = buffers[3];

	// Barrier to ensure that input buffer transfer is finished before compute shader reads from it
	_input_buffer->transfer_in_barrier(command_buffer);
	_data_buffer->transfer_in_barrier(command_buffer);

	for (uint32_t layer_index = 0; layer_index < 2; layer_index++)
	{
		constants.layer_weights_offset = layer_index * 4;

		//collect weighted inputs
		compute->pass("compute").bind_and_dispatch(command_buffer, descriptor_set, 2, 2, 1, constants);

		//barrier on write to output_buffer before it can be read
		_output_buffer->compute_write_read_barrier(command_buffer);

		//calculate activations
		compute->pass("activate").bind_and_dispatch(command_buffer, descriptor_set, 2, 1, 1, constants);

		//barrier on write to activations_buffer before it can be read
		_activated_buffer->compute_write_read_barrier(command_buffer);

		//reset output buffer and copy activated to new input
		compute->pass("reset").bind_and_dispatch(command_buffer, descriptor_set, 2, 1, 1, constants);
	}

	// Barrier to ensure that shader writes are finished before buffer is read back from GPU
	_activated_buffer->shader_write_barrier(command_buffer);

	// Read back to host visible buffer
	vk::BufferCopy copyRegion(0, 0, 20);
	command_buffer.copyBuffer(_activated_buffer->device.buffer, _activated_buffer->host.buffer, 1, &copyRegion);
	_activated_buffer->transfer_out_barrier(command_buffer);

	command_buffer.end();
		
	compute->run();

	// Copy to output
	std::vector<float> computeOutput(20);
	//_input_buffer.host.read_back(computeOutput.data(), kBufferSize);
	//_output_buffer.host.read_back(computeOutput.data(), kBufferSize);
	activated_buffer.host.read_back(computeOutput.data(), 20);

	context.queue.waitIdle();

	LOG("Compute output:\n");
	for (auto v : computeOutput) {
		LOG("%f \t", v);
	}

	std::cout << std::endl;



	input_buffer.destroy();
	output_buffer.destroy();
	data_buffer.destroy();
	activated_buffer.destroy();


	delete(compute);
	return 0;
}
