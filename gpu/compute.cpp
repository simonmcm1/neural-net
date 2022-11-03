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

Compute::Compute()
{
	LOG("Running headless compute example\n");
	_context.open();

	/*
		Prepare storage buffers
	*/
	std::vector<float_t> computeInput(kBufferElements);

	// Fill input data
	uint32_t n = 0;
	std::generate(computeInput.begin(), computeInput.end(), [&n] { return n++; });

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
	_input_buffer = HostDeviceBufferPair(&_context, inputs.size() * sizeof(float_t));
	_input_buffer.store(inputs.data(), inputs.size() * sizeof(float_t));
	_output_buffer = HostDeviceBufferPair(&_context, kBufferSize);
	_activated_buffer = HostDeviceBufferPair(&_context, kBufferSize);
	_data_buffer = HostDeviceBufferPair(&_context, weights.size() * sizeof(float_t));
	_data_buffer.store(weights.data(), weights.size() * sizeof(float_t));


	/*
		Prepare compute pipeline
	*/
	{
		std::vector<vk::DescriptorPoolSize> poolSizes = {
			vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1)
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo({}, 1, static_cast<uint32_t>(poolSizes.size()), poolSizes.data());
		descriptorPool = _context.device.createDescriptorPool(descriptorPoolInfo);
		
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
			vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
			vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
			vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
			vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout({}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data());
		descriptorSetLayout = _context.device.createDescriptorSetLayout(descriptorLayout);

		vk::PushConstantRange push_constants(vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants));
		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo({}, 1, & descriptorSetLayout, 1, &push_constants);
		pipelineLayout = _context.device.createPipelineLayout(pipelineLayoutCreateInfo);

		vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, 1, &descriptorSetLayout);
		auto sets = _context.device.allocateDescriptorSets(allocInfo);
		assert(sets.size() > 0);
		descriptorSet = sets[0];

		vk::DescriptorBufferInfo inputBufferDescriptor(_input_buffer.device.buffer, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo outputBufferDescriptor(_output_buffer.device.buffer, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo dataBufferDescriptor(_data_buffer.device.buffer, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo activatedBufferDescriptor(_activated_buffer.device.buffer, 0, VK_WHOLE_SIZE);

		std::vector<vk::WriteDescriptorSet> computeWriteDescriptorSets = {
			vk::WriteDescriptorSet(
						descriptorSet, 0, {}, 1,
						vk::DescriptorType::eStorageBuffer, {}, &inputBufferDescriptor),
			vk::WriteDescriptorSet(
						descriptorSet, 1, {}, 1,
						vk::DescriptorType::eStorageBuffer, {}, &outputBufferDescriptor),
			vk::WriteDescriptorSet(
						descriptorSet, 2, {}, 1,
						vk::DescriptorType::eStorageBuffer, {}, &dataBufferDescriptor),
			vk::WriteDescriptorSet(
						descriptorSet, 3, {}, 1,
						vk::DescriptorType::eStorageBuffer, {}, &activatedBufferDescriptor)
		};
		_context.device.updateDescriptorSets(static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, nullptr);
	
		vk::PipelineCacheCreateInfo pipelineCacheCreateInfo;
		pipelineCache = _context.device.createPipelineCache(pipelineCacheCreateInfo);

		//collect inputs pipeline
		collect_inputs_pipeline = create_pipeline(_context, "compute", pipelineLayout, pipelineCache);

		//activations pipeline
		activate_pipeline = create_pipeline(_context, "activate", pipelineLayout, pipelineCache);

		//reset layer pipeline
		reset_pipeline = create_pipeline(_context, "reset", pipelineLayout, pipelineCache);

		// Create a command buffer for compute operations
		vk::CommandBufferAllocateInfo cmdBufAllocateInfo(_context.command_pool, vk::CommandBufferLevel::ePrimary, 1);
		commandBuffer = _context.device.allocateCommandBuffers(cmdBufAllocateInfo)[0];

		// Fence for compute CB sync
		vk::FenceCreateInfo fenceCreateInfo(vk::FenceCreateFlagBits::eSignaled);
		fence = _context.device.createFence(fenceCreateInfo);
	}

	/*
		Command buffer creation (for compute work submission)
	*/
	{
		vk::CommandBufferBeginInfo cmdBufInfo;
		PushConstants constants{ 0 };
		commandBuffer.begin(&cmdBufInfo);


		// Barrier to ensure that input buffer transfer is finished before compute shader reads from it
		_input_buffer.transfer_in_barrier(commandBuffer);
		_data_buffer.transfer_in_barrier(commandBuffer);
		//_output_buffer.transfer_in_barrier(commandBuffer);


		for (uint32_t layer_index = 0; layer_index < 2; layer_index++)
		{		
			//collect weighted inputs
			commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, collect_inputs_pipeline);
			commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

			constants.layer_weights_offset = layer_index * 4;
			commandBuffer.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants), &constants);
			commandBuffer.dispatch(2, 2, 1);

			//barrier on write to output_buffer before it can be read
			_output_buffer.compute_write_read_barrier(commandBuffer);

			//calculate activations
			commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, activate_pipeline);
			commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
			commandBuffer.dispatch(2, 1, 1);

			//barrier on write to activations_buffer before it can be read
			_activated_buffer.compute_write_read_barrier(commandBuffer);

			//reset output buffer and copy activated to new input
			commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, reset_pipeline);
			commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
			commandBuffer.dispatch(2, 1, 1);

		}



		// Barrier to ensure that shader writes are finished before buffer is read back from GPU
		//_input_buffer.shader_write_barrier(commandBuffer);
		//_output_buffer.shader_write_barrier(commandBuffer);
		_activated_buffer.shader_write_barrier(commandBuffer);

		// Read back to host visible buffer
		vk::BufferCopy copyRegion(0, 0, kBufferSize);
		//commandBuffer.copyBuffer(_input_buffer.device.buffer, _input_buffer.host.buffer, 1, &copyRegion);
		//commandBuffer.copyBuffer(_output_buffer.device.buffer, _output_buffer.host.buffer, 1, &copyRegion);
		commandBuffer.copyBuffer(_activated_buffer.device.buffer, _activated_buffer.host.buffer, 1, &copyRegion);

		// Barrier to ensure that buffer copy is finished before host reading from it
		//_input_buffer.transfer_out_barrier(commandBuffer);
		//_output_buffer.transfer_out_barrier(commandBuffer);
		_activated_buffer.transfer_out_barrier(commandBuffer);

		commandBuffer.end();
	}
}

void Compute::run() {

	// Submit compute work
	_context.device.resetFences(1, &fence);
	const vk::PipelineStageFlags waitStageMask = vk::PipelineStageFlagBits::eTransfer;
	vk::SubmitInfo computeSubmitInfo(0, nullptr, &waitStageMask, 1, &commandBuffer, 0, nullptr);
	_context.queue.submit(1, &computeSubmitInfo, fence);
	_context.device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX);

	// Copy to output
	std::vector<float> computeOutput(kBufferElements);
	//_input_buffer.host.read_back(computeOutput.data(), kBufferSize);
	//_output_buffer.host.read_back(computeOutput.data(), kBufferSize);
	_activated_buffer.host.read_back(computeOutput.data(), kBufferSize);

	_context.queue.waitIdle();
	//std::this_thread::sleep_for(std::chrono::milliseconds(3000));
	// Output buffer contents
	//LOG("Compute input:\n");
	//for (auto v : computeInput) {
	//	LOG("%d \t", v);
	//}
	//std::cout << std::endl;

	LOG("Compute output:\n");
	for (auto v : computeOutput) {
		LOG("%f \t", v);
	}

	std::cout << std::endl;
}


Compute::~Compute()
{
	// Clean up
	_input_buffer.destroy();
	_output_buffer.destroy();
	_data_buffer.destroy();
	_activated_buffer.destroy();

	_context.device.destroyPipelineLayout(pipelineLayout);
	_context.device.destroyDescriptorSetLayout(descriptorSetLayout);
	_context.device.destroyDescriptorPool(descriptorPool);
	_context.device.destroyPipeline(collect_inputs_pipeline);
	_context.device.destroyPipeline(activate_pipeline);
	_context.device.destroyPipelineCache(pipelineCache);
	_context.device.destroyFence(fence);
	
	_context.device.destroyShaderModule(shaderModule);

	_context.close();
}


int go() {
	Compute* compute = new Compute();
	compute->run();
	//compute->run();
	//compute->run();
	//compute->run();
	std::cout << "Finished. Press enter to terminate...";
	delete(compute);
	return 0;
}
