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

#define LOG(...) printf(__VA_ARGS__)

vk::ShaderModule load_shader(vk::Device &device, const std::vector<char>& bytes) {
	vk::ShaderModuleCreateInfo create_info{};
	create_info.codeSize = bytes.size();
	create_info.pCode = reinterpret_cast<const uint32_t*>(bytes.data());

	vk::ShaderModule shader = device.createShaderModule(create_info);
	return shader;
}
	
Compute::Compute()
{
	LOG("Running headless compute example\n");
	_context.open();

	/*
		Prepare storage buffers
	*/
	std::vector<uint32_t> computeInput(kBufferElements);

	// Fill input data
	uint32_t n = 0;
	std::generate(computeInput.begin(), computeInput.end(), [&n] { return n++; });

	// Copy input data to VRAM using a staging buffer
	_input_buffer = HostDeviceBufferPair(&_context, kBufferSize);
	_input_buffer.store(computeInput.data(), kBufferSize);

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
			vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout({}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data());
		descriptorSetLayout = _context.device.createDescriptorSetLayout(descriptorLayout);

		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo({}, 1, & descriptorSetLayout);
		pipelineLayout = _context.device.createPipelineLayout(pipelineLayoutCreateInfo);

		vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, 1, &descriptorSetLayout);
		auto sets = _context.device.allocateDescriptorSets(allocInfo);
		assert(sets.size() > 0);
		descriptorSet = sets[0];

		vk::DescriptorBufferInfo bufferDescriptor(_input_buffer.device.buffer, 0, VK_WHOLE_SIZE);
		std::vector<vk::WriteDescriptorSet> computeWriteDescriptorSets = {
			vk::WriteDescriptorSet(
						descriptorSet, 0, {}, 1,
						vk::DescriptorType::eStorageBuffer, {}, &bufferDescriptor)
		};
		_context.device.updateDescriptorSets(static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, nullptr);
	
		vk::PipelineCacheCreateInfo pipelineCacheCreateInfo;
		pipelineCache = _context.device.createPipelineCache(pipelineCacheCreateInfo);

		// Create pipeline
		vk::ComputePipelineCreateInfo computePipelineCreateInfo;
		computePipelineCreateInfo.layout = pipelineLayout;

		// Pass SSBO size via specialization constant
		struct SpecializationData {
			uint32_t BUFFER_ELEMENT_COUNT = kBufferElements;
		} specializationData;

		vk::SpecializationMapEntry specializationMapEntry(0, 0, sizeof(uint32_t));
		vk::SpecializationInfo specializationinfo(1, &specializationMapEntry, sizeof(SpecializationData), &specializationData);
			
		auto shader_code = read_file("../../../gpu/assets/compute.spv");
		shaderModule = load_shader(_context.device, shader_code);
		vk::PipelineShaderStageCreateInfo shaderStage({}, vk::ShaderStageFlagBits::eCompute, shaderModule, "main", & specializationinfo);

		assert(shaderStage.module != VK_NULL_HANDLE);
		computePipelineCreateInfo.stage = shaderStage;

		auto pipelineRes = _context.device.createComputePipeline(pipelineCache, computePipelineCreateInfo);
		assert(pipelineRes.result == vk::Result::eSuccess);
		pipeline = pipelineRes.value;

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
		commandBuffer.begin(&cmdBufInfo);


		// Barrier to ensure that input buffer transfer is finished before compute shader reads from it
		_input_buffer.transfer_in_barrier(commandBuffer);

		commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
			
		commandBuffer.dispatch(kBufferElements, 1, 1);

		// Barrier to ensure that shader writes are finished before buffer is read back from GPU
		_input_buffer.shader_write_barrier(commandBuffer);

		// Read back to host visible buffer
		vk::BufferCopy copyRegion(0, 0, kBufferSize);
		commandBuffer.copyBuffer(_input_buffer.device.buffer, _input_buffer.host.buffer, 1, &copyRegion);

		// Barrier to ensure that buffer copy is finished before host reading from it
		_input_buffer.transfer_out_barrier(commandBuffer);

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
	std::vector<uint32_t> computeOutput(kBufferElements);
	_input_buffer.host.read_back(computeOutput.data(), kBufferSize);

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
		LOG("%d \t", v);
	}

	std::cout << std::endl;
}


Compute::~Compute()
{
	// Clean up
	_input_buffer.destroy();

	_context.device.destroyPipelineLayout(pipelineLayout);
	_context.device.destroyDescriptorSetLayout(descriptorSetLayout);
	_context.device.destroyDescriptorPool(descriptorPool);
	_context.device.destroyPipeline(pipeline);
	_context.device.destroyPipelineCache(pipelineCache);
	_context.device.destroyFence(fence);
	
	_context.device.destroyShaderModule(shaderModule);

	_context.close();
}


int go() {
	Compute* compute = new Compute();
	compute->run();
	compute->run();
	compute->run();
	compute->run();
	std::cout << "Finished. Press enter to terminate...";
	delete(compute);
	return 0;
}
