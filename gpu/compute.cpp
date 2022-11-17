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

		std::vector<vk::DescriptorBufferInfo> buffer_descriptors(0);
		std::vector<vk::WriteDescriptorSet> computeWriteDescriptorSets(0);
		for (size_t i = 0; i < buffers.size(); i++) {
			vk::DescriptorBufferInfo info(buffers[i]->device.buffer, 0, VK_WHOLE_SIZE);
			buffer_descriptors.push_back(info);
			vk::WriteDescriptorSet write_set(
					descriptor_set, i, {}, 1,
					vk::DescriptorType::eStorageBuffer, {}, &buffer_descriptors[i]);
			computeWriteDescriptorSets.push_back(write_set);
		}

		// TODO: Why is this needed? 
		// 		 computeWriteDescriptorSets[i].pBufferInfo[0].buffer is nullptr here even though we set it in the loop above
		//       this also worked fine in MSVC and was only noticed in linux
		for (size_t i = 0; i < buffers.size(); i++) {
			computeWriteDescriptorSets[i].pBufferInfo = &buffer_descriptors[i];
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
}


int go() {
	return 0;
}
