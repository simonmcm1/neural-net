#include "Buffer.h"

Buffer Buffer::create(Context& context, vk::BufferUsageFlags usageFlags, vk::MemoryPropertyFlags memoryPropertyFlags, vk::DeviceSize size)
{
	// Create the buffer handle	
	vk::BufferCreateInfo bufferCreateInfo({}, size, usageFlags, vk::SharingMode::eExclusive);
	vk::Buffer buffer = context.device.createBuffer(bufferCreateInfo);

	if (buffer == (vk::Buffer)nullptr) {
		throw std::runtime_error("failed to create buffer");
	}

	// Create the memory backing up the buffer handle
	vk::PhysicalDeviceMemoryProperties deviceMemoryProperties = context.physical_device.getMemoryProperties();
	vk::MemoryRequirements memReqs = context.device.getBufferMemoryRequirements(buffer);

	// Find a memory type index that fits the properties of the buffer
	uint32_t memoryTypeIndex = -1;
	bool memTypeFound = false;
	for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++) {
		if ((memReqs.memoryTypeBits & 1) == 1) {
			if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & memoryPropertyFlags) == memoryPropertyFlags) {
				memoryTypeIndex = i;
				memTypeFound = true;
			}
		}
		memReqs.memoryTypeBits >>= 1;
	}
	assert(memTypeFound);

	vk::MemoryAllocateInfo memAlloc(memReqs.size, memoryTypeIndex);
	vk::DeviceMemory memory = context.device.allocateMemory(memAlloc);

	context.device.bindBufferMemory(buffer, memory, 0);

	Buffer res(&context);
	res.buffer = buffer;
	res.memory = memory;
	res.size = size;
	
	return res;
}

void Buffer::store(void* data, vk::DeviceSize data_len, bool flush) {
	if (data != nullptr) {
		void* mapped = context->device.mapMemory(memory, 0, size);
		memcpy(mapped, data, data_len);
		if (flush) {
			vk::MappedMemoryRange mappedRange(memory, 0, VK_WHOLE_SIZE);
			context->device.flushMappedMemoryRanges(1, &mappedRange);
		}
		context->device.unmapMemory(memory);
	}
}

void Buffer::read_back(void* dest, vk::DeviceSize data_len) {
	void* mapped = context->device.mapMemory(memory, 0, data_len);
	vk::MappedMemoryRange mappedRange(memory, 0, VK_WHOLE_SIZE);
	context->device.invalidateMappedMemoryRanges(1, &mappedRange);

	memcpy(dest, mapped, data_len);
	context->device.unmapMemory(memory);
}

void HostDeviceBufferPair::init(Context* ctx, vk::DeviceSize size)
{
	host = Buffer::create(
		*context,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible,
		size);

	device = Buffer::create(
		*context,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		size
	);

}

void HostDeviceBufferPair::store(void* data, vk::DeviceSize len)
{
	host.store(data, len, true);

	// Copy to staging buffer		
	vk::CommandBufferAllocateInfo cmdBufAllocateInfo(context->command_pool, vk::CommandBufferLevel::ePrimary, 1);
	vk::CommandBuffer copyCmd = context->device.allocateCommandBuffers(cmdBufAllocateInfo)[0];
	vk::CommandBufferBeginInfo cmdBufInfo;
	copyCmd.begin(cmdBufInfo);

	vk::BufferCopy copyRegion(0, 0, len);
	copyCmd.copyBuffer(host.buffer, device.buffer, 1, &copyRegion);
	copyCmd.end();

	vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &copyCmd);
	vk::FenceCreateInfo fenceInfo;
	vk::Fence lfence = context->device.createFence(fenceInfo);

	// Submit to the queue
	context->queue.submit(1, &submitInfo, lfence);
	context->device.waitForFences(1, &lfence, VK_TRUE, UINT64_MAX);

	context->device.destroyFence(lfence);
	context->device.freeCommandBuffers(context->command_pool, 1, &copyCmd);
}

void HostDeviceBufferPair::transfer_in_barrier(vk::CommandBuffer& command_buffer)
{
	vk::BufferMemoryBarrier barrier(
		vk::AccessFlagBits::eHostWrite,
		vk::AccessFlagBits::eShaderRead,
		VK_QUEUE_FAMILY_IGNORED,
		VK_QUEUE_FAMILY_IGNORED,
		device.buffer,
		0, VK_WHOLE_SIZE);

	command_buffer.pipelineBarrier(
		vk::PipelineStageFlagBits::eHost,
		vk::PipelineStageFlagBits::eComputeShader,
		vk::DependencyFlags(0),
		0, nullptr,
		1, &barrier,
		0, nullptr);
}

void HostDeviceBufferPair::transfer_out_barrier(vk::CommandBuffer& command_buffer)
{
	vk::BufferMemoryBarrier barrier(
		vk::AccessFlagBits::eTransferWrite,
		vk::AccessFlagBits::eHostRead,
		VK_QUEUE_FAMILY_IGNORED,
		VK_QUEUE_FAMILY_IGNORED,
		host.buffer,
		0, VK_WHOLE_SIZE);

	command_buffer.pipelineBarrier(
		vk::PipelineStageFlagBits::eTransfer,
		vk::PipelineStageFlagBits::eHost,
		vk::DependencyFlags(0),
		0, nullptr,
		1, &barrier,
		0, nullptr);
}


void HostDeviceBufferPair::compute_write_read_barrier(vk::CommandBuffer& command_buffer)
{
	vk::BufferMemoryBarrier barrier(
		vk::AccessFlagBits::eShaderWrite,
		vk::AccessFlagBits::eShaderRead,
		VK_QUEUE_FAMILY_IGNORED,
		VK_QUEUE_FAMILY_IGNORED,
		device.buffer,
		0, VK_WHOLE_SIZE);

	command_buffer.pipelineBarrier(
		vk::PipelineStageFlagBits::eComputeShader,
		vk::PipelineStageFlagBits::eComputeShader,
		vk::DependencyFlags(0),
		0, nullptr,
		1, &barrier,
		0, nullptr);

}

void HostDeviceBufferPair::compute_write_readwrite_barrier(vk::CommandBuffer& command_buffer)
{
	vk::BufferMemoryBarrier barrier(
		vk::AccessFlagBits::eShaderWrite,
		vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
		VK_QUEUE_FAMILY_IGNORED,
		VK_QUEUE_FAMILY_IGNORED,
		device.buffer,
		0, VK_WHOLE_SIZE);

	command_buffer.pipelineBarrier(
		vk::PipelineStageFlagBits::eComputeShader,
		vk::PipelineStageFlagBits::eComputeShader,
		vk::DependencyFlags(0),
		0, nullptr,
		1, &barrier,
		0, nullptr);

}

void HostDeviceBufferPair::shader_write_barrier(vk::CommandBuffer& command_buffer)
{
	vk::BufferMemoryBarrier barrier(
		vk::AccessFlagBits::eShaderWrite,
		vk::AccessFlagBits::eTransferRead,
		VK_QUEUE_FAMILY_IGNORED,
		VK_QUEUE_FAMILY_IGNORED,
		device.buffer,
		0, VK_WHOLE_SIZE);

	command_buffer.pipelineBarrier(
		vk::PipelineStageFlagBits::eComputeShader,
		vk::PipelineStageFlagBits::eTransfer,
		vk::DependencyFlags(0),
		0, nullptr,
		1, &barrier,
		0, nullptr);
}

HostDeviceBufferPair::HostDeviceBufferPair(Context* ctx, vk::DeviceSize size) : context(ctx)
{
	init(ctx, size);
}

void HostDeviceBufferPair::destroy()
{
	context->device.destroyBuffer(device.buffer);
	context->device.freeMemory(device.memory);
	context->device.destroyBuffer(host.buffer);
	context->device.freeMemory(host.memory);
}



