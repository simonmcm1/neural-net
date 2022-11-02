#pragma once
#include "Context.h"

class Buffer {
private: 
	Context *context = nullptr;
public:
	vk::Buffer buffer = nullptr;
	vk::DeviceMemory memory = nullptr;
	vk::DeviceSize size = 0;

	Buffer() = default;
	Buffer(Context *ctx) : context(ctx) {};

	static Buffer create(Context& context, vk::BufferUsageFlags usageFlags, vk::MemoryPropertyFlags memoryPropertyFlags, vk::DeviceSize buffer_size);
	void store(void* data, vk::DeviceSize data_len, bool flush=false);
	void read_back(void* dest, vk::DeviceSize len);
};


class HostDeviceBufferPair {
private:
	Context* context = nullptr;
public:
	Buffer host;
	Buffer device;

	HostDeviceBufferPair() = default;
	HostDeviceBufferPair(Context* ctx, vk::DeviceSize size);
	void init(Context* ctx, vk::DeviceSize size);
	void store(void* data, vk::DeviceSize len);

	void transfer_in_barrier(vk::CommandBuffer& command_buffer);
	void shader_write_barrier(vk::CommandBuffer& command_buffer);
	void compute_write_read_barrier(vk::CommandBuffer& command_buffer);
	void transfer_out_barrier(vk::CommandBuffer& command_buffer);

	void destroy();
};
