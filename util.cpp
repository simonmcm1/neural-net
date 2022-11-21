#include <fstream>
#include <vector>
#include <random>
#include <assert.h>
#include <math.h>
#include <functional>
#include <algorithm>
#include <filesystem>
#include "util.h"
#include "ThreadPool.h"
#include <memory>

double random01()
{
	static std::default_random_engine e;
	static std::uniform_real_distribution<> dis(-1, 1);
	return dis(e);
}

double relu(double x)
{
	return x > 0.0 ? x : 0.0;
}

double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x) {
	double fx = sigmoid(x);
	return fx * (1 - fx);
}

double cost(const std::vector<double>& output, const std::vector<double>& expected)
{
	assert(output.size() == expected.size());
	double error = 0.0;
	for (int i = 0; i < output.size(); i++) {
		double diff = output[i] - expected[i];
		error += diff * diff;
	}
	return 0.5 * error;
}

double cost_derivative(double output, double expected) {
	return output - expected;
}


void read_file(const std::string& path, std::vector<uint8_t>& buffer)
{
	std::ifstream file(path, std::ios::binary);
	if (file.fail()) {
		throw std::runtime_error("failed to read file " + path);
	}
	file.seekg(0, std::ios::end);
	auto size = file.tellg();
	file.seekg(0, std::ios::beg);

	buffer.resize(size);
	file.read((char*)buffer.data(), size);
}

std::vector<char> read_file(const std::string& path) {
	std::ifstream file(path, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file " + path);
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();
	return buffer;
}

uint32_t from_big_endian(uint8_t* data) {
	return (data[3] << 0) | (data[2] << 8) | (data[1] << 16) | ((unsigned)data[0] << 24);
}



void batch_jobs(batch_function &task, int nthreads, size_t data_len)
{
	std::vector<std::thread> threads;
	size_t batch_size = data_len / nthreads + 1;
	size_t data_index = 0;
	for (int i = 0; i < nthreads; i++) {
		size_t len = std::min(batch_size, data_len - data_index - 1);
		auto ttask = std::thread(task, i, data_index, len);
		threads.push_back(std::move(ttask));
		data_index += len;
	}
	assert(data_index == data_len);
	for (auto& thread : threads) {
		thread.join();
	}

}

void double_vector_to_float(const std::vector<double>& a, std::vector<float>& b) 
{
	for (const auto val : a) {
		b.push_back(static_cast<float_t>(val));
	}
}
