#pragma once
#include <string>
#include <vector>
#include <thread>
#include <functional>

double get_random();

double random01();

double relu(double x);

double sigmoid(double x);

double sigmoid_derivative(double x);

double cost(const std::vector<double>& output, const std::vector<double>& expected);

double cost_derivative(double output, double expected);

void read_file(const std::string& path, std::vector<uint8_t>& buffer);

std::vector<char> read_file(const std::string& path);

uint32_t from_big_endian(uint8_t* data);

using batch_function = std::function<void(size_t, size_t, size_t)>;
void batch_jobs(batch_function& task, int nthreads, size_t data_len);

void double_vector_to_float(const std::vector<double>& a, std::vector<float>& b);