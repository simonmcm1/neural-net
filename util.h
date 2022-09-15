#pragma once
#include <string>
#include <vector>

double get_random();

double random01();

double relu(double x);

double sigmoid(double x);

double sigmoid_derivative(double x);

double cost(const std::vector<double>& output, const std::vector<double>& expected);

double cost_derivative(double output, double expected);

void read_file(const std::string& path, std::vector<uint8_t>& buffer);

uint32_t from_big_endian(uint8_t* data);
