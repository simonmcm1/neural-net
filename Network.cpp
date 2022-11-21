#include "Network.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <assert.h>
#include <math.h>
#include <functional>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include "util.h"
#include "Timer.h"
#include "Logging.h"

double Layer::calculate_node(int node_index, const std::vector<double>& inputs) {
	assert(node_index < size);

	double input = biases[node_index];
	for (int i = 0; i < input_size; i++) {
		int index = node_index * input_size + i;
		input += inputs[i] * weights[index];
	}
	return input;
}

std::vector<double>Layer::calculate(const std::vector<double>& inputs, LayerTrainingData *training_data) {
	//Timer t("Layer::Calculate");
	std::vector<double> output(size);
	for (int node = 0; node < size; node++) {
		double weighted_input = calculate_node(node, inputs);
		double res = sigmoid(weighted_input);
		if (training_data != nullptr) {
			training_data->set_activation_input(index, node, weighted_input);
			training_data->set_output(index, node, res);
		}
		output[node] = res;
	}
	return output;
}

void Layer::init() 
{
	weights.resize(input_size * size);
	biases.resize(size);

	for (size_t i = 0; i < size; i++) {
		for (size_t j = 0; j < input_size; j++) {
			weights[i * input_size + j] = random01() / sqrt(input_size);
		}
		biases[i] = random01() / sqrt(input_size);
	}
}

LayerTrainingData::LayerTrainingData(const std::vector<Layer>& layers) {
	activation_inputs.resize(layers.size());
	deltas.resize(layers.size());
	output.resize(layers.size());
	for (int i = 0; i < layers.size(); i++) {
		activation_inputs[i].resize(layers[i].size, 0.0);
		deltas[i].resize(layers[i].size, 0.0);
		output[i].resize(layers[i].size, 0.0);
	}
}

void LayerTrainingData::reset() {
	for (int i = 0; i < activation_inputs.size(); i++) {
		std::fill(activation_inputs[i].begin(), activation_inputs[i].end(), 0.0);
		std::fill(deltas[i].begin(), deltas[i].end(), 0.0);
		std::fill(output[i].begin(), output[i].end(), 0.0);
	}
}

void LayerTrainingData::set_activation_input(size_t layer, size_t index, double val) {
	activation_inputs[layer][index] = val;
}
double LayerTrainingData::get_activation_input(size_t layer, size_t index) const {
	return activation_inputs[layer][index];
}
void LayerTrainingData::set_delta(size_t layer, size_t index, double val) {
	deltas[layer][index] = val;
}
double LayerTrainingData::get_delta(size_t layer, size_t index) const {
	return deltas[layer][index];
}
void LayerTrainingData::set_output(size_t layer, size_t index, double val) {
	output[layer][index] = val;
}
double LayerTrainingData::get_output(size_t layer, size_t index) const {
	return output[layer][index];
}
const std::vector<double>& LayerTrainingData::get_full_output(size_t layer) const {
	return output[layer];
}
const std::vector<double>& LayerTrainingData::get_full_activation_inputs(size_t layer) const {
	return activation_inputs[layer];
}
const std::vector<double>& LayerTrainingData::get_full_deltas(size_t layer) const {
	return deltas[layer];
}
//std::vector<double>& Network::get_result()
//{
//	return layers.back().output;
//}

double Network::mean_squared_error() 
{
	double error = 0.0;
	for (const auto& data : training_data) {

		auto output = calculate(data.get_input());
		error += cost(output, data.get_expected());
	}
	return error / training_data.size();
}

void Network::debug() {};

void Network::test()
{
	double correct = 0.0;
	for (const auto& point : test_data) {
		std::vector<double> input = point.get_input();
		auto result = calculate(input);
		if (point.is_correct(result)) {
			correct += 1;
		}
	}
	double rate = correct / test_data.size() * 100;
	std::cout << "RESULT: " << correct << "/" << test_data.size() << " -- " << rate << "%" << std::endl;
	
}

std::vector<double> Network::calculate(const std::vector<double>& input)
{
	std::vector<double> res = layers[0].calculate(input, nullptr);
	for (size_t i = 1; i < layers.size(); i++) {
		res = layers[i].calculate(res, nullptr);
	}
	return res;
}


void Network::calculate(const std::vector<double>& input, LayerTrainingData *layer_training_data)
{
	layers[0].calculate(input, layer_training_data);
	for (size_t i = 1; i < layers.size(); i++) {
		layers[i].calculate(layer_training_data->get_full_output(i - 1), layer_training_data);
	}
	
}

double Network::get_accuracy()
{
	return 0.0;// _training_accuracy;
}
