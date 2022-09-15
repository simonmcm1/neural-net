#include "Network.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <assert.h>
#include <math.h>
#include <functional>
#include <algorithm>
#include <filesystem>
#include "util.h"

double Layer::calculate_node(int node_index, const std::vector<double>& inputs) {
	assert(node_index < size);

	double input = biases[node_index];
	for (int i = 0; i < input_size; i++) {
		int index = node_index * input_size + i;
		input += inputs[i] * weights[index];
	}
	activation_inputs[node_index] = input;

	return sigmoid(input);
}

void Layer::calculate(const std::vector<double>& inputs) {
	for (int node = 0; node < size; node++) {
		output[node] = calculate_node(node, inputs);
	}
}

void Layer::init() 
{
	weights.resize(input_size * size);
	biases.resize(size);
	output.resize(size);
	activation_inputs.resize(size);
	deltas.resize(size);

	for (size_t i = 0; i < size; i++) {
		for (size_t j = 0; j < input_size; j++) {
			weights[i * input_size + j] = random01() / sqrt(input_size);
		}
		biases[i] = random01() / sqrt(input_size);
	}
}

std::vector<double>& Network::get_result()
{
	return layers.back().output;
}

double Network::mean_squared_error() 
{
	double error = 0.0;
	for (const auto& data : training_data) {

		calculate(data.get_input());
		error += cost(get_result(), data.get_expected());
	}
	return error / training_data.size();
}

void Network::process_batch(size_t batch_start, size_t batch_len, 
	std::vector<std::vector<double>> *weight_gradients, 
	std::vector<std::vector<double>> *bias_gradients) 
{
	for (int data_index = 0; data_index < batch_len; data_index++) {

		const auto& data = training_data[batch_start + data_index];
		const std::vector<double> &input = data.get_input();
		const std::vector<double> &expected = data.get_expected();

		calculate(input);
		const auto& result = get_result();

		//backpropogation
		//output layer
		int layer_index = layers.size() - 1;
		auto& out_layer = layers[layer_index];
		assert(expected.size() == out_layer.size);
		for (int node_index = 0; node_index < out_layer.size; node_index++) {
			double o = out_layer.output[node_index];
			double cd = cost_derivative(o, expected[node_index]);
			double ad = sigmoid_derivative(out_layer.activation_inputs[node_index]);
			double delta = cd * ad;
			out_layer.deltas[node_index] = delta;
		}

		//hidden layers
		for (layer_index = layers.size() - 2; layer_index >= 0; layer_index--) {
			auto& layer = layers[layer_index];
			auto& last_layer = layers[layer_index + 1];
			for (int node_index = 0; node_index < layer.size; node_index++) {

				double sum_of_weighted_errors = 0.0;
				for (int last_node_index = 0; last_node_index < last_layer.size; last_node_index++) {
					int weight_index = (last_node_index * last_layer.input_size + node_index);
					double we = last_layer.deltas[last_node_index] * last_layer.weights[weight_index];
					sum_of_weighted_errors += we;
				}
				layer.deltas[node_index] = sum_of_weighted_errors * sigmoid_derivative(layer.activation_inputs[node_index]);
			}
		}

		//feed gradients forward
		const std::vector<double>* cur_input = &input;
		for (layer_index = 0; layer_index < layers.size(); layer_index++) {
			auto& layer = layers[layer_index];

			for (int node_index = 0; node_index < layer.size; node_index++) {
				for (int input_index = 0; input_index < layer.input_size; input_index++) {
					int i = node_index * layer.input_size + input_index;
					double grad = cur_input->at(input_index) * layer.deltas[node_index];
					weight_gradients->at(layer_index)[i] += grad;
				}
				bias_gradients->at(layer_index)[node_index] += 1 * layer.deltas[node_index];
			}
			cur_input = &layer.output;
		}
	}
}

void Network::train() 
{
	std::vector<std::vector<double>> weight_gradients;
	std::vector<std::vector<double>> bias_gradients;
	weight_gradients.resize(layers.size());
	bias_gradients.resize(layers.size());
	for (int i = 0; i < layers.size(); i++) {
		weight_gradients[i].resize(layers[i].weights.size(), 0.0);
		bias_gradients[i].resize(layers[i].biases.size(), 0.0);
	}
	_training = true;
	while (_training) {
		_training_accuracy = test_training_accuracy();
		std::cout << "a" << _training_accuracy << std::endl;

		for (int batch_index = 0; batch_index < training_data.size(); batch_index += batch_size) {
			//reset all the gradients
			for (int i = 0; i < layers.size(); i++) {
				//weight_gradients[i].resize(layers[i].weights.size(), 0.0);
				//bias_gradients[i].resize(layers[i].biases.size(), 0.0);
				for (int w = 0; w < layers[i].weights.size(); w++) {
					weight_gradients[i][w] = 0.0;
				}
				for (int b = 0; b < layers[i].biases.size(); b++) {
					bias_gradients[i][b] = 0.0;
				}
			}


			int real_batch_size = std::min((size_t)batch_size, training_data.size() - batch_index);
			process_batch(batch_index, real_batch_size, &weight_gradients, &bias_gradients);

			//now apply all the gradients
			for (int layer_index = 0; layer_index < layers.size(); layer_index++) {
				auto& layer = layers[layer_index];
				for (int weight_index = 0; weight_index < layer.weights.size(); weight_index++) {
					double original = layer.weights[weight_index];
					layer.weights[weight_index] = original - weight_gradients[layer_index][weight_index] * learn_rate / real_batch_size;
					if (layer_index == 1 && weight_index == 0) {
					//	std::cout << "WTF: " << layer.weights[weight_index] << std::endl;
					}
				}
				for (int bias_index = 0; bias_index < layer.biases.size(); bias_index++) {
					double original = layer.biases[bias_index];
					layer.biases[bias_index] = original - bias_gradients[layer_index][bias_index] * learn_rate / real_batch_size;
				}
			}

			//_training_accuracy = (double)correct / real_batch_size;
			//std::cout << "CORRECT: " << correct << std::endl;
			//std::cout << "pb" << std::endl;
			//debug();
		}
		//debug();

		//error = error / training_data.size();
	}
}

void Network::debug() {};

double Network::test_training_accuracy() {
	int correct = 0;
	for (const auto& point : training_data) {
		calculate(point.get_input());
		if (point.is_correct(get_result())) {
			correct++;
		}
	}
	return (double)correct / training_data.size();
}

void Network::test()
{
	double correct = 0.0;
	for (const auto& point : test_data) {
		std::vector<double> input = point.get_input();
		calculate(input);
		auto& result = get_result();
		if (point.is_correct(result)) {
			correct += 1;
		}
	}
	double rate = correct / test_data.size() * 100;
	std::cout << "RESULT: " << correct << "/" << test_data.size() << " -- " << rate << "%" << std::endl;
}

void Network::calculate(const std::vector<double>& input)
{
	layers[0].calculate(input);
	for (size_t i = 1; i < layers.size(); i++) {
		layers[i].calculate(layers[i-1].output);
	}
}

double Network::get_accuracy()
{
	return _training_accuracy;
}
