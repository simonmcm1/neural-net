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
	std::vector<double> output(size);
	for (int node = 0; node < size; node++) {
		double weighted_input = calculate_node(node, inputs);
		double res = sigmoid(weighted_input);
		if (training_data != nullptr) {
			training_data->activation_inputs[node] = weighted_input;
			training_data->output[node] = res;
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

Gradients::Gradients(std::vector<Layer>& layers)
{
	weight_gradients.resize(layers.size());
	bias_gradients.resize(layers.size());
	for (int i = 0; i < layers.size(); i++) {
		weight_gradients[i].resize(layers[i].weights.size(), 0.0);
		bias_gradients[i].resize(layers[i].biases.size(), 0.0);
	}
}

void Gradients::reset() {
	for (size_t layer = 0; layer < weight_gradients.size(); layer++) {
		std::fill(weight_gradients[layer].begin(), weight_gradients[layer].end(), 0.0);
		std::fill(bias_gradients[layer].begin(), bias_gradients[layer].end(), 0.0);
	}
}

void Gradients::lock()
{
	_mutex.lock();
}

void Gradients::release()
{
	_mutex.unlock();
}

double Gradients::get_weight(size_t layer, size_t index)
{
	return weight_gradients.at(layer).at(index);
}

double Gradients::get_bias(size_t layer, size_t index)
{
	return bias_gradients.at(layer).at(index);
}

void Gradients::add_to_weight(size_t layer, size_t index, double delta)
{
	weight_gradients.at(layer)[index] += delta;
}

void Gradients::add_to_bias(size_t layer, size_t index, double delta)
{
	bias_gradients.at(layer)[index] += delta;
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

void Network::process_batch(size_t batch_start, size_t batch_len, Gradients *gradients) 
{
	int nthreads = 4;
	batch_function task = [&](size_t thread_index, size_t start_index, size_t count) {
		//todo: reuse?
		std::vector<LayerTrainingData> layer_data(layers.size());
		for (int layer = 0; layer < layers.size(); layer++) {
			layer_data[layer].activation_inputs.resize(layers[layer].size);
			layer_data[layer].deltas.resize(layers[layer].size);
			layer_data[layer].output.resize(layers[layer].size);
		}

		for (int data_index = start_index; data_index < count; data_index++) {

			const auto& data = training_data[batch_start + data_index];
			const std::vector<double>& input = data.get_input();
			const std::vector<double>& expected = data.get_expected();

			calculate(input, layer_data);
			std::vector<double>& output = layer_data[layers.size() - 1].output;
			//const auto& result = get_result();

			//backpropogation
			//output layer
			int layer_index = layers.size() - 1;
			auto& out_layer = layers[layer_index];
			assert(expected.size() == out_layer.size);
			for (int node_index = 0; node_index < out_layer.size; node_index++) {
				double o = output[node_index];
				double cd = cost_derivative(o, expected[node_index]);
				double ad = sigmoid_derivative(layer_data[layers.size() - 1].activation_inputs[node_index]);
				double delta = cd * ad;
				layer_data[layers.size() - 1].deltas[node_index] = delta;
			}

			//hidden layers
			for (layer_index = layers.size() - 2; layer_index >= 0; layer_index--) {
				auto& layer = layers[layer_index];
				auto& this_layer_data = layer_data[layer_index];
				auto& last_layer = layers[layer_index + 1];
				auto& last_layer_data = layer_data[layer_index + 1];
				for (int node_index = 0; node_index < layer.size; node_index++) {

					double sum_of_weighted_errors = 0.0;
					for (int last_node_index = 0; last_node_index < last_layer.size; last_node_index++) {
						int weight_index = (last_node_index * last_layer.input_size + node_index);
						double we = last_layer_data.deltas[last_node_index] * last_layer.weights[weight_index];
						sum_of_weighted_errors += we;
					}
					this_layer_data.deltas[node_index] = sum_of_weighted_errors * sigmoid_derivative(this_layer_data.activation_inputs[node_index]);
				}
			}

			//feed gradients forward
			const std::vector<double>* cur_input = &input;
			gradients->lock();
			for (layer_index = 0; layer_index < layers.size(); layer_index++) {
				auto& layer = layers[layer_index];
				auto& this_layer_data = layer_data[layer_index];

				for (int node_index = 0; node_index < layer.size; node_index++) {
					for (int input_index = 0; input_index < layer.input_size; input_index++) {
						int i = node_index * layer.input_size + input_index;
						double grad = cur_input->at(input_index) * this_layer_data.deltas[node_index];
						gradients->add_to_weight(layer_index, i, grad);
					}
					gradients->add_to_bias(layer_index, node_index, 1 * this_layer_data.deltas[node_index]);
				}
				cur_input = &this_layer_data.output;
			}
			gradients->release();
		}
	};
	thread_pool.batch_jobs(task, batch_len);
}

void Network::train() 
{
	
	Gradients gradients(layers);

	_training = true;
	Timer epoch_timer("Epoch");
	while (_training) {
		_training_accuracy = test_training_accuracy();
		std::cout << "training_accuracy" << _training_accuracy << std::endl;
		epoch_timer.reset();
		for (int batch_index = 0; batch_index < training_data.size(); batch_index += batch_size) {
			//reset all the gradients
			gradients.reset();


			int real_batch_size = std::min((size_t)batch_size, training_data.size() - batch_index);
			process_batch(batch_index, real_batch_size, &gradients);

			//now apply all the gradients
			for (int layer_index = 0; layer_index < layers.size(); layer_index++) {
				auto& layer = layers[layer_index];
				for (int weight_index = 0; weight_index < layer.weights.size(); weight_index++) {
					double original = layer.weights[weight_index];
					layer.weights[weight_index] = original - gradients.get_weight(layer_index, weight_index) * learn_rate / real_batch_size;
					if (layer_index == 1 && weight_index == 0) {
					//	std::cout << "WTF: " << layer.weights[weight_index] << std::endl;
					}
				}
				for (int bias_index = 0; bias_index < layer.biases.size(); bias_index++) {
					double original = layer.biases[bias_index];
					layer.biases[bias_index] = original - gradients.get_bias(layer_index, bias_index) * learn_rate / real_batch_size;
				}
			}
			//_training_accuracy = (double)correct / real_batch_size;
			//std::cout << "CORRECT: " << correct << std::endl;
			//std::cout << "pb" << std::endl;
			//debug();
		}
		epoch_timer.end();
		//debug();

		//error = error / training_data.size();
	}
}

void Network::debug() {};

double Network::test_training_accuracy() {
	Timer t("training_accuracy");
	int nthreads = 4;
	std::vector<int> correct(nthreads);

	batch_function task = [&](size_t thread_index, size_t start_index, size_t count) {
		for (size_t i = start_index; i < start_index + count; i++) {
			const auto &point = training_data[i];
			auto output = calculate(point.get_input());
			if (point.is_correct(output)) {
				correct[thread_index] += 1;
			}
		}
	};
	batch_jobs(task, nthreads, training_data.size());
	int total_correct = std::accumulate(correct.begin(), correct.end(), 0);

	return (double)total_correct / training_data.size();
}

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

void Network::calculate(const std::vector<double>& input, std::vector<LayerTrainingData> &training_data)
{
	layers[0].calculate(input, &training_data[0]);
	for (size_t i = 1; i < layers.size(); i++) {
		layers[i].calculate(training_data[i-1].output, &training_data[i]);
	}
}

double Network::get_accuracy()
{
	return _training_accuracy;
}
