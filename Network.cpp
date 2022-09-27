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


Gradients::Gradients(const std::vector<Layer>& layers)
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
	//Timer t("Network::process_batch");
	if (per_thread_gradients.size() == 0) {
		//initialize on first entry
		for (int i = 0; i < thread_pool.nthreads(); i++) {
			per_thread_gradients.push_back(std::make_unique<Gradients>(layers));
		}
	}
	else {
		for (auto& gradient : per_thread_gradients) {
			gradient->reset();
		}
	}

	if (per_thread_training_data.size() == 0) {
		//initialize on first entry
		for (int i = 0; i < thread_pool.nthreads(); i++) {
			per_thread_training_data.push_back(LayerTrainingData(layers));
		}
	}
	else {
		for (auto& training_data : per_thread_training_data) {
			training_data.reset();
		}
	}
	
	batch_function task = [&](size_t thread_index, size_t start_index, size_t count) {
		LayerTrainingData &layer_data = per_thread_training_data.at(thread_index);

		for (int data_index = start_index; data_index < start_index + count; data_index++) {

			const auto& data = training_data[batch_start + data_index];
			const std::vector<double>& input = data.get_input();
			const std::vector<double>& expected = data.get_expected();

			calculate(input, &layer_data);
			const std::vector<double>& output = layer_data.get_full_output(layers.size() - 1);
			//const auto& result = get_result();

			//backpropogation
			//output layer
			int layer_index = layers.size() - 1;
			auto& out_layer = layers[layer_index];
			assert(expected.size() == out_layer.size);
			for (int node_index = 0; node_index < out_layer.size; node_index++) {
				double o = output[node_index];
				double cd = cost_derivative(o, expected[node_index]);
				double ad = sigmoid_derivative(layer_data.get_activation_input(layers.size() - 1, node_index));
				double delta = cd * ad;
				layer_data.set_delta(layers.size() - 1, node_index, delta);
			}

			//hidden layers
			for (layer_index = layers.size() - 2; layer_index >= 0; layer_index--) {
				auto& layer = layers[layer_index];
				//auto& this_layer_data = layer_data[layer_index];
				int last_layer_index = layer_index + 1;
				auto& last_layer = layers[last_layer_index];
				
				//auto& last_layer_data = layer_data[layer_index + 1];
				for (int node_index = 0; node_index < layer.size; node_index++) {

					double sum_of_weighted_errors = 0.0;
					for (int last_node_index = 0; last_node_index < last_layer.size; last_node_index++) {
						int weight_index = (last_node_index * last_layer.input_size + node_index);
						double we = layer_data.get_delta(last_layer_index, last_node_index) * last_layer.weights[weight_index];
						sum_of_weighted_errors += we;
					}
					double new_delta = sum_of_weighted_errors * sigmoid_derivative(layer_data.get_activation_input(layer_index, node_index));
					layer_data.set_delta(layer_index, node_index, new_delta);
				}
			}

			//feed gradients forward
			const std::vector<double>* cur_input = &input;
			for (layer_index = 0; layer_index < layers.size(); layer_index++) {
				auto& layer = layers[layer_index];

				for (int node_index = 0; node_index < layer.size; node_index++) {
					for (int input_index = 0; input_index < layer.input_size; input_index++) {
						int i = node_index * layer.input_size + input_index;
						double grad = cur_input->at(input_index) * layer_data.get_delta(layer_index, node_index);
						per_thread_gradients.at(thread_index)->add_to_weight(layer_index, i, grad);
					}
					per_thread_gradients.at(thread_index)->add_to_bias(layer_index, node_index, 1 * layer_data.get_delta(layer_index, node_index));
				}
				//cur_input = &layer_data.get_full_output(layer_index);
			}
		}
	};

	thread_pool.batch_jobs(task, batch_len);

	//add up all the per-thread results
	for (auto& per_thread : per_thread_gradients) {
		for (int layer_index = 0; layer_index < layers.size(); layer_index++) {
			for (int weight_index = 0; weight_index < layers.at(layer_index).weights.size(); weight_index++) {
				double w = per_thread->get_weight(layer_index, weight_index);
				gradients->add_to_weight(layer_index, weight_index, w);
			}
			for (int bias_index = 0; bias_index < layers.at(layer_index).biases.size(); bias_index++) {
				gradients->add_to_bias(layer_index, bias_index, per_thread->get_bias(layer_index, bias_index));
			}
		}
	}
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
				}
				for (int bias_index = 0; bias_index < layer.biases.size(); bias_index++) {
					double original = layer.biases[bias_index];
					layer.biases[bias_index] = original - gradients.get_bias(layer_index, bias_index) * learn_rate / real_batch_size;
				}
			}
			//debug();
		}
		epoch_timer.end();
		Timer::print_usage_report();
		//debug();
	}
}

void Network::debug() {};

double Network::test_training_accuracy() {
	Timer t("training_accuracy");

	std::vector<int> correct(thread_pool.nthreads(), 0);
	batch_function task = [&](size_t thread_index, size_t start_index, size_t count) {
		for (size_t i = start_index; i < start_index + count; i++) {
			const auto &point = training_data[i];
			auto output = calculate(point.get_input());
			if (point.is_correct(output)) {
				correct[thread_index] += 1;
			}
		}
	};
	thread_pool.batch_jobs(task, training_data.size());
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

void Network::calculate(const std::vector<double>& input, LayerTrainingData *layer_training_data)
{
	layers[0].calculate(input, layer_training_data);
	for (size_t i = 1; i < layers.size(); i++) {
		layers[i].calculate(layer_training_data->get_full_output(i - 1), layer_training_data);
	}
	
}

double Network::get_accuracy()
{
	return _training_accuracy;
}
