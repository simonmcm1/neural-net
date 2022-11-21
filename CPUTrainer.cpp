#include "CPUTrainer.h"
#include <assert.h>
#include <algorithm>
#include <functional>
#include <numeric>
#include "Logging.h"

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
	auto& lw = weight_gradients[layer];
	auto& weight = lw[index];
	weight += delta;
}

void Gradients::add_to_bias(size_t layer, size_t index, double delta)
{
	bias_gradients[layer][index] += delta;
}

double CPUTrainer::test_training_accuracy() {
	Timer t("training_accuracy");

	std::vector<int> correct(_thread_pool.nthreads(), 0);
	batch_function task = [&](size_t thread_index, size_t start_index, size_t count) {
		for (size_t i = start_index; i < start_index + count; i++) {
			const auto& point = _network.training_data[i];
			auto output = _network.calculate(point.get_input());
			if (point.is_correct(output)) {
				correct[thread_index] += 1;
			}
		}
	};
	_thread_pool.batch_jobs(task, _network.training_data.size());
	int total_correct = std::accumulate(correct.begin(), correct.end(), 0);

	return (double)total_correct / _network.training_data.size();
}

void CPUTrainer::calculate_deltas(const std::vector<double>& input, const std::vector<double>& expected, LayerTrainingData &layer_data)
{
	int nlayers = _network.layers.size();
	const std::vector<double>& output = layer_data.get_full_output(nlayers - 1);

	//backpropogation
	//output layer
	int layer_index = nlayers - 1;
	auto& out_layer = _network.layers[layer_index];
	assert(expected.size() == out_layer.size);
	for (int node_index = 0; node_index < out_layer.size; node_index++) {
		double o = output[node_index];
		double cd = cost_derivative(o, expected[node_index]);
		double ad = sigmoid_derivative(layer_data.get_activation_input(nlayers - 1, node_index));
		double delta = cd * ad;
		layer_data.set_delta(nlayers - 1, node_index, delta);
	}

	//hidden layers
	for (layer_index = nlayers - 2; layer_index >= 0; layer_index--) {
		auto& layer = _network.layers[layer_index];

		int last_layer_index = layer_index + 1;
		auto& last_layer = _network.layers[last_layer_index];

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
}

void CPUTrainer::process_batch(size_t batch_start, size_t batch_len, Gradients* gradients)
{
	//Timer t("Network::process_batch");
	if (per_thread_gradients.size() == 0) {
		//initialize on first entry
		for (int i = 0; i < _thread_pool.nthreads(); i++) {
			per_thread_gradients.push_back(std::make_unique<Gradients>(_network.layers));
		}
	}
	else {
		for (auto& gradient : per_thread_gradients) {
			gradient->reset();
		}
	}

	if (per_thread_training_data.size() == 0) {
		//initialize on first entry
		for (int i = 0; i < _thread_pool.nthreads(); i++) {
			per_thread_training_data.push_back(LayerTrainingData(_network.layers));
		}
	}
	else {
		for (auto& training_data : per_thread_training_data) {
			training_data.reset();
		}
	}

	int nlayers = _network.layers.size();
	batch_function task = [&](size_t thread_index, size_t start_index, size_t count) {
		LayerTrainingData& layer_data = per_thread_training_data.at(thread_index);

		for (int data_index = start_index; data_index < start_index + count; data_index++) {

			const auto& data = _network.training_data[batch_start + data_index];
			const std::vector<double>& input = data.get_input();
			const std::vector<double>& expected = data.get_expected();

			_network.calculate(input, &layer_data);

			calculate_deltas(input, expected, layer_data);

			//feed gradients forward
			const std::vector<double>* cur_input = &input;
			for (size_t layer_index = 0; layer_index < nlayers; layer_index++) {
				auto& layer = _network.layers[layer_index];

				for (int node_index = 0; node_index < layer.size; node_index++) {
					for (int input_index = 0; input_index < layer.input_size; input_index++) {
						int i = node_index * layer.input_size + input_index;
						double grad = cur_input->at(input_index) * layer_data.get_delta(layer_index, node_index);
						per_thread_gradients.at(thread_index)->add_to_weight(layer_index, i, grad);
					}
					per_thread_gradients.at(thread_index)->add_to_bias(layer_index, node_index, 1 * layer_data.get_delta(layer_index, node_index));
				}
				cur_input = &layer_data.get_full_output(layer_index);
			}
		}
	};

	_thread_pool.batch_jobs(task, batch_len);
	

	//add up all the per-thread results
	
	for (auto& per_thread : per_thread_gradients) {
		for (int layer_index = 0; layer_index < nlayers; layer_index++) {
			batch_function post_process = [&](size_t thread_index, size_t start_index, size_t count) {
				for (size_t i = start_index; i < start_index + count; i++) {
					double w = per_thread->get_weight(layer_index, i);
					gradients->add_to_weight(layer_index, i, w);
				}
			};
			_thread_pool.batch_jobs(post_process, _network.layers.at(layer_index).weights.size());

			for (int bias_index = 0; bias_index < _network.layers.at(layer_index).biases.size(); bias_index++) {
				gradients->add_to_bias(layer_index, bias_index, per_thread->get_bias(layer_index, bias_index));
			}
		}
	}
}

void CPUTrainer::train()
{
	Gradients gradients(_network.layers);

	bool training = true;
	Timer epoch_timer("Epoch");
	while (training) {
		_training_accuracy = test_training_accuracy();
		LOG_DEBUG("Training Accuracy: {}", _training_accuracy);
		epoch_timer.reset();
		for (int batch_index = 0; batch_index < _network.training_data.size(); batch_index += _network.batch_size) {
			//reset all the gradients
			gradients.reset();

			int real_batch_size = std::min((size_t)_network.batch_size, _network.training_data.size() - batch_index);
			process_batch(batch_index, real_batch_size, &gradients);

			//now apply all the gradients
			for (int layer_index = 0; layer_index < _network.layers.size(); layer_index++) {
				auto& layer = _network.layers[layer_index];
				for (int weight_index = 0; weight_index < layer.weights.size(); weight_index++) {
					double original = layer.weights[weight_index];
					layer.weights[weight_index] = original - gradients.get_weight(layer_index, weight_index) * _network.learn_rate / real_batch_size;
				}
				for (int bias_index = 0; bias_index < layer.biases.size(); bias_index++) {
					double original = layer.biases[bias_index];
					layer.biases[bias_index] = original - gradients.get_bias(layer_index, bias_index) * _network.learn_rate / real_batch_size;
				}
			}
			//debug();
		}
		epoch_timer.end();
		Timer::print_usage_report();
		//debug();
	}
}