#pragma once
#include <vector>
#include "DataPoint.h"

class Layer {
public:
	int input_size;
	int size;
	std::vector<double> weights;
	std::vector<double> biases;
	std::vector<double> output;

	std::vector<double> activation_inputs;
	std::vector<double> deltas;

	void init();
	double calculate_node(int node_index, const std::vector<double>& inputs);
	void calculate(const std::vector<double>& inputs);
};

class Network {
protected:
	std::vector<Layer> layers;
	std::vector<DataPoint> training_data;
	std::vector<DataPoint> test_data;
	double mean_squared_error();
	double _training_accuracy = 0.0;

	bool _training = false;
	
	
	virtual void debug();
public:
	int batch_size = 32;
	double learn_rate = 0.05;

	virtual void build() = 0;
	virtual void load_data() = 0;
	double test_training_accuracy();

	void process_batch(size_t batch_start, size_t batch_len,
		std::vector<std::vector<double>>* weight_gradients,
		std::vector<std::vector<double>>* bias_gradients);
	void train();
	void test();
	void calculate(const std::vector<double>& input);
	std::vector<double> &get_result();
	double get_accuracy();

};