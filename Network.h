#pragma once
#include <vector>
#include <deque>
#include <atomic>
#include "DataPoint.h"
#include <mutex>

class LayerTrainingData {
public:
	std::vector<double> activation_inputs;
	std::vector<double> deltas;
	std::vector<double> output;
};

class Layer {
public:
	int input_size;
	int size;
	std::vector<double> weights;
	std::vector<double> biases;

	void init();
	double calculate_node(int node_index, const std::vector<double>& inputs);
	std::vector<double> calculate(const std::vector<double>& inputs, LayerTrainingData *training_data);
};

class Gradients {
private:
	std::vector<std::vector<double>> weight_gradients;
	std::vector<std::vector<double>> bias_gradients;
	std::mutex _mutex;

public:
	Gradients() = delete;
	Gradients(std::vector<Layer>& layers);
	void reset();

	void lock();
	void release();

	double get_weight(size_t layer, size_t index);
	double get_bias(size_t layer, size_t index);
	void add_to_weight(size_t layer, size_t index, double delta);
	void add_to_bias(size_t layer, size_t index, double delta);
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

	void process_batch(size_t batch_start, size_t batch_len, Gradients *gradients);
	void train();
	void test();
	std::vector<double> calculate(const std::vector<double>& input);
	void calculate(const std::vector<double>& input, std::vector<LayerTrainingData>& training_data);
	//std::vector<double> &get_result();
	double get_accuracy();

};