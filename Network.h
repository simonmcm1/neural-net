#pragma once
#include <vector>
#include <deque>
#include <atomic>
#include "DataPoint.h"
#include <mutex>
#include "ThreadPool.h"

//#define SINGLE_THREADED

class LayerTrainingData;

class Layer {
public:
	int input_size;
	int size;
	int index;
	std::vector<double> weights;
	std::vector<double> biases;

	void init();
	double calculate_node(int node_index, const std::vector<double>& inputs);
	std::vector<double> calculate(const std::vector<double>& inputs, LayerTrainingData* training_data);
};

class LayerTrainingData {
private:
	std::vector<std::vector<double>> activation_inputs;
	std::vector<std::vector<double>> deltas;
	std::vector<std::vector<double>> output;
public:
	LayerTrainingData(const std::vector<Layer>& layers);
	void reset();

	void set_activation_input(size_t layer, size_t index, double val);
	double get_activation_input(size_t layer, size_t index) const;

	void set_delta(size_t layer, size_t index, double val);
	double get_delta(size_t layer, size_t index) const;

	void set_output(size_t layer, size_t index, double val);
	double get_output(size_t layer, size_t index) const;

	const std::vector<double>& get_full_output(size_t layer) const;
};

class Gradients {
private:
	std::vector<std::vector<double>> weight_gradients;
	std::vector<std::vector<double>> bias_gradients;

public:
	Gradients() = delete;
	Gradients(const std::vector<Layer>& layers);
	void reset();

	double get_weight(size_t layer, size_t index);
	double get_bias(size_t layer, size_t index);
	void add_to_weight(size_t layer, size_t index, double delta);
	void add_to_bias(size_t layer, size_t index, double delta);
};

class Network {
protected:
	std::vector<std::unique_ptr<Gradients>> per_thread_gradients;
	std::vector<LayerTrainingData> per_thread_training_data;

	std::vector<Layer> layers;
	std::vector<DataPoint> training_data;
	std::vector<DataPoint> test_data;
	double mean_squared_error();
	double _training_accuracy = 0.0;

	bool _training = false;
	
	
	virtual void debug();

	ThreadPool thread_pool;
public:
#ifdef SINGLE_THREADED
	Network() : thread_pool(1) {};
#else
	Network() : thread_pool(std::thread::hardware_concurrency() - 1) {};
#endif

	int batch_size = 32;
	double learn_rate = 0.05;

	virtual void build() = 0;
	virtual void load_data() = 0;
	double test_training_accuracy();

	void process_batch(size_t batch_start, size_t batch_len, Gradients *gradients);
	void train();
	void test();
	std::vector<double> calculate(const std::vector<double>& input);
	void calculate(const std::vector<double>& input, LayerTrainingData* layer_training_data);
	//std::vector<double> &get_result();
	double get_accuracy();

};