#pragma once
#include <vector>
#include <deque>
#include <atomic>
#include "DataPoint.h"
#include <mutex>
#include "ThreadPool.h"

//#define SINGLE_THREADED
class Layer;

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
	const std::vector<double>& get_full_activation_inputs(size_t layer) const;
	const std::vector<double>& get_full_deltas(size_t layer) const;

};

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

class Network {
protected:
	double mean_squared_error();
	bool _training = false;
	
	virtual void debug();
public:
	std::vector<Layer> layers;
	std::vector<DataPoint> training_data;
	std::vector<DataPoint> test_data;

	int batch_size = 128;
	double learn_rate = 0.05;

	virtual void build() = 0;
	virtual void load_data() = 0;

	void test();
	std::vector<double> calculate(const std::vector<double>& input);
	void calculate(const std::vector<double>& input, LayerTrainingData* layer_training_data);
	//std::vector<double> &get_result();
	double get_accuracy();

};