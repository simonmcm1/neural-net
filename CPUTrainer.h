#pragma once
#include <vector>
#include "Network.h"
#include "ThreadPool.h"
#include "util.h"
#include "Timer.h"
#include <memory>

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

class CPUTrainer {
private:
	std::vector<std::unique_ptr<Gradients>> per_thread_gradients;
	std::vector<LayerTrainingData> per_thread_training_data;

	ThreadPool _thread_pool;
	Network& _network;

	double _training_accuracy = 0.0;

public:
#ifdef SINGLE_THREADED
	CPUTrainer(Network& network) : _network(network), _thread_pool(1) {};
#else
	CPUTrainer(Network &network) : _thread_pool(std::thread::hardware_concurrency() - 2), _network(network) {};
#endif
	double test_training_accuracy();
	void process_batch(size_t batch_start, size_t batch_len, Gradients* gradients);
	void train();
};