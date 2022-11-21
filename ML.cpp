// ML.cpp : Defines the entry point for the application.
//

#include "ML.h"
#include "networks/mnist.h"
#include "networks/test.h"

#include "CPUTrainer.h"
#include "matplotlibcpp.h"
#include <thread>

#include "gpu/GPUNetwork.h"

namespace plt = matplotlibcpp;
bool training = false;

void train_task(Network& n) {
	std::cout << "TRAINING" << std::endl;
	//n.train();
	CPUTrainer trainer(n);
	trainer.train();
	std::cout << "DONE" << std::endl;
	training = false;
}

void mnist() 
{
	MNISTNetwork n;
	n.batch_size = 32;
	n.learn_rate = 0.05;
	n.build();
	std::cout << "Loading data" << std::endl;
	n.load_data();
	std::cout << "Training" << std::endl;
	training = true;
	std::thread train_thread(train_task, std::ref(n));
	train_thread.detach();
	//n.train();
	//n.test();
	std::vector<double> x{};
	std::vector<double> y{};
	plt::Plot plot("data");
	plt::ylim(0.0, 1.0);
	double t = 0.0;
	while (training) {
		x.push_back(t);
		double acc = n.get_accuracy();
		y.push_back(acc);
		t += 0.1;
		int xmax = std::ceil(t);
		plt::xlim(0, xmax);
		plot.update(x, y);
		plt::pause(0.1);
	}
}

void test() {
	TestNetwork n;
	//MNISTNetwork n;
	n.learn_rate = 0.5;
	n.build();
	n.load_data();
	CPUTrainer(n).train();
}

void gputest() {
	TestNetwork n;
	//MNISTNetwork n;
	n.build();
	n.load_data();
	CPUTrainer trainer(n);
	
	GPUNetwork g;
	g.init(n);
	g.setup_calculate_only_pipeline(n);

	std::vector<float> output;
	std::vector<float> activations;
	std::vector<float> deltas;


	g.calculate({ &n.training_data[0].data, nullptr, &output, &activations, nullptr });

	std::cout << "output:" << std::endl;
	for (auto v : activations) {
		std::cout << v << std::endl;
	}
	std::cout << "activations:" << std::endl;
	for (auto v : output) {
		std::cout << v << std::endl;
	}

	std::cout << "deltas:" << std::endl;
	for (auto v : deltas) {
		std::cout << v << std::endl;
	}

	std::cout << "EXPECTED:" << std::endl;
	LayerTrainingData ltd(n.layers);
	n.calculate(n.training_data[0].data, &ltd);
	trainer.calculate_deltas(n.training_data[0].data, n.training_data[0].get_expected(), ltd);
	std::cout << "output:" << std::endl;
	for (auto layer : n.layers) {
		for (auto v : ltd.get_full_output(layer.index)) {
			std::cout << v << std::endl;
		}
	}
	std::cout << "activations" << std::endl;
	for (auto layer : n.layers) {
		for (auto v : ltd.get_full_activation_inputs(layer.index)) {
			std::cout << v << std::endl;
		}
	}
	std::cout << "deltas" << std::endl;
	for (auto layer : n.layers) {
		for (auto v : ltd.get_full_deltas(layer.index)) {
			std::cout << v << std::endl;
		}
	}

	//std::cin.get();
}

int main()
{
	gputest();
	//mnist();
	//test();
	return 0;
}

