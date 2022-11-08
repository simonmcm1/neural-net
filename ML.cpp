// ML.cpp : Defines the entry point for the application.
//

#include "ML.h"
#include "networks/mnist.h"
#include "networks/test.h"
#include "networks/seb.h"

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
	//SebNetwork n;
	n.build();
	n.load_data();
	GPUNetwork g;
	g.init(n);

	std::vector<float> gout;
	g.calculate(n.training_data[0].data, gout);
	std::cout << "RES:" << std::endl;
	for (auto v : gout) {
		std::cout << v << std::endl;
	}

	std::cout << "EXPECTED:" << std::endl;
	auto res = n.calculate(n.training_data[0].data);
	for (auto v : res) {
		std::cout << v << std::endl;
	}
	//std::cin.get();
}

int main()
{
	gputest();
	//mnist();
	//test();
	//seb();
	return 0;
}

