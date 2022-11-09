#include "test.h"
#include "../util.h"
#include <assert.h>

void TestNetwork::build()
{
	layers.resize(2);
	Layer* hidden = &layers[0];
	hidden->input_size = 2;
	hidden->size = 2;
	hidden->init();
	hidden->weights[0] = 0.15;
	hidden->weights[1] = 0.2;
	hidden->weights[2] = 0.25;
	hidden->weights[3] = 0.3;
	hidden->biases[0] = 0.35;
	hidden->biases[1] = 0.35;


	Layer* output = &layers[1];
	output->input_size = hidden->size;
	output->size = 2;
	output->init();
	output->weights[0] = 0.4;
	output->weights[1] = 0.45;
	output->weights[2] = 0.5;
	output->weights[3] = 0.55;
	output->biases[0] = 0.60;
	output->biases[1] = 0.60;

	for (int i = 0; i < layers.size(); i++) {
		layers[i].index = i;
	}
}

void TestNetwork::load_data() {
	training_data.resize(1, {});


	DataPoint& point = training_data[0];
	point.label = 1;
	point.data.push_back(0.05);
	point.data.push_back(0.1);
	point.expected.push_back(0.01);
	point.expected.push_back(0.99);
}