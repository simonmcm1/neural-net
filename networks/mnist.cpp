#include "mnist.h"
#include "../util.h"
#include <assert.h>
#include <iostream>

void MNISTNetwork::build()
{
	layers.resize(2);
	Layer *hidden = &layers[0];
	hidden->input_size = 28*28;
	hidden->size = 300;
	hidden->init();

	Layer *output = &layers[1];
	output->input_size = hidden->size;
	output->size = 10;
	output->init();

	for (int i = 0; i < layers.size(); i++) {
		layers[i].index = i;
	}
}

void MNISTNetwork::load_data() {
	std::vector<uint8_t> labels_buffer;
	read_file(std::string(DATA_ROOT) + "train-labels.idx1-ubyte", labels_buffer);
	uint32_t magic = from_big_endian(&labels_buffer[0]);
	assert(magic == 0x801);

	std::vector<uint8_t> data_buffer;
	read_file(std::string(DATA_ROOT) + "train-images.idx3-ubyte", data_buffer);
	magic = from_big_endian(&data_buffer[0]);
	assert(magic == 0x803);

	uint32_t label_len = from_big_endian(&labels_buffer[4]);
	uint32_t data_len = from_big_endian(&data_buffer[4]);
	assert(label_len == data_len);

	uint32_t rows = from_big_endian(&data_buffer[8]);
	uint32_t cols = from_big_endian(&data_buffer[12]);

	training_data.resize(data_len, {});

	size_t label_pos = 8;
	size_t data_pos = 16;
	for (size_t i = 0; i < data_len; i++) {
		DataPoint& point = training_data[i];
		point.label = labels_buffer[label_pos++];
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				point.data.push_back(data_buffer[data_pos++] / 255.0);
			}
		}
		point.set_expected_from_label(10);
		//stdf::cout << point.label << std::endl;
	}
}