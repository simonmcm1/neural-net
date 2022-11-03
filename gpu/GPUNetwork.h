#pragma once
#include "../Network.h"

class GPUNetwork {
private:
	std::vector<float> weights;
	std::vector<float> input;
	std::vector<float> activated;
public:
	void init(Network& network);
	std::vector<float> calculate(const std::vector<double>& input_data);
};