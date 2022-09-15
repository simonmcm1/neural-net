#include "DataPoint.h"


const std::vector<double>& DataPoint::get_input() const
{
	return data;
}

const std::vector<double>& DataPoint::get_expected() const
{
	return expected;
}

void DataPoint::set_expected_from_label(size_t output_len)
{
	expected.resize(output_len, 0.0);
	expected[label] = 1.0;
}

bool DataPoint::is_correct(const std::vector<double>& output) const {
	uint8_t maxi = 0;
	double max = -9999999;
	for (int i = 0; i < output.size(); i++) {
		if (output[i] > max) {
			maxi = i;
			max = output[i];
		}
	}
	return maxi == label;
}