#pragma once
#include <vector>
#include <cstdint>

class DataPoint {
public:
	std::vector<double> data;
	std::vector<double> expected;
	uint32_t label;

	const std::vector<double>& get_input() const;
	const std::vector<double>& get_expected() const;
	bool is_correct(const std::vector<double>& output) const;

	void set_expected_from_label(size_t output_len);
};