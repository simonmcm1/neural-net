#pragma once
#include "../DataPoint.h"
#include "../Network.h"

class MNISTNetwork : public Network {
public:
	void build() override;
	void load_data() override;
#ifdef _WIN32
	static constexpr auto DATA_ROOT = "../../../data/mnist/";
#else
	static constexpr auto DATA_ROOT = "../data/mnist/";
#endif
};