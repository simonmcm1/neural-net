#pragma once
#include "../DataPoint.h"
#include "../Network.h"

class TestNetwork : public Network {
public:
	TestNetwork() {
		batch_size = 1;
	}
	void build() override;
	void load_data() override;
};