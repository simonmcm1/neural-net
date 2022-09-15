#pragma once
#include "../DataPoint.h"
#include "../Network.h"

class MNISTNetwork : public Network {
public:
	void build() override;
	void load_data() override;
};