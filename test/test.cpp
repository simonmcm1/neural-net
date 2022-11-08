#include <gtest/gtest.h>

#include "../networks/test.h"
#include "../networks/mnist.h"
#include "../gpu/GPUNetwork.h"

// Demonstrate some basic assertions.
TEST(GPUCompute, TestNetwork) {
	TestNetwork n;
	n.build();
	n.load_data();
	GPUNetwork g;
	g.init(n);

	auto expected = n.calculate(n.training_data[0].data);
	std::vector<float> result;
	g.calculate(n.training_data[0].data, result);

	EXPECT_EQ(result.size(), expected.size());
	for (int i = 0; i < result.size(); i++) {
		EXPECT_FLOAT_EQ(result[i], expected[i]);
	}
	g.destroy();
}

TEST(GPUCompute, MNISTNetwork) {
	MNISTNetwork n;
	n.build();
	n.load_data();
	GPUNetwork g;
	g.init(n);

	auto expected = n.calculate(n.training_data[0].data);
	std::vector<float> result;
	g.calculate(n.training_data[0].data, result);

	EXPECT_EQ(result.size(), expected.size());
	for (int i = 0; i < result.size(); i++) {
		EXPECT_FLOAT_EQ(result[i], expected[i]);
	}
	g.destroy();
}