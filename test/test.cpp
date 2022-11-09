#include <gtest/gtest.h>

#include "../networks/test.h"
#include "../networks/mnist.h"
#include "../gpu/GPUNetwork.h"

TEST(GPUCompute, TestNetwork) {
	TestNetwork n;
	n.build();
	n.load_data();
	GPUNetwork g;
	g.init(n);

	LayerTrainingData expected(n.layers);
	n.calculate(n.training_data[0].data, &expected);
	std::vector<float> result;
	g.calculate(n.training_data[0].data, result);

	int ri = 0;
	for (int l = 0; l < n.layers.size(); l++) {
		auto o = expected.get_full_output(l);
		for (int i = 0; i < n.layers[l].size; i++) {
			EXPECT_FLOAT_EQ(result[ri++], o[i]);
		}
	}

	g.destroy();
}


TEST(GPUCompute, MNISTNetwork) {
	MNISTNetwork n;
	n.build();
	n.load_data();
	GPUNetwork g;
	g.init(n);

	uint32_t di = 0;
	for (auto d : n.training_data) {
		LayerTrainingData expected(n.layers);
		n.calculate(d.data, &expected);
		std::vector<float> result;
		g.calculate(d.data, result);

		int ri = 0;
		for (int l = 0; l < n.layers.size(); l++) {
			auto o = expected.get_full_output(l);
			for (int i = 0; i < n.layers[l].size; i++) {
				ASSERT_NEAR(result[ri++], o[i], 0.0001) << "index " << i << " on case " << di;
			}
		}
		di++;
	}

	g.destroy();
}

