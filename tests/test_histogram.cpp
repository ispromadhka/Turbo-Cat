/**
 * TurboCat Histogram Tests
 */

#include <gtest/gtest.h>
#include "turbocat/histogram.hpp"

using namespace turbocat;

TEST(HistogramTest, Creation) {
    Histogram hist(10, 255);
    EXPECT_EQ(hist.n_features(), 10);
    EXPECT_EQ(hist.max_bins(), 255);
}

TEST(HistogramTest, Clear) {
    Histogram hist(5, 100);
    hist.bin(0, 10).grad = 1.5f;
    hist.bin(0, 10).hess = 2.0f;
    
    hist.clear();
    
    EXPECT_FLOAT_EQ(hist.bin(0, 10).grad, 0.0f);
    EXPECT_FLOAT_EQ(hist.bin(0, 10).hess, 0.0f);
}

TEST(HistogramTest, BinAccess) {
    Histogram hist(3, 50);
    
    hist.bin(1, 25).grad = 3.14f;
    hist.bin(1, 25).hess = 2.71f;
    hist.bin(1, 25).count = 42;
    
    EXPECT_FLOAT_EQ(hist.bin(1, 25).grad, 3.14f);
    EXPECT_FLOAT_EQ(hist.bin(1, 25).hess, 2.71f);
    EXPECT_EQ(hist.bin(1, 25).count, 42u);
}

TEST(SplitFinderTest, GainComputation) {
    TreeConfig config;
    config.lambda_l2 = 1.0f;

    SplitFinder finder(config);

    // For a good split, we need gradients that separate well
    // Opposite signs on left/right produce positive gain when they
    // cancel in parent (variance reduction)
    GradientPair left(5.0f, 2.0f, 10);   // positive gradient
    GradientPair right(-5.0f, 2.0f, 5);  // negative gradient
    GradientPair parent = left + right;   // gradients cancel: sum=0

    Float gain = finder.compute_gain_variance(left, right, parent);

    // Gain should be positive for a good split
    // gain = 0.5 * (25/3 + 25/3 - 0/5) = 8.33
    EXPECT_GT(gain, 0.0f);
}
