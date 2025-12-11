/**
 * TurboCat GradTree Tests
 */

#include <gtest/gtest.h>
#include "turbocat/tree.hpp"

using namespace turbocat;

TEST(GradTreeTest, Construction) {
    TreeConfig config;
    config.max_depth = 4;
    
    GradTree tree(config, 10);
    
    EXPECT_EQ(tree.depth(), 4);
    EXPECT_EQ(tree.n_internal_nodes(), 15u);  // 2^4 - 1
    EXPECT_EQ(tree.n_leaves(), 16u);          // 2^4
}

TEST(GradTreeTest, RandomInitialization) {
    TreeConfig config;
    config.max_depth = 3;
    
    GradTree tree(config, 5);
    tree.initialize_random(42);
    
    // Check that parameters are initialized
    EXPECT_EQ(tree.feature_weights().rows(), 7);  // 2^3 - 1
    EXPECT_EQ(tree.feature_weights().cols(), 5);
    EXPECT_EQ(tree.thresholds().size(), 7);
    EXPECT_EQ(tree.leaf_values().size(), 8);     // 2^3
}

TEST(GradTreeTest, HardPrediction) {
    TreeConfig config;
    config.max_depth = 2;
    
    GradTree tree(config, 3);
    tree.initialize_random(123);
    
    std::vector<float> features = {0.5f, -0.5f, 1.0f};
    Float pred = tree.predict_hard(features.data());
    
    // Should return some value (exact value depends on random init)
    EXPECT_TRUE(std::isfinite(pred));
}
