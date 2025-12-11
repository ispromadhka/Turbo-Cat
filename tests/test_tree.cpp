/**
 * TurboCat Tree Tests
 */

#include <gtest/gtest.h>
#include "turbocat/tree.hpp"

using namespace turbocat;

TEST(TreeTest, DefaultConstruction) {
    TreeConfig config;
    Tree tree(config);
    
    EXPECT_EQ(tree.n_nodes(), 0u);
    EXPECT_EQ(tree.n_leaves(), 0u);
}

TEST(TreeNodeTest, DefaultValues) {
    TreeNode node;
    
    EXPECT_EQ(node.is_leaf, 1);
    EXPECT_EQ(node.split_feature, 0);
    EXPECT_FLOAT_EQ(node.value, 0.0f);
}

TEST(TreeEnsembleTest, EmptyPrediction) {
    TreeEnsemble ensemble;
    
    std::vector<float> features = {1.0f, 2.0f, 3.0f};
    Float pred = ensemble.predict(features.data(), 3);
    
    EXPECT_FLOAT_EQ(pred, 0.0f);
}
