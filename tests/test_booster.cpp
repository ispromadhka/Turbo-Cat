/**
 * TurboCat Booster Tests
 */

#include <gtest/gtest.h>
#include "turbocat/booster.hpp"

using namespace turbocat;

TEST(BoosterTest, DefaultConstruction) {
    Booster booster;
    
    EXPECT_EQ(booster.n_trees(), 0u);
}

TEST(BoosterTest, ConfigConstruction) {
    Config config = Config::binary_classification();
    config.boosting.n_estimators = 100;
    
    Booster booster(config);
    
    EXPECT_EQ(booster.n_trees(), 0u);
    EXPECT_EQ(booster.config().boosting.n_estimators, 100u);
}

TEST(BoosterTest, FeatureImportanceEmpty) {
    Booster booster;
    
    auto importance = booster.feature_importance();
    
    // Empty model should have zero importance
    for (Float imp : importance.gain) {
        EXPECT_FLOAT_EQ(imp, 0.0f);
    }
}
