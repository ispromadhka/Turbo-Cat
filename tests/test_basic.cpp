/**
 * TurboCat Tests
 */

#include <gtest/gtest.h>
#include "turbocat/turbocat.hpp"

// ============================================================================
// Types Tests
// ============================================================================

TEST(TypesTest, GradientPairAddition) {
    turbocat::GradientPair a(1.0f, 2.0f, 10);
    turbocat::GradientPair b(3.0f, 4.0f, 20);
    
    auto c = a + b;
    
    EXPECT_FLOAT_EQ(c.grad, 4.0f);
    EXPECT_FLOAT_EQ(c.hess, 6.0f);
    EXPECT_EQ(c.count, 30u);
}

TEST(TypesTest, GradientPairSubtraction) {
    turbocat::GradientPair a(5.0f, 6.0f, 30);
    turbocat::GradientPair b(2.0f, 3.0f, 10);
    
    auto c = a - b;
    
    EXPECT_FLOAT_EQ(c.grad, 3.0f);
    EXPECT_FLOAT_EQ(c.hess, 3.0f);
    EXPECT_EQ(c.count, 20u);
}

TEST(TypesTest, SplitInfoComparison) {
    turbocat::SplitInfo a, b;
    a.gain = 1.5f;
    b.gain = 2.0f;
    
    EXPECT_TRUE(b > a);
    EXPECT_FALSE(a > b);
}

// ============================================================================
// Config Tests
// ============================================================================

TEST(ConfigTest, DefaultValues) {
    turbocat::Config config;
    
    EXPECT_EQ(config.tree.max_depth, 6);
    EXPECT_EQ(config.boosting.n_estimators, 1000u);
    EXPECT_FLOAT_EQ(config.boosting.learning_rate, 0.05f);
}

TEST(ConfigTest, BinaryClassificationPreset) {
    auto config = turbocat::Config::binary_classification();
    
    EXPECT_EQ(config.task, turbocat::TaskType::BinaryClassification);
    EXPECT_EQ(config.loss.loss_type, turbocat::LossType::LogLoss);
}

TEST(ConfigTest, RobustClassificationPreset) {
    auto config = turbocat::Config::robust_classification();
    
    EXPECT_EQ(config.loss.loss_type, turbocat::LossType::RobustFocal);
}

TEST(ConfigTest, Validation) {
    turbocat::Config config;
    
    // Valid config should not throw
    EXPECT_NO_THROW(config.validate());
    
    // Invalid max_depth should throw
    config.tree.max_depth = 100;
    EXPECT_THROW(config.validate(), std::invalid_argument);
}

// ============================================================================
// Version Tests
// ============================================================================

TEST(VersionTest, VersionInfo) {
    EXPECT_EQ(turbocat::Version::major, 0);
    EXPECT_EQ(turbocat::Version::minor, 1);
    EXPECT_EQ(turbocat::Version::patch, 0);
    EXPECT_STREQ(turbocat::Version::string, "0.1.0");
}

// Main
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
