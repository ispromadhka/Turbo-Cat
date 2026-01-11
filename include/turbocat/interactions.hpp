#pragma once

/**
 * TurboCat Feature Interaction Detection and Generation
 *
 * Automatically detects informative feature interactions and generates
 * combination features to improve model quality.
 */

#include "types.hpp"
#include "config.hpp"
#include <vector>
#include <utility>
#include <unordered_map>
#include <cmath>

namespace turbocat {

// Forward declaration
class Dataset;

/**
 * Represents a detected feature interaction
 */
struct FeatureInteraction {
    FeatureIndex feature_a;
    FeatureIndex feature_b;
    Float interaction_score;      // Higher = more informative
    Float individual_gain_a;      // Gain from feature a alone
    Float individual_gain_b;      // Gain from feature b alone
    Float combined_gain;          // Gain from a|b (a conditioned on b)
    bool is_categorical_a;
    bool is_categorical_b;

    // Interaction strength = combined_gain - max(individual_gain_a, individual_gain_b)
    Float interaction_strength() const {
        return combined_gain - std::max(individual_gain_a, individual_gain_b);
    }
};

/**
 * Detects informative feature interactions
 */
class InteractionDetector {
public:
    InteractionDetector() = default;

    /**
     * Detect feature interactions using the configured method
     * @param data Dataset with binned features and targets
     * @param config Interaction configuration
     * @return Vector of detected interactions, sorted by score (descending)
     */
    std::vector<FeatureInteraction> detect(
        const Dataset& data,
        const InteractionConfig& config
    );

    /**
     * Detect interactions using split-based method (fast)
     * Measures how often feature pairs appear in consecutive tree splits
     */
    std::vector<FeatureInteraction> detect_split_based(
        const Dataset& data,
        const InteractionConfig& config
    );

    /**
     * Detect interactions using mutual information (accurate)
     * Measures I(Y; X1, X2) - max(I(Y; X1), I(Y; X2))
     */
    std::vector<FeatureInteraction> detect_mutual_info(
        const Dataset& data,
        const InteractionConfig& config
    );

    /**
     * Detect interactions using correlation (simple)
     * Measures correlation of X1*X2 with Y
     */
    std::vector<FeatureInteraction> detect_correlation(
        const Dataset& data,
        const InteractionConfig& config
    );

private:
    // Helper: compute gain for a single feature
    Float compute_feature_gain(
        const Dataset& data,
        FeatureIndex feature
    );

    // Helper: compute conditional gain (feature_a | feature_b)
    Float compute_conditional_gain(
        const Dataset& data,
        FeatureIndex feature_a,
        FeatureIndex feature_b
    );

    // Helper: compute mutual information I(Y; X)
    Float compute_mutual_info(
        const Dataset& data,
        FeatureIndex feature
    );

    // Helper: compute joint mutual information I(Y; X1, X2)
    Float compute_joint_mutual_info(
        const Dataset& data,
        FeatureIndex feature_a,
        FeatureIndex feature_b
    );
};

/**
 * Generates combination features from detected interactions
 */
class InteractionGenerator {
public:
    InteractionGenerator() = default;

    /**
     * Generate interaction features and add them to the dataset
     * @param data Dataset to modify (adds new features)
     * @param interactions Detected interactions
     * @param config Configuration for combination types
     * @return Number of features added
     */
    FeatureIndex generate(
        Dataset& data,
        const std::vector<FeatureInteraction>& interactions,
        const InteractionConfig& config
    );

    /**
     * Apply same transformations to new data (for prediction)
     * @param data New dataset to transform
     * @param reference Reference dataset with interaction metadata
     */
    void apply(Dataset& data, const Dataset& reference);

    /**
     * Get mapping of generated feature index to source features
     */
    const std::vector<std::pair<FeatureIndex, FeatureIndex>>& get_interaction_map() const {
        return interaction_map_;
    }

private:
    // Mapping: generated_feature_idx -> (source_a, source_b)
    std::vector<std::pair<FeatureIndex, FeatureIndex>> interaction_map_;

    // Combination types used for each generated feature
    std::vector<InteractionConfig::CombinationType> combination_types_;

    // Helper: create product feature
    void create_product_feature(
        Dataset& data,
        FeatureIndex src_a,
        FeatureIndex src_b
    );

    // Helper: create ratio feature
    void create_ratio_feature(
        Dataset& data,
        FeatureIndex src_a,
        FeatureIndex src_b
    );

    // Helper: create categorical combination (hash)
    void create_concat_feature(
        Dataset& data,
        FeatureIndex src_a,
        FeatureIndex src_b
    );
};

} // namespace turbocat
