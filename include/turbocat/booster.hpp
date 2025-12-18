#pragma once

/**
 * TurboCat Booster
 * 
 * Main gradient boosting interface combining all components:
 * - Histogram-based tree building
 * - GradTree optimization (optional)
 * - Advanced loss functions
 * - GOSS sampling
 * - Early stopping
 * - Feature importance
 * - Uncertainty quantification
 */

#include "types.hpp"
#include "config.hpp"
#include "dataset.hpp"
#include "histogram.hpp"
#include "tree.hpp"
#include "symmetric_tree.hpp"
#include "loss.hpp"
#include <memory>
#include <vector>
#include <functional>
#include <chrono>

namespace turbocat {

// ============================================================================
// Training Callback
// ============================================================================

struct TrainingInfo {
    uint32_t iteration;
    Float train_loss;
    Float valid_loss;
    Float best_valid_loss;
    uint32_t best_iteration;
    double elapsed_seconds;
    size_t n_trees;
};

using TrainingCallback = std::function<bool(const TrainingInfo&)>;

// ============================================================================
// Feature Importance
// ============================================================================

struct FeatureImportance {
    std::vector<Float> gain;         // Total gain from splits
    std::vector<Float> split_count;  // Number of times used for splits
    std::vector<Float> cover;        // Total samples affected
    
    // Normalized versions (sum to 1)
    std::vector<Float> gain_normalized;
    std::vector<Float> split_normalized;
    std::vector<Float> cover_normalized;
};

// ============================================================================
// Cross-Validation Result
// ============================================================================

struct CVResult {
    std::vector<Float> train_scores;
    std::vector<Float> valid_scores;
    Float mean_valid_score;
    Float std_valid_score;
    uint32_t best_n_estimators;
};

// ============================================================================
// TurboCat Booster
// ============================================================================

class Booster {
public:
    Booster();
    explicit Booster(const Config& config);
    
    // ========================================================================
    // Training
    // ========================================================================
    
    /**
     * Train the model
     * @param train_data Training dataset
     * @param valid_data Validation dataset (optional, for early stopping)
     * @param callback Optional callback for progress monitoring
     */
    void train(
        Dataset& train_data,
        Dataset* valid_data = nullptr,
        TrainingCallback callback = nullptr
    );
    
    /**
     * Continue training (add more trees)
     */
    void continue_training(
        Dataset& train_data,
        uint32_t n_additional_trees,
        Dataset* valid_data = nullptr
    );
    
    /**
     * Cross-validation
     */
    CVResult cross_validate(
        Dataset& data,
        uint32_t n_folds = 5,
        uint64_t seed = 42
    );
    
    // ========================================================================
    // Prediction
    // ========================================================================
    
    /**
     * Predict raw scores (binary/regression)
     */
    void predict_raw(
        const Dataset& data,
        Float* output,
        int n_trees = -1  // -1 = all trees
    ) const;

    /**
     * Predict raw scores (multiclass): output is n_samples * n_classes
     */
    void predict_raw_multiclass(
        const Dataset& data,
        Float* output,
        int n_trees = -1
    ) const;

    /**
     * Predict probabilities (binary classification)
     */
    void predict_proba(
        const Dataset& data,
        Float* output,
        int n_trees = -1
    ) const;

    /**
     * Predict probabilities (multiclass): output is n_samples * n_classes
     */
    void predict_proba_multiclass(
        const Dataset& data,
        Float* output,
        int n_trees = -1
    ) const;
    
    /**
     * Predict with uncertainty (if enabled)
     */
    std::vector<Prediction> predict_with_uncertainty(
        const Dataset& data,
        int n_trees = -1
    ) const;
    
    /**
     * Predict from raw features (no binning required)
     */
    Float predict_single(const Float* features, FeatureIndex n_features) const;
    
    // ========================================================================
    // Model Information
    // ========================================================================
    
    size_t n_trees() const {
        return config_.tree.use_symmetric ? symmetric_ensemble_.n_trees() : ensemble_.n_trees();
    }
    const Config& config() const { return config_; }
    
    /**
     * Get feature importance
     */
    FeatureImportance feature_importance() const;
    
    /**
     * Get training history
     */
    struct TrainingHistory {
        std::vector<Float> train_loss;
        std::vector<Float> valid_loss;
        std::vector<double> iteration_time;
    };
    
    const TrainingHistory& training_history() const { return history_; }
    
    // ========================================================================
    // Serialization
    // ========================================================================
    
    void save(const std::string& path) const;
    static Booster load(const std::string& path);
    
    void save_binary(std::ostream& out) const;
    static Booster load_binary(std::istream& in);
    
    // Export to other formats
    std::string to_json() const;
    std::string to_pmml() const;  // Predictive Model Markup Language
    
    // ========================================================================
    // Model Analysis
    // ========================================================================
    
    /**
     * SHAP values (TreeSHAP algorithm)
     */
    void compute_shap_values(
        const Dataset& data,
        Float* shap_values  // Output: n_samples × n_features
    ) const;
    
    /**
     * Partial dependence
     */
    std::vector<std::pair<Float, Float>> partial_dependence(
        const Dataset& data,
        FeatureIndex feature,
        uint32_t n_points = 100
    ) const;
    
private:
    Config config_;
    TreeEnsemble ensemble_;
    SymmetricEnsemble symmetric_ensemble_;  // For use_symmetric=true
    std::unique_ptr<Loss> loss_;
    std::unique_ptr<HistogramBuilder> hist_builder_;
    
    // Training state
    Float base_prediction_ = 0.0f;
    std::vector<Float> base_predictions_multiclass_;  // For multiclass: K base predictions
    TrainingHistory history_;
    uint32_t best_iteration_ = 0;
    Float best_valid_loss_ = 1e30f;
    
    // Dataset metadata (for prediction without binning)
    std::vector<FeatureInfo> feature_info_;
    std::vector<std::vector<Float>> bin_edges_;
    
    // Random state
    uint64_t rng_state_;
    
    // ========================================================================
    // Internal Methods
    // ========================================================================
    
    /**
     * Build single tree and add to ensemble (binary/regression)
     */
    void build_tree(
        Dataset& data,
        const std::vector<Index>& sample_indices,
        const std::vector<FeatureIndex>& feature_indices,
        AlignedVector<Float>& predictions
    );

    /**
     * Build single tree for multiclass and add to ensemble
     */
    void build_tree_multiclass(
        Dataset& data,
        const std::vector<Index>& sample_indices,
        const std::vector<FeatureIndex>& feature_indices,
        std::vector<Float>& predictions,  // n_samples * n_classes
        const std::vector<Float>& gradients,  // n_samples * n_classes
        const std::vector<Float>& hessians   // n_samples * n_classes
    );
    
    /**
     * Build GradTree and add to ensemble
     */
    void build_gradtree(
        Dataset& data,
        const std::vector<Index>& sample_indices,
        AlignedVector<Float>& predictions
    );
    
    /**
     * Update gradients and hessians
     */
    void update_gradients(Dataset& data, const AlignedVector<Float>& predictions);
    
    /**
     * Compute loss on dataset
     */
    Float compute_loss(const Dataset& data, const AlignedVector<Float>& predictions) const;
    
    /**
     * GOSS sampling
     */
    std::vector<Index> goss_sample(
        const Dataset& data,
        Float top_rate,
        Float other_rate
    );
    
    /**
     * Random number generation
     */
    uint64_t next_random();
    Float random_float();
    
    /**
     * Initialize loss and base prediction
     */
    void initialize_training(Dataset& data);

    /**
     * Multiclass training loop
     */
    void train_multiclass(
        Dataset& train_data,
        Dataset* valid_data,
        TrainingCallback callback
    );
};

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Quick train function with sensible defaults
 */
inline Booster quick_train(
    const Float* X, Index n_samples, FeatureIndex n_features,
    const Float* y,
    TaskType task = TaskType::BinaryClassification
) {
    Config config;
    config.task = task;
    
    Dataset data;
    data.from_dense(X, n_samples, n_features, y);
    data.compute_bins(config);
    
    Booster booster(config);
    booster.train(data);
    
    return booster;
}

/**
 * Grid search for hyperparameter tuning
 */
struct GridSearchResult {
    Config best_config;
    Float best_score;
    std::vector<std::pair<Config, Float>> all_results;
};

GridSearchResult grid_search(
    Dataset& data,
    const std::vector<Config>& configs,
    uint32_t n_folds = 5,
    uint64_t seed = 42
);

} // namespace turbocat
