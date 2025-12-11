#pragma once

/**
 * TurboCat Configuration
 * 
 * Comprehensive hyperparameter settings designed for superior out-of-box performance.
 * Default values are carefully tuned based on TabZilla/OpenML benchmarks.
 */

#include "types.hpp"
#include <string>
#include <optional>

namespace turbocat {

// ============================================================================
// Tree Configuration
// ============================================================================

struct TreeConfig {
    // Structure
    uint16_t max_depth = 6;                    // Maximum tree depth
    uint32_t max_leaves = 64;                  // Maximum leaves (2^max_depth)
    uint32_t min_samples_leaf = 1;             // Minimum samples per leaf (was 20)
    Float min_child_weight = 0.1f;             // Minimum sum of hessians (was 1.0)
    
    // Split finding
    SplitCriterion criterion = SplitCriterion::Variance;
    Float min_split_gain = 1e-7f;              // Minimum gain to make a split (was 1e-5)
    Float tsallis_q = 1.5f;                    // Tsallis entropy parameter
    
    // Histogram
    uint16_t max_bins = 255;                   // Number of histogram bins
    bool use_quantized_grad = false;           // Use 3-bit gradient quantization (was true - disabled for now)
    
    // Regularization
    Float lambda_l2 = 0.1f;                    // L2 regularization on leaf values (was 1.0)
    Float lambda_l1 = 0.0f;                    // L1 regularization (sparsity)
    Float max_delta_step = 0.0f;               // Maximum delta step (0 = unlimited)
    
    // Missing values
    bool learn_missing_direction = true;       // Learn optimal direction for NaN
    
    // GradTree specific (gradient-based optimization)
    bool use_gradtree = false;                 // Enable differentiable tree optimization
    uint16_t gradtree_iterations = 100;        // Optimization iterations per tree
    Float gradtree_lr = 0.1f;                  // Learning rate for tree parameters
    Float gradtree_momentum = 0.9f;            // Momentum for optimization
};

// ============================================================================
// Boosting Configuration
// ============================================================================

struct BoostingConfig {
    // Ensemble
    uint32_t n_estimators = 1000;              // Number of trees
    Float learning_rate = 0.05f;               // Shrinkage factor
    Float subsample = 0.8f;                    // Row subsampling ratio
    Float colsample_bytree = 0.8f;             // Column subsampling per tree
    Float colsample_bylevel = 1.0f;            // Column subsampling per level
    
    // Early stopping
    uint32_t early_stopping_rounds = 50;       // Stop if no improvement
    Float early_stopping_tolerance = 1e-5f;    // Minimum improvement
    
    // GOSS (Gradient-based One-Side Sampling)
    bool use_goss = true;                      // Enable GOSS
    Float goss_top_rate = 0.2f;                // Keep top 20% by gradient
    Float goss_other_rate = 0.1f;              // Sample 10% from rest
    
    // Dart (Dropouts meet Multiple Additive Regression Trees)
    bool use_dart = false;                     // Enable DART
    Float dart_drop_rate = 0.1f;               // Dropout rate
    
    // Ordered boosting (like CatBoost, but optional)
    bool use_ordered_boosting = false;         // Disable by default (overhead for large data)
    uint8_t n_permutations = 4;                // Number of permutations if enabled
};

// ============================================================================
// Loss Configuration
// ============================================================================

struct LossConfig {
    LossType loss_type = LossType::LogLoss;
    
    // Robust Focal Loss parameters
    Float focal_gamma = 2.0f;                  // Focusing parameter
    Float focal_alpha = 0.25f;                 // Class balance parameter
    Float robust_mu = 0.1f;                    // Robustness to label noise
    
    // LDAM (Label-Distribution-Aware Margin)
    bool auto_ldam_margins = true;             // Auto-compute margins from class distribution
    std::vector<Float> ldam_margins;           // Manual margins if not auto
    
    // Logit-adjusted loss
    bool auto_class_priors = true;             // Auto-compute from data
    std::vector<Float> class_priors;           // Manual priors if not auto
    Float logit_adjustment_tau = 1.0f;         // Temperature for adjustment
    
    // Huber loss
    Float huber_delta = 1.0f;                  // Transition point
};

// ============================================================================
// Categorical Feature Configuration
// ============================================================================

struct CategoricalConfig {
    // Encoding method
    enum class EncodingMethod : uint8_t {
        TargetStatistics = 0,   // Ordered target encoding (like CatBoost)
        CrossValidatedTS = 1,   // CV target statistics (our improvement)
        OneHot = 2,             // One-hot encoding
        Embedding = 3,          // Learned embeddings (for text-like)
        Hybrid = 4,             // Auto-select based on cardinality
    };
    
    EncodingMethod method = EncodingMethod::CrossValidatedTS;
    
    // Target statistics parameters
    Float ts_prior_weight = 1.0f;              // Smoothing prior weight
    uint8_t ts_cv_folds = 5;                   // CV folds for CrossValidatedTS
    
    // Cardinality thresholds for hybrid
    uint32_t one_hot_max_cardinality = 10;     // Use one-hot below this
    uint32_t embedding_min_cardinality = 100;  // Use embeddings above this
    
    // Feature interaction combinations
    bool auto_feature_combinations = true;     // Auto-generate cat×cat features
    uint8_t max_combination_depth = 2;         // Maximum combination depth
};

// ============================================================================
// Device Configuration
// ============================================================================

struct DeviceConfig {
    enum class DeviceType : uint8_t {
        Auto = 0,
        CPU = 1,
        CUDA = 2,
        Metal = 3,
    };
    
    DeviceType device = DeviceType::Auto;
    int32_t device_id = 0;                     // GPU device ID
    
    // CPU specific
    int32_t n_threads = -1;                    // Number of threads (-1 = auto)
    bool use_simd = true;                      // Enable SIMD optimizations
    
    // GPU specific
    uint32_t gpu_batch_size = 65536;           // Batch size for GPU operations
    bool gpu_use_fp16 = false;                 // Use half precision on GPU
};

// ============================================================================
// Main Configuration
// ============================================================================

struct Config {
    TaskType task = TaskType::BinaryClassification;
    
    TreeConfig tree;
    BoostingConfig boosting;
    LossConfig loss;
    CategoricalConfig categorical;
    DeviceConfig device;
    
    // Verbosity and logging
    int32_t verbosity = 1;                     // 0=silent, 1=progress, 2=debug
    uint32_t log_period = 100;                 // Log every N iterations
    
    // Random state
    uint64_t seed = 42;
    
    // Validation
    Float validation_fraction = 0.1f;          // Auto-split for early stopping
    
    // Feature importance
    bool compute_feature_importance = true;
    
    // Uncertainty quantification (GP-derived)
    bool compute_uncertainty = false;          // Enable variance estimation
    uint32_t uncertainty_samples = 100;        // MC samples for uncertainty
    
    // ========================================================================
    // Factory Methods for Common Tasks
    // ========================================================================
    
    static Config binary_classification() {
        Config cfg;
        cfg.task = TaskType::BinaryClassification;
        cfg.loss.loss_type = LossType::LogLoss;
        return cfg;
    }
    
    static Config multiclass_classification(uint32_t n_classes) {
        Config cfg;
        cfg.task = TaskType::MulticlassClassification;
        cfg.loss.loss_type = LossType::CrossEntropy;
        return cfg;
    }
    
    static Config regression() {
        Config cfg;
        cfg.task = TaskType::Regression;
        cfg.loss.loss_type = LossType::MSE;
        cfg.tree.criterion = SplitCriterion::Variance;
        return cfg;
    }
    
    static Config robust_classification() {
        // For noisy labels or class imbalance
        Config cfg;
        cfg.task = TaskType::BinaryClassification;
        cfg.loss.loss_type = LossType::RobustFocal;
        cfg.categorical.method = CategoricalConfig::EncodingMethod::CrossValidatedTS;
        return cfg;
    }
    
    static Config fast_training() {
        Config cfg;
        cfg.boosting.use_goss = true;
        cfg.tree.use_quantized_grad = true;
        cfg.boosting.subsample = 0.5f;
        cfg.boosting.early_stopping_rounds = 20;
        return cfg;
    }
    
    static Config maximum_accuracy() {
        Config cfg;
        cfg.tree.use_gradtree = true;
        cfg.boosting.n_estimators = 3000;
        cfg.boosting.learning_rate = 0.02f;
        cfg.boosting.use_goss = false;
        cfg.tree.use_quantized_grad = false;
        cfg.categorical.method = CategoricalConfig::EncodingMethod::CrossValidatedTS;
        cfg.loss.loss_type = LossType::RobustFocal;
        return cfg;
    }
    
    // ========================================================================
    // Validation
    // ========================================================================
    
    void validate() const {
        if (tree.max_depth > 32) {
            throw std::invalid_argument("max_depth cannot exceed 32");
        }
        if (tree.max_bins > 255) {
            throw std::invalid_argument("max_bins cannot exceed 255 (uint8 limit)");
        }
        if (boosting.learning_rate <= 0 || boosting.learning_rate > 1) {
            throw std::invalid_argument("learning_rate must be in (0, 1]");
        }
        if (boosting.subsample <= 0 || boosting.subsample > 1) {
            throw std::invalid_argument("subsample must be in (0, 1]");
        }
    }
};

} // namespace turbocat
