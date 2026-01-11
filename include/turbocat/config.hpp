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

// Tree growing policy
enum class GrowPolicy : uint8_t {
    Depthwise = 0,  // Level-wise: grow all nodes at each depth (XGBoost/CatBoost style)
    Lossguide = 1,  // Leaf-wise: grow leaf with highest gain (LightGBM style)
};

struct TreeConfig {
    // Structure
    uint16_t max_depth = 6;                    // Maximum tree depth
    uint32_t max_leaves = 64;                  // Maximum leaves (2^max_depth)
    uint32_t min_samples_leaf = 1;             // Minimum samples per leaf
    Float min_child_weight = 1.0f;             // Minimum sum of hessians (XGBoost default=1.0)
    GrowPolicy grow_policy = GrowPolicy::Depthwise;  // Tree growing strategy
    
    // Split finding
    SplitCriterion criterion = SplitCriterion::Variance;
    Float min_split_gain = 0.0f;               // Minimum gain threshold (0 = disabled)
    Float gamma = 0.0f;                        // Complexity penalty subtracted from gain (XGBoost-style)
    Float tsallis_q = 1.5f;                    // Tsallis entropy parameter
    
    // Histogram
    uint16_t max_bins = 255;                   // Number of histogram bins
    bool use_quantized_grad = false;           // Use 3-bit gradient quantization (was true - disabled for now)
    
    // Regularization
    Float lambda_l2 = 1.0f;                    // L2 regularization on leaf values (XGBoost default=1.0)
    Float lambda_l1 = 0.0f;                    // L1 regularization (sparsity)
    Float max_delta_step = 0.0f;               // Maximum delta step (0 = unlimited)
    Float leaf_smooth = 0.0f;                  // Leaf value smoothing weight (0 = disabled)
                                               // shrink = count / (count + smooth), reduces overfitting
    
    // Missing values
    bool learn_missing_direction = true;       // Learn optimal direction for NaN
    
    // GradTree specific (gradient-based optimization)
    bool use_gradtree = false;                 // Enable differentiable tree optimization
    uint16_t gradtree_iterations = 100;        // Optimization iterations per tree
    Float gradtree_lr = 0.1f;                  // Learning rate for tree parameters
    Float gradtree_momentum = 0.9f;            // Momentum for optimization

    // Tree type
    bool use_symmetric = false;                // Use oblivious/symmetric trees (CatBoost-style)
                                               // NOTE: Currently regular trees perform better
};

// ============================================================================
// Boosting Configuration
// ============================================================================

struct BoostingConfig {
    // Ensemble
    uint32_t n_estimators = 1000;              // Number of trees
    Float learning_rate = 0.1f;                // Shrinkage factor (XGBoost default=0.3, LightGBM=0.1)
    Float subsample = 0.8f;                    // Row subsampling ratio
    Float colsample_bytree = 0.8f;             // Column subsampling per tree
    Float colsample_bylevel = 1.0f;            // Column subsampling per level
    
    // Early stopping
    uint32_t early_stopping_rounds = 50;       // Stop if no improvement
    Float early_stopping_tolerance = 1e-4f;    // Relative improvement threshold (0.01%)
    std::string early_stopping_metric = "loss"; // "loss", "roc_auc", "pr_auc", "f1"
    
    // GOSS (Gradient-based One-Side Sampling)
    bool use_goss = false;                     // Disabled by default (can hurt some datasets)
    Float goss_top_rate = 0.2f;                // Keep top 20% by gradient
    Float goss_other_rate = 0.1f;              // Sample 10% from rest

    // MVS (Minimum Variance Sampling) - better for large datasets
    bool use_mvs = false;                      // Use MVS instead of GOSS
    Float mvs_subsample = 0.5f;                // MVS sample ratio (more aggressive than GOSS)
    
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
// Feature Interaction Configuration
// ============================================================================

struct InteractionConfig {
    // Detection settings
    bool auto_detect = false;                  // Auto-detect feature interactions
    uint16_t max_interactions = 20;            // Maximum number of interactions to detect
    Float min_interaction_gain = 0.01f;        // Minimum gain threshold for interaction

    // Detection method
    enum class DetectionMethod : uint8_t {
        SplitBased = 0,       // Based on consecutive tree splits (fast)
        MutualInfo = 1,       // Mutual information (accurate but slower)
        Correlation = 2,      // Correlation-based (simple)
    };
    DetectionMethod method = DetectionMethod::SplitBased;

    // Interaction types
    bool numerical_numerical = true;           // Detect num×num interactions
    bool numerical_categorical = true;         // Detect num×cat interactions
    bool categorical_categorical = true;       // Detect cat×cat interactions

    // Generated feature settings
    enum class CombinationType : uint8_t {
        Product = 0,          // a * b (numerical)
        Ratio = 1,            // a / (b + eps) (numerical)
        Sum = 2,              // a + b (numerical)
        Difference = 3,       // a - b (numerical)
        Concat = 4,           // hash(a, b) (categorical)
    };
    std::vector<CombinationType> combination_types = {CombinationType::Product};

    // Manual interactions (always included)
    std::vector<std::pair<FeatureIndex, FeatureIndex>> manual_interactions;
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
    uint32_t n_classes = 2;  // Number of classes (2 for binary, >2 for multiclass)

    TreeConfig tree;
    BoostingConfig boosting;
    LossConfig loss;
    CategoricalConfig categorical;
    InteractionConfig interactions;
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
    
    static Config multiclass_classification(uint32_t num_classes) {
        Config cfg;
        cfg.task = TaskType::MulticlassClassification;
        cfg.n_classes = num_classes;
        cfg.loss.loss_type = LossType::CrossEntropy;
        return cfg;
    }
    
    static Config regression() {
        Config cfg;
        cfg.task = TaskType::Regression;
        cfg.loss.loss_type = LossType::MSE;
        cfg.tree.criterion = SplitCriterion::Variance;
        cfg.tree.lambda_l2 = 0.0f;  // Lower regularization for regression (MSE has hessian=2)
        cfg.tree.grow_policy = GrowPolicy::Lossguide;  // Leaf-wise growth for better regression
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

    /**
     * Adaptive configuration for large datasets (50K+ samples)
     * Optimizes for both speed and quality on large data
     */
    static Config large_dataset() {
        Config cfg;
        cfg.tree.max_depth = 8;                        // Deeper trees for more data
        cfg.tree.max_leaves = 128;
        cfg.tree.min_samples_leaf = 20;                // Prevent overfitting
        cfg.tree.min_child_weight = 20.0f;             // Higher weight threshold
        cfg.tree.grow_policy = GrowPolicy::Lossguide;  // Leaf-wise for better accuracy
        cfg.tree.lambda_l2 = 5.0f;                     // More regularization
        cfg.tree.leaf_smooth = 10.0f;                  // Smooth leaf values
        cfg.boosting.n_estimators = 2000;
        cfg.boosting.learning_rate = 0.05f;            // Lower LR for stability
        cfg.boosting.subsample = 0.7f;
        cfg.boosting.colsample_bytree = 0.8f;
        cfg.boosting.use_mvs = true;                   // MVS for large data
        cfg.boosting.mvs_subsample = 0.5f;
        cfg.boosting.early_stopping_rounds = 100;
        return cfg;
    }

    /**
     * Adaptive configuration for large regression datasets
     * Specifically tuned for regression quality
     */
    static Config large_regression() {
        Config cfg = large_dataset();
        cfg.task = TaskType::Regression;
        cfg.loss.loss_type = LossType::Huber;          // Robust to outliers
        cfg.loss.huber_delta = 1.35f;                  // Good default
        cfg.tree.criterion = SplitCriterion::Variance;
        cfg.tree.lambda_l2 = 10.0f;                    // Strong regularization
        cfg.tree.max_depth = 10;                       // Deeper for regression
        cfg.tree.leaf_smooth = 20.0f;                  // More smoothing
        cfg.boosting.learning_rate = 0.03f;            // Lower LR
        cfg.boosting.n_estimators = 3000;
        return cfg;
    }

    /**
     * Automatically adapt config based on dataset characteristics
     */
    void adapt_to_data(Index n_samples, FeatureIndex n_features) {
        // For smaller datasets: enable ordered boosting (like CatBoost)
        // Prevents prediction shift and improves generalization
        if (n_samples < 50000) {
            boosting.use_ordered_boosting = true;
            boosting.n_permutations = 4;
        }

        // For large datasets: deeper trees, more regularization, use MVS
        if (n_samples >= 50000) {
            tree.max_depth = std::max(tree.max_depth, static_cast<uint16_t>(8));
            tree.max_leaves = std::max(tree.max_leaves, 128u);
            tree.min_samples_leaf = std::max(tree.min_samples_leaf, 20u);
            tree.min_child_weight = std::max(tree.min_child_weight, 10.0f);
            tree.lambda_l2 = std::max(tree.lambda_l2, 5.0f);
            tree.leaf_smooth = std::max(tree.leaf_smooth, 10.0f);
            boosting.use_mvs = true;
            boosting.mvs_subsample = 0.5f;
            boosting.early_stopping_rounds = std::max(boosting.early_stopping_rounds, 100u);
        }

        // For very large datasets: even more aggressive sampling
        if (n_samples >= 200000) {
            tree.max_depth = std::max(tree.max_depth, static_cast<uint16_t>(10));
            tree.max_leaves = std::max(tree.max_leaves, 256u);
            tree.min_samples_leaf = std::max(tree.min_samples_leaf, 50u);
            tree.min_child_weight = std::max(tree.min_child_weight, 30.0f);
            tree.lambda_l2 = std::max(tree.lambda_l2, 10.0f);
            boosting.mvs_subsample = 0.3f;
            boosting.learning_rate = std::min(boosting.learning_rate, 0.03f);
        }

        // For high-dimensional data: more regularization, column sampling
        if (n_features >= 100) {
            boosting.colsample_bytree = std::min(boosting.colsample_bytree, 0.7f);
            tree.lambda_l2 = std::max(tree.lambda_l2, 3.0f);
        }

        // For wide data (many features, fewer samples): prevent overfitting
        if (n_features > n_samples / 10) {
            boosting.colsample_bytree = std::min(boosting.colsample_bytree, 0.5f);
            tree.max_depth = std::min(tree.max_depth, static_cast<uint16_t>(6));
        }
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
        if (n_classes < 2) {
            throw std::invalid_argument("n_classes must be at least 2");
        }
        if (task == TaskType::MulticlassClassification && n_classes < 3) {
            throw std::invalid_argument("MulticlassClassification requires n_classes >= 3");
        }
    }
};

} // namespace turbocat
