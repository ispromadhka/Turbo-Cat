/**
 * TurboCat Python Bindings
 * 
 * Provides sklearn-compatible API for easy integration.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <set>
#include <chrono>

#include "turbocat/turbocat.hpp"
#include "turbocat/metrics.hpp"
#include "turbocat/interactions.hpp"

namespace py = pybind11;
using namespace turbocat;

// ============================================================================
// NumPy Conversion Utilities
// ============================================================================

template<typename T>
py::array_t<T> vector_to_numpy(const std::vector<T>& vec) {
    auto result = py::array_t<T>(vec.size());
    auto buf = result.request();
    std::memcpy(buf.ptr, vec.data(), vec.size() * sizeof(T));
    return result;
}

// ============================================================================
// Python Classifier Class
// ============================================================================

class TurboCatClassifier {
public:
    TurboCatClassifier(
        int n_estimators = 500,          // Optimized: fewer trees, higher LR
        float learning_rate = 0.2f,      // Optimized: faster convergence
        int max_depth = 6,
        int max_bins = 255,
        float subsample = 1.0f,      // Default: no subsampling (faster training)
        float colsample_bytree = 1.0f,  // Default: use all features
        float min_child_weight = 1.0f,
        float lambda_l2 = 1.0f,
        const std::string& loss = "auto",
        bool use_goss = false,
        float goss_top_rate = 0.2f,
        float goss_other_rate = 0.1f,
        bool use_gradtree = false,
        bool use_symmetric = false,  // Oblivious trees (experimental)
        bool use_ordered_boosting = false,  // Ordered boosting (like CatBoost)
        const std::string& grow_policy = "depthwise",  // "depthwise" or "lossguide"
        const std::string& mode = "auto",  // "small", "large", or "auto"
        int early_stopping_rounds = 50,
        const std::string& early_stopping_metric = "loss",  // "loss", "roc_auc", "pr_auc", "f1"
        const std::string& cat_encoding = "ordered",  // "ordered", "cv", "onehot"
        bool auto_interactions = false,  // Auto-detect feature interactions
        int max_interactions = 10,       // Maximum interactions to detect
        const std::string& interaction_method = "split",  // "split", "mutual_info", "correlation"
        int n_threads = -1,
        int seed = 42,
        int verbosity = 1
    ) : loss_type_str_(loss), mode_(mode) {
        // Will configure properly in fit() when we know n_classes
        config_.boosting.n_estimators = n_estimators;
        config_.boosting.learning_rate = learning_rate;
        config_.tree.max_depth = max_depth;
        config_.tree.max_bins = max_bins;
        config_.boosting.subsample = subsample;
        config_.boosting.colsample_bytree = colsample_bytree;
        config_.tree.min_child_weight = min_child_weight;
        config_.tree.lambda_l2 = lambda_l2;
        config_.boosting.use_goss = use_goss;
        config_.boosting.goss_top_rate = goss_top_rate;
        config_.boosting.goss_other_rate = goss_other_rate;
        config_.tree.use_gradtree = use_gradtree;
        config_.tree.use_symmetric = use_symmetric;
        config_.boosting.use_ordered_boosting = use_ordered_boosting;
        config_.boosting.early_stopping_rounds = early_stopping_rounds;
        config_.boosting.early_stopping_metric = early_stopping_metric;
        config_.device.n_threads = n_threads;
        config_.seed = seed;
        config_.verbosity = verbosity;

        // Parse grow_policy
        if (grow_policy == "lossguide" || grow_policy == "leaf" || grow_policy == "leafwise") {
            config_.tree.grow_policy = GrowPolicy::Lossguide;
        } else {
            config_.tree.grow_policy = GrowPolicy::Depthwise;
        }

        // Parse mode - controls tree architecture based on data size
        // "small" = regular trees (best quality, fast training)
        // "large" = symmetric trees with bit-ops (fastest inference on large data)
        // "auto" = automatically select based on dataset size in fit()
        if (mode == "large") {
            config_.tree.use_symmetric = true;  // Use symmetric trees for large data
        } else if (mode == "small") {
            config_.tree.use_symmetric = false;  // Use regular trees for small data
        }
        // "auto" mode will be handled in fit() based on data size

        // Parse categorical encoding method
        if (cat_encoding == "ordered" || cat_encoding == "target") {
            config_.categorical.method = CategoricalConfig::EncodingMethod::TargetStatistics;
        } else if (cat_encoding == "cv" || cat_encoding == "cross_validated") {
            config_.categorical.method = CategoricalConfig::EncodingMethod::CrossValidatedTS;
        } else if (cat_encoding == "onehot" || cat_encoding == "one_hot") {
            config_.categorical.method = CategoricalConfig::EncodingMethod::OneHot;
        } else {
            // Default: ordered target statistics (like CatBoost)
            config_.categorical.method = CategoricalConfig::EncodingMethod::TargetStatistics;
        }

        // Parse interaction detection settings
        config_.interactions.auto_detect = auto_interactions;
        config_.interactions.max_interactions = max_interactions;

        if (interaction_method == "split" || interaction_method == "split_based") {
            config_.interactions.method = InteractionConfig::DetectionMethod::SplitBased;
        } else if (interaction_method == "mutual_info" || interaction_method == "mi") {
            config_.interactions.method = InteractionConfig::DetectionMethod::MutualInfo;
        } else if (interaction_method == "correlation" || interaction_method == "corr") {
            config_.interactions.method = InteractionConfig::DetectionMethod::Correlation;
        }
    }

    void fit(
        py::array_t<float> X,
        py::array_t<float> y,
        py::array_t<float> X_val = py::array_t<float>(),
        py::array_t<float> y_val = py::array_t<float>(),
        std::vector<int> cat_features = {}
    ) {
        auto X_buf = X.request();
        auto y_buf = y.request();

        if (X_buf.ndim != 2) {
            throw std::runtime_error("X must be 2-dimensional");
        }
        if (y_buf.ndim != 1) {
            throw std::runtime_error("y must be 1-dimensional");
        }

        Index n_samples = static_cast<Index>(X_buf.shape[0]);
        FeatureIndex n_features = static_cast<FeatureIndex>(X_buf.shape[1]);

        // Handle "auto" mode - choose tree type based on dataset size
        // "small" data: regular trees (best quality)
        // "large" data: symmetric trees (fastest batch inference with no-binning path)
        if (mode_ == "auto") {
            // Use symmetric trees by default for fast batch inference
            // Symmetric trees support float_threshold for direct comparisons
            // This avoids the binning overhead in predict (4-5x speedup)
            config_.tree.use_symmetric = true;
        }

        // Detect number of classes from labels
        float* y_ptr = static_cast<float*>(y_buf.ptr);
        float max_label = 0.0f;
        for (Index i = 0; i < n_samples; ++i) {
            max_label = std::max(max_label, y_ptr[i]);
        }
        n_classes_ = static_cast<uint32_t>(max_label) + 1;

        // Configure based on n_classes
        if (n_classes_ <= 2) {
            config_.task = TaskType::BinaryClassification;
            config_.n_classes = 2;

            // Parse loss type for binary
            if (loss_type_str_ == "auto" || loss_type_str_ == "logloss" || loss_type_str_ == "binary_crossentropy") {
                config_.loss.loss_type = LossType::LogLoss;
            } else if (loss_type_str_ == "focal" || loss_type_str_ == "robust_focal") {
                config_.loss.loss_type = LossType::RobustFocal;
            } else if (loss_type_str_ == "ldam") {
                config_.loss.loss_type = LossType::LDAM;
            } else if (loss_type_str_ == "logit_adjusted") {
                config_.loss.loss_type = LossType::LogitAdjusted;
            } else if (loss_type_str_ == "tsallis") {
                config_.loss.loss_type = LossType::Tsallis;
            } else if (loss_type_str_ == "asymmetric") {
                config_.loss.loss_type = LossType::Asymmetric;
            } else if (loss_type_str_ == "auc" || loss_type_str_ == "auc_loss") {
                config_.loss.loss_type = LossType::AUCLoss;
            } else if (loss_type_str_ == "class_balanced" || loss_type_str_ == "balanced") {
                config_.loss.loss_type = LossType::ClassBalanced;
            } else if (loss_type_str_ == "pr_auc" || loss_type_str_ == "prauc") {
                config_.loss.loss_type = LossType::PRAUCLoss;
            }
        } else {
            // Multiclass
            config_.task = TaskType::MulticlassClassification;
            config_.n_classes = n_classes_;
            config_.loss.loss_type = LossType::CrossEntropy;
        }

        // Convert cat_features to FeatureIndex
        std::vector<FeatureIndex> cat_features_typed;
        for (int f : cat_features) {
            cat_features_typed.push_back(static_cast<FeatureIndex>(f));
        }

        // Create dataset
        train_data_ = std::make_unique<Dataset>();
        train_data_->from_dense(
            static_cast<float*>(X_buf.ptr),
            n_samples,
            n_features,
            static_cast<float*>(y_buf.ptr),
            nullptr,
            cat_features_typed
        );

        // Compute categorical encodings BEFORE binning (CatBoost-style)
        if (!cat_features_typed.empty() &&
            config_.categorical.method != CategoricalConfig::EncodingMethod::OneHot) {
            train_data_->compute_target_statistics(config_);
        }

        train_data_->compute_bins(config_);

        // Detect and generate feature interactions
        if (config_.interactions.auto_detect) {
            if (config_.verbosity > 0) {
                std::printf("Detecting feature interactions...\n");
            }

            InteractionDetector detector;
            auto interactions = detector.detect(*train_data_, config_.interactions);

            if (!interactions.empty()) {
                if (config_.verbosity > 0) {
                    std::printf("Found %zu interactions:\n", interactions.size());
                    for (size_t i = 0; i < std::min(interactions.size(), size_t(5)); ++i) {
                        const auto& inter = interactions[i];
                        std::printf("  Feature %u x Feature %u (score: %.4f)\n",
                                  inter.feature_a, inter.feature_b, inter.interaction_score);
                    }
                }

                // Generate interaction features
                interaction_generator_ = std::make_unique<InteractionGenerator>();
                FeatureIndex added = interaction_generator_->generate(
                    *train_data_, interactions, config_.interactions);

                if (config_.verbosity > 0) {
                    std::printf("Added %u interaction features\n", added);
                }

                // Re-bin with new features (only bin the new features)
                train_data_->compute_bins(config_);
            }
        }

        // Validation data
        std::unique_ptr<Dataset> valid_data;
        if (X_val.size() > 0) {
            auto X_val_buf = X_val.request();
            auto y_val_buf = y_val.request();

            valid_data = std::make_unique<Dataset>();
            valid_data->from_dense(
                static_cast<float*>(X_val_buf.ptr),
                static_cast<Index>(X_val_buf.shape[0]),
                static_cast<FeatureIndex>(X_val_buf.shape[1]),
                static_cast<float*>(y_val_buf.ptr)
            );

            // Apply interaction transformations if they were generated
            if (interaction_generator_) {
                interaction_generator_->apply(*valid_data, *train_data_);
            }

            valid_data->apply_bins(*train_data_);
        }

        // Train
        booster_ = std::make_unique<Booster>(config_);
        booster_->train(*train_data_, valid_data.get());

        is_fitted_ = true;
    }

    py::array_t<float> predict_proba(py::array_t<float> X) {
        if (!is_fitted_) {
            throw std::runtime_error("Model not fitted. Call fit() first.");
        }

        auto X_buf = X.request();
        Index n_samples = static_cast<Index>(X_buf.shape[0]);
        FeatureIndex n_features = static_cast<FeatureIndex>(X_buf.shape[1]);

        // OPTIMIZATION: Use fast no-binning path for binary classification
        // This avoids Dataset creation, data copy, and binning overhead
        // Only use slow path if we have interactions or multiclass
        if (n_classes_ <= 2 && !interaction_generator_) {
            return predict_proba_nobinning_fast(X);
        }

        // Slow path: required for multiclass or when we have interactions
        // If train_data_ is not available (loaded model), fall back to nobinning
        if (!train_data_) {
            return predict_proba_nobinning_fast(X);
        }

        Dataset test_data;
        test_data.from_dense(
            static_cast<float*>(X_buf.ptr),
            n_samples,
            n_features
        );

        // Apply interaction transformations if they were generated
        if (interaction_generator_) {
            interaction_generator_->apply(test_data, *train_data_);
        }

        test_data.apply_bins(*train_data_);

        if (n_classes_ > 2) {
            // Multiclass: return (n_samples, n_classes)
            auto result = py::array_t<float>({n_samples, static_cast<Index>(n_classes_)});
            auto result_buf = result.request();
            booster_->predict_proba_multiclass(test_data, static_cast<float*>(result_buf.ptr));
            return result;
        } else {
            // Binary: return (n_samples, 2) for sklearn compatibility
            std::vector<float> proba_1(n_samples);
            booster_->predict_proba(test_data, proba_1.data());

            auto result = py::array_t<float>({n_samples, static_cast<Index>(2)});
            auto result_buf = result.request();
            float* r = static_cast<float*>(result_buf.ptr);

            for (Index i = 0; i < n_samples; ++i) {
                r[i * 2] = 1.0f - proba_1[i];      // P(class=0)
                r[i * 2 + 1] = proba_1[i];         // P(class=1)
            }
            return result;
        }
    }

    // FASTEST probability prediction - no binning + cached flat tree data + SIMD
    // Only works with symmetric trees (mode='large' or 'auto' for large data)
    py::array_t<float> predict_proba_nobinning_fast(py::array_t<float> X) {
        if (!is_fitted_) {
            throw std::runtime_error("Model not fitted. Call fit() first.");
        }

        auto X_buf = X.request();
        Index n_samples = static_cast<Index>(X_buf.shape[0]);
        FeatureIndex n_features = static_cast<FeatureIndex>(X_buf.shape[1]);

        if (n_classes_ > 2) {
            // Multiclass: fall back to standard predict_proba for now
            // TODO: implement predict_proba_multiclass_nobinning_fast
            return predict_proba(X);
        }

        // Binary: return (n_samples, 2) for sklearn compatibility
        std::vector<float> proba_1(n_samples);
        booster_->predict_proba_nobinning_fast(
            static_cast<float*>(X_buf.ptr),
            n_samples,
            n_features,
            proba_1.data()
        );

        auto result = py::array_t<float>({n_samples, static_cast<Index>(2)});
        auto result_buf = result.request();
        float* r = static_cast<float*>(result_buf.ptr);

        for (Index i = 0; i < n_samples; ++i) {
            r[i * 2] = 1.0f - proba_1[i];      // P(class=0)
            r[i * 2 + 1] = proba_1[i];         // P(class=1)
        }
        return result;
    }

    py::array_t<int> predict(py::array_t<float> X) {
        if (!is_fitted_) {
            throw std::runtime_error("Model not fitted. Call fit() first.");
        }

        // Use predict_proba to ensure consistency
        auto proba = predict_proba(X);
        auto proba_buf = proba.request();
        float* p = static_cast<float*>(proba_buf.ptr);
        Index n_samples = static_cast<Index>(proba_buf.shape[0]);

        auto result = py::array_t<int>(n_samples);
        auto result_buf = result.request();
        int* r = static_cast<int*>(result_buf.ptr);

        if (n_classes_ > 2) {
            // Multiclass: argmax over probabilities
            for (Index i = 0; i < n_samples; ++i) {
                int best_class = 0;
                float best_prob = p[i * n_classes_];
                for (uint32_t c = 1; c < n_classes_; ++c) {
                    if (p[i * n_classes_ + c] > best_prob) {
                        best_prob = p[i * n_classes_ + c];
                        best_class = static_cast<int>(c);
                    }
                }
                r[i] = best_class;
            }
        } else {
            // Binary: threshold at 0.5
            for (Index i = 0; i < n_samples; ++i) {
                r[i] = p[i * 2 + 1] >= 0.5f ? 1 : 0;
            }
        }

        return result;
    }
    
    py::dict feature_importance() {
        if (!is_fitted_) {
            throw std::runtime_error("Model not fitted");
        }
        
        auto imp = booster_->feature_importance();
        
        py::dict result;
        result["gain"] = vector_to_numpy(imp.gain);
        result["gain_normalized"] = vector_to_numpy(imp.gain_normalized);
        
        return result;
    }
    
    void save(const std::string& path) {
        if (!is_fitted_) {
            throw std::runtime_error("Model not fitted");
        }
        booster_->save(path);
    }

    void load(const std::string& path) {
        booster_ = std::make_unique<Booster>(Booster::load(path));
        // Restore n_classes from loaded config
        n_classes_ = booster_->config().n_classes;
        is_fitted_ = true;
    }

    // Get tree structure for ONNX export
    py::list get_booster_dump() const {
        if (!is_fitted_) {
            throw std::runtime_error("Model not fitted");
        }

        py::list trees_list;
        const auto& ensemble = booster_->ensemble();
        size_t n_trees = ensemble.n_trees();

        for (size_t t = 0; t < n_trees; ++t) {
            const auto& tree = ensemble.tree(t);
            const auto& nodes = tree.nodes();
            Float weight = ensemble.tree_weight(t);

            py::dict tree_dict;
            tree_dict["weight"] = weight;
            tree_dict["n_nodes"] = nodes.size();
            tree_dict["depth"] = tree.depth();

            py::list nodes_list;
            for (size_t i = 0; i < nodes.size(); ++i) {
                const auto& node = nodes[i];
                py::dict node_dict;
                node_dict["is_leaf"] = static_cast<bool>(node.is_leaf);
                node_dict["feature"] = static_cast<int>(node.split_feature);
                node_dict["threshold"] = static_cast<int>(node.split_bin);
                node_dict["left_child"] = static_cast<int>(node.left_child);
                node_dict["right_child"] = static_cast<int>(node.right_child);
                node_dict["value"] = node.value;
                node_dict["default_left"] = static_cast<bool>(node.default_left);
                nodes_list.append(node_dict);
            }
            tree_dict["nodes"] = nodes_list;
            trees_list.append(tree_dict);
        }

        return trees_list;
    }

    Float get_base_prediction() const {
        if (!is_fitted_) {
            throw std::runtime_error("Model not fitted");
        }
        return booster_->base_prediction();
    }

    FeatureIndex get_n_features() const {
        if (!train_data_) {
            throw std::runtime_error("Model not fitted");
        }
        return train_data_->n_features();
    }
    
    size_t n_trees() const {
        return booster_ ? booster_->n_trees() : 0;
    }
    
    py::dict get_params() const {
        py::dict params;
        params["n_estimators"] = config_.boosting.n_estimators;
        params["learning_rate"] = config_.boosting.learning_rate;
        params["max_depth"] = config_.tree.max_depth;
        params["max_bins"] = config_.tree.max_bins;
        params["subsample"] = config_.boosting.subsample;
        params["colsample_bytree"] = config_.boosting.colsample_bytree;
        params["lambda_l2"] = config_.tree.lambda_l2;
        params["use_goss"] = config_.boosting.use_goss;
        params["use_gradtree"] = config_.tree.use_gradtree;
        params["n_classes"] = n_classes_;
        return params;
    }

    uint32_t n_classes() const { return n_classes_; }

private:
    Config config_;
    std::unique_ptr<Booster> booster_;
    std::unique_ptr<Dataset> train_data_;
    std::unique_ptr<InteractionGenerator> interaction_generator_;
    std::string loss_type_str_;
    std::string mode_;
    uint32_t n_classes_ = 2;
    bool is_fitted_ = false;
};

// ============================================================================
// Python Regressor Class
// ============================================================================

class TurboCatRegressor {
public:
    TurboCatRegressor(
        int n_estimators = 500,              // Optimized: fewer trees, higher LR (like classifier)
        float learning_rate = 0.1f,          // Slightly lower for regression stability
        int max_depth = 6,                   // Deeper trees for regression (CatBoost default)
        const std::string& loss = "mse",
        const std::string& mode = "auto",   // "small", "large", or "auto"
        float subsample = 1.0f,              // OPTIMIZED: no subsampling (faster, no quality loss)
        float colsample_bytree = 1.0f,       // All features for regression
        bool use_goss = false,
        float goss_top_rate = 0.2f,
        float goss_other_rate = 0.1f,
        int early_stopping_rounds = 50,
        float lambda_l2 = 3.0f,              // CatBoost default
        float min_child_weight = 1.0f,       // Minimum hessian sum
        int max_leaves = 64,                 // 2^6 = 64 (matching max_depth)
        const std::string& grow_policy = "depthwise",  // Depthwise is more stable
        int n_threads = -1,
        int seed = 42,
        int verbosity = 1
    ) : mode_(mode) {
        config_ = Config::regression();

        config_.boosting.n_estimators = n_estimators;
        config_.boosting.learning_rate = learning_rate;
        config_.tree.max_depth = max_depth;
        config_.tree.max_leaves = max_leaves;
        config_.boosting.subsample = subsample;
        config_.boosting.colsample_bytree = colsample_bytree;
        config_.boosting.use_goss = use_goss;
        config_.boosting.goss_top_rate = goss_top_rate;
        config_.boosting.goss_other_rate = goss_other_rate;
        config_.boosting.early_stopping_rounds = early_stopping_rounds;
        config_.tree.lambda_l2 = lambda_l2;
        config_.tree.min_child_weight = min_child_weight;
        config_.device.n_threads = n_threads;
        config_.seed = seed;
        config_.verbosity = verbosity;

        // Parse mode - controls tree architecture for inference speed
        // "small" = regular trees (best quality, fast training)
        // "large" = symmetric trees with bit-ops (fastest inference on large data)
        // "auto" = automatically select based on dataset size in fit()
        if (mode == "large") {
            config_.tree.use_symmetric = true;
        } else if (mode == "small") {
            config_.tree.use_symmetric = false;
        }
        // "auto" mode will be handled in fit() based on data size

        // Parse grow_policy - depthwise is default for stability
        if (grow_policy == "lossguide" || grow_policy == "leaf" || grow_policy == "leafwise") {
            config_.tree.grow_policy = GrowPolicy::Lossguide;
        } else {
            config_.tree.grow_policy = GrowPolicy::Depthwise;  // Default for stability
        }

        if (loss == "mse" || loss == "l2") {
            config_.loss.loss_type = LossType::MSE;
        } else if (loss == "mae" || loss == "l1") {
            config_.loss.loss_type = LossType::MAE;
        } else if (loss == "huber") {
            config_.loss.loss_type = LossType::Huber;
        }
    }
    
    void fit(py::array_t<float> X, py::array_t<float> y) {
        auto X_buf = X.request();
        auto y_buf = y.request();

        Index n_samples = static_cast<Index>(X_buf.shape[0]);
        FeatureIndex n_features = static_cast<FeatureIndex>(X_buf.shape[1]);

        // Handle "auto" mode - choose tree type based on dataset size
        if (mode_ == "auto") {
            // Use symmetric trees by default for fast batch inference
            // Symmetric trees support float_threshold for direct comparisons
            // This avoids the binning overhead in predict (4-5x speedup)
            config_.tree.use_symmetric = true;
        }

        if (config_.verbosity > 0) {
            std::printf("[REGRESSOR] fit() called, n_samples=%u, n_features=%u, n_estimators=%u, use_symmetric=%s\n",
                       n_samples, n_features, config_.boosting.n_estimators,
                       config_.tree.use_symmetric ? "true" : "false");
            std::fflush(stdout);
        }

        train_data_ = std::make_unique<Dataset>();
        train_data_->from_dense(
            static_cast<float*>(X_buf.ptr),
            n_samples,
            n_features,
            static_cast<float*>(y_buf.ptr)
        );
        train_data_->compute_bins(config_);

        if (config_.verbosity > 0) {
            std::printf("[REGRESSOR] calling booster_->train()\n");
            std::fflush(stdout);
        }

        booster_ = std::make_unique<Booster>(config_);
        booster_->train(*train_data_);

        if (config_.verbosity > 0) {
            std::printf("[REGRESSOR] training done, n_trees=%zu\n", booster_->n_trees());
            std::fflush(stdout);
        }

        is_fitted_ = true;
    }
    
    std::vector<float> predict(py::array_t<float> X, bool timing = false) {
        if (!is_fitted_) {
            throw std::runtime_error("Model not fitted");
        }

        // OPTIMIZATION: Always use fast no-binning path for regression
        // This avoids Dataset creation, data copy, and binning overhead
        return predict_nobinning_fast(X, timing);
    }

    // Fast prediction - avoids data copy by binning directly from numpy array
    std::vector<float> predict_fast(py::array_t<float> X, bool timing = false) {
        if (!is_fitted_) {
            throw std::runtime_error("Model not fitted");
        }

        auto X_buf = X.request();
        Index n_samples = static_cast<Index>(X_buf.shape[0]);
        FeatureIndex n_features = static_cast<FeatureIndex>(X_buf.shape[1]);

        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<float> predictions(n_samples);
        booster_->predict_raw_fast(
            static_cast<float*>(X_buf.ptr),
            n_samples,
            n_features,
            predictions.data()
        );

        auto t1 = std::chrono::high_resolution_clock::now();

        if (timing) {
            double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::printf("[TIMING] predict_fast total: %.1fms\n", total_ms);
            std::fflush(stdout);
        }

        return predictions;
    }

    // No-binning prediction - fastest path using raw float thresholds
    // Only works with symmetric trees (mode='symmetric')
    std::vector<float> predict_nobinning(py::array_t<float> X, bool timing = false) {
        if (!is_fitted_) {
            throw std::runtime_error("Model not fitted");
        }

        auto X_buf = X.request();
        Index n_samples = static_cast<Index>(X_buf.shape[0]);
        FeatureIndex n_features = static_cast<FeatureIndex>(X_buf.shape[1]);

        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<float> predictions(n_samples);
        booster_->predict_raw_nobinning(
            static_cast<float*>(X_buf.ptr),
            n_samples,
            n_features,
            predictions.data()
        );

        auto t1 = std::chrono::high_resolution_clock::now();

        if (timing) {
            double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::printf("[TIMING] predict_nobinning total: %.1fms\n", total_ms);
            std::fflush(stdout);
        }

        return predictions;
    }

    // FASTEST prediction - no binning + cached flat tree data + SIMD transpose
    // This is the recommended method for production inference
    // Only works with symmetric trees (mode='large' or 'auto' with large data)
    std::vector<float> predict_nobinning_fast(py::array_t<float> X, bool timing = false) {
        if (!is_fitted_) {
            throw std::runtime_error("Model not fitted");
        }

        auto X_buf = X.request();
        Index n_samples = static_cast<Index>(X_buf.shape[0]);
        FeatureIndex n_features = static_cast<FeatureIndex>(X_buf.shape[1]);

        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<float> predictions(n_samples);
        booster_->predict_raw_nobinning_fast(
            static_cast<float*>(X_buf.ptr),
            n_samples,
            n_features,
            predictions.data()
        );

        auto t1 = std::chrono::high_resolution_clock::now();

        if (timing) {
            double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::printf("[TIMING] predict_nobinning_fast total: %.1fms\n", total_ms);
            std::fflush(stdout);
        }

        return predictions;
    }

    size_t n_trees() const {
        return booster_ ? booster_->n_trees() : 0;
    }

    float base_prediction() const {
        return booster_ ? booster_->base_prediction() : 0.0f;
    }

    py::dict debug_info() const {
        py::dict info;
        info["n_trees"] = n_trees();
        info["base_prediction"] = base_prediction();
        info["is_fitted"] = is_fitted_;

        // Add tree structure info
        if (booster_ && booster_->n_trees() > 0) {
            const auto& ensemble = booster_->ensemble();
            size_t single_leaf_count = 0;
            size_t total_nodes = 0;
            size_t total_leaves = 0;

            for (size_t t = 0; t < ensemble.n_trees(); ++t) {
                const auto& tree = ensemble.tree(t);
                const auto& nodes = tree.nodes();
                total_nodes += nodes.size();

                size_t leaves = 0;
                for (const auto& node : nodes) {
                    if (node.is_leaf) leaves++;
                }
                total_leaves += leaves;

                if (nodes.size() == 1) single_leaf_count++;
            }

            info["single_leaf_trees"] = single_leaf_count;
            info["total_nodes"] = total_nodes;
            info["total_leaves"] = total_leaves;
            info["avg_nodes_per_tree"] = static_cast<float>(total_nodes) / ensemble.n_trees();
        }

        return info;
    }

    py::list tree_info() const {
        py::list trees;
        if (!booster_) return trees;

        const auto& ensemble = booster_->ensemble();
        for (size_t t = 0; t < ensemble.n_trees(); ++t) {
            py::dict tree_dict;
            const auto& tree = ensemble.tree(t);
            const auto& nodes = tree.nodes();

            size_t n_leaves = 0;
            float leaf_sum = 0.0f;
            for (const auto& node : nodes) {
                if (node.is_leaf) {
                    n_leaves++;
                    leaf_sum += node.value;
                }
            }

            tree_dict["n_nodes"] = nodes.size();
            tree_dict["n_leaves"] = n_leaves;
            tree_dict["leaf_sum"] = leaf_sum;
            tree_dict["is_single_leaf"] = (nodes.size() == 1);
            trees.append(tree_dict);
        }
        return trees;
    }

    // Debug prediction: trace path through trees for a few samples
    py::dict debug_predict(py::array_t<float> X) const {
        if (!is_fitted_ || !booster_) {
            throw std::runtime_error("Model not fitted");
        }

        auto X_buf = X.request();
        Index n_samples = static_cast<Index>(X_buf.shape[0]);
        FeatureIndex n_features = static_cast<FeatureIndex>(X_buf.shape[1]);

        // Create test dataset
        Dataset test_data;
        test_data.from_dense(
            static_cast<float*>(X_buf.ptr),
            n_samples,
            n_features
        );
        test_data.apply_bins(*train_data_);

        py::dict result;
        const auto& ensemble = booster_->ensemble();

        // Sample up to 5 samples for detailed trace
        Index samples_to_trace = std::min(n_samples, Index(5));

        // Check bin variance across samples for first few features
        py::list bin_variance_info;
        for (FeatureIndex f = 0; f < std::min(n_features, FeatureIndex(10)); ++f) {
            std::set<BinIndex> unique_bins;
            for (Index i = 0; i < std::min(n_samples, Index(100)); ++i) {
                unique_bins.insert(test_data.binned().get(i, f));
            }
            py::dict finfo;
            finfo["feature"] = f;
            finfo["unique_bins"] = unique_bins.size();

            // Get first few bins as list
            py::list sample_bins;
            for (Index i = 0; i < std::min(n_samples, Index(10)); ++i) {
                sample_bins.append(static_cast<int>(test_data.binned().get(i, f)));
            }
            finfo["sample_bins"] = sample_bins;
            bin_variance_info.append(finfo);
        }
        result["bin_variance"] = bin_variance_info;

        // Trace tree paths for first few samples
        py::list sample_traces;
        for (Index row = 0; row < samples_to_trace && ensemble.n_trees() > 0; ++row) {
            py::dict trace;
            trace["sample_idx"] = row;

            // Trace first tree only
            const auto& tree = ensemble.tree(0);
            const auto& nodes = tree.nodes();

            py::list path;
            TreeIndex node_idx = 0;

            while (!nodes[node_idx].is_leaf) {
                const auto& node = nodes[node_idx];
                BinIndex bin = test_data.binned().get(row, node.split_feature);
                bool go_right = (bin > node.split_bin);

                py::dict step;
                step["node_idx"] = node_idx;
                step["split_feature"] = node.split_feature;
                step["split_bin"] = static_cast<int>(node.split_bin);
                step["sample_bin"] = static_cast<int>(bin);
                step["go_right"] = go_right;
                path.append(step);

                node_idx = go_right ? node.right_child : node.left_child;
            }

            trace["path"] = path;
            trace["leaf_idx"] = node_idx;
            trace["leaf_value"] = nodes[node_idx].value;
            sample_traces.append(trace);
        }
        result["sample_traces"] = sample_traces;

        // Compare with training data bins
        py::list train_bin_info;
        for (FeatureIndex f = 0; f < std::min(n_features, FeatureIndex(10)); ++f) {
            std::set<BinIndex> unique_bins;
            Index train_samples = train_data_->n_samples();
            for (Index i = 0; i < std::min(train_samples, Index(100)); ++i) {
                unique_bins.insert(train_data_->binned().get(i, f));
            }
            py::dict finfo;
            finfo["feature"] = f;
            finfo["unique_bins"] = unique_bins.size();
            train_bin_info.append(finfo);
        }
        result["train_bin_variance"] = train_bin_info;

        // Add raw value comparison
        py::list raw_compare;
        for (FeatureIndex f = 0; f < std::min(n_features, FeatureIndex(5)); ++f) {
            py::dict cmp;
            cmp["feature"] = f;

            // Train data range
            float train_min = 1e30f, train_max = -1e30f;
            for (Index i = 0; i < train_data_->n_samples(); ++i) {
                float val = train_data_->raw_value(i, f);
                if (!std::isnan(val)) {
                    train_min = std::min(train_min, val);
                    train_max = std::max(train_max, val);
                }
            }
            cmp["train_min"] = train_min;
            cmp["train_max"] = train_max;

            // Test data range
            float* X_ptr = static_cast<float*>(X_buf.ptr);
            float test_min = 1e30f, test_max = -1e30f;
            for (Index i = 0; i < n_samples; ++i) {
                float val = X_ptr[i * n_features + f];
                if (!std::isnan(val)) {
                    test_min = std::min(test_min, val);
                    test_max = std::max(test_max, val);
                }
            }
            cmp["test_min"] = test_min;
            cmp["test_max"] = test_max;

            raw_compare.append(cmp);
        }
        result["raw_value_comparison"] = raw_compare;

        return result;
    }

    void save(const std::string& path) {
        if (!is_fitted_) {
            throw std::runtime_error("Model not fitted");
        }
        booster_->save(path);
    }

    void load(const std::string& path) {
        booster_ = std::make_unique<Booster>(Booster::load(path));
        is_fitted_ = true;
    }

    // Get tree structure for ONNX export
    py::list get_booster_dump() const {
        if (!is_fitted_) {
            throw std::runtime_error("Model not fitted");
        }

        py::list trees_list;
        const auto& ensemble = booster_->ensemble();
        size_t n_trees_count = ensemble.n_trees();

        for (size_t t = 0; t < n_trees_count; ++t) {
            const auto& tree = ensemble.tree(t);
            const auto& nodes = tree.nodes();
            Float weight = ensemble.tree_weight(t);

            py::dict tree_dict;
            tree_dict["weight"] = weight;
            tree_dict["n_nodes"] = nodes.size();
            tree_dict["depth"] = tree.depth();

            py::list nodes_list;
            for (size_t i = 0; i < nodes.size(); ++i) {
                const auto& node = nodes[i];
                py::dict node_dict;
                node_dict["is_leaf"] = static_cast<bool>(node.is_leaf);
                node_dict["feature"] = static_cast<int>(node.split_feature);
                node_dict["threshold"] = static_cast<int>(node.split_bin);
                node_dict["left_child"] = static_cast<int>(node.left_child);
                node_dict["right_child"] = static_cast<int>(node.right_child);
                node_dict["value"] = node.value;
                node_dict["default_left"] = static_cast<bool>(node.default_left);
                nodes_list.append(node_dict);
            }
            tree_dict["nodes"] = nodes_list;
            trees_list.append(tree_dict);
        }

        return trees_list;
    }

    FeatureIndex get_n_features() const {
        if (!train_data_) {
            throw std::runtime_error("Model not fitted");
        }
        return train_data_->n_features();
    }

private:
    Config config_;
    std::unique_ptr<Booster> booster_;
    std::unique_ptr<Dataset> train_data_;
    std::string mode_;
    bool is_fitted_ = false;
};

// ============================================================================
// Module Definition
// ============================================================================

PYBIND11_MODULE(_turbocat, m) {
    m.doc() = "TurboCat: Next-Generation Gradient Boosting";
    
    // Version info
    m.attr("__version__") = TURBOCAT_VERSION_STRING;
    
    // Classifier
    py::class_<TurboCatClassifier>(m, "TurboCatClassifier")
        .def(py::init<int, float, int, int, float, float, float, float,
                      const std::string&, bool, float, float, bool, bool, bool,
                      const std::string&, const std::string&, int, const std::string&,
                      const std::string&, bool, int, const std::string&, int, int, int>(),
             py::arg("n_estimators") = 500,   // Optimized: fewer trees, higher LR
             py::arg("learning_rate") = 0.2f, // Optimized: faster convergence
             py::arg("max_depth") = 6,
             py::arg("max_bins") = 255,
             py::arg("subsample") = 1.0f,         // No subsampling (faster training)
             py::arg("colsample_bytree") = 1.0f,  // Use all features
             py::arg("min_child_weight") = 1.0f,
             py::arg("lambda_l2") = 1.0f,
             py::arg("loss") = "logloss",
             py::arg("use_goss") = false,
             py::arg("goss_top_rate") = 0.2f,
             py::arg("goss_other_rate") = 0.1f,
             py::arg("use_gradtree") = false,
             py::arg("use_symmetric") = false,
             py::arg("use_ordered_boosting") = false,
             py::arg("grow_policy") = "depthwise",
             py::arg("mode") = "auto",  // "small", "large", or "auto"
             py::arg("early_stopping_rounds") = 50,
             py::arg("early_stopping_metric") = "loss",  // "loss", "roc_auc", "pr_auc", "f1"
             py::arg("cat_encoding") = "ordered",  // "ordered", "cv", "onehot"
             py::arg("auto_interactions") = false,  // Auto-detect feature interactions
             py::arg("max_interactions") = 10,      // Maximum interactions to detect
             py::arg("interaction_method") = "split",  // "split", "mutual_info", "correlation"
             py::arg("n_threads") = -1,
             py::arg("seed") = 42,
             py::arg("verbosity") = 1)
        .def("fit", &TurboCatClassifier::fit,
             py::arg("X"),
             py::arg("y"),
             py::arg("X_val") = py::array_t<float>(),
             py::arg("y_val") = py::array_t<float>(),
             py::arg("cat_features") = std::vector<int>())
        .def("predict", &TurboCatClassifier::predict,
             py::arg("X"))
        .def("predict_proba", &TurboCatClassifier::predict_proba)
        .def("predict_proba_nobinning_fast", &TurboCatClassifier::predict_proba_nobinning_fast)
        .def("feature_importance", &TurboCatClassifier::feature_importance)
        .def("save", &TurboCatClassifier::save)
        .def("load", &TurboCatClassifier::load)
        .def("get_params", &TurboCatClassifier::get_params)
        .def("get_booster_dump", &TurboCatClassifier::get_booster_dump)
        .def("get_base_prediction", &TurboCatClassifier::get_base_prediction)
        .def("get_n_features", &TurboCatClassifier::get_n_features)
        .def_property_readonly("n_trees", &TurboCatClassifier::n_trees)
        .def_property_readonly("n_classes_", &TurboCatClassifier::n_classes);
    
    // Regressor
    py::class_<TurboCatRegressor>(m, "TurboCatRegressor")
        .def(py::init<int, float, int, const std::string&, const std::string&, float, float, bool, float, float, int, float,
                      float, int, const std::string&, int, int, int>(),
             py::arg("n_estimators") = 500,        // Optimized: fewer trees, higher LR
             py::arg("learning_rate") = 0.1f,       // Slightly lower for regression stability
             py::arg("max_depth") = 6,              // Deeper trees for regression (CatBoost default)
             py::arg("loss") = "mse",
             py::arg("mode") = "auto",              // "small", "large", or "auto" for tree architecture
             py::arg("subsample") = 1.0f,           // No subsampling (faster training)
             py::arg("colsample_bytree") = 1.0f,    // All features for regression
             py::arg("use_goss") = false,
             py::arg("goss_top_rate") = 0.2f,
             py::arg("goss_other_rate") = 0.1f,
             py::arg("early_stopping_rounds") = 50,
             py::arg("lambda_l2") = 3.0f,           // CatBoost default
             py::arg("min_child_weight") = 1.0f,    // Minimum hessian sum
             py::arg("max_leaves") = 64,            // 2^6 = 64 (matching max_depth)
             py::arg("grow_policy") = "depthwise",  // Depthwise is more stable
             py::arg("n_threads") = -1,
             py::arg("seed") = 42,
             py::arg("verbosity") = 1)
        .def("fit", &TurboCatRegressor::fit)
        .def("predict", &TurboCatRegressor::predict, py::arg("X"), py::arg("timing") = false)
        .def("predict_fast", &TurboCatRegressor::predict_fast, py::arg("X"), py::arg("timing") = false)
        .def("predict_nobinning", &TurboCatRegressor::predict_nobinning, py::arg("X"), py::arg("timing") = false)
        .def("predict_nobinning_fast", &TurboCatRegressor::predict_nobinning_fast, py::arg("X"), py::arg("timing") = false)
        .def("debug_info", &TurboCatRegressor::debug_info)
        .def("tree_info", &TurboCatRegressor::tree_info)
        .def("debug_predict", &TurboCatRegressor::debug_predict)
        .def("save", &TurboCatRegressor::save)
        .def("load", &TurboCatRegressor::load)
        .def("get_booster_dump", &TurboCatRegressor::get_booster_dump)
        .def("get_n_features", &TurboCatRegressor::get_n_features)
        .def_property_readonly("n_trees", &TurboCatRegressor::n_trees)
        .def_property_readonly("base_prediction", &TurboCatRegressor::base_prediction);

    // Utility functions
    m.def("print_info", &print_info, "Print TurboCat library information");

    // ========================================================================
    // Metrics Module
    // ========================================================================
    auto metrics = m.def_submodule("metrics", "Evaluation metrics for TurboCat");

    // MetricType enum
    py::enum_<MetricType>(metrics, "MetricType")
        .value("LogLoss", MetricType::LogLoss)
        .value("Accuracy", MetricType::Accuracy)
        .value("Precision", MetricType::Precision)
        .value("Recall", MetricType::Recall)
        .value("F1", MetricType::F1)
        .value("ROC_AUC", MetricType::ROC_AUC)
        .value("PR_AUC", MetricType::PR_AUC)
        .value("MSE", MetricType::MSE)
        .value("MAE", MetricType::MAE)
        .value("RMSE", MetricType::RMSE)
        .export_values();

    // Classification metrics
    metrics.def("roc_auc", [](py::array_t<float> y_true, py::array_t<float> y_pred) {
        auto yt = y_true.unchecked<1>();
        auto yp = y_pred.unchecked<1>();
        return Metrics::roc_auc(yt.data(0), yp.data(0), yt.size());
    }, "Compute ROC-AUC score", py::arg("y_true"), py::arg("y_pred"));

    metrics.def("pr_auc", [](py::array_t<float> y_true, py::array_t<float> y_pred) {
        auto yt = y_true.unchecked<1>();
        auto yp = y_pred.unchecked<1>();
        return Metrics::pr_auc(yt.data(0), yp.data(0), yt.size());
    }, "Compute PR-AUC score", py::arg("y_true"), py::arg("y_pred"));

    metrics.def("log_loss", [](py::array_t<float> y_true, py::array_t<float> y_pred, float eps) {
        auto yt = y_true.unchecked<1>();
        auto yp = y_pred.unchecked<1>();
        return Metrics::log_loss(yt.data(0), yp.data(0), yt.size(), eps);
    }, "Compute log loss", py::arg("y_true"), py::arg("y_pred"), py::arg("eps") = 1e-15f);

    metrics.def("accuracy", [](py::array_t<float> y_true, py::array_t<float> y_pred, float threshold) {
        auto yt = y_true.unchecked<1>();
        auto yp = y_pred.unchecked<1>();
        return Metrics::accuracy(yt.data(0), yp.data(0), yt.size(), threshold);
    }, "Compute accuracy", py::arg("y_true"), py::arg("y_pred"), py::arg("threshold") = 0.5f);

    metrics.def("precision", [](py::array_t<float> y_true, py::array_t<float> y_pred, float threshold) {
        auto yt = y_true.unchecked<1>();
        auto yp = y_pred.unchecked<1>();
        return Metrics::precision(yt.data(0), yp.data(0), yt.size(), threshold);
    }, "Compute precision", py::arg("y_true"), py::arg("y_pred"), py::arg("threshold") = 0.5f);

    metrics.def("recall", [](py::array_t<float> y_true, py::array_t<float> y_pred, float threshold) {
        auto yt = y_true.unchecked<1>();
        auto yp = y_pred.unchecked<1>();
        return Metrics::recall(yt.data(0), yp.data(0), yt.size(), threshold);
    }, "Compute recall", py::arg("y_true"), py::arg("y_pred"), py::arg("threshold") = 0.5f);

    metrics.def("f1_score", [](py::array_t<float> y_true, py::array_t<float> y_pred, float threshold) {
        auto yt = y_true.unchecked<1>();
        auto yp = y_pred.unchecked<1>();
        return Metrics::f1_score(yt.data(0), yp.data(0), yt.size(), threshold);
    }, "Compute F1 score", py::arg("y_true"), py::arg("y_pred"), py::arg("threshold") = 0.5f);

    metrics.def("find_optimal_threshold", [](py::array_t<float> y_true, py::array_t<float> y_pred,
                                              MetricType metric, int n_thresholds) {
        auto yt = y_true.unchecked<1>();
        auto yp = y_pred.unchecked<1>();
        return Metrics::find_optimal_threshold(yt.data(0), yp.data(0), yt.size(), metric, n_thresholds);
    }, "Find optimal threshold for a metric",
       py::arg("y_true"), py::arg("y_pred"),
       py::arg("metric") = MetricType::F1, py::arg("n_thresholds") = 100);

    // Regression metrics
    metrics.def("mse", [](py::array_t<float> y_true, py::array_t<float> y_pred) {
        auto yt = y_true.unchecked<1>();
        auto yp = y_pred.unchecked<1>();
        return Metrics::mse(yt.data(0), yp.data(0), yt.size());
    }, "Compute mean squared error", py::arg("y_true"), py::arg("y_pred"));

    metrics.def("mae", [](py::array_t<float> y_true, py::array_t<float> y_pred) {
        auto yt = y_true.unchecked<1>();
        auto yp = y_pred.unchecked<1>();
        return Metrics::mae(yt.data(0), yp.data(0), yt.size());
    }, "Compute mean absolute error", py::arg("y_true"), py::arg("y_pred"));

    metrics.def("rmse", [](py::array_t<float> y_true, py::array_t<float> y_pred) {
        auto yt = y_true.unchecked<1>();
        auto yp = y_pred.unchecked<1>();
        return std::sqrt(Metrics::mse(yt.data(0), yp.data(0), yt.size()));
    }, "Compute root mean squared error", py::arg("y_true"), py::arg("y_pred"));
}
