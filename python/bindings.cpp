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

#include "turbocat/turbocat.hpp"

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
        int n_estimators = 1000,
        float learning_rate = 0.05f,
        int max_depth = 6,
        int max_bins = 255,
        float subsample = 0.8f,
        float colsample_bytree = 0.8f,
        float min_child_weight = 1.0f,
        float lambda_l2 = 1.0f,
        const std::string& loss = "auto",
        bool use_goss = true,
        float goss_top_rate = 0.2f,
        float goss_other_rate = 0.1f,
        bool use_gradtree = false,
        bool use_symmetric = false,  // Oblivious trees (experimental)
        int early_stopping_rounds = 50,
        int n_threads = -1,
        int seed = 42,
        int verbosity = 1
    ) : loss_type_str_(loss) {
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
        config_.boosting.early_stopping_rounds = early_stopping_rounds;
        config_.device.n_threads = n_threads;
        config_.seed = seed;
        config_.verbosity = verbosity;
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
        train_data_->compute_bins(config_);

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

        Dataset test_data;
        test_data.from_dense(
            static_cast<float*>(X_buf.ptr),
            n_samples,
            n_features
        );
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
        is_fitted_ = true;
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
    std::string loss_type_str_;
    uint32_t n_classes_ = 2;
    bool is_fitted_ = false;
};

// ============================================================================
// Python Regressor Class
// ============================================================================

class TurboCatRegressor {
public:
    TurboCatRegressor(
        int n_estimators = 1000,
        float learning_rate = 0.05f,
        int max_depth = 6,
        const std::string& loss = "mse",
        float subsample = 0.8f,
        float colsample_bytree = 0.8f,
        bool use_goss = true,
        float goss_top_rate = 0.2f,
        float goss_other_rate = 0.1f,
        int early_stopping_rounds = 50,
        int n_threads = -1,
        int seed = 42,
        int verbosity = 1
    ) {
        config_ = Config::regression();

        config_.boosting.n_estimators = n_estimators;
        config_.boosting.learning_rate = learning_rate;
        config_.tree.max_depth = max_depth;
        config_.boosting.subsample = subsample;
        config_.boosting.colsample_bytree = colsample_bytree;
        config_.boosting.use_goss = use_goss;
        config_.boosting.goss_top_rate = goss_top_rate;
        config_.boosting.goss_other_rate = goss_other_rate;
        config_.boosting.early_stopping_rounds = early_stopping_rounds;
        config_.device.n_threads = n_threads;
        config_.seed = seed;
        config_.verbosity = verbosity;

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

        if (config_.verbosity > 0) {
            std::printf("[REGRESSOR] fit() called, n_samples=%u, n_features=%u, n_estimators=%u\n",
                       n_samples, n_features, config_.boosting.n_estimators);
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
    
    py::array_t<float> predict(py::array_t<float> X) {
        if (!is_fitted_) {
            throw std::runtime_error("Model not fitted");
        }

        auto X_buf = X.request();
        Index n_samples = static_cast<Index>(X_buf.shape[0]);
        FeatureIndex n_features = static_cast<FeatureIndex>(X_buf.shape[1]);

        Dataset test_data;
        test_data.from_dense(
            static_cast<float*>(X_buf.ptr),
            n_samples,
            n_features
        );
        test_data.apply_bins(*train_data_);

        // Manual prediction to avoid predict_batch_optimized issues on GCC
        const auto& ensemble = booster_->ensemble();
        std::vector<float> predictions(n_samples, 0.0f);

        // Debug: print info about first 3 samples
        if (config_.verbosity > 0) {
            py::print("[CPP-DEBUG] n_trees=", ensemble.n_trees(), ", n_samples=", n_samples);
        }

        for (Index row = 0; row < n_samples; ++row) {
            float sum = 0.0f;
            for (size_t t = 0; t < ensemble.n_trees(); ++t) {
                const auto& tree = ensemble.tree(t);
                const auto& nodes = tree.nodes();
                if (nodes.empty()) continue;

                TreeIndex node_idx = 0;
                while (!nodes[node_idx].is_leaf) {
                    const TreeNode& node = nodes[node_idx];
                    BinIndex bin = test_data.binned().get(row, node.split_feature);
                    node_idx = (bin > node.split_bin) ? node.right_child : node.left_child;
                }
                float leaf_val = nodes[node_idx].value;
                sum += leaf_val;

                // Debug first 3 samples, first 2 trees
                if (config_.verbosity > 0 && row < 3 && t < 2) {
                    py::print("  row=", row, " tree=", t, " leaf_idx=", node_idx, " leaf_val=", leaf_val);
                }
            }
            predictions[row] = sum + booster_->base_prediction();

            if (config_.verbosity > 0 && row < 3) {
                py::print("  row=", row, " sum=", sum, " + base=", booster_->base_prediction(), " = ", predictions[row]);
            }
        }

        // Debug: check variance
        if (config_.verbosity > 0) {
            float min_p = predictions[0], max_p = predictions[0];
            for (Index i = 1; i < n_samples; ++i) {
                min_p = std::min(min_p, predictions[i]);
                max_p = std::max(max_p, predictions[i]);
            }
            py::print("[CPP-DEBUG] predictions range: [", min_p, ",", max_p, "]");
        }

        auto result = py::array_t<float>(n_samples);
        auto result_buf = result.request();
        std::memcpy(result_buf.ptr, predictions.data(), n_samples * sizeof(float));

        return result;
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

private:
    Config config_;
    std::unique_ptr<Booster> booster_;
    std::unique_ptr<Dataset> train_data_;
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
                      const std::string&, bool, float, float, bool, bool, int, int, int, int>(),
             py::arg("n_estimators") = 1000,
             py::arg("learning_rate") = 0.05f,
             py::arg("max_depth") = 6,
             py::arg("max_bins") = 255,
             py::arg("subsample") = 0.8f,
             py::arg("colsample_bytree") = 0.8f,
             py::arg("min_child_weight") = 1.0f,
             py::arg("lambda_l2") = 1.0f,
             py::arg("loss") = "logloss",
             py::arg("use_goss") = true,
             py::arg("goss_top_rate") = 0.2f,
             py::arg("goss_other_rate") = 0.1f,
             py::arg("use_gradtree") = false,
             py::arg("use_symmetric") = false,
             py::arg("early_stopping_rounds") = 50,
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
        .def("feature_importance", &TurboCatClassifier::feature_importance)
        .def("save", &TurboCatClassifier::save)
        .def("load", &TurboCatClassifier::load)
        .def("get_params", &TurboCatClassifier::get_params)
        .def_property_readonly("n_trees", &TurboCatClassifier::n_trees)
        .def_property_readonly("n_classes_", &TurboCatClassifier::n_classes);
    
    // Regressor
    py::class_<TurboCatRegressor>(m, "TurboCatRegressor")
        .def(py::init<int, float, int, const std::string&, float, float, bool, float, float, int, int, int, int>(),
             py::arg("n_estimators") = 1000,
             py::arg("learning_rate") = 0.05f,
             py::arg("max_depth") = 6,
             py::arg("loss") = "mse",
             py::arg("subsample") = 0.8f,
             py::arg("colsample_bytree") = 0.8f,
             py::arg("use_goss") = true,
             py::arg("goss_top_rate") = 0.2f,
             py::arg("goss_other_rate") = 0.1f,
             py::arg("early_stopping_rounds") = 50,
             py::arg("n_threads") = -1,
             py::arg("seed") = 42,
             py::arg("verbosity") = 1)
        .def("fit", &TurboCatRegressor::fit)
        .def("predict", &TurboCatRegressor::predict)
        .def("debug_info", &TurboCatRegressor::debug_info)
        .def("tree_info", &TurboCatRegressor::tree_info)
        .def("debug_predict", &TurboCatRegressor::debug_predict)
        .def_property_readonly("n_trees", &TurboCatRegressor::n_trees)
        .def_property_readonly("base_prediction", &TurboCatRegressor::base_prediction);

    // Utility functions
    m.def("print_info", &print_info, "Print TurboCat library information");
}
