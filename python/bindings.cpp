/**
 * TurboCat Python Bindings
 * 
 * Provides sklearn-compatible API for easy integration.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>

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
        const std::string& loss = "logloss",
        bool use_goss = true,
        float goss_top_rate = 0.2f,
        float goss_other_rate = 0.1f,
        bool use_gradtree = false,
        int early_stopping_rounds = 50,
        int n_threads = -1,
        int seed = 42,
        int verbosity = 1
    ) {
        config_ = Config::binary_classification();
        
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
        config_.boosting.early_stopping_rounds = early_stopping_rounds;
        config_.device.n_threads = n_threads;
        config_.seed = seed;
        config_.verbosity = verbosity;
        
        // Parse loss type
        if (loss == "logloss" || loss == "binary_crossentropy") {
            config_.loss.loss_type = LossType::LogLoss;
        } else if (loss == "focal" || loss == "robust_focal") {
            config_.loss.loss_type = LossType::RobustFocal;
        } else if (loss == "ldam") {
            config_.loss.loss_type = LossType::LDAM;
        } else if (loss == "logit_adjusted") {
            config_.loss.loss_type = LossType::LogitAdjusted;
        } else if (loss == "tsallis") {
            config_.loss.loss_type = LossType::Tsallis;
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
    
    std::vector<float> predict_proba(py::array_t<float> X) {
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
        
        // Return vector - pybind11 will convert to numpy
        std::vector<float> output(n_samples);
        booster_->predict_proba(test_data, output.data());
        
        return output;
    }
    
    py::array_t<int> predict(py::array_t<float> X, float threshold = 0.5f) {
        auto proba = predict_proba(X);  // Now returns std::vector<float>
        
        auto result = py::array_t<int>(proba.size());
        auto result_buf = result.request();
        int* r = static_cast<int*>(result_buf.ptr);
        
        for (size_t i = 0; i < proba.size(); ++i) {
            r[i] = proba[i] >= threshold ? 1 : 0;
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
        return params;
    }
    
private:
    Config config_;
    std::unique_ptr<Booster> booster_;
    std::unique_ptr<Dataset> train_data_;
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
        int seed = 42,
        int verbosity = 1
    ) {
        config_ = Config::regression();
        
        config_.boosting.n_estimators = n_estimators;
        config_.boosting.learning_rate = learning_rate;
        config_.tree.max_depth = max_depth;
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
        
        train_data_ = std::make_unique<Dataset>();
        train_data_->from_dense(
            static_cast<float*>(X_buf.ptr),
            n_samples,
            n_features,
            static_cast<float*>(y_buf.ptr)
        );
        train_data_->compute_bins(config_);
        
        booster_ = std::make_unique<Booster>(config_);
        booster_->train(*train_data_);
        
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
        
        auto result = py::array_t<float>(n_samples);
        auto result_buf = result.request();
        
        booster_->predict_raw(test_data, static_cast<float*>(result_buf.ptr));
        
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
                      const std::string&, bool, float, float, bool, int, int, int, int>(),
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
             py::arg("X"),
             py::arg("threshold") = 0.5f)
        .def("predict_proba", &TurboCatClassifier::predict_proba)
        .def("feature_importance", &TurboCatClassifier::feature_importance)
        .def("save", &TurboCatClassifier::save)
        .def("load", &TurboCatClassifier::load)
        .def("get_params", &TurboCatClassifier::get_params)
        .def_property_readonly("n_trees", &TurboCatClassifier::n_trees);
    
    // Regressor
    py::class_<TurboCatRegressor>(m, "TurboCatRegressor")
        .def(py::init<int, float, int, const std::string&, int, int>(),
             py::arg("n_estimators") = 1000,
             py::arg("learning_rate") = 0.05f,
             py::arg("max_depth") = 6,
             py::arg("loss") = "mse",
             py::arg("seed") = 42,
             py::arg("verbosity") = 1)
        .def("fit", &TurboCatRegressor::fit)
        .def("predict", &TurboCatRegressor::predict);
    
    // Utility functions
    m.def("print_info", &print_info, "Print TurboCat library information");
}
