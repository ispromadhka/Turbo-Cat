/**
 * TurboCat Booster Implementation
 * 
 * Main gradient boosting training loop with all innovations:
 * - GOSS sampling
 * - GradTree support
 * - Early stopping
 * - Feature importance
 */

#include "turbocat/booster.hpp"
#include <chrono>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace turbocat {

// ============================================================================
// Construction
// ============================================================================

Booster::Booster() : config_(Config::binary_classification()) {
    rng_state_ = config_.seed;
}

Booster::Booster(const Config& config) : config_(config) {
    config_.validate();
    rng_state_ = config_.seed;
}

// ============================================================================
// Training
// ============================================================================

void Booster::train(
    Dataset& train_data,
    Dataset* valid_data,
    TrainingCallback callback
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize
    initialize_training(train_data);
    
    // Store feature info for later prediction
    feature_info_.clear();
    for (FeatureIndex f = 0; f < train_data.n_features(); ++f) {
        feature_info_.push_back(train_data.feature_info(f));
    }
    
    // Predictions array
    AlignedVector<Float> train_preds(train_data.n_samples(), base_prediction_);
    AlignedVector<Float> valid_preds;
    if (valid_data) {
        valid_preds.resize(valid_data->n_samples(), base_prediction_);
    }
    
    // All sample indices - preallocate once
    std::vector<Index> all_indices(train_data.n_samples());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    
    // All feature indices - preallocate once
    std::vector<FeatureIndex> all_features(train_data.n_features());
    std::iota(all_features.begin(), all_features.end(), static_cast<FeatureIndex>(0));
    
    // Preallocate buffers for GOSS (avoid repeated allocations)
    std::vector<std::pair<Float, Index>> grad_indices;
    grad_indices.reserve(train_data.n_samples());
    
    // Training history
    history_.train_loss.clear();
    history_.valid_loss.clear();
    history_.iteration_time.clear();
    
    best_iteration_ = 0;
    best_valid_loss_ = 1e30f;
    uint32_t no_improvement_count = 0;
    
    // Main training loop
    for (uint32_t iter = 0; iter < config_.boosting.n_estimators; ++iter) {
        auto iter_start = std::chrono::high_resolution_clock::now();

        // Update gradients
        update_gradients(train_data, train_preds);
        
        // Quantize gradients if configured
        if (config_.tree.use_quantized_grad) {
            train_data.quantize_gradients();
        }
        
        // Sample selection
        std::vector<Index> sample_indices;
        
        if (config_.boosting.use_goss) {
            sample_indices = goss_sample(
                train_data,
                config_.boosting.goss_top_rate,
                config_.boosting.goss_other_rate
            );
        } else if (config_.boosting.subsample < 1.0f) {
            sample_indices = train_data.random_subsample(
                config_.boosting.subsample, next_random()
            );
        } else {
            sample_indices = all_indices;
        }
        
        // Feature selection
        std::vector<FeatureIndex> feature_indices;
        if (config_.boosting.colsample_bytree < 1.0f) {
            feature_indices = train_data.random_feature_subsample(
                config_.boosting.colsample_bytree, next_random()
            );
        } else {
            feature_indices = all_features;
        }
        
        // Build tree
        build_tree(train_data, sample_indices, feature_indices, train_preds);
        
        // Compute training loss
        Float train_loss = compute_loss(train_data, train_preds);
        history_.train_loss.push_back(train_loss);
        
        // Validation
        Float valid_loss = 0.0f;
        if (valid_data) {
            // Update validation predictions - only from last tree for efficiency
            const Float lr = config_.boosting.learning_rate;
            const auto& last_tree = ensemble_.n_trees() > 0 ? true : false;
            
            #pragma omp parallel for
            for (Index i = 0; i < valid_data->n_samples(); ++i) {
                Float pred = ensemble_.predict(*valid_data, i);
                valid_preds[i] = base_prediction_ + pred;
            }
            
            valid_loss = compute_loss(*valid_data, valid_preds);
            history_.valid_loss.push_back(valid_loss);
            
            // Early stopping
            if (valid_loss < best_valid_loss_ - config_.boosting.early_stopping_tolerance) {
                best_valid_loss_ = valid_loss;
                best_iteration_ = iter;
                no_improvement_count = 0;
            } else {
                no_improvement_count++;
            }
            
            if (no_improvement_count >= config_.boosting.early_stopping_rounds) {
                if (config_.verbosity > 0) {
                    std::printf("Early stopping at iteration %u (best: %u)\n", 
                               iter, best_iteration_);
                }
                break;
            }
        }
        
        auto iter_end = std::chrono::high_resolution_clock::now();
        double iter_time = std::chrono::duration<double>(iter_end - iter_start).count();
        history_.iteration_time.push_back(iter_time);
        
        // Logging
        if (config_.verbosity > 0 && (iter + 1) % config_.log_period == 0) {
            auto elapsed = std::chrono::duration<double>(iter_end - start_time).count();
            
            std::printf("[%4u] train_loss: %.6f", iter + 1, train_loss);
            if (valid_data) {
                std::printf("  valid_loss: %.6f", valid_loss);
            }
            std::printf("  (%.2fs)\n", elapsed);
        }
        
        // Callback
        if (callback) {
            TrainingInfo info;
            info.iteration = iter;
            info.train_loss = train_loss;
            info.valid_loss = valid_loss;
            info.best_valid_loss = best_valid_loss_;
            info.best_iteration = best_iteration_;
            info.elapsed_seconds = std::chrono::duration<double>(
                iter_end - start_time).count();
            info.n_trees = ensemble_.n_trees();
            
            if (!callback(info)) {
                break;  // Callback requested stop
            }
        }
    }
    
    if (config_.verbosity > 0) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end_time - start_time).count();
        std::printf("Training completed in %.2fs with %zu trees\n", 
                   total_time, ensemble_.n_trees());
    }
}

void Booster::continue_training(
    Dataset& train_data,
    uint32_t n_additional_trees,
    Dataset* valid_data
) {
    uint32_t original_n = config_.boosting.n_estimators;
    config_.boosting.n_estimators = static_cast<uint32_t>(ensemble_.n_trees()) + n_additional_trees;
    
    // Continue training...
    // (simplified - full implementation would resume from current state)
    
    config_.boosting.n_estimators = original_n;
}

// ============================================================================
// Tree Building
// ============================================================================

void Booster::build_tree(
    Dataset& data,
    const std::vector<Index>& sample_indices,
    const std::vector<FeatureIndex>& feature_indices,
    AlignedVector<Float>& predictions
) {
    auto tree = std::make_unique<Tree>(config_.tree);
    tree->build(data, sample_indices, *hist_builder_);
    
    // Update predictions
    Float lr = config_.boosting.learning_rate;
    
    #pragma omp parallel for
    for (Index i = 0; i < data.n_samples(); ++i) {
        Float tree_pred = tree->predict(data, i);
        predictions[i] += lr * tree_pred;
    }
    
    ensemble_.add_tree(std::move(tree), lr);
}

// ============================================================================
// Gradient Computation
// ============================================================================

void Booster::update_gradients(Dataset& data, const AlignedVector<Float>& predictions) {
    AlignedVector<Float> grads(data.n_samples());
    AlignedVector<Float> hess(data.n_samples());
    
    loss_->compute_gradients(
        data.labels().data(),
        predictions.data(),
        grads.data(),
        hess.data(),
        data.n_samples()
    );
    
    data.set_gradients(std::move(grads), std::move(hess));
}

Float Booster::compute_loss(const Dataset& data, const AlignedVector<Float>& predictions) const {
    return loss_->compute_loss(
        data.labels().data(),
        predictions.data(),
        data.n_samples()
    );
}

// ============================================================================
// GOSS Sampling
// ============================================================================

std::vector<Index> Booster::goss_sample(
    const Dataset& data,
    Float top_rate,
    Float other_rate
) {
    auto subsample = data.goss_subsample(top_rate, other_rate, next_random());
    return subsample.row_indices;
}

// ============================================================================
// Initialization
// ============================================================================

void Booster::initialize_training(Dataset& data) {
    // Create loss function
    loss_ = Loss::create(config_.loss, config_.task);
    
    // Set class-specific loss parameters
    if (config_.loss.loss_type == LossType::LDAM && config_.loss.auto_ldam_margins) {
        auto stats = data.compute_stats();
        if (!stats.class_distribution.empty()) {
            std::vector<Index> counts;
            for (Float f : stats.class_distribution) {
                counts.push_back(static_cast<Index>(f * data.n_samples()));
            }
            static_cast<LDAMLoss*>(loss_.get())->set_margins_from_counts(counts);
        }
    }
    
    if (config_.loss.loss_type == LossType::LogitAdjusted && config_.loss.auto_class_priors) {
        static_cast<LogitAdjustedLoss*>(loss_.get())->set_priors_from_labels(
            data.labels().data(), data.n_samples()
        );
    }
    
    // Initialize base prediction
    base_prediction_ = loss_->init_prediction(data.labels().data(), data.n_samples());
    
    // Create histogram builder
    hist_builder_ = HistogramBuilder::create(config_.device);
    
    // Compute categorical encodings if needed
    if (config_.categorical.method != CategoricalConfig::EncodingMethod::OneHot) {
        data.compute_target_statistics(config_);
    }
}

// ============================================================================
// Prediction
// ============================================================================

void Booster::predict_raw(const Dataset& data, Float* output, int n_trees) const {
    std::memset(output, 0, data.n_samples() * sizeof(Float));
    
    #pragma omp parallel for
    for (Index i = 0; i < data.n_samples(); ++i) {
        Float tree_pred = ensemble_.predict(data, i);
        output[i] = base_prediction_ + tree_pred;
    }
}

void Booster::predict_proba(const Dataset& data, Float* output, int n_trees) const {
    predict_raw(data, output, n_trees);
    
    // Transform to probabilities
    #pragma omp parallel for
    for (Index i = 0; i < data.n_samples(); ++i) {
        output[i] = loss_->transform_prediction(output[i]);
    }
}

std::vector<Prediction> Booster::predict_with_uncertainty(
    const Dataset& data,
    int n_trees
) const {
    std::vector<Prediction> results(data.n_samples());
    
    // Simple variance estimation via ensemble diversity
    // More sophisticated: GP posterior interpretation
    
    AlignedVector<Float> raw_preds(data.n_samples());
    predict_raw(data, raw_preds.data(), n_trees);
    
    for (Index i = 0; i < data.n_samples(); ++i) {
        Float mean = raw_preds[i];
        Float var = 0.1f;  // Placeholder - real implementation would compute from trees
        
        results[i] = Prediction(loss_->transform_prediction(mean), var);
    }
    
    return results;
}

Float Booster::predict_single(const Float* features, FeatureIndex n_features) const {
    Float raw = base_prediction_ + ensemble_.predict(features, n_features);
    return loss_->transform_prediction(raw);
}

// ============================================================================
// Feature Importance
// ============================================================================

FeatureImportance Booster::feature_importance() const {
    FeatureImportance imp;
    
    auto raw_importance = ensemble_.feature_importance();
    
    imp.gain = raw_importance;
    
    // Normalize
    Float sum = std::accumulate(raw_importance.begin(), raw_importance.end(), 0.0f);
    
    imp.gain_normalized.resize(raw_importance.size());
    if (sum > 0) {
        for (size_t i = 0; i < raw_importance.size(); ++i) {
            imp.gain_normalized[i] = raw_importance[i] / sum;
        }
    }
    
    return imp;
}

// ============================================================================
// Cross-Validation
// ============================================================================

CVResult Booster::cross_validate(Dataset& data, uint32_t n_folds, uint64_t seed) {
    CVResult result;
    
    Index n = data.n_samples();
    std::vector<Index> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    
    // Shuffle indices
    std::mt19937_64 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    Index fold_size = n / n_folds;
    
    for (uint32_t fold = 0; fold < n_folds; ++fold) {
        // Split indices
        Index valid_start = fold * fold_size;
        Index valid_end = (fold == n_folds - 1) ? n : (fold + 1) * fold_size;
        
        std::vector<Index> train_indices, valid_indices;
        for (Index i = 0; i < n; ++i) {
            if (i >= valid_start && i < valid_end) {
                valid_indices.push_back(indices[i]);
            } else {
                train_indices.push_back(indices[i]);
            }
        }
        
        // Create fold datasets (simplified - real impl would create proper subsets)
        // Train model and evaluate
        // ...
        
        // Placeholder
        result.train_scores.push_back(0.0f);
        result.valid_scores.push_back(0.0f);
    }
    
    result.mean_valid_score = std::accumulate(
        result.valid_scores.begin(), result.valid_scores.end(), 0.0f
    ) / n_folds;
    
    Float var = 0.0f;
    for (Float s : result.valid_scores) {
        var += (s - result.mean_valid_score) * (s - result.mean_valid_score);
    }
    result.std_valid_score = std::sqrt(var / n_folds);
    
    return result;
}

// ============================================================================
// Serialization
// ============================================================================

void Booster::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }
    save_binary(out);
}

Booster Booster::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file for reading: " + path);
    }
    return load_binary(in);
}

void Booster::save_binary(std::ostream& out) const {
    // Write magic number and version
    const char magic[] = "TCAT";
    out.write(magic, 4);
    
    uint32_t version = 1;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Write base prediction
    out.write(reinterpret_cast<const char*>(&base_prediction_), sizeof(base_prediction_));
    
    // Write number of trees
    size_t n_trees = ensemble_.n_trees();
    out.write(reinterpret_cast<const char*>(&n_trees), sizeof(n_trees));
    
    // Trees would be serialized here...
}

Booster Booster::load_binary(std::istream& in) {
    // Read and verify magic
    char magic[4];
    in.read(magic, 4);
    if (std::strncmp(magic, "TCAT", 4) != 0) {
        throw std::runtime_error("Invalid TurboCat model file");
    }
    
    uint32_t version;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    
    Booster booster;
    in.read(reinterpret_cast<char*>(&booster.base_prediction_), sizeof(booster.base_prediction_));
    
    // Read trees...
    
    return booster;
}

std::string Booster::to_json() const {
    // JSON serialization
    return "{}";
}

std::string Booster::to_pmml() const {
    // PMML export
    return "<PMML></PMML>";
}

// ============================================================================
// Random Number Generation
// ============================================================================

uint64_t Booster::next_random() {
    // xorshift64*
    rng_state_ ^= rng_state_ >> 12;
    rng_state_ ^= rng_state_ << 25;
    rng_state_ ^= rng_state_ >> 27;
    return rng_state_ * 0x2545F4914F6CDD1DULL;
}

Float Booster::random_float() {
    return static_cast<Float>(next_random() & 0xFFFFFFFF) / 4294967296.0f;
}

// ============================================================================
// Grid Search
// ============================================================================

GridSearchResult grid_search(
    Dataset& data,
    const std::vector<Config>& configs,
    uint32_t n_folds,
    uint64_t seed
) {
    GridSearchResult result;
    result.best_score = -1e30f;
    
    for (const auto& cfg : configs) {
        Booster booster(cfg);
        CVResult cv = booster.cross_validate(data, n_folds, seed);
        
        result.all_results.emplace_back(cfg, cv.mean_valid_score);
        
        if (cv.mean_valid_score > result.best_score) {
            result.best_score = cv.mean_valid_score;
            result.best_config = cfg;
        }
    }
    
    return result;
}

} // namespace turbocat
