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

    if (config_.verbosity > 0) {
        std::printf("[DEBUG] train() started, n_samples=%u, n_features=%u\n",
                   train_data.n_samples(), train_data.n_features());
        std::fflush(stdout);
    }

    // Initialize
    initialize_training(train_data);

    if (config_.verbosity > 0) {
        std::printf("[DEBUG] initialized, base_prediction=%.4f, n_estimators=%u\n",
                   base_prediction_, config_.boosting.n_estimators);
        std::fflush(stdout);
    }

    // Store feature info for later prediction
    feature_info_.clear();
    for (FeatureIndex f = 0; f < train_data.n_features(); ++f) {
        feature_info_.push_back(train_data.feature_info(f));
    }

    // Check if multiclass
    bool is_multiclass = (config_.task == TaskType::MulticlassClassification && config_.n_classes > 2);

    if (config_.verbosity > 0) {
        std::printf("[DEBUG] is_multiclass=%d, task=%d, n_classes=%u\n",
                   is_multiclass, static_cast<int>(config_.task), config_.n_classes);
        std::fflush(stdout);
    }

    if (is_multiclass) {
        // Use multiclass training loop
        train_multiclass(train_data, valid_data, callback);
        return;
    }

    // Predictions array (binary/regression)
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

    if (config_.verbosity > 0) {
        std::printf("[DEBUG] Starting main loop, n_estimators=%u\n", config_.boosting.n_estimators);
        std::fflush(stdout);
    }

    // Ordered Boosting setup: create permutation folds
    const uint8_t n_perms = config_.boosting.n_permutations;
    std::vector<std::vector<Index>> perm_folds;
    std::vector<AlignedVector<Float>> perm_preds;  // Separate predictions per permutation

    if (config_.boosting.use_ordered_boosting) {
        // Create shuffled fold assignments
        std::vector<Index> shuffled = all_indices;
        std::mt19937_64 rng(config_.seed);
        std::shuffle(shuffled.begin(), shuffled.end(), rng);

        // Assign to folds
        perm_folds.resize(n_perms);
        Index fold_size = train_data.n_samples() / n_perms;
        for (Index i = 0; i < train_data.n_samples(); ++i) {
            uint8_t fold = std::min(static_cast<uint8_t>(i / fold_size), static_cast<uint8_t>(n_perms - 1));
            perm_folds[fold].push_back(shuffled[i]);
        }

        // Separate predictions for each permutation
        perm_preds.resize(n_perms);
        for (uint8_t p = 0; p < n_perms; ++p) {
            perm_preds[p].resize(train_data.n_samples(), base_prediction_);
        }

        if (config_.verbosity > 0) {
            std::printf("[DEBUG] Ordered Boosting: %u permutation folds\n", n_perms);
            std::fflush(stdout);
        }
    }

    // Main training loop
    for (uint32_t iter = 0; iter < config_.boosting.n_estimators; ++iter) {
        auto iter_start = std::chrono::high_resolution_clock::now();

        // Ordered Boosting: use out-of-fold predictions for gradients
        if (config_.boosting.use_ordered_boosting) {
            // For each sample, use prediction from fold that DIDN'T train on it
            uint8_t current_perm = iter % n_perms;
            #pragma omp parallel for
            for (Index i = 0; i < train_data.n_samples(); ++i) {
                train_preds[i] = perm_preds[current_perm][i];
            }
        }

        // Update gradients
        update_gradients(train_data, train_preds);

        // Debug: Check gradients on first iteration
        if (iter == 0 && config_.verbosity > 0) {
            Float grad_sum = 0.0f, hess_sum = 0.0f;
            Float grad_max = 0.0f, grad_min = 0.0f;
            const Float* grads = train_data.gradients();
            const Float* hess = train_data.hessians();
            for (Index i = 0; i < train_data.n_samples(); ++i) {
                grad_sum += std::abs(grads[i]);
                hess_sum += hess[i];
                grad_max = std::max(grad_max, grads[i]);
                grad_min = std::min(grad_min, grads[i]);
            }
            std::printf("[DEBUG] iter=0: grad_sum=%.4f, hess_sum=%.4f, grad_range=[%.4f, %.4f]\n",
                       grad_sum, hess_sum, grad_min, grad_max);
            std::fflush(stdout);
        }

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

        // Ordered Boosting: update only OOF predictions
        if (config_.boosting.use_ordered_boosting && ensemble_.n_trees() > 0) {
            uint8_t train_fold = iter % n_perms;
            const Float lr = config_.boosting.learning_rate;

            // This tree was trained on train_fold, so update predictions for OTHER folds
            for (uint8_t p = 0; p < n_perms; ++p) {
                if (p != train_fold) {
                    #pragma omp parallel for
                    for (Index i = 0; i < train_data.n_samples(); ++i) {
                        Float tree_pred = ensemble_.tree(ensemble_.n_trees() - 1).predict(train_data, i);
                        perm_preds[p][i] += lr * tree_pred;
                    }
                }
            }
        }
        
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
            std::fflush(stdout);
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
        std::fflush(stdout);
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
    Float lr = config_.boosting.learning_rate;

    if (config_.tree.use_symmetric) {
        // Build symmetric (oblivious) tree for faster inference
        auto tree = std::make_unique<SymmetricTree>(config_.tree);
        tree->build(data, sample_indices, *hist_builder_);

        // Update predictions using batch method (faster)
        std::vector<Float> tree_preds(data.n_samples());
        tree->predict_batch(data, tree_preds.data());

        #pragma omp parallel for
        for (Index i = 0; i < data.n_samples(); ++i) {
            predictions[i] += lr * tree_preds[i];
        }

        symmetric_ensemble_.add_tree(std::move(tree), lr);
    } else {
        // Build regular tree
        auto tree = std::make_unique<Tree>(config_.tree);
        tree->build(data, sample_indices, *hist_builder_);

        // Debug: Check tree structure
        size_t n_nodes = tree->nodes().size();
        size_t n_leaves = tree->n_leaves();
        Float first_leaf_value = 0.0f;
        for (const auto& node : tree->nodes()) {
            if (node.is_leaf) {
                first_leaf_value = node.value;
                break;
            }
        }

        // Debug output on first few trees
        static int debug_tree_count = 0;
        if (config_.verbosity > 0 && debug_tree_count < 3) {
            std::printf("[DEBUG] Tree %d: n_nodes=%zu, n_leaves=%zu, first_leaf=%.6f\n",
                       debug_tree_count, n_nodes, n_leaves, first_leaf_value);
            std::fflush(stdout);
            debug_tree_count++;
        }

        // Update predictions using batch method (faster than per-sample)
        std::vector<Float> tree_preds(data.n_samples());
        tree->predict_batch(data, tree_preds.data());

        #pragma omp parallel for
        for (Index i = 0; i < data.n_samples(); ++i) {
            predictions[i] += lr * tree_preds[i];
        }

        ensemble_.add_tree(std::move(tree), lr);
    }
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

void Booster::init_loss() {
    // Create loss function based on config (used when loading from file)
    loss_ = Loss::create(config_.loss, config_.task, config_.n_classes);
}

void Booster::initialize_training(Dataset& data) {
    // Create loss function
    loss_ = Loss::create(config_.loss, config_.task, config_.n_classes);
    
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
    Index n_samples = data.n_samples();

    if (config_.verbosity > 0) {
        std::printf("[DEBUG] predict_raw: n_samples=%u, n_trees=%zu, base_prediction=%.4f\n",
                   n_samples, ensemble_.n_trees(), base_prediction_);
        std::fflush(stdout);
    }

    // Use appropriate ensemble based on tree type
    if (config_.tree.use_symmetric) {
        symmetric_ensemble_.predict_batch(data, output);
    } else {
        // Use batched prediction with local caching
        ensemble_.predict_batch_optimized(data, output, config_.device.n_threads);
    }

    // Debug: check ensemble output before adding base
    if (config_.verbosity > 0 && n_samples > 0) {
        Float min_out = output[0], max_out = output[0], sum_out = 0.0f;
        for (Index i = 0; i < n_samples; ++i) {
            min_out = std::min(min_out, output[i]);
            max_out = std::max(max_out, output[i]);
            sum_out += output[i];
        }
        std::printf("[DEBUG] predict_raw: ensemble output range=[%.6f, %.6f], mean=%.6f\n",
                   min_out, max_out, sum_out / n_samples);
        std::fflush(stdout);
    }

    // Add base prediction
    #pragma omp parallel for
    for (Index i = 0; i < n_samples; ++i) {
        output[i] += base_prediction_;
    }

    // Debug: check final output
    if (config_.verbosity > 0 && n_samples > 0) {
        Float min_out = output[0], max_out = output[0];
        for (Index i = 0; i < n_samples; ++i) {
            min_out = std::min(min_out, output[i]);
            max_out = std::max(max_out, output[i]);
        }
        std::printf("[DEBUG] predict_raw: final output range=[%.6f, %.6f]\n", min_out, max_out);
        std::fflush(stdout);
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
    Float raw;
    if (config_.tree.use_symmetric) {
        // For symmetric trees, use base prediction only (single sample prediction not optimized)
        raw = base_prediction_;  // TODO: implement single prediction for symmetric trees
    } else {
        raw = base_prediction_ + ensemble_.predict(features, n_features);
    }
    return loss_->transform_prediction(raw);
}

// ============================================================================
// Feature Importance
// ============================================================================

FeatureImportance Booster::feature_importance() const {
    FeatureImportance imp;

    std::vector<Float> raw_importance;
    if (config_.tree.use_symmetric) {
        raw_importance = symmetric_ensemble_.feature_importance();
    } else {
        raw_importance = ensemble_.feature_importance();
    }

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

    uint32_t version = 2;  // Version 2 with full tree serialization
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write base prediction
    out.write(reinterpret_cast<const char*>(&base_prediction_), sizeof(base_prediction_));

    // Write task type and n_classes
    uint8_t task_type = static_cast<uint8_t>(config_.task);
    out.write(reinterpret_cast<const char*>(&task_type), sizeof(task_type));
    out.write(reinterpret_cast<const char*>(&config_.n_classes), sizeof(config_.n_classes));

    // Write essential config
    out.write(reinterpret_cast<const char*>(&config_.boosting.learning_rate), sizeof(config_.boosting.learning_rate));
    out.write(reinterpret_cast<const char*>(&config_.tree.max_depth), sizeof(config_.tree.max_depth));
    out.write(reinterpret_cast<const char*>(&config_.tree.lambda_l2), sizeof(config_.tree.lambda_l2));

    // Write number of trees
    size_t n_trees = ensemble_.n_trees();
    out.write(reinterpret_cast<const char*>(&n_trees), sizeof(n_trees));

    // Write tree weights
    for (size_t i = 0; i < n_trees; ++i) {
        Float weight = ensemble_.tree_weight(i);
        out.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
    }

    // Write each tree
    for (size_t i = 0; i < n_trees; ++i) {
        ensemble_.tree(i).save(out);
    }
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

    if (version < 1 || version > 2) {
        throw std::runtime_error("Unsupported TurboCat model version: " + std::to_string(version));
    }

    Booster booster;
    in.read(reinterpret_cast<char*>(&booster.base_prediction_), sizeof(booster.base_prediction_));

    if (version >= 2) {
        // Read task type and n_classes
        uint8_t task_type;
        in.read(reinterpret_cast<char*>(&task_type), sizeof(task_type));
        booster.config_.task = static_cast<TaskType>(task_type);
        in.read(reinterpret_cast<char*>(&booster.config_.n_classes), sizeof(booster.config_.n_classes));

        // Read essential config
        in.read(reinterpret_cast<char*>(&booster.config_.boosting.learning_rate),
               sizeof(booster.config_.boosting.learning_rate));
        in.read(reinterpret_cast<char*>(&booster.config_.tree.max_depth),
               sizeof(booster.config_.tree.max_depth));
        in.read(reinterpret_cast<char*>(&booster.config_.tree.lambda_l2),
               sizeof(booster.config_.tree.lambda_l2));
    }

    // Read number of trees
    size_t n_trees;
    in.read(reinterpret_cast<char*>(&n_trees), sizeof(n_trees));

    if (version >= 2) {
        // Read tree weights and trees
        std::vector<Float> weights(n_trees);
        for (size_t i = 0; i < n_trees; ++i) {
            in.read(reinterpret_cast<char*>(&weights[i]), sizeof(Float));
        }

        // Set n_classes for ensemble
        booster.ensemble_.set_n_classes(booster.config_.n_classes);

        // Read each tree
        for (size_t i = 0; i < n_trees; ++i) {
            auto tree = std::make_unique<Tree>(Tree::load(in));
            booster.ensemble_.add_tree(std::move(tree), weights[i]);
        }
    }

    // Initialize loss function
    booster.init_loss();

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

// ============================================================================
// Multiclass Training
// ============================================================================

void Booster::train_multiclass(
    Dataset& train_data,
    Dataset* valid_data,
    TrainingCallback callback
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    uint32_t n_classes = config_.n_classes;
    Index n_train = train_data.n_samples();

    // Set ensemble n_classes
    ensemble_.set_n_classes(n_classes);

    // Initialize base predictions (zeros for softmax)
    base_predictions_multiclass_.resize(n_classes, 0.0f);

    // Predictions arrays: n_samples * n_classes
    std::vector<Float> train_preds(n_train * n_classes, 0.0f);
    std::vector<Float> valid_preds;
    if (valid_data) {
        valid_preds.resize(valid_data->n_samples() * n_classes, 0.0f);
    }

    // Gradients and hessians: n_samples * n_classes
    std::vector<Float> gradients(n_train * n_classes);
    std::vector<Float> hessians(n_train * n_classes);

    // All sample indices
    std::vector<Index> all_indices(n_train);
    std::iota(all_indices.begin(), all_indices.end(), 0);

    // All feature indices
    std::vector<FeatureIndex> all_features(train_data.n_features());
    std::iota(all_features.begin(), all_features.end(), static_cast<FeatureIndex>(0));

    // Get loss function (CrossEntropyLoss)
    auto* ce_loss = dynamic_cast<CrossEntropyLoss*>(loss_.get());
    if (!ce_loss) {
        throw std::runtime_error("Multiclass training requires CrossEntropyLoss");
    }

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

        // Compute multiclass gradients
        ce_loss->compute_multiclass_gradients(
            train_data.labels().data(),
            train_preds.data(),
            gradients.data(),
            hessians.data(),
            n_train
        );

        // Sample selection (shared across all K trees for this iteration)
        std::vector<Index> sample_indices;
        if (config_.boosting.use_goss) {
            // For GOSS, use max absolute gradient across classes
            AlignedVector<Float> max_grads(n_train);
            AlignedVector<Float> sum_hess(n_train);
            #pragma omp parallel for
            for (Index i = 0; i < n_train; ++i) {
                Float max_g = 0.0f;
                Float h_sum = 0.0f;
                for (uint32_t c = 0; c < n_classes; ++c) {
                    max_g = std::max(max_g, std::abs(gradients[i * n_classes + c]));
                    h_sum += hessians[i * n_classes + c];
                }
                max_grads[i] = max_g;
                sum_hess[i] = h_sum;
            }
            train_data.set_gradients(std::move(max_grads), std::move(sum_hess));
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

        // Build K trees - one per class (like XGBoost/LightGBM)
        for (uint32_t k = 0; k < n_classes; ++k) {
            // Set gradients for class k
            AlignedVector<Float> class_grads(n_train);
            AlignedVector<Float> class_hess(n_train);
            #pragma omp parallel for
            for (Index i = 0; i < n_train; ++i) {
                class_grads[i] = gradients[i * n_classes + k];
                class_hess[i] = hessians[i * n_classes + k];
            }
            train_data.set_gradients(std::move(class_grads), std::move(class_hess));

            // Build tree for class k
            auto tree = std::make_unique<Tree>(config_.tree);
            tree->build(train_data, sample_indices, *hist_builder_);

            // Update predictions for class k using batch method (faster)
            Float lr = config_.boosting.learning_rate;
            std::vector<Float> tree_preds_batch(n_train);
            tree->predict_batch(train_data, tree_preds_batch.data());

            #pragma omp parallel for
            for (Index i = 0; i < n_train; ++i) {
                train_preds[i * n_classes + k] += lr * tree_preds_batch[i];
            }

            // Store tree with class index
            ensemble_.add_tree_for_class(std::move(tree), lr, k);
        }

        // Compute training loss
        Float train_loss = ce_loss->compute_loss(
            train_data.labels().data(),
            train_preds.data(),
            n_train
        );
        history_.train_loss.push_back(train_loss);

        // Validation
        Float valid_loss = 0.0f;
        if (valid_data) {
            // Update validation predictions
            ensemble_.predict_batch_multiclass(*valid_data, valid_preds.data());

            valid_loss = ce_loss->compute_loss(
                valid_data->labels().data(),
                valid_preds.data(),
                valid_data->n_samples()
            );
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
            std::fflush(stdout);
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
                break;
            }
        }
    }

    if (config_.verbosity > 0) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end_time - start_time).count();
        std::printf("Training completed in %.2fs with %zu trees\n",
                   total_time, ensemble_.n_trees());
        std::fflush(stdout);
    }
}

void Booster::build_tree_multiclass(
    Dataset& data,
    const std::vector<Index>& sample_indices,
    const std::vector<FeatureIndex>& feature_indices,
    std::vector<Float>& predictions,
    const std::vector<Float>& gradients,
    const std::vector<Float>& hessians
) {
    uint32_t n_classes = config_.n_classes;

    auto tree = std::make_unique<Tree>(config_.tree, n_classes);
    tree->build_multiclass(data, sample_indices, *hist_builder_, gradients, hessians);

    // Update predictions
    Float lr = config_.boosting.learning_rate;
    Index n_samples = data.n_samples();

    #pragma omp parallel
    {
        std::vector<Float> tree_pred(n_classes);

        #pragma omp for
        for (Index i = 0; i < n_samples; ++i) {
            tree->predict_multiclass(data, i, tree_pred.data());
            for (uint32_t c = 0; c < n_classes; ++c) {
                predictions[i * n_classes + c] += lr * tree_pred[c];
            }
        }
    }

    ensemble_.add_tree(std::move(tree), lr);
}

void Booster::predict_raw_multiclass(const Dataset& data, Float* output, int n_trees) const {
    Index n_samples = data.n_samples();
    uint32_t n_classes = config_.n_classes;

    // Use appropriate ensemble based on tree type
    if (config_.tree.use_symmetric) {
        symmetric_ensemble_.predict_batch_multiclass(data, output);
    } else {
        ensemble_.predict_batch_multiclass_optimized(data, output, config_.device.n_threads);
    }

    // Add base predictions
    if (!base_predictions_multiclass_.empty()) {
        #pragma omp parallel for
        for (Index i = 0; i < n_samples; ++i) {
            for (uint32_t c = 0; c < n_classes; ++c) {
                output[i * n_classes + c] += base_predictions_multiclass_[c];
            }
        }
    }
}

void Booster::predict_proba_multiclass(const Dataset& data, Float* output, int n_trees) const {
    predict_raw_multiclass(data, output, n_trees);

    // Transform to probabilities using softmax
    auto* ce_loss = dynamic_cast<CrossEntropyLoss*>(loss_.get());
    if (ce_loss) {
        ce_loss->transform_to_proba(output, output, data.n_samples());
    }
}

} // namespace turbocat
