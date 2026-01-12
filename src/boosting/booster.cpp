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
#include "turbocat/metrics.hpp"
#include <chrono>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TURBOCAT_AVX2
#include <immintrin.h>
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

    // Store feature info and bin edges for fast prediction
    feature_info_.clear();
    for (FeatureIndex f = 0; f < train_data.n_features(); ++f) {
        feature_info_.push_back(train_data.feature_info(f));
    }
    bin_edges_ = train_data.bin_edges();

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

    // Initialize best metric based on early stopping metric type
    const std::string& es_metric = config_.boosting.early_stopping_metric;
    bool es_higher_is_better = (es_metric == "roc_auc" || es_metric == "auc" ||
                                es_metric == "pr_auc" || es_metric == "f1");
    best_iteration_ = 0;
    best_valid_loss_ = es_higher_is_better ? -1e30f : 1e30f;
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

        if (config_.boosting.use_mvs) {
            // MVS (Minimum Variance Sampling) - best for large datasets
            auto subsample = train_data.mvs_subsample(
                config_.boosting.mvs_subsample, next_random()
            );
            sample_indices = std::move(subsample.row_indices);
        } else if (config_.boosting.use_goss) {
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
        
        // Validation: incrementally update predictions from the last tree only
        // This is O(n_valid * tree_depth) instead of O(n_valid * n_trees * tree_depth)
        Float valid_loss = 0.0f;
        Float valid_metric = 0.0f;
        if (valid_data) {
            const Float lr = config_.boosting.learning_rate;
            const Index n_valid = valid_data->n_samples();

            // Ensure buffer is large enough
            if (valid_pred_buffer_.size() < static_cast<size_t>(n_valid)) {
                valid_pred_buffer_.resize(n_valid);
            }

            // Get predictions from the last tree only
            if (config_.tree.use_symmetric) {
                size_t last_idx = symmetric_ensemble_.n_trees() - 1;
                symmetric_ensemble_.tree(last_idx).predict_batch(*valid_data, valid_pred_buffer_.data());
            } else {
                size_t last_idx = ensemble_.n_trees() - 1;
                ensemble_.tree(last_idx).predict_batch(*valid_data, valid_pred_buffer_.data());
            }

            // Incrementally add to validation predictions
            #pragma omp parallel for
            for (Index i = 0; i < n_valid; ++i) {
                valid_preds[i] += lr * valid_pred_buffer_[i];
            }

            valid_loss = compute_loss(*valid_data, valid_preds);
            history_.valid_loss.push_back(valid_loss);

            // Compute early stopping metric
            const std::string& es_metric = config_.boosting.early_stopping_metric;
            bool higher_is_better = false;

            if (es_metric == "loss" || es_metric == "logloss") {
                valid_metric = valid_loss;
                higher_is_better = false;
            } else if (es_metric == "roc_auc" || es_metric == "auc") {
                // Transform predictions to probabilities
                std::vector<Float> probs(valid_data->n_samples());
                for (Index i = 0; i < valid_data->n_samples(); ++i) {
                    probs[i] = 1.0f / (1.0f + std::exp(-valid_preds[i]));
                }
                valid_metric = Metrics::roc_auc(
                    valid_data->labels().data(),
                    probs.data(),
                    valid_data->n_samples()
                );
                higher_is_better = true;
            } else if (es_metric == "pr_auc") {
                std::vector<Float> probs(valid_data->n_samples());
                for (Index i = 0; i < valid_data->n_samples(); ++i) {
                    probs[i] = 1.0f / (1.0f + std::exp(-valid_preds[i]));
                }
                valid_metric = Metrics::pr_auc(
                    valid_data->labels().data(),
                    probs.data(),
                    valid_data->n_samples()
                );
                higher_is_better = true;
            } else if (es_metric == "f1") {
                std::vector<Float> probs(valid_data->n_samples());
                for (Index i = 0; i < valid_data->n_samples(); ++i) {
                    probs[i] = 1.0f / (1.0f + std::exp(-valid_preds[i]));
                }
                Float opt_thresh = Metrics::find_optimal_threshold(
                    valid_data->labels().data(),
                    probs.data(),
                    valid_data->n_samples(),
                    MetricType::F1
                );
                valid_metric = Metrics::f1_score(
                    valid_data->labels().data(),
                    probs.data(),
                    valid_data->n_samples(),
                    opt_thresh
                );
                higher_is_better = true;
            } else {
                valid_metric = valid_loss;
                higher_is_better = false;
            }

            // Early stopping: check if validation metric improved
            // Use ABSOLUTE comparison with tiny tolerance to avoid floating point issues
            // This matches CatBoost/XGBoost behavior - any improvement resets the counter
            constexpr Float eps = 1e-10f;  // Tiny tolerance for floating point comparison

            bool improved = higher_is_better
                ? (valid_metric > best_valid_loss_ + eps)
                : (valid_metric < best_valid_loss_ - eps);

            if (improved) {
                best_valid_loss_ = valid_metric;
                best_iteration_ = iter;
                no_improvement_count = 0;
            } else {
                no_improvement_count++;
            }

            // Debug: print improvement status every 10 iterations
            if (config_.verbosity > 1 && iter % 10 == 0) {
                std::printf("[ES Debug] iter=%u: metric=%.8f, best=%.8f, improved=%d, no_improve_count=%u\n",
                           iter, valid_metric, best_valid_loss_, improved ? 1 : 0, no_improvement_count);
            }

            // Only check early stopping after minimum iterations (10% of n_estimators or 50)
            uint32_t min_iterations = std::max(50u, config_.boosting.n_estimators / 10);
            if (iter >= min_iterations && no_improvement_count >= config_.boosting.early_stopping_rounds) {
                if (config_.verbosity > 0) {
                    std::printf("Early stopping at iteration %u (best: %u, %s: %.6f)\n",
                               iter, best_iteration_, es_metric.c_str(), best_valid_loss_);
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
                if (es_higher_is_better) {
                    std::printf("  %s: %.6f", es_metric.c_str(), valid_metric);
                }
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
            info.n_trees = config_.tree.use_symmetric ? symmetric_ensemble_.n_trees() : ensemble_.n_trees();

            if (!callback(info)) {
                break;  // Callback requested stop
            }
        }
    }

    if (config_.verbosity > 0) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end_time - start_time).count();
        size_t n_trees = config_.tree.use_symmetric ? symmetric_ensemble_.n_trees() : ensemble_.n_trees();
        std::printf("Training completed in %.2fs with %zu trees\n",
                   total_time, n_trees);
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
    const Float lr = config_.boosting.learning_rate;
    const Index n_samples = data.n_samples();

    // Reuse tree prediction buffer (avoid allocation every tree)
    if (tree_pred_buffer_.size() < static_cast<size_t>(n_samples)) {
        tree_pred_buffer_.resize(n_samples);
    }

    if (config_.tree.use_symmetric) {
        // Build symmetric (oblivious) tree for faster inference
        auto tree = std::make_unique<SymmetricTree>(config_.tree);
        tree->build(data, sample_indices, *hist_builder_);

        // Update predictions using batch method with reusable buffer
        tree->predict_batch(data, tree_pred_buffer_.data());

        #pragma omp parallel for
        for (Index i = 0; i < n_samples; ++i) {
            predictions[i] += lr * tree_pred_buffer_[i];
        }

        symmetric_ensemble_.add_tree(std::move(tree), lr);
    } else {
        // Build regular tree
        auto tree = std::make_unique<Tree>(config_.tree);
        tree->build(data, sample_indices, *hist_builder_);

        // Debug: Check tree structure (only first few trees)
        static int debug_tree_count = 0;
        if (config_.verbosity > 0 && debug_tree_count < 3) {
            size_t n_nodes = tree->nodes().size();
            size_t n_leaves = tree->n_leaves();
            Float first_leaf_value = 0.0f;
            for (const auto& node : tree->nodes()) {
                if (node.is_leaf) {
                    first_leaf_value = node.value;
                    break;
                }
            }
            std::printf("[DEBUG] Tree %d: n_nodes=%zu, n_leaves=%zu, first_leaf=%.6f\n",
                       debug_tree_count, n_nodes, n_leaves, first_leaf_value);
            std::fflush(stdout);
            debug_tree_count++;
        }

        // Update predictions using batch method with reusable buffer
        tree->predict_batch(data, tree_pred_buffer_.data());

        #pragma omp parallel for
        for (Index i = 0; i < n_samples; ++i) {
            predictions[i] += lr * tree_pred_buffer_[i];
        }

        ensemble_.add_tree(std::move(tree), lr);
    }
}

// ============================================================================
// Gradient Computation
// ============================================================================

void Booster::update_gradients(Dataset& data, const AlignedVector<Float>& predictions) {
    const Index n_samples = data.n_samples();

    // Ensure gradient arrays are allocated
    data.ensure_gradients_allocated(n_samples);

    // Compute gradients directly into dataset's arrays (avoid copy)
    loss_->compute_gradients(
        data.labels().data(),
        predictions.data(),
        data.gradients(),    // Direct write
        data.hessians(),     // Direct write
        n_samples
    );
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

void Booster::predict_raw_fast(
    const Float* data,
    Index n_samples,
    FeatureIndex n_features,
    Float* output
) const {
    // Fast prediction: bin data inline without copying raw values
    // This avoids the from_dense + apply_bins overhead

    // OPTIMIZED: Use thread-local buffers to reduce allocation overhead
    // For small batches, allocate on stack
    static thread_local std::vector<BinIndex> tl_binned_data;

    const size_t total_size = static_cast<size_t>(n_samples) * n_features;

    // Resize thread-local buffer if needed (amortized O(1))
    if (tl_binned_data.size() < total_size) {
        tl_binned_data.resize(total_size);
    }

    // Create BinnedData that uses our pre-allocated buffer via placement
    BinnedData binned(n_samples, n_features, 255);

    // Bin the data directly from the input pointer using stored bin_edges_
    // OPTIMIZED: Process feature-by-feature for better vectorization

    #pragma omp parallel for schedule(dynamic)
    for (FeatureIndex f = 0; f < n_features; ++f) {
        const auto& edges = bin_edges_[f];
        const size_t n_edges = edges.size();
        BinIndex* col = binned.column(f);

        if (n_edges == 0) {
            // No edges, all bins = 0
            std::memset(col, 0, n_samples * sizeof(BinIndex));
            continue;
        }

        // SIMD-accelerated linear scan for small bin counts
        if (n_edges <= 16) {
#ifdef TURBOCAT_AVX2
            // AVX2 accelerated binning: process 8 samples at a time
            // For each sample, compare against all edges using SIMD
            alignas(32) float edges_arr[16] = {0};
            for (size_t e = 0; e < n_edges && e < 16; ++e) {
                edges_arr[e] = edges[e];
            }

            Index i = 0;
            for (; i + 8 <= n_samples; i += 8) {
                // Load 8 values (with stride)
                alignas(32) float vals[8];
                for (int j = 0; j < 8; ++j) {
                    vals[j] = data[(i + j) * n_features + f];
                }
                __m256 v_vals = _mm256_load_ps(vals);

                // Check for NaN
                __m256 v_nan_mask = _mm256_cmp_ps(v_vals, v_vals, _CMP_UNORD_Q);

                // Count how many edges each value is >= using SIMD comparison
                __m256i v_bins = _mm256_setzero_si256();
                for (size_t e = 0; e < n_edges; ++e) {
                    __m256 v_edge = _mm256_set1_ps(edges_arr[e]);
                    __m256 v_cmp = _mm256_cmp_ps(v_vals, v_edge, _CMP_GE_OQ);
                    __m256i v_inc = _mm256_and_si256(_mm256_castps_si256(v_cmp), _mm256_set1_epi32(1));
                    v_bins = _mm256_add_epi32(v_bins, v_inc);
                }

                // Apply NaN mask
                __m256i v_nan_bin = _mm256_set1_epi32(255);
                v_bins = _mm256_blendv_epi8(v_bins, v_nan_bin, _mm256_castps_si256(v_nan_mask));

                // Store results
                alignas(32) int32_t bins_arr[8];
                _mm256_store_si256(reinterpret_cast<__m256i*>(bins_arr), v_bins);
                for (int j = 0; j < 8; ++j) {
                    col[i + j] = static_cast<BinIndex>(bins_arr[j]);
                }
            }

            // Handle remainder
            for (; i < n_samples; ++i) {
                Float val = data[i * n_features + f];
                if (std::isnan(val)) {
                    col[i] = 255;
                } else {
                    BinIndex bin = 0;
                    for (size_t e = 0; e < n_edges; ++e) {
                        if (val >= edges[e]) bin = static_cast<BinIndex>(e + 1);
                    }
                    col[i] = bin;
                }
            }
#else
            // Scalar fallback
            for (Index i = 0; i < n_samples; ++i) {
                Float val = data[i * n_features + f];
                if (std::isnan(val)) {
                    col[i] = 255;
                } else {
                    BinIndex bin = 0;
                    for (size_t e = 0; e < n_edges; ++e) {
                        if (val >= edges[e]) bin = static_cast<BinIndex>(e + 1);
                    }
                    col[i] = bin;
                }
            }
#endif
        } else {
            // Binary search for larger bin counts - 4x unrolled with prefetching
            Index i = 0;
            for (; i + 4 <= n_samples; i += 4) {
                // Prefetch
                if (i + 64 < n_samples) {
                    #ifdef _MSC_VER
                    _mm_prefetch(reinterpret_cast<const char*>(&data[(i + 64) * n_features + f]), _MM_HINT_T0);
                    #else
                    __builtin_prefetch(&data[(i + 64) * n_features + f], 0, 3);
                    #endif
                }

                Float v0 = data[i * n_features + f];
                Float v1 = data[(i+1) * n_features + f];
                Float v2 = data[(i+2) * n_features + f];
                Float v3 = data[(i+3) * n_features + f];

                if (std::isnan(v0)) {
                    col[i] = 255;
                } else {
                    auto it = std::lower_bound(edges.begin(), edges.end(), v0);
                    col[i] = static_cast<BinIndex>(it - edges.begin());
                }

                if (std::isnan(v1)) {
                    col[i+1] = 255;
                } else {
                    auto it = std::lower_bound(edges.begin(), edges.end(), v1);
                    col[i+1] = static_cast<BinIndex>(it - edges.begin());
                }

                if (std::isnan(v2)) {
                    col[i+2] = 255;
                } else {
                    auto it = std::lower_bound(edges.begin(), edges.end(), v2);
                    col[i+2] = static_cast<BinIndex>(it - edges.begin());
                }

                if (std::isnan(v3)) {
                    col[i+3] = 255;
                } else {
                    auto it = std::lower_bound(edges.begin(), edges.end(), v3);
                    col[i+3] = static_cast<BinIndex>(it - edges.begin());
                }
            }

            // Handle remainder
            for (; i < n_samples; ++i) {
                Float val = data[i * n_features + f];
                if (std::isnan(val)) {
                    col[i] = 255;
                } else {
                    auto it = std::lower_bound(edges.begin(), edges.end(), val);
                    col[i] = static_cast<BinIndex>(it - edges.begin());
                }
            }
        }
    }

    // Create temporary dataset with the binned data
    // Set sample/feature counts for predict_batch to work correctly
    Dataset temp;
    temp.set_n_samples(n_samples);
    temp.set_n_features(n_features);
    temp.binned() = std::move(binned);

    // Initialize output
    std::memset(output, 0, n_samples * sizeof(Float));

    // Predict using the appropriate ensemble
    if (config_.tree.use_symmetric) {
        symmetric_ensemble_.predict_batch(temp, output);
    } else {
        ensemble_.predict_batch_optimized(temp, output, config_.device.n_threads);
    }

    // Add base prediction
    #pragma omp parallel for
    for (Index i = 0; i < n_samples; ++i) {
        output[i] += base_prediction_;
    }
}

void Booster::predict_raw_nobinning(
    const Float* data,
    Index n_samples,
    FeatureIndex n_features,
    Float* output
) const {
    // Fastest prediction path - no binning required!
    // Uses raw float thresholds stored in trees for direct comparison
    // Only works with symmetric trees

    if (!config_.tree.use_symmetric) {
        // Fall back to predict_raw_fast for non-symmetric trees
        predict_raw_fast(data, n_samples, n_features, output);
        return;
    }

    // Use raw float prediction on symmetric ensemble
    symmetric_ensemble_.predict_batch_raw(data, n_samples, n_features, output);

    // Add base prediction
    #pragma omp parallel for
    for (Index i = 0; i < n_samples; ++i) {
        output[i] += base_prediction_;
    }
}

void Booster::predict_raw_nobinning_fast(
    const Float* data,
    Index n_samples,
    FeatureIndex n_features,
    Float* output
) const {
    // FASTEST prediction path:
    // - No binning required (raw float thresholds)
    // - Cached flat tree data (no per-call allocation)
    // - SIMD transpose for column-major access (8 consecutive floats per load)
    // Only works with symmetric trees

    if (!config_.tree.use_symmetric) {
        // Fall back to predict_raw_fast for non-symmetric trees
        predict_raw_fast(data, n_samples, n_features, output);
        return;
    }

    // Use the optimized raw float prediction with cached FastFloatEnsemble
    symmetric_ensemble_.predict_batch_raw_fast(data, n_samples, n_features, output);

    // Add base prediction
    #pragma omp parallel for
    for (Index i = 0; i < n_samples; ++i) {
        output[i] += base_prediction_;
    }
}

void Booster::predict_proba_nobinning_fast(
    const Float* data,
    Index n_samples,
    FeatureIndex n_features,
    Float* output
) const {
    // FASTEST probability prediction:
    // - Uses predict_raw_nobinning_fast for raw scores
    // - Applies sigmoid transformation for probabilities

    // Get raw predictions
    predict_raw_nobinning_fast(data, n_samples, n_features, output);

    // Transform to probabilities using SIMD-optimized sigmoid
    #pragma omp parallel for
    for (Index i = 0; i < n_samples; ++i) {
        output[i] = loss_->transform_prediction(output[i]);
    }
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

    uint32_t version = 3;  // Version 3 with bin_edges for prediction after load
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

    // Write use_symmetric flag
    uint8_t use_symmetric = config_.tree.use_symmetric ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&use_symmetric), sizeof(use_symmetric));

    // Write bin_edges_ for prediction after loading
    size_t n_features = bin_edges_.size();
    out.write(reinterpret_cast<const char*>(&n_features), sizeof(n_features));
    for (size_t f = 0; f < n_features; ++f) {
        size_t n_edges = bin_edges_[f].size();
        out.write(reinterpret_cast<const char*>(&n_edges), sizeof(n_edges));
        if (n_edges > 0) {
            out.write(reinterpret_cast<const char*>(bin_edges_[f].data()), n_edges * sizeof(Float));
        }
    }

    // Write number of trees
    size_t n_trees = config_.tree.use_symmetric ? symmetric_ensemble_.n_trees() : ensemble_.n_trees();
    out.write(reinterpret_cast<const char*>(&n_trees), sizeof(n_trees));

    if (config_.tree.use_symmetric) {
        // Write symmetric tree weights and trees
        for (size_t i = 0; i < n_trees; ++i) {
            Float weight = symmetric_ensemble_.tree_weight(i);
            out.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
        }
        for (size_t i = 0; i < n_trees; ++i) {
            symmetric_ensemble_.tree(i).save(out);
        }
    } else {
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

    if (version < 1 || version > 3) {
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

    if (version >= 3) {
        // Read use_symmetric flag
        uint8_t use_symmetric;
        in.read(reinterpret_cast<char*>(&use_symmetric), sizeof(use_symmetric));
        booster.config_.tree.use_symmetric = (use_symmetric != 0);

        // Read bin_edges_
        size_t n_features;
        in.read(reinterpret_cast<char*>(&n_features), sizeof(n_features));
        booster.bin_edges_.resize(n_features);
        for (size_t f = 0; f < n_features; ++f) {
            size_t n_edges;
            in.read(reinterpret_cast<char*>(&n_edges), sizeof(n_edges));
            booster.bin_edges_[f].resize(n_edges);
            if (n_edges > 0) {
                in.read(reinterpret_cast<char*>(booster.bin_edges_[f].data()), n_edges * sizeof(Float));
            }
        }
    }

    // Read number of trees
    size_t n_trees;
    in.read(reinterpret_cast<char*>(&n_trees), sizeof(n_trees));

    if (version >= 3 && booster.config_.tree.use_symmetric) {
        // Read symmetric tree weights and trees
        std::vector<Float> weights(n_trees);
        for (size_t i = 0; i < n_trees; ++i) {
            in.read(reinterpret_cast<char*>(&weights[i]), sizeof(Float));
        }

        booster.symmetric_ensemble_ = SymmetricEnsemble(booster.config_.n_classes);

        for (size_t i = 0; i < n_trees; ++i) {
            auto tree = std::make_unique<SymmetricTree>(SymmetricTree::load(in));
            booster.symmetric_ensemble_.add_tree(std::move(tree), weights[i]);
        }
    } else if (version >= 2) {
        // Read regular tree weights and trees
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

            // Early stopping: check if validation loss improved
            // Use tiny tolerance to avoid floating point comparison issues
            constexpr Float eps = 1e-10f;

            if (valid_loss < best_valid_loss_ - eps) {
                best_valid_loss_ = valid_loss;
                best_iteration_ = iter;
                no_improvement_count = 0;
            } else {
                no_improvement_count++;
            }

            // Only check after minimum iterations
            uint32_t min_iterations = std::max(50u, config_.boosting.n_estimators / 10);
            if (iter >= min_iterations && no_improvement_count >= config_.boosting.early_stopping_rounds) {
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
