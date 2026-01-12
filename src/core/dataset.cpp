/**
 * TurboCat Dataset Implementation
 */

#include "turbocat/dataset.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <cstring>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace turbocat {

// ============================================================================
// BinnedData Implementation
// ============================================================================

BinnedData::BinnedData(Index n_rows, FeatureIndex n_features, BinIndex max_bins)
    : n_rows_(n_rows), n_features_(n_features) {
    // Allocate column-major storage
    data_.resize(static_cast<size_t>(n_rows) * n_features);
}

void BinnedData::prepare_for_prediction() const {
    if (!row_major_data_.empty()) return;  // Already prepared

    // Transpose from column-major to row-major
    row_major_data_.resize(static_cast<size_t>(n_rows_) * n_features_);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (Index row = 0; row < n_rows_; ++row) {
        BinIndex* dst = row_major_data_.data() + row * n_features_;
        for (FeatureIndex feat = 0; feat < n_features_; ++feat) {
            dst[feat] = data_[feat * n_rows_ + row];
        }
    }
}

// ============================================================================
// Dataset Construction
// ============================================================================

void Dataset::from_dense(
    const Float* data,
    Index n_samples,
    FeatureIndex n_features,
    const Float* labels,
    const Float* weights,
    const std::vector<FeatureIndex>& cat_features
) {
    n_samples_ = n_samples;
    n_features_ = n_features;
    
    // Store raw data (row-major)
    raw_data_.resize(static_cast<size_t>(n_samples) * n_features);
    std::memcpy(raw_data_.data(), data, raw_data_.size() * sizeof(Float));
    
    // Store labels
    if (labels) {
        labels_.resize(n_samples);
        std::memcpy(labels_.data(), labels, n_samples * sizeof(Float));
    }
    
    // Store weights
    if (weights) {
        weights_.resize(n_samples);
        std::memcpy(weights_.data(), weights, n_samples * sizeof(Float));
    }
    
    // Initialize feature info
    feature_info_.resize(n_features);
    categorical_features_ = cat_features;
    
    // Mark categorical features
    for (auto f : cat_features) {
        if (f < n_features) {
            feature_info_[f].type = FeatureType::Categorical;
        }
    }
    
    // Detect types for remaining features
    detect_feature_types();
    
    // Allocate gradient storage
    gradients_.resize(n_samples, 0.0f);
    hessians_.resize(n_samples, 1.0f);
}

void Dataset::from_sparse_csr(
    const Float* data,
    const Index* indices,
    const Index* indptr,
    Index n_samples,
    FeatureIndex n_features,
    const Float* labels
) {
    n_samples_ = n_samples;
    n_features_ = n_features;
    
    // Convert CSR to dense (for simplicity; can optimize later)
    raw_data_.resize(static_cast<size_t>(n_samples) * n_features, 0.0f);
    
    for (Index i = 0; i < n_samples; ++i) {
        for (Index j = indptr[i]; j < indptr[i + 1]; ++j) {
            raw_data_[i * n_features + indices[j]] = data[j];
        }
    }
    
    if (labels) {
        labels_.resize(n_samples);
        std::memcpy(labels_.data(), labels, n_samples * sizeof(Float));
    }
    
    feature_info_.resize(n_features);
    detect_feature_types();
    
    gradients_.resize(n_samples, 0.0f);
    hessians_.resize(n_samples, 1.0f);
}

// ============================================================================
// Feature Type Detection
// ============================================================================

void Dataset::detect_feature_types() {
    for (FeatureIndex f = 0; f < n_features_; ++f) {
        if (feature_info_[f].type == FeatureType::Categorical) {
            continue;  // Already marked
        }
        
        // Check if binary
        bool is_binary = true;
        Float min_val = 1e30f;
        Float max_val = -1e30f;
        bool has_nan = false;
        
        for (Index i = 0; i < n_samples_; ++i) {
            Float val = raw_data_[i * n_features_ + f];
            
            if (std::isnan(val)) {
                has_nan = true;
                continue;
            }
            
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            
            if (val != 0.0f && val != 1.0f) {
                is_binary = false;
            }
        }
        
        feature_info_[f].min_value = min_val;
        feature_info_[f].max_value = max_val;
        feature_info_[f].has_missing = has_nan;
        
        if (is_binary && !has_nan) {
            feature_info_[f].type = FeatureType::Boolean;
        } else {
            feature_info_[f].type = FeatureType::Numerical;
        }
    }
}

// ============================================================================
// Binning
// ============================================================================

void Dataset::compute_bins(const Config& config) {
    BinIndex max_bins = static_cast<BinIndex>(config.tree.max_bins);
    binned_data_ = BinnedData(n_samples_, n_features_, max_bins);
    bin_edges_.resize(n_features_);
    
    #pragma omp parallel for schedule(dynamic) if(n_features_ > 10)
    for (FeatureIndex f = 0; f < n_features_; ++f) {
        if (feature_info_[f].type == FeatureType::Categorical) {
            // Check if we have ordered target statistics encoding
            bool use_ts_encoding = (f < ordered_cat_encodings_.size() &&
                                    !ordered_cat_encodings_[f].empty());

            if (use_ts_encoding) {
                // Use target statistics encoding: treat as numerical for binning
                // The encoded values are continuous, so quantile binning works well
                std::vector<Float> values;
                values.reserve(n_samples_);

                for (Index i = 0; i < n_samples_; ++i) {
                    Float encoded = ordered_cat_encodings_[f][i];
                    if (!std::isnan(encoded)) {
                        values.push_back(encoded);
                    }
                }

                if (values.empty()) {
                    feature_info_[f].num_bins = 1;
                    bin_edges_[f] = {0.0f};
                    for (Index i = 0; i < n_samples_; ++i) {
                        binned_data_.set(i, f, max_bins);
                    }
                } else {
                    // Sort for quantile computation
                    std::sort(values.begin(), values.end());
                    auto last = std::unique(values.begin(), values.end());
                    values.erase(last, values.end());

                    // Compute bin edges
                    BinIndex actual_bins = std::min(max_bins, static_cast<BinIndex>(values.size()));
                    std::vector<Float>& edges = bin_edges_[f];
                    edges.clear();
                    edges.reserve(actual_bins);

                    for (BinIndex b = 0; b < actual_bins; ++b) {
                        size_t idx = (b + 1) * values.size() / (actual_bins + 1);
                        idx = std::min(idx, values.size() - 1);
                        edges.push_back(values[idx]);
                    }

                    // Bin the encoded values
                    for (Index i = 0; i < n_samples_; ++i) {
                        Float val = ordered_cat_encodings_[f][i];

                        if (std::isnan(val)) {
                            binned_data_.set(i, f, max_bins);
                            continue;
                        }

                        // Binary search for bin
                        auto it = std::lower_bound(edges.begin(), edges.end(), val);
                        BinIndex bin = static_cast<BinIndex>(it - edges.begin());
                        binned_data_.set(i, f, bin);
                    }

                    feature_info_[f].num_bins = static_cast<BinIndex>(edges.size() + 1);
                }
            } else {
                // Fallback: each unique categorical value gets a bin
                std::unordered_map<Float, BinIndex> value_to_bin;
                BinIndex next_bin = 0;

                for (Index i = 0; i < n_samples_; ++i) {
                    Float val = raw_data_[i * n_features_ + f];

                    if (std::isnan(val)) {
                        binned_data_.set(i, f, max_bins);  // Special bin for NaN
                        continue;
                    }

                    auto it = value_to_bin.find(val);
                    if (it == value_to_bin.end()) {
                        value_to_bin[val] = next_bin;
                        binned_data_.set(i, f, next_bin);
                        next_bin = std::min(static_cast<BinIndex>(next_bin + 1),
                                           static_cast<BinIndex>(max_bins - 1));
                    } else {
                        binned_data_.set(i, f, it->second);
                    }
                }

                feature_info_[f].num_bins = next_bin;
            }
        } else {
            // For numerical: quantile-based binning
            compute_quantile_bins(f, max_bins);
        }
    }

    // Compute pre-sorted feature orderings for fast histogram building
    orderings_.compute(*this);
}

void Dataset::compute_quantile_bins(FeatureIndex feature, BinIndex n_bins) {
    // Collect non-NaN values
    std::vector<Float> values;
    values.reserve(n_samples_);
    
    for (Index i = 0; i < n_samples_; ++i) {
        Float val = raw_data_[i * n_features_ + feature];
        if (!std::isnan(val)) {
            values.push_back(val);
        }
    }
    
    if (values.empty()) {
        // All NaN
        feature_info_[feature].num_bins = 1;
        bin_edges_[feature] = {0.0f};
        for (Index i = 0; i < n_samples_; ++i) {
            binned_data_.set(i, feature, n_bins);  // NaN bin
        }
        return;
    }
    
    // Sort for quantile computation
    std::sort(values.begin(), values.end());
    
    // Remove duplicates to find unique values
    auto last = std::unique(values.begin(), values.end());
    values.erase(last, values.end());
    
    // Compute bin edges using quantiles
    BinIndex actual_bins = std::min(n_bins, static_cast<BinIndex>(values.size()));
    std::vector<Float>& edges = bin_edges_[feature];
    edges.clear();
    edges.reserve(actual_bins);
    
    for (BinIndex b = 1; b < actual_bins; ++b) {
        size_t idx = static_cast<size_t>(b) * values.size() / actual_bins;
        if (edges.empty() || values[idx] > edges.back()) {
            edges.push_back(values[idx]);
        }
    }
    
    feature_info_[feature].num_bins = static_cast<BinIndex>(edges.size() + 1);
    feature_info_[feature].bin_edges = edges;

    // Bin the data
    for (Index i = 0; i < n_samples_; ++i) {
        Float val = raw_data_[i * n_features_ + feature];

        if (std::isnan(val)) {
            binned_data_.set(i, feature, n_bins);  // NaN bin
        } else {
            BinIndex bin = find_bin(val, edges);
            binned_data_.set(i, feature, bin);
        }
    }
}

BinIndex Dataset::find_bin(Float value, const std::vector<Float>& edges) const {
    // Binary search for the bin
    // Use upper_bound: finds first element > value
    // This ensures values equal to an edge go to the bin BEFORE that edge
    // Example: edges = [1.0], value = 1 → upper_bound returns end() → bin 1
    //          edges = [1.0], value = 0 → upper_bound returns 1.0 → bin 0
    auto it = std::upper_bound(edges.begin(), edges.end(), value);
    return static_cast<BinIndex>(it - edges.begin());
}

void Dataset::apply_bins(const Dataset& reference) {
    bin_edges_ = reference.bin_edges_;
    BinIndex max_bins = 255;

    binned_data_ = BinnedData(n_samples_, n_features_, max_bins);

    // Copy feature info from reference to enable is_categorical checks
    if (feature_info_.empty() && !reference.feature_info_.empty()) {
        feature_info_ = reference.feature_info_;
    }

    #pragma omp parallel for schedule(dynamic)
    for (FeatureIndex f = 0; f < n_features_; ++f) {
        const auto& edges = bin_edges_[f];

        // Check if this is a categorical feature with target statistics
        bool use_target_stats = reference.is_categorical(f) && reference.has_target_stats(f);

        for (Index i = 0; i < n_samples_; ++i) {
            Float val = raw_data_[i * n_features_ + f];

            if (std::isnan(val)) {
                binned_data_.set(i, f, max_bins);
            } else if (use_target_stats) {
                // For categorical features: encode using global target statistics, then bin
                Index category = static_cast<Index>(val);
                Float encoded = reference.encode_category(f, category);
                binned_data_.set(i, f, find_bin(encoded, edges));
            } else {
                binned_data_.set(i, f, find_bin(val, edges));
            }
        }
    }
}

void Dataset::add_feature(const std::vector<Float>& values, FeatureType type) {
    if (values.size() != static_cast<size_t>(n_samples_)) {
        throw std::invalid_argument("Feature values size must match n_samples");
    }

    // Expand raw_data_ to include new feature
    AlignedVector<Float> new_raw_data(n_samples_ * (n_features_ + 1));

    // Copy existing data and add new feature
    for (Index i = 0; i < n_samples_; ++i) {
        // Copy existing features for this row
        for (FeatureIndex f = 0; f < n_features_; ++f) {
            new_raw_data[i * (n_features_ + 1) + f] = raw_data_[i * n_features_ + f];
        }
        // Add new feature value
        new_raw_data[i * (n_features_ + 1) + n_features_] = values[i];
    }

    raw_data_ = std::move(new_raw_data);

    // Add feature info
    FeatureInfo info;
    info.type = type;
    info.num_bins = 0;  // Will be set during binning
    info.has_missing = false;
    info.min_value = std::numeric_limits<Float>::max();
    info.max_value = std::numeric_limits<Float>::lowest();

    // Compute min/max and check for missing values
    for (Index i = 0; i < n_samples_; ++i) {
        Float v = values[i];
        if (std::isnan(v)) {
            info.has_missing = true;
        } else {
            info.min_value = std::min(info.min_value, v);
            info.max_value = std::max(info.max_value, v);
        }
    }

    // For categorical features, compute cardinality
    if (type == FeatureType::Categorical) {
        std::unordered_set<uint32_t> unique_cats;
        for (Index i = 0; i < n_samples_; ++i) {
            if (!std::isnan(values[i])) {
                unique_cats.insert(static_cast<uint32_t>(values[i]));
            }
        }
        info.cardinality = static_cast<uint32_t>(unique_cats.size());

        // Add to categorical features list
        categorical_features_.push_back(n_features_);
    }

    feature_info_.push_back(info);
    n_features_++;

    // Add empty bin edges for new feature (will be computed in compute_bins)
    bin_edges_.push_back({});
}

// ============================================================================
// Gradient Management
// ============================================================================

void Dataset::set_gradients(const AlignedVector<Float>& grads, const AlignedVector<Float>& hess) {
    gradients_ = grads;
    hessians_ = hess;
}

void Dataset::set_gradients(AlignedVector<Float>&& grads, AlignedVector<Float>&& hess) {
    gradients_ = std::move(grads);
    hessians_ = std::move(hess);
}

void Dataset::set_gradients_copy(const Float* grads, const Float* hess, Index n_samples) {
    // Resize only if needed (avoid reallocation on every call)
    if (gradients_.size() != static_cast<size_t>(n_samples)) {
        gradients_.resize(n_samples);
        hessians_.resize(n_samples);
    }

    // Fast copy using memcpy (SIMD optimized by compiler)
    std::memcpy(gradients_.data(), grads, n_samples * sizeof(Float));
    std::memcpy(hessians_.data(), hess, n_samples * sizeof(Float));
}

void Dataset::quantize_gradients() {
    // Find min/max gradient
    Float g_min = *std::min_element(gradients_.begin(), gradients_.end());
    Float g_max = *std::max_element(gradients_.begin(), gradients_.end());
    
    grad_min_ = g_min;
    grad_scale_ = (g_max - g_min) / 255.0f;  // 8-bit quantization for simplicity
    
    if (grad_scale_ < 1e-10f) {
        grad_scale_ = 1.0f;
    }
    
    quantized_grads_.resize(n_samples_);
    
    #pragma omp parallel for
    for (Index i = 0; i < n_samples_; ++i) {
        Float normalized = (gradients_[i] - grad_min_) / grad_scale_;
        quantized_grads_[i] = static_cast<QuantizedGrad>(
            std::max(0.0f, std::min(255.0f, normalized))
        );
    }
}

// ============================================================================
// Sampling
// ============================================================================

Dataset::SubsampleIndices Dataset::goss_subsample(
    Float top_rate, Float other_rate, uint64_t seed
) const {
    SubsampleIndices result;

    Index top_n = static_cast<Index>(n_samples_ * top_rate);
    Index other_n = static_cast<Index>(n_samples_ * other_rate);

    if (top_n == 0 && other_n == 0) {
        return result;
    }

    // Create index array with absolute gradients
    std::vector<std::pair<Float, Index>> grad_idx(n_samples_);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (Index i = 0; i < n_samples_; ++i) {
        grad_idx[i] = {std::abs(gradients_[i]), i};
    }

    // Use nth_element for O(n) partitioning instead of O(n log n) sort
    // This partitions so that top_n largest elements are at the beginning
    if (top_n > 0 && top_n < n_samples_) {
        std::nth_element(grad_idx.begin(), grad_idx.begin() + top_n, grad_idx.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    result.row_indices.reserve(top_n + other_n);

    // Add top gradient samples (they are now in first top_n positions, unordered)
    for (Index i = 0; i < top_n; ++i) {
        result.row_indices.push_back(grad_idx[i].second);
    }

    // Efficient random sampling from remaining using reservoir sampling
    // No need to create separate vector or shuffle all remaining elements
    if (other_n > 0 && top_n < n_samples_) {
        Index remaining_count = n_samples_ - top_n;
        other_n = std::min(other_n, remaining_count);

        std::mt19937_64 rng(seed);

        // Use Fisher-Yates partial shuffle - only shuffle first other_n elements
        for (Index i = 0; i < other_n; ++i) {
            std::uniform_int_distribution<Index> dist(i, remaining_count - 1);
            Index j = dist(rng);
            if (i != j) {
                std::swap(grad_idx[top_n + i], grad_idx[top_n + j]);
            }
            result.row_indices.push_back(grad_idx[top_n + i].second);
        }
    }

    return result;
}

std::vector<Index> Dataset::random_subsample(Float ratio, uint64_t seed) const {
    Index n = static_cast<Index>(n_samples_ * ratio);
    std::vector<Index> indices(n_samples_);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::mt19937_64 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    indices.resize(n);
    return indices;
}

std::vector<FeatureIndex> Dataset::random_feature_subsample(Float ratio, uint64_t seed) const {
    FeatureIndex n = static_cast<FeatureIndex>(n_features_ * ratio);
    std::vector<FeatureIndex> indices(n_features_);
    std::iota(indices.begin(), indices.end(), static_cast<FeatureIndex>(0));

    std::mt19937_64 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    indices.resize(n);
    return indices;
}

Dataset::SubsampleIndices Dataset::mvs_subsample(Float subsample_ratio, uint64_t seed) const {
    // MVS (Minimum Variance Sampling) - CatBoost-style
    // Key insight: sample with probability proportional to gradient magnitude
    // but bounded to maintain low variance in gradient estimation
    //
    // For each sample, compute: score = max(|g|, threshold)
    // Sample with probability: p_i = min(1, s * score / sum(scores))
    // where s is adjusted to achieve target subsample_ratio
    //
    // This is better than GOSS because it doesn't completely drop low-gradient samples

    SubsampleIndices result;

    if (n_samples_ == 0 || subsample_ratio <= 0.0f) {
        return result;
    }

    Index target_n = static_cast<Index>(n_samples_ * subsample_ratio);
    target_n = std::max(target_n, static_cast<Index>(1));

    // Compute absolute gradients
    std::vector<Float> abs_grads(n_samples_);
    Float grad_sum = 0.0f;
    Float grad_max = 0.0f;

    #pragma omp parallel for reduction(+:grad_sum) reduction(max:grad_max)
    for (Index i = 0; i < n_samples_; ++i) {
        abs_grads[i] = std::abs(gradients_[i]);
        grad_sum += abs_grads[i];
        grad_max = std::max(grad_max, abs_grads[i]);
    }

    // Compute threshold at ~subsample_ratio percentile
    // This ensures low-gradient samples have a floor probability
    std::vector<Float> sorted_grads = abs_grads;
    Index threshold_idx = static_cast<Index>((1.0f - subsample_ratio) * n_samples_);
    threshold_idx = std::min(threshold_idx, n_samples_ - 1);

    std::nth_element(sorted_grads.begin(), sorted_grads.begin() + threshold_idx,
                     sorted_grads.end());
    Float threshold = sorted_grads[threshold_idx];

    // Small threshold to avoid division issues
    threshold = std::max(threshold, grad_max * 0.01f);
    if (threshold < 1e-10f) threshold = 1e-10f;

    // Compute sampling scores with floor
    std::vector<Float> scores(n_samples_);
    Float score_sum = 0.0f;

    #pragma omp parallel for reduction(+:score_sum)
    for (Index i = 0; i < n_samples_; ++i) {
        scores[i] = std::max(abs_grads[i], threshold);
        score_sum += scores[i];
    }

    // Compute sampling probabilities
    // Scale factor to achieve target sample count
    Float scale = static_cast<Float>(target_n) / score_sum * score_sum / n_samples_;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<Float> uniform(0.0f, 1.0f);

    result.row_indices.reserve(target_n + target_n / 4);  // Some extra capacity

    for (Index i = 0; i < n_samples_; ++i) {
        Float prob = std::min(1.0f, scale * scores[i] / score_sum * n_samples_);

        if (uniform(rng) < prob) {
            result.row_indices.push_back(i);
        }
    }

    // If we got too few samples, add more randomly
    if (result.row_indices.size() < static_cast<size_t>(target_n * 0.8f)) {
        std::vector<bool> selected(n_samples_, false);
        for (Index idx : result.row_indices) {
            selected[idx] = true;
        }

        std::vector<Index> remaining;
        remaining.reserve(n_samples_ - result.row_indices.size());
        for (Index i = 0; i < n_samples_; ++i) {
            if (!selected[i]) {
                remaining.push_back(i);
            }
        }

        std::shuffle(remaining.begin(), remaining.end(), rng);

        Index need = target_n - static_cast<Index>(result.row_indices.size());
        for (Index i = 0; i < need && i < static_cast<Index>(remaining.size()); ++i) {
            result.row_indices.push_back(remaining[i]);
        }
    }

    return result;
}

// ============================================================================
// Categorical Encoding
// ============================================================================

void Dataset::compute_target_statistics(const Config& config) {
    if (labels_.empty()) return;

    target_stats_.resize(n_features_);

    // Compute prior (global mean)
    Float prior = std::accumulate(labels_.begin(), labels_.end(), 0.0f) / n_samples_;
    Float prior_weight = config.categorical.ts_prior_weight;

    // Choose encoding method
    if (config.categorical.method == CategoricalConfig::EncodingMethod::TargetStatistics) {
        // Ordered target statistics (CatBoost style) - prevents target leakage
        compute_ordered_target_statistics(prior, prior_weight);
    } else if (config.categorical.method == CategoricalConfig::EncodingMethod::CrossValidatedTS) {
        // Cross-validated target statistics - our improvement
        compute_cv_target_statistics(config.categorical.ts_cv_folds, prior, prior_weight);
    } else {
        // Default: global statistics (may have leakage but simpler)
        for (FeatureIndex f : categorical_features_) {
            auto& stats = target_stats_[f];
            stats.clear();

            // Count and sum per category
            std::unordered_map<Index, std::pair<Float, Index>> cat_stats;  // sum, count

            for (Index i = 0; i < n_samples_; ++i) {
                Float val = raw_data_[i * n_features_ + f];
                if (std::isnan(val)) continue;

                Index cat = static_cast<Index>(val);
                auto& s = cat_stats[cat];
                s.first += labels_[i];
                s.second += 1;
            }

            // Compute smoothed target statistics
            for (const auto& [cat, s] : cat_stats) {
                Float smoothed = (s.first + prior_weight * prior) / (s.second + prior_weight);
                stats[cat] = smoothed;
            }
        }
    }
}

void Dataset::compute_ordered_target_statistics(Float prior, Float prior_weight) {
    // CatBoost-style ordered target statistics:
    // For each sample, compute statistics using only samples that appear
    // before it in a random permutation. This prevents target leakage.

    // Create random permutation
    std::vector<Index> permutation(n_samples_);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(permutation.begin(), permutation.end(), gen);

    // Per-sample categorical encodings (stored instead of per-category stats)
    ordered_cat_encodings_.resize(n_features_);

    for (FeatureIndex f : categorical_features_) {
        ordered_cat_encodings_[f].resize(n_samples_);

        // Running statistics per category: (sum, count)
        std::unordered_map<Index, std::pair<Float, Index>> running_stats;

        // Process samples in permutation order
        for (Index perm_idx = 0; perm_idx < n_samples_; ++perm_idx) {
            Index sample_idx = permutation[perm_idx];
            Float val = raw_data_[sample_idx * n_features_ + f];

            if (std::isnan(val)) {
                ordered_cat_encodings_[f][sample_idx] = prior;
                continue;
            }

            Index cat = static_cast<Index>(val);

            // Compute encoding using ONLY samples seen before this one
            auto it = running_stats.find(cat);
            if (it != running_stats.end()) {
                Float sum = it->second.first;
                Index count = it->second.second;
                ordered_cat_encodings_[f][sample_idx] =
                    (sum + prior_weight * prior) / (count + prior_weight);
            } else {
                // First occurrence of this category: use prior
                ordered_cat_encodings_[f][sample_idx] = prior;
            }

            // Update running stats with this sample's label
            auto& s = running_stats[cat];
            s.first += labels_[sample_idx];
            s.second += 1;
        }

        // Also compute global stats for test data
        target_stats_[f].clear();
        for (const auto& [cat, s] : running_stats) {
            Float smoothed = (s.first + prior_weight * prior) / (s.second + prior_weight);
            target_stats_[f][cat] = smoothed;
        }
    }
}

void Dataset::compute_cv_target_statistics(uint8_t n_folds, Float prior, Float prior_weight) {
    // Cross-validated target statistics:
    // Split data into folds, compute statistics for each sample using
    // only samples from other folds. Less aggressive than ordered but
    // still prevents direct leakage.

    // Create fold assignments
    std::vector<uint8_t> fold_assignment(n_samples_);
    std::random_device rd;
    std::mt19937 gen(rd());
    for (Index i = 0; i < n_samples_; ++i) {
        fold_assignment[i] = gen() % n_folds;
    }

    ordered_cat_encodings_.resize(n_features_);

    for (FeatureIndex f : categorical_features_) {
        ordered_cat_encodings_[f].resize(n_samples_);

        // Compute per-fold statistics
        std::vector<std::unordered_map<Index, std::pair<Float, Index>>> fold_stats(n_folds);

        // First pass: compute stats per fold
        for (Index i = 0; i < n_samples_; ++i) {
            Float val = raw_data_[i * n_features_ + f];
            if (std::isnan(val)) continue;

            Index cat = static_cast<Index>(val);
            uint8_t fold = fold_assignment[i];

            auto& s = fold_stats[fold][cat];
            s.first += labels_[i];
            s.second += 1;
        }

        // Compute out-of-fold encoding for each sample
        for (Index i = 0; i < n_samples_; ++i) {
            Float val = raw_data_[i * n_features_ + f];

            if (std::isnan(val)) {
                ordered_cat_encodings_[f][i] = prior;
                continue;
            }

            Index cat = static_cast<Index>(val);
            uint8_t sample_fold = fold_assignment[i];

            // Aggregate stats from all OTHER folds
            Float sum = 0.0f;
            Index count = 0;

            for (uint8_t fold = 0; fold < n_folds; ++fold) {
                if (fold == sample_fold) continue;  // Skip sample's own fold

                auto it = fold_stats[fold].find(cat);
                if (it != fold_stats[fold].end()) {
                    sum += it->second.first;
                    count += it->second.second;
                }
            }

            // Compute smoothed encoding
            ordered_cat_encodings_[f][i] = (sum + prior_weight * prior) / (count + prior_weight);
        }

        // Global stats for test data
        target_stats_[f].clear();
        std::unordered_map<Index, std::pair<Float, Index>> global_stats;

        for (uint8_t fold = 0; fold < n_folds; ++fold) {
            for (const auto& [cat, s] : fold_stats[fold]) {
                auto& g = global_stats[cat];
                g.first += s.first;
                g.second += s.second;
            }
        }

        for (const auto& [cat, s] : global_stats) {
            Float smoothed = (s.first + prior_weight * prior) / (s.second + prior_weight);
            target_stats_[f][cat] = smoothed;
        }
    }
}

Float Dataset::get_categorical_encoding(FeatureIndex feature, Index category) const {
    if (feature >= target_stats_.size()) return 0.0f;
    
    const auto& stats = target_stats_[feature];
    auto it = stats.find(category);
    
    if (it != stats.end()) {
        return it->second;
    }
    
    // Unknown category: return prior
    if (!labels_.empty()) {
        return std::accumulate(labels_.begin(), labels_.end(), 0.0f) / n_samples_;
    }
    return 0.0f;
}

// ============================================================================
// Statistics
// ============================================================================

Dataset::DatasetStats Dataset::compute_stats() const {
    DatasetStats stats;
    stats.n_samples = n_samples_;
    stats.n_features = n_features_;
    stats.n_categorical = static_cast<FeatureIndex>(categorical_features_.size());
    stats.n_numerical = n_features_ - stats.n_categorical;
    
    // Count missing values
    stats.n_missing = 0;
    for (size_t i = 0; i < raw_data_.size(); ++i) {
        if (std::isnan(raw_data_[i])) {
            stats.n_missing++;
        }
    }
    
    // Label statistics
    if (!labels_.empty()) {
        stats.label_mean = std::accumulate(labels_.begin(), labels_.end(), 0.0f) / n_samples_;
        
        Float var = 0.0f;
        for (Float y : labels_) {
            var += (y - stats.label_mean) * (y - stats.label_mean);
        }
        stats.label_std = std::sqrt(var / n_samples_);
        
        // Class distribution (for classification)
        std::unordered_map<int, Index> class_counts;
        for (Float y : labels_) {
            class_counts[static_cast<int>(y)]++;
        }
        
        for (const auto& [cls, cnt] : class_counts) {
            stats.class_distribution.push_back(static_cast<Float>(cnt) / n_samples_);
        }
    }
    
    return stats;
}

} // namespace turbocat
