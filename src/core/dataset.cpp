/**
 * TurboCat Dataset Implementation
 */

#include "turbocat/dataset.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <cstring>

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
            // For categorical: each unique value gets a bin
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
        } else {
            // For numerical: quantile-based binning
            compute_quantile_bins(f, max_bins);
        }
    }
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
            binned_data_.set(i, feature, find_bin(val, edges));
        }
    }
}

BinIndex Dataset::find_bin(Float value, const std::vector<Float>& edges) const {
    // Binary search for the bin
    auto it = std::lower_bound(edges.begin(), edges.end(), value);
    return static_cast<BinIndex>(it - edges.begin());
}

void Dataset::apply_bins(const Dataset& reference) {
    bin_edges_ = reference.bin_edges_;
    BinIndex max_bins = 255;
    
    binned_data_ = BinnedData(n_samples_, n_features_, max_bins);
    
    #pragma omp parallel for schedule(dynamic)
    for (FeatureIndex f = 0; f < n_features_; ++f) {
        const auto& edges = bin_edges_[f];
        
        for (Index i = 0; i < n_samples_; ++i) {
            Float val = raw_data_[i * n_features_ + f];
            
            if (std::isnan(val)) {
                binned_data_.set(i, f, max_bins);
            } else {
                binned_data_.set(i, f, find_bin(val, edges));
            }
        }
    }
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
    
    #pragma omp parallel for simd
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
    
    // Sort by absolute gradient
    std::vector<std::pair<Float, Index>> grad_idx(n_samples_);
    for (Index i = 0; i < n_samples_; ++i) {
        grad_idx[i] = {std::abs(gradients_[i]), i};
    }
    
    std::sort(grad_idx.begin(), grad_idx.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Select top samples
    Index top_n = static_cast<Index>(n_samples_ * top_rate);
    Index other_n = static_cast<Index>(n_samples_ * other_rate);
    
    result.row_indices.reserve(top_n + other_n);
    
    // Add top gradient samples
    for (Index i = 0; i < top_n; ++i) {
        result.row_indices.push_back(grad_idx[i].second);
    }
    
    // Random sample from remaining
    std::mt19937_64 rng(seed);
    std::vector<Index> remaining;
    remaining.reserve(n_samples_ - top_n);
    
    for (Index i = top_n; i < n_samples_; ++i) {
        remaining.push_back(grad_idx[i].second);
    }
    
    std::shuffle(remaining.begin(), remaining.end(), rng);
    
    for (Index i = 0; i < std::min(other_n, static_cast<Index>(remaining.size())); ++i) {
        result.row_indices.push_back(remaining[i]);
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

// ============================================================================
// Categorical Encoding
// ============================================================================

void Dataset::compute_target_statistics(const Config& config) {
    if (labels_.empty()) return;
    
    target_stats_.resize(n_features_);
    
    // Compute prior (global mean)
    Float prior = std::accumulate(labels_.begin(), labels_.end(), 0.0f) / n_samples_;
    Float prior_weight = config.categorical.ts_prior_weight;
    
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
