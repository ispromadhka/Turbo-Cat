#pragma once

/**
 * TurboCat Dataset
 * 
 * Efficient columnar data storage with:
 * - Histogram binning (255 bins)
 * - Categorical encoding
 * - SIMD-aligned memory layout
 * - Row/column subsampling support
 */

#include "types.hpp"
#include "config.hpp"
#include "feature_orderings.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

namespace turbocat {

// ============================================================================
// Binned Data Storage (Column-major, cache-efficient)
// ============================================================================

class BinnedData {
public:
    BinnedData() = default;
    BinnedData(Index n_rows, FeatureIndex n_features, BinIndex max_bins = 255);

    // Access binned values (column-major)
    BinIndex get(Index row, FeatureIndex feature) const {
        return data_[feature * n_rows_ + row];
    }

    // Access from row-major layout (for prediction)
    BinIndex get_row_major(Index row, FeatureIndex feature) const {
        return row_major_data_[row * n_features_ + feature];
    }

    void set(Index row, FeatureIndex feature, BinIndex bin) {
        data_[feature * n_rows_ + row] = bin;
    }

    // Column access for histogram building
    const BinIndex* column(FeatureIndex feature) const {
        return data_.data() + feature * n_rows_;
    }

    BinIndex* column(FeatureIndex feature) {
        return data_.data() + feature * n_rows_;
    }

    // Row access for prediction (returns pointer to row in row-major layout)
    const BinIndex* row(Index row_idx) const {
        return row_major_data_.data() + row_idx * n_features_;
    }

    // Prepare row-major layout for prediction
    void prepare_for_prediction() const;
    bool has_row_major() const { return !row_major_data_.empty(); }

    // Prepare column-major layout for SIMD prediction (same as internal storage)
    // This is a no-op since data is already column-major, but ensures it's ready
    void prepare_column_major() const { /* Data is already column-major */ }
    bool has_column_major() const { return !data_.empty(); }

    // Access from column-major layout (same as get(), for SIMD prediction)
    BinIndex get_column_major(Index row, FeatureIndex feature) const {
        return data_[feature * n_rows_ + row];
    }

    Index n_rows() const { return n_rows_; }
    FeatureIndex n_features() const { return n_features_; }

private:
    AlignedVector<BinIndex> data_;  // Column-major for training
    mutable AlignedVector<BinIndex> row_major_data_;  // Row-major for prediction
    Index n_rows_ = 0;
    FeatureIndex n_features_ = 0;
};

// ============================================================================
// Dataset Class
// ============================================================================

class Dataset {
public:
    Dataset() = default;
    
    // ========================================================================
    // Construction from raw data
    // ========================================================================
    
    /**
     * Create dataset from dense matrix (row-major)
     * @param data Pointer to row-major data [n_samples × n_features]
     * @param n_samples Number of samples
     * @param n_features Number of features
     * @param labels Optional labels (nullptr if not provided)
     * @param weights Optional sample weights (nullptr for uniform)
     * @param cat_features Indices of categorical features
     */
    void from_dense(
        const Float* data,
        Index n_samples,
        FeatureIndex n_features,
        const Float* labels = nullptr,
        const Float* weights = nullptr,
        const std::vector<FeatureIndex>& cat_features = {}
    );
    
    /**
     * Create dataset from CSR sparse matrix
     */
    void from_sparse_csr(
        const Float* data,
        const Index* indices,
        const Index* indptr,
        Index n_samples,
        FeatureIndex n_features,
        const Float* labels = nullptr
    );
    
    // ========================================================================
    // Binning
    // ========================================================================
    
    /**
     * Compute histogram bins for numerical features
     * Uses quantile-based binning for better split finding
     */
    void compute_bins(const Config& config);
    
    /**
     * Apply precomputed bins from another dataset (for test data)
     */
    void apply_bins(const Dataset& reference);

    /**
     * Add a new feature to the dataset (for interaction features)
     * @param values Feature values for each sample
     * @param type Feature type (Numerical or Categorical)
     */
    void add_feature(const std::vector<Float>& values, FeatureType type = FeatureType::Numerical);

    // ========================================================================
    // Gradient/Hessian Management
    // ========================================================================
    
    void set_gradients(const AlignedVector<Float>& grads, const AlignedVector<Float>& hess);
    void set_gradients(AlignedVector<Float>&& grads, AlignedVector<Float>&& hess);

    // Fast copy from raw pointers (avoids reallocating internal vectors)
    void set_gradients_copy(const Float* grads, const Float* hess, Index n_samples);

    // Ensure gradient arrays are allocated (for direct write)
    void ensure_gradients_allocated(Index n_samples) {
        if (gradients_.size() != static_cast<size_t>(n_samples)) {
            gradients_.resize(n_samples);
            hessians_.resize(n_samples);
        }
    }

    const Float* gradients() const { return gradients_.data(); }
    const Float* hessians() const { return hessians_.data(); }
    Float* gradients() { return gradients_.data(); }
    Float* hessians() { return hessians_.data(); }
    
    // Quantized gradients (3-bit)
    void quantize_gradients();
    const QuantizedGrad* quantized_gradients() const { return quantized_grads_.data(); }
    Float gradient_scale() const { return grad_scale_; }
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    Index n_samples() const { return n_samples_; }
    FeatureIndex n_features() const { return n_features_; }

    // Setters for fast prediction path (when binned data is set directly)
    void set_n_samples(Index n) { n_samples_ = n; }
    void set_n_features(FeatureIndex n) { n_features_ = n; }

    const BinnedData& binned() const { return binned_data_; }
    BinnedData& binned() { return binned_data_; }
    
    const AlignedVector<Float>& labels() const { return labels_; }
    const AlignedVector<Float>& weights() const { return weights_; }
    Float label(Index i) const { return labels_[i]; }
    Float weight(Index i) const { return weights_.empty() ? 1.0f : weights_[i]; }
    
    const std::vector<FeatureInfo>& feature_info() const { return feature_info_; }
    const FeatureInfo& feature_info(FeatureIndex f) const { return feature_info_[f]; }
    
    bool is_categorical(FeatureIndex f) const {
        return feature_info_[f].type == FeatureType::Categorical;
    }

    // Categorical features list
    const std::vector<FeatureIndex>& categorical_features() const {
        return categorical_features_;
    }

    // Global target statistics for categorical encoding (used for test data)
    const std::vector<std::unordered_map<Index, Float>>& target_stats() const {
        return target_stats_;
    }

    // Check if target stats have been computed for a feature
    bool has_target_stats(FeatureIndex f) const {
        return f < target_stats_.size() && !target_stats_[f].empty();
    }

    // Get encoded value for a category using global target statistics
    Float encode_category(FeatureIndex f, Index category, Float prior = 0.5f) const {
        if (f < target_stats_.size()) {
            auto it = target_stats_[f].find(category);
            if (it != target_stats_[f].end()) {
                return it->second;
            }
        }
        return prior;  // Unknown category falls back to prior
    }

    // Raw unbinned data (for GradTree optimization)
    const Float* raw_data() const { return raw_data_.data(); }
    Float raw_value(Index row, FeatureIndex feature) const {
        return raw_data_[row * n_features_ + feature];
    }

    // Bin edges (for fast prediction without rebinning)
    const std::vector<std::vector<Float>>& bin_edges() const { return bin_edges_; }

    // Pre-sorted feature orderings (for fast histogram building)
    const FeatureOrderings& orderings() const { return orderings_; }
    bool has_orderings() const { return orderings_.is_computed(); }
    
    // ========================================================================
    // Subsampling Support
    // ========================================================================
    
    struct SubsampleIndices {
        std::vector<Index> row_indices;
        std::vector<FeatureIndex> col_indices;
    };
    
    /**
     * Create row subsample using GOSS
     * @param top_rate Keep top_rate% by gradient magnitude
     * @param other_rate Sample other_rate% from remainder
     */
    SubsampleIndices goss_subsample(Float top_rate, Float other_rate, uint64_t seed) const;
    
    /**
     * Simple random row subsample
     */
    std::vector<Index> random_subsample(Float ratio, uint64_t seed) const;

    /**
     * MVS (Minimum Variance Sampling) - CatBoost-style
     * Better than GOSS for large datasets, maintains gradient variance
     * @param subsample_ratio Fraction of samples to keep
     * @param seed Random seed
     * @return Subsample indices with adjusted weights
     */
    SubsampleIndices mvs_subsample(Float subsample_ratio, uint64_t seed) const;
    
    /**
     * Random column subsample
     */
    std::vector<FeatureIndex> random_feature_subsample(Float ratio, uint64_t seed) const;
    
    // ========================================================================
    // Categorical Encoding
    // ========================================================================
    
    /**
     * Compute target statistics for categorical features
     */
    void compute_target_statistics(const Config& config);
    
    /**
     * Get encoded value for categorical feature (for test data)
     */
    Float get_categorical_encoding(FeatureIndex feature, Index category) const;

    /**
     * Get per-sample encoding for categorical feature (for training with ordered/CV stats)
     */
    Float get_sample_categorical_encoding(FeatureIndex feature, Index sample_idx) const {
        if (feature < ordered_cat_encodings_.size() &&
            sample_idx < ordered_cat_encodings_[feature].size()) {
            return ordered_cat_encodings_[feature][sample_idx];
        }
        return 0.0f;
    }

    /**
     * Check if per-sample encodings are available
     */
    bool has_ordered_encodings() const {
        return !ordered_cat_encodings_.empty() &&
               !ordered_cat_encodings_[0].empty();
    }
    
    // ========================================================================
    // Statistics
    // ========================================================================
    
    struct DatasetStats {
        Index n_samples;
        FeatureIndex n_features;
        FeatureIndex n_categorical;
        FeatureIndex n_numerical;
        Index n_missing;
        Float label_mean;
        Float label_std;
        std::vector<Float> class_distribution;  // For classification
    };
    
    DatasetStats compute_stats() const;
    
private:
    Index n_samples_ = 0;
    FeatureIndex n_features_ = 0;
    
    // Raw data (row-major)
    AlignedVector<Float> raw_data_;
    
    // Binned data (column-major for histogram efficiency)
    BinnedData binned_data_;
    
    // Labels and weights
    AlignedVector<Float> labels_;
    AlignedVector<Float> weights_;
    
    // Gradients and hessians
    AlignedVector<Float> gradients_;
    AlignedVector<Float> hessians_;
    
    // Quantized gradients
    AlignedVector<QuantizedGrad> quantized_grads_;
    Float grad_scale_ = 1.0f;
    Float grad_min_ = 0.0f;
    
    // Feature metadata
    std::vector<FeatureInfo> feature_info_;
    std::vector<FeatureIndex> categorical_features_;
    
    // Target statistics for categorical encoding
    // Map: feature_idx -> (category -> encoded_value)
    std::vector<std::unordered_map<Index, Float>> target_stats_;

    // Per-sample categorical encodings (for ordered/CV target statistics)
    // Map: feature_idx -> (sample_idx -> encoded_value)
    std::vector<std::vector<Float>> ordered_cat_encodings_;

    // Bin edges for numerical features
    std::vector<std::vector<Float>> bin_edges_;

    // Missing value indicators
    AlignedVector<uint8_t> missing_mask_;  // Bit-packed

    // Pre-sorted feature orderings for fast histogram building
    FeatureOrderings orderings_;

    // Helper methods
    void detect_feature_types();
    void compute_quantile_bins(FeatureIndex feature, BinIndex n_bins);

    // Categorical encoding methods
    void compute_ordered_target_statistics(Float prior, Float prior_weight);
    void compute_cv_target_statistics(uint8_t n_folds, Float prior, Float prior_weight);
    BinIndex find_bin(Float value, const std::vector<Float>& edges) const;
};

// ============================================================================
// Dataset Builder (fluent API)
// ============================================================================

class DatasetBuilder {
public:
    DatasetBuilder& data(const Float* ptr, Index rows, FeatureIndex cols) {
        data_ptr_ = ptr;
        n_rows_ = rows;
        n_cols_ = cols;
        return *this;
    }
    
    DatasetBuilder& labels(const Float* ptr) {
        labels_ptr_ = ptr;
        return *this;
    }
    
    DatasetBuilder& weights(const Float* ptr) {
        weights_ptr_ = ptr;
        return *this;
    }
    
    DatasetBuilder& categorical_features(std::vector<FeatureIndex> indices) {
        cat_features_ = std::move(indices);
        return *this;
    }
    
    DatasetBuilder& feature_names(std::vector<std::string> names) {
        feature_names_ = std::move(names);
        return *this;
    }
    
    std::unique_ptr<Dataset> build(const Config& config) {
        auto dataset = std::make_unique<Dataset>();
        dataset->from_dense(data_ptr_, n_rows_, n_cols_, labels_ptr_, weights_ptr_, cat_features_);
        dataset->compute_bins(config);
        return dataset;
    }
    
private:
    const Float* data_ptr_ = nullptr;
    const Float* labels_ptr_ = nullptr;
    const Float* weights_ptr_ = nullptr;
    Index n_rows_ = 0;
    FeatureIndex n_cols_ = 0;
    std::vector<FeatureIndex> cat_features_;
    std::vector<std::string> feature_names_;
};

} // namespace turbocat
