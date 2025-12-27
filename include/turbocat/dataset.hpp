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
    
    // ========================================================================
    // Gradient/Hessian Management
    // ========================================================================
    
    void set_gradients(const AlignedVector<Float>& grads, const AlignedVector<Float>& hess);
    void set_gradients(AlignedVector<Float>&& grads, AlignedVector<Float>&& hess);
    
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
    
    // Raw unbinned data (for GradTree optimization)
    const Float* raw_data() const { return raw_data_.data(); }
    Float raw_value(Index row, FeatureIndex feature) const {
        return raw_data_[row * n_features_ + feature];
    }
    
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
     * Get encoded value for categorical feature
     */
    Float get_categorical_encoding(FeatureIndex feature, Index category) const;
    
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
    
    // Bin edges for numerical features
    std::vector<std::vector<Float>> bin_edges_;
    
    // Missing value indicators
    AlignedVector<uint8_t> missing_mask_;  // Bit-packed
    
    // Helper methods
    void detect_feature_types();
    void compute_quantile_bins(FeatureIndex feature, BinIndex n_bins);
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
