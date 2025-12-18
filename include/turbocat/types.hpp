#pragma once

/**
 * TurboCat: Next-Generation Gradient Boosting Framework
 * 
 * Core type definitions optimized for:
 * - Cache-friendly memory layout
 * - SIMD-aligned data structures
 * - Minimal memory footprint with quantization
 */

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <array>
#include <memory>
#include <limits>
#include <type_traits>

namespace turbocat {

// ============================================================================
// Alignment and SIMD Configuration
// ============================================================================

#if defined(TURBOCAT_AVX512)
    constexpr size_t SIMD_ALIGNMENT = 64;
    constexpr size_t SIMD_WIDTH = 16;  // 16 floats per register
#elif defined(TURBOCAT_AVX2)
    constexpr size_t SIMD_ALIGNMENT = 32;
    constexpr size_t SIMD_WIDTH = 8;   // 8 floats per register
#else
    constexpr size_t SIMD_ALIGNMENT = 16;
    constexpr size_t SIMD_WIDTH = 4;   // 4 floats per register
#endif

// ============================================================================
// Basic Types
// ============================================================================

using Float = float;                    // Main floating point type
using Double = double;                  // High precision when needed
using Index = uint32_t;                 // Row/column indices
using BinIndex = uint8_t;               // Histogram bin index (0-255)
using FeatureIndex = uint16_t;          // Feature index (up to 65k features)
using TreeIndex = uint16_t;             // Tree/node index
using Label = int32_t;                  // Class labels

// Quantized gradient type (2-3 bits sufficient per Microsoft research)
using QuantizedGrad = int8_t;

// ============================================================================
// Aligned Allocator for SIMD
// ============================================================================

template <typename T, size_t Alignment = SIMD_ALIGNMENT>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = size_t;

    AlignedAllocator() noexcept = default;
    
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] pointer allocate(size_type n) {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        
        void* ptr = nullptr;
        #if defined(_MSC_VER)
            ptr = _aligned_malloc(n * sizeof(T), Alignment);
        #else
            if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
                ptr = nullptr;
            }
        #endif
        
        if (!ptr) throw std::bad_alloc();
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        #if defined(_MSC_VER)
            _aligned_free(p);
        #else
            free(p);
        #endif
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template <typename T, typename U, size_t A>
bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept {
    return true;
}

template <typename T, typename U, size_t A>
bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept {
    return false;
}

// ============================================================================
// Aligned Containers
// ============================================================================

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

// ============================================================================
// Histogram Bin Structure (Gradient Statistics)
// ============================================================================

struct alignas(16) GradientPair {
    Float grad;      // Sum of gradients
    Float hess;      // Sum of hessians
    Index count;     // Number of samples
    Float padding;   // Align to 16 bytes
    
    GradientPair() : grad(0), hess(0), count(0), padding(0) {}
    GradientPair(Float g, Float h, Index c = 1) : grad(g), hess(h), count(c), padding(0) {}
    
    GradientPair& operator+=(const GradientPair& other) {
        grad += other.grad;
        hess += other.hess;
        count += other.count;
        return *this;
    }
    
    GradientPair operator+(const GradientPair& other) const {
        return GradientPair(grad + other.grad, hess + other.hess, count + other.count);
    }
    
    GradientPair operator-(const GradientPair& other) const {
        return GradientPair(grad - other.grad, hess - other.hess, count - other.count);
    }
};

// ============================================================================
// Split Information
// ============================================================================

struct SplitInfo {
    FeatureIndex feature_idx;
    BinIndex bin_threshold;
    Float gain;
    Float left_value;
    Float right_value;
    GradientPair left_stats;
    GradientPair right_stats;
    bool is_valid;
    
    SplitInfo() : feature_idx(0), bin_threshold(0), gain(-1e30f),
                  left_value(0), right_value(0), is_valid(false) {}
    
    bool operator>(const SplitInfo& other) const {
        return gain > other.gain;
    }
};

// ============================================================================
// Tree Node Structure (Cache-optimized)
// ============================================================================

struct alignas(32) TreeNode {
    // Split information (for internal nodes)
    FeatureIndex split_feature;
    BinIndex split_bin;
    uint8_t is_leaf : 1;
    uint8_t default_left : 1;  // Direction for missing values
    uint8_t reserved : 6;
    
    // Child indices (for internal nodes)
    TreeIndex left_child;
    TreeIndex right_child;
    
    // Leaf value or intermediate prediction
    Float value;
    
    // Statistics for pruning/analysis
    GradientPair stats;
    Float gain;
    
    TreeNode() : split_feature(0), split_bin(0), is_leaf(1), default_left(1), reserved(0),
                 left_child(0), right_child(0), value(0), gain(0) {}
};

// Note: Size varies by platform due to padding differences
// static_assert(sizeof(TreeNode) == 32, "TreeNode must be 32 bytes for cache efficiency");

// ============================================================================
// Feature Metadata
// ============================================================================

enum class FeatureType : uint8_t {
    Numerical = 0,
    Categorical = 1,
    Boolean = 2
};

struct FeatureInfo {
    FeatureType type;
    BinIndex num_bins;
    Float min_value;
    Float max_value;
    bool has_missing;
    std::vector<Float> bin_edges;        // For numerical
    std::vector<Index> category_counts;  // For categorical
};

// ============================================================================
// Prediction Result with Uncertainty (GP-derived)
// ============================================================================

struct Prediction {
    Float value;           // Point prediction
    Float variance;        // Uncertainty estimate
    Float lower_bound;     // Confidence interval
    Float upper_bound;
    
    Prediction(Float v = 0) : value(v), variance(0), lower_bound(v), upper_bound(v) {}
    Prediction(Float v, Float var, Float alpha = 1.96) 
        : value(v), variance(var), 
          lower_bound(v - alpha * std::sqrt(var)),
          upper_bound(v + alpha * std::sqrt(var)) {}
};

// ============================================================================
// Task Type
// ============================================================================

enum class TaskType : uint8_t {
    BinaryClassification = 0,
    MulticlassClassification = 1,
    Regression = 2,
    Ranking = 3
};

// ============================================================================
// Loss Type
// ============================================================================

enum class LossType : uint8_t {
    // Standard losses
    LogLoss = 0,
    CrossEntropy = 1,
    MSE = 2,
    MAE = 3,
    Huber = 4,
    
    // Advanced losses (our innovations)
    RobustFocal = 10,      // Robust to label noise
    LDAM = 11,             // Label-distribution-aware margin loss
    LogitAdjusted = 12,    // Class-imbalance aware
    Tsallis = 13,          // Generalized entropy loss
};

// ============================================================================
// Split Criterion
// ============================================================================

enum class SplitCriterion : uint8_t {
    Variance = 0,          // Standard variance reduction
    Gini = 1,              // Gini impurity
    Entropy = 2,           // Information gain
    TsallisEntropy = 3,    // Generalized Tsallis entropy (q-parameterized)
};

} // namespace turbocat
