#pragma once

/**
 * TurboCat: Next-Generation Gradient Boosting Framework
 * 
 * A high-performance gradient boosting library designed to outperform CatBoost.
 * 
 * Key innovations:
 * - GradTree: Gradient-based global tree optimization (not greedy splitting)
 * - Advanced losses: Robust Focal, LDAM, Logit-adjusted, Tsallis
 * - Tsallis entropy splitting criterion
 * - Cross-validated target statistics for categoricals
 * - 3-bit gradient quantization
 * - SIMD-optimized histogram building (AVX2/AVX-512)
 * 
 * Usage:
 * ```cpp
 * #include <turbocat/turbocat.hpp>
 * 
 * turbocat::Config config = turbocat::Config::binary_classification();
 * turbocat::Booster model(config);
 * 
 * turbocat::Dataset train_data;
 * train_data.from_dense(X, n_samples, n_features, y);
 * train_data.compute_bins(config);
 * 
 * model.train(train_data);
 * model.predict_proba(test_data, predictions);
 * ```
 * 
 * For maximum accuracy:
 * ```cpp
 * turbocat::Config config = turbocat::Config::maximum_accuracy();
 * config.tree.use_gradtree = true;  // Enable GradTree
 * config.loss.loss_type = turbocat::LossType::RobustFocal;
 * ```
 * 
 * @author TurboCat Team
 * @version 0.1.0
 */

#define TURBOCAT_VERSION_MAJOR 0
#define TURBOCAT_VERSION_MINOR 2
#define TURBOCAT_VERSION_PATCH 7
#define TURBOCAT_VERSION_STRING "0.2.7-dev1"

#include "turbocat/types.hpp"
#include "turbocat/config.hpp"
#include "turbocat/dataset.hpp"
#include "turbocat/histogram.hpp"
#include "turbocat/tree.hpp"
#include "turbocat/loss.hpp"
#include "turbocat/booster.hpp"

namespace turbocat {

/**
 * Library version information
 */
struct Version {
    static constexpr int major = TURBOCAT_VERSION_MAJOR;
    static constexpr int minor = TURBOCAT_VERSION_MINOR;
    static constexpr int patch = TURBOCAT_VERSION_PATCH;
    static constexpr const char* string = TURBOCAT_VERSION_STRING;
};

/**
 * Get compile-time feature flags
 */
struct CompileFeatures {
    static constexpr bool has_avx2 = 
        #ifdef TURBOCAT_AVX2
            true;
        #else
            false;
        #endif
    
    static constexpr bool has_avx512 = 
        #ifdef TURBOCAT_AVX512
            true;
        #else
            false;
        #endif
    
    static constexpr bool has_openmp = 
        #ifdef _OPENMP
            true;
        #else
            false;
        #endif
    
    static constexpr bool has_cuda = 
        #ifdef TURBOCAT_CUDA
            true;
        #else
            false;
        #endif
    
    static constexpr bool has_metal = 
        #ifdef TURBOCAT_METAL
            true;
        #else
            false;
        #endif
};

/**
 * Print library info
 */
inline void print_info() {
    std::printf("TurboCat v%s\n", Version::string);
    std::printf("  SIMD: %s\n", 
        CompileFeatures::has_avx512 ? "AVX-512" : 
        CompileFeatures::has_avx2 ? "AVX2" : "SSE");
    std::printf("  OpenMP: %s\n", CompileFeatures::has_openmp ? "Yes" : "No");
    std::printf("  CUDA: %s\n", CompileFeatures::has_cuda ? "Yes" : "No");
    std::printf("  Metal: %s\n", CompileFeatures::has_metal ? "Yes" : "No");
}

} // namespace turbocat
