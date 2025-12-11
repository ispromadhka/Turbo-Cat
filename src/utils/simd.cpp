/**
 * TurboCat SIMD Utilities
 * 
 * CPU feature detection and SIMD dispatch.
 */

#include "turbocat/types.hpp"
#include <cstdio>

#ifdef TURBOCAT_AVX2
#include <immintrin.h>
#endif

#ifdef TURBOCAT_AVX512
#include <immintrin.h>
#endif

namespace turbocat {
namespace simd {

// ============================================================================
// CPU Feature Detection
// ============================================================================

struct CPUFeatures {
    bool has_sse = false;
    bool has_sse2 = false;
    bool has_sse4_1 = false;
    bool has_avx = false;
    bool has_avx2 = false;
    bool has_avx512f = false;
    bool has_fma = false;
    
    static CPUFeatures detect() {
        CPUFeatures f;
        
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        
        #ifdef __GNUC__
        unsigned int eax, ebx, ecx, edx;
        
        // Check basic features (CPUID leaf 1)
        __asm__ __volatile__(
            "cpuid"
            : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
            : "a"(1)
        );
        
        f.has_sse = (edx >> 25) & 1;
        f.has_sse2 = (edx >> 26) & 1;
        f.has_sse4_1 = (ecx >> 19) & 1;
        f.has_avx = (ecx >> 28) & 1;
        f.has_fma = (ecx >> 12) & 1;
        
        // Check extended features (CPUID leaf 7)
        __asm__ __volatile__(
            "cpuid"
            : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
            : "a"(7), "c"(0)
        );
        
        f.has_avx2 = (ebx >> 5) & 1;
        f.has_avx512f = (ebx >> 16) & 1;
        
        #endif
        
        #endif
        
        return f;
    }
};

static CPUFeatures cpu_features = CPUFeatures::detect();

bool has_avx2() {
    return cpu_features.has_avx2;
}

bool has_avx512() {
    return cpu_features.has_avx512f;
}

bool has_fma() {
    return cpu_features.has_fma;
}

void print_cpu_features() {
    std::printf("CPU Features:\n");
    std::printf("  SSE:     %s\n", cpu_features.has_sse ? "Yes" : "No");
    std::printf("  SSE2:    %s\n", cpu_features.has_sse2 ? "Yes" : "No");
    std::printf("  SSE4.1:  %s\n", cpu_features.has_sse4_1 ? "Yes" : "No");
    std::printf("  AVX:     %s\n", cpu_features.has_avx ? "Yes" : "No");
    std::printf("  AVX2:    %s\n", cpu_features.has_avx2 ? "Yes" : "No");
    std::printf("  AVX-512: %s\n", cpu_features.has_avx512f ? "Yes" : "No");
    std::printf("  FMA:     %s\n", cpu_features.has_fma ? "Yes" : "No");
}

// ============================================================================
// Vectorized Operations
// ============================================================================

#ifdef TURBOCAT_AVX2

void add_vectors_avx2(float* dst, const float* src, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_load_ps(dst + i);
        __m256 b = _mm256_load_ps(src + i);
        __m256 c = _mm256_add_ps(a, b);
        _mm256_store_ps(dst + i, c);
    }
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
}

float dot_product_avx2(const float* a, const float* b, size_t n) {
    __m256 sum = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    
    float result = _mm_cvtss_f32(sum128);
    
    // Handle remainder
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

#endif

#ifdef TURBOCAT_AVX512

void add_vectors_avx512(float* dst, const float* src, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 a = _mm512_load_ps(dst + i);
        __m512 b = _mm512_load_ps(src + i);
        __m512 c = _mm512_add_ps(a, b);
        _mm512_store_ps(dst + i, c);
    }
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
}

float dot_product_avx512(const float* a, const float* b, size_t n) {
    __m512 sum = _mm512_setzero_ps();
    
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_load_ps(a + i);
        __m512 vb = _mm512_load_ps(b + i);
        sum = _mm512_fmadd_ps(va, vb, sum);
    }
    
    float result = _mm512_reduce_add_ps(sum);
    
    // Handle remainder
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

#endif

// Dispatch to appropriate implementation
void add_vectors(float* dst, const float* src, size_t n) {
    #ifdef TURBOCAT_AVX512
    if (cpu_features.has_avx512f) {
        add_vectors_avx512(dst, src, n);
        return;
    }
    #endif
    
    #ifdef TURBOCAT_AVX2
    if (cpu_features.has_avx2) {
        add_vectors_avx2(dst, src, n);
        return;
    }
    #endif
    
    // Scalar fallback
    for (size_t i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

float dot_product(const float* a, const float* b, size_t n) {
    #ifdef TURBOCAT_AVX512
    if (cpu_features.has_avx512f) {
        return dot_product_avx512(a, b, n);
    }
    #endif
    
    #ifdef TURBOCAT_AVX2
    if (cpu_features.has_avx2) {
        return dot_product_avx2(a, b, n);
    }
    #endif
    
    // Scalar fallback
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

} // namespace simd
} // namespace turbocat
