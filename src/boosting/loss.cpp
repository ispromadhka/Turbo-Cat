/**
 * TurboCat Loss Functions Implementation
 * 
 * Advanced loss functions for robust gradient boosting.
 */

#include "turbocat/loss.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD includes
#if defined(TURBOCAT_AVX2) || defined(TURBOCAT_AVX512)
#include <immintrin.h>
#define HAS_AVX2 1
#elif defined(TURBOCAT_NEON) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAS_NEON 1
#endif

namespace turbocat {

// ============================================================================
// SIMD Sigmoid using accurate exp approximation
// Uses Schraudolph's exp approximation with improved accuracy
// ============================================================================

#ifdef HAS_AVX2
// Accurate exp approximation for AVX2
inline __m256 fast_exp_avx2(__m256 x) {
    // Clamp to avoid overflow/underflow
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    // exp(x) = 2^(x * log2(e)) = 2^(x * 1.4426950408889634)
    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 c1 = _mm256_set1_ps(0.693359375f);
    const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);

    __m256 t = _mm256_mul_ps(x, log2e);
    __m256 ti = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 tf = _mm256_sub_ps(t, ti);

    // Polynomial approximation for 2^tf where tf in [-0.5, 0.5]
    const __m256 p0 = _mm256_set1_ps(1.0f);
    const __m256 p1 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 p2 = _mm256_set1_ps(0.2402265069591007f);
    const __m256 p3 = _mm256_set1_ps(0.05550410866482046f);
    const __m256 p4 = _mm256_set1_ps(0.009618129107628477f);
    const __m256 p5 = _mm256_set1_ps(0.0013333558146428443f);

    __m256 poly = _mm256_fmadd_ps(p5, tf, p4);
    poly = _mm256_fmadd_ps(poly, tf, p3);
    poly = _mm256_fmadd_ps(poly, tf, p2);
    poly = _mm256_fmadd_ps(poly, tf, p1);
    poly = _mm256_fmadd_ps(poly, tf, p0);

    // Multiply by 2^ti using integer manipulation
    __m256i ti_int = _mm256_cvtps_epi32(ti);
    ti_int = _mm256_add_epi32(ti_int, _mm256_set1_epi32(127));
    ti_int = _mm256_slli_epi32(ti_int, 23);
    __m256 pow2 = _mm256_castsi256_ps(ti_int);

    return _mm256_mul_ps(poly, pow2);
}

inline __m256 fast_sigmoid_avx2(__m256 x) {
    // Clip to [-15, 15] to prevent overflow
    x = _mm256_max_ps(x, _mm256_set1_ps(-15.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(15.0f));

    // sigmoid(x) = 1 / (1 + exp(-x))
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg_x = fast_exp_avx2(neg_x);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 denom = _mm256_add_ps(one, exp_neg_x);
    return _mm256_div_ps(one, denom);
}
#endif

#ifdef HAS_NEON
// Standard accurate sigmoid for NEON (using scalar exp for accuracy)
inline float32x4_t fast_sigmoid_neon(float32x4_t x) {
    // Clip to [-15, 15]
    x = vmaxq_f32(x, vdupq_n_f32(-15.0f));
    x = vminq_f32(x, vdupq_n_f32(15.0f));

    // Use scalar exp for accuracy - NEON doesn't have exp intrinsic
    alignas(16) float vals[4];
    vst1q_f32(vals, x);

    alignas(16) float results[4];
    for (int i = 0; i < 4; ++i) {
        float exp_neg = std::exp(-vals[i]);
        results[i] = 1.0f / (1.0f + exp_neg);
    }

    return vld1q_f32(results);
}
#endif

// ============================================================================
// Log Loss (Binary Classification)
// ============================================================================

Float LogLoss::compute_loss(const Float* labels, const Float* predictions, Index n) const {
    Float loss = 0.0f;
    
    #pragma omp parallel for reduction(+:loss)
    for (Index i = 0; i < n; ++i) {
        Float p = clip(sigmoid(predictions[i]));
        Float y = labels[i];
        loss += -y * std::log(p) - (1 - y) * std::log(1 - p);
    }
    
    return loss / n;
}

void LogLoss::compute_gradients(
    const Float* labels,
    const Float* predictions,
    Float* gradients,
    Float* hessians,
    Index n
) const {
#ifdef HAS_AVX2
    // AVX2 vectorized version - process 8 samples at a time
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 min_hess = _mm256_set1_ps(1e-6f);

    Index n_vec = (n / 8) * 8;

    #pragma omp parallel for
    for (Index i = 0; i < n_vec; i += 8) {
        __m256 preds = _mm256_loadu_ps(predictions + i);
        __m256 y = _mm256_loadu_ps(labels + i);

        // Fast sigmoid
        __m256 p = fast_sigmoid_avx2(preds);

        // Gradient: p - y
        __m256 grad = _mm256_sub_ps(p, y);
        _mm256_storeu_ps(gradients + i, grad);

        // Hessian: max(p * (1 - p), 1e-6)
        __m256 one_minus_p = _mm256_sub_ps(one, p);
        __m256 hess = _mm256_mul_ps(p, one_minus_p);
        hess = _mm256_max_ps(hess, min_hess);
        _mm256_storeu_ps(hessians + i, hess);
    }

    // Handle remaining elements
    for (Index i = n_vec; i < n; ++i) {
        Float pred = std::max(-15.0f, std::min(15.0f, predictions[i]));
        Float p = sigmoid(pred);
        Float y = labels[i];
        gradients[i] = p - y;
        hessians[i] = std::max(p * (1 - p), 1e-6f);
    }

#elif defined(HAS_NEON)
    // NEON vectorized version - process 4 samples at a time
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t min_hess = vdupq_n_f32(1e-6f);

    Index n_vec = (n / 4) * 4;

    #pragma omp parallel for
    for (Index i = 0; i < n_vec; i += 4) {
        float32x4_t preds = vld1q_f32(predictions + i);
        float32x4_t y = vld1q_f32(labels + i);

        // Fast sigmoid
        float32x4_t p = fast_sigmoid_neon(preds);

        // Gradient: p - y
        float32x4_t grad = vsubq_f32(p, y);
        vst1q_f32(gradients + i, grad);

        // Hessian: max(p * (1 - p), 1e-6)
        float32x4_t one_minus_p = vsubq_f32(one, p);
        float32x4_t hess = vmulq_f32(p, one_minus_p);
        hess = vmaxq_f32(hess, min_hess);
        vst1q_f32(hessians + i, hess);
    }

    // Handle remaining elements
    for (Index i = n_vec; i < n; ++i) {
        Float pred = std::max(-15.0f, std::min(15.0f, predictions[i]));
        Float p = sigmoid(pred);
        Float y = labels[i];
        gradients[i] = p - y;
        hessians[i] = std::max(p * (1 - p), 1e-6f);
    }

#else
    // Scalar fallback
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i) {
        Float pred = std::max(-15.0f, std::min(15.0f, predictions[i]));
        Float p = sigmoid(pred);
        Float y = labels[i];
        gradients[i] = p - y;
        hessians[i] = std::max(p * (1 - p), 1e-6f);
    }
#endif
}

Float LogLoss::transform_prediction(Float raw) const {
    return sigmoid(raw);
}

Float LogLoss::init_prediction(const Float* labels, Index n) const {
    Float sum = 0.0f;
    for (Index i = 0; i < n; ++i) {
        sum += labels[i];
    }
    Float p = sum / n;
    p = clip(p);
    // Initial raw score: log(p / (1-p))
    return std::log(p / (1 - p));
}

// ============================================================================
// Cross Entropy (Multiclass)
// ============================================================================

Float CrossEntropyLoss::compute_loss(const Float* labels, const Float* predictions, Index n) const {
    Float loss = 0.0f;
    
    std::vector<Float> probs(n_classes_);
    
    for (Index i = 0; i < n; ++i) {
        softmax(predictions + i * n_classes_, probs.data(), n_classes_);
        Index label = static_cast<Index>(labels[i]);
        loss += -std::log(std::max(probs[label], 1e-7f));
    }
    
    return loss / n;
}

void CrossEntropyLoss::compute_gradients(
    const Float* labels,
    const Float* predictions,
    Float* gradients,
    Float* hessians,
    Index n
) const {
    std::vector<Float> probs(n_classes_);
    
    for (Index i = 0; i < n; ++i) {
        softmax(predictions + i * n_classes_, probs.data(), n_classes_);
        Index label = static_cast<Index>(labels[i]);
        
        for (uint32_t c = 0; c < n_classes_; ++c) {
            Float p = probs[c];
            Float target = (c == label) ? 1.0f : 0.0f;
            
            gradients[i * n_classes_ + c] = p - target;
            hessians[i * n_classes_ + c] = std::max(p * (1 - p), 1e-6f);
        }
    }
}

Float CrossEntropyLoss::init_prediction(const Float* labels, Index n) const {
    return 0.0f;  // Start with uniform predictions
}

void CrossEntropyLoss::softmax(const Float* input, Float* output, uint32_t n) const {
    Float max_val = *std::max_element(input, input + n);
    Float sum = 0.0f;

    for (uint32_t i = 0; i < n; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    for (uint32_t i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}

void CrossEntropyLoss::compute_multiclass_gradients(
    const Float* labels,
    const Float* predictions,
    Float* gradients,
    Float* hessians,
    Index n_samples
) const {
    #pragma omp parallel
    {
        // Thread-local buffer to avoid firstprivate issues with std::vector on GCC
        std::vector<Float> probs(n_classes_);

        #pragma omp for
        for (Index i = 0; i < n_samples; ++i) {
            // Compute softmax probabilities
            softmax(predictions + i * n_classes_, probs.data(), n_classes_);

            Index label = static_cast<Index>(labels[i]);

            for (uint32_t c = 0; c < n_classes_; ++c) {
                Float p = probs[c];
                Float target = (c == label) ? 1.0f : 0.0f;

                // Gradient: p_c - y_c (where y_c is one-hot encoded)
                gradients[i * n_classes_ + c] = p - target;

                // Hessian (diagonal approximation): p_c * (1 - p_c)
                hessians[i * n_classes_ + c] = std::max(p * (1.0f - p), 1e-6f);
            }
        }
    }
}

void CrossEntropyLoss::transform_to_proba(const Float* raw_scores, Float* proba, Index n_samples) const {
    #pragma omp parallel
    {
        std::vector<Float> temp(n_classes_);

        #pragma omp for
        for (Index i = 0; i < n_samples; ++i) {
            softmax(raw_scores + i * n_classes_, proba + i * n_classes_, n_classes_);
        }
    }
}

// ============================================================================
// MSE Loss
// ============================================================================

Float MSELoss::compute_loss(const Float* labels, const Float* predictions, Index n) const {
    Float loss = 0.0f;
    
    #pragma omp parallel for reduction(+:loss)
    for (Index i = 0; i < n; ++i) {
        Float diff = predictions[i] - labels[i];
        loss += diff * diff;
    }
    
    return loss / n;
}

void MSELoss::compute_gradients(
    const Float* labels,
    const Float* predictions,
    Float* gradients,
    Float* hessians,
    Index n
) const {
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i) {
        // Gradient: 2 * (pred - label)
        gradients[i] = 2.0f * (predictions[i] - labels[i]);
        // Hessian: 2
        hessians[i] = 2.0f;
    }
}

Float MSELoss::init_prediction(const Float* labels, Index n) const {
    Float sum = 0.0f;
    for (Index i = 0; i < n; ++i) {
        sum += labels[i];
    }
    return sum / n;
}

// ============================================================================
// MAE Loss
// ============================================================================

Float MAELoss::compute_loss(const Float* labels, const Float* predictions, Index n) const {
    Float loss = 0.0f;
    
    #pragma omp parallel for reduction(+:loss)
    for (Index i = 0; i < n; ++i) {
        loss += std::abs(predictions[i] - labels[i]);
    }
    
    return loss / n;
}

void MAELoss::compute_gradients(
    const Float* labels,
    const Float* predictions,
    Float* gradients,
    Float* hessians,
    Index n
) const {
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i) {
        Float diff = predictions[i] - labels[i];
        // Gradient: sign(pred - label)
        gradients[i] = (diff > 0) ? 1.0f : ((diff < 0) ? -1.0f : 0.0f);
        // Hessian: small constant (MAE is not twice differentiable)
        hessians[i] = 1.0f;
    }
}

Float MAELoss::init_prediction(const Float* labels, Index n) const {
    // Median for MAE
    std::vector<Float> sorted(labels, labels + n);
    std::sort(sorted.begin(), sorted.end());
    return sorted[n / 2];
}

// ============================================================================
// Huber Loss
// ============================================================================

Float HuberLoss::compute_loss(const Float* labels, const Float* predictions, Index n) const {
    Float loss = 0.0f;
    
    #pragma omp parallel for reduction(+:loss)
    for (Index i = 0; i < n; ++i) {
        Float diff = std::abs(predictions[i] - labels[i]);
        if (diff <= delta_) {
            loss += 0.5f * diff * diff;
        } else {
            loss += delta_ * (diff - 0.5f * delta_);
        }
    }
    
    return loss / n;
}

void HuberLoss::compute_gradients(
    const Float* labels,
    const Float* predictions,
    Float* gradients,
    Float* hessians,
    Index n
) const {
    // Improved Huber gradient with smooth transition
    // Uses modified pseudo-huber for better convergence
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i) {
        Float diff = predictions[i] - labels[i];
        Float abs_diff = std::abs(diff);

        if (abs_diff <= delta_) {
            // Quadratic region: g = diff, h = 1
            gradients[i] = diff;
            hessians[i] = 1.0f;
        } else {
            // Linear region: g = delta * sign(diff), h = delta / |diff|
            // Using smooth transition: h = delta^2 / (delta + |diff|)
            // This gives better convergence than constant small hessian
            Float sign = (diff > 0) ? 1.0f : -1.0f;
            gradients[i] = delta_ * sign;
            // Smooth hessian that decays from 1.0 at boundary to smaller values
            // This helps Newton's method converge better
            hessians[i] = delta_ * delta_ / (delta_ + abs_diff);
            hessians[i] = std::max(hessians[i], 0.01f);  // Floor to avoid issues
        }
    }
}

Float HuberLoss::init_prediction(const Float* labels, Index n) const {
    Float sum = 0.0f;
    for (Index i = 0; i < n; ++i) {
        sum += labels[i];
    }
    return sum / n;
}

// ============================================================================
// Robust Focal Loss
// ============================================================================

Float RobustFocalLoss::compute_loss(const Float* labels, const Float* predictions, Index n) const {
    Float loss = 0.0f;
    
    #pragma omp parallel for reduction(+:loss)
    for (Index i = 0; i < n; ++i) {
        Float p = sigmoid(predictions[i]);
        Float y = labels[i];
        
        // p_t = p if y=1, (1-p) if y=0
        Float p_t = y * p + (1 - y) * (1 - p);
        p_t = std::max(p_t, 1e-7f);
        
        // Alpha weighting
        Float alpha_t = y * alpha_ + (1 - y) * (1 - alpha_);
        
        // Focal term: (1 - p_t)^gamma
        Float focal = std::pow(1 - p_t, gamma_);
        
        // Robustness weight
        Float robust = robustness_weight(p, y);
        
        // Class weights
        Float class_weight = y * pos_weight_ + (1 - y) * neg_weight_;
        
        loss += -alpha_t * focal * robust * class_weight * std::log(p_t);
    }
    
    return loss / n;
}

void RobustFocalLoss::compute_gradients(
    const Float* labels,
    const Float* predictions,
    Float* gradients,
    Float* hessians,
    Index n
) const {
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i) {
        Float raw = predictions[i];
        Float p = sigmoid(raw);
        Float y = labels[i];

        Float p_t = y * p + (1 - y) * (1 - p);
        p_t = std::max(p_t, 1e-7f);

        Float alpha_t = y * alpha_ + (1 - y) * (1 - alpha_);
        Float class_weight = y * pos_weight_ + (1 - y) * neg_weight_;

        // Focal weight: (1 - p_t)^gamma
        Float focal_weight = std::pow(1.0f - p_t, gamma_);

        // Full coefficient
        Float coef = alpha_t * class_weight * focal_weight;

        // Simplified focal loss gradient (weighted cross-entropy approach)
        // This is numerically stable and proven to work
        // grad = coef * (p - y)
        gradients[i] = coef * (p - y);

        // Hessian approximation: coef * p * (1 - p)
        Float hess = coef * p * (1.0f - p);
        hessians[i] = std::max(hess, 1e-6f);
    }
}

Float RobustFocalLoss::transform_prediction(Float raw) const {
    return sigmoid(raw);
}

Float RobustFocalLoss::init_prediction(const Float* labels, Index n) const {
    Float sum = 0.0f;
    for (Index i = 0; i < n; ++i) {
        sum += labels[i];
    }
    Float p = sum / n;
    p = std::max(1e-7f, std::min(1.0f - 1e-7f, p));
    return std::log(p / (1 - p));
}

void RobustFocalLoss::set_class_weights(Float pos_weight, Float neg_weight) {
    pos_weight_ = pos_weight;
    neg_weight_ = neg_weight;
}

// ============================================================================
// LDAM Loss
// ============================================================================

LDAMLoss::LDAMLoss(const std::vector<Index>& class_counts, Float C) {
    set_margins_from_counts(class_counts, C);
}

Float LDAMLoss::compute_loss(const Float* labels, const Float* predictions, Index n) const {
    Float loss = 0.0f;
    
    // For binary classification
    #pragma omp parallel for reduction(+:loss)
    for (Index i = 0; i < n; ++i) {
        Float y = labels[i];
        Float raw = predictions[i];
        
        // Get margin for this class
        Index cls = static_cast<Index>(y);
        Float margin = (cls < margins_.size()) ? margins_[cls] : 0.0f;
        
        // Adjusted logit
        Float adjusted = raw - margin * (2 * y - 1);  // Subtract margin for positive class
        
        // Binary cross entropy with adjusted logit
        Float p = sigmoid(adjusted);
        loss += -y * std::log(std::max(p, 1e-7f)) - (1 - y) * std::log(std::max(1 - p, 1e-7f));
    }
    
    return loss / n;
}

void LDAMLoss::compute_gradients(
    const Float* labels,
    const Float* predictions,
    Float* gradients,
    Float* hessians,
    Index n
) const {
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i) {
        Float y = labels[i];
        Float raw = predictions[i];
        
        Index cls = static_cast<Index>(y);
        Float margin = (cls < margins_.size()) ? margins_[cls] : 0.0f;
        
        Float adjusted = raw - margin * (2 * y - 1);
        Float p = sigmoid(adjusted);
        
        // Gradient of adjusted logit
        gradients[i] = p - y;
        hessians[i] = std::max(p * (1 - p), 1e-6f);
    }
}

Float LDAMLoss::transform_prediction(Float raw) const {
    return sigmoid(raw);
}

Float LDAMLoss::init_prediction(const Float* labels, Index n) const {
    Float sum = 0.0f;
    for (Index i = 0; i < n; ++i) {
        sum += labels[i];
    }
    Float p = sum / n;
    p = std::max(1e-7f, std::min(1.0f - 1e-7f, p));
    return std::log(p / (1 - p));
}

void LDAMLoss::set_margins_from_counts(const std::vector<Index>& counts, Float C) {
    margins_.resize(counts.size());
    
    for (size_t i = 0; i < counts.size(); ++i) {
        // Margin = C / n^{1/4}
        margins_[i] = C / std::pow(static_cast<Float>(counts[i]), 0.25f);
    }
    
    n_classes_ = static_cast<uint32_t>(counts.size());
}

void LDAMLoss::set_margins(const std::vector<Float>& margins) {
    margins_ = margins;
    n_classes_ = static_cast<uint32_t>(margins.size());
}

// ============================================================================
// Logit-Adjusted Loss
// ============================================================================

LogitAdjustedLoss::LogitAdjustedLoss(const std::vector<Float>& class_priors, Float tau)
    : tau_(tau) {
    log_priors_.resize(class_priors.size());
    for (size_t i = 0; i < class_priors.size(); ++i) {
        log_priors_[i] = std::log(std::max(class_priors[i], 1e-7f));
    }
}

Float LogitAdjustedLoss::compute_loss(const Float* labels, const Float* predictions, Index n) const {
    Float loss = 0.0f;
    
    #pragma omp parallel for reduction(+:loss)
    for (Index i = 0; i < n; ++i) {
        Float y = labels[i];
        Float raw = predictions[i];
        
        // Binary case: adjust by log(π_1/π_0)
        Float adjustment = 0.0f;
        if (log_priors_.size() >= 2) {
            adjustment = tau_ * (log_priors_[1] - log_priors_[0]);
        }
        
        // Adjusted prediction for positive class
        Float adjusted = raw + adjustment * (2 * y - 1);
        
        Float p = sigmoid(adjusted);
        loss += -y * std::log(std::max(p, 1e-7f)) - (1 - y) * std::log(std::max(1 - p, 1e-7f));
    }
    
    return loss / n;
}

void LogitAdjustedLoss::compute_gradients(
    const Float* labels,
    const Float* predictions,
    Float* gradients,
    Float* hessians,
    Index n
) const {
    Float adjustment = 0.0f;
    if (log_priors_.size() >= 2) {
        adjustment = tau_ * (log_priors_[1] - log_priors_[0]);
    }
    
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i) {
        Float y = labels[i];
        Float raw = predictions[i];
        
        Float adjusted = raw + adjustment * (2 * y - 1);
        Float p = sigmoid(adjusted);
        
        gradients[i] = p - y;
        hessians[i] = std::max(p * (1 - p), 1e-6f);
    }
}

Float LogitAdjustedLoss::transform_prediction(Float raw) const {
    return sigmoid(raw);
}

Float LogitAdjustedLoss::init_prediction(const Float* labels, Index n) const {
    Float sum = 0.0f;
    for (Index i = 0; i < n; ++i) {
        sum += labels[i];
    }
    Float p = sum / n;
    p = std::max(1e-7f, std::min(1.0f - 1e-7f, p));
    return std::log(p / (1 - p));
}

void LogitAdjustedLoss::set_priors_from_labels(const Float* labels, Index n) {
    Float pos_count = 0.0f;
    for (Index i = 0; i < n; ++i) {
        pos_count += labels[i];
    }
    
    Float pi_1 = pos_count / n;
    Float pi_0 = 1 - pi_1;
    
    log_priors_ = {std::log(std::max(pi_0, 1e-7f)), std::log(std::max(pi_1, 1e-7f))};
}

// ============================================================================
// Tsallis Loss
// ============================================================================

Float TsallisLoss::compute_loss(const Float* labels, const Float* predictions, Index n) const {
    Float loss = 0.0f;
    
    #pragma omp parallel for reduction(+:loss)
    for (Index i = 0; i < n; ++i) {
        Float p = sigmoid(predictions[i]);
        Float y = labels[i];
        
        Float p_t = y * p + (1 - y) * (1 - p);
        p_t = std::max(p_t, 1e-7f);
        
        // Tsallis loss: -ln_q(p_t)
        loss += -tsallis_log(p_t);
    }
    
    return loss / n;
}

void TsallisLoss::compute_gradients(
    const Float* labels,
    const Float* predictions,
    Float* gradients,
    Float* hessians,
    Index n
) const {
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i) {
        Float raw = predictions[i];
        Float p = sigmoid(raw);
        Float y = labels[i];
        
        Float p_t = y * p + (1 - y) * (1 - p);
        p_t = std::max(p_t, 1e-7f);
        
        Float dpt_dp = (y > 0.5f) ? 1.0f : -1.0f;
        Float dp_draw = p * (1 - p);
        
        // Gradient of -ln_q(p_t)
        // d/dp_t [-ln_q(p_t)] = -p_t^{-q}
        Float dloss_dpt = -tsallis_log_derivative(p_t);
        
        gradients[i] = dloss_dpt * dpt_dp * dp_draw;
        
        // Hessian approximation
        hessians[i] = std::max(std::abs(dloss_dpt) * dp_draw, 1e-6f);
    }
}

Float TsallisLoss::transform_prediction(Float raw) const {
    return sigmoid(raw);
}

Float TsallisLoss::init_prediction(const Float* labels, Index n) const {
    Float sum = 0.0f;
    for (Index i = 0; i < n; ++i) {
        sum += labels[i];
    }
    Float p = sum / n;
    p = std::max(1e-7f, std::min(1.0f - 1e-7f, p));
    return std::log(p / (1 - p));
}

void TsallisLoss::auto_tune_q(const Float* labels, Index n) {
    // Heuristic: q closer to 1 for balanced, closer to 2 for imbalanced
    Float sum = 0.0f;
    for (Index i = 0; i < n; ++i) {
        sum += labels[i];
    }
    Float ratio = sum / n;
    Float imbalance = std::abs(ratio - 0.5f) * 2;  // 0 = balanced, 1 = extreme
    
    // q = 1 + imbalance (ranges from 1 to 2)
    q_ = 1.0f + imbalance;
}

// ============================================================================
// Asymmetric Loss Implementation
// ============================================================================

Float AsymmetricLoss::compute_loss(const Float* labels, const Float* predictions, Index n) const {
    Float loss = 0.0f;

    #pragma omp parallel for reduction(+:loss)
    for (Index i = 0; i < n; ++i) {
        Float p = sigmoid(predictions[i]);
        Float y = labels[i];
        p = std::max(1e-7f, std::min(1.0f - 1e-7f, p));

        // Asymmetric weighted cross-entropy
        // alpha weight for positive class (FN penalty)
        // (1-alpha) weight for negative class (FP penalty)
        Float pos_loss = -alpha_ * y * std::log(p);
        Float neg_loss = -(1.0f - alpha_) * (1.0f - y) * std::log(1.0f - p);

        // Apply focal modulation if gamma > 0
        if (gamma_pos_ > 0 && y > 0.5f) {
            pos_loss *= std::pow(1.0f - p, gamma_pos_);
        }
        if (gamma_neg_ > 0 && y < 0.5f) {
            neg_loss *= std::pow(p, gamma_neg_);
        }

        loss += pos_loss + neg_loss;
    }

    return loss / n;
}

void AsymmetricLoss::compute_gradients(
    const Float* labels,
    const Float* predictions,
    Float* gradients,
    Float* hessians,
    Index n
) const {
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i) {
        Float raw = predictions[i];
        Float p = sigmoid(raw);
        Float y = labels[i];

        // Gradient for asymmetric loss
        // d/dz [-alpha*y*log(p) - (1-alpha)*(1-y)*log(1-p)]
        // = alpha*y*(p-1) + (1-alpha)*(1-y)*p  (using d/dz log(sigmoid(z)) = 1 - sigmoid(z))
        // Simplified: alpha*(p-y) for positive, (1-alpha)*(p-y) for negative
        // = p - y weighted by alpha or (1-alpha)

        Float weight = (y > 0.5f) ? alpha_ : (1.0f - alpha_);

        // Apply focal modulation to gradient
        Float focal_weight = 1.0f;
        if (y > 0.5f && gamma_pos_ > 0) {
            focal_weight = std::pow(1.0f - p, gamma_pos_);
        } else if (y < 0.5f && gamma_neg_ > 0) {
            focal_weight = std::pow(p, gamma_neg_);
        }

        gradients[i] = weight * focal_weight * (p - y);
        hessians[i] = std::max(weight * focal_weight * p * (1.0f - p), 1e-6f);
    }
}

Float AsymmetricLoss::transform_prediction(Float raw) const {
    return sigmoid(raw);
}

Float AsymmetricLoss::init_prediction(const Float* labels, Index n) const {
    Float sum = 0.0f;
    for (Index i = 0; i < n; ++i) {
        sum += labels[i];
    }
    Float p = sum / n;
    p = std::max(1e-7f, std::min(1.0f - 1e-7f, p));
    return std::log(p / (1.0f - p));
}

// ============================================================================
// AUC Loss Implementation (Pairwise Ranking)
// ============================================================================

void AUCLoss::update_indices(const Float* labels, Index n) const {
    pos_indices_.clear();
    neg_indices_.clear();

    for (Index i = 0; i < n; ++i) {
        if (labels[i] > 0.5f) {
            pos_indices_.push_back(i);
        } else {
            neg_indices_.push_back(i);
        }
    }
}

Float AUCLoss::compute_loss(const Float* labels, const Float* predictions, Index n) const {
    update_indices(labels, n);

    if (pos_indices_.empty() || neg_indices_.empty()) {
        return 0.0f;  // Can't compute pairwise loss without both classes
    }

    Float loss = 0.0f;
    Index n_pairs = 0;

    // Sample pairs for efficiency (max 10000 pairs)
    Index max_pairs = std::min(static_cast<Index>(10000),
                               static_cast<Index>(pos_indices_.size() * neg_indices_.size()));
    Index step_pos = std::max(1u, static_cast<Index>(pos_indices_.size() * neg_indices_.size() / max_pairs));

    for (Index pi = 0; pi < pos_indices_.size(); pi += step_pos) {
        Index i = pos_indices_[pi];
        Float s_pos = predictions[i];

        for (Index nj = 0; nj < neg_indices_.size(); nj += step_pos) {
            Index j = neg_indices_[nj];
            Float s_neg = predictions[j];

            // Pairwise logistic loss: log(1 + exp(-(s_pos - s_neg - margin)))
            Float diff = s_pos - s_neg - margin_;
            loss += std::log(1.0f + std::exp(-diff));
            n_pairs++;
        }
    }

    return (n_pairs > 0) ? loss / n_pairs : 0.0f;
}

void AUCLoss::compute_gradients(
    const Float* labels,
    const Float* predictions,
    Float* gradients,
    Float* hessians,
    Index n
) const {
    update_indices(labels, n);

    // Initialize gradients and hessians
    std::fill(gradients, gradients + n, 0.0f);
    std::fill(hessians, hessians + n, 1e-6f);

    if (pos_indices_.empty() || neg_indices_.empty()) {
        return;
    }

    // For each positive-negative pair, compute gradients
    // Loss for pair (i,j): log(1 + exp(-(s_i - s_j)))
    // dL/ds_i = -sigma(-(s_i - s_j)) = -(1 - sigma(s_i - s_j))
    // dL/ds_j = sigma(-(s_i - s_j)) = 1 - sigma(s_i - s_j)

    Index max_pairs = std::min(static_cast<Index>(5000),
                               static_cast<Index>(pos_indices_.size() * neg_indices_.size()));
    Index step = std::max(1u, static_cast<Index>(pos_indices_.size() * neg_indices_.size() / max_pairs));

    std::vector<Float> grad_accum(n, 0.0f);
    std::vector<Index> pair_counts(n, 0);

    for (Index pi = 0; pi < pos_indices_.size(); pi += step) {
        Index i = pos_indices_[pi];
        Float s_pos = predictions[i];

        for (Index nj = 0; nj < neg_indices_.size(); nj += step) {
            Index j = neg_indices_[nj];
            Float s_neg = predictions[j];

            Float diff = s_pos - s_neg - margin_;
            Float sig = sigmoid(diff);

            // Gradient: want to increase s_pos and decrease s_neg
            grad_accum[i] += -(1.0f - sig);  // Negative gradient for pos (minimize loss)
            grad_accum[j] += (1.0f - sig);   // Positive gradient for neg

            pair_counts[i]++;
            pair_counts[j]++;
        }
    }

    // Normalize and set gradients
    for (Index i = 0; i < n; ++i) {
        if (pair_counts[i] > 0) {
            gradients[i] = grad_accum[i] / pair_counts[i];
            // Hessian approximation
            Float p = sigmoid(predictions[i]);
            hessians[i] = std::max(p * (1.0f - p), 1e-6f);
        }
    }
}

Float AUCLoss::transform_prediction(Float raw) const {
    return sigmoid(raw);
}

Float AUCLoss::init_prediction(const Float* labels, Index n) const {
    return 0.0f;  // Start from zero
}

// ============================================================================
// Class-Balanced Loss Implementation
// ============================================================================

void ClassBalancedLoss::set_class_counts(Index pos_count, Index neg_count) {
    // Effective number of samples: E_n = (1 - beta^n) / (1 - beta)
    Float E_pos = (1.0f - std::pow(beta_, static_cast<Float>(pos_count))) / (1.0f - beta_);
    Float E_neg = (1.0f - std::pow(beta_, static_cast<Float>(neg_count))) / (1.0f - beta_);

    // Weights inversely proportional to effective number
    Float total = E_pos + E_neg;
    pos_weight_ = total / (2.0f * E_pos);
    neg_weight_ = total / (2.0f * E_neg);
}

void ClassBalancedLoss::set_weights(Float pos_weight, Float neg_weight) {
    pos_weight_ = pos_weight;
    neg_weight_ = neg_weight;
}

Float ClassBalancedLoss::compute_loss(const Float* labels, const Float* predictions, Index n) const {
    Float loss = 0.0f;

    #pragma omp parallel for reduction(+:loss)
    for (Index i = 0; i < n; ++i) {
        Float p = sigmoid(predictions[i]);
        Float y = labels[i];
        p = std::max(1e-7f, std::min(1.0f - 1e-7f, p));

        Float weight = (y > 0.5f) ? pos_weight_ : neg_weight_;
        loss += weight * (-y * std::log(p) - (1.0f - y) * std::log(1.0f - p));
    }

    return loss / n;
}

void ClassBalancedLoss::compute_gradients(
    const Float* labels,
    const Float* predictions,
    Float* gradients,
    Float* hessians,
    Index n
) const {
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i) {
        Float p = sigmoid(predictions[i]);
        Float y = labels[i];

        Float weight = (y > 0.5f) ? pos_weight_ : neg_weight_;

        gradients[i] = weight * (p - y);
        hessians[i] = std::max(weight * p * (1.0f - p), 1e-6f);
    }
}

Float ClassBalancedLoss::transform_prediction(Float raw) const {
    return sigmoid(raw);
}

Float ClassBalancedLoss::init_prediction(const Float* labels, Index n) const {
    Float sum = 0.0f;
    for (Index i = 0; i < n; ++i) {
        sum += labels[i];
    }
    Float p = sum / n;
    p = std::max(1e-7f, std::min(1.0f - 1e-7f, p));
    return std::log(p / (1.0f - p));
}

// ============================================================================
// PR-AUC Loss Implementation
// ============================================================================

Float PRAUCLoss::compute_loss(const Float* labels, const Float* predictions, Index n) const {
    // Similar to AUC loss but weights pairs by precision at each threshold
    std::vector<Index> pos_indices, neg_indices;

    for (Index i = 0; i < n; ++i) {
        if (labels[i] > 0.5f) {
            pos_indices.push_back(i);
        } else {
            neg_indices.push_back(i);
        }
    }

    if (pos_indices.empty() || neg_indices.empty()) {
        return 0.0f;
    }

    Float loss = 0.0f;
    Index n_pairs = 0;

    // Weight by inverse of negative count (emphasize precision)
    Float neg_weight = 1.0f / std::max(1.0f, static_cast<Float>(neg_indices.size()));

    Index max_pairs = std::min(static_cast<Index>(5000),
                               static_cast<Index>(pos_indices.size() * neg_indices.size()));
    Index step = std::max(1u, static_cast<Index>(pos_indices.size() * neg_indices.size() / max_pairs));

    for (Index pi = 0; pi < pos_indices.size(); pi += step) {
        Index i = pos_indices[pi];
        Float s_pos = predictions[i];

        for (Index nj = 0; nj < neg_indices.size(); nj += step) {
            Index j = neg_indices[nj];
            Float s_neg = predictions[j];

            // Higher weight for violations (neg ranked higher than pos)
            Float diff = s_pos - s_neg - margin_;
            Float pair_loss = std::log(1.0f + std::exp(-diff));

            // Weight by position to emphasize precision
            loss += neg_weight * pair_loss;
            n_pairs++;
        }
    }

    return (n_pairs > 0) ? loss / n_pairs : 0.0f;
}

void PRAUCLoss::compute_gradients(
    const Float* labels,
    const Float* predictions,
    Float* gradients,
    Float* hessians,
    Index n
) const {
    std::vector<Index> pos_indices, neg_indices;

    for (Index i = 0; i < n; ++i) {
        if (labels[i] > 0.5f) {
            pos_indices.push_back(i);
        } else {
            neg_indices.push_back(i);
        }
    }

    std::fill(gradients, gradients + n, 0.0f);
    std::fill(hessians, hessians + n, 1e-6f);

    if (pos_indices.empty() || neg_indices.empty()) {
        return;
    }

    Float neg_weight = 1.0f / std::max(1.0f, static_cast<Float>(neg_indices.size()));

    std::vector<Float> grad_accum(n, 0.0f);
    std::vector<Index> pair_counts(n, 0);

    Index max_pairs = std::min(static_cast<Index>(5000),
                               static_cast<Index>(pos_indices.size() * neg_indices.size()));
    Index step = std::max(1u, static_cast<Index>(pos_indices.size() * neg_indices.size() / max_pairs));

    for (Index pi = 0; pi < pos_indices.size(); pi += step) {
        Index i = pos_indices[pi];
        Float s_pos = predictions[i];

        for (Index nj = 0; nj < neg_indices.size(); nj += step) {
            Index j = neg_indices[nj];
            Float s_neg = predictions[j];

            Float diff = s_pos - s_neg - margin_;
            Float sig = sigmoid(diff);

            // Weighted gradients
            grad_accum[i] += neg_weight * (-(1.0f - sig));
            grad_accum[j] += neg_weight * (1.0f - sig);

            pair_counts[i]++;
            pair_counts[j]++;
        }
    }

    for (Index i = 0; i < n; ++i) {
        if (pair_counts[i] > 0) {
            gradients[i] = grad_accum[i] / pair_counts[i];
            Float p = sigmoid(predictions[i]);
            hessians[i] = std::max(p * (1.0f - p), 1e-6f);
        }
    }
}

Float PRAUCLoss::transform_prediction(Float raw) const {
    return sigmoid(raw);
}

Float PRAUCLoss::init_prediction(const Float* labels, Index n) const {
    return 0.0f;
}

} // namespace turbocat
