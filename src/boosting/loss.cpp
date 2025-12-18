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

namespace turbocat {

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
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i) {
        Float p = sigmoid(predictions[i]);
        Float y = labels[i];

        // Gradient: p - y
        gradients[i] = p - y;

        // Hessian: p * (1 - p)
        hessians[i] = std::max(p * (1 - p), 1e-6f);
    }
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
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i) {
        Float diff = predictions[i] - labels[i];
        Float abs_diff = std::abs(diff);

        if (abs_diff <= delta_) {
            gradients[i] = diff;
            hessians[i] = 1.0f;
        } else {
            gradients[i] = delta_ * ((diff > 0) ? 1.0f : -1.0f);
            hessians[i] = 1e-6f;  // Small for linear region
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
        
        // Focal components
        Float one_minus_pt = 1 - p_t;
        Float focal = std::pow(one_minus_pt, gamma_);
        Float log_pt = std::log(p_t);
        
        // Robustness
        Float robust = robustness_weight(p, y);
        
        // Full coefficient
        Float coef = alpha_t * class_weight * robust;
        
        // Gradient of focal loss
        // d/dp [-alpha * (1-pt)^gamma * log(pt)]
        // = -alpha * [gamma * (1-pt)^(gamma-1) * (-1) * log(pt) + (1-pt)^gamma * (1/pt)]
        // = alpha * (1-pt)^(gamma-1) * [gamma * log(pt) + (1-pt)/pt]
        
        Float dp_draw = p * (1 - p);  // sigmoid derivative
        Float dpt_dp = (y > 0.5f) ? 1.0f : -1.0f;
        
        Float focal_grad = std::pow(one_minus_pt, gamma_ - 1) * 
                          (gamma_ * log_pt * one_minus_pt + one_minus_pt - 1);
        
        gradients[i] = -coef * focal_grad * dpt_dp * dp_draw;
        
        // Hessian (approximation)
        hessians[i] = std::max(coef * focal * dp_draw, 1e-6f);
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

} // namespace turbocat
