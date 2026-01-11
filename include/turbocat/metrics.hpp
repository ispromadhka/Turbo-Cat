#pragma once

/**
 * TurboCat: Evaluation Metrics
 *
 * Comprehensive metrics for classification and regression:
 * - ROC-AUC: Area under ROC curve
 * - PR-AUC: Area under Precision-Recall curve
 * - F1 Score: Harmonic mean of precision and recall
 * - Precision, Recall, Accuracy
 * - Log Loss: Cross-entropy loss
 * - Optimal threshold finding
 */

#include "types.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace turbocat {

// ============================================================================
// Metric Types
// ============================================================================

enum class MetricType : uint8_t {
    LogLoss = 0,
    Accuracy = 1,
    Precision = 2,
    Recall = 3,
    F1 = 4,
    ROC_AUC = 5,
    PR_AUC = 6,
    MSE = 7,
    MAE = 8,
    RMSE = 9
};

// ============================================================================
// Confusion Matrix
// ============================================================================

struct ConfusionMatrix {
    Index tp = 0;  // True positives
    Index tn = 0;  // True negatives
    Index fp = 0;  // False positives
    Index fn = 0;  // False negatives

    Float precision() const {
        return (tp + fp > 0) ? static_cast<Float>(tp) / (tp + fp) : 0.0f;
    }

    Float recall() const {
        return (tp + fn > 0) ? static_cast<Float>(tp) / (tp + fn) : 0.0f;
    }

    Float f1_score() const {
        Float p = precision();
        Float r = recall();
        return (p + r > 0) ? 2.0f * p * r / (p + r) : 0.0f;
    }

    Float accuracy() const {
        Index total = tp + tn + fp + fn;
        return (total > 0) ? static_cast<Float>(tp + tn) / total : 0.0f;
    }

    Float specificity() const {
        return (tn + fp > 0) ? static_cast<Float>(tn) / (tn + fp) : 0.0f;
    }

    Float balanced_accuracy() const {
        return (recall() + specificity()) / 2.0f;
    }
};

// ============================================================================
// Metrics Class
// ============================================================================

class Metrics {
public:
    /**
     * Compute confusion matrix for binary classification
     */
    static ConfusionMatrix confusion_matrix(
        const Float* y_true,
        const Float* y_pred,
        Index n_samples,
        Float threshold = 0.5f
    );

    static ConfusionMatrix confusion_matrix(
        const std::vector<Float>& y_true,
        const std::vector<Float>& y_pred,
        Float threshold = 0.5f
    ) {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("y_true and y_pred must have same size");
        }
        return confusion_matrix(y_true.data(), y_pred.data(), y_true.size(), threshold);
    }

    /**
     * Log Loss (Cross-Entropy)
     */
    static Float log_loss(
        const Float* y_true,
        const Float* y_pred,
        Index n_samples,
        Float eps = 1e-15f
    );

    static Float log_loss(
        const std::vector<Float>& y_true,
        const std::vector<Float>& y_pred,
        Float eps = 1e-15f
    ) {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("y_true and y_pred must have same size");
        }
        return log_loss(y_true.data(), y_pred.data(), y_true.size(), eps);
    }

    /**
     * ROC-AUC: Area Under ROC Curve
     * Uses trapezoidal integration with O(n log n) complexity
     */
    static Float roc_auc(
        const Float* y_true,
        const Float* y_pred,
        Index n_samples
    );

    static Float roc_auc(
        const std::vector<Float>& y_true,
        const std::vector<Float>& y_pred
    ) {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("y_true and y_pred must have same size");
        }
        return roc_auc(y_true.data(), y_pred.data(), y_true.size());
    }

    /**
     * PR-AUC: Area Under Precision-Recall Curve
     * More informative than ROC-AUC for imbalanced datasets
     */
    static Float pr_auc(
        const Float* y_true,
        const Float* y_pred,
        Index n_samples
    );

    static Float pr_auc(
        const std::vector<Float>& y_true,
        const std::vector<Float>& y_pred
    ) {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("y_true and y_pred must have same size");
        }
        return pr_auc(y_true.data(), y_pred.data(), y_true.size());
    }

    /**
     * Precision at given threshold
     */
    static Float precision(
        const Float* y_true,
        const Float* y_pred,
        Index n_samples,
        Float threshold = 0.5f
    );

    static Float precision(
        const std::vector<Float>& y_true,
        const std::vector<Float>& y_pred,
        Float threshold = 0.5f
    ) {
        return confusion_matrix(y_true, y_pred, threshold).precision();
    }

    /**
     * Recall at given threshold
     */
    static Float recall(
        const Float* y_true,
        const Float* y_pred,
        Index n_samples,
        Float threshold = 0.5f
    );

    static Float recall(
        const std::vector<Float>& y_true,
        const std::vector<Float>& y_pred,
        Float threshold = 0.5f
    ) {
        return confusion_matrix(y_true, y_pred, threshold).recall();
    }

    /**
     * F1 Score at given threshold
     */
    static Float f1_score(
        const Float* y_true,
        const Float* y_pred,
        Index n_samples,
        Float threshold = 0.5f
    );

    static Float f1_score(
        const std::vector<Float>& y_true,
        const std::vector<Float>& y_pred,
        Float threshold = 0.5f
    ) {
        return confusion_matrix(y_true, y_pred, threshold).f1_score();
    }

    /**
     * Accuracy at given threshold
     */
    static Float accuracy(
        const Float* y_true,
        const Float* y_pred,
        Index n_samples,
        Float threshold = 0.5f
    );

    static Float accuracy(
        const std::vector<Float>& y_true,
        const std::vector<Float>& y_pred,
        Float threshold = 0.5f
    ) {
        return confusion_matrix(y_true, y_pred, threshold).accuracy();
    }

    /**
     * Find optimal threshold for a given metric
     * Searches thresholds from predictions to maximize the metric
     */
    static Float find_optimal_threshold(
        const Float* y_true,
        const Float* y_pred,
        Index n_samples,
        MetricType metric = MetricType::F1,
        int n_thresholds = 100
    );

    static Float find_optimal_threshold(
        const std::vector<Float>& y_true,
        const std::vector<Float>& y_pred,
        MetricType metric = MetricType::F1,
        int n_thresholds = 100
    ) {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("y_true and y_pred must have same size");
        }
        return find_optimal_threshold(y_true.data(), y_pred.data(), y_true.size(), metric, n_thresholds);
    }

    /**
     * Compute metric at optimal threshold
     * Returns pair of (optimal_threshold, metric_value)
     */
    static std::pair<Float, Float> compute_at_optimal_threshold(
        const std::vector<Float>& y_true,
        const std::vector<Float>& y_pred,
        MetricType metric = MetricType::F1,
        int n_thresholds = 100
    );

    // ========================================================================
    // Regression Metrics
    // ========================================================================

    /**
     * Mean Squared Error
     */
    static Float mse(
        const Float* y_true,
        const Float* y_pred,
        Index n_samples
    );

    static Float mse(
        const std::vector<Float>& y_true,
        const std::vector<Float>& y_pred
    ) {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("y_true and y_pred must have same size");
        }
        return mse(y_true.data(), y_pred.data(), y_true.size());
    }

    /**
     * Root Mean Squared Error
     */
    static Float rmse(
        const std::vector<Float>& y_true,
        const std::vector<Float>& y_pred
    ) {
        return std::sqrt(mse(y_true, y_pred));
    }

    /**
     * Mean Absolute Error
     */
    static Float mae(
        const Float* y_true,
        const Float* y_pred,
        Index n_samples
    );

    static Float mae(
        const std::vector<Float>& y_true,
        const std::vector<Float>& y_pred
    ) {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("y_true and y_pred must have same size");
        }
        return mae(y_true.data(), y_pred.data(), y_true.size());
    }

    /**
     * R-squared (coefficient of determination)
     */
    static Float r2_score(
        const std::vector<Float>& y_true,
        const std::vector<Float>& y_pred
    );

    // ========================================================================
    // Generic metric computation
    // ========================================================================

    /**
     * Compute any supported metric
     */
    static Float compute(
        MetricType metric,
        const std::vector<Float>& y_true,
        const std::vector<Float>& y_pred,
        Float threshold = 0.5f
    );

    /**
     * Check if higher values are better for a metric
     */
    static bool higher_is_better(MetricType metric);

    /**
     * Get metric name as string
     */
    static const char* metric_name(MetricType metric);
};

// ============================================================================
// Inline Implementations for Performance
// ============================================================================

inline ConfusionMatrix Metrics::confusion_matrix(
    const Float* y_true,
    const Float* y_pred,
    Index n_samples,
    Float threshold
) {
    ConfusionMatrix cm;

    for (Index i = 0; i < n_samples; ++i) {
        bool actual = y_true[i] > 0.5f;
        bool predicted = y_pred[i] >= threshold;

        if (actual && predicted) cm.tp++;
        else if (!actual && !predicted) cm.tn++;
        else if (!actual && predicted) cm.fp++;
        else cm.fn++;
    }

    return cm;
}

inline Float Metrics::log_loss(
    const Float* y_true,
    const Float* y_pred,
    Index n_samples,
    Float eps
) {
    Float loss = 0.0f;

    for (Index i = 0; i < n_samples; ++i) {
        Float p = std::max(eps, std::min(1.0f - eps, y_pred[i]));
        Float y = y_true[i];
        loss -= y * std::log(p) + (1.0f - y) * std::log(1.0f - p);
    }

    return loss / n_samples;
}

inline Float Metrics::precision(
    const Float* y_true,
    const Float* y_pred,
    Index n_samples,
    Float threshold
) {
    return confusion_matrix(y_true, y_pred, n_samples, threshold).precision();
}

inline Float Metrics::recall(
    const Float* y_true,
    const Float* y_pred,
    Index n_samples,
    Float threshold
) {
    return confusion_matrix(y_true, y_pred, n_samples, threshold).recall();
}

inline Float Metrics::f1_score(
    const Float* y_true,
    const Float* y_pred,
    Index n_samples,
    Float threshold
) {
    return confusion_matrix(y_true, y_pred, n_samples, threshold).f1_score();
}

inline Float Metrics::accuracy(
    const Float* y_true,
    const Float* y_pred,
    Index n_samples,
    Float threshold
) {
    return confusion_matrix(y_true, y_pred, n_samples, threshold).accuracy();
}

inline Float Metrics::mse(
    const Float* y_true,
    const Float* y_pred,
    Index n_samples
) {
    Float sum = 0.0f;
    for (Index i = 0; i < n_samples; ++i) {
        Float diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }
    return sum / n_samples;
}

inline Float Metrics::mae(
    const Float* y_true,
    const Float* y_pred,
    Index n_samples
) {
    Float sum = 0.0f;
    for (Index i = 0; i < n_samples; ++i) {
        sum += std::abs(y_true[i] - y_pred[i]);
    }
    return sum / n_samples;
}

inline bool Metrics::higher_is_better(MetricType metric) {
    switch (metric) {
        case MetricType::LogLoss:
        case MetricType::MSE:
        case MetricType::MAE:
        case MetricType::RMSE:
            return false;
        default:
            return true;
    }
}

inline const char* Metrics::metric_name(MetricType metric) {
    switch (metric) {
        case MetricType::LogLoss: return "log_loss";
        case MetricType::Accuracy: return "accuracy";
        case MetricType::Precision: return "precision";
        case MetricType::Recall: return "recall";
        case MetricType::F1: return "f1";
        case MetricType::ROC_AUC: return "roc_auc";
        case MetricType::PR_AUC: return "pr_auc";
        case MetricType::MSE: return "mse";
        case MetricType::MAE: return "mae";
        case MetricType::RMSE: return "rmse";
        default: return "unknown";
    }
}

} // namespace turbocat
