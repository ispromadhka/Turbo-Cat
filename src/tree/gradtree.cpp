/**
 * TurboCat GradTree Implementation
 * 
 * Gradient-based global optimization of decision trees.
 * Based on AAAI 2024 paper but with our own improvements.
 * 
 * Key idea: Instead of greedy splitting, represent tree as differentiable
 * parameters and optimize jointly via gradient descent.
 */

#include "turbocat/tree.hpp"
#include <random>
#include <cmath>

namespace turbocat {

// ============================================================================
// GradTree Construction
// ============================================================================

GradTree::GradTree(const TreeConfig& config, FeatureIndex n_features)
    : config_(config), n_features_(n_features) {
    
    depth_ = std::min(config.max_depth, static_cast<uint16_t>(8));  // Limit for memory
    
    TreeIndex n_internal = n_internal_nodes();
    TreeIndex n_leaf = n_leaves();
    
    // Initialize parameters
    W_ = Matrix::Zero(n_internal, n_features);
    t_ = Vector::Zero(n_internal);
    v_ = Vector::Zero(n_leaf);
    
    // Adam optimizer state
    W_m_ = Matrix::Zero(n_internal, n_features);
    W_v_ = Matrix::Zero(n_internal, n_features);
    t_m_ = Vector::Zero(n_internal);
    t_v_ = Vector::Zero(n_internal);
    v_m_ = Vector::Zero(n_leaf);
    v_v_ = Vector::Zero(n_leaf);
}

void GradTree::initialize_random(uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<Float> normal(0.0f, 0.1f);
    std::uniform_real_distribution<Float> uniform(-1.0f, 1.0f);
    
    // Initialize feature weights (sparse initialization)
    for (TreeIndex i = 0; i < n_internal_nodes(); ++i) {
        // Pick a random feature to weight highly
        FeatureIndex main_feature = rng() % n_features_;
        W_(i, main_feature) = 2.0f + normal(rng);
        
        // Small weights for others
        for (FeatureIndex f = 0; f < n_features_; ++f) {
            if (f != main_feature) {
                W_(i, f) = normal(rng) * 0.01f;
            }
        }
    }
    
    // Initialize thresholds
    for (TreeIndex i = 0; i < n_internal_nodes(); ++i) {
        t_(i) = uniform(rng);
    }
    
    // Initialize leaf values
    for (TreeIndex i = 0; i < n_leaves(); ++i) {
        v_(i) = normal(rng);
    }
}

void GradTree::initialize_from_tree(const Tree& tree, const Dataset& dataset) {
    const auto& nodes = tree.nodes();
    
    // Map standard tree nodes to GradTree parameters
    std::function<void(TreeIndex, TreeIndex)> map_node;
    
    map_node = [&](TreeIndex std_idx, TreeIndex grad_idx) {
        if (grad_idx >= n_internal_nodes()) return;
        
        const TreeNode& node = nodes[std_idx];
        
        if (node.is_leaf) {
            // This shouldn't happen for internal grad nodes
            return;
        }
        
        // Set feature weight (one-hot like)
        W_.row(grad_idx).setZero();
        W_(grad_idx, node.split_feature) = 5.0f;  // Strong weight
        
        // Set threshold from bin
        // Approximate: use middle of bin range
        const auto& info = dataset.feature_info(node.split_feature);
        if (!info.bin_edges.empty() && node.split_bin < info.bin_edges.size()) {
            t_(grad_idx) = info.bin_edges[node.split_bin];
        } else {
            t_(grad_idx) = static_cast<Float>(node.split_bin);
        }
        
        // Recurse to children
        if (node.left_child < nodes.size()) {
            map_node(node.left_child, 2 * grad_idx + 1);
        }
        if (node.right_child < nodes.size()) {
            map_node(node.right_child, 2 * grad_idx + 2);
        }
    };
    
    if (!nodes.empty()) {
        map_node(0, 0);
    }
    
    // Initialize leaf values
    v_.setZero();
}

// ============================================================================
// Forward Pass
// ============================================================================

GradTree::ForwardCache GradTree::forward(const Matrix& X) const {
    Index n_samples = X.rows();
    TreeIndex n_internal = n_internal_nodes();
    TreeIndex n_leaf = n_leaves();
    
    ForwardCache cache;
    cache.routing_probs = Matrix::Zero(n_samples, n_internal);
    cache.leaf_probs = Matrix::Zero(n_samples, n_leaf);
    cache.predictions = Vector::Zero(n_samples);
    
    // Compute internal node routing decisions
    // For each internal node i: p_i = σ((W_i · x) - t_i)
    // p_i is probability of going RIGHT
    
    for (TreeIndex i = 0; i < n_internal; ++i) {
        for (Index s = 0; s < n_samples; ++s) {
            Float wx = W_.row(i).dot(X.row(s).transpose());
            cache.routing_probs(s, i) = sigmoid(wx - t_(i));
        }
    }
    
    // Compute leaf probabilities
    // Leaf l's probability = product of routing decisions on path from root
    // Using binary tree indexing: 
    // - left child of node i = 2i + 1
    // - right child of node i = 2i + 2
    // - leaf indices start at n_internal
    
    for (Index s = 0; s < n_samples; ++s) {
        for (TreeIndex l = 0; l < n_leaf; ++l) {
            Float prob = 1.0f;
            
            // Trace path from leaf to root
            TreeIndex leaf_global = n_internal + l;
            TreeIndex current = leaf_global;
            
            while (current > 0) {
                TreeIndex parent = (current - 1) / 2;
                bool is_right_child = (current == 2 * parent + 2);
                
                Float routing = cache.routing_probs(s, parent);
                
                if (is_right_child) {
                    prob *= routing;  // Went right
                } else {
                    prob *= (1.0f - routing);  // Went left
                }
                
                current = parent;
            }
            
            cache.leaf_probs(s, l) = prob;
        }
        
        // Prediction = weighted sum of leaf values
        cache.predictions(s) = cache.leaf_probs.row(s).dot(v_);
    }
    
    return cache;
}

// ============================================================================
// Backward Pass
// ============================================================================

GradTree::Gradients GradTree::backward(
    const Matrix& X,
    const ForwardCache& cache,
    const Vector& grad_output
) const {
    Index n_samples = X.rows();
    TreeIndex n_internal = n_internal_nodes();
    TreeIndex n_leaf = n_leaves();
    
    Gradients grads;
    grads.dW = Matrix::Zero(n_internal, n_features_);
    grads.dt = Vector::Zero(n_internal);
    grads.dv = Vector::Zero(n_leaf);
    
    // Gradient w.r.t. leaf values: dL/dv_l = Σ_s (dL/dy_s) * P(leaf=l|x_s)
    for (TreeIndex l = 0; l < n_leaf; ++l) {
        for (Index s = 0; s < n_samples; ++s) {
            grads.dv(l) += grad_output(s) * cache.leaf_probs(s, l);
        }
    }
    
    // Gradient w.r.t. routing probabilities
    // dL/dp_i = Σ_l Σ_s (dL/dy_s) * v_l * (dP_l/dp_i)
    
    for (Index s = 0; s < n_samples; ++s) {
        Float dl_dy = grad_output(s);
        
        for (TreeIndex i = 0; i < n_internal; ++i) {
            Float p_i = cache.routing_probs(s, i);
            Float sigmoid_grad = p_i * (1.0f - p_i);  // σ'(x) = σ(x)(1-σ(x))
            
            // Compute contribution from each leaf that goes through node i
            Float dp_contribution = 0.0f;
            
            for (TreeIndex l = 0; l < n_leaf; ++l) {
                TreeIndex leaf_global = n_internal + l;
                
                // Check if this leaf's path goes through node i
                TreeIndex current = leaf_global;
                bool through_node = false;
                bool is_right = false;
                
                while (current > 0) {
                    TreeIndex parent = (current - 1) / 2;
                    if (parent == i) {
                        through_node = true;
                        is_right = (current == 2 * parent + 2);
                        break;
                    }
                    current = parent;
                }
                
                if (through_node) {
                    // dP_l/dp_i = P_l / factor
                    // where factor is p_i (if right) or (1-p_i) (if left)
                    Float P_l = cache.leaf_probs(s, l);
                    Float factor = is_right ? p_i : (1.0f - p_i);
                    
                    if (factor > 1e-10f) {
                        Float dP_l_dp_i = P_l / factor * (is_right ? 1.0f : -1.0f);
                        dp_contribution += v_(l) * dP_l_dp_i;
                    }
                }
            }
            
            Float dl_dp = dl_dy * dp_contribution;
            
            // Chain rule: dL/dt_i = dL/dp_i * dp_i/dt_i = dL/dp_i * (-sigmoid_grad)
            grads.dt(i) += dl_dp * (-sigmoid_grad);
            
            // dL/dW_i = dL/dp_i * dp_i/dW_i = dL/dp_i * sigmoid_grad * x
            grads.dW.row(i) += dl_dp * sigmoid_grad * X.row(s);
        }
    }
    
    return grads;
}

// ============================================================================
// Optimization
// ============================================================================

void GradTree::optimize(
    const Dataset& dataset,
    const std::vector<Index>& indices,
    std::function<void(const Vector&, const Vector&, Vector&, Vector&)> loss_fn
) {
    // Convert data to Eigen matrix
    Matrix X(indices.size(), n_features_);
    Vector y(indices.size());
    
    for (size_t i = 0; i < indices.size(); ++i) {
        Index idx = indices[i];
        for (FeatureIndex f = 0; f < n_features_; ++f) {
            X(i, f) = dataset.raw_value(idx, f);
        }
        y(i) = dataset.label(idx);
    }
    
    // Gradient descent iterations
    Float best_loss = 1e30f;
    int patience = 0;
    
    for (uint16_t iter = 0; iter < config_.gradtree_iterations; ++iter) {
        Float loss = optimization_step(dataset, indices, y);
        
        // Temperature annealing
        temperature_ = std::max(0.1f, temperature_ * 0.99f);
        
        // Early stopping
        if (loss < best_loss - 1e-5f) {
            best_loss = loss;
            patience = 0;
        } else {
            patience++;
            if (patience > 10) break;
        }
    }
}

Float GradTree::optimization_step(
    const Dataset& dataset,
    const std::vector<Index>& indices,
    const Vector& targets
) {
    // Build feature matrix
    Matrix X(indices.size(), n_features_);
    for (size_t i = 0; i < indices.size(); ++i) {
        Index idx = indices[i];
        for (FeatureIndex f = 0; f < n_features_; ++f) {
            X(i, f) = dataset.raw_value(idx, f);
        }
    }
    
    // Forward pass
    ForwardCache cache = forward(X);
    
    // Compute loss and gradient
    // Using MSE for simplicity: L = 0.5 * Σ(y - ŷ)²
    Float loss = 0.0f;
    Vector grad_output(indices.size());
    
    for (size_t i = 0; i < indices.size(); ++i) {
        Float diff = cache.predictions(i) - targets(i);
        loss += 0.5f * diff * diff;
        grad_output(i) = diff;  // dL/dŷ = (ŷ - y)
    }
    loss /= indices.size();
    
    // Backward pass
    Gradients grads = backward(X, cache, grad_output);
    
    // Scale gradients
    Float scale = 1.0f / indices.size();
    grads.dW *= scale;
    grads.dt *= scale;
    grads.dv *= scale;
    
    // Adam update
    adam_update(grads);
    
    return loss;
}

void GradTree::adam_update(const Gradients& grads) {
    opt_step_++;
    
    Float lr = config_.gradtree_lr;
    Float beta1 = 0.9f;
    Float beta2 = 0.999f;
    Float eps = 1e-8f;
    
    // Bias correction
    Float bc1 = 1.0f - std::pow(beta1, opt_step_);
    Float bc2 = 1.0f - std::pow(beta2, opt_step_);
    
    // Update W
    W_m_ = beta1 * W_m_ + (1 - beta1) * grads.dW;
    W_v_ = beta2 * W_v_ + (1 - beta2) * grads.dW.array().square().matrix();
    Matrix W_m_hat = W_m_ / bc1;
    Matrix W_v_hat = W_v_ / bc2;
    W_ -= lr * (W_m_hat.array() / (W_v_hat.array().sqrt() + eps)).matrix();
    
    // Update t
    t_m_ = beta1 * t_m_ + (1 - beta1) * grads.dt;
    t_v_ = beta2 * t_v_ + (1 - beta2) * grads.dt.array().square().matrix();
    Vector t_m_hat = t_m_ / bc1;
    Vector t_v_hat = t_v_ / bc2;
    t_ -= lr * (t_m_hat.array() / (t_v_hat.array().sqrt() + eps)).matrix();
    
    // Update v
    v_m_ = beta1 * v_m_ + (1 - beta1) * grads.dv;
    v_v_ = beta2 * v_v_ + (1 - beta2) * grads.dv.array().square().matrix();
    Vector v_m_hat = v_m_ / bc1;
    Vector v_v_hat = v_v_ / bc2;
    v_ -= lr * (v_m_hat.array() / (v_v_hat.array().sqrt() + eps)).matrix();
}

// ============================================================================
// Prediction
// ============================================================================

Float GradTree::predict_soft(const Float* features) const {
    // Single sample prediction
    Vector x(n_features_);
    for (FeatureIndex f = 0; f < n_features_; ++f) {
        x(f) = features[f];
    }
    
    Matrix X(1, n_features_);
    X.row(0) = x;
    
    ForwardCache cache = forward(X);
    return cache.predictions(0);
}

Float GradTree::predict_hard(const Float* features) const {
    // Argmax routing (no soft probabilities)
    TreeIndex current = 0;
    
    while (current < n_internal_nodes()) {
        Float wx = 0.0f;
        for (FeatureIndex f = 0; f < n_features_; ++f) {
            wx += W_(current, f) * features[f];
        }
        
        bool go_right = wx > t_(current);
        current = go_right ? (2 * current + 2) : (2 * current + 1);
    }
    
    // current is now a leaf index (offset by n_internal)
    TreeIndex leaf_idx = current - n_internal_nodes();
    return v_(leaf_idx);
}

void GradTree::predict_soft_batch(
    const Float* features,
    Index n_samples,
    Float* output
) const {
    Matrix X(n_samples, n_features_);
    for (Index i = 0; i < n_samples; ++i) {
        for (FeatureIndex f = 0; f < n_features_; ++f) {
            X(i, f) = features[i * n_features_ + f];
        }
    }
    
    ForwardCache cache = forward(X);
    
    for (Index i = 0; i < n_samples; ++i) {
        output[i] = cache.predictions(i);
    }
}

GradTree::Vector GradTree::compute_leaf_probs(const Float* features) const {
    Vector x(n_features_);
    for (FeatureIndex f = 0; f < n_features_; ++f) {
        x(f) = features[f];
    }
    
    Matrix X(1, n_features_);
    X.row(0) = x;
    
    ForwardCache cache = forward(X);
    return cache.leaf_probs.row(0);
}

// ============================================================================
// Conversion to Standard Tree
// ============================================================================

Tree GradTree::to_standard_tree() const {
    TreeConfig config = config_;
    config.max_depth = depth_;
    
    Tree tree(config);
    // Would need to implement proper conversion
    // For now, return empty tree
    return tree;
}

// ============================================================================
// Entmax (Sparse Softmax Alternative)
// ============================================================================

GradTree::Vector GradTree::entmax(const Vector& input, Float alpha) {
    // Simplified entmax-alpha implementation
    // For alpha = 1.5 (default), this provides sparse outputs
    
    if (std::abs(alpha - 1.0f) < 1e-6f) {
        // Standard softmax
        Vector exp_x = (input.array() - input.maxCoeff()).exp();
        return exp_x / exp_x.sum();
    }
    
    // Approximate entmax via iterative thresholding
    Vector sorted = input;
    std::sort(sorted.data(), sorted.data() + sorted.size(), std::greater<Float>());
    
    Float cumsum = 0.0f;
    Float tau = 0.0f;
    
    for (Index k = 0; k < sorted.size(); ++k) {
        cumsum += sorted(k);
        Float candidate = (cumsum - 1.0f) / (k + 1);
        if (sorted(k) > candidate) {
            tau = candidate;
        }
    }
    
    Vector output = (input.array() - tau).max(0.0f);
    Float sum = output.sum();
    if (sum > 0) {
        output /= sum;
    }
    
    return output;
}

GradTree::Matrix GradTree::entmax_rows(const Matrix& input, Float alpha) {
    Matrix output(input.rows(), input.cols());
    for (Index i = 0; i < input.rows(); ++i) {
        output.row(i) = entmax(input.row(i), alpha);
    }
    return output;
}

std::vector<std::pair<TreeIndex, bool>> GradTree::get_path_to_leaf(TreeIndex leaf_idx) const {
    std::vector<std::pair<TreeIndex, bool>> path;
    
    TreeIndex leaf_global = n_internal_nodes() + leaf_idx;
    TreeIndex current = leaf_global;
    
    while (current > 0) {
        TreeIndex parent = (current - 1) / 2;
        bool is_right = (current == 2 * parent + 2);
        path.emplace_back(parent, is_right);
        current = parent;
    }
    
    std::reverse(path.begin(), path.end());
    return path;
}

} // namespace turbocat
