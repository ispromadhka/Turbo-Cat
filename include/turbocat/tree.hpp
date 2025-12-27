#pragma once

/**
 * TurboCat Tree Structures
 *
 * Includes:
 * - Standard decision tree (histogram-based)
 * - GradTree: Differentiable axis-aligned trees (AAAI 2024)
 * - FlatTreeEnsemble: Ultra-fast decision tables for inference
 *
 * GradTree innovation: Instead of greedy splitting, jointly optimize all
 * tree parameters via gradient descent. Uses dense representation with
 * differentiable feature selection and threshold optimization.
 */

#include "types.hpp"
#include "config.hpp"
#include "dataset.hpp"
#include "histogram.hpp"
#include <Eigen/Dense>
#include <memory>
#include <functional>
#include "turbocat/flat_tree.hpp"

namespace turbocat {

// ============================================================================
// Standard Decision Tree
// ============================================================================

class Tree {
public:
    Tree() = default;
    explicit Tree(const TreeConfig& config, uint32_t n_classes = 1);

    // Build tree from histogram
    void build(
        const Dataset& dataset,
        const std::vector<Index>& sample_indices,
        HistogramBuilder& hist_builder
    );

    // Build tree for multiclass (K gradients/hessians per sample)
    void build_multiclass(
        const Dataset& dataset,
        const std::vector<Index>& sample_indices,
        HistogramBuilder& hist_builder,
        const std::vector<Float>& all_gradients,   // n_samples * n_classes
        const std::vector<Float>& all_hessians     // n_samples * n_classes
    );

    // Predict single sample (binary/regression)
    Float predict(const Float* features, FeatureIndex n_features) const;
    Float predict(const Dataset& dataset, Index row) const;

    // Predict multiclass (returns K values)
    void predict_multiclass(const Dataset& dataset, Index row, Float* output) const;
    void predict_multiclass(const Float* features, FeatureIndex n_features, Float* output) const;

    // Batch prediction
    void predict_batch(
        const Float* features,
        Index n_samples,
        FeatureIndex n_features,
        Float* output
    ) const;

    // Batch prediction using binned data (faster for training)
    void predict_batch(const Dataset& data, Float* output) const;

    // Batch multiclass prediction (output: n_samples * n_classes)
    void predict_batch_multiclass(
        const Dataset& dataset,
        Float* output
    ) const;

    // Update leaf values (for refinement)
    void update_leaf_values(const Dataset& dataset, const std::vector<Index>& indices);

    // Access tree structure
    const std::vector<TreeNode>& nodes() const { return nodes_; }
    TreeIndex n_nodes() const { return static_cast<TreeIndex>(nodes_.size()); }
    TreeIndex n_leaves() const { return n_leaves_; }
    uint16_t depth() const { return depth_; }
    uint32_t n_classes() const { return n_classes_; }

    // Access multiclass leaf values
    const std::vector<Float>& multiclass_leaf_values() const { return multiclass_leaf_values_; }

    // Feature importance
    std::vector<Float> feature_importance() const;

    // Serialization
    void save(std::ostream& out) const;
    static Tree load(std::istream& in);

private:
    TreeConfig config_;
    std::vector<TreeNode> nodes_;
    TreeIndex n_leaves_ = 0;
    uint16_t depth_ = 0;
    uint32_t n_classes_ = 1;  // Number of classes (1 for binary/regression)

    // For multiclass: leaf_values[leaf_idx * n_classes + class_idx]
    // Maps node_idx to leaf_idx for lookup
    std::vector<Float> multiclass_leaf_values_;
    std::vector<TreeIndex> node_to_leaf_idx_;  // Maps node index to leaf index
    
    // Build helpers
    void build_recursive(
        TreeIndex node_idx,
        const Dataset& dataset,
        const std::vector<Index>& indices,
        const std::vector<FeatureIndex>& features,
        HistogramBuilder& hist_builder,
        Histogram& histogram,
        uint16_t current_depth
    );

    // Optimized build using pre-computed histogram (histogram subtraction)
    void build_recursive_with_hist(
        TreeIndex node_idx,
        const Dataset& dataset,
        const std::vector<Index>& indices,
        const std::vector<FeatureIndex>& features,
        HistogramBuilder& hist_builder,
        Histogram& histogram,
        uint16_t current_depth
    );

    // Optimized build with histogram subtraction trick and pooled memory
    void build_recursive_optimized(
        TreeIndex node_idx,
        const Dataset& dataset,
        const std::vector<Index>& indices,
        const std::vector<FeatureIndex>& features,
        HistogramBuilder& hist_builder,
        std::vector<Histogram>& hist_pool,
        int hist_idx,
        uint16_t current_depth
    );

    // Build helpers for multiclass
    void build_recursive_multiclass(
        TreeIndex node_idx,
        const Dataset& dataset,
        const std::vector<Index>& indices,
        const std::vector<FeatureIndex>& features,
        HistogramBuilder& hist_builder,
        Histogram& histogram,
        uint16_t current_depth,
        const std::vector<Float>& all_gradients,
        const std::vector<Float>& all_hessians
    );

    TreeIndex add_node();
    void make_leaf(TreeIndex node_idx, const GradientPair& stats);

    // Leaf-wise (loss-guided) tree building
    void build_leafwise(
        const Dataset& dataset,
        const std::vector<Index>& sample_indices,
        HistogramBuilder& hist_builder
    );

    // Make multiclass leaf (K values per leaf)
    void make_leaf_multiclass(
        TreeIndex node_idx,
        const std::vector<Index>& indices,
        const std::vector<Float>& all_gradients,
        const std::vector<Float>& all_hessians
    );

    // Get leaf index for a sample (traverse tree)
    TreeIndex get_leaf_idx(const Dataset& dataset, Index row) const;

    // Inference helpers
    TreeIndex traverse(const Float* features, const std::vector<Float>& bin_edges,
                       FeatureIndex feature) const;
};

// ============================================================================
// GradTree: Gradient-Optimized Decision Tree (AAAI 2024)
// ============================================================================

/**
 * GradTree represents a decision tree using a differentiable dense formulation:
 * 
 * For a tree of depth D with 2^D leaves:
 * - Feature selection: Soft one-hot W ∈ R^{(2^D-1) × F} for internal nodes
 * - Thresholds: t ∈ R^{(2^D-1)} for split points
 * - Leaf values: v ∈ R^{2^D}
 * 
 * The routing probability from root to leaf l is:
 *   P(leaf=l | x) = Π_{nodes on path} σ(±(Wx - t))
 * 
 * where σ is sigmoid (or entmax for sparsity) and ± depends on direction.
 * 
 * Prediction: ŷ = Σ_l P(leaf=l | x) · v_l
 * 
 * This allows gradient descent optimization of the entire tree structure,
 * overcoming the local optimality of greedy splitting.
 */

class GradTree {
public:
    using Matrix = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Vector = Eigen::Matrix<Float, Eigen::Dynamic, 1>;
    
    GradTree() = default;
    explicit GradTree(const TreeConfig& config, FeatureIndex n_features);
    
    // ========================================================================
    // Initialization
    // ========================================================================
    
    /**
     * Initialize from a standard tree (warm start)
     * This converts greedy splits to differentiable parameters
     */
    void initialize_from_tree(const Tree& tree, const Dataset& dataset);
    
    /**
     * Random initialization
     */
    void initialize_random(uint64_t seed);
    
    // ========================================================================
    // Training
    // ========================================================================
    
    /**
     * Optimize tree parameters via gradient descent
     * @param dataset Training data
     * @param indices Sample indices to use
     * @param loss_fn Loss function (gradient/hessian computation)
     */
    void optimize(
        const Dataset& dataset,
        const std::vector<Index>& indices,
        std::function<void(const Vector&, const Vector&, Vector&, Vector&)> loss_fn
    );
    
    /**
     * Single optimization step
     * Returns loss value
     */
    Float optimization_step(
        const Dataset& dataset,
        const std::vector<Index>& indices,
        const Vector& targets
    );
    
    // ========================================================================
    // Prediction
    // ========================================================================
    
    /**
     * Soft prediction (differentiable)
     * Returns weighted sum over leaf values
     */
    Float predict_soft(const Float* features) const;
    
    /**
     * Hard prediction (for inference)
     * Uses argmax routing
     */
    Float predict_hard(const Float* features) const;
    
    /**
     * Batch soft prediction
     */
    void predict_soft_batch(
        const Float* features,
        Index n_samples,
        Float* output
    ) const;
    
    /**
     * Compute leaf routing probabilities
     * Returns 2^D probabilities for each leaf
     */
    Vector compute_leaf_probs(const Float* features) const;
    
    // ========================================================================
    // Conversion
    // ========================================================================
    
    /**
     * Convert to standard tree (discretize parameters)
     * Uses argmax for feature selection and median for thresholds
     */
    Tree to_standard_tree() const;
    
    // ========================================================================
    // Access Parameters
    // ========================================================================
    
    const Matrix& feature_weights() const { return W_; }
    const Vector& thresholds() const { return t_; }
    const Vector& leaf_values() const { return v_; }
    
    Matrix& feature_weights() { return W_; }
    Vector& thresholds() { return t_; }
    Vector& leaf_values() { return v_; }
    
    uint16_t depth() const { return depth_; }
    TreeIndex n_internal_nodes() const { return (1 << depth_) - 1; }
    TreeIndex n_leaves() const { return 1 << depth_; }
    
private:
    TreeConfig config_;
    FeatureIndex n_features_ = 0;
    uint16_t depth_ = 0;
    
    // Learnable parameters
    Matrix W_;      // Feature selection weights: (2^D - 1) × F
    Vector t_;      // Split thresholds: 2^D - 1
    Vector v_;      // Leaf values: 2^D
    
    // Optimizer state (Adam)
    Matrix W_m_, W_v_;  // First/second moment for W
    Vector t_m_, t_v_;  // For thresholds
    Vector v_m_, v_v_;  // For leaf values
    uint32_t opt_step_ = 0;
    
    // Temperature for sigmoid (annealing)
    Float temperature_ = 1.0f;
    
    // ========================================================================
    // Internal Methods
    // ========================================================================
    
    /**
     * Forward pass: compute predictions and cache intermediate values
     */
    struct ForwardCache {
        Matrix routing_probs;      // N × (2^D - 1) internal node probs
        Matrix leaf_probs;         // N × 2^D leaf probabilities
        Vector predictions;        // N predictions
    };
    
    ForwardCache forward(const Matrix& X) const;
    
    /**
     * Backward pass: compute gradients
     */
    struct Gradients {
        Matrix dW;
        Vector dt;
        Vector dv;
    };
    
    Gradients backward(
        const Matrix& X,
        const ForwardCache& cache,
        const Vector& grad_output  // dL/d(predictions)
    ) const;
    
    /**
     * Apply Adam update
     */
    void adam_update(const Gradients& grads);
    
    /**
     * Entmax activation (sparse alternative to softmax)
     * Parameter alpha: 1.5 for entmax-1.5, 2.0 for sparsemax
     */
    static Vector entmax(const Vector& input, Float alpha = 1.5f);
    static Matrix entmax_rows(const Matrix& input, Float alpha = 1.5f);
    
    /**
     * Sigmoid with temperature
     */
    Float sigmoid(Float x) const {
        return 1.0f / (1.0f + std::exp(-x / temperature_));
    }
    
    /**
     * Get path to leaf (for standard tree conversion)
     */
    std::vector<std::pair<TreeIndex, bool>> get_path_to_leaf(TreeIndex leaf_idx) const;
};

// ============================================================================
// Tree Ensemble
// ============================================================================

class TreeEnsemble {
public:
    TreeEnsemble() = default;
    explicit TreeEnsemble(uint32_t n_classes) : n_classes_(n_classes) {}

    // Move operations
    TreeEnsemble(TreeEnsemble&&) = default;
    TreeEnsemble& operator=(TreeEnsemble&&) = default;

    // Disable copy (has unique_ptr members)
    TreeEnsemble(const TreeEnsemble&) = delete;
    TreeEnsemble& operator=(const TreeEnsemble&) = delete;

    void add_tree(std::unique_ptr<Tree> tree, Float weight = 1.0f);
    // void add_gradtree(std::unique_ptr<GradTree> tree, Float weight = 1.0f);

    // Add tree for specific class (multiclass K-trees-per-iteration)
    void add_tree_for_class(std::unique_ptr<Tree> tree, Float weight, uint32_t class_idx);

    // Binary/regression prediction
    Float predict(const Float* features, FeatureIndex n_features) const;
    Float predict(const Dataset& data, Index row) const;  // Using binned data
    void predict_batch(const Float* features, Index n_samples,
                       FeatureIndex n_features, Float* output) const;
    void predict_batch(const Dataset& data, Float* output) const;  // Using binned data

    // Optimized batch prediction - processes samples in cache-friendly manner
    void predict_batch_optimized(const Dataset& data, Float* output, int n_threads = -1) const;

    // Ultra-fast prediction using decision tables (FlatTree)
    void predict_batch_flat(const Dataset& data, Float* output, int n_threads = -1) const;

    // Multiclass prediction (K-trees-per-iteration approach)
    void predict_multiclass(const Dataset& data, Index row, Float* output) const;
    void predict_batch_multiclass(const Dataset& data, Float* output) const;  // output: n_samples * n_classes
    void predict_batch_multiclass_optimized(const Dataset& data, Float* output, int n_threads = -1) const;

    size_t n_trees() const { return trees_.size() + gradtrees_.size(); }
    uint32_t n_classes() const { return n_classes_; }
    void set_n_classes(uint32_t n) { n_classes_ = n; }
    const Tree& tree(size_t idx) const { return *trees_[idx]; }
    Float tree_weight(size_t idx) const { return tree_weights_[idx]; }

    // Ensemble sparsification (LP-based thinning)
    void sparsify(Float target_sparsity);

    // Feature importance (aggregated)
    std::vector<Float> feature_importance() const;

    // Prepare optimized inference structures
    void prepare_for_inference();

private:
    std::vector<std::unique_ptr<Tree>> trees_;
    std::vector<std::unique_ptr<GradTree>> gradtrees_;
    std::vector<Float> tree_weights_;
    std::vector<Float> gradtree_weights_;
    std::vector<uint32_t> tree_class_indices_;  // Which class each tree belongs to (for multiclass)
    uint32_t n_classes_ = 1;

    // Cached flat representation for fast inference
    mutable bool inference_prepared_ = false;
    mutable std::vector<TreeNode> flat_nodes_;  // All nodes from all trees concatenated
    mutable std::vector<size_t> tree_offsets_;   // Offset into flat_nodes_ for each tree

    // Ultra-fast decision table representation
    mutable bool flat_trees_prepared_ = false;
    mutable std::unique_ptr<FlatTreeEnsemble> flat_ensemble_;
};

} // namespace turbocat
