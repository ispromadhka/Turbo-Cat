/**
 * TurboCat Fast Ensemble Implementation
 */

#include "turbocat/fast_ensemble.hpp"
#include <algorithm>

namespace turbocat {

void FastEnsemble::from_symmetric_ensemble(const SymmetricEnsemble& ensemble) {
    n_trees_ = ensemble.n_trees();
    if (n_trees_ == 0) return;

    // Find max depth
    max_depth_ = 0;
    for (size_t t = 0; t < n_trees_; ++t) {
        max_depth_ = std::max(max_depth_, ensemble.tree(t).depth());
    }

    if (max_depth_ == 0) {
        n_trees_ = 0;
        return;
    }

    leaves_per_tree_ = 1u << max_depth_;

    // Allocate flat arrays
    features_.resize(n_trees_ * max_depth_, 0);
    thresholds_.resize(n_trees_ * max_depth_, 0);
    leaf_values_.resize(n_trees_ * leaves_per_tree_, 0.0f);
    weights_.resize(n_trees_);
    depths_.resize(n_trees_);

    // Copy data from each tree
    for (size_t t = 0; t < n_trees_; ++t) {
        const SymmetricTree& tree = ensemble.tree(t);
        uint16_t depth = tree.depth();
        depths_[t] = depth;
        weights_[t] = ensemble.tree_weight(t);

        // Copy splits
        const auto& splits = tree.splits();
        for (uint16_t d = 0; d < depth; ++d) {
            features_[t * max_depth_ + d] = splits[d].feature;
            thresholds_[t * max_depth_ + d] = splits[d].threshold;
        }

        // Copy leaf values (may be fewer than leaves_per_tree_ if depth < max_depth_)
        const auto& leaves = tree.leaf_values();
        uint32_t n_leaves = 1u << depth;
        for (uint32_t i = 0; i < n_leaves; ++i) {
            leaf_values_[t * leaves_per_tree_ + i] = leaves[i];
        }
    }
}

} // namespace turbocat
