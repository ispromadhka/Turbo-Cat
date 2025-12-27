/**
 * TurboCat Flat Tree Implementation
 */

#include "turbocat/flat_tree.hpp"
#include "turbocat/tree.hpp"
#include <cmath>
#include <stdexcept>
#include <queue>

namespace turbocat {

// Helper structure for tree traversal
struct TraversalInfo {
    TreeIndex node_idx;
    uint32_t path;      // Bit pattern of decisions taken
    uint8_t depth;      // Current depth
};

void FlatTreeEnsemble::from_ensemble(const TreeEnsemble& ensemble) {
    trees_.clear();
    max_depth_ = 0;

    for (size_t t = 0; t < ensemble.n_trees(); ++t) {
        const Tree& tree = ensemble.tree(t);
        Float weight = ensemble.tree_weight(t);

        if (tree.nodes().empty()) continue;

        FlatTree flat = flatten_tree(tree, weight);
        if (flat.depth > 0) {
            max_depth_ = std::max(max_depth_, flat.depth);
            trees_.push_back(std::move(flat));
        }
    }
}

FlatTree FlatTreeEnsemble::flatten_tree(const Tree& tree, Float weight) {
    FlatTree flat;
    flat.weight = weight;

    const auto& nodes = tree.nodes();
    if (nodes.empty()) return flat;

    // If it's a single leaf, create trivial flat tree
    if (nodes[0].is_leaf) {
        flat.depth = 1;
        flat.n_leaves = 2;
        flat.leaf_values = static_cast<Float*>(std::aligned_alloc(32, 2 * sizeof(Float)));
        flat.leaf_values[0] = nodes[0].value;
        flat.leaf_values[1] = nodes[0].value;
        flat.features[0] = 0;
        flat.thresholds[0] = 127;  // Middle threshold - doesn't matter for single leaf
        return flat;
    }

    // Determine tree depth and collect level info
    uint8_t tree_depth = static_cast<uint8_t>(tree.depth());
    if (tree_depth > MAX_FLAT_DEPTH) {
        tree_depth = MAX_FLAT_DEPTH;  // Truncate deep trees
    }

    flat.depth = tree_depth;
    flat.n_leaves = 1u << tree_depth;
    flat.leaf_values = static_cast<Float*>(std::aligned_alloc(32, flat.n_leaves * sizeof(Float)));

    // Initialize leaf values to 0
    std::memset(flat.leaf_values, 0, flat.n_leaves * sizeof(Float));

    // For non-oblivious trees, we need to find the most common feature at each level
    // and create a consistent decision table

    // First pass: collect all nodes at each level and their split info
    std::vector<std::vector<TreeIndex>> level_nodes(tree_depth);
    std::vector<std::pair<FeatureIndex, BinIndex>> level_splits(tree_depth, {0, 0});

    // BFS to collect level info
    std::queue<std::pair<TreeIndex, uint8_t>> bfs;
    bfs.push({0, 0});

    while (!bfs.empty()) {
        auto [node_idx, depth] = bfs.front();
        bfs.pop();

        if (depth >= tree_depth) continue;

        const TreeNode& node = nodes[node_idx];
        if (!node.is_leaf) {
            level_nodes[depth].push_back(node_idx);

            // Track most common split at this level
            // For simplicity, use the first encountered split
            if (level_splits[depth].first == 0 && level_splits[depth].second == 0) {
                level_splits[depth] = {node.split_feature, node.split_bin};
            }

            if (depth + 1 < tree_depth) {
                bfs.push({node.left_child, static_cast<uint8_t>(depth + 1)});
                bfs.push({node.right_child, static_cast<uint8_t>(depth + 1)});
            }
        }
    }

    // Set level features and thresholds
    for (uint8_t d = 0; d < tree_depth; ++d) {
        flat.features[d] = level_splits[d].first;
        flat.thresholds[d] = level_splits[d].second;
    }

    // Second pass: traverse tree and fill leaf values
    // For each possible path (0 to 2^depth - 1), find the corresponding leaf value
    for (uint32_t path = 0; path < flat.n_leaves; ++path) {
        TreeIndex node_idx = 0;
        uint8_t current_depth = 0;

        while (current_depth < tree_depth && !nodes[node_idx].is_leaf) {
            const TreeNode& node = nodes[node_idx];

            // Decision based on path bit
            bool go_right = (path >> (tree_depth - 1 - current_depth)) & 1;

            // Check if this node's split matches the level's split
            // If not, we need to make the decision based on the actual node split
            // This is a simplification - for best results, use oblivious trees

            if (go_right) {
                node_idx = node.right_child;
            } else {
                node_idx = node.left_child;
            }
            current_depth++;
        }

        // Store leaf value at this path index
        flat.leaf_values[path] = nodes[node_idx].value;
    }

    return flat;
}

void FlatTreeEnsemble::fill_leaves(const Tree& tree, const std::vector<TreeNode>& nodes,
                                   TreeIndex node_idx, uint32_t path, uint8_t current_depth,
                                   uint8_t target_depth, Float* leaf_values,
                                   const FeatureIndex* level_features, const BinIndex* level_thresholds) {
    const TreeNode& node = nodes[node_idx];

    if (node.is_leaf || current_depth >= target_depth) {
        // Fill all paths from here with this leaf's value
        uint32_t remaining_bits = target_depth - current_depth;
        uint32_t n_paths = 1u << remaining_bits;
        uint32_t base_idx = path << remaining_bits;

        for (uint32_t i = 0; i < n_paths; ++i) {
            leaf_values[base_idx + i] = node.value;
        }
        return;
    }

    // Recurse
    fill_leaves(tree, nodes, node.left_child, (path << 1) | 0, current_depth + 1,
                target_depth, leaf_values, level_features, level_thresholds);
    fill_leaves(tree, nodes, node.right_child, (path << 1) | 1, current_depth + 1,
                target_depth, leaf_values, level_features, level_thresholds);
}

} // namespace turbocat
