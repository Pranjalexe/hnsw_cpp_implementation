#pragma once
#include <vector>
#include <mutex>
#include <cstdint>

namespace hnsw {

using NodeId = uint32_t;
static constexpr NodeId INVALID_ID = UINT32_MAX;

// One node in the HNSW graph.
// neighbors[lc] holds the adjacency list at layer lc.
// Layer 0 allows up to 2*M neighbors; upper layers allow M.
struct Node {
    std::vector<std::vector<NodeId>> neighbors; // neighbors[layer] = list of neighbor ids
    std::mutex                       mtx;       // per-node lock for concurrent inserts

    explicit Node(int max_layer, int M) {
        neighbors.resize(max_layer + 1);
        // pre-reserve to avoid small reallocations during construction
        neighbors[0].reserve(2 * M);
        for (int l = 1; l <= max_layer; ++l)
            neighbors[l].reserve(M);
    }

    // Non-copyable (mutex), movable
    Node(const Node&)            = delete;
    Node& operator=(const Node&) = delete;
    Node(Node&&)                 = default;
    Node& operator=(Node&&)      = default;

    int  max_layer() const { return static_cast<int>(neighbors.size()) - 1; }
    bool has_layer(int lc) const { return lc < static_cast<int>(neighbors.size()); }
};

} // namespace hnsw