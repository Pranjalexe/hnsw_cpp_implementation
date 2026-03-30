#pragma once
#include "node.hpp"
#include "distance.hpp"

#include <vector>
#include <queue>
#include <random>
#include <mutex>
#include <shared_mutex>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <fstream>

namespace hnsw {

// --------------------------------------------------------------------------
// Result type
// --------------------------------------------------------------------------
struct Neighbor {
    NodeId id;
    float  dist;
    bool operator<(const Neighbor& o) const { return dist < o.dist; }
    bool operator>(const Neighbor& o) const { return dist > o.dist; }
};

// --------------------------------------------------------------------------
// HNSW Index
// --------------------------------------------------------------------------
class Index {
public:
    using DistFn    = std::function<float(const float*, const float*, size_t)>;
    using RawDistFn = float(*)(const float*, const float*, size_t);

    struct Params {
        int    M              = 16;
        int    ef_construction = 100;
        int    ef_search       = 50;
        size_t dim            = 128;
        DistFn dist_fn;               // defaults to L2 if empty
    };

    explicit Index(Params p);

    // ----- Mutation -----
    NodeId insert(const float* vec);
    NodeId insert(const Vector& v) { return insert(v.data()); }
    void   insert_batch(const float* data, size_t n);
    bool   remove(NodeId id);

    // ----- Query -----
    std::vector<Neighbor> search(const float* query, int K, int ef = -1) const;
    std::vector<Neighbor> search(const Vector& q, int K, int ef = -1) const {
        return search(q.data(), K, ef);
    }

    // ----- Access -----
    const float* get_vector(NodeId id) const;

    // ----- Persistence -----
    void save(const std::string& path) const;
    void load(const std::string& path);

    // ----- Stats -----
    size_t size()     const;
    size_t capacity() const;
    bool   empty()    const { return size() == 0; }
    bool   is_deleted(NodeId id) const;

    size_t dim()             const { return params_.dim; }
    int    M()               const { return params_.M; }
    int    ef_construction() const { return params_.ef_construction; }
    int    max_layer()       const { return entry_layer_; }

private:
    using MaxHeap = std::priority_queue<Neighbor>;
    using MinHeap = std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>>;

    int   sample_level() const;
    float dist(NodeId a, NodeId b)       const;
    float dist(const float* q, NodeId b) const;

    MaxHeap search_layer(const float*               q,
                         const std::vector<NodeId>& entry_points,
                         int ef, int lc,
                         std::vector<bool>&         visited_buf) const;

    std::vector<NodeId> select_neighbors_heuristic(
        const float* q, MaxHeap candidates, int M, int lc,
        bool extend_candidates = false, bool keep_pruned = true) const;

    void prune_connections(NodeId u, int lc, int max_conn);
    void repair_entry_point();

    // ---- Data ----
    Params    params_;
    RawDistFn dist_fn_;

    // Flat contiguous storage: data_[id * dim .. id * dim + dim - 1]
    std::vector<float>                 data_;
    std::vector<std::unique_ptr<Node>> nodes_;
    std::vector<bool>                  deleted_;

    NodeId entry_point_ = INVALID_ID;
    int    entry_layer_ = -1;

    mutable std::shared_mutex index_mtx_;
    mutable std::mt19937      rng_;
    mutable std::mutex        rng_mtx_;

    double mL_;

    static constexpr uint32_t MAGIC   = 0x484E5357; // "HNSW"
    static constexpr uint32_t VERSION = 2;           // bumped: flat storage format
};

} // namespace hnsw

namespace hnsw {
inline Index make_index(size_t dim = 128, int M = 16, int ef_construction = 100) {
    Index::Params p;
    p.dim             = dim;
    p.M               = M;
    p.ef_construction = ef_construction;
    return Index(p);
}
}