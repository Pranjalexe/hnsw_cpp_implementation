#pragma once
#include "node.hpp"
#include "distance.hpp"

#include <vector>
#include <queue>
#include <random>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <fstream>
#include <thread>
#include <condition_variable>

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
        int    M               = 16;
        int    ef_construction  = 100;
        int    ef_search        = 50;
        size_t dim             = 128;
        int    num_threads      = 0;   // 0 = use hardware_concurrency()
        DistFn dist_fn;                // defaults to L2 if empty
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
    size_t size()         const;
    size_t capacity()     const;
    bool   empty()        const { return size() == 0; }
    bool   is_deleted(NodeId id) const;
    int    thread_count() const { return num_threads_; }

    size_t dim()             const { return params_.dim; }
    int    M()               const { return params_.M; }
    int    ef_construction() const { return params_.ef_construction; }
    int    max_layer()       const { return entry_layer_; }

private:
    using MaxHeap = std::priority_queue<Neighbor>;
    using MinHeap = std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>>;

    // -------------------------------------------------------------------
    // Generation-counter visited table.
    //
    // Instead of a vector<bool> that must be cleared after every
    // search_layer call, we keep a vector<uint32_t> of "stamps" and a
    // monotonically increasing generation counter.  A node is considered
    // visited iff visited_gen[id] == current_gen.  Clearing is O(1):
    // just increment the counter.  Each search_layer call gets its own
    // generation so the table can be reused across calls within the same
    // insert/search without any per-element reset.
    // -------------------------------------------------------------------
    struct VisitedTable {
        std::vector<uint32_t> gen;
        uint32_t              cur = 0;

        void reset(size_t n) {
            if (gen.size() < n) gen.assign(n, 0);
            if (++cur == 0) { gen.assign(gen.size(), 0); cur = 1; }
        }
        bool visited(NodeId id) const {
            return id < gen.size() && gen[id] == cur;
        }
        void mark(NodeId id) {
            if (id >= gen.size()) gen.resize(id + 1, 0); // grow on demand
            gen[id] = cur;
        }
    };

    int   sample_level() const;
    int   sample_level_thread(int rng_slot) const;
    float dist(NodeId a, NodeId b)       const;
    float dist(const float* q, NodeId b) const;

    // Core insert used by both the public insert() and parallel workers.
    // rng_slot selects which per-thread RNG to use (0 = single-threaded path).
    NodeId insert_impl(const float* vec, int rng_slot);

    void insert_batch_parallel(const float* data, size_t n);

    MaxHeap search_layer(const float*               q,
                         const std::vector<NodeId>& entry_points,
                         int ef, int lc,
                         VisitedTable&              vt,
                         bool                       construction = false) const;

    std::vector<NodeId> select_neighbors_heuristic(
        const float* q, MaxHeap candidates, int M, int lc,
        bool extend_candidates = false, bool keep_pruned = true) const;

    void prune_connections(NodeId u, int lc, int max_conn);
    void repair_entry_point();

    // ---- Data ----
    Params    params_;
    RawDistFn dist_fn_;
    int       num_threads_;   // resolved from params_.num_threads at construction

    // Flat contiguous storage: data_[id * dim .. id * dim + dim - 1]
    std::vector<float>                 data_;
    std::vector<std::unique_ptr<Node>> nodes_;
    // uint8_t not bool: vector<bool> is bit-packed, meaning two threads touching
    // adjacent elements share the same word — a guaranteed data race under
    // parallel insert even with append_mtx_ protecting the push_back itself.
    // uint8_t gives each element its own byte; reads and writes are independent.
    std::vector<uint8_t>               deleted_;

    NodeId entry_point_ = INVALID_ID;
    int    entry_layer_ = -1;

    // Live (non-deleted) node count — updated atomically, avoids O(N) size().
    std::atomic<size_t> live_count_{0};

    // Total node count and committed deleted_ entries — both updated after
    // their respective push_backs complete, making them safe to read
    // without append_mtx_ from any thread.
    std::atomic<size_t> node_count_{0};
    std::atomic<size_t> deleted_committed_{0};

    // Two-level locking:
    //   append_mtx_  — exclusive when growing data_/nodes_/deleted_ (very brief)
    //   index_mtx_   — shared during graph traversal, exclusive only for
    //                  entry_point_ / entry_layer_ updates
    mutable std::mutex        append_mtx_;
    mutable std::shared_mutex index_mtx_;

    // Per-thread RNGs — eliminates the rng_mtx_ bottleneck under parallel inserts.
    // Indexed by the worker slot [0, num_threads_).  Single-threaded insert()
    // uses slot 0 (rng_pool_[0]).
    mutable std::vector<std::mt19937> rng_pool_;
    mutable std::mutex                rng_mtx_;   // guards rng_pool_[0] for non-batch inserts

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
}s
