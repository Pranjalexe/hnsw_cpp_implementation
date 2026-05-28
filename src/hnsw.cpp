#include "hnsw.hpp"
#include <cassert>
#include <stdexcept>
#include <cstring>
#include <unordered_set>

namespace hnsw {

// ============================================================================
// Constructor
// ============================================================================
Index::Index(Params p)
    : params_(std::move(p))
    , dist_fn_(params_.dist_fn
               ? (params_.dist_fn.target<RawDistFn>()
                  ? *params_.dist_fn.target<RawDistFn>()
                  : static_cast<RawDistFn>(l2_sq))
               : static_cast<RawDistFn>(l2_sq))
    , num_threads_(params_.num_threads > 0
                   ? params_.num_threads
                   : static_cast<int>(std::thread::hardware_concurrency()))
    , mL_(1.0 / std::log(static_cast<double>(params_.M)))
{
    // Seed one RNG per thread slot.  We use splitmix64-style seed spacing so
    // threads don't start with correlated sequences.
    std::random_device rd;
    uint64_t seed = (static_cast<uint64_t>(rd()) << 32) | rd();
    rng_pool_.reserve(num_threads_);
    for (int i = 0; i < num_threads_; ++i) {
        seed += 0x9e3779b97f4a7c15ULL; // splitmix step
        rng_pool_.emplace_back(static_cast<uint32_t>(seed >> 33));
    }
}

// ============================================================================
// Stats
// ============================================================================
size_t Index::size() const {
    return live_count_.load(std::memory_order_relaxed);
}

size_t Index::capacity() const {
    return node_count_.load(std::memory_order_relaxed);
}

bool Index::is_deleted(NodeId id) const {
    std::lock_guard<std::mutex> lg(append_mtx_);
    return id >= deleted_.size() || deleted_[id];
}

const float* Index::get_vector(NodeId id) const {
    std::lock_guard<std::mutex> lg(append_mtx_);
    if (id >= nodes_.size())
        throw std::out_of_range("hnsw::Index::get_vector: id out of range");
    if (deleted_[id])
        throw std::invalid_argument("hnsw::Index::get_vector: node is deleted");
    return data_.data() + id * params_.dim;
}

// ============================================================================
// Level sampling
// ============================================================================
int Index::sample_level() const {
    return sample_level_thread(0);
}

int Index::sample_level_thread(int slot) const {
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    if (slot == 0) {
        // Single-threaded path: guard with rng_mtx_
        std::lock_guard<std::mutex> lg(rng_mtx_);
        return static_cast<int>(-std::log(ud(rng_pool_[0])) * mL_);
    }
    // Parallel path: each slot is owned by exactly one thread — no lock needed.
    return static_cast<int>(-std::log(ud(rng_pool_[slot])) * mL_);
}

// ============================================================================
// Distance helpers
// ============================================================================
float Index::dist(NodeId a, NodeId b) const {
    return dist_fn_(data_.data() + a * params_.dim,
                    data_.data() + b * params_.dim,
                    params_.dim);
}

float Index::dist(const float* q, NodeId b) const {
    return dist_fn_(q, data_.data() + b * params_.dim, params_.dim);
}

// ============================================================================
// search_layer
//
// Two read paths for neighbor lists, selected by the `construction` flag:
//
// construction=false (pure search, default):
//   Reads neighbor lists directly without locking — no allocation, no copy.
//   Safe when no concurrent inserts are running. Matches hnswlib's approach
//   and gives ~30% better QPS over the locking version.
//
// construction=true (called during insert):
//   Snapshots each neighbor list under the per-node lock before iterating.
//   Required because push_back during neighbor wiring can reallocate the
//   underlying vector storage, making a held reference dangling.
// ============================================================================
Index::MaxHeap Index::search_layer(const float*               q,
                                   const std::vector<NodeId>& entry_points,
                                   int ef, int lc,
                                   VisitedTable&              vt,
                                   bool                       construction) const {
    const size_t nc  = node_count_.load(std::memory_order_acquire);
    const size_t del = deleted_committed_.load(std::memory_order_acquire);
    const size_t dim = params_.dim;

    vt.reset(nc);

    MaxHeap result;
    MinHeap candidates;

    for (NodeId ep : entry_points) {
        if (ep >= nc || vt.visited(ep)) continue;
        vt.mark(ep);
        if (ep < del && deleted_[ep]) continue;
        float d = dist(q, ep);
        result.push({ep, d});
        candidates.push({ep, d});
    }

    while (!candidates.empty()) {
        Neighbor cur = candidates.top(); candidates.pop();

        // Prefetch the next candidate's node and vector data while we
        // process the current one — hides pointer-chasing latency.
        if (!candidates.empty()) {
            NodeId next = candidates.top().id;
            if (next < nc) {
                __builtin_prefetch(nodes_[next].get(), 0, 1);
                const char* p = reinterpret_cast<const char*>(
                    data_.data() + next * dim);
                for (size_t off = 0; off < dim * sizeof(float); off += 64)
                    __builtin_prefetch(p + off, 0, 1);
            }
        }

        if (static_cast<int>(result.size()) >= ef &&
            cur.dist > result.top().dist)
            break;

        const auto& node = *nodes_[cur.id];
        if (!node.has_layer(lc)) continue;

        if (construction) {
            // Snapshot under lock: safe against concurrent push_back/resize
            // during neighbor wiring in the same or other threads.
            std::vector<NodeId> nbrs;
            {
                std::lock_guard<std::mutex> nlg(nodes_[cur.id]->mtx);
                if (nodes_[cur.id]->has_layer(lc))
                    nbrs = nodes_[cur.id]->neighbors[lc];
            }
            const int nbr_n = static_cast<int>(nbrs.size());
            for (int ni = 0; ni < nbr_n; ++ni) {
                // Prefetch next neighbor's vector while computing current distance.
                if (ni + 1 < nbr_n && nbrs[ni+1] < nc) {
                    const char* p = reinterpret_cast<const char*>(
                        data_.data() + nbrs[ni+1] * dim);
                    for (size_t off = 0; off < dim * sizeof(float); off += 64)
                        __builtin_prefetch(p + off, 0, 1);
                }
                const NodeId nb = nbrs[ni];
                if (nb >= nc || vt.visited(nb)) continue;
                vt.mark(nb);
                if (nb < del && deleted_[nb]) continue;
                float d_nb  = dist(q, nb);
                float worst = static_cast<int>(result.size()) >= ef
                              ? result.top().dist
                              : std::numeric_limits<float>::max();
                if (d_nb < worst) {
                    candidates.push({nb, d_nb});
                    result.push({nb, d_nb});
                    if (static_cast<int>(result.size()) > ef) result.pop();
                }
            }
        } else {
            // Direct read: no lock, no allocation. Safe for pure search.
            const auto& nbr_list = node.neighbors[lc];
            const int nbr_n = static_cast<int>(nbr_list.size());

            for (int ni = 0; ni < nbr_n; ++ni) {
                // Prefetch next neighbor's vector while computing current distance.
                // Locality hint=1 (L2): these will be reused across candidates.
                if (ni + 1 < nbr_n && nbr_list[ni+1] < nc) {
                    const char* p = reinterpret_cast<const char*>(
                        data_.data() + nbr_list[ni+1] * dim);
                    for (size_t off = 0; off < dim * sizeof(float); off += 64)
                        __builtin_prefetch(p + off, 0, 1);
                }
                const NodeId nb = nbr_list[ni];
                if (nb >= nc || vt.visited(nb)) continue;
                vt.mark(nb);
                if (nb < del && deleted_[nb]) continue;
                float d_nb  = dist(q, nb);
                float worst = static_cast<int>(result.size()) >= ef
                              ? result.top().dist
                              : std::numeric_limits<float>::max();
                if (d_nb < worst) {
                    candidates.push({nb, d_nb});
                    result.push({nb, d_nb});
                    if (static_cast<int>(result.size()) > ef) result.pop();
                }
            }
        }
    }

    return result;
}

// ============================================================================
// select_neighbors_heuristic  (Algorithm 4)
// ============================================================================
std::vector<NodeId> Index::select_neighbors_heuristic(
    const float* q, MaxHeap candidates, int M, int lc,
    bool extend_candidates, bool keep_pruned) const
{
    const size_t del = deleted_committed_.load(std::memory_order_acquire);

    std::vector<Neighbor> W_vec;
    W_vec.reserve(candidates.size());
    while (!candidates.empty()) { W_vec.push_back(candidates.top()); candidates.pop(); }
    std::reverse(W_vec.begin(), W_vec.end());

    if (extend_candidates) {
        std::unordered_set<NodeId> seen;
        for (auto& e : W_vec) seen.insert(e.id);

        std::vector<Neighbor> extras;
        for (auto& e : W_vec) {
            std::vector<NodeId> nbrs;
            {
                std::lock_guard<std::mutex> nlg(nodes_[e.id]->mtx);
                if (nodes_[e.id]->has_layer(lc))
                    nbrs = nodes_[e.id]->neighbors[lc];
            }
            for (NodeId nb : nbrs) {
                if (nb < del && deleted_[nb]) continue;
                if (!seen.insert(nb).second) continue;
                extras.push_back({nb, dist(q, nb)});
            }
        }
        std::sort(extras.begin(), extras.end());
        std::vector<Neighbor> merged;
        merged.reserve(W_vec.size() + extras.size());
        std::merge(W_vec.begin(), W_vec.end(),
                   extras.begin(), extras.end(),
                   std::back_inserter(merged));
        W_vec = std::move(merged);
    }

    std::vector<NodeId> R;
    std::vector<Neighbor> pruned;
    R.reserve(M);

    for (auto& e : W_vec) {
        if (!keep_pruned && static_cast<int>(R.size()) >= M) break;
        if (e.id < del && deleted_[e.id]) continue;

        bool good = true;
        for (NodeId r : R) {
            if (dist(e.id, r) < e.dist) { good = false; break; }
        }

        if (good) {
            if (static_cast<int>(R.size()) < M) R.push_back(e.id);
        } else if (keep_pruned) {
            pruned.push_back(e);
        }
    }

    if (keep_pruned) {
        for (auto& e : pruned) {
            if (static_cast<int>(R.size()) >= M) break;
            if (e.id >= del || !deleted_[e.id]) R.push_back(e.id);
        }
    }

    return R;
}

// ============================================================================
// prune_connections
// Called with nodes_[u]->mtx already held by the caller.
// Keeps the max_conn closest neighbors by distance (simple truncation).
// ============================================================================
void Index::prune_connections(NodeId u, int lc, int max_conn) {
    auto& nbrs = nodes_[u]->neighbors[lc];
    if (static_cast<int>(nbrs.size()) <= max_conn) return;

    const float* u_vec = data_.data() + u * params_.dim;
    const int sz = static_cast<int>(nbrs.size());

    std::vector<std::pair<float, NodeId>> scored(sz);
    for (int i = 0; i < sz; ++i)
        scored[i] = {dist(u_vec, nbrs[i]), nbrs[i]};
    std::sort(scored.begin(), scored.end());

    for (int i = 0; i < max_conn; ++i) nbrs[i] = scored[i].second;
    nbrs.resize(max_conn);
}

// ============================================================================
// insert_impl  — core of both insert() and parallel workers
//
// rng_slot:  which rng_pool_ entry to use.
//            0  → guarded by rng_mtx_  (single-threaded / public insert())
//            1+ → owned exclusively by one worker thread (no lock needed)
// ============================================================================
NodeId Index::insert_impl(const float* vec, int rng_slot) {
    const int new_level = sample_level_thread(rng_slot);
    NodeId new_id;

    {
        std::lock_guard<std::mutex> alock(append_mtx_);
        new_id = static_cast<NodeId>(nodes_.size());
        data_.insert(data_.end(), vec, vec + params_.dim);
        nodes_.push_back(std::make_unique<Node>(new_level, params_.M));
        deleted_.push_back(0);
    }
    // Publish node immediately after append so other threads can see it.
    // search_layer uses construction=true snapshot mode during insert, so
    // visiting a node with empty neighbor lists just finds no candidates — safe.
    node_count_.fetch_add(1, std::memory_order_release);
    deleted_committed_.fetch_add(1, std::memory_order_release);
    live_count_.fetch_add(1, std::memory_order_relaxed);

    NodeId ep; int cur_top;
    {
        std::shared_lock<std::shared_mutex> rlock(index_mtx_);
        ep = entry_point_; cur_top = entry_layer_;
    }

    if (ep == INVALID_ID) {
        std::unique_lock<std::shared_mutex> wlock(index_mtx_);
        if (entry_point_ == INVALID_ID) { entry_point_ = new_id; entry_layer_ = new_level; }
        return new_id;
    }

    VisitedTable vt;
    std::vector<NodeId> eps = {ep};

    // Greedy descent to new_level — ef=1 per layer
    for (int lc = cur_top; lc > new_level; --lc) {
        auto W = search_layer(vec, eps, 1, lc, vt, true);
        if (!W.empty()) eps = {W.top().id};
    }

    // Construction layers — find and wire neighbors
    for (int lc = std::min(new_level, cur_top); lc >= 0; --lc) {
        const int max_conn = (lc == 0) ? 2 * params_.M : params_.M;

        auto W = search_layer(vec, eps, params_.ef_construction, lc, vt, true);

        // Drain once into sorted vector; rebuild heap for select_neighbors_heuristic
        std::vector<Neighbor> sorted_vec;
        sorted_vec.reserve(W.size());
        while (!W.empty()) { sorted_vec.push_back(W.top()); W.pop(); }
        std::reverse(sorted_vec.begin(), sorted_vec.end());
        MaxHeap W_for_heuristic;
        for (auto& n : sorted_vec) W_for_heuristic.push(n);

        auto nbrs = select_neighbors_heuristic(vec, std::move(W_for_heuristic), max_conn, lc);

        {
            std::lock_guard<std::mutex> lg(nodes_[new_id]->mtx);
            nodes_[new_id]->neighbors[lc] = nbrs;
        }

        for (NodeId nb : nbrs) {
            std::lock_guard<std::mutex> lg(nodes_[nb]->mtx);
            if (nodes_[nb]->has_layer(lc)) {
                nodes_[nb]->neighbors[lc].push_back(new_id);
                prune_connections(nb, lc, max_conn);
            }
        }

        eps.clear();
        for (auto& n : sorted_vec) eps.push_back(n.id);
    }

    if (new_level > cur_top) {
        std::unique_lock<std::shared_mutex> wlock(index_mtx_);
        if (new_level > entry_layer_) { entry_point_ = new_id; entry_layer_ = new_level; }
    }

    return new_id;
}

// ============================================================================
// Public insert — thin wrapper around insert_impl using rng slot 0
// ============================================================================
NodeId Index::insert(const float* vec) {
    return insert_impl(vec, 0);
}

// ============================================================================
// insert_batch_parallel
//
// ID assignment must be deterministic (input index i → new_id = base + i) so
// that callers can predict which ID holds which vector.  We achieve this by
// pre-allocating all slots serially under append_mtx_ before launching workers.
// Workers then only do graph wiring — the expensive part — in parallel.
// ============================================================================
void Index::insert_batch_parallel(const float* data, size_t n) {
    const int T = std::min(num_threads_, static_cast<int>(n));
    if (T <= 1) {
        for (size_t i = 0; i < n; ++i) insert_impl(data + i * params_.dim, 0);
        return;
    }

    // ---- Phase 1: Pre-allocate all node slots in input order (serial, fast) ----
    // Under append_mtx_ we write each vector into data_, create the Node, and
    // set deleted_=0.  We do NOT do any graph wiring here.
    // Publishing node_count_ / deleted_committed_ happens after all slots are
    // allocated so workers start with a fully-visible node table.
    const NodeId base_id = static_cast<NodeId>(nodes_.size());
    {
        std::lock_guard<std::mutex> alock(append_mtx_);
        for (size_t i = 0; i < n; ++i) {
            const float* vec = data + i * params_.dim;
            data_.insert(data_.end(), vec, vec + params_.dim);
            // Level is sampled per-node; we need it now so the Node is
            // constructed with the right number of layers.
            // Use rng slot 0 (guarded by rng_mtx_ inside sample_level_thread).
            const int lv = sample_level_thread(0);
            nodes_.push_back(std::make_unique<Node>(lv, params_.M));
            deleted_.push_back(0);
        }
    }
    // Publish all n new nodes — node_count_ and deleted_committed_ together so
    // search_layer sees a consistent view during parallel construction.
    node_count_.fetch_add(n, std::memory_order_release);
    deleted_committed_.fetch_add(n, std::memory_order_release);
    live_count_.fetch_add(n, std::memory_order_relaxed);

    // ---- Phase 2: Bootstrap — wire the first chunk serially ----
    // ef_construction*2 nodes gives workers a well-connected backbone to
    // navigate into without serialising too much of the total work.
    // (With N=1M, T=8, ef_c=200: bootstrap=400 instead of 125,000 — huge difference.)
    const size_t bootstrap = std::min(n,
        static_cast<size_t>(params_.ef_construction) * 2);

    // Helper: drain a MaxHeap into a vector sorted ascending (closest-first).
    // Used for both entry-point selection and select_neighbors_heuristic input.
    // Builds a MaxHeap copy for select_neighbors_heuristic from the same data.
    auto drain_heap = [](MaxHeap& W,
                         std::vector<Neighbor>& sorted_out,
                         MaxHeap& heap_copy_out)
    {
        sorted_out.reserve(W.size());
        while (!W.empty()) { sorted_out.push_back(W.top()); W.pop(); }
        std::reverse(sorted_out.begin(), sorted_out.end()); // ascending
        for (auto& n : sorted_out) heap_copy_out.push(n);
    };

    // update_entry_point: helper used after each bootstrap insert
    auto wire_node = [&](NodeId new_id) {
        const float* vec = data_.data() + new_id * params_.dim;
        const int new_level = nodes_[new_id]->max_layer();

        NodeId ep; int cur_top;
        {
            std::shared_lock<std::shared_mutex> rlock(index_mtx_);
            ep = entry_point_; cur_top = entry_layer_;
        }

        if (ep == INVALID_ID) {
            std::unique_lock<std::shared_mutex> wlock(index_mtx_);
            if (entry_point_ == INVALID_ID) { entry_point_ = new_id; entry_layer_ = new_level; }
            return;
        }

        VisitedTable vt;
        std::vector<NodeId> eps = {ep};

        for (int lc = cur_top; lc > new_level; --lc) {
            auto W = search_layer(vec, eps, 1, lc, vt, true);
            if (!W.empty()) eps = {W.top().id};
        }

        for (int lc = std::min(new_level, cur_top); lc >= 0; --lc) {
            const int max_conn = (lc == 0) ? 2 * params_.M : params_.M;
            auto W = search_layer(vec, eps, params_.ef_construction, lc, vt, true);

            // Single drain: sorted_vec for entry points, heap_for_heuristic for selection
            std::vector<Neighbor> sorted_vec; MaxHeap heap_for_heuristic;
            drain_heap(W, sorted_vec, heap_for_heuristic);

            auto nbrs = select_neighbors_heuristic(vec, std::move(heap_for_heuristic), max_conn, lc);
            { std::lock_guard<std::mutex> lg(nodes_[new_id]->mtx); nodes_[new_id]->neighbors[lc] = nbrs; }

            for (NodeId nb : nbrs) {
                std::lock_guard<std::mutex> lg(nodes_[nb]->mtx);
                if (nodes_[nb]->has_layer(lc)) {
                    nodes_[nb]->neighbors[lc].push_back(new_id);
                    prune_connections(nb, lc, max_conn);
                }
            }
            eps.clear();
            for (auto& nn : sorted_vec) eps.push_back(nn.id);
        }

        if (new_level > cur_top) {
            std::unique_lock<std::shared_mutex> wlock(index_mtx_);
            if (new_level > entry_layer_) { entry_point_ = new_id; entry_layer_ = new_level; }
        }
    };

    for (size_t i = 0; i < bootstrap; ++i)
        wire_node(base_id + static_cast<NodeId>(i));

    const size_t remaining = n - bootstrap;
    if (remaining == 0) return;

    // ---- Phase 3: Parallel graph wiring for the rest ----
    const int slots = static_cast<int>(rng_pool_.size());
    std::vector<std::thread> workers;
    workers.reserve(T);

    const size_t chunk = (remaining + T - 1) / static_cast<size_t>(T);
    for (int t = 0; t < T; ++t) {
        const size_t start = bootstrap + static_cast<size_t>(t) * chunk;
        if (start >= n) break;
        const size_t end = std::min(start + chunk, n);
        const int slot   = (t + 1) % slots;

        workers.emplace_back([this, base_id, start, end, &drain_heap]() {
            for (size_t i = start; i < end; ++i) {
                NodeId new_id = base_id + static_cast<NodeId>(i);
                const float* vec = data_.data() + new_id * params_.dim;
                const int new_level = nodes_[new_id]->max_layer();

                NodeId ep; int cur_top;
                {
                    std::shared_lock<std::shared_mutex> rlock(index_mtx_);
                    ep = entry_point_; cur_top = entry_layer_;
                }
                if (ep == INVALID_ID) {
                    std::unique_lock<std::shared_mutex> wlock(index_mtx_);
                    if (entry_point_ == INVALID_ID) { entry_point_ = new_id; entry_layer_ = new_level; }
                    continue;
                }

                VisitedTable vt;
                std::vector<NodeId> eps = {ep};

                for (int lc = cur_top; lc > new_level; --lc) {
                    auto W = search_layer(vec, eps, 1, lc, vt, true);
                    if (!W.empty()) eps = {W.top().id};
                }

                for (int lc = std::min(new_level, cur_top); lc >= 0; --lc) {
                    const int max_conn = (lc == 0) ? 2 * params_.M : params_.M;
                    auto W = search_layer(vec, eps, params_.ef_construction, lc, vt, true);

                    std::vector<Neighbor> sorted_vec; MaxHeap heap_for_heuristic;
                    drain_heap(W, sorted_vec, heap_for_heuristic);

                    auto nbrs = select_neighbors_heuristic(vec, std::move(heap_for_heuristic), max_conn, lc);
                    { std::lock_guard<std::mutex> lg(nodes_[new_id]->mtx); nodes_[new_id]->neighbors[lc] = nbrs; }

                    for (NodeId nb : nbrs) {
                        std::lock_guard<std::mutex> lg(nodes_[nb]->mtx);
                        if (nodes_[nb]->has_layer(lc)) {
                            nodes_[nb]->neighbors[lc].push_back(new_id);
                            prune_connections(nb, lc, max_conn);
                        }
                    }
                    eps.clear();
                    for (auto& nn : sorted_vec) eps.push_back(nn.id);
                }

                if (new_level > cur_top) {
                    std::unique_lock<std::shared_mutex> wlock(index_mtx_);
                    if (new_level > entry_layer_) { entry_point_ = new_id; entry_layer_ = new_level; }
                }
            }
        });
    }
    for (auto& w : workers) w.join();
}

// ============================================================================
// insert_batch — pre-reserve then dispatch to parallel implementation
// ============================================================================
void Index::insert_batch(const float* data, size_t n) {
    {
        std::lock_guard<std::mutex> alock(append_mtx_);
        data_.reserve(data_.size() + n * params_.dim);
        nodes_.reserve(nodes_.size() + n);
        deleted_.reserve(deleted_.size() + n);
    }
    insert_batch_parallel(data, n);
}

// ============================================================================
// remove
// ============================================================================
bool Index::remove(NodeId id) {
    {
        std::lock_guard<std::mutex> alock(append_mtx_);
        if (id >= deleted_.size() || deleted_[id]) return false;
        deleted_[id] = true;
    }
    live_count_.fetch_sub(1, std::memory_order_relaxed);
    {
        std::shared_lock<std::shared_mutex> rlock(index_mtx_);
        if (entry_point_ == id) { rlock.unlock(); repair_entry_point(); }
    }
    return true;
}

void Index::repair_entry_point() {
    std::unique_lock<std::shared_mutex> wlock(index_mtx_);
    if (entry_point_ == INVALID_ID || !deleted_[entry_point_]) return;

    NodeId old_ep = entry_point_;
    entry_point_  = INVALID_ID;
    entry_layer_  = -1;

    for (int lc = nodes_[old_ep]->max_layer(); lc >= 0; --lc) {
        std::vector<NodeId> nbrs;
        { std::lock_guard<std::mutex> nlg(nodes_[old_ep]->mtx);
          if (nodes_[old_ep]->has_layer(lc)) nbrs = nodes_[old_ep]->neighbors[lc]; }
        for (NodeId nb : nbrs) {
            if (nb >= deleted_.size() || deleted_[nb]) continue;
            int lv = nodes_[nb]->max_layer();
            if (entry_point_ == INVALID_ID || lv > entry_layer_)
                { entry_point_ = nb; entry_layer_ = lv; }
        }
    }
    if (entry_point_ == INVALID_ID) {
        std::lock_guard<std::mutex> alock(append_mtx_);
        for (NodeId i = 0; i < static_cast<NodeId>(nodes_.size()); ++i) {
            if (deleted_[i]) continue;
            int lv = nodes_[i]->max_layer();
            if (entry_point_ == INVALID_ID || lv > entry_layer_)
                { entry_point_ = i; entry_layer_ = lv; }
        }
    }
}

// ============================================================================
// search  (Algorithm 5)
// ============================================================================
std::vector<Neighbor> Index::search(const float* query, int K, int ef) const {
    if (ef < 0) ef = params_.ef_search;
    ef = std::max(ef, K);

    NodeId ep; int top_layer;
    {
        std::shared_lock<std::shared_mutex> rlock(index_mtx_);
        ep = entry_point_; top_layer = entry_layer_;
    }
    if (ep == INVALID_ID) return {};

    VisitedTable vt;
    std::vector<NodeId> eps = {ep};

    for (int lc = top_layer; lc > 0; --lc) {
        auto W = search_layer(query, eps, 1, lc, vt);
        if (!W.empty()) eps = {W.top().id};
    }

    auto W = search_layer(query, eps, ef, 0, vt);

    // Snapshot deleted_ state once for the result drain.
    std::vector<uint8_t> del_snap;
    {
        std::lock_guard<std::mutex> alock(append_mtx_);
        del_snap = deleted_;
    }

    std::vector<Neighbor> all;
    all.reserve(W.size());
    while (!W.empty()) {
        Neighbor n = W.top(); W.pop();
        if (n.id < del_snap.size() && !del_snap[n.id]) all.push_back(n);
    }
    std::sort(all.begin(), all.end());
    if (static_cast<int>(all.size()) > K) all.resize(K);
    return all;
}

// ============================================================================
// save / load
// Layout: magic(4) version(4) n(8) dim(8) M(4) ef_c(4) ef_s(4)
//         ep(4) el(4)
//         [per node: deleted(1) max_layer(4)
//           [per layer: degree(4) neighbors(degree*4)]]
//         flat vector data (n * dim * 4 bytes) at end
// ============================================================================
namespace {
    template<typename T> void wpod(std::ofstream& f, const T& v)
        { f.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
    template<typename T> void rpod(std::ifstream& f, T& v) {
        f.read(reinterpret_cast<char*>(&v), sizeof(T));
        if (!f) throw std::runtime_error("hnsw: unexpected end of file");
    }
}

void Index::save(const std::string& path) const {
    // Snapshot everything under both locks to get a consistent view.
    std::lock_guard<std::mutex>           alock(append_mtx_);
    std::shared_lock<std::shared_mutex>   rlock(index_mtx_);

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("hnsw::save: cannot open " + path);

    const uint64_t n = nodes_.size(), dim = params_.dim;
    wpod(f, MAGIC); wpod(f, VERSION);
    wpod(f, n); wpod(f, dim);
    wpod(f, params_.M); wpod(f, params_.ef_construction); wpod(f, params_.ef_search);
    wpod(f, entry_point_); wpod(f, entry_layer_);

    for (uint64_t i = 0; i < n; ++i) {
        wpod(f, static_cast<uint8_t>(deleted_[i] ? 1 : 0));
        const int32_t ml = nodes_[i]->max_layer();
        wpod(f, ml);
        for (int lc = 0; lc <= ml; ++lc) {
            std::vector<NodeId> nbrs;
            { std::lock_guard<std::mutex> nlg(nodes_[i]->mtx); nbrs = nodes_[i]->neighbors[lc]; }
            const uint32_t deg = static_cast<uint32_t>(nbrs.size());
            wpod(f, deg);
            for (NodeId nb : nbrs) wpod(f, nb);
        }
    }
    f.write(reinterpret_cast<const char*>(data_.data()),
            static_cast<std::streamsize>(n * dim * sizeof(float)));
    if (!f) throw std::runtime_error("hnsw::save: write error");
}

void Index::load(const std::string& path) {
    std::lock_guard<std::mutex>         alock(append_mtx_);
    std::unique_lock<std::shared_mutex> wlock(index_mtx_);

    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("hnsw::load: cannot open " + path);

    uint32_t magic, version; rpod(f, magic); rpod(f, version);
    if (magic != MAGIC) throw std::runtime_error("hnsw::load: bad magic");
    if (version != VERSION) throw std::runtime_error("hnsw::load: unsupported version");

    uint64_t n, dim; rpod(f, n); rpod(f, dim);
    int32_t M, ef_c, ef_s; rpod(f, M); rpod(f, ef_c); rpod(f, ef_s);
    NodeId ep; int32_t el; rpod(f, ep); rpod(f, el);

    params_.dim = dim; params_.M = M;
    params_.ef_construction = ef_c; params_.ef_search = ef_s;
    dist_fn_ = params_.dist_fn
               ? (params_.dist_fn.target<RawDistFn>()
                  ? *params_.dist_fn.target<RawDistFn>()
                  : static_cast<RawDistFn>(l2_sq))
               : static_cast<RawDistFn>(l2_sq);
    mL_ = 1.0 / std::log(static_cast<double>(M));

    nodes_.clear(); deleted_.clear(); data_.clear();
    nodes_.reserve(n); deleted_.reserve(n);
    entry_point_ = ep; entry_layer_ = static_cast<int>(el);

    for (uint64_t i = 0; i < n; ++i) {
        uint8_t del; rpod(f, del);
        deleted_.push_back(del != 0);
        int32_t ml; rpod(f, ml);
        nodes_.push_back(std::make_unique<Node>(static_cast<int>(ml), M));
        for (int32_t lc = 0; lc <= ml; ++lc) {
            uint32_t deg; rpod(f, deg);
            auto& nbrs = nodes_[i]->neighbors[static_cast<size_t>(lc)];
            nbrs.resize(deg);
            for (uint32_t j = 0; j < deg; ++j) rpod(f, nbrs[j]);
        }
    }
    data_.resize(n * dim);
    f.read(reinterpret_cast<char*>(data_.data()),
           static_cast<std::streamsize>(n * dim * sizeof(float)));
    if (!f) throw std::runtime_error("hnsw::load: read error on vector data");

    size_t live = 0;
    for (uint8_t d : deleted_) if (!d) ++live;
    live_count_.store(live, std::memory_order_relaxed);
    node_count_.store(nodes_.size(), std::memory_order_release);
    deleted_committed_.store(deleted_.size(), std::memory_order_release);
}

} // namespace hnsw
