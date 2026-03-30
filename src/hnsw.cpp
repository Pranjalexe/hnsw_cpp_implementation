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
    , rng_(std::random_device{}())
    , mL_(1.0 / std::log(static_cast<double>(params_.M)))
{}

// ============================================================================
// Stats
// ============================================================================
size_t Index::size() const {
    std::shared_lock<std::shared_mutex> rlock(index_mtx_);
    size_t c = 0;
    for (bool d : deleted_) if (!d) ++c;
    return c;
}

size_t Index::capacity() const {
    std::shared_lock<std::shared_mutex> rlock(index_mtx_);
    return nodes_.size();
}

bool Index::is_deleted(NodeId id) const {
    std::shared_lock<std::shared_mutex> rlock(index_mtx_);
    return id >= deleted_.size() || deleted_[id];
}

const float* Index::get_vector(NodeId id) const {
    std::shared_lock<std::shared_mutex> rlock(index_mtx_);
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
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    std::lock_guard<std::mutex> lg(rng_mtx_);
    return static_cast<int>(-std::log(ud(rng_)) * mL_);
}

// ============================================================================
// Distance helpers — index directly into the flat data_ array
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
// Uses a caller-supplied std::vector<bool> as the visited array.
// The vector is sized to the current graph on first use and reused across
// calls within the same insert/search to avoid repeated allocation.
// After each call we clear only the entries we set (O(visited) not O(N)).
// ============================================================================
Index::MaxHeap Index::search_layer(const float*               q,
                                   const std::vector<NodeId>& entry_points,
                                   int ef, int lc,
                                   std::vector<bool>&         visited_buf) const {
    // Grow the visited buffer if the index has grown since last call.
    if (visited_buf.size() < nodes_.size())
        visited_buf.assign(nodes_.size(), false);

    MaxHeap result;
    MinHeap candidates;

    std::vector<NodeId> to_clear; // nodes marked visited this call
    to_clear.reserve(ef * static_cast<size_t>(params_.M));

    auto visit = [&](NodeId id) -> bool {
        if (visited_buf[id]) return false;
        visited_buf[id] = true;
        to_clear.push_back(id);
        return true;
    };

    for (NodeId ep : entry_points) {
        if (!visit(ep)) continue;
        float d = dist(q, ep);
        result.push({ep, d});
        candidates.push({ep, d});
    }

    while (!candidates.empty()) {
        Neighbor cur = candidates.top(); candidates.pop();

        if (static_cast<int>(result.size()) >= ef &&
            cur.dist > result.top().dist)
            break;

        if (!nodes_[cur.id]->has_layer(lc)) continue;
        const auto& nbrs = nodes_[cur.id]->neighbors[lc];

        for (NodeId nb : nbrs) {
            if (!visit(nb)) continue;
            if (deleted_[nb])  continue;

            float d_nb  = dist(q, nb);
            float worst = static_cast<int>(result.size()) >= ef
                          ? result.top().dist
                          : std::numeric_limits<float>::max();

            if (d_nb < worst) {
                candidates.push({nb, d_nb});
                result.push({nb, d_nb});
                if (static_cast<int>(result.size()) > ef)
                    result.pop();
            }
        }
    }

    // Reset only the entries we touched — O(visited) not O(N)
    for (NodeId id : to_clear) visited_buf[id] = false;

    return result;
}

// ============================================================================
// select_neighbors_heuristic  (Algorithm 4)
// ============================================================================
std::vector<NodeId> Index::select_neighbors_heuristic(
    const float* q, MaxHeap candidates, int M, int lc,
    bool extend_candidates, bool keep_pruned) const
{
    MinHeap W;
    while (!candidates.empty()) { W.push(candidates.top()); candidates.pop(); }

    if (extend_candidates) {
        std::unordered_set<NodeId> seen;
        std::vector<Neighbor> init;
        { MinHeap tmp = W; while (!tmp.empty()) { init.push_back(tmp.top()); tmp.pop(); } }
        for (auto& e : init) {
            if (!nodes_[e.id]->has_layer(lc)) continue;
            for (NodeId nb : nodes_[e.id]->neighbors[lc])
                if (nb < deleted_.size() && !deleted_[nb] && seen.insert(nb).second)
                    W.push({nb, dist(q, nb)});
        }
    }

    std::vector<NodeId> R;
    MinHeap             Wd;
    R.reserve(M);

    while (!W.empty() && static_cast<int>(R.size()) < M) {
        Neighbor e = W.top(); W.pop();
        if (e.id < deleted_.size() && deleted_[e.id]) continue;

        bool good = true;
        for (NodeId r : R)
            if (dist(e.id, r) < e.dist) { good = false; break; }

        if (good)             R.push_back(e.id);
        else if (keep_pruned) Wd.push(e);
    }

    if (keep_pruned)
        while (!Wd.empty() && static_cast<int>(R.size()) < M) {
            NodeId id = Wd.top().id; Wd.pop();
            if (id < deleted_.size() && !deleted_[id]) R.push_back(id);
        }

    return R;
}

// ============================================================================
// prune_connections
// ============================================================================
void Index::prune_connections(NodeId u, int lc, int max_conn) {
    auto& nbrs = nodes_[u]->neighbors[lc];
    if (static_cast<int>(nbrs.size()) <= max_conn) return;

    const float* u_vec = data_.data() + u * params_.dim;
    const int sz = static_cast<int>(nbrs.size());

    // Precompute distances once, then sort — avoids recomputing in comparator
    std::vector<std::pair<float, NodeId>> scored(sz);
    for (int i = 0; i < sz; ++i)
        scored[i] = {dist(u_vec, nbrs[i]), nbrs[i]};
    std::sort(scored.begin(), scored.end());

    for (int i = 0; i < max_conn; ++i) nbrs[i] = scored[i].second;
    nbrs.resize(max_conn);
}

// ============================================================================
// insert  (Algorithm 1)
// ============================================================================
NodeId Index::insert(const float* vec) {
    int    new_level = sample_level();
    NodeId new_id;

    {
        std::unique_lock<std::shared_mutex> wlock(index_mtx_);
        new_id = static_cast<NodeId>(nodes_.size());
        // Append to flat array
        data_.insert(data_.end(), vec, vec + params_.dim);
        nodes_.push_back(std::make_unique<Node>(new_level, params_.M));
        deleted_.push_back(false);
    }

    NodeId ep; int cur_top;
    {
        std::shared_lock<std::shared_mutex> rlock(index_mtx_);
        ep      = entry_point_;
        cur_top = entry_layer_;
    }

    if (ep == INVALID_ID) {
        std::unique_lock<std::shared_mutex> wlock(index_mtx_);
        if (entry_point_ == INVALID_ID) { entry_point_ = new_id; entry_layer_ = new_level; }
        return new_id;
    }

    // Shared visited buffer — reused across all search_layer calls in this insert
    std::vector<bool> visited_buf;

    std::vector<NodeId> eps = {ep};
    {
        std::shared_lock<std::shared_mutex> rlock(index_mtx_);
        for (int lc = cur_top; lc > new_level; --lc) {
            auto W = search_layer(vec, eps, 1, lc, visited_buf);
            eps = {W.top().id};
        }
    }

    {
        std::shared_lock<std::shared_mutex> rlock(index_mtx_);
        for (int lc = std::min(new_level, cur_top); lc >= 0; --lc) {
            const int max_conn = (lc == 0) ? 2 * params_.M : params_.M;

            auto W = search_layer(vec, eps, params_.ef_construction, lc, visited_buf);

            // Snapshot W before handing to select_neighbors (which drains it)
            std::vector<Neighbor> W_snap;
            { MaxHeap tmp = W; while (!tmp.empty()) { W_snap.push_back(tmp.top()); tmp.pop(); } }
            std::sort(W_snap.begin(), W_snap.end());

            auto nbrs = select_neighbors_heuristic(vec, std::move(W), max_conn, lc);

            { std::lock_guard<std::mutex> lg(nodes_[new_id]->mtx); nodes_[new_id]->neighbors[lc] = nbrs; }

            for (NodeId nb : nbrs) {
                std::lock_guard<std::mutex> lg(nodes_[nb]->mtx);
                if (nodes_[nb]->has_layer(lc))
                    nodes_[nb]->neighbors[lc].push_back(new_id);
                prune_connections(nb, lc, max_conn);
            }

            eps.clear();
            for (auto& n : W_snap) eps.push_back(n.id);
        }
    }

    if (new_level > cur_top) {
        std::unique_lock<std::shared_mutex> wlock(index_mtx_);
        if (new_level > entry_layer_) { entry_point_ = new_id; entry_layer_ = new_level; }
    }

    return new_id;
}

void Index::insert_batch(const float* data, size_t n) {
    for (size_t i = 0; i < n; ++i) insert(data + i * params_.dim);
}

// ============================================================================
// remove
// ============================================================================
bool Index::remove(NodeId id) {
    {
        std::unique_lock<std::shared_mutex> wlock(index_mtx_);
        if (id >= deleted_.size() || deleted_[id]) return false;
        deleted_[id] = true;
    }
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

    std::shared_lock<std::shared_mutex> rlock(index_mtx_);

    std::vector<bool> visited_buf; // reused across all search_layer calls
    std::vector<NodeId> eps = {ep};

    for (int lc = top_layer; lc > 0; --lc) {
        auto W = search_layer(query, eps, 1, lc, visited_buf);
        eps = {W.top().id};
    }

    auto W = search_layer(query, eps, ef, 0, visited_buf);

    std::vector<Neighbor> all;
    all.reserve(W.size());
    while (!W.empty()) {
        Neighbor n = W.top(); W.pop();
        if (!deleted_[n.id]) all.push_back(n);
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
    std::shared_lock<std::shared_mutex> rlock(index_mtx_);
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
    // Write all vector data as one flat block
    f.write(reinterpret_cast<const char*>(data_.data()),
            static_cast<std::streamsize>(n * dim * sizeof(float)));
    if (!f) throw std::runtime_error("hnsw::save: write error");
}

void Index::load(const std::string& path) {
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
    // Read flat vector block
    data_.resize(n * dim);
    f.read(reinterpret_cast<char*>(data_.data()),
           static_cast<std::streamsize>(n * dim * sizeof(float)));
    if (!f) throw std::runtime_error("hnsw::load: read error on vector data");
}

} // namespace hnsw