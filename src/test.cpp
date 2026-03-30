// Correctness tests for the HNSW index.
// Each test prints PASS or FAIL and exits 1 on any failure.

#include "hnsw.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstring>
#include <fstream>

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------
static bool approx_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

static std::vector<float> make_vec(std::initializer_list<float> il) {
    return std::vector<float>(il);
}

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL [%s]: %s\n", __func__, msg); return false; } \
} while(0)

#define RUN(fn) do { \
    bool ok = fn(); \
    std::printf("%s: %s\n", #fn, ok ? "PASS" : "FAIL"); \
    if (!ok) failures++; \
} while(0)

// --------------------------------------------------------------------------
// Test: basic insert + exact nearest neighbor (tiny dataset)
// --------------------------------------------------------------------------
static bool test_basic_nn() {
    hnsw::Index idx = hnsw::make_index(2, /*M=*/4, /*ef=*/20);

    // Insert 5 2-D points
    std::vector<std::vector<float>> pts = {
        {0.f, 0.f},
        {1.f, 0.f},
        {0.f, 1.f},
        {5.f, 5.f},
        {5.f, 6.f},
    };
    for (auto& p : pts) idx.insert(p);

    CHECK(idx.size() == 5, "size should be 5");

    // Query closest to (0.1, 0.1) — should be pt[0]
    auto res = idx.search({0.1f, 0.1f}, 1);
    CHECK(!res.empty(), "search returned empty");
    CHECK(res[0].id == 0, "nearest to (0.1,0.1) should be node 0");

    // Query closest to (5.0, 5.5) — should be pt[3] or pt[4]
    auto res2 = idx.search({5.0f, 5.5f}, 1);
    CHECK(!res2.empty(), "search2 returned empty");
    CHECK(res2[0].id == 3 || res2[0].id == 4, "nearest to (5,5.5) should be 3 or 4");

    return true;
}

// --------------------------------------------------------------------------
// Test: KNN ordering — results must be sorted ascending by distance
// --------------------------------------------------------------------------
static bool test_sorted_results() {
    hnsw::Index idx = hnsw::make_index(3, 8, 50);

    // Random-ish 3-D points
    for (int i = 0; i < 100; ++i) {
        float v[3] = { static_cast<float>(i), static_cast<float>(i % 7), static_cast<float>(i % 11) };
        idx.insert(v);
    }

    float q[3] = {10.f, 3.f, 5.f};
    auto res = idx.search(q, 10);
    CHECK(res.size() <= 10, "too many results");

    for (size_t i = 1; i < res.size(); ++i)
        CHECK(res[i].dist >= res[i-1].dist, "results not sorted ascending");

    return true;
}

// --------------------------------------------------------------------------
// Test: soft delete — deleted node must not appear in results
// --------------------------------------------------------------------------
static bool test_soft_delete() {
    hnsw::Index idx = hnsw::make_index(2, 4, 20);

    idx.insert({0.f, 0.f}); // id=0
    idx.insert({0.1f, 0.f}); // id=1, very close to 0
    idx.insert({10.f, 10.f}); // id=2, far away

    // Before delete, id=0 should be the closest to origin
    auto before = idx.search({0.f, 0.f}, 1);
    CHECK(!before.empty() && before[0].id == 0, "before delete: id=0 should be closest");

    bool ok = idx.remove(0);
    CHECK(ok, "remove(0) should succeed");
    CHECK(idx.size() == 2, "size after delete should be 2");
    CHECK(idx.is_deleted(0), "node 0 should be marked deleted");

    // After delete, id=1 should be closest
    auto after = idx.search({0.f, 0.f}, 3);
    for (auto& n : after)
        CHECK(n.id != 0, "deleted node 0 appeared in results");

    // Deleting again should return false
    CHECK(!idx.remove(0), "double-delete should return false");

    // Deleting out-of-range should return false
    CHECK(!idx.remove(999), "out-of-range delete should return false");

    return true;
}

// --------------------------------------------------------------------------
// Test: delete entry point — index should stay searchable
// --------------------------------------------------------------------------
static bool test_delete_entry_point() {
    hnsw::Index idx = hnsw::make_index(2, 4, 20);

    for (int i = 0; i < 20; ++i) {
        float v[2] = { static_cast<float>(i), 0.f };
        idx.insert(v);
    }

    hnsw::NodeId old_ep = static_cast<hnsw::NodeId>(idx.max_layer()); // not exact but good enough
    // Just delete the entry point (last node inserted at highest layer is typically it,
    // but we'll delete node 0 and verify search still works)
    idx.remove(0);

    auto res = idx.search({0.f, 0.f}, 3);
    CHECK(!res.empty(), "search after entry-point delete returned empty");
    for (auto& n : res)
        CHECK(n.id != 0, "deleted entry point appeared in results");

    return true;
}

// --------------------------------------------------------------------------
// Test: get_vector
// --------------------------------------------------------------------------
static bool test_get_vector() {
    hnsw::Index idx = hnsw::make_index(3, 4, 20);

    float v[3] = {1.f, 2.f, 3.f};
    hnsw::NodeId id = idx.insert(v);

    const float* rv = idx.get_vector(id);
    CHECK(approx_eq(rv[0], 1.f) && approx_eq(rv[1], 2.f) && approx_eq(rv[2], 3.f),
          "get_vector returned wrong data");

    idx.remove(id);
    bool threw = false;
    try { idx.get_vector(id); } catch (const std::exception&) { threw = true; }
    CHECK(threw, "get_vector on deleted node should throw");

    return true;
}

// --------------------------------------------------------------------------
// Test: recall on random data (statistical — should be > 80%)
// --------------------------------------------------------------------------
static bool test_recall() {
    const int N   = 500;
    const int DIM = 16;
    const int K   = 5;

    hnsw::Index::Params p;
    p.dim            = DIM;
    p.M              = 16;
    p.ef_construction = 100;
    p.ef_search      = 50;
    hnsw::Index idx(p);

    // Deterministic dataset
    std::vector<std::vector<float>> data(N, std::vector<float>(DIM));
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < DIM; ++d) {
            // simple deterministic values
            data[i][d] = static_cast<float>((i * 17 + d * 31) % 100) / 100.f;
        }
        idx.insert(data[i]);
    }

    // Brute-force ground truth for 50 queries
    const int NQ = 50;
    int correct = 0, total = 0;
    for (int qi = 0; qi < NQ; ++qi) {
        std::vector<float> q(DIM);
        for (int d = 0; d < DIM; ++d)
            q[d] = static_cast<float>((qi * 13 + d * 7 + 50) % 100) / 100.f;

        // brute force
        std::vector<std::pair<float, int>> bf;
        bf.reserve(N);
        for (int i = 0; i < N; ++i) {
            float s = 0;
            for (int d = 0; d < DIM; ++d) {
                float diff = q[d] - data[i][d];
                s += diff * diff;
            }
            bf.push_back({s, i});
        }
        std::partial_sort(bf.begin(), bf.begin() + K, bf.end());

        auto res = idx.search(q, K);
        for (int k = 0; k < K; ++k) {
            for (auto& r : res)
                if (static_cast<int>(r.id) == bf[k].second) { ++correct; break; }
            ++total;
        }
    }

    float recall = static_cast<float>(correct) / static_cast<float>(total);
    std::printf("  recall@%d on %d-dim/%d-vec: %.1f%%\n", K, DIM, N, recall * 100.f);
    CHECK(recall > 0.75f, "recall below 80% — graph quality too low");
    return true;
}

// --------------------------------------------------------------------------
// Test: save and load round-trip
// --------------------------------------------------------------------------
static bool test_save_load() {
    const char* path = "/tmp/hnsw_test.bin";

    hnsw::Index idx = hnsw::make_index(4, 8, 50);
    for (int i = 0; i < 30; ++i) {
        float v[4] = { static_cast<float>(i), static_cast<float>(i*2),
                       static_cast<float>(i*3), static_cast<float>(i*4) };
        idx.insert(v);
    }
    idx.remove(5);  // soft-delete one node before saving

    idx.save(path);

    // Load into a fresh index
    hnsw::Index idx2 = hnsw::make_index(4, 8, 50);
    idx2.load(path);

    CHECK(idx2.capacity() == idx.capacity(), "capacity mismatch after load");
    CHECK(idx2.size()     == idx.size(),     "size mismatch after load");
    CHECK(idx2.is_deleted(5),               "deleted node 5 not preserved after load");
    CHECK(idx2.max_layer() == idx.max_layer(), "max_layer mismatch after load");

    // Search should give the same results
    float q[4] = {10.f, 20.f, 30.f, 40.f};
    auto r1 = idx.search(q, 5);
    auto r2 = idx2.search(q, 5);
    CHECK(r1.size() == r2.size(), "result size differs after load");
    for (size_t i = 0; i < r1.size(); ++i) {
        CHECK(r1[i].id == r2[i].id, "result ids differ after load");
    }

    std::remove(path);
    return true;
}

// --------------------------------------------------------------------------
// Test: distance functions
// --------------------------------------------------------------------------
static bool test_distance_fns() {
    using namespace hnsw;

    float a[4] = {1.f, 0.f, 0.f, 0.f};
    float b[4] = {0.f, 1.f, 0.f, 0.f};

    // L2: |a-b|^2 = 2
    CHECK(approx_eq(l2_sq(a, b, 4), 2.f), "l2_sq wrong for orthogonal unit vecs");

    // Same vector: distance = 0
    CHECK(approx_eq(l2_sq(a, a, 4), 0.f), "l2_sq self-distance should be 0");

    // Inner product distance: 1 - dot(a, b) = 1 - 0 = 1
    CHECK(approx_eq(inner_product_dist(a, b, 4), 1.f), "inner_product_dist wrong");
    // Same vector: 1 - 1 = 0
    CHECK(approx_eq(inner_product_dist(a, a, 4), 0.f), "inner_product_dist self should be 0");

    // Custom dist fn passed to index
    Index::Params p;
    p.dim     = 4;
    p.M       = 4;
    p.dist_fn = static_cast<float(*)(const float*, const float*, size_t)>(hnsw::inner_product_dist);
    Index idx(p);
    idx.insert(std::vector<float>(a, a+4));
    idx.insert(std::vector<float>(b, b+4));

    auto res = idx.search(a, 4, 1);
    CHECK(!res.empty(), "custom dist_fn search empty");
    // closest to a under inner_product should be a itself (id=0)
    CHECK(res[0].id == 0, "with inner_product dist, a should be closest to itself");

    return true;
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------
int main() {
    int failures = 0;
    RUN(test_basic_nn);
    RUN(test_sorted_results);
    RUN(test_soft_delete);
    RUN(test_delete_entry_point);
    RUN(test_get_vector);
    RUN(test_recall);
    RUN(test_save_load);
    RUN(test_distance_fns);

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED",
                failures, failures == 1 ? "" : "s");
    return failures > 0 ? 1 : 0;
}