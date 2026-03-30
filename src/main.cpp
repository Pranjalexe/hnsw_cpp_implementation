// HNSW Benchmark Harness
// Measures build time, query latency, throughput, and recall@K
// against a synthetic dataset of random unit vectors.
//
// Usage:
//   ./bench                     # quick: 50k vectors, 128-dim
//   ./bench --full              # full:  100k / 500k / 1M vectors

#include "hnsw.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;
using Us    = std::chrono::duration<double, std::micro>;

// -------------------------------------------------------------------------
// Dataset generation: random unit vectors in R^dim
// -------------------------------------------------------------------------
static std::vector<float> make_dataset(size_t n, size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> data(n * dim);
    for (size_t i = 0; i < n; ++i) {
        float norm = 0.f;
        for (size_t d = 0; d < dim; ++d) {
            float v = nd(rng);
            data[i * dim + d] = v;
            norm += v * v;
        }
        norm = std::sqrt(norm);
        for (size_t d = 0; d < dim; ++d)
            data[i * dim + d] /= norm;
    }
    return data;
}

// -------------------------------------------------------------------------
// Brute-force exact KNN (ground truth)
// -------------------------------------------------------------------------
static std::vector<std::vector<uint32_t>>
exact_knn(const float* base, size_t nb, size_t dim,
          const float* queries, size_t nq, int K)
{
    std::vector<std::vector<uint32_t>> gt(nq, std::vector<uint32_t>(K));
    for (size_t qi = 0; qi < nq; ++qi) {
        const float* q = queries + qi * dim;
        std::vector<std::pair<float, uint32_t>> dists(nb);
        for (size_t i = 0; i < nb; ++i) {
            float s = 0.f;
            for (size_t d = 0; d < dim; ++d) {
                float diff = q[d] - base[i * dim + d];
                s += diff * diff;
            }
            dists[i] = {s, static_cast<uint32_t>(i)};
        }
        std::partial_sort(dists.begin(), dists.begin() + K, dists.end());
        for (int k = 0; k < K; ++k)
            gt[qi][k] = dists[k].second;
    }
    return gt;
}

// -------------------------------------------------------------------------
// Recall@K
// -------------------------------------------------------------------------
static float recall_at_k(
    const std::vector<std::vector<uint32_t>>& gt,
    const std::vector<std::vector<hnsw::Neighbor>>& results,
    int K)
{
    size_t correct = 0;
    const size_t nq = gt.size();
    for (size_t qi = 0; qi < nq; ++qi) {
        for (int k = 0; k < K; ++k) {
            for (auto& r : results[qi]) {
                if (r.id == gt[qi][k]) { ++correct; break; }
            }
        }
    }
    return static_cast<float>(correct) / static_cast<float>(nq * K);
}

// -------------------------------------------------------------------------
// Single benchmark run
// -------------------------------------------------------------------------
struct Result {
    size_t n, dim;
    int    M, ef_construction, ef_search;
    double build_ms;
    double inserts_per_sec;
    double p50_us, p95_us, p99_us;  // per-query latency percentiles
    double qps;
    float  recall;
};

static Result run(size_t n, size_t dim, int M, int ef_c, int ef_s,
                  size_t nq = 1000, int K = 10)
{
    auto base    = make_dataset(n,  dim, 42);
    auto queries = make_dataset(nq, dim, 99);

    // --- Build ---
    hnsw::Index::Params p;
    p.dim             = dim;
    p.M               = M;
    p.ef_construction = ef_c;
    p.ef_search       = ef_s;
    hnsw::Index idx(p);

    auto t0 = Clock::now();
    idx.insert_batch(base.data(), n);
    auto t1 = Clock::now();
    double build_ms = Ms(t1 - t0).count();

    // --- Ground truth (brute force) ---
    auto gt = exact_knn(base.data(), n, dim, queries.data(), nq, K);

    // --- Per-query latency ---
    std::vector<double> latencies(nq);
    std::vector<std::vector<hnsw::Neighbor>> results(nq);

    for (size_t qi = 0; qi < nq; ++qi) {
        auto ts = Clock::now();
        results[qi] = idx.search(queries.data() + qi * dim, K, ef_s);
        auto te = Clock::now();
        latencies[qi] = Us(te - ts).count();
    }

    std::sort(latencies.begin(), latencies.end());
    double p50 = latencies[static_cast<size_t>(nq * 0.50)];
    double p95 = latencies[static_cast<size_t>(nq * 0.95)];
    double p99 = latencies[static_cast<size_t>(nq * 0.99)];

    double total_search_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / 1000.0;

    Result r;
    r.n                = n;
    r.dim              = dim;
    r.M                = M;
    r.ef_construction  = ef_c;
    r.ef_search        = ef_s;
    r.build_ms         = build_ms;
    r.inserts_per_sec  = n / (build_ms / 1000.0);
    r.p50_us           = p50;
    r.p95_us           = p95;
    r.p99_us           = p99;
    r.qps              = nq / (total_search_ms / 1000.0);
    r.recall           = recall_at_k(gt, results, K);
    return r;
}

// -------------------------------------------------------------------------
// Printing
// -------------------------------------------------------------------------
static void print_header() {
    std::printf("\n%-8s %-5s %-5s %-6s %-6s  %-10s %-12s  %-8s %-8s %-8s  %-8s  %s\n",
                "N", "dim", "M", "ef_c", "ef_s",
                "build(ms)", "inserts/sec",
                "p50(us)", "p95(us)", "p99(us)",
                "QPS", "recall@10");
    std::printf("%s\n", std::string(110, '-').c_str());
}

static void print_row(const Result& r) {
    std::printf("%-8zu %-5zu %-5d %-6d %-6d  %-10.0f %-12.0f  %-8.1f %-8.1f %-8.1f  %-8.0f  %.4f\n",
                r.n, r.dim, r.M, r.ef_construction, r.ef_search,
                r.build_ms, r.inserts_per_sec,
                r.p50_us, r.p95_us, r.p99_us,
                r.qps, r.recall);
    std::fflush(stdout);
}

static void section(const char* title) {
    std::printf("\n=== %s ===\n", title);
    print_header();
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------
int main(int argc, char** argv) {
    bool full = (argc > 1 && std::string(argv[1]) == "--full");

    std::printf("HNSW Benchmark  |  dataset: random unit vectors, K=10, nq=1000\n");

    if (!full) {
        // ---- Quick run: 50k vectors, vary M and ef_search ----
        section("Quick  |  N=50k, dim=32, ef_construction=400");
        for (int M : {16, 32}) {
            for (int ef_s : {50, 100, 200}) {
                print_row(run(50'000, 32, M, 400, ef_s));
            }
        }

    } else {
        // ---- Full suite ----

        // 1. Scaling with N
        section("Scale N  |  dim=128, M=16, ef_construction=200, ef_search=100");
        for (size_t n : {100'000, 500'000, 1'000'000}) {
            print_row(run(n, 128, 16, 200, 100));
        }

        // 2. ef_search recall/latency tradeoff at 100k
        section("ef_search tradeoff  |  N=100k, dim=128, M=16, ef_construction=200");
        for (int ef_s : {10, 20, 50, 100, 200, 400}) {
            print_row(run(100'000, 128, 16, 200, ef_s));
        }

        // 3. M tradeoff at 100k
        section("M tradeoff  |  N=100k, dim=128, ef_construction=200, ef_search=100");
        for (int M : {4, 8, 16, 32, 64}) {
            print_row(run(100'000, 128, M, 200, 100));
        }

        // 4. Dimensionality
        section("Dimensionality  |  N=100k, M=16, ef_construction=200, ef_search=100");
        for (size_t dim : {32, 64, 128, 256, 512}) {
            print_row(run(100'000, dim, 16, 200, 100));
        }
    }

    std::printf("\n");
    return 0;
}