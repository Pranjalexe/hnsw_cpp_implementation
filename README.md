# HNSW — Approximate Nearest Neighbor Index in C++

A from-scratch C++17 implementation of the **Hierarchical Navigable Small World (HNSW)** graph algorithm for high-performance approximate nearest neighbor (ANN) search, based on the paper by Malkov & Yashunin (2018).

The same algorithm powers the vector search layer in production systems like Pinecone, Weaviate, Milvus, and pgvector.

---

## What it does

Given a dataset of float vectors, HNSW builds a multi-layer proximity graph that supports sub-millisecond K-nearest-neighbor queries with high recall — far faster than brute-force search at the cost of a small, tunable approximation error.

The core idea: nodes are connected to their nearest neighbors in a layered graph. Upper layers are sparse and enable fast coarse navigation; the bottom layer is dense and enables precise local search. A query descends through layers greedily, arriving at a tight neighborhood in O(log N) steps.

---

## Features

- **O(log N) approximate KNN search** via hierarchical graph traversal
- **AVX2-accelerated distance computation** — L2 and inner product, with scalar fallback
- **Heuristic neighbor selection** (Algorithm 4 from the paper) — selects spatially diverse neighbors to preserve navigability, not just the M closest
- **Configurable recall/latency tradeoff** at query time via `ef` parameter, without rebuilding the index
- **Soft deletion** with automatic entry point repair
- **Binary save/load** for index persistence
- **Pluggable distance functions** — pass any `float(const float*, const float*, size_t)` callable
- **Thread-safe reads** via `std::shared_mutex` (concurrent searches, sequential inserts)
- **Flat contiguous vector storage** for cache-friendly distance computation
- **O(1) visited node tracking** using a generation-reset boolean array instead of a hash set

---

## Build

**Requirements:** GCC 9+ or Clang 10+, CMake 3.16+, C++17

```bash
git clone https://github.com/yourusername/hnsw
cd hnsw
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This produces two binaries:

```bash
./test_hnsw    # correctness test suite (8 tests)
./bench        # latency, throughput, and recall benchmarks
./bench --full # extended suite: 100k / 500k / 1M vectors
```

AVX2 is enabled automatically via `-march=native` if your CPU supports it. Run `grep -m1 avx2 /proc/cpuinfo` to check.

---

## Usage

```cpp
#include "hnsw.hpp"

// Build index
hnsw::Index::Params p;
p.dim             = 128;   // vector dimensionality
p.M               = 16;    // max neighbors per node
p.ef_construction = 200;   // beam width during build
p.ef_search       = 100;   // default beam width at query time

hnsw::Index idx(p);

// Insert vectors
std::vector<float> vec(128, 0.f); // your 128-dim vector
hnsw::NodeId id = idx.insert(vec);

// Bulk insert
idx.insert_batch(data_ptr, num_vectors);

// Search — returns up to K nearest neighbors sorted by distance
auto results = idx.search(query_ptr, /*K=*/10);
for (auto& nb : results)
    printf("id=%u  dist=%.4f\n", nb.id, nb.dist);

// Override ef at query time for higher recall
auto results_hq = idx.search(query_ptr, 10, /*ef=*/400);

// Soft delete
idx.remove(id);

// Persist
idx.save("index.bin");
idx.load("index.bin");
```

### Custom distance function

```cpp
hnsw::Index::Params p;
p.dim     = 128;
p.dist_fn = hnsw::inner_product_dist; // cosine similarity on unit vectors
hnsw::Index idx(p);
```

---

## Parameters

| Parameter | What it controls | Typical range |
|---|---|---|
| `M` | Max neighbors per node per layer. Higher → better recall, more memory, slower build. Layer 0 uses `2*M`. | 8–64 |
| `ef_construction` | Beam width during graph construction. Higher → better graph quality, slower build. Does not affect query speed. | 100–500 |
| `ef` (search) | Beam width at query time. Higher → better recall, higher latency. Can be changed per-query without rebuilding. | 50–400 |

**Rule of thumb:** set `ef_construction` ≥ `M`. For production recall targets above 95%, use M=16 and ef_construction=200 as a starting point, then tune `ef` at query time to hit your latency budget.

---

## Benchmark results

Synthetic dataset of random unit vectors, K=10, 1000 queries, `ef_construction=400`.

```
N        dim   M     ef_c   ef_s    build(ms)  inserts/sec   p50(us)  p95(us)  p99(us)   QPS       recall@10
--------------------------------------------------------------------------------------------------------------
50000    32    16    400    50      24095      2075          73.8     136.9    218.6     12090     0.8987
50000    32    16    400    100     26373      1896          137.5    303.9    422.0     6090      0.9766
50000    32    16    400    200     30104      1661          406.5    958.4    1834.8    2067      0.9973
50000    32    32    400    50      53137      941           104.8    238.0    362.5     7916      0.9781
50000    32    32    400    100     36106      1385          178.0    396.3    620.2     4613      0.9980
50000    32    32    400    200     34808      1436          336.5    648.1    978.4     2556      0.9998
    
```

## Updated Benchmark Results (after adding parallelised inserts)

Synthetic dataset of random unit vectors, K=10, 1000 queries, `ef_construction=400`.

```
N        dim   M     ef_c   ef_s    build(ms)  inserts/sec   p50(us)  p95(us)  p99(us)   QPS       recall@10
--------------------------------------------------------------------------------------------------------------
50000    32    16    400    50      5140       9728          68.8     142.6    220.3     12250     0.8981
50000    32    16    400    100     5883       8499          135.5    268.0    515.8     6237      0.9762
50000    32    16    400    200     5603       8923          261.1    484.8    951.2     3255      0.9972
50000    32    32    400    50      8841       5656          103.9    227.0    352.4     7945      0.9770
50000    32    32    400    100     8427       5933          212.1    424.1    673.5     4038      0.9976
50000    32    32    400    200     7932       6304          372.3    719.6    976.8     2339      0.9994
```


> Run `./bench` on your machine to get numbers for your hardware. Results vary significantly with CPU, cache size, and available RAM.

**Note on dimensionality:** random unit vectors in high dimensions (128-dim) are nearly equidistant from each other — the ratio of nearest to 10th-nearest neighbor distance is ~1.09, making recall@10 inherently harder than in real datasets. Real embeddings (text, image) have structure that HNSW exploits well, typically achieving 95%+ recall at 128+ dims.

---

## Project structure

```
hnsw/
├── include/
│   ├── hnsw.hpp        # Index class — insert, search, remove, save/load
│   ├── node.hpp        # Node struct with per-layer adjacency lists
│   └── distance.hpp    # L2 and inner product with AVX2 SIMD
├── src/
│   ├── hnsw.cpp        # Full algorithm implementation
│   ├── test.cpp        # Correctness test suite (8 tests)
│   └── main.cpp        # Benchmark harness
└── CMakeLists.txt
```

---


## References

- Malkov, Y. A., & Yashunin, D. A. (2018). [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320). *IEEE Transactions on Pattern Analysis and Machine Intelligence.*
