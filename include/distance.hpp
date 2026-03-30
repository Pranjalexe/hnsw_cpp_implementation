#pragma once
#include <cmath>
#include <cstddef>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace hnsw {

using Vector = std::vector<float>;

// L2 squared distance (skip sqrt — monotone, saves cycles)
inline float l2_sq(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    size_t i   = 0;

#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m256 va   = _mm256_loadu_ps(a + i);
        __m256 vb   = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc         = _mm256_fmadd_ps(diff, diff, acc);
    }
    // Horizontal sum of 8 floats
    __m128 lo  = _mm256_castps256_ps128(acc);
    __m128 hi  = _mm256_extractf128_ps(acc, 1);
    __m128 s   = _mm_add_ps(lo, hi);
    s          = _mm_hadd_ps(s, s);
    s          = _mm_hadd_ps(s, s);
    sum        = _mm_cvtss_f32(s);
#endif

    // Scalar tail (or full scalar if no AVX2)
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum    += d * d;
    }
    return sum;
}

inline float l2_sq(const Vector& a, const Vector& b) {
    return l2_sq(a.data(), b.data(), a.size());
}

// Inner product distance (1 - dot) for unit vectors
inline float inner_product_dist(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    size_t i  = 0;

#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc       = _mm256_fmadd_ps(va, vb, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s         = _mm_hadd_ps(s, s);
    s         = _mm_hadd_ps(s, s);
    dot       = _mm_cvtss_f32(s);
#endif

    for (; i < dim; ++i) dot += a[i] * b[i];
    return 1.0f - dot;
}

inline float inner_product_dist(const Vector& a, const Vector& b) {
    return inner_product_dist(a.data(), b.data(), a.size());
}

} // namespace hnsw