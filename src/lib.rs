//! # fast-cosine-rs
//!
//! A zero-dependency, cache-friendly library for high-throughput vector similarity
//! and normalization operations designed for dense RAG (Retrieval-Augmented Generation) embeddings.
//!
//! ## Overview
//!
//! Vector similarity in dense retrieval often becomes the CPU bottleneck when re-ranking
//! or filtering embedding candidates before passing them to a cross-encoder or LLM.
//! `fast-cosine-rs` provides:
//! - Unrolled loops structured to enable auto-vectorization across SIMD architectures (SSE, AVX2, NEON).
//! - Specialized zero-allocation operations for pre-normalized embedding spaces (e.g. OpenAI `text-embedding-3`, Cohere, Voyage).
//! - Top-K ranking utilities for in-memory approximate or exact nearest neighbors.
//! - Thorough error types handling dimension mismatches, empty vectors, and zero norms.

use std::cmp::Ordering;
use std::fmt;

/// Errors arising during vector similarity computations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VectorError {
    /// Vectors have different dimensions (e.g., 768 vs 1536).
    DimensionMismatch { expected: usize, found: usize },
    /// Input vector contains zero elements.
    EmptyVector,
    /// Vector magnitude is zero or degenerate, preventing division by norm.
    ZeroNorm,
}

impl fmt::Display for VectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, found } => {
                write!(f, "vector dimension mismatch: expected {}, got {}", expected, found)
            }
            Self::EmptyVector => write!(f, "vector cannot be empty"),
            Self::ZeroNorm => write!(f, "vector has zero or degenerate magnitude (L2 norm is 0)"),
        }
    }
}

impl std::error::Error for VectorError {}

/// Computes the dot product of two slices of `f32`.
///
/// Uses 4-way loop unrolling to maximize CPU pipeline throughput and facilitate
/// compiler auto-vectorization into SIMD FMA (Fused Multiply-Add) instructions.
///
/// # Errors
/// Returns [`VectorError::EmptyVector`] if input slices are empty, or
/// [`VectorError::DimensionMismatch`] if slice lengths differ.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
    if a.is_empty() {
        return Err(VectorError::EmptyVector);
    }
    if a.len() != b.len() {
        return Err(VectorError::DimensionMismatch {
            expected: a.len(),
            found: b.len(),
        });
    }

    let len = a.len();
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let idx = i * 4;
        sum0 += a[idx] * b[idx];
        sum1 += a[idx + 1] * b[idx + 1];
        sum2 += a[idx + 2] * b[idx + 2];
        sum3 += a[idx + 3] * b[idx + 3];
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3);

    let offset = chunks * 4;
    for i in 0..remainder {
        total += a[offset + i] * b[offset + i];
    }

    Ok(total)
}

/// Computes the squared L2 norm (magnitude) of a vector.
#[inline]
pub fn norm_squared(a: &[f32]) -> Result<f32, VectorError> {
    if a.is_empty() {
        return Err(VectorError::EmptyVector);
    }

    let len = a.len();
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let idx = i * 4;
        sum0 += a[idx] * a[idx];
        sum1 += a[idx + 1] * a[idx + 1];
        sum2 += a[idx + 2] * a[idx + 2];
        sum3 += a[idx + 3] * a[idx + 3];
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3);

    let offset = chunks * 4;
    for i in 0..remainder {
        total += a[offset + i] * a[offset + i];
    }

    Ok(total)
}

/// Computes the Euclidean L2 norm of a vector.
#[inline]
pub fn l2_norm(a: &[f32]) -> Result<f32, VectorError> {
    norm_squared(a).map(|sq| sq.sqrt())
}

/// In-place L2 normalization of an `f32` slice.
///
/// Modifies the vector such that `||a||_2 == 1.0`.
///
/// # Errors
/// Returns [`VectorError::ZeroNorm`] if the vector magnitude is less than `1e-12`.
pub fn l2_normalize(a: &mut [f32]) -> Result<(), VectorError> {
    let norm = l2_norm(a)?;
    if norm < 1e-12 {
        return Err(VectorError::ZeroNorm);
    }
    let inv_norm = 1.0 / norm;
    for x in a.iter_mut() {
        *x *= inv_norm;
    }
    Ok(())
}

/// Computes the standard Cosine Similarity between vectors `a` and `b`.
///
/// Returns a value in range `[-1.0, 1.0]`, where `1.0` indicates identical direction.
///
/// # Errors
/// Returns [`VectorError::ZeroNorm`] if either vector has near-zero magnitude.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
    let dot = dot_product(a, b)?;
    let norm_a = l2_norm(a)?;
    let norm_b = l2_norm(b)?;

    let denom = norm_a * norm_b;
    if denom < 1e-12 {
        return Err(VectorError::ZeroNorm);
    }

    let sim = dot / denom;
    // Clamp to valid mathematical interval [-1.0, 1.0] to guard against floating-point drift
    Ok(sim.clamp(-1.0, 1.0))
}

/// Computes Cosine Similarity between vectors already known to be unit normalized.
///
/// For unit vectors, cosine similarity is strictly equal to the dot product, avoiding
/// two square root and two division operations.
#[inline]
pub fn cosine_similarity_normalized(a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
    dot_product(a, b).map(|dot| dot.clamp(-1.0, 1.0))
}

/// Computes the Euclidean distance (L2 distance) between two vectors.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
    if a.is_empty() {
        return Err(VectorError::EmptyVector);
    }
    if a.len() != b.len() {
        return Err(VectorError::DimensionMismatch {
            expected: a.len(),
            found: b.len(),
        });
    }

    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }

    Ok(sum.sqrt())
}

/// Computes the Manhattan (L1) distance between two vectors.
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
    if a.is_empty() {
        return Err(VectorError::EmptyVector);
    }
    if a.len() != b.len() {
        return Err(VectorError::DimensionMismatch {
            expected: a.len(),
            found: b.len(),
        });
    }

    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += (a[i] - b[i]).abs();
    }

    Ok(sum)
}

/// Computes cosine similarity of `query` against each vector in `candidates`.
///
/// Pre-allocates the result vector and validates dimension on each candidate.
pub fn batch_cosine_similarity<'a>(
    query: &[f32],
    candidates: &[&'a [f32]],
) -> Result<Vec<f32>, VectorError> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let mut results = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        results.push(cosine_similarity(query, candidate)?);
    }
    Ok(results)
}

/// Ranks the candidates by cosine similarity relative to `query`, returning the top `k` indices and scores.
///
/// Output is sorted in descending order of similarity (highest first).
pub fn top_k_cosine<'a>(
    query: &[f32],
    candidates: &[&'a [f32]],
    k: usize,
) -> Result<Vec<(usize, f32)>, VectorError> {
    if k == 0 || candidates.is_empty() {
        return Ok(Vec::new());
    }

    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(candidates.len());

    for (idx, candidate) in candidates.iter().enumerate() {
        let score = cosine_similarity(query, candidate)?;
        scored.push((idx, score));
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scored.truncate(k);

    Ok(scored)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_dot_product_identical() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0];
        let res = dot_product(&a, &b).expect("dot product failed");
        // 1 + 4 + 9 + 16 + 25 = 55
        assert!((res - 55.0).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b).expect("cosine failed");
        assert!(sim.abs() < EPSILON);
    }

    #[test]
    fn test_cosine_similarity_parallel() {
        let a = [2.0, 4.0, 6.0];
        let b = [1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b).expect("cosine failed");
        assert!((sim - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = [1.0, 2.0, 3.0];
        let b = [-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b).expect("cosine failed");
        assert!((sim - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 3.0];
        let err = cosine_similarity(&a, &b).unwrap_err();
        assert_eq!(
            err,
            VectorError::DimensionMismatch {
                expected: 2,
                found: 3
            }
        );
    }

    #[test]
    fn test_zero_norm_handling() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 3.0];
        let err = cosine_similarity(&a, &b).unwrap_err();
        assert_eq!(err, VectorError::ZeroNorm);
    }

    #[test]
    fn test_l2_normalization() {
        let mut a = [3.0, 4.0];
        l2_normalize(&mut a).expect("normalize failed");
        assert!((a[0] - 0.6).abs() < EPSILON);
        assert!((a[1] - 0.8).abs() < EPSILON);

        let norm = l2_norm(&a).expect("norm failed");
        assert!((norm - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 0.0, -1.0];
        let dist = manhattan_distance(&a, &b).expect("manhattan failed");
        // |1-4| + |2-0| + |3 - (-1)| = 3 + 2 + 4 = 9
        assert!((dist - 9.0).abs() < EPSILON);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        let dist = euclidean_distance(&a, &b).expect("distance failed");
        assert!((dist - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_batch_cosine_similarity() {
        let query = [1.0, 0.0];
        let c0 = [1.0, 0.0];
        let c1 = [0.0, 1.0];
        let candidates = [&c0[..], &c1[..]];
        let results = batch_cosine_similarity(&query, &candidates).unwrap();
        assert_eq!(results.len(), 2);
        assert!((results[0] - 1.0).abs() < EPSILON);
        assert!(results[1].abs() < EPSILON);
    }

    #[test]
    fn test_top_k_ranking() {
        let query = [1.0, 0.0];
        let c0: [f32; 2] = [1.0, 0.0];      // sim = 1.0
        let c1: [f32; 2] = [0.7071, 0.7071];// sim ~= 0.707
        let c2: [f32; 2] = [0.0, 1.0];      // sim = 0.0
        let c3: [f32; 2] = [-1.0, 0.0];     // sim = -1.0

        let candidates = [&c0[..], &c1[..], &c2[..], &c3[..]];
        let top = top_k_cosine(&query, &candidates, 2).expect("ranking failed");

        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 0); // c0 is #1
        assert_eq!(top[1].0, 1); // c1 is #2
        assert!((top[0].1 - 1.0).abs() < EPSILON);
    }
}
