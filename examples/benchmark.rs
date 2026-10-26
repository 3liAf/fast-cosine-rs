//! Benchmark example for fast-cosine-rs
//!
//! Simulates vector similarity search across dense embedding spaces (e.g. 1536-dim text-embedding-3).

use fast_cosine_rs::{cosine_similarity, cosine_similarity_normalized, dot_product, l2_normalize, top_k_cosine};
use std::time::Instant;

fn main() {
    println!("=== fast-cosine-rs High-Throughput Retrieval Benchmark ===");

    let dim = 1536;
    let num_vectors = 10_000;

    println!("Generating {} synthetic embedding vectors (dim = {})...", num_vectors, dim);

    // Deterministic pseudo-random generation without external dependencies
    let mut rng_state: u64 = 42;
    let mut next_f32 = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng_state >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let mut query = vec![0.0f32; dim];
    for x in query.iter_mut() {
        *x = next_f32();
    }
    l2_normalize(&mut query).unwrap();

    let mut corpus: Vec<Vec<f32>> = Vec::with_capacity(num_vectors);
    for _ in 0..num_vectors {
        let mut v = vec![0.0f32; dim];
        for x in v.iter_mut() {
            *x = next_f32();
        }
        l2_normalize(&mut v).unwrap();
        corpus.push(v);
    }

    let corpus_refs: Vec<&[f32]> = corpus.iter().map(|v| v.as_slice()).collect();

    // 1. Raw dot product throughput
    let start = Instant::now();
    let mut sum_dots = 0.0f32;
    for doc in &corpus_refs {
        sum_dots += dot_product(&query, doc).unwrap();
    }
    let dot_duration = start.elapsed();
    let dot_throughput = (num_vectors as f64) / dot_duration.as_secs_f64();

    println!(
        "Dot product (10k x 1536): {:?} ({:.0} vectors/sec, sum: {:.4})",
        dot_duration, dot_throughput, sum_dots
    );

    // 2. Full Cosine Similarity
    let start = Instant::now();
    let mut sum_cosine = 0.0f32;
    for doc in &corpus_refs {
        sum_cosine += cosine_similarity(&query, doc).unwrap();
    }
    let cos_duration = start.elapsed();
    let cos_throughput = (num_vectors as f64) / cos_duration.as_secs_f64();

    println!(
        "Cosine similarity (10k x 1536): {:?} ({:.0} vectors/sec, sum: {:.4})",
        cos_duration, cos_throughput, sum_cosine
    );

    // 3. Normalized Cosine (dot-product fast path)
    let start = Instant::now();
    let mut sum_norm_cosine = 0.0f32;
    for doc in &corpus_refs {
        sum_norm_cosine += cosine_similarity_normalized(&query, doc).unwrap();
    }
    let norm_cos_duration = start.elapsed();
    let norm_cos_throughput = (num_vectors as f64) / norm_cos_duration.as_secs_f64();

    println!(
        "Pre-normalized Cosine (10k x 1536): {:?} ({:.0} vectors/sec, sum: {:.4})",
        norm_cos_duration, norm_cos_throughput, sum_norm_cosine
    );

    // 4. Top-K Selection
    let k = 10;
    let start = Instant::now();
    let top_results = top_k_cosine(&query, &corpus_refs, k).unwrap();
    let top_k_duration = start.elapsed();

    println!(
        "Top-{} retrieval across 10k vectors: {:?}",
        k, top_k_duration
    );
    for (rank, (idx, score)) in top_results.iter().enumerate() {
        println!("  #{}: doc_idx={}, similarity={:.4}", rank + 1, idx, score);
    }
    println!("Benchmark completed successfully.");
}
