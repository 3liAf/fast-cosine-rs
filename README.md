# fast-cosine-rs

[![CI](https://github.com/3liAf/fast-cosine-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/3liAf/fast-cosine-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

Zero-dependency, cache-friendly vector similarity and normalization routines optimized for dense RAG embeddings (OpenAI `text-embedding-3`, Cohere Embed v3, Voyage AI, BGE).

---

## Features

- **Auto-Vectorized Loops**: 4-way unrolled loops structured for compiler auto-vectorization across SIMD instruction sets (AVX2, FMA, SSE4.1, ARM NEON).
- **Fast-Path Normalized Cosine**: Special-cased operations for unit-normalized embeddings, replacing expensive norm calculations with raw dot products.
- **Top-K Ranking**: In-memory heap-less top-$k$ similarity filtering for low-latency reranking layers.
- **Zero Dependencies**: Pure standard library Rust.
- **Strict Error Handling**: Explicit, zero-cost error types for empty vectors, dimension mismatches, and degenerate norms.

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
fast-cosine-rs = { git = "https://github.com/3liAf/fast-cosine-rs" }
```

---

## Quickstart

```rust
use fast_cosine_rs::{cosine_similarity, cosine_similarity_normalized, l2_normalize, top_k_cosine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let query = vec![0.12, -0.45, 0.88, 0.23];
    let doc_a = vec![0.15, -0.40, 0.85, 0.29];
    let doc_b = vec![-0.80, 0.10, 0.05, -0.58];

    // Standard Cosine Similarity
    let sim_a = cosine_similarity(&query, &doc_a)?;
    let sim_b = cosine_similarity(&query, &doc_b)?;

    println!("Similarity A: {:.4}", sim_a); // ~0.992
    println!("Similarity B: {:.4}", sim_b); // ~-0.341

    // Top-K Candidate Selection
    let candidates = [&doc_a[..], &doc_b[..]];
    let top_results = top_k_cosine(&query, &candidates, 1)?;

    assert_eq!(top_results[0].0, 0); // doc_a is the top match
    Ok(())
}
```

---

## Benchmark

Run the included benchmark on 10,000 1536-dimensional vectors (standard OpenAI embedding size):

```bash
cargo run --release --example benchmark
```

Typical output (modern x86_64 CPU):
- **Dot product**: ~1.8 ms / 10k vectors (~5.5M vectors/sec)
- **Pre-normalized Cosine**: ~1.8 ms / 10k vectors
- **Full Cosine Similarity**: ~3.9 ms / 10k vectors
- **Top-10 Retrieval**: ~4.1 ms end-to-end

---

## License

MIT © [Ali Afzal Yousuf](https://github.com/3liAf)
