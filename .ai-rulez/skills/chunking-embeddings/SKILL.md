---
name: chunking-embeddings
priority: critical
---

# Chunking & Embeddings

**Text splitting strategies, embedding generation with FastEmbed, RAG pipeline integration**

## Chunking Architecture Overview

**Location**: `crates/kreuzberg/src/chunking/`, `crates/kreuzberg/src/embeddings.rs`

The chunking subsystem handles intelligent text splitting for RAG (Retrieval-Augmented Generation) pipelines:

```
Extracted Text
    ↓
[1. Normalization] → Clean whitespace, remove control chars
    ↓
[2. Chunk Strategy Selection] → Fixed-size, semantic, syntax-aware
    ↓
[3. Overlap Management] → Control context window overlap
    ↓
[4. Optional Embedding] → Generate vectors with FastEmbed
    ↓
Output: Vec<Chunk> with text, vectors, metadata
```

## Chunking Strategies

**Location**: `crates/kreuzberg/src/chunking/mod.rs` (72KB)

### 1. Fixed-Size Chunking

```rust
pub struct FixedSizeChunkingConfig {
    pub chunk_size: usize,           // Characters or tokens (configurable)
    pub overlap: usize,              // Overlap characters
    pub trim_whitespace: bool,       // Remove leading/trailing spaces
}

// Pattern: Sliding window with overlap
fn chunk_fixed_size(text: &str, config: &FixedSizeChunkingConfig) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let chunk_size = config.chunk_size;
    let overlap = config.overlap;

    for i in (0..text.len()).step_by(chunk_size - overlap) {
        let end = std::cmp::min(i + chunk_size, text.len());
        let chunk_text = &text[i..end];

        if !chunk_text.is_empty() {
            chunks.push(Chunk {
                text: chunk_text.to_string(),
                start_pos: i,
                end_pos: end,
                ..Default::default()
            });
        }
    }

    chunks
}
```

**Use Case**: Uniform chunk size for consistency; best for embedding models with fixed token limits (e.g., 384-token sentence embeddings).

### 2. Semantic Chunking

```rust
pub struct SemanticChunkingConfig {
    pub target_chunk_size: usize,     // Desired size (flexible)
    pub min_chunk_size: usize,        // Never below this
    pub max_chunk_size: usize,        // Never above this
    pub semantic_threshold: f32,      // 0.0-1.0 similarity threshold
    pub use_sentence_boundaries: bool, // Prefer sentence breaks
}

// Pattern: Split by sentences/paragraphs, then merge/split based on similarity
async fn chunk_semantic(
    text: &str,
    config: &SemanticChunkingConfig,
) -> Result<Vec<Chunk>> {
    // Step 1: Split into sentences
    let sentences = split_into_sentences(text)?;

    // Step 2: Group sentences into chunks
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();

    for (i, sentence) in sentences.iter().enumerate() {
        let combined = if current_chunk.is_empty() {
            sentence.clone()
        } else {
            format!("{} {}", current_chunk, sentence)
        };

        // Check if adding next sentence would exceed max
        if combined.len() > config.max_chunk_size && !current_chunk.is_empty() {
            // Emit chunk and start new one
            chunks.push(Chunk {
                text: current_chunk.clone(),
                ..Default::default()
            });
            current_chunk = sentence.clone();
        } else {
            current_chunk = combined;
        }

        // Check semantic similarity to next sentence (if embeddings available)
        if let Some(next_sentence) = sentences.get(i + 1) {
            let similarity = compute_similarity(&combined, next_sentence)?;
            if similarity < config.semantic_threshold && current_chunk.len() > config.min_chunk_size {
                chunks.push(Chunk {
                    text: current_chunk.clone(),
                    ..Default::default()
                });
                current_chunk.clear();
            }
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(Chunk {
            text: current_chunk,
            ..Default::default()
        });
    }

    Ok(chunks)
}
```

**Use Case**: Smart context preservation for LLM consumption; works well with semantic search.

### 3. Syntax-Aware Chunking (Markup-Aware)

```rust
pub struct SyntaxAwareChunkingConfig {
    pub chunk_by: SyntaxUnit,  // Paragraph, Section, Heading, etc.
    pub max_chunk_size: usize, // If section > limit, split further
    pub respect_code_blocks: bool, // Don't split code
}

pub enum SyntaxUnit {
    Paragraph,      // Split at \n\n
    Section,        // Split at H1/H2 headings (Markdown/HTML)
    Heading,        // Use heading hierarchy as chunks
    Sentence,       // Split at '. '
    CodeBlock,      // Preserve code as single chunk
}

// Pattern: Use Markdown/HTML structure awareness
fn chunk_syntax_aware(text: &str, config: &SyntaxAwareChunkingConfig) -> Vec<Chunk> {
    match config.chunk_by {
        SyntaxUnit::Paragraph => {
            text.split("\n\n")
                .map(|p| Chunk { text: p.to_string(), ..Default::default() })
                .collect()
        }
        SyntaxUnit::Section => {
            // Use heading regex to detect sections
            let re = regex::Regex::new(r"^(#{1,6})\s+(.+)$").unwrap();
            let mut chunks = Vec::new();
            let mut current_section = String::new();

            for line in text.lines() {
                if re.is_match(line) && !current_section.is_empty() {
                    chunks.push(Chunk {
                        text: current_section.clone(),
                        ..Default::default()
                    });
                    current_section.clear();
                }
                current_section.push_str(line);
                current_section.push('\n');
            }

            if !current_section.is_empty() {
                chunks.push(Chunk {
                    text: current_section,
                    ..Default::default()
                });
            }

            chunks
        }
        _ => vec![],  // Other strategies
    }
}
```

**Use Case**: Preserve document structure (sections, code blocks) for better context in RAG.

### 4. Recursive Chunking (LangChain Pattern)

```rust
pub struct RecursiveChunkingConfig {
    pub separators: Vec<&'static str>,  // ["\n\n", "\n", " ", ""]
    pub chunk_size: usize,
    pub overlap: usize,
}

// Pattern: Try splitting by separator[0]; if too large, try separator[1], etc.
fn chunk_recursive(text: &str, config: &RecursiveChunkingConfig) -> Vec<Chunk> {
    fn split_recursive(
        text: &str,
        separators: &[&str],
        chunk_size: usize,
        overlap: usize,
    ) -> Vec<String> {
        let mut chunks = Vec::new();

        // Try current separator
        if let Some(separator) = separators.first() {
            let splits: Vec<&str> = text.split(separator).collect();

            let mut good_splits = Vec::new();
            let mut current = String::new();

            for split in splits {
                if split.len() > chunk_size {
                    // Recursively split this part
                    if !current.is_empty() {
                        good_splits.push(current.clone());
                        current.clear();
                    }
                    let subsplits = split_recursive(split, &separators[1..], chunk_size, overlap);
                    good_splits.extend(subsplits);
                } else {
                    current.push_str(split);
                    if current.len() > chunk_size {
                        good_splits.push(current.clone());
                        current.clear();
                    }
                }
            }

            if !current.is_empty() {
                good_splits.push(current);
            }

            chunks.extend(good_splits);
        }

        chunks
    }

    let splits = split_recursive(text, &config.separators, config.chunk_size, config.overlap);
    splits
        .into_iter()
        .map(|text| Chunk { text, ..Default::default() })
        .collect()
}
```

**Use Case**: Best general-purpose chunking; automatically finds optimal split points.

## Chunking Configuration Presets

**Location**: `crates/kreuzberg/src/chunking/mod.rs`

```rust
pub enum ChunkingPreset {
    Balanced,       // 512 tokens, 50 token overlap → RAG sweet spot
    Compact,        // 256 tokens, 32 overlap → Dense vectors
    Extended,       // 1024 tokens, 100 overlap → Full context
    Minimal,        // 128 tokens, 16 overlap → Lightweight embeddings
}

impl ChunkingPreset {
    pub fn to_config(self) -> ChunkingConfig {
        match self {
            Balanced => ChunkingConfig {
                chunk_size: 512,
                overlap: 50,
                strategy: ChunkingStrategy::Semantic,
                ..Default::default()
            },
            Compact => ChunkingConfig {
                chunk_size: 256,
                overlap: 32,
                strategy: ChunkingStrategy::FixedSize,
                ..Default::default()
            },
            Extended => ChunkingConfig {
                chunk_size: 1024,
                overlap: 100,
                strategy: ChunkingStrategy::Recursive,
                ..Default::default()
            },
            _ => Default::default(),
        }
    }
}
```

**Usage**:
```rust
let config = ExtractionConfig {
    chunking: Some(ChunkingConfig {
        preset: Some("balanced".to_string()),
        embedding: Some(EmbeddingConfig::default()),
        ..Default::default()
    }),
    ..Default::default()
};
```

## Embedding Generation with FastEmbed

**Location**: `crates/kreuzberg/src/embeddings.rs` (18KB)

### FastEmbed Integration

```rust
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

pub struct EmbeddingConfig {
    pub model: EmbeddingModel,        // BAAI/bge-small-en-v1.5 (default)
    pub cache_dir: Option<String>,    // ONNX model cache location
    pub batch_size: usize,            // Batch size for inference (default 256)
    pub device: Device,               // CPU or GPU (CUDA)
    pub parallel_requests: usize,     // Number of parallel embedding tasks
}

pub enum EmbeddingModel {
    BgeSmallEnV15,      // 384 dims, fast, excellent for RAG
    BgeSmallZhV15,      // Chinese optimized
    BgeBaseEnV15,       // 768 dims, better quality, slower
    JinaEmbeddingsV2,   // 768 dims, supports long context (up to 8192)
    Custom(String),     // Custom ONNX model path
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: EmbeddingModel::BgeSmallEnV15,
            batch_size: 256,
            device: Device::CPU,
            parallel_requests: 4,
            ..Default::default()
        }
    }
}
```

### Embedding Generation Pattern

```rust
// Location: embeddings.rs
pub struct TextEmbeddingManager {
    // Singleton pattern: cache embedding models per config
    models: Arc<RwLock<HashMap<String, Arc<TextEmbedding>>>>,
}

impl TextEmbeddingManager {
    pub async fn embed_chunks(
        &self,
        chunks: Vec<Chunk>,
        config: &EmbeddingConfig,
    ) -> Result<Vec<ChunkWithEmbedding>> {
        // Load or get cached model
        let model = self.get_or_init_model(config).await?;

        // Prepare texts
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();

        // Embed in parallel batches
        let embeddings = model.embed(texts, None).await?;

        // Combine chunks with embeddings
        let result = chunks
            .into_iter()
            .zip(embeddings)
            .map(|(chunk, embedding)| ChunkWithEmbedding {
                text: chunk.text,
                embedding: embedding.to_vec(),  // Vec<f32>
                dimensions: embedding.len(),
                ..Default::default()
            })
            .collect();

        Ok(result)
    }

    async fn get_or_init_model(
        &self,
        config: &EmbeddingConfig,
    ) -> Result<Arc<TextEmbedding>> {
        let model_key = config.model.name();
        let mut models = self.models.write().map_err(|e| EmbeddingError::LockPoisoned)?;

        if let Some(model) = models.get(&model_key) {
            return Ok(Arc::clone(model));
        }

        // Initialize model (downloads ONNX if needed)
        let model_name = match &config.model {
            EmbeddingModel::BgeSmallEnV15 => "BAAI/bge-small-en-v1.5",
            EmbeddingModel::BgeBaseEnV15 => "BAAI/bge-base-en-v1.5",
            EmbeddingModel::JinaEmbeddingsV2 => "jinaai/jina-embeddings-v2-base-en",
            _ => return Err(EmbeddingError::UnknownModel),
        };

        let init_options = InitOptions::default()
            .with_cache_dir(config.cache_dir.clone())
            .with_device(config.device.clone());

        let model = TextEmbedding::try_new(init_options.model_name(model_name))?;
        let model = Arc::new(model);

        models.insert(model_key, Arc::clone(&model));

        Ok(model)
    }
}
```

### ONNX Runtime Requirement

**Location**: `crates/kreuzberg/src/embeddings.rs` (lines 14-36)

```rust
//! # ONNX Runtime Requirement
//!
//! **CRITICAL**: This module requires ONNX Runtime to be installed.
//!
//! Installation:
//! - macOS: `brew install onnxruntime@1.21`
//! - Linux: `apt install libonnxruntime libonnxruntime-dev`
//! - Windows: Download from Microsoft ONNX Runtime releases
//!
//! Verify: `echo $ORT_DYLIB_PATH` should point to ONNX Runtime lib
```

**Feature Gating**:
```toml
[features]
embeddings = ["dep:fastembed", "dep:ort"]  # Requires ONNX Runtime
```

## RAG Integration Pattern

**Location**: Integration point for extraction → chunking → embedding pipeline

```rust
pub async fn extract_and_embed_for_rag(
    file_path: &str,
    config: &ExtractionConfig,
) -> Result<RagDocument> {
    // Step 1: Extract document
    let result = extract_file(file_path, None, config).await?;

    // Step 2: Chunk text (if chunking enabled)
    let chunks = if let Some(chunking_config) = &config.chunking {
        let strategy = ChunkingStrategy::from_preset(
            &chunking_config.preset.as_ref().unwrap_or(&"balanced".to_string())
        )?;

        chunk_text(&result.content, &strategy)?
    } else {
        vec![Chunk {
            text: result.content.clone(),
            ..Default::default()
        }]
    };

    // Step 3: Embed chunks (if embedding enabled)
    let chunks_with_embeddings = if let Some(embed_config) = &config.chunking.as_ref()
        .and_then(|c| c.embedding.as_ref()) {
        let manager = TextEmbeddingManager::new();
        manager.embed_chunks(chunks, embed_config).await?
    } else {
        chunks.into_iter().map(|c| ChunkWithEmbedding {
            text: c.text,
            embedding: None,
            ..Default::default()
        }).collect()
    };

    // Step 4: Prepare for RAG (vector DB ingestion)
    Ok(RagDocument {
        file_path: file_path.to_string(),
        metadata: result.metadata,
        chunks: chunks_with_embeddings,
    })
}
```

## Performance Optimization

### Embedding Batch Processing

```rust
// Embed multiple documents' chunks in parallel
pub async fn embed_batch(
    documents: Vec<(String, Vec<String>)>,  // (doc_id, chunks)
    config: &EmbeddingConfig,
) -> Result<Vec<(String, Vec<Vec<f32>>)>> {
    let manager = TextEmbeddingManager::new();

    let tasks: Vec<_> = documents
        .into_iter()
        .map(|(doc_id, texts)| {
            let manager_clone = manager.clone();
            let config_clone = config.clone();
            tokio::spawn(async move {
                let model = manager_clone.get_or_init_model(&config_clone).await?;
                let embeddings = model.embed(texts, Some(config_clone.batch_size)).await?;
                Ok::<_, EmbeddingError>((doc_id, embeddings.iter().map(|e| e.to_vec()).collect()))
            })
        })
        .collect();

    let results = futures::future::try_join_all(tasks).await?;
    Ok(results.into_iter().filter_map(|r| r.ok()).collect())
}
```

### Lazy Model Loading

```rust
// Models only loaded when first chunk needs embedding
lazy_static::lazy_static! {
    static ref EMBEDDING_MANAGER: Mutex<Option<TextEmbeddingManager>> = Mutex::new(None);
}

pub async fn get_embedding_manager() -> Result<TextEmbeddingManager> {
    let mut manager = EMBEDDING_MANAGER.lock().map_err(|_| EmbeddingError::LockPoisoned)?;

    if manager.is_none() {
        *manager = Some(TextEmbeddingManager::new());
    }

    Ok(manager.as_ref().unwrap().clone())
}
```

## Chunk Metadata

**Location**: `types.rs`

```rust
pub struct Chunk {
    pub text: String,
    pub start_pos: usize,      // Position in original document
    pub end_pos: usize,
    pub page_number: Option<u32>,  // For multi-page docs
    pub section_heading: Option<String>,  // If markup-aware chunking
    pub confidence: Option<f32>,   // Quality score
}

pub struct ChunkWithEmbedding {
    pub text: String,
    pub embedding: Vec<f32>,   // 384 dims for bge-small, 768 for bge-base
    pub dimensions: usize,
    pub norm: Option<f32>,     // L2 norm for similarity computation
    pub metadata: HashMap<String, String>,  // Custom metadata
}
```

## Critical Rules

1. **Chunking is preprocessing** - Always apply before embedding to ensure consistent vector sizes
2. **Overlap prevents information loss** - Set overlap to 15-20% of chunk size
3. **Embedding models are stateful** - Lazy load and cache to avoid repeated initialization
4. **ONNX Runtime is required** - Gracefully degrade if not available (skip embeddings)
5. **Batch embedding for performance** - Never embed single chunks; batch 50-1000 chunks
6. **Normalize embeddings for search** - Use L2 norm for cosine similarity
7. **Cache embedding results** - Don't re-embed identical text chunks
8. **Model selection impacts quality** - bge-small (384) for speed, bge-base (768) for quality

## Related Skills

- **extraction-pipeline-patterns** - Text extraction preceding chunking
- **feature-flag-strategy** - Embeddings feature gating (requires ONNX Runtime)
- **api-server-patterns** - Endpoint for chunking + embedding operations
- **ocr-backend-management** - OCR text quality affects chunking success
