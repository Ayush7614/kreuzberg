---
name: feature-flag-strategy
priority: high
---

# Feature Flag Strategy

**Managing 20+ feature combinations, platform-specific builds, and conditional compilation in Kreuzberg's polyglot architecture**

## Feature Matrix Overview

**Location**: `Cargo.toml` (workspace), `crates/kreuzberg/Cargo.toml`, `FEATURE_MATRIX.md`

Kreuzberg manages 20+ interdependent features across 9 language bindings:

```
Rust Core Crate (crates/kreuzberg)
├── OCR Backends
│   ├── tesseract (default)
│   ├── tesseract-static (bundled binary)
│   └── ocr-minimal (text extraction only)
├── Document Formats
│   ├── pdf (default, with PDFium)
│   ├── pdf-minimal (text only)
│   ├── office (DOCX, XLSX, PPTX)
│   └── office-minimal (text only)
├── AI/ML Features
│   ├── embeddings (ONNX Runtime + fastembed)
│   ├── keywords-yake (YAKE keyword extraction)
│   ├── keywords-rake (RAKE keyword extraction)
│   └── language-detection (fast-langdetect)
├── API & Server
│   ├── api (Axum server, REST endpoints)
│   ├── mcp (Model Context Protocol server)
│   └── tokio-runtime (full async runtime)
├── Platform Support
│   ├── python-bindings (PyO3)
│   ├── ruby-bindings (Rutie)
│   ├── php-bindings (Extism)
│   ├── node-bindings (NAPI-RS)
│   └── wasm (wasm32-unknown-unknown target)
└── Development
    ├── otel (OpenTelemetry tracing)
    ├── bench (Criterion benchmarks)
    └── dev-tools (debugging, profiling)
```

## Cargo.toml Feature Definition

**Location**: `Cargo.toml` (workspace root), `crates/kreuzberg/Cargo.toml`

```toml
[package]
name = "kreuzberg"
version = "4.0.0-rc.22"

[features]
# Default: Minimal + essential features
default = ["api", "tesseract", "pdf", "office", "embeddings", "keywords-yake"]

# OCR Backends (mutually exclusive recommendation)
tesseract = ["dep:kreuzberg-tesseract"]
tesseract-static = ["tesseract", "kreuzberg-tesseract/bundled"]
ocr-minimal = []  # No OCR backend; image extraction only

# Document Format Support
pdf = ["dep:pdfium-render"]
pdf-minimal = []  # Text extraction from PDFs only
office = []  # DOCX, XLSX, PPTX support
office-minimal = []  # Metadata extraction only

# AI/ML Features
embeddings = ["dep:fastembed", "dep:ort"]  # Requires ONNX Runtime
keywords-yake = ["dep:yake"]
keywords-rake = ["dep:rake"]
language-detection = ["dep:fast-langdetect"]

# Server & API
api = ["dep:axum", "dep:tower", "dep:tower-http", "tokio-runtime"]
mcp = []  # Model Context Protocol server mode

# Runtime
tokio-runtime = ["dep:tokio"]  # Full async runtime
lite-runtime = []  # Minimal embedded runtime

# Platform Bindings
python-bindings = ["dep:pyo3"]
ruby-bindings = ["dep:rutie"]
php-bindings = ["dep:extism"]
node-bindings = ["dep:napi-rs"]
wasm = []  # wasm32-unknown-unknown target

# Observability
otel = ["dep:opentelemetry"]

# Build & Testing
bench = ["criterion"]
```

## Platform-Specific Feature Combinations

### 1. Core Rust Library (crates/kreuzberg)

**Typical Build**:
```bash
cargo build --release --features "api,tesseract,pdf,office,embeddings,keywords-yake,language-detection"
```

**Minimal Build** (for embedded):
```bash
cargo build --release --no-default-features --features "pdf-minimal,ocr-minimal"
```

**Static Tesseract** (Docker/reproducible):
```bash
cargo build --release --features "tesseract-static,api"
```

### 2. Python Binding (crates/kreuzberg-py)

**Location**: `crates/kreuzberg-py/Cargo.toml`, `packages/python/`

```toml
[dependencies]
kreuzberg = { path = "../kreuzberg", features = [
    "python-bindings",
    "api",
    "tesseract",
    "pdf",
    "office",
    "embeddings",
    "keywords-yake",
    "language-detection",
    "otel",  # Optional tracing support
] }

[features]
default = []
full = ["kreuzberg/embeddings", "kreuzberg/keywords-rake"]
lite = ["kreuzberg/ocr-minimal", "kreuzberg/pdf-minimal"]
```

**Build Matrix** (via `maturin`):
```yaml
# .github/workflows/python-wheels.yml
python-versions: [3.8, 3.9, 3.10, 3.11, 3.12]
platforms: [x86_64-unknown-linux-gnu, x86_64-apple-darwin, aarch64-apple-darwin, x86_64-pc-windows-msvc]
features:
  - default (full features)
  - lite (minimal embeddings/ocr)
  - staticlibs (bundled Tesseract)
```

### 3. Node.js Binding (crates/kreuzberg-node)

**Location**: `crates/kreuzberg-node/Cargo.toml`

```toml
[dependencies]
kreuzberg = { path = "../kreuzberg", features = [
    "node-bindings",
    "api",
    "tesseract",
    "pdf",
    "office",
    "embeddings",
    "keywords-yake",
    "language-detection",
] }

[features]
# Node specific
napi-rs = ["dep:napi-rs"]
```

**Runtime Detection** (JavaScript):
```javascript
// packages/typescript/node/src/index.ts
import addon from '../native/index.node';

// Feature detection at runtime
export function getCapabilities(): Capabilities {
  return {
    ocr: addon.hasOcr(),          // true if tesseract feature enabled
    embeddings: addon.hasEmbeddings(),  // true if embeddings feature enabled
    keywords: addon.hasKeywords(),      // true if keyword extraction available
    pdf: addon.hasPdf(),                // true if pdf feature enabled
    office: addon.hasOffice(),          // true if office feature enabled
  };
}
```

### 4. WebAssembly (crates/kreuzberg-wasm)

**Location**: `crates/kreuzberg-wasm/Cargo.toml`

```toml
[dependencies]
kreuzberg = { path = "../kreuzberg", features = [
    "wasm",
    "pdf-minimal",        # Light PDF parsing
    "ocr-minimal",        # No Tesseract (browser-unsafe)
    # NO embeddings (ONNX Runtime incompatible with WASM)
    # NO keywords (language models too large for browser)
], default-features = false }

[lib]
crate-type = ["cdylib"]

[profile.release]
opt-level = "z"  # Minimize binary size
lto = true
codegen-units = 1
```

**Feature Limitations** (WASM):
- NO OCR (Tesseract not available in browsers)
- NO Embeddings (ONNX Runtime incompatible)
- NO Keywords extraction (YAKE/RAKE models too large)
- Limited PDF support (use `pdf-minimal` for client-side parsing)
- Table extraction only (no reconstruction)

### 5. Go Binding (packages/go/v4)

**No Cargo.toml** - Uses kreuzberg-ffi C binding. Feature flags applied at Rust compile time.

```bash
# Compile Rust FFI with all features
cd crates/kreuzberg-ffi
cargo build --release --features "api,tesseract,pdf,office,embeddings,keywords-yake,language-detection"

# Go then imports C header
# packages/go/v4/binding.go includes <kreuzberg.h>
```

### 6. Other Bindings (Ruby, PHP, Elixir, C#, Java)

All use FFI/bindings to pre-compiled Rust core → feature flags set at Rust build time, not in language-specific config.

## Conditional Compilation Patterns

### Location-Based Feature Gating

**Location**: `crates/kreuzberg/src/lib.rs`, module structure

```rust
// src/lib.rs
#[cfg(feature = "api")]
pub mod api;

#[cfg(feature = "mcp")]
pub mod mcp;

#[cfg(feature = "embeddings")]
pub mod embeddings;

#[cfg(any(feature = "keywords-yake", feature = "keywords-rake"))]
pub mod keywords;

// Only include extraction format if feature enabled
#[cfg(feature = "pdf")]
pub mod pdf;
#[cfg(feature = "office")]
pub mod office_extractors;
```

### Function-Level Feature Gating

```rust
// src/extraction/image.rs
pub async fn extract_text_from_image(
    image_data: &[u8],
    config: &ImageConfig,
) -> Result<ExtractionResult> {
    // ... common image processing ...

    // OCR is optional
    #[cfg(feature = "tesseract")]
    {
        if config.enable_ocr {
            return ocr_text_extraction(image_data, config).await;
        }
    }

    // Fallback to direct text extraction (no OCR)
    Ok(ExtractionResult {
        content: extract_text_direct(image_data)?,
        ..Default::default()
    })
}
```

### Runtime Feature Detection

```rust
// src/core/config_validation.rs
pub fn validate_config(config: &ExtractionConfig) -> Result<ValidationReport> {
    let mut warnings = Vec::new();

    // Check if requested feature is compiled in
    #[cfg(not(feature = "embeddings"))]
    {
        if config.chunking.as_ref().map_or(false, |c| c.embedding.is_some()) {
            warnings.push("Embeddings requested but not compiled in. Set feature 'embeddings'".to_string());
        }
    }

    #[cfg(not(feature = "tesseract"))]
    {
        if config.image.as_ref().map_or(false, |c| c.enable_ocr) {
            warnings.push("OCR requested but tesseract feature not enabled".to_string());
        }
    }

    Ok(ValidationReport { warnings, ..Default::default() })
}
```

## Build Target Configuration

### Supported Targets

```toml
# Cargo.toml - [profile.*.package] overrides
[profile.release.package.kreuzberg-wasm]
opt-level = "z"  # Minimize WASM size
codegen-units = 1
lto = true

[profile.release]
lto = "thin"  # Balance between compile time and performance
opt-level = 3
strip = true
codegen-units = 1
```

### Target-Specific Builds

**Cross-compilation**:
```bash
# Compile for ARM64 on macOS
cargo build --release --target aarch64-apple-darwin

# Compile for Linux on macOS (requires cross)
cross build --release --target x86_64-unknown-linux-gnu

# Windows MSVC (for embeddings support)
cargo build --release --target x86_64-pc-windows-msvc

# WebAssembly
wasm-pack build --target web crates/kreuzberg-wasm --release
```

### Docker Multi-Stage Build

**Location**: `docker/Dockerfile`

```dockerfile
# Stage 1: Build with all features
FROM rust:1.91 as builder
WORKDIR /build
COPY . .
RUN cargo build --release --features "api,tesseract-static,pdf,office,embeddings,keywords-yake,language-detection,otel"

# Stage 2: Minimal runtime
FROM debian:bookworm-slim
COPY --from=builder /build/target/release/kreuzberg-api /usr/local/bin/
RUN apt-get update && apt-get install -y libssl3 && rm -rf /var/lib/apt/lists/*
ENTRYPOINT ["kreuzberg-api"]
```

## Feature Flag Testing Strategy

**Location**: `tests/`, `.github/workflows/`

### Matrix Testing

```yaml
# .github/workflows/feature-tests.yml
strategy:
  matrix:
    features:
      - "default"
      - "api,tesseract,pdf,office,embeddings"
      - "pdf-minimal,ocr-minimal"  # Minimal build
      - "api,tesseract,language-detection"  # No embeddings
      - "api,keywords-yake,keywords-rake"  # No PDF
    os: [ubuntu-latest, macos-latest, windows-latest]

steps:
  - run: cargo test --features "${{ matrix.features }}" --release
  - run: cargo build --target wasm32-unknown-unknown --no-default-features
```

### Feature Availability Tests

```rust
// tests/feature_detection.rs
#[test]
fn test_embeddings_available() {
    #[cfg(feature = "embeddings")]
    {
        assert!(can_use_embeddings());
    }

    #[cfg(not(feature = "embeddings"))]
    {
        assert!(must_skip_embeddings_tests());
    }
}

#[test]
fn test_ocr_backends_available() {
    #[cfg(feature = "tesseract")]
    {
        assert!(ocr_processor().is_ok());
    }

    #[cfg(all(not(feature = "tesseract"), not(feature = "ocr-minimal")))]
    {
        assert_eq!(extract_image_text().unwrap().len(), 0);  // No OCR
    }
}
```

## Language Binding Feature Exposure

### Python

```python
# kreuzberg/__init__.pyi (type stubs)
import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

# Feature availability
HAS_EMBEDDINGS: bool
HAS_KEYWORDS: bool
HAS_OCR: bool
HAS_PDF: bool

def get_features() -> dict[str, bool]:
    """Return which features are compiled in"""
    ...
```

### TypeScript/Node.js

```typescript
// @kreuzberg/node
export interface Capabilities {
  ocr: boolean;
  embeddings: boolean;
  keywords: boolean;
  pdf: boolean;
  office: boolean;
  mcp: boolean;
}

export function getCapabilities(): Capabilities {
  // Runtime detection via NAPI-RS
}
```

## Critical Rules

1. **Never mix conflicting features** - e.g., `ocr-minimal` + `tesseract` should error at compile time
2. **Always provide feature diagnostics** - Config validation must warn if feature unavailable
3. **Default to maximum feature set** - Unless embedded/minimal specifically requested
4. **Test all feature combinations** - Matrix testing in CI catches regressions
5. **Document feature dependencies**:
   - `embeddings` requires ONNX Runtime installed
   - `tesseract-static` includes Tesseract binary (larger binary)
   - `api` requires tokio runtime
   - WASM incompatible with embeddings, keywords, OCR
6. **Version gate features carefully** - Python 3.8 vs 3.12 differ in typing
7. **CI/CD gate features by platform** - Windows MSVC required for embeddings, WASM only on web

## Related Skills

- **extraction-pipeline-patterns** - Feature-gated extraction paths
- **ocr-backend-management** - Feature gates for different OCR backends
- **api-server-patterns** - `api` feature gating Axum server
- **chunking-embeddings** - Feature gating embeddings/FastEmbed
- **mcp-protocol-integration** - MCP feature compilation
