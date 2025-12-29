---
name: extraction-pipeline-patterns
priority: critical
---

# Extraction Pipeline Patterns

**Kreuzberg's format detection → extraction → fallback orchestration for 56+ file formats**

## Core Pipeline Architecture

The extraction pipeline (`crates/kreuzberg/src/core/pipeline.rs`, `crates/kreuzberg/src/extraction/`) orchestrates:
1. **Format Detection** - MIME type inference + extension validation → select appropriate extractor
2. **Intelligent Extraction** - Route to format-specific extractors (PDF, DOCX, Excel, HTML, images, archives, etc.)
3. **Fallback Strategies** - Password-protected PDFs, OCR for images, nested archive handling, corrupted file recovery
4. **Post-Processing Pipeline** - Validators, quality processing, chunking, custom hooks (see `core/pipeline.rs`)

## Format Detection Strategy

**Location**: `crates/kreuzberg/src/core/mime.rs`, `crates/kreuzberg/src/core/formats.rs`

- Detect format via MIME type (from Content-Type header or magic bytes)
- Validate extension alignment to prevent spoofing
- Route to correct extractor module in `extraction/{email,html,image,docx,excel,pptx,table,archive}.rs`
- Handle format upgrades (`.xls` → `.xlsx` support in Excel extractor)

**Pattern**: Multiple extractors match same format (e.g., PDF tables via both `table.rs` and `pdf/` module) → choose highest confidence/specificity.

## Extraction Modules (56 Formats)

| Category | Extractors | Key Modules |
|----------|-----------|------------|
| **Office** | DOCX, XLSX, XLSM, XLSB, XLS, PPTX, ODP, ODS | `extraction/{docx,excel,pptx}.rs` |
| **PDF** | Standard + encrypted, password attempts | `pdf/` subdirectory (13 files) |
| **Images** | PNG, JPG, TIFF, WebP, JP2, SVG (OCR-enabled) | `extraction/image.rs` + `ocr/` |
| **Web** | HTML, XHTML, XML, SVG (DOM parsing) | `extraction/html.rs` (67KB - complex table handling) |
| **Email** | EML, MSG (headers, body, attachments, threading) | `extraction/email.rs` |
| **Archives** | ZIP, TAR, GZ, 7Z (recursive extraction) | `extraction/archive.rs` (31KB) |
| **Markdown** | MD, TXT, RST, Org Mode, RTF | `extraction/markdown.rs` |
| **Academic** | LaTeX, BibTeX, JATS, Jupyter, DocBook | `extraction/{structured,xml}.rs` |

## Fallback Strategies

### Password-Protected PDFs
```rust
// Pattern: Multiple password attempts (single password or list)
// Location: crates/kreuzberg/src/pdf/ module
- Try primary password
- Fallback to secondary password list
- Return with `is_encrypted=true` in metadata if decryption fails
```

### OCR Fallback for Images
```rust
// Pattern: Image extraction → OCR when text extraction insufficient
// Location: crates/kreuzberg/src/extraction/image.rs + ocr/
- Attempt direct image text extraction
- If confidence < threshold: trigger OCR backend (Tesseract, EasyOCR, PaddleOCR)
- Return both direct + OCR results with confidence scores
```

### Nested Archive Handling
```rust
// Pattern: Recursive extraction with depth limiting
// Location: crates/kreuzberg/src/extraction/archive.rs (31KB)
- Extract ZIP/TAR/7Z
- If nested archives found: recursive extraction (depth limit configurable)
- Flatten results or preserve hierarchy (config option)
- Include file metadata (size, type, position in hierarchy)
```

### Corrupted File Recovery
```rust
// Pattern: Graceful degradation with partial results
- Stream-based parsing (avoid memory exhaustion)
- Catch parsing errors, emit content up to error point
- Include error location in metadata
- Continue extraction on next chunk (if applicable)
```

## Configuration Integration

**Location**: `crates/kreuzberg/src/core/config.rs`, `crates/kreuzberg/src/core/config_validation.rs`

```rust
pub struct ExtractionConfig {
    // Format-specific config
    pub pdf: Option<PdfConfig>,  // password, max_pages, ocr thresholds
    pub image: Option<ImageConfig>,  // ocr_backend, confidence, table_detection
    pub html: Option<HtmlConfig>,  // preserve_tables, extract_links, metadata
    pub office: Option<OfficeConfig>,  // extract_metadata, preserve_formatting

    // Fallback orchestration
    pub fallback: Option<FallbackConfig>,  // enable ocr, retry count, timeout

    // Post-processing
    pub postprocessor: Option<PostProcessorConfig>,  // validators, chunking, custom hooks
    pub chunking: Option<ChunkingConfig>,  // text splitting with FastEmbed
    pub keywords: Option<KeywordConfig>,  // YAKE/RAKE extraction
}
```

## Plugin System Integration

**Location**: `crates/kreuzberg/src/plugins/`

- **CustomExtractor**: Register format-specific extractors (override built-ins)
- **PostProcessor**: Modify extraction results after main extraction (Early/Middle/Late stages)
- **Validator**: Fail-fast validation (e.g., minimum text length, required fields)
- **OCRBackend**: Swap OCR engine (Tesseract ↔ custom backend)

**Pattern**: Plugin registry loaded at startup → cached for zero-cost lookup during extraction.

## Key Code Patterns

### Format Detection Pattern
```rust
// Location: core/mime.rs
fn detect_format(content: &[u8], extension: Option<&str>) -> Result<FileFormat> {
    match (magic_bytes(content), extension) {
        (Some(fmt), Some(ext)) if fmt == ext => Ok(fmt),  // Aligned
        (Some(fmt), Some(ext)) => Err(FormatMismatch(fmt, ext)),  // Spoofing
        (Some(fmt), None) => Ok(fmt),  // Magic bytes only
        (None, Some(ext)) => Ok(FileFormat::from_extension(ext)),  // Extension only
        _ => Err(UnknownFormat),
    }
}
```

### Extraction Dispatcher Pattern
```rust
// Location: extraction/mod.rs
async fn extract_with_detector(
    source: DocumentSource,
    config: &ExtractionConfig,
) -> Result<ExtractionResult> {
    let format = detect_format(&source.bytes, source.extension)?;

    let result = match format {
        FileFormat::Pdf => extract_pdf(&source, config).await?,
        FileFormat::Docx => extract_docx(&source, config).await?,
        FileFormat::Image => extract_image_with_ocr_fallback(&source, config).await?,
        FileFormat::Archive => extract_archive_recursive(&source, config).await?,
        _ => extract_with_plugin(format, &source, config)?,
    };

    // Post-processing pipeline
    run_pipeline(result, config).await
}
```

### Fallback Orchestration Pattern
```rust
// Location: core/pipeline.rs
async fn run_extraction_with_fallbacks(
    source: &DocumentSource,
    config: &ExtractionConfig,
) -> Result<ExtractionResult> {
    // Try primary extraction
    match primary_extraction(source, config).await {
        Ok(result) => Ok(result),
        Err(e) => {
            if config.fallback.ocr_enabled && is_image_like(source) {
                // Fallback to OCR
                ocr_extraction(source, &config.image).await
            } else if config.fallback.retry_count > 0 {
                // Retry with partial config
                retry_extraction(source, config, config.fallback.retry_count).await
            } else {
                Err(e)
            }
        }
    }
}
```

## Testing Strategy

**Location**: Tests in `crates/kreuzberg/tests/`, `e2e/` directory

- **Format Coverage**: 56+ format test documents in `fixtures/` (real-world samples)
- **Fallback Scenarios**: Password-protected PDFs, corrupted files, nested archives
- **Format Detection**: Spoofing tests, extension mismatch, missing magic bytes
- **Benchmark Suite**: `tools/benchmark-harness/` for format-specific performance

## Critical Rules

1. **Always use format detection** before routing to extractors (prevent confusion attacks)
2. **Stream-based parsing** for PDFs/archives to handle multi-GB files
3. **Post-pipeline is mandatory**: All extraction results flow through `run_pipeline()` for validators/hooks
4. **Plugin overrides are order-dependent**: Plugins registered first take priority
5. **Fallback timeouts**: Set reasonable OCR/archive extraction timeouts (config-driven)
6. **Metadata preservation**: Include format detection confidence, extraction method used, any fallbacks applied

## Related Skills

- **ocr-backend-management** - OCR engine selection and image preprocessing
- **chunking-embeddings** - Post-extraction text splitting with FastEmbed
- **api-server-patterns** - Axum endpoint for extraction pipeline exposure
- **feature-flag-strategy** - Conditional feature compilation (embeddings, OCR backends)
