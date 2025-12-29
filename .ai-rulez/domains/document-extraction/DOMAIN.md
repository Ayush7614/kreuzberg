# Document Extraction Domain

## Purpose

The Document Extraction domain is responsible for orchestrating the complete document intelligence and content extraction pipeline within Kreuzberg. It serves as the central hub that detects document formats, routes documents to appropriate extractors, manages fallback chains, and coordinates the entire extraction lifecycle from initial file inspection through final post-processing.

## Key Responsibilities

### 1. Multi-Format Detection
- **MIME Type Identification**: Detect document format using file extension, magic bytes, and content analysis
- **Format Classification**: Classify documents as PDF, Office (Word/Excel/PowerPoint), Markdown, HTML, Images (PNG/JPG), or other binary formats
- **Dynamic Detection**: Support runtime detection of document types independent of file extension
- **Legacy Format Handling**: Convert legacy Office formats (DOC, PPT) to modern formats (DOCX, PPTX) for consistent processing

### 2. Extractor Routing & Selection
- **Registry Management**: Maintain registry of available DocumentExtractor plugins with priority levels
- **Format-to-Extractor Mapping**: Route documents to the highest-priority extractor that supports the detected MIME type
- **Priority-Based Selection**: Support extractor priority system (default 50) to enable:
  - Custom extractors to override built-in implementations (priority > 50)
  - Specialized extractors for specific use cases (priority between 0-100)
  - Fallback extractors for graceful degradation (priority < 50)
- **Multi-Extractor Support**: Allow multiple extractors to claim the same MIME type with priority-based arbitration

### 3. Fallback Chains & Error Recovery
- **Graceful Degradation**: Implement fallback extraction strategies when primary extractor fails
- **Error Context Preservation**: Track which extractor failed and why for debugging and fallback decisions
- **Batch Processing Resilience**: Continue batch extraction operations even when individual documents fail
- **Partial Result Handling**: Return best-effort extraction results with error metadata when complete extraction fails

### 4. Cache Integration
- **Result Caching**: Store extraction results using file content hash to avoid re-extraction of identical documents
- **Cache Invalidation**: Update cache when extractor configuration or version changes
- **Performance Optimization**: Provide sub-millisecond result retrieval for cached documents

## Core Components

### Extract Entry Points (`core/extractor.rs`)
- `extract_file()` - Extract content from file path with async support
- `extract_bytes()` - Extract content from byte array
- `batch_extract_file()` - Concurrent extraction of multiple files
- `batch_extract_bytes()` - Concurrent extraction of multiple byte arrays
- Synchronous wrappers for FFI and blocking contexts

### MIME Type System (`core/mime.rs`)
- MIME type detection and routing
- Legacy Office format conversion coordination
- Format-specific processing configuration

### Error Handling & Telemetry
- Structured error recording in OpenTelemetry spans
- Error type classification (ValidationError, ParsingError, OCRError, MissingDependencyError)
- Sanitized telemetry (excludes sensitive file path information)

## Integration with Kreuzberg Architecture

### Document Extractor Plugin Trait
The DocumentExtractor trait defines the interface for all document extraction implementations:

```rust
#[async_trait]
pub trait DocumentExtractor: Plugin {
    async fn extract_bytes(&self, content: &[u8], mime_type: &str, config: &ExtractionConfig)
        -> Result<ExtractionResult>;
    async fn extract_file(&self, path: &Path, mime_type: &str, config: &ExtractionConfig)
        -> Result<ExtractionResult>;
    fn supported_mime_types(&self) -> &[&str];
}
```

Key aspects:
- **Priority System**: Extractors declare their priority level for MIME type arbitration
- **Thread Safety**: All extractors must be Send + Sync for concurrent extraction
- **Config-Driven**: Each extractor receives ExtractionConfig to customize behavior
- **Result Standardization**: All extractors produce ExtractionResult with unified structure

### Built-in Extractors
- **PDF Extractor**: Supports PDF/A, encrypted PDFs, complex layouts via multiple backends (pdfplumber, PyMuPDF, Docling, MinerU)
- **Office Extractor**: Processes DOCX, XLSX, PPTX with layout preservation and table reconstruction
- **HTML/Markdown Extractor**: Extracts structured content from web documents
- **Image Extractor**: Orchestrates OCR pipeline for image documents
- **Text Extractor**: Direct plaintext extraction with encoding detection

### Extraction Configuration
The ExtractionConfig object controls behavior across all extractors:

```rust
pub struct ExtractionConfig {
    pub pdf_config: PdfConfig,
    pub image_extraction: ImageExtractionConfig,
    pub ocr_config: OcrConfig,
    pub chunking: ChunkingConfig,
    pub embedding: EmbeddingConfig,
    pub language_detection: LanguageDetectionConfig,
    pub post_processors: Vec<PostProcessorConfig>,
    // ... additional config fields
}
```

## Data Flow

### Extraction Pipeline
1. **Input** → File path or byte array
2. **Cache Check** → Return cached result if available
3. **MIME Detection** → Identify document format
4. **Format Conversion** → Convert legacy formats if needed
5. **Extractor Selection** → Find highest-priority matching extractor
6. **Extraction** → Execute extractor with appropriate config
7. **Error Handling** → Try fallback extractors if primary fails
8. **Post-Processing** → Apply document enhancement (keyword extraction, entity recognition)
9. **Caching** → Store result for future requests
10. **Output** → ExtractionResult with content, metadata, tables, chunks, images

### Result Structure
```rust
pub struct ExtractionResult {
    pub content: String,                    // Main extracted text
    pub mime_type: String,                  // Detected document format
    pub metadata: Metadata,                 // Document metadata
    pub tables: Vec<ExtractedTable>,        // Structured table data
    pub detected_languages: Option<Vec<DetectedLanguage>>, // Language detection
    pub chunks: Option<Vec<DocumentChunk>>, // Chunked content
    pub images: Option<Vec<ExtractedImage>>, // Extracted images
    pub pages: Option<Vec<PageInfo>>,       // Page-by-page information
}
```

## Dependencies & Relationships

### Upstream Dependencies
- **Kreuzberg Core**: Rust extraction library providing core PDF/Office/HTML extraction
- **LibreOffice**: Legacy document format conversion
- **MIME Detection Library**: File format detection

### Downstream Dependencies
- **OCR Integration Domain**: Processes image documents and embedded images
- **Plugin System Domain**: Loads and manages DocumentExtractor plugins
- **Caching Layer**: Stores extraction results for performance optimization

## Performance Characteristics

- **Single Document**: 50-500ms depending on format and document size
- **Cached Lookup**: <1ms for previously extracted documents
- **Batch Processing**: Concurrent extraction with configurable worker pool size
- **Memory**: Linear with document size (streaming where possible for large PDFs)
- **Cache Hit Rate**: Typically 40-60% in production due to repeated document processing

## Testing & Validation

- **Format Coverage**: Test with sample documents for each major format (PDF, DOCX, XLSX, etc.)
- **Error Scenarios**: Validate fallback behavior with corrupted/malformed documents
- **Performance**: Benchmark extraction speeds for various document sizes
- **Cache Consistency**: Verify cache hits and invalidation work correctly
- **Extractor Priority**: Test custom extractor registration and priority-based selection

## Future Enhancements

- Adaptive extractor selection based on document complexity analysis
- Streaming extraction for very large documents
- Incremental extraction with resume capability
- Content-aware format detection using ML models
- Extractor performance profiling and auto-tuning
