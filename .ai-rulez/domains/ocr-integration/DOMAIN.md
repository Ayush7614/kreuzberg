# OCR Integration Domain

## Purpose

The OCR Integration domain manages optical character recognition (OCR) for document images and embedded images within documents. It provides a pluggable backend system supporting multiple OCR engines (Tesseract, EasyOCR, PaddleOCR), implements sophisticated image preprocessing, manages OCR result caching, and integrates hOCR parsing with table reconstruction for high-fidelity document intelligence extraction.

## Key Responsibilities

### 1. Multiple OCR Backends
- **Backend Registry**: Maintain pluggable registry of OCR backend implementations
- **Tesseract Integration**: Native Tesseract backend via `kreuzberg-tesseract` C bindings with:
  - All Page Segmentation Modes (0-13) for flexible text layout handling
  - Multi-language support via language pack detection and validation
  - hOCR output format for detailed text positioning and table reconstruction
  - PSM configuration per document region for specialized extraction
- **Python OCR Backends**: Support for EasyOCR and PaddleOCR via Python plugin interface with:
  - Async execution using tokio blocking tasks
  - GIL management for safe Python-Rust FFI
  - Configurable model loading and caching
  - Language support inheritance from Python backend
- **Backend Selection**: Choose optimal OCR engine based on:
  - Image characteristics (resolution, text type, layout complexity)
  - Supported languages for detected text
  - Performance requirements and latency constraints
  - Accuracy preferences for specific document types

### 2. Image Preprocessing
- **Format Normalization**: Convert all image types (PNG, JPG, TIFF, WebP) to standardized format for processing
- **Resolution Optimization**: Upscale low-resolution images (< 150 DPI) and downsample extremely high-resolution images
- **Noise Reduction**: Apply Gaussian blur, morphological operations to reduce OCR errors from noise
- **Contrast Enhancement**: Improve readability through histogram equalization and CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Deskewing**: Automatic text angle detection and correction for misaligned scans
- **Binarization**: Convert grayscale/color images to pure black-and-white for certain OCR backends
- **Region Isolation**: Extract and process specific image regions (text blocks, tables) separately for specialized handling

### 3. OCR Result Processing & Caching
- **Result Caching**: Persistent cache using file content hash to avoid re-OCR of identical images
- **hOCR to Markdown**: Convert Tesseract hOCR format to clean Markdown with preserved text layout
- **Table Reconstruction**: Extract and reconstruct table structure from hOCR bounding boxes and positioning data
- **Word-Level Confidence**: Track OCR confidence scores per word for downstream confidence filtering
- **Language Detection**: Identify text languages from OCR output for multi-lingual document handling
- **Batch Processing**: Efficient concurrent OCR processing of multiple images with resource pooling

### 4. Language Management
- **Language Support Validation**: Verify Tesseract language packs are installed and valid
- **Multi-Language Support**: Handle documents with mixed languages across pages or regions
- **Language Configuration**: Allow per-document language specification to improve OCR accuracy
- **Dynamic Language Loading**: Support runtime installation hints for missing language packs

## Core Components

### OCR Processor (`ocr/processor.rs`)
Primary entry point for OCR operations:
- `OcrProcessor::new()` - Initialize with optional cache configuration
- `process_image()` - Process single image with TesseractConfig
- `batch_process_images()` - Concurrent processing of image batches
- Cache management and eviction strategies

### OCR Backend Traits
Extensible backend interface supporting custom implementations:

```rust
#[async_trait]
pub trait OcrBackend: Send + Sync {
    async fn process_image(&self, image_bytes: &[u8], config: &TesseractConfig)
        -> Result<ExtractionResult>;
    fn supported_languages(&self) -> Vec<String>;
    fn name(&self) -> String;
}
```

### Tesseract Backend (`ocr/tesseract_backend.rs`)
Native Tesseract integration via kreuzberg-tesseract:
- Direct C FFI bindings for performance
- All PSM modes (0-13) support for flexible text recognition
- hOCR output parsing and conversion
- Configuration per document region

### hOCR Processing (`ocr/hocr.rs`)
Convert Tesseract hOCR XML to structured data:
- Parse bounding box information for word positioning
- Extract confidence scores per word and paragraph
- Convert formatting information (bold, italic) to Markdown
- Preserve spatial layout for table reconstruction

### Table Reconstruction (`ocr/table/mod.rs`)
Extract and reconstruct tables from OCR output:
- `reconstruct_table()` - Convert hOCR positioned text to table structure
- `extract_words_from_tsv()` - Parse Tesseract TSV output format
- `table_to_markdown()` - Format reconstructed tables as Markdown
- Cell boundary detection using positional data

### OCR Caching (`ocr/cache.rs`)
Performance optimization through caching:
- Content-based hash for cache key (file content, not filename)
- Persistent storage with configurable backends (file system, Redis)
- Cache statistics tracking (hits, misses, evictions)
- Invalidation when OCR config changes

### Language Registry (`ocr/language_registry.rs`)
Tesseract language management:
- Detect available language packs
- Validate language codes against Tesseract requirements
- Provide language pack installation hints

### Configuration Types (`ocr/types.rs`)
```rust
pub struct TesseractConfig {
    pub languages: Vec<String>,     // OCR languages
    pub psm: PSMMode,               // Page Segmentation Mode
    pub oem: usize,                 // OCR Engine Mode
    pub preprocessing: ImagePreprocessingConfig,
    pub confidence_threshold: f32,  // Filter low-confidence words
}

pub enum PSMMode {
    SingleCharacter = 8,
    WordAsLine = 6,
    UniformBlock = 3,
    SparseText = 11,
    // ... 10 additional modes
}
```

## Integration with Kreuzberg Architecture

### Image Extraction Pipeline
The document extraction domain routes image documents to the OCR integration:

1. **Image Detection** → Detect image format and size
2. **Preprocessing** → Apply preprocessing transformations
3. **OCR Selection** → Choose optimal backend based on config
4. **OCR Execution** → Process image and extract text
5. **Result Enhancement** → Parse hOCR, reconstruct tables, detect languages
6. **Caching** → Store OCR result for future requests
7. **Return to Extractor** → Provide text and table results to document extraction pipeline

### Embedded Image Handling
Documents containing embedded images (PDFs with scanned content, Office docs with image content):

```rust
pub struct ExtractionResult {
    pub images: Option<Vec<ExtractedImage>>, // Embedded images
    // OCR results from images incorporated into main content
}

pub struct ExtractedImage {
    pub image_data: Vec<u8>,        // Original image bytes
    pub extracted_text: String,     // OCR output
    pub position: (u32, u32, u32, u32), // Page coordinates
    pub confidence: f32,            // OCR confidence
}
```

### Configuration Integration
OCR behavior controlled through ExtractionConfig cascade:

```rust
pub struct ExtractionConfig {
    pub ocr_config: OcrConfig,
    pub image_extraction: ImageExtractionConfig,
}

pub struct OcrConfig {
    pub enabled: bool,
    pub default_languages: Vec<String>,
    pub cache_enabled: bool,
    pub preprocessing: ImagePreprocessingConfig,
    pub backend_priority: Vec<String>, // ["tesseract", "easyocr", "paddle"]
}
```

### Python Plugin Interface
Support for custom OCR backends via Python:

```rust
pub struct PythonOcrBackend {
    python_obj: Py<PyAny>,
    name: String,                    // Cached Python name
    supported_languages: Vec<String>, // Cached from Python
}
```

Features:
- GIL management for safe Python-Rust FFI
- Async execution via tokio::task::spawn_blocking
- Python exception handling with error translation

## Data Flow

### Single Image OCR
1. **Input** → Image bytes, TesseractConfig
2. **Cache Check** → Return cached result if available
3. **Preprocessing** → Apply transformations if enabled
4. **Backend Selection** → Choose OCR engine
5. **OCR Execution** → Run backend and collect output
6. **hOCR Parsing** → Extract bounding boxes and confidence
7. **Table Reconstruction** → Identify and reconstruct tables
8. **Language Detection** → Identify text languages
9. **Caching** → Store processed result
10. **Output** → ExtractionResult with text, tables, metadata

### Batch OCR Processing
- Concurrent processing with configurable worker pool
- Shared OCR processor instance to amortize initialization cost
- Individual error handling per image
- Progress tracking for monitoring

## Dependencies & Relationships

### Upstream Dependencies
- **kreuzberg-tesseract**: Rust-C bindings for Tesseract engine
- **image**: Image format detection and basic manipulation
- **imageproc**: Advanced image processing operations
- **pyo3**: Python FFI for plugin backends
- **serde**: Configuration serialization

### Downstream Dependencies
- **Document Extraction Domain**: Calls OCR for image documents and embedded images
- **Plugin System Domain**: Loads and manages Python OCR backends
- **Caching Layer**: Persistent storage for OCR results

## Performance Characteristics

### Processing Times (per image)
- **Preprocessing**: 5-50ms (deskew, denoise, contrast enhancement)
- **Tesseract OCR**: 50-500ms depending on resolution and language count
- **hOCR Parsing**: <5ms for typical documents
- **Cache Lookup**: <1ms for previously processed images
- **Batch Processing**: Linear with image count (parallel execution)

### Memory Usage
- Tesseract instance: ~50-100MB per process
- Image preprocessing: Temporary memory proportional to image resolution
- hOCR parsing: Memory proportional to word count

### Cache Efficiency
- Hit rate: 30-50% in production (repeated document processing)
- Storage: ~5KB per OCR result (text + metadata)

## Testing & Validation

- **Format Coverage**: Test with PNG, JPG, TIFF, WebP image formats
- **Resolution Testing**: Verify preprocessing handles low/high resolution images
- **Language Support**: Test multi-language documents with language switching
- **PSM Modes**: Validate different page segmentation modes for various layouts
- **Table Detection**: Test table reconstruction accuracy on complex layouts
- **Error Cases**: Validate handling of blurry/noisy/inverted images
- **Cache Consistency**: Verify cache hits work correctly across process boundaries
- **Preprocessing Impact**: Benchmark accuracy improvements from preprocessing

## Future Enhancements

- Machine learning-based OCR backend selection
- Parallel preprocessing and OCR execution on multi-core systems
- Adaptive PSM mode selection based on document layout analysis
- Handwriting recognition support via specialized backends
- Real-time progress tracking for long OCR operations
- Confidence-based result filtering and flagging
- Integration with specialized document type handlers (forms, tables)
- Hardware acceleration support (CUDA, Metal, OpenVINO)
