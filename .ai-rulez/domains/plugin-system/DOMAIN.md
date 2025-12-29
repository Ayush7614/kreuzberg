# Plugin System Domain

## Purpose

The Plugin System domain provides the core architecture for extensibility within Kreuzberg, enabling third-party developers to implement custom document extractors, OCR backends, post-processors, validators, and other enhancement modules. It manages plugin discovery, lifecycle management, trait-based registry systems, and priority-based selection logic to ensure seamless integration of plugins with the core extraction engine.

## Key Responsibilities

### 1. Plugin Architecture
- **Base Plugin Trait**: Define common interface for all plugin types:
  ```rust
  pub trait Plugin: Send + Sync {
      fn name(&self) -> &str;
      fn version(&self) -> String;
      fn initialize(&self) -> Result<()>;
      fn shutdown(&self) -> Result<()>;
  }
  ```
- **Type-Specific Traits**: Specialized traits for different plugin categories:
  - `DocumentExtractor` - Custom document format support
  - `OcrBackend` - Alternative OCR engines (EasyOCR, PaddleOCR, custom models)
  - `PostProcessor` - Result enhancement (entity extraction, keyword extraction, classification)
  - `Validator` - Content validation and quality checks
- **Thread Safety**: All plugins must be Send + Sync for concurrent operation
- **Lifecycle Methods**: Initialize/shutdown hooks for resource management

### 2. Plugin Discovery
- **Static Registration**: Direct plugin instantiation and registration via Rust code:
  ```rust
  let plugin = MyCustomExtractor::new();
  register_document_extractor(plugin)?;
  ```
- **Python Plugin Discovery**: Load Python plugins dynamically with:
  - Module path scanning and class detection
  - Automatic instantiation of classes implementing plugin protocols
  - Validation that Python classes implement required methods
  - GIL management for safe Python object handling
- **Dynamic Loading**: Runtime plugin loading for deployment flexibility
- **Plugin Validation**: Verify plugins implement required traits before registration
- **Dependency Resolution**: Detect and validate plugin dependencies

### 3. Priority Selection System
- **Priority Levels**: Numeric priority (0-255) for arbitrating between multiple plugins claiming the same capability:
  - 255: Highest priority (critical overrides)
  - 100-200: High priority (custom implementations override built-ins)
  - 50: Default priority (standard behavior)
  - 0-49: Low priority (fallback implementations)
- **MIME Type Arbitration**: When multiple DocumentExtractor plugins support the same MIME type:
  - Select highest-priority plugin
  - Fall back to next-highest on failure
  - Continue chain until successful extraction or all fail
- **Capability Arbitration**: Select optimal plugin based on:
  - Supported capabilities
  - Language support (for OCR/NLP plugins)
  - Required external dependencies
  - Performance characteristics
- **Conflict Resolution**: When plugins conflict, apply priority rules consistently

### 4. Registry Management
- **Plugin Registries**: Maintain separate registries for each plugin type:
  - DocumentExtractorRegistry: All document extraction plugins
  - OcrBackendRegistry: All OCR engine implementations
  - PostProcessorRegistry: All result enhancement plugins
  - ValidatorRegistry: All validation plugins
- **Thread-Safe Registry Operations**: RwLock-protected registries for concurrent access
- **Registration/Unregistration**: Add and remove plugins dynamically
- **Registry Queries**: List plugins, query capabilities, filter by criteria
- **Registry Events**: Optional hooks for registration/unregistration notifications

## Core Components

### Plugin Trait System (`plugins/mod.rs`)
Base trait hierarchy:

```rust
pub trait Plugin: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> String;
    fn initialize(&self) -> Result<()>;
    fn shutdown(&self) -> Result<()>;
}
```

### Document Extractor Plugin (`plugins/extractor.rs`)
Interface for custom document extraction:

```rust
#[async_trait]
pub trait DocumentExtractor: Plugin {
    async fn extract_bytes(&self, content: &[u8], mime_type: &str, config: &ExtractionConfig)
        -> Result<ExtractionResult>;
    async fn extract_file(&self, path: &Path, mime_type: &str, config: &ExtractionConfig)
        -> Result<ExtractionResult>;
    fn supported_mime_types(&self) -> &[&str];
    fn priority(&self) -> u8 { 50 } // Override for custom priority
}
```

Key features:
- Multiple MIME types per extractor
- Async execution for non-blocking I/O
- Config-driven behavior customization
- Optional priority override

### OCR Backend Plugin (`plugins/ocr.rs`)
Interface for custom OCR engines:

```rust
#[async_trait]
pub trait OcrBackend: Plugin {
    async fn process_image(&self, image_bytes: &[u8], config: &TesseractConfig)
        -> Result<ExtractionResult>;
    fn supported_languages(&self) -> Vec<String>;
    fn capabilities(&self) -> OcrCapabilities;
}

pub struct OcrCapabilities {
    pub supports_table_detection: bool,
    pub supports_hocr_output: bool,
    pub supports_confidence_scores: bool,
    pub min_resolution_dpi: u32,
}
```

### Post-Processor Plugin (`plugins/postprocessor.rs`)
Interface for result enhancement:

```rust
#[async_trait]
pub trait PostProcessor: Plugin {
    async fn process(&self, result: &mut ExtractionResult, config: &PostProcessorConfig)
        -> Result<()>;
    fn supported_mime_types(&self) -> &[&str];
    fn priority(&self) -> u8 { 50 }
}
```

Examples:
- Keyword extraction (YAKE, RAKE algorithms)
- Named entity recognition
- Language detection
- Sentiment analysis
- Content classification

### Validator Plugin (`plugins/validator.rs`)
Interface for result validation:

```rust
#[async_trait]
pub trait Validator: Plugin {
    async fn validate(&self, result: &ExtractionResult) -> Result<ValidationReport>;
    fn validation_type(&self) -> ValidationType;
}

pub struct ValidationReport {
    pub is_valid: bool,
    pub issues: Vec<ValidationIssue>,
    pub recommendations: Vec<String>,
}
```

### Python Plugin FFI Bridge (`crates/kreuzberg-py/src/plugins.rs`)
Enable Python implementations of plugin traits:

```python
class MyCustomExtractor:
    def name(self) -> str:
        return "my-extractor"

    def version(self) -> str:
        return "1.0.0"

    async def extract_bytes(self, content: bytes, mime_type: str, config):
        # Custom extraction logic
        return ExtractionResult(...)

    def supported_mime_types(self) -> list[str]:
        return ["application/custom"]
```

Features:
- Class-based plugin definition in Python
- Async method support with proper GIL management
- Type hints for configuration objects
- Exception handling and error translation

### Plugin Registry Implementation
Example DocumentExtractorRegistry:

```rust
pub struct DocumentExtractorRegistry {
    extractors: Arc<RwLock<Vec<Arc<dyn DocumentExtractor>>>>,
    mime_type_index: Arc<RwLock<HashMap<String, Vec<usize>>>>, // MIME -> extractor indices
}

impl DocumentExtractorRegistry {
    pub fn register(&self, extractor: Arc<dyn DocumentExtractor>) -> Result<()>;
    pub fn unregister(&self, name: &str) -> Result<()>;
    pub fn get_for_mime(&self, mime_type: &str) -> Result<Arc<dyn DocumentExtractor>>;
    pub fn list_all(&self) -> Vec<PluginInfo>;
    pub fn clear(&self) -> Result<()>;
}
```

## Integration with Kreuzberg Architecture

### Document Extraction Pipeline
Plugins are invoked in the document extraction process:

```
1. Detect MIME type
2. Query DocumentExtractorRegistry for matching plugins
3. Sort plugins by priority (highest first)
4. Iterate through plugins attempting extraction
5. On success → return result
6. On failure → try next-priority plugin
7. If all fail → return error or partial result
```

### OCR Pipeline Integration
OCR plugins selected in image processing:

```
1. Receive image and TesseractConfig
2. Determine required OCR languages
3. Query OcrBackendRegistry for capable backends
4. Sort by priority and capability match
5. Select highest-priority capable backend
6. Execute async OCR operation
7. Cache result for future use
```

### Post-Processing Chain
Plugins applied sequentially to results:

```rust
pub struct PostProcessorConfig {
    pub enabled: bool,
    pub plugins: Vec<PostProcessorSpec>,
}

pub struct PostProcessorSpec {
    pub name: String,
    pub priority: u8,
    pub config: serde_json::Value,
}
```

### Python Plugin Registration
Python plugins registered from Rust-Python FFI:

```rust
pub fn register_ocr_backend(
    name: String,
    python_obj: Py<PyAny>,
    priority: u8,
) -> PyResult<()>;

pub fn register_post_processor(
    name: String,
    python_obj: Py<PyAny>,
    priority: u8,
) -> PyResult<()>;
```

## GIL Management for Python Plugins

### Critical GIL Patterns Used

1. **Temporary GIL Acquisition** (Python::attach)
   ```rust
   Python::attach(|py| {
       let result = python_obj.bind(py).call_method0("name")?;
       result.extract::<String>()
   })
   ```
   Use for quick operations (reading attributes, simple calls)

2. **GIL Release During Expensive Operations** (py.detach)
   ```rust
   py.detach(|| {
       let registry = get_registry();
       let mut registry = registry.write()?;
       registry.register(backend)
   })
   ```
   Use for I/O, lock acquisition, expensive computation

3. **Async Python Calls** (tokio::task::spawn_blocking)
   ```rust
   let python_obj = Python::attach(|py| python_obj.clone_ref(py));
   tokio::task::spawn_blocking(move || {
       Python::attach(|py| {
           let obj = python_obj.bind(py);
           obj.call_method1("process_image", (bytes, config))
       })
   }).await?
   ```
   Use for async trait implementations (must block during Python call)

4. **Caching to Minimize GIL Acquisitions**
   ```rust
   pub struct PythonOcrBackend {
       python_obj: Py<PyAny>,
       name: String,                    // Cached - no GIL
       supported_languages: Vec<String>, // Cached - no GIL
   }
   ```
   Cache frequently-accessed Python data in Rust

## Data Flow

### Plugin Registration
1. **Plugin Creation** → Instantiate plugin object (Rust or Python)
2. **Validation** → Verify plugin implements required trait
3. **Registry Lookup** → Find appropriate registry (Extractor, OCR, etc.)
4. **Priority Assignment** → Determine plugin priority level
5. **Index Update** → Update registries to include new plugin
6. **Initialization** → Call plugin.initialize() if needed

### Plugin Selection & Execution
1. **Capability Query** → Request plugins for specific capability
2. **Priority Sort** → Order matching plugins by priority (highest first)
3. **Fallback Chain** → Iterate through plugins until success
4. **Execution** → Invoke plugin with appropriate configuration
5. **Error Handling** → Catch exceptions and continue to next plugin
6. **Result Caching** → Store result (may vary by plugin type)

## Dependencies & Relationships

### Upstream Dependencies
- **Rust Core**: Base plugin traits and registries
- **PyO3**: Python FFI for Python plugins
- **async-trait**: Async trait support
- **serde**: Configuration serialization

### Downstream Dependencies
- **Document Extraction Domain**: Uses DocumentExtractor plugins for format support
- **OCR Integration Domain**: Uses OcrBackend plugins for text recognition
- **Post-Processing**: Applies PostProcessor plugins to enhance results
- **Validation**: Applies Validator plugins for quality assurance

## Extension Points for Users

### Creating a Custom Document Extractor
```rust
pub struct MyExtractor;

impl Plugin for MyExtractor {
    fn name(&self) -> &str { "my-extractor" }
    fn version(&self) -> String { "1.0.0".to_string() }
    fn initialize(&self) -> Result<()> { Ok(()) }
    fn shutdown(&self) -> Result<()> { Ok(()) }
}

#[async_trait]
impl DocumentExtractor for MyExtractor {
    async fn extract_bytes(&self, content: &[u8], mime_type: &str, config: &ExtractionConfig)
        -> Result<ExtractionResult> {
        // Custom extraction logic
    }

    fn supported_mime_types(&self) -> &[&str] {
        &["application/custom"]
    }

    fn priority(&self) -> u8 { 75 } // Override default
}
```

### Creating a Python OCR Backend
```python
class MyOcrBackend:
    def name(self) -> str:
        return "my-ocr-backend"

    def version(self) -> str:
        return "1.0.0"

    async def process_image(self, image_bytes: bytes, config: TesseractConfig) -> ExtractionResult:
        # Load your ML model
        # Process image
        # Return results
        pass

    def supported_languages(self) -> list[str]:
        return ["en", "de", "fr"]
```

## Performance Characteristics

- **Plugin Registration**: O(1) for Rust plugins, O(n) for Python plugin discovery
- **Plugin Selection**: O(log n) with indexed MIME type lookup and priority sorting
- **Fallback Execution**: Linear in number of plugins (short-circuit on success)
- **Registry Memory**: ~100 bytes per registered plugin
- **GIL Overhead**: ~5-55µs per Python method call (mitigated by caching)

## Testing & Validation

- **Plugin Interface Compliance**: Verify plugins implement all required methods
- **Priority Selection**: Test priority-based plugin arbitration
- **Fallback Chains**: Validate fallback behavior with multiple plugins
- **Error Handling**: Test error handling in plugin registration/execution
- **Python FFI**: Verify Python plugins work correctly through FFI bridge
- **GIL Management**: Validate GIL is released during expensive operations
- **Thread Safety**: Test concurrent plugin access from multiple threads
- **Performance**: Benchmark plugin selection overhead

## Future Enhancements

- Plugin dependency system with version resolution
- Plugin hot-reloading without process restart
- Plugin marketplace/registry for discovering third-party plugins
- Plugin performance profiling and optimization hints
- Declarative plugin configuration (TOML/YAML)
- Plugin sandboxing for security isolation
- Plugin versioning and compatibility checking
- Async plugin initialization with streaming results
