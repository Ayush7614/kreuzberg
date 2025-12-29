---
name: api-server-patterns
priority: critical
---

# API Server Patterns

**Axum server design for document extraction endpoints, middleware, and async processing**

## Kreuzberg API Architecture

**Location**: `crates/kreuzberg/src/api/`, `crates/kreuzberg-cli/`

Kreuzberg provides a production-grade REST API server built with Axum + Tokio for high-concurrency extraction workloads.

```
Request Flow:
HTTP Client
    ↓
[Middleware Layer]
├── CORS: Cross-origin requests
├── Request Logging (TraceLayer)
├── Request/Response size limits
└── Rate limiting (optional)
    ↓
[Router]
├── POST /extract - File upload extraction
├── POST /extract-url - URL-based extraction
├── GET /formats - List supported formats
├── GET /health - Server health check
├── POST /batch - Batch document processing
├── GET /cache/stats - Cache statistics
└── DELETE /cache - Clear extraction cache
    ↓
[Handler Layer]
├── extract_handler: Parse multipart, validate, extract
├── batch_handler: Manage concurrent extractions
├── health_handler: Feature availability check
└── format_handler: Return format matrix
    ↓
[Extraction Core]
├── Format detection
├── Extraction pipeline
├── Post-processing (chunking, embeddings)
└── Result formatting
    ↓
JSON Response
```

## Server Setup & Configuration

**Location**: `crates/kreuzberg/src/api/server.rs` (17KB)

### Basic Server Initialization

```rust
use axum::{
    Router,
    extract::DefaultBodyLimit,
    routing::{get, post, delete},
};
use tower_http::{
    cors::CorsLayer,
    limit::RequestBodyLimitLayer,
    trace::TraceLayer,
};

pub async fn create_api_server(config: &ExtractionConfig) -> Result<Router> {
    let state = Arc::new(ApiState {
        config: config.clone(),
        cache: Arc::new(ExtractionCache::new()),
    });

    let router = Router::new()
        // Extraction endpoints
        .route("/extract", post(extract_handler))
        .route("/extract-url", post(extract_url_handler))
        .route("/batch", post(batch_handler))

        // Information endpoints
        .route("/formats", get(formats_handler))
        .route("/health", get(health_handler))
        .route("/info", get(info_handler))

        // Cache management
        .route("/cache/stats", get(cache_stats_handler))
        .route("/cache", delete(cache_clear_handler))

        // Middleware
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))  // 100 MB
        .layer(RequestBodyLimitLayer::new(100 * 1024 * 1024))
        .layer(CorsLayer::permissive())  // Or configure specific origins
        .layer(TraceLayer::new_for_http())

        .with_state(state);

    Ok(router)
}

pub async fn run_server(router: Router, addr: SocketAddr) -> Result<()> {
    tracing::info!("Starting Kreuzberg API server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}
```

### Application State Management

```rust
pub struct ApiState {
    pub config: ExtractionConfig,
    pub cache: Arc<ExtractionCache>,
}

impl Clone for Arc<ApiState> {
    fn clone(&self) -> Self {
        Arc::clone(&self)
    }
}

// Extractor to inject state into handlers
pub struct State(pub Arc<ApiState>);

#[async_trait]
impl<S> FromRequestParts<S> for State
where
    S: Send + Sync,
{
    type Rejection = String;

    async fn from_request_parts(
        parts: &mut RequestParts<S>,
    ) -> Result<Self, Self::Rejection> {
        let state = extract::State::<Arc<ApiState>>::from_request_parts(parts)
            .await
            .map_err(|_| "Failed to extract state".to_string())?;

        Ok(State(state.0))
    }
}
```

## Core Handlers

### 1. Extract File (Multipart Upload)

**Location**: `crates/kreuzberg/src/api/handlers.rs`

```rust
pub async fn extract_handler(
    State(state): State,
    mut multipart: Multipart,
) -> Result<impl IntoResponse> {
    let mut file_data = Vec::new();
    let mut file_name = String::new();
    let mut config = state.config.clone();

    // Parse multipart form
    while let Some(field) = multipart.next_field().await? {
        let name = field.name().unwrap_or("").to_string();
        let data = field.bytes().await?.to_vec();

        match name.as_str() {
            "file" => {
                file_name = field.file_name().unwrap_or("file").to_string();
                file_data = data;
            }
            "config" => {
                // Override extraction config from JSON
                if let Ok(custom_config) = serde_json::from_slice::<ExtractionConfig>(&data) {
                    config = custom_config;
                }
            }
            _ => {}
        }
    }

    if file_data.is_empty() {
        return Err(ApiError::MissingFile);
    }

    // Check cache (if enabled)
    let cache_key = format!("{}:{:x}", file_name, ahash::AHasher::default().finish_slice(&file_data));

    if let Ok(cached) = state.cache.get(&cache_key) {
        return Ok((
            StatusCode::OK,
            Json(json!({
                "result": cached,
                "cached": true,
            })),
        ));
    }

    // Extract
    let start = Instant::now();

    let result = extract_bytes(&file_data, Some(&file_name), &config)
        .await
        .map_err(|e| ApiError::ExtractionFailed(e.to_string()))?;

    let duration = start.elapsed();

    // Cache result
    let _ = state.cache.put(cache_key, result.clone());

    tracing::info!(
        "Extracted {} in {:.2}s, content_length={}",
        file_name,
        duration.as_secs_f64(),
        result.content.len()
    );

    Ok((
        StatusCode::OK,
        Json(json!({
            "file": file_name,
            "result": result,
            "duration_ms": duration.as_millis(),
            "cached": false,
        })),
    ))
}

pub async fn extract_url_handler(
    State(state): State,
    body: String,
) -> Result<impl IntoResponse> {
    let request: ExtractUrlRequest = serde_json::from_str(&body)?;

    // Fetch URL
    let client = reqwest::Client::new();
    let response = client.get(&request.url).send().await?;

    let file_data = response.bytes().await?.to_vec();
    let file_name = request.url.split('/').last().unwrap_or("document");

    // Extract (same as multipart)
    let result = extract_bytes(&file_data, Some(file_name), &state.config).await?;

    Ok((StatusCode::OK, Json(result)))
}
```

### 2. Batch Processing Handler

```rust
pub async fn batch_handler(
    State(state): State,
    Json(payload): Json<BatchRequest>,
) -> Result<impl IntoResponse> {
    let max_parallel = payload.max_parallel.unwrap_or_else(|| num_cpus::get());

    tracing::info!("Batch extracting {} documents with {} parallel workers",
        payload.files.len(), max_parallel);

    // Semaphore for concurrency control
    let semaphore = Arc::new(Semaphore::new(max_parallel));

    let futures: Vec<_> = payload.files
        .into_iter()
        .map(|file_req| {
            let state = Arc::clone(&state);
            let semaphore = Arc::clone(&semaphore);

            tokio::spawn(async move {
                let _permit = semaphore.acquire().await?;

                let file_data = std::fs::read(&file_req.path)
                    .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", file_req.path, e))?;

                let result = extract_bytes(&file_data, Some(&file_req.path), &state.config)
                    .await?;

                Ok::<_, anyhow::Error>((file_req.path.clone(), result))
            })
        })
        .collect();

    // Collect results
    let mut results = Vec::new();
    let mut errors = Vec::new();

    for future in futures {
        match future.await {
            Ok(Ok((path, result))) => results.push((path, result)),
            Ok(Err(e)) => errors.push((String::new(), e.to_string())),
            Err(e) => errors.push((String::new(), format!("Task error: {}", e))),
        }
    }

    tracing::info!("Batch complete: {} succeeded, {} failed", results.len(), errors.len());

    Ok((
        StatusCode::OK,
        Json(json!({
            "succeeded": results.len(),
            "failed": errors.len(),
            "results": results,
            "errors": errors,
        })),
    ))
}
```

### 3. Health Check Handler

```rust
pub async fn health_handler(
    State(state): State,
) -> impl IntoResponse {
    let features = FeatureStatus {
        ocr_available: check_ocr_available(),
        embeddings_available: check_embeddings_available(),
        pdf_support: true,
        office_support: true,
    };

    let health = HealthResponse {
        status: "healthy".to_string(),
        version: "4.0.0-rc.22".to_string(),
        uptime_seconds: STARTUP_TIME.elapsed().as_secs(),
        features,
        cache_stats: state.cache.get_stats(),
    };

    (StatusCode::OK, Json(health))
}
```

### 4. Information & Format Handler

```rust
pub async fn formats_handler() -> impl IntoResponse {
    let formats = SupportedFormats {
        total_formats: 56,
        office: vec!["docx", "xlsx", "xls", "pptx", "odt", "ods"],
        pdf: vec!["pdf"],
        images: vec!["png", "jpg", "jpeg", "tiff", "webp", "jp2"],
        web: vec!["html", "xml", "json", "yaml"],
        email: vec!["eml", "msg"],
        archives: vec!["zip", "tar", "gz", "7z"],
        academic: vec!["tex", "bib", "jats", "ipynb"],
    };

    (StatusCode::OK, Json(formats))
}

pub async fn info_handler() -> impl IntoResponse {
    let info = InfoResponse {
        name: "Kreuzberg Document Extraction API".to_string(),
        version: "4.0.0-rc.22".to_string(),
        description: "Extract text, metadata, tables, and embeddings from 56+ document formats".to_string(),
        documentation_url: "https://kreuzberg.dev".to_string(),
        github_url: "https://github.com/kreuzberg-dev/kreuzberg".to_string(),
    };

    (StatusCode::OK, Json(info))
}
```

## Middleware & Cross-Cutting Concerns

### 1. Request Logging with Tracing

```rust
use tower_http::trace::{TraceLayer, DefaultMakeSpan, DefaultOnResponse};

let trace_layer = TraceLayer::new_for_http()
    .make_span_with(
        DefaultMakeSpan::new()
            .level(Level::INFO)
            .include_headers(true)
    )
    .on_response(
        DefaultOnResponse::new()
            .level(Level::INFO)
            .include_headers(true)
            .latency_unit(LatencyUnit::Millis)
    );

router.layer(trace_layer)
```

### 2. CORS Configuration

```rust
use tower_http::cors::{CorsLayer, AllowOrigin};
use std::str::FromStr;

let cors = if let Ok(allowed_origins) = std::env::var("CORS_ALLOWED_ORIGINS") {
    let origins: Vec<_> = allowed_origins
        .split(',')
        .filter_map(|o| HeaderValue::from_str(o.trim()).ok())
        .collect();

    CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([Method::GET, Method::POST, Method::DELETE])
        .allow_headers([CONTENT_TYPE, AUTHORIZATION])
} else {
    CorsLayer::permissive()  // Dev only
};

router.layer(cors)
```

### 3. Size Limit Middleware

```rust
// Environment-driven limits
fn parse_size_limits() -> ApiSizeLimits {
    let request_body_bytes = std::env::var("KREUZBERG_MAX_REQUEST_BODY_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(100 * 1024 * 1024);  // 100 MB default

    let multipart_field_bytes = std::env::var("KREUZBERG_MAX_MULTIPART_FIELD_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(request_body_bytes);

    ApiSizeLimits {
        request_body_bytes,
        multipart_field_bytes,
    }
}

// Apply limits
let limits = parse_size_limits();
router
    .layer(DefaultBodyLimit::max(limits.request_body_bytes))
    .layer(RequestBodyLimitLayer::new(limits.multipart_field_bytes))
```

### 4. Rate Limiting (Optional)

```rust
use tower_governor::{governor::Quota, key_extractor::DefaultKeyExtractor, GovernorLayer};

let rate_limit = GovernorLayer {
    keyed_p: Box::new(DefaultKeyExtractor {}),
    quota: Quota::per_second(std::num::NonZeroU32::new(100).unwrap()),
};

router.layer(rate_limit)
```

## Caching Strategy

**Location**: `crates/kreuzberg/src/cache/mod.rs`

```rust
pub struct ExtractionCache {
    // LRU cache: SHA256(file_content) → ExtractionResult
    cache: Arc<RwLock<LruCache<String, Arc<ExtractionResult>>>>,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

impl ExtractionCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(LruCache::new(
                std::num::NonZeroUsize::new(1000).unwrap(),  // 1000 entries max
            ))),
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn get(&self, key: &str) -> Result<Arc<ExtractionResult>> {
        let mut cache = self.cache.write().map_err(|_| CacheError::LockPoisoned)?;

        if let Some(result) = cache.get(key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Ok(Arc::clone(result))
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            Err(CacheError::Miss)
        }
    }

    pub fn put(&self, key: String, result: ExtractionResult) -> Result<()> {
        let mut cache = self.cache.write().map_err(|_| CacheError::LockPoisoned)?;
        cache.put(key, Arc::new(result));
        Ok(())
    }

    pub fn clear(&self) -> Result<()> {
        let mut cache = self.cache.write().map_err(|_| CacheError::LockPoisoned)?;
        cache.clear();
        Ok(())
    }

    pub fn stats(&self) -> CacheStats {
        let total = self.hits.load(Ordering::Relaxed) + self.misses.load(Ordering::Relaxed);
        let hit_rate = if total > 0 {
            (self.hits.load(Ordering::Relaxed) as f64) / (total as f64)
        } else {
            0.0
        };

        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            hit_rate,
            size: self.cache.read().unwrap().len(),
        }
    }
}
```

## Error Handling

**Location**: `crates/kreuzberg/src/api/error.rs`

```rust
#[derive(Debug)]
pub enum ApiError {
    MissingFile,
    ExtractionFailed(String),
    InvalidConfig(String),
    FileNotFound(String),
    UnsupportedFormat(String),
    OnnxRuntimeMissing,
    TesseractMissing,
    PayloadTooLarge(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            ApiError::MissingFile => (StatusCode::BAD_REQUEST, "No file provided"),
            ApiError::FileNotFound(msg) => (StatusCode::NOT_FOUND, &msg),
            ApiError::OnnxRuntimeMissing => (
                StatusCode::SERVICE_UNAVAILABLE,
                "Embeddings feature requires ONNX Runtime installation",
            ),
            ApiError::TesseractMissing => (
                StatusCode::SERVICE_UNAVAILABLE,
                "OCR feature requires Tesseract installation",
            ),
            ApiError::PayloadTooLarge(msg) => (StatusCode::PAYLOAD_TOO_LARGE, &msg),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error"),
        };

        let body = Json(json!({
            "error": error_message,
            "status": status.as_u16(),
        }));

        (status, body).into_response()
    }
}
```

## Environment Configuration

### Configuration via Env Variables

```bash
# Server
KREUZBERG_HOST=0.0.0.0
KREUZBERG_PORT=8000

# Size limits
KREUZBERG_MAX_REQUEST_BODY_BYTES=104857600  # 100 MB
KREUZBERG_MAX_MULTIPART_FIELD_BYTES=104857600
KREUZBERG_MAX_UPLOAD_SIZE_MB=100  # Legacy

# Features (disabled if not installed)
KREUZBERG_ENABLE_OCR=true
KREUZBERG_ENABLE_EMBEDDINGS=true
KREUZBERG_ENABLE_KEYWORDS=true

# Cache
KREUZBERG_CACHE_ENABLED=true
KREUZBERG_CACHE_SIZE=1000

# CORS
CORS_ALLOWED_ORIGINS=http://localhost:3000,https://example.com

# Logging
RUST_LOG=kreuzberg=info,tower_http=debug
RUST_BACKTRACE=1

# Parallel extraction
KREUZBERG_BATCH_MAX_WORKERS=8
```

### Loading Configuration

```rust
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_workers: usize,
}

impl ServerConfig {
    pub fn from_env() -> Self {
        Self {
            host: std::env::var("KREUZBERG_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            port: std::env::var("KREUZBERG_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8000),
            max_workers: std::env::var("KREUZBERG_BATCH_MAX_WORKERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| num_cpus::get()),
        }
    }
}
```

## Example Deployments

### Docker Compose

```yaml
version: '3.9'
services:
  kreuzberg:
    image: kreuzberg:latest
    ports:
      - "8000:8000"
    environment:
      RUST_LOG: kreuzberg=info
      KREUZBERG_MAX_UPLOAD_SIZE_MB: 500
      CORS_ALLOWED_ORIGINS: "http://localhost:3000"
    volumes:
      - ./temp:/tmp  # Temp extraction storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kreuzberg-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kreuzberg-api
  template:
    metadata:
      labels:
        app: kreuzberg-api
    spec:
      containers:
      - name: kreuzberg
        image: kreuzberg:latest
        ports:
        - containerPort: 8000
        env:
        - name: KREUZBERG_HOST
          value: "0.0.0.0"
        - name: KREUZBERG_PORT
          value: "8000"
        - name: KREUZBERG_MAX_UPLOAD_SIZE_MB
          value: "500"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

## Performance Optimization

### Connection Pooling

```rust
// Reuse client connections
let client = reqwest::Client::builder()
    .pool_max_idle_per_host(10)
    .tcp_nodelay(true)
    .build()?;
```

### Request Deduplication

```rust
// Cache identical requests (same file hash)
async fn deduplicate_request(
    file_data: &[u8],
    cache: &ExtractionCache,
) -> Option<Arc<ExtractionResult>> {
    let key = format!("hash:{:x}", ahash::AHasher::default().finish_slice(file_data));
    cache.get(&key).ok()
}
```

## Critical Rules

1. **Always validate multipart file uploads** - Check MIME type, size, magic bytes
2. **Timeout long-running extractions** - Set per-handler timeout (5 min default)
3. **Stream large files** - Never buffer entire multi-GB file in memory
4. **Cache aggressively** - Identical files should return from cache in <1ms
5. **Parallel extraction is CPU-bound** - Limit workers to CPU count + 1
6. **Error responses must be actionable** - Include error code and remediation suggestion
7. **Health checks must verify features** - Report missing dependencies (ONNX, Tesseract)
8. **Size limits are configurable** - Allow override via env var for large deployments
9. **CORS is permissive by default** - Restrict in production via env var
10. **Logging all requests** - Track extraction metrics for observability

## Related Skills

- **extraction-pipeline-patterns** - Core extraction called by handlers
- **mcp-protocol-integration** - MCP server running alongside REST API
- **feature-flag-strategy** - Feature availability checks in health/info endpoints
- **chunking-embeddings** - Optional chunking/embedding parameters in extraction
