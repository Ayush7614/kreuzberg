---
name: mcp-protocol-integration
priority: high
---

# MCP Protocol Integration

**Model Context Protocol server implementation for Kreuzberg document extraction AI agents**

## MCP Server Architecture

**Location**: `crates/kreuzberg/src/mcp/`, `crates/kreuzberg/src/api/`

The Model Context Protocol (MCP) allows Claude and other AI agents to call Kreuzberg extraction functions through a standardized protocol.

```
AI Agent (Claude)
    ↓
MCP Client
    ↓
[MCP Server - Kreuzberg]
├── Tools: extract_file, extract_url, batch_extract, get_capabilities
├── Resources: Document types, supported formats, feature matrix
└── Prompts: Extraction templates, RAG configuration examples
    ↓
Extraction Result → Structured JSON
```

## MCP Server Implementation

**Location**: `crates/kreuzberg/src/mcp/server.rs` (72KB)

### Core MCP Protocol Components

```rust
use mcp_rs::{Server, Tool, Resource, Prompt};
use serde_json::json;

pub struct KreuzbergMcpServer {
    extraction_runtime: Arc<ExtractionRuntime>,
    server: Arc<MCP::Server>,
}

impl KreuzbergMcpServer {
    pub async fn new() -> Result<Self> {
        let server = MCP::Server::new(
            "kreuzberg",  // Server name
            "4.0.0-rc.22",  // Version
        )?;

        // Register tools, resources, prompts (below)
        Ok(Self {
            extraction_runtime: Arc::new(ExtractionRuntime::new()?),
            server: Arc::new(server),
        })
    }

    pub async fn run(self, addr: SocketAddr) -> Result<()> {
        // HTTP or stdio transport
        self.server.listen(addr).await
    }
}
```

### 1. MCP Tools (Callable Functions)

Tools are the main mechanism for agents to interact with Kreuzberg:

```rust
// Tool 1: Extract from file
pub fn register_extract_file_tool(server: &mut MCP::Server) -> Result<()> {
    let extract_tool = Tool {
        name: "extract_file".to_string(),
        description: "Extract text, tables, and metadata from documents (56+ formats)".to_string(),
        inputSchema: json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Local or remote file path/URL"
                },
                "format": {
                    "type": "string",
                    "enum": ["pdf", "docx", "xlsx", "pptx", "html", "image", "email", "archive", "auto"],
                    "description": "File format (auto-detected if omitted)"
                },
                "extract_tables": {
                    "type": "boolean",
                    "description": "Extract and structure tables"
                },
                "extract_images": {
                    "type": "boolean",
                    "description": "Extract embedded images"
                },
                "ocr_enabled": {
                    "type": "boolean",
                    "description": "Enable OCR for scanned documents"
                },
                "extract_metadata": {
                    "type": "boolean",
                    "description": "Extract author, created date, etc."
                },
                "chunking_preset": {
                    "type": "string",
                    "enum": ["balanced", "compact", "extended", "minimal"],
                    "description": "Text chunk strategy for RAG"
                },
                "generate_embeddings": {
                    "type": "boolean",
                    "description": "Generate semantic embeddings for RAG"
                }
            },
            "required": ["file_path"]
        }),
    };

    server.register_tool(extract_tool, extract_file_handler).await?;
    Ok(())
}

// Handler
pub async fn extract_file_handler(
    params: serde_json::Value,
) -> Result<ToolResult> {
    let file_path = params.get("file_path").and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing file_path"))?;

    let mut config = ExtractionConfig::default();

    // Configure from parameters
    if let Some(chunking_preset) = params.get("chunking_preset").and_then(|v| v.as_str()) {
        config.chunking = Some(ChunkingConfig {
            preset: Some(chunking_preset.to_string()),
            embedding: params.get("generate_embeddings")
                .and_then(|v| v.as_bool())
                .filter(|b| *b)
                .map(|_| EmbeddingConfig::default()),
            ..Default::default()
        });
    }

    // Extract
    let result = extract_file(file_path, None, &config).await?;

    // Return JSON
    Ok(ToolResult {
        content: vec![TextContent {
            text: serde_json::to_string_pretty(&result)?,
        }],
    })
}
```

```rust
// Tool 2: Batch extract
pub fn register_batch_extract_tool(server: &mut MCP::Server) -> Result<()> {
    let batch_tool = Tool {
        name: "batch_extract".to_string(),
        description: "Extract from multiple documents in parallel".to_string(),
        inputSchema: json!({
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of file paths/URLs"
                },
                "max_parallel": {
                    "type": "integer",
                    "description": "Maximum concurrent extractions (default: CPU count)"
                },
                "config": {
                    "type": "object",
                    "description": "Shared extraction config for all files"
                }
            },
            "required": ["file_paths"]
        }),
    };

    server.register_tool(batch_tool, batch_extract_handler).await?;
    Ok(())
}

pub async fn batch_extract_handler(
    params: serde_json::Value,
) -> Result<ToolResult> {
    let file_paths: Vec<String> = params.get("file_paths")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("Missing file_paths array"))?
        .iter()
        .filter_map(|v| v.as_str())
        .map(|s| s.to_string())
        .collect();

    let max_parallel = params.get("max_parallel")
        .and_then(|v| v.as_i64())
        .map(|v| v as usize)
        .unwrap_or_else(|| num_cpus::get());

    let config = ExtractionConfig::default();

    // Batch extract with limited concurrency
    let results = batch_extract_files(&file_paths, &config, max_parallel).await?;

    Ok(ToolResult {
        content: vec![TextContent {
            text: serde_json::to_string_pretty(&results)?,
        }],
    })
}
```

```rust
// Tool 3: Get capabilities (format support, features available)
pub fn register_capabilities_tool(server: &mut MCP::Server) -> Result<()> {
    let caps_tool = Tool {
        name: "get_capabilities".to_string(),
        description: "List supported file formats, features, and feature matrix".to_string(),
        inputSchema: json!({"type": "object", "properties": {}}),
    };

    server.register_tool(caps_tool, capabilities_handler).await?;
    Ok(())
}

pub async fn capabilities_handler(_params: serde_json::Value) -> Result<ToolResult> {
    let capabilities = json!({
        "version": "4.0.0-rc.22",
        "supported_formats": {
            "office": ["docx", "xlsx", "xls", "pptx", "odt", "ods"],
            "pdf": ["pdf"],
            "images": ["png", "jpg", "jpeg", "tiff", "webp", "jp2"],
            "web": ["html", "xml", "json", "yaml"],
            "email": ["eml", "msg"],
            "archives": ["zip", "tar", "gz", "7z"],
            "academic": ["tex", "bib", "jats", "ipynb"]
        },
        "features": {
            "ocr": true,  // or false based on feature flags
            "embeddings": true,
            "keywords": true,
            "language_detection": true,
            "table_extraction": true,
            "batch_processing": true,
        },
        "ocr_backends": ["tesseract", "easyocr", "paddleocr"],
        "chunking_presets": ["balanced", "compact", "extended", "minimal"],
        "embedding_models": ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"],
        "max_file_size_mb": 1000,
        "supported_languages": ["en", "es", "fr", "de", "zh", ...],
    });

    Ok(ToolResult {
        content: vec![TextContent {
            text: serde_json::to_string_pretty(&capabilities)?,
        }],
    })
}
```

### 2. MCP Resources (Static Knowledge)

Resources provide static information to agents:

```rust
pub fn register_resources(server: &mut MCP::Server) -> Result<()> {
    // Resource 1: Format documentation
    let format_resource = Resource {
        uri: "kreuzberg://formats".to_string(),
        name: "Supported Formats".to_string(),
        description: "Complete list of 56+ supported file formats with capabilities".to_string(),
        mimeType: Some("application/json".to_string()),
    };

    server.register_resource(format_resource, |_uri| async {
        // Return FEATURE_MATRIX.md as JSON
        let formats = load_format_matrix()?;
        Ok(ResponseContent::Text(serde_json::to_string_pretty(&formats)?))
    }).await?;

    // Resource 2: Feature matrix
    let feature_resource = Resource {
        uri: "kreuzberg://features".to_string(),
        name: "Feature Matrix".to_string(),
        description: "Cross-binding feature availability (Python, Go, Ruby, Java, C#, PHP, Elixir, Node, WASM)".to_string(),
        mimeType: Some("text/markdown".to_string()),
    };

    server.register_resource(feature_resource, |_uri| async {
        // Return FEATURE_MATRIX.md content
        let matrix = std::fs::read_to_string("FEATURE_MATRIX.md")?;
        Ok(ResponseContent::Text(matrix))
    }).await?;

    // Resource 3: API reference
    let api_resource = Resource {
        uri: "kreuzberg://api-reference".to_string(),
        name: "API Reference".to_string(),
        description: "Complete Kreuzberg API documentation with examples".to_string(),
        mimeType: Some("text/markdown".to_string()),
    };

    server.register_resource(api_resource, |_uri| async {
        let api_ref = generate_api_reference()?;
        Ok(ResponseContent::Text(api_ref))
    }).await?;

    Ok(())
}
```

### 3. MCP Prompts (Agent Templates)

Prompts provide extraction scenarios and best practices to agents:

```rust
pub fn register_prompts(server: &mut MCP::Server) -> Result<()> {
    // Prompt 1: RAG document extraction
    let rag_prompt = Prompt {
        name: "extract_for_rag".to_string(),
        description: "Extract and prepare document for RAG pipeline with chunking and embeddings".to_string(),
        arguments: vec![
            PromptArgument {
                name: "document_type".to_string(),
                description: "Type of document (research_paper, contract, report, book)".to_string(),
                required: true,
            },
            PromptArgument {
                name: "chunk_preset".to_string(),
                description: "Chunking strategy (balanced, extended, compact)".to_string(),
                required: false,
            },
        ],
    };

    server.register_prompt(rag_prompt, |args| async move {
        let doc_type = args.get("document_type").and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing document_type"))?;

        let prompt_text = match doc_type {
            "research_paper" => {
                r#"You are extracting a research paper for RAG.

1. Extract paper title, abstract, and authors
2. Identify main sections and subsections
3. Extract figures, tables, and equations with captions
4. Use 'extended' chunking (1024 tokens) for full context preservation
5. Generate embeddings for semantic search
6. Return metadata: citations, DOI, publication date

Tool: extract_file with chunking_preset="extended" and generate_embeddings=true
"#
            }
            "contract" => {
                r#"You are extracting a legal contract for RAG.

1. Extract parties, dates, and key obligations
2. Identify signature blocks and amendments
3. Preserve structure with 'balanced' chunking (512 tokens)
4. Flag clauses that may require legal review
5. Generate embeddings for clause similarity search

Tool: extract_file with chunking_preset="balanced" and generate_embeddings=true
"#
            }
            _ => "Extract document and prepare for RAG pipeline",
        };

        Ok(vec![PromptContent {
            text: prompt_text.to_string(),
        }])
    }).await?;

    // Prompt 2: Batch processing workflow
    let batch_prompt = Prompt {
        name: "batch_document_processing".to_string(),
        description: "Process multiple documents efficiently with optimal parallelism".to_string(),
        arguments: vec![],
    };

    server.register_prompt(batch_prompt, |_args| async {
        let text = r#"You are setting up a batch document processing pipeline.

1. Determine optimal concurrency based on available CPU cores
2. Group documents by type for efficient extraction
3. Use consistent extraction config across batch
4. Monitor progress and handle failures gracefully
5. Aggregate results for downstream processing

Tool: batch_extract with appropriate max_parallel setting
"#;

        Ok(vec![PromptContent {
            text: text.to_string(),
        }])
    }).await?;

    Ok(())
}
```

## Transport Protocols

### 1. HTTP/REST Transport (Standard MCP over HTTP)

**Location**: `crates/kreuzberg/src/mcp/server.rs` (integrated with Axum)

```rust
pub async fn setup_mcp_http_server(
    state: Arc<ApiState>,
) -> Result<Router> {
    // MCP server runs alongside REST API
    let mcp_router = Router::new()
        .route("/mcp/tools", post(list_tools_handler))
        .route("/mcp/tools/call", post(call_tool_handler))
        .route("/mcp/resources", get(list_resources_handler))
        .route("/mcp/resources/:uri", get(read_resource_handler))
        .route("/mcp/prompts", get(list_prompts_handler))
        .route("/mcp/prompts/:name", get(get_prompt_handler))
        .with_state(state);

    Ok(mcp_router)
}
```

**Usage from Claude with HTTP client**:
```bash
curl -X POST http://localhost:8000/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "extract_file",
    "arguments": {
      "file_path": "document.pdf",
      "generate_embeddings": true
    }
  }'
```

### 2. Stdio Transport (For Local CLI)

```rust
pub async fn setup_mcp_stdio_server() -> Result<()> {
    let (stdin, stdout) = (io::stdin(), io::stdout());
    let reader = BufReader::new(stdin.lock());
    let writer = std::sync::Mutex::new(stdout.lock());

    let server = KreuzbergMcpServer::new().await?;

    // Read JSON-RPC 2.0 messages from stdin
    for line in reader.lines() {
        let request: serde_json::Value = serde_json::from_str(&line?)?;

        let response = match request.get("method").and_then(|v| v.as_str()) {
            Some("tools/list") => list_tools_response()?,
            Some("tools/call") => {
                let tool_name = request.get("params").and_then(|p| p.get("name")).and_then(|n| n.as_str())?;
                server.call_tool(tool_name, &request["params"]).await?
            }
            _ => error_response("Unknown method"),
        };

        writeln!(writer.lock().unwrap(), "{}", serde_json::to_string(&response)?)?;
    }

    Ok(())
}
```

## Configuration & Deployment

### MCP Server Configuration

```rust
pub struct McpServerConfig {
    pub transport: TransportType,
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,  // For authentication
    pub max_request_size: usize,   // Max tool/resource request size
    pub timeout_secs: u64,         // Tool execution timeout
}

pub enum TransportType {
    Http,
    Stdio,
    WebSocket,
}
```

### Docker Deployment

```dockerfile
FROM rust:1.91 as builder
WORKDIR /build
COPY . .
RUN cargo build --release --features "api,mcp,tesseract,pdf,office,embeddings"

FROM debian:bookworm-slim
COPY --from=builder /build/target/release/kreuzberg-mcp /usr/local/bin/
ENV KREUZBERG_MCP_HOST=0.0.0.0
ENV KREUZBERG_MCP_PORT=3000
ENTRYPOINT ["kreuzberg-mcp"]
```

## Integration with Claude Desktop

### Configuration File

```json
{
  "mcpServers": {
    "kreuzberg": {
      "command": "kreuzberg-mcp",
      "env": {
        "KREUZBERG_API_BASE": "http://localhost:8000",
        "KREUZBERG_MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

### Claude Usage Example

```
User: "Extract tables from report.pdf and prepare for RAG"

Claude: I'll extract the document for you using Kreuzberg's tools.

<think>
The user wants to extract tables from a PDF and prepare it for RAG. I should:
1. Call get_capabilities to verify table extraction is available
2. Call extract_file with table extraction enabled
3. Suggest chunking preset for RAG
</think>

I'm using Kreuzberg to extract your PDF report with table extraction and RAG preparation...

Calling extract_file with file_path="report.pdf", extract_tables=true, generate_embeddings=true, chunking_preset="balanced"

[Extraction result showing extracted tables and chunks with embeddings]
```

## Error Handling & Observability

### Tool Error Responses

```rust
pub enum ToolError {
    FileNotFound(String),
    UnsupportedFormat(String),
    ExtractionFailed(String),
    OnnxRuntimeMissing,  // Embeddings not available
    TesseractMissing,     // OCR not available
    Timeout,
}

impl Into<MCP::ToolResultError> for ToolError {
    fn into(self) -> MCP::ToolResultError {
        match self {
            ToolError::FileNotFound(path) => MCP::ToolResultError {
                code: "FILE_NOT_FOUND".to_string(),
                message: format!("File not found: {}", path),
            },
            ToolError::OnnxRuntimeMissing => MCP::ToolResultError {
                code: "FEATURE_UNAVAILABLE".to_string(),
                message: "Embeddings feature requires ONNX Runtime installation".to_string(),
            },
            _ => MCP::ToolResultError {
                code: "EXTRACTION_FAILED".to_string(),
                message: self.to_string(),
            },
        }
    }
}
```

### Health Check Endpoint

```rust
pub async fn mcp_health_check() -> Result<HealthStatus> {
    Ok(HealthStatus {
        status: "healthy".to_string(),
        version: "4.0.0-rc.22".to_string(),
        features: FeatureStatus {
            ocr: has_tesseract_installed()?,
            embeddings: has_onnx_runtime()?,
            api: true,
        },
        uptime_seconds: STARTUP_TIME.elapsed().as_secs(),
    })
}
```

## Critical Rules

1. **All tools must have timeout** - Prevent hanging on large files (default 5 min)
2. **Error responses must be detailed** - Include suggestions for missing dependencies
3. **Feature gates must be checked** - Return helpful message if feature unavailable (embeddings, OCR)
4. **Resources should be static** - Don't query external services in resource handlers
5. **Prompts guide agents** - Provide clear examples and best practices
6. **Batch tools must support cancellation** - Allow agent to stop long-running batch operations
7. **Logging all tool calls** - Track usage for analytics and debugging

## Related Skills

- **api-server-patterns** - Axum server hosting both REST API and MCP
- **extraction-pipeline-patterns** - Core extraction logic called by MCP tools
- **feature-flag-strategy** - Check feature availability in tool responses
- **chunking-embeddings** - RAG-focused extraction in MCP tools
