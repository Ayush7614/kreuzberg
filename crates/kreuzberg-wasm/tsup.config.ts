import { defineConfig } from "tsup";

export default defineConfig({
	entry: [
		"typescript/index.ts",
		"typescript/runtime.ts",
		"typescript/adapters/wasm-adapter.ts",
		"typescript/ocr/registry.ts",
		"typescript/ocr/tesseract-wasm-backend.ts",
	],
	// ESM only - CJS is not supported due to top-level await in WASM initialization
	// Modern Node.js (>= 14), Deno, and browsers all support ESM natively
	format: ["esm"],
	bundle: true,
	dts: {
		compilerOptions: {
			skipLibCheck: true,
			skipDefaultLibCheck: true,
			verbatimModuleSyntax: true,
		},
	},
	splitting: false,
	sourcemap: true,
	clean: true,
	shims: false,
	platform: "neutral",
	target: "es2022",
	external: [
		"@kreuzberg/core",
		"tesseract-wasm",
		"fs",
		"node:fs/promises",
		"path",
		"node:path",
		"url",
		"node:url",
		"util",
		"node:util",
		// WASM module - keep external to avoid bundling Node.js fs dependency
		// The wasm-pack generated module uses require('fs') which cannot be bundled
		// for neutral platform targets
		"../pkg/kreuzberg_wasm.js",
		"./pkg/kreuzberg_wasm.js",
		"./kreuzberg_wasm.js",
		/\.wasm$/,
		/@kreuzberg\/wasm-.*/,
		"./index.js",
		"../index.js",
	],
});
