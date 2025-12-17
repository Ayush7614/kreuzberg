#!/usr/bin/env node
/**
 * Post-build script to fix missing type exports in generated .d.ts files
 * Ensures ExtractionConfig and ExtractionResult are exported from the main entry point
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const distDir = path.join(__dirname, "..", "dist");

/**
 * Fix type exports in a .d.ts or .d.mts file
 * @param {string} filePath - Path to the file to fix
 */
function fixTypeExports(filePath) {
	try {
		if (!fs.existsSync(filePath)) {
			console.warn(`File not found: ${filePath}`);
			return;
		}

		let content = fs.readFileSync(filePath, "utf-8");

		// Detect the actual types file name from imports
		let moduleRef = null;
		const moduleAlias = null;

		// Look for imports from the types file (could be types-*.mjs or types-*.d.mts)
		const typeImportMatch = content.match(/import\s+{([^}]+)}\s+from\s+['"](\.\/(types-[^\s'"]+))['"];?/);
		if (typeImportMatch) {
			const importPath = typeImportMatch[2];
			// Extract the module reference without extension
			const baseModule = importPath.replace(/\.(mjs|d\.mts|d\.ts|js)$/, "");
			// For both .mjs and .d.mts references, use .mjs
			moduleRef = `${baseModule}.mjs`;

			// Parse the named imports to get type aliases
			const imports = typeImportMatch[1];
			// This regex captures the mappings like "E as ExtractionConfig"
		}

		if (!moduleRef) {
			console.log(`- Could not determine types module for ${path.basename(filePath)}`);
			return;
		}

		// Build the corrected export statement with all types
		const correctedExport = `export { C as Chunk, b as ChunkMetadata, c as ChunkingConfig, d as ExtractedImage, I as ImageExtractionConfig, L as LanguageDetectionConfig, M as Metadata, O as OcrBackendProtocol, e as OcrConfig, P as PageContent, f as PageExtractionConfig, g as PdfConfig, h as PostProcessorConfig, T as Table, i as TesseractConfig, j as TokenReductionConfig, E as ExtractionConfig, a as ExtractionResult } from '${moduleRef}';`;

		// Find and replace the export statement that doesn't include ExtractionConfig
		const lines = content.split("\n");
		let replaced = false;
		let foundCorrectExport = false;

		for (let i = 0; i < lines.length; i++) {
			const line = lines[i];
			// Match any export from types-*.mjs
			if (line.startsWith("export {") && /from\s+['"]\.\/types-[^'"]+['"]/.test(line)) {
				// Check if it already has both key types
				if (line.includes("ExtractionConfig") && line.includes("ExtractionResult")) {
					foundCorrectExport = true;
				} else if (line.includes("from")) {
					// Replace with corrected export
					lines[i] = correctedExport;
					replaced = true;
				}
				break;
			}
		}

		if (replaced) {
			content = lines.join("\n");
			fs.writeFileSync(filePath, content);
			console.log(`✓ Fixed type exports in ${path.basename(filePath)}`);
		} else if (foundCorrectExport) {
			console.log(`✓ ${path.basename(filePath)} already has correct exports`);
		} else {
			console.log(`- No changes needed for ${path.basename(filePath)}`);
		}
	} catch (error) {
		console.error(`✗ Error fixing ${filePath}:`, error.message);
		process.exit(1);
	}
}

// Main execution
console.log("Fixing type exports in generated .d.ts files...\n");

fixTypeExports(path.join(distDir, "index.d.ts"));
fixTypeExports(path.join(distDir, "index.d.mts"));

console.log("\nType export fixes complete!");
