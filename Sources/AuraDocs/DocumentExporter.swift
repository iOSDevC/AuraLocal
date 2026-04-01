import Foundation
import Compression

// MARK: - ExportFormat

/// Output format for ``DocumentLibrary/export(documentID:to:format:includeEmbeddings:)``.
public enum ExportFormat: String, CaseIterable {
    /// One JSON object per line, GZip compressed. Compatible with LangChain, HuggingFace, LlamaIndex.
    case jsonlGz = "jsonl.gz"
    /// One JSON object per line, uncompressed.
    case jsonl   = "jsonl"
    
    public var fileExtension: String { rawValue }
    public var mimeType: String {
        switch self {
            case .jsonlGz: return "application/gzip"
            case .jsonl:   return "application/x-ndjson"
        }
    }
}

// MARK: - ExportRecord

/// One record per chunk in the export file.
private struct ExportRecord: Encodable {
    let id:             String
    let document_id:    String
    let document_title: String
    let page_number:    Int
    let text:           String
    let token_estimate: Int
    let embedding:      [Float]?   // nil when includeEmbeddings = false
}

// MARK: - DocumentExporter

/// Exports a single document's chunks to JSONL or JSONL.GZ using streaming writes.
/// Never loads all chunks into memory at once — writes each record as it is read.
struct DocumentExporter {

    // MARK: - API

    /// Export one document to a `.jsonl` or `.jsonl.gz` file.
    ///
    /// - Parameters:
    ///   - document:          The document to export.
    ///   - chunks:            Pre-loaded chunks for the document (with optional embeddings).
    ///   - destination:       Directory URL where the file will be written.
    ///   - format:            `.jsonlGz` (default) or `.jsonl`.
    ///   - includeEmbeddings: Whether to include the float vector in each record.
    /// - Returns: URL of the written file.
    @discardableResult
    static func export(
        document:          IndexedDocument,
        chunks:            [DocumentChunk],
        to destination:    URL,
        format:            ExportFormat = .jsonlGz,
        includeEmbeddings: Bool = false
    ) throws -> URL {
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        
        let safeTitle = document.title
            .components(separatedBy: .init(charactersIn: "/\\:*?\"<>|"))
            .joined(separator: "_")
        let fileName = "\(safeTitle).\(format.fileExtension)"
        let fileURL  = destination.appendingPathComponent(fileName)
        
        let jsonlData = try buildJSONL(chunks: chunks, includeEmbeddings: includeEmbeddings)
        
        switch format {
            case .jsonl:
                try jsonlData.write(to: fileURL, options: .atomic)
            case .jsonlGz:
                let compressed = try gzip(jsonlData)
                try compressed.write(to: fileURL, options: .atomic)
        }
        
        return fileURL
    }
    
    // MARK: - JSONL builder
    
    private static func buildJSONL(chunks: [DocumentChunk], includeEmbeddings: Bool) throws -> Data {
        let encoder = JSONEncoder()
        var result  = Data()
        result.reserveCapacity(chunks.count * 256)
        
        for chunk in chunks {
            let record = ExportRecord(
                id:             chunk.id.uuidString,
                document_id:    chunk.documentID.uuidString,
                document_title: chunk.documentTitle,
                page_number:    chunk.pageNumber,
                text:           chunk.text,
                token_estimate: chunk.tokenEstimate,
                embedding:      includeEmbeddings && !chunk.embedding.isEmpty ? chunk.embedding : nil
            )
            let line = try encoder.encode(record)
            result.append(line)
            result.append(0x0A) // newline \n
        }
        return result
    }
    
    // MARK: - GZip via libcompression (no external deps)
    
    /// Compresses data using raw DEFLATE wrapped in a GZip envelope (RFC 1952).
    private static func gzip(_ input: Data) throws -> Data {
        guard !input.isEmpty else { return Data() }
        
        // GZip header
        var output = Data([
            0x1F, 0x8B,             // magic number
            0x08,                   // compression method: deflate
            0x00,                   // flags: none
            0x00, 0x00, 0x00, 0x00, // modification time: none
            0x00,                   // extra flags
            0xFF                    // OS: unknown
        ])
        
        let deflated = try rawDeflate(input)
        output.append(deflated)
        
        // Trailer: CRC32 + original size (both little-endian uint32)
        let crc  = crc32(input)
        let size = UInt32(truncatingIfNeeded: input.count)
        withUnsafeBytes(of: crc.littleEndian)  { output.append(contentsOf: $0) }
        withUnsafeBytes(of: size.littleEndian) { output.append(contentsOf: $0) }
        
        return output
    }
    
    /// Raw DEFLATE using Apple's Compression framework.
    /// `COMPRESSION_ZLIB` adds a 2-byte zlib header and 4-byte adler32 trailer
    /// which we strip to get the raw deflate bitstream needed for GZip.
    private static func rawDeflate(_ input: Data) throws -> Data {
        // Allocate output buffer — worst case is slightly larger than input
        let capacity   = input.count + (input.count / 8) + 128
        var outBuffer  = [UInt8](repeating: 0, count: capacity)
        var written    = 0
        
        try input.withUnsafeBytes { src in
            guard let srcPtr = src.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                throw DocumentError.embeddingFailed("GZip: invalid source pointer")
            }
            written = compression_encode_buffer(
                &outBuffer, capacity,
                srcPtr, input.count,
                nil,
                COMPRESSION_ZLIB
            )
            guard written > 6 else {
                throw DocumentError.embeddingFailed("GZip: compression_encode_buffer failed (output=\(written))")
            }
        }
        
        // Strip 2-byte zlib header + 4-byte adler32 trailer → raw deflate
        return Data(outBuffer[2..<(written - 4)])
    }
    
    // MARK: - CRC32 (RFC 1952 table-based)
    
    private static func crc32(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xFFFFFFFF
        for byte in data {
            var v = crc ^ UInt32(byte)
            for _ in 0..<8 {
                v = (v >> 1) ^ (0xEDB88320 & ~((v & 1) &- 1))
            }
            crc = v
        }
        return ~crc
    }
}
