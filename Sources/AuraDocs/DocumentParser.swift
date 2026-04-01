import Foundation
import PDFKit
import AuraCore

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - ParsedDocument

struct ParsedDocument: Sendable {
    let url: URL
    let title: String
    let pages: [ParsedPage]
    var fullText: String { pages.map(\.text).joined(separator: "\n\n") }
}

struct ParsedPage: Sendable {
    let pageNumber: Int   // 1-based, 0 = no page concept
    let text: String
}

// MARK: - DocumentParser protocol

protocol DocumentParser {
    func canParse(url: URL) -> Bool
    func parse(url: URL) async throws -> ParsedDocument
}

// MARK: - DocumentParserDispatcher

/// Tries each registered parser in order, throws if none can handle the URL.
struct DocumentParserDispatcher {
    private var parsers: [DocumentParser]
    
    init(visionLLM: AuraLocal? = nil) {
        parsers = [
            PDFDocumentParser(),
            DocxDocumentParser(),
            PlainTextDocumentParser(),
        ]
        if let vlm = visionLLM {
            parsers.append(ImageDocumentParser(llm: vlm))
        }
    }
    
    func parse(url: URL) async throws -> ParsedDocument {
        for parser in parsers {
            if parser.canParse(url: url) {
                return try await parser.parse(url: url)
            }
        }
        throw DocumentError.unsupportedFormat(url.pathExtension)
    }
    
    static var supportedExtensions: [String] {
        ["pdf", "docx", "txt", "md", "markdown", "png", "jpg", "jpeg", "heic", "tiff"]
    }
}

// MARK: - PDFDocumentParser

struct PDFDocumentParser: DocumentParser {
    func canParse(url: URL) -> Bool {
        url.pathExtension.lowercased() == "pdf"
    }
    
    func parse(url: URL) async throws -> ParsedDocument {
        guard let pdf = PDFDocument(url: url) else {
            throw DocumentError.parseFailed("Cannot open PDF at \(url.lastPathComponent)")
        }
        
        var pages: [ParsedPage] = []
        for i in 0..<pdf.pageCount {
            guard let page = pdf.page(at: i) else { continue }
            let text = page.string ?? ""
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
            pages.append(ParsedPage(pageNumber: i + 1, text: text))
        }
        
        let title = pdf.documentAttributes?[PDFDocumentAttribute.titleAttribute] as? String
        ?? url.deletingPathExtension().lastPathComponent
        
        return ParsedDocument(url: url, title: title, pages: pages)
    }
}

// MARK: - DocxDocumentParser
// Reads .docx as a ZIP archive and extracts word/document.xml — no external deps.

struct DocxDocumentParser: DocumentParser {
    func canParse(url: URL) -> Bool {
        url.pathExtension.lowercased() == "docx"
    }
    
    func parse(url: URL) async throws -> ParsedDocument {
        let data = try Data(contentsOf: url)
        let text = try extractDocxText(from: data)
        let title = url.deletingPathExtension().lastPathComponent
        let page  = ParsedPage(pageNumber: 0, text: text)
        return ParsedDocument(url: url, title: title, pages: [page])
    }
    
    private func extractDocxText(from data: Data) throws -> String {
        // .docx is a ZIP — find the PK header and locate word/document.xml entry
        guard let xml = extractZipEntry(named: "word/document.xml", from: data) else {
            throw DocumentError.parseFailed("word/document.xml not found in .docx")
        }
        return stripXMLTags(from: xml)
    }
    
    /// Minimal ZIP local file entry reader (no external deps).
    /// Uses `withUnsafeBytes` to scan the Data buffer directly — no `[UInt8]` copy.
    private func extractZipEntry(named target: String, from data: Data) -> String? {
        let sig: [UInt8] = [0x50, 0x4B, 0x03, 0x04]   // PK local file header

        return data.withUnsafeBytes { buffer -> String? in
            guard let base = buffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return nil }
            let count = buffer.count
            var i = 0

            while i < count - 30 {
                guard base[i] == sig[0], base[i+1] == sig[1],
                      base[i+2] == sig[2], base[i+3] == sig[3]
                else { i += 1; continue }

                let fnLen  = Int(base[i+26]) | Int(base[i+27]) << 8
                let exLen  = Int(base[i+28]) | Int(base[i+29]) << 8
                let compSz = Int(base[i+18]) | Int(base[i+19]) << 8
                           | Int(base[i+20]) << 16 | Int(base[i+21]) << 24
                let nameStart = i + 30
                let nameEnd   = nameStart + fnLen
                guard nameEnd <= count else { i += 1; continue }

                let nameData = Data(bytes: base + nameStart, count: fnLen)
                guard let name = String(data: nameData, encoding: .utf8) else { i += 1; continue }

                let dataStart = nameEnd + exLen
                let dataEnd   = dataStart + compSz
                guard dataEnd <= count else { i += 1; continue }

                if name == target {
                    // compression method: 0 = stored, 8 = deflate
                    let method = Int(base[i+8]) | Int(base[i+9]) << 8
                    let entryData = Data(bytes: base + dataStart, count: compSz)
                    if method == 0 {
                        return String(data: entryData, encoding: .utf8)
                    } else {
                        // Deflate — use zlib via NSData
                        var header = Data([0x78, 0x9C])
                        header.append(entryData)
                        if let decompressed = try? (header as NSData).decompressed(using: .zlib) {
                            return String(data: decompressed as Data, encoding: .utf8)
                        }
                    }
                }
                i = dataEnd
            }
            return nil
        }
    }
    
    private func stripXMLTags(from xml: String) -> String {
        // Extract text between <w:t> tags, preserving paragraph breaks
        var result = ""
        var scanner = xml.startIndex
        let wt    = "<w:t"
        let wtEnd = "</w:t>"
        let wp    = "</w:p>"
        
        while scanner < xml.endIndex {
            if xml[scanner...].hasPrefix(wp) {
                result += "\n"
                scanner = xml.index(scanner, offsetBy: wp.count, limitedBy: xml.endIndex) ?? xml.endIndex
            } else if xml[scanner...].hasPrefix(wt) {
                // Skip to > then capture until </w:t>
                if let gt  = xml[scanner...].firstIndex(of: ">"),
                   let end = xml[gt...].range(of: wtEnd) {
                    let textStart = xml.index(after: gt)
                    result += xml[textStart..<end.lowerBound]
                    scanner = end.upperBound
                } else { scanner = xml.index(after: scanner) }
            } else {
                scanner = xml.index(after: scanner)
            }
        }
        return result
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }
}

// MARK: - PlainTextDocumentParser

struct PlainTextDocumentParser: DocumentParser {
    private static let extensions = ["txt", "md", "markdown", "rtf"]
    
    func canParse(url: URL) -> Bool {
        Self.extensions.contains(url.pathExtension.lowercased())
    }
    
    func parse(url: URL) async throws -> ParsedDocument {
        let text  = try String(contentsOf: url, encoding: .utf8)
        let title = url.deletingPathExtension().lastPathComponent
        return ParsedDocument(url: url, title: title, pages: [ParsedPage(pageNumber: 0, text: text)])
    }
}

// MARK: - ImageDocumentParser

struct ImageDocumentParser: DocumentParser {
    let llm: AuraLocal
    private static let extensions = ["png", "jpg", "jpeg", "heic", "tiff", "bmp"]
    
    func canParse(url: URL) -> Bool {
        Self.extensions.contains(url.pathExtension.lowercased())
    }
    
    func parse(url: URL) async throws -> ParsedDocument {
        guard let data  = try? Data(contentsOf: url),
              let image = PlatformImage(data: data)
        else { throw DocumentError.parseFailed("Cannot load image at \(url.lastPathComponent)") }
        
        let text = try await llm.analyze(
            "Extract all text from this image verbatim. Output only the extracted text, no commentary.",
            image: image,
            maxTokens: 2048
        )
        let title = url.deletingPathExtension().lastPathComponent
        return ParsedDocument(url: url, title: title, pages: [ParsedPage(pageNumber: 0, text: text)])
    }
}

// MARK: - DocumentError

/// Errors thrown by the document parsing, embedding, and indexing pipeline.
public enum DocumentError: LocalizedError {
    /// The file extension is not supported by any registered parser.
    case unsupportedFormat(String)
    /// The document could not be parsed (corrupted file, missing content, etc.).
    case parseFailed(String)
    /// The embedding provider failed to produce vectors for the given text.
    case embeddingFailed(String)
    /// ``DocumentLibrary/open()`` has not been called yet, or ``DocumentLibrary/configure(embeddingProvider:llm:visionLLM:)`` is missing.
    case libraryNotReady
    
    public var errorDescription: String? {
        switch self {
            case .unsupportedFormat(let ext): return "Unsupported document format: .\(ext)"
            case .parseFailed(let msg):       return "Parse error: \(msg)"
            case .embeddingFailed(let msg):   return "Embedding error: \(msg)"
            case .libraryNotReady:            return "DocumentLibrary is not open. Call open() first."
        }
    }
}
