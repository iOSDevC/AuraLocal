import Foundation

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - AuraLocal

/// On-device LLM/VLM powered by mlx-swift.
///
/// Use the factory methods to get a typed instance:
/// ```swift
/// // Text
/// let llm = try await AuraLocal.text(.qwen3_1_7b)
/// let reply = try await llm.chat("Summarize my expenses")
///
/// // Vision
/// let vlm = try await AuraLocal.vision(.qwen35_0_8b)
/// let desc = try await vlm.analyze("What's in this image?", image: photo)
///
/// // Specialized (OCR / document)
/// let ocr = try await AuraLocal.specialized(.fastVLM_0_5b_fp16)
/// let json = try await ocr.extractDocument(receiptImage)
/// ```
@MainActor
public final class AuraLocal {
    
    // MARK: - Nested types
    
    public enum VisionRunMode: String, CaseIterable, Identifiable {
        case standard = "Standard"
        case stream   = "Stream"
        public var id: String { rawValue }
    }
    
    // MARK: - Properties
    
    let engine: AuraEngine
    public let model: Model
    
    // MARK: - Init

    /// Internal init — creates an instance with backend auto-selected by ``BackendRouter``.
    /// Use ``ModelManager/load(_:onProgress:)`` or factory methods (``text``, ``vision``, ``specialized``) instead.
    init(model: Model, temperature: Float? = nil) {
        self.model = model
        self.engine = AuraEngine(model: model, temperature: temperature)
    }
    
    // MARK: - Factory methods
    
    /// Load a text-generation model.
    public static func text(
        _ model: Model = .qwen3_1_7b,
        onProgress: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> AuraLocal {
        guard case .text = model.purpose else {
            throw AuraError.invalidResponse("Model \(model.displayName) is not a text model.")
        }
        let instance = AuraLocal(model: model)
        try await instance.engine.load(onProgress: onProgress)
        return instance
    }
    
    /// Load a general-purpose vision model.
    public static func vision(
        _ model: Model = .qwen35_0_8b,
        onProgress: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> AuraLocal {
        guard case .vision = model.purpose else {
            throw AuraError.invalidResponse("Model \(model.displayName) is not a vision model.")
        }
        let instance = AuraLocal(model: model)
        try await instance.engine.load(onProgress: onProgress)
        return instance
    }
    
    /// Load an OCR / document-specialized vision model.
    public static func specialized(
        _ model: Model = .fastVLM_0_5b_fp16,
        onProgress: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> AuraLocal {
        guard case .visionSpecialized = model.purpose else {
            throw AuraError.invalidResponse("Model \(model.displayName) is not a specialized model.")
        }
        let instance = AuraLocal(model: model)
        try await instance.engine.load(onProgress: onProgress)
        return instance
    }
    
    // MARK: - Text API
    
    /// Send a message and receive the full response.
    public func chat(
        _ prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 1024
    ) async throws -> String {
        try await engine.generate(
            prompt: prompt,
            systemPrompt: systemPrompt,
            maxTokens: maxTokens,
            onToken: { _ in }
        )
    }
    
    /// Stream tokens one by one.
    public func stream(
        _ prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 1024
    ) -> AsyncThrowingStream<String, Error> {
        let engine = self.engine
        return AsyncThrowingStream { continuation in
            Task { @MainActor in
                do {
                    var lastLength = 0
                    _ = try await engine.generate(
                        prompt: prompt,
                        systemPrompt: systemPrompt,
                        maxTokens: maxTokens
                    ) { @MainActor partial in
                        let newText = String(partial.dropFirst(lastLength))
                        lastLength = partial.count
                        if !newText.isEmpty { continuation.yield(newText) }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    // MARK: - Vision API
    
    /// Ask a question about an image (or text-only if image is nil).
    public func analyze(
        _ prompt: String,
        image: PlatformImage? = nil,
        maxTokens: Int = 800
    ) async throws -> String {
        try await engine.generate(
            prompt: prompt,
            image: image,
            maxTokens: maxTokens,
            onToken: { _ in }
        )
    }
    
    /// Stream tokens for a vision + text query.
    public func streamVision(
        _ prompt: String,
        image: PlatformImage? = nil,
        maxTokens: Int = 800
    ) -> AsyncThrowingStream<String, Error> {
        let engine = self.engine
        let img = image
        return AsyncThrowingStream { continuation in
            Task { @MainActor in
                do {
                    var lastLength = 0
                    _ = try await engine.generate(
                        prompt: prompt,
                        image: img,
                        maxTokens: maxTokens
                    ) { @MainActor partial in
                        let newText = String(partial.dropFirst(lastLength))
                        lastLength = partial.count
                        if !newText.isEmpty { continuation.yield(newText) }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    // MARK: - Specialized API
    
    /// Extract structured data from a document/receipt image using the model's default prompt.
    public func extractDocument(
        _ image: PlatformImage,
        maxTokens: Int? = nil
    ) async throws -> String {
        guard let prompt = model.defaultDocumentPrompt else {
            throw AuraError.invalidResponse("Model \(model.displayName) has no default document prompt.")
        }
        let tokens: Int
        if let maxTokens {
            tokens = maxTokens
        } else if case .visionSpecialized(let docTags) = model.purpose {
            tokens = docTags ? 2048 : 600
        } else {
            tokens = 600
        }
        return try await engine.generate(
            prompt: prompt,
            image: image,
            maxTokens: tokens,
            onToken: { _ in }
        )
    }
    
    // MARK: - Static convenience
    
    /// One-liner text chat (loads model each call — prefer instance for reuse).
    public static func chat(
        _ prompt: String,
        model: Model = .qwen3_1_7b,
        systemPrompt: String? = nil,
        onProgress: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> String {
        let llm = try await AuraLocal.text(model, onProgress: onProgress)
        return try await llm.chat(prompt, systemPrompt: systemPrompt)
    }

    /// One-liner receipt/document extraction.
    public static func extractDocument(
        _ image: PlatformImage,
        model: Model = .fastVLM_0_5b_fp16,
        onProgress: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> String {
        let ocr = try await AuraLocal.specialized(model, onProgress: onProgress)
        return try await ocr.extractDocument(image)
    }
}

// MARK: - DocTags → Markdown

public extension AuraLocal {
    /// Convert Granite Docling's DocTags output to readable Markdown.
    static func parseDocTags(_ docTags: String) -> String {
        var md = docTags
        for tag in ["<doctag>", "</doctag>", "<page>", "</page>", "<body>", "</body>"] {
            md = md.replacingOccurrences(of: tag, with: "")
        }
        for (tag, prefix) in [("section-header-1", "#"), ("section-header-2", "##"), ("section-header-3", "###")] {
            md = md.replacingOccurrences(of: "<\(tag)>(.*?)</\(tag)>", with: "\(prefix) $1", options: .regularExpression)
        }
        md = md.replacingOccurrences(of: #"<paragraph>(.*?)</paragraph>"#,  with: "$1\n",   options: .regularExpression)
        md = md.replacingOccurrences(of: #"<list-item>(.*?)</list-item>"#,   with: "- $1",   options: .regularExpression)
        md = md.replacingOccurrences(of: #"<formula>(.*?)</formula>"#,       with: "`$1`",   options: .regularExpression)
        md = md.replacingOccurrences(of: #"(?s)<table>(.*?)</table>"#,       with: "\n```\n$1\n```\n", options: .regularExpression)
        return md
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
    }
}
