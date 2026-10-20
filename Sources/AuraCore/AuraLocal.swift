import Foundation
import LocalLLMClientCore

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
    /// What the backend samples with; part of ``ModelManager``'s cache key.
    let sampling: SamplingSettings

    /// Token counts and timings of the last generation that returned, labelled by
    /// ``GenerationStats/Provenance``. Read it after `chat` returns or a stream finishes; `nil` before the first
    /// generation and after one that threw.
    public var lastGenerationStats: GenerationStats? { engine.lastGenerationStats }

    /// Opt-in ``DeviceConditions`` guard for this instance's generations; `nil` (the default) is off. Instances
    /// from ``ModelManager`` are shared, so this applies to every caller of the cached instance.
    public var deviceGuard: DeviceConditions? {
        get { engine.deviceGuard }
        set { engine.deviceGuard = newValue }
    }

    /// How an instance loaded with tools (a GGUF model or an MLX text model) runs the calls the model makes: at most
    /// ``ToolLoopOptions/maxToolRounds`` rounds per turn (default 1), and tool errors reported to the model. Read when
    /// a turn starts; no effect on an instance loaded without tools.
    public var toolLoop: ToolLoopOptions {
        get { engine.toolLoop }
        set { engine.toolLoop = newValue }
    }

    /// Receives each ``ToolEvent`` on the main actor while a turn runs its tools; read when a turn starts. No effect
    /// on an instance loaded without tools.
    public var onToolEvent: (@MainActor (ToolEvent) -> Void)? {
        get { engine.onToolEvent }
        set { engine.onToolEvent = newValue }
    }

    /// What the tool loop did in the last generation. `nil` without tools, before the first turn, after one that
    /// threw, and after a constrained call.
    public var lastToolTurn: ToolTurnSummary? { engine.lastToolTurn }

    /// Set when the last generation outgrew the MLX KV window, which drops the oldest context without an error.
    /// `nil` when it fit, the model does not run on MLX, or the last generation threw.
    public var lastContextOverflow: ContextWindowOverflow? { engine.lastContextOverflow }

    /// The last `stream(_:in:)` run. History calls wait for it, so a stopped stream stores its reply before the
    /// next prompt.
    var historyInFlight: Task<Void, Never>?

    // MARK: - Init

    /// Internal init — creates an instance with backend auto-selected by ``BackendRouter``. `tools` (when
    /// non-empty) enables function-calling on a GGUF model or an MLX text model; with an MLX vision or document model
    /// the backend's load throws ``ModelConfigurationError/toolsUnsupported(_:)``. `nil` sampling takes the backend
    /// defaults.
    /// Use ``ModelManager/load(_:tools:temperature:topP:onProgress:)`` or the factories (``text``, ``vision``…).
    init(model: Model, sampling: SamplingSettings? = nil, tools: [any LLMTool] = []) {
        let settings = sampling ?? .defaults(for: model)
        self.model = model
        self.sampling = settings
        self.engine = AuraEngine(model: model, sampling: settings, tools: tools)
    }

    /// An instance around a prepared engine (tests, and MLX tool instances sharing loaded weights).
    init(engine: AuraEngine, sampling: SamplingSettings? = nil) {
        self.model = engine.model
        self.sampling = sampling ?? .defaults(for: engine.model)
        self.engine = engine
    }
    
    // MARK: - Factory methods
    
    /// Load a text-generation model. `temperature` and `topP` hold for the instance's lifetime; `nil` keeps the
    /// backend default, and temperature 0 is greedy. Out-of-range values throw ``ModelConfigurationError``.
    public static func text(
        _ model: Model = .qwen3_1_7b,
        temperature: Float? = nil,
        topP: Float? = nil,
        onProgress: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> AuraLocal {
        guard case .text = model.purpose else {
            throw AuraError.invalidResponse("Model \(model.displayName) is not a text model.")
        }
        return try await loaded(model, temperature: temperature, topP: topP, onProgress: onProgress)
    }
    
    /// Load a general-purpose vision model. `temperature` and `topP` behave as in
    /// ``text(_:temperature:topP:onProgress:)``.
    public static func vision(
        _ model: Model = .qwen35_0_8b,
        temperature: Float? = nil,
        topP: Float? = nil,
        onProgress: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> AuraLocal {
        guard case .vision = model.purpose else {
            throw AuraError.invalidResponse("Model \(model.displayName) is not a vision model.")
        }
        return try await loaded(model, temperature: temperature, topP: topP, onProgress: onProgress)
    }
    
    /// Load an OCR / document-specialized vision model. `temperature` and `topP` behave as in
    /// ``text(_:temperature:topP:onProgress:)``.
    public static func specialized(
        _ model: Model = .fastVLM_0_5b_fp16,
        temperature: Float? = nil,
        topP: Float? = nil,
        onProgress: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> AuraLocal {
        guard case .visionSpecialized = model.purpose else {
            throw AuraError.invalidResponse("Model \(model.displayName) is not a specialized model.")
        }
        return try await loaded(model, temperature: temperature, topP: topP, onProgress: onProgress)
    }

    private static func loaded(
        _ model: Model,
        temperature: Float?,
        topP: Float?,
        onProgress: @escaping @MainActor (String) -> Void
    ) async throws -> AuraLocal {
        let sampling = try SamplingSettings.resolve(temperature: temperature, topP: topP, for: model)
        let instance = AuraLocal(model: model, sampling: sampling)
        try await instance.engine.load(onProgress: onProgress)
        return instance
    }
    
    // MARK: - Text API
    
    /// Send a message and receive the full response.
    ///
    /// `constraint` restricts the output to JSON or a GBNF grammar. Only GGUF models on the llama.cpp
    /// backend honour it; any other backend throws ``GenerationConstraintError/unsupportedBackend(_:)``.
    public func chat(
        _ prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 1024,
        constraint: GenerationConstraint? = nil
    ) async throws -> String {
        try await engine.generate(
            prompt: prompt,
            systemPrompt: systemPrompt,
            maxTokens: maxTokens,
            constraint: constraint,
            onToken: { _ in }
        )
    }

    /// Stream tokens one by one. `constraint` behaves as in ``chat(_:systemPrompt:maxTokens:constraint:)``.
    public func stream(
        _ prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 1024,
        constraint: GenerationConstraint? = nil
    ) -> AsyncThrowingStream<String, Error> {
        let engine = self.engine
        return AsyncThrowingStream { continuation in
            let task = Task { @MainActor in
                do {
                    var lastLength = 0
                    _ = try await engine.generate(
                        prompt: prompt,
                        systemPrompt: systemPrompt,
                        maxTokens: maxTokens,
                        constraint: constraint
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
            // Cancelling the consumer (e.g. the user hits Stop or switches providers) must cancel the
            // decode — otherwise the orphaned generation runs to EOS and races the shared llama.cpp
            // context on the next request. `Task.isCancelled` is checked inside the backend loop.
            continuation.onTermination = { _ in task.cancel() }
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
            let task = Task { @MainActor in
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
            continuation.onTermination = { _ in task.cancel() }
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
