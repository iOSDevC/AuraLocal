import Foundation
import MLX
import MLXLLM
import MLXVLM
import MLXLMCommon
import Tokenizers

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - MLXBackend

/// Inference backend powered by Apple's MLX framework.
///
/// Loads the full model into unified memory via `LLMModelFactory` or
/// `VLMModelFactory` and delegates generation to `MLXLMCommon.generate()`.
/// Best for models that fit entirely in device RAM (typically up to 4B on iOS).
@MainActor
final class MLXBackend: InferenceBackend {

    // MARK: - State

    private var modelContainer: ModelContainer?
    private let model: Model
    private let generateParameters: GenerateParameters

    var isLoaded: Bool { modelContainer != nil }

    // MARK: - Init

    init(model: Model, temperature: Float? = nil) {
        self.model = model
        let defaultTemp: Float
        switch model.purpose {
        case .text:              defaultTemp = 0.7
        case .vision:            defaultTemp = 0.1
        case .visionSpecialized: defaultTemp = 0.0
        }
        self.generateParameters = GenerateParameters(temperature: temperature ?? defaultTemp)
    }

    // MARK: - InferenceBackend

    func load(onProgress: @escaping @MainActor (String) -> Void) async throws {
        guard modelContainer == nil else { return }

        // Scale GPU cache limit proportionally to model size.
        let modelWeightsBytes = model.approximateSizeMB * 1024 * 1024
        let proportionalCache = modelWeightsBytes / 12
        let cacheLimitBytes   = max(128 * 1024 * 1024, min(proportionalCache, 512 * 1024 * 1024))
        MLX.GPU.set(cacheLimit: cacheLimitBytes)

        let config = ModelConfiguration(id: model.rawValue)
        let downloader = AuraHFDownloader()
        let tokenizer = AuraTokenizerLoader()

        switch model.purpose {
        case .text:
            modelContainer = try await LLMModelFactory.shared.loadContainer(
                from: downloader,
                using: tokenizer,
                configuration: config
            ) { [model] progress in
                let pct = Int(progress.fractionCompleted * 100)
                Task { @MainActor in
                    onProgress("Downloading \(model.displayName): \(pct)%")
                }
            }

        case .vision, .visionSpecialized:
            modelContainer = try await VLMModelFactory.shared.loadContainer(
                from: downloader,
                using: tokenizer,
                configuration: config
            ) { [model] progress in
                let pct = Int(progress.fractionCompleted * 100)
                Task { @MainActor in
                    onProgress("Downloading \(model.displayName): \(pct)%")
                }
            }
        }

        onProgress("\(model.displayName) ready")
    }

    func generate(
        prompt: String,
        systemPrompt: String?,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        guard let container = modelContainer else {
            throw AuraError.modelNotLoaded
        }

        var msgs: [[String: String]] = []
        if let sys = systemPrompt {
            msgs.append(["role": "system", "content": sys])
        }
        msgs.append(["role": "user", "content": prompt])
        let capturedMessages = msgs

        return try await performGeneration(
            container: container,
            prepareInput: { context in
                try await context.processor.prepare(input: .init(messages: capturedMessages))
            },
            maxTokens: maxTokens,
            onToken: onToken
        )
    }

    func generate(
        prompt: String,
        image: PlatformImage?,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        guard let container = modelContainer else {
            throw AuraError.modelNotLoaded
        }

        var tempURL: URL?
        let capturedInput: UserInput
        if let img = image, let url = saveImageToTemp(img) {
            tempURL = url
            capturedInput = UserInput(prompt: prompt, images: [.url(url)])
        } else {
            capturedInput = UserInput(prompt: prompt)
        }

        defer {
            if let url = tempURL {
                try? FileManager.default.removeItem(at: url)
            }
        }

        nonisolated(unsafe) let unsafeInput = capturedInput
        return try await performGeneration(
            container: container,
            prepareInput: { context in
                try await context.processor.prepare(input: unsafeInput)
            },
            maxTokens: maxTokens,
            onToken: onToken
        )
    }

    func generate(
        messages: [[String: String]],
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        guard let container = modelContainer else {
            throw AuraError.modelNotLoaded
        }

        return try await performGeneration(
            container: container,
            prepareInput: { context in
                try await context.processor.prepare(input: .init(messages: messages))
            },
            maxTokens: maxTokens,
            onToken: onToken
        )
    }

    func unload() {
        modelContainer = nil
    }

    // MARK: - Core generation

    private func performGeneration(
        container: ModelContainer,
        prepareInput: @escaping @Sendable (ModelContext) async throws -> LMInput,
        maxTokens: Int,
        onToken: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        let params = self.generateParameters
        return try await container.perform { (context: ModelContext) async throws -> sending String in
            let input = try await prepareInput(context)
            let result = try MLXLMCommon.generate(
                input: input,
                parameters: params,
                context: context
            ) { tokens in
                guard !Task.isCancelled else { return .stop }

                let partial = context.tokenizer.decode(tokenIds: tokens)
                Task { @MainActor in
                    guard !Task.isCancelled else { return }
                    onToken(partial)
                }
                return tokens.count >= maxTokens ? .stop : .more
            }
            return context.tokenizer.decode(tokenIds: result.tokens)
        }
    }

    // MARK: - Helpers

    private func saveImageToTemp(_ image: PlatformImage) -> URL? {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "mlxedge_\(UUID().uuidString).jpg")
#if canImport(UIKit)
        guard let data = image.jpegData(compressionQuality: 0.9) else { return nil }
#else
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil),
              let data = NSBitmapImageRep(cgImage: cgImage)
            .representation(using: .jpeg, properties: [:]) else { return nil }
#endif
        try? data.write(to: url)
        return url
    }
}

// MARK: - HuggingFace Downloader (MLXLMCommon.Downloader)

/// Downloads model snapshots from HuggingFace Hub.
struct AuraHFDownloader: Downloader {
    private static let base = "https://huggingface.co"

    func download(
        id: String, revision: String?, matching patterns: [String],
        useLatest: Bool, progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        let sanitized = id.replacingOccurrences(of: "/", with: "--")
        let dir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appending(path: "huggingface/hub/models--\(sanitized)/snapshots/main")

        if !useLatest, FileManager.default.fileExists(atPath: dir.path()) { return dir }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let rev = revision ?? "main"
        let (data, _) = try await URLSession.shared.data(
            from: URL(string: "\(Self.base)/api/models/\(id)/tree/\(rev)")!)

        struct F: Codable { let path: String?; let type: String? }
        let files = ((try? JSONDecoder().decode([F].self, from: data)) ?? [])
            .filter { $0.type == "file" }
            .compactMap(\.path)

        let progress = Progress(totalUnitCount: Int64(files.count))
        for name in files {
            let dest = dir.appending(path: name)
            guard !FileManager.default.fileExists(atPath: dest.path()) else {
                progress.completedUnitCount += 1; progressHandler(progress); continue
            }
            try FileManager.default.createDirectory(
                at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            let (tmp, _) = try await URLSession.shared.download(
                from: URL(string: "\(Self.base)/\(id)/resolve/\(rev)/\(name)")!)
            try FileManager.default.moveItem(at: tmp, to: dest)
            progress.completedUnitCount += 1; progressHandler(progress)
        }
        return dir
    }
}

// MARK: - Tokenizer Loader (MLXLMCommon.TokenizerLoader)

/// Loads tokenizer via swift-transformers and bridges to MLXLMCommon.Tokenizer.
struct AuraTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let t = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
        return Bridge(t)
    }

    private struct Bridge: MLXLMCommon.Tokenizer, @unchecked Sendable {
        let t: any Tokenizers.Tokenizer
        init(_ t: any Tokenizers.Tokenizer) { self.t = t }
        var bosToken: String? { t.bosToken }
        var eosToken: String? { t.eosToken }
        var unknownToken: String? { t.unknownToken }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            t.encode(text: text, addSpecialTokens: addSpecialTokens)
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            t.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
        }
        func convertTokenToId(_ token: String) -> Int? { t.convertTokenToId(token) }
        func convertIdToToken(_ id: Int) -> String? { t.convertIdToToken(id) }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            try t.applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext)
        }
    }
}
