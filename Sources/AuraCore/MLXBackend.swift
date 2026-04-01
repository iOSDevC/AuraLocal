import Foundation
import MLX
import MLXLLM
import MLXVLM
import MLXLMCommon

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

        switch model.purpose {
        case .text:
            modelContainer = try await LLMModelFactory.shared.loadContainer(
                configuration: config
            ) { [model] progress in
                let pct = Int(progress.fractionCompleted * 100)
                Task { @MainActor in
                    onProgress("Downloading \(model.displayName): \(pct)%")
                }
            }

        case .vision, .visionSpecialized:
            modelContainer = try await VLMModelFactory.shared.loadContainer(
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
        return try await container.perform { context in
            let input = try await prepareInput(context)
            let result = try MLXLMCommon.generate(
                input: input,
                parameters: self.generateParameters,
                context: context
            ) { tokens in
                guard !Task.isCancelled else { return .stop }

                let partial = context.tokenizer.decode(tokens: tokens)
                Task { @MainActor in
                    guard !Task.isCancelled else { return }
                    onToken(partial)
                }
                return tokens.count >= maxTokens ? .stop : .more
            }
            return context.tokenizer.decode(tokens: result.tokens)
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
