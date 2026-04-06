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

        let downloader = HFSnapshotDownloader()
        let tokenizerLoader = JSONTokenizerLoader()

        switch model.purpose {
        case .text:
            modelContainer = try await LLMModelFactory.shared.loadContainer(
                from: downloader,
                using: tokenizerLoader,
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
                using: tokenizerLoader,
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
        let stream: AsyncStream<Generation> = try await container.perform { (context: ModelContext) in
            let input = try await prepareInput(context)
            return try MLXLMCommon.generate(
                input: input,
                parameters: self.generateParameters,
                context: context
            )
        }
        var fullText = ""
        var tokenCount = 0
        for await generation in stream {
            guard !Task.isCancelled else { break }
            switch generation {
            case .chunk(let text):
                fullText += text
                tokenCount += 1
                Task { @MainActor in
                    guard !Task.isCancelled else { return }
                    onToken(text)
                }
                if tokenCount >= maxTokens { break }
            case .info:
                break
            default:
                break
            }
        }
        return fullText
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

// MARK: - HuggingFace Snapshot Downloader

/// Downloads model snapshots from HuggingFace Hub using URLSession.
/// Caches downloaded files in `~/Library/Caches/huggingface/hub/`.
struct HFSnapshotDownloader: Downloader {

    private static let baseURL = "https://huggingface.co"
    private static let cacheRoot: URL = {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appending(path: "huggingface/hub", directoryHint: .isDirectory)
    }()

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        let sanitizedID = id.replacingOccurrences(of: "/", with: "--")
        let modelDir = Self.cacheRoot
            .appending(path: "models--\(sanitizedID)/snapshots/main", directoryHint: .isDirectory)

        // Return cached directory if it exists and we're not forcing latest
        if !useLatest, FileManager.default.fileExists(atPath: modelDir.path()) {
            return modelDir
        }

        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        // Determine which files to download from the model's file list
        let rev = revision ?? "main"
        let apiURL = URL(string: "\(Self.baseURL)/api/models/\(id)/tree/\(rev)")!
        let (apiData, _) = try await URLSession.shared.data(from: apiURL)

        struct HFFile: Codable { let rfilename: String }
        let files = (try? JSONDecoder().decode([HFFile].self, from: apiData)) ?? []

        let matchedFiles = files.map(\.rfilename).filter { filename in
            patterns.isEmpty || patterns.contains { pattern in
                matchGlob(pattern: pattern, string: filename)
            }
        }

        let progress = Progress(totalUnitCount: Int64(matchedFiles.count))

        for filename in matchedFiles {
            let fileURL = modelDir.appending(path: filename)
            guard !FileManager.default.fileExists(atPath: fileURL.path()) else {
                progress.completedUnitCount += 1
                progressHandler(progress)
                continue
            }

            // Create parent directory if needed
            let parent = fileURL.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)

            let downloadURL = URL(string: "\(Self.baseURL)/\(id)/resolve/\(rev)/\(filename)")!
            let (tempURL, _) = try await URLSession.shared.download(from: downloadURL)
            try FileManager.default.moveItem(at: tempURL, to: fileURL)

            progress.completedUnitCount += 1
            progressHandler(progress)
        }

        return modelDir
    }

    private func matchGlob(pattern: String, string: String) -> Bool {
        let regexPattern = "^" + NSRegularExpression.escapedPattern(for: pattern)
            .replacingOccurrences(of: "\\*", with: ".*")
            .replacingOccurrences(of: "\\?", with: ".") + "$"
        return string.range(of: regexPattern, options: .regularExpression) != nil
    }
}

// MARK: - JSON Tokenizer Loader

struct JSONTokenizerLoader: TokenizerLoader {

    func load(from directory: URL) async throws -> any Tokenizer {
        let configURL = directory.appending(path: "tokenizer.json")
        guard FileManager.default.fileExists(atPath: configURL.path()) else {
            throw AuraError.modelNotLoaded
        }
        return try SimpleTokenizer(directory: directory)
    }
}

private final class SimpleTokenizer: Tokenizer, @unchecked Sendable {
    private let vocab: [String: Int]
    private let reverseVocab: [Int: String]
    private let _bosToken: String?
    private let _eosToken: String?

    var bosToken: String? { _bosToken }
    var eosToken: String? { _eosToken }
    var unknownToken: String? { nil }

    init(directory: URL) throws {
        let configURL = directory.appending(path: "tokenizer_config.json")
        let vocabURL = directory.appending(path: "tokenizer.json")

        var bos: String?
        var eos: String?
        if let configData = try? Data(contentsOf: configURL),
           let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] {
            bos = config["bos_token"] as? String
            eos = config["eos_token"] as? String
        }
        _bosToken = bos
        _eosToken = eos

        var tempVocab: [String: Int] = [:]
        if let vocabData = try? Data(contentsOf: vocabURL),
           let json = try? JSONSerialization.jsonObject(with: vocabData) as? [String: Any],
           let model = json["model"] as? [String: Any],
           let v = model["vocab"] as? [String: Int] {
            tempVocab = v
        }
        vocab = tempVocab
        reverseVocab = Dictionary(uniqueKeysWithValues: tempVocab.map { ($1, $0) })
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.split(separator: " ").compactMap { vocab[String($0)] }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { reverseVocab[$0] }.joined(separator: "")
    }

    func convertTokenToId(_ token: String) -> Int? { vocab[token] }
    func convertIdToToken(_ id: Int) -> String? { reverseVocab[id] }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let text = messages.compactMap { $0["content"] as? String }.joined(separator: "\n")
        return encode(text: text)
    }
}
