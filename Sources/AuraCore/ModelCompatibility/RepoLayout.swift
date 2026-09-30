import Foundation

// MARK: - DetectedFormat

/// What kind of weights a repository ships, from AuraLocal's point of view.
public enum DetectedFormat: String, Sendable, CaseIterable {
    /// MLX safetensors: MLX quantization in `config.json`, or the `mlx` tag / library.
    case mlx
    /// llama.cpp GGUF files.
    case gguf
    /// A diffusion / text-to-image pipeline (AuraImageGen territory, not the LLM runtimes).
    case imageGeneration
    /// Plain transformers safetensors (full precision or a non-MLX quantizer).
    case unconvertedSafetensors
    /// Neither safetensors nor GGUF (e.g. `pytorch_model.bin` only).
    case noWeights
    /// The repository listing could not be read.
    case unknown

    public var label: String {
        switch self {
        case .mlx: "MLX"
        case .gguf: "GGUF"
        case .imageGeneration: "Image generation"
        case .unconvertedSafetensors: "Transformers safetensors"
        case .noWeights: "No MLX/GGUF weights"
        case .unknown: "Unknown"
        }
    }

    static let imageGenerationPipelines: Set<String> = ["text-to-image", "image-to-image"]

    public static func detect(_ snapshot: RepoSnapshot) -> DetectedFormat {
        guard let listing = snapshot.listing else { return .unknown }
        let hasSafetensors = listing.files.contains { $0.isSafetensors }
        let hasGGUF = listing.files.contains { $0.isGGUF && !$0.isProjector }
        let isImagePipeline = listing.pipelineTag.map(imageGenerationPipelines.contains) ?? false
        if isImagePipeline || listing.libraryName == "diffusers" || listing.tags.contains("diffusers") {
            return .imageGeneration
        }
        let mlxMarked = listing.tags.contains("mlx") || listing.libraryName == "mlx"
            || snapshot.configuration?.isMLXQuantized == true
        if hasSafetensors && mlxMarked { return .mlx }
        if hasGGUF { return .gguf }
        if hasSafetensors { return .unconvertedSafetensors }
        return .noWeights
    }
}

// MARK: - GGUFQuantGroup

/// One downloadable quant of a GGUF repo: a single file, or every part of a split (`-0000N-of-0000M`) file.
public struct GGUFQuantGroup: Sendable, Equatable, Identifiable {
    /// Repo-relative paths, first part first.
    public let paths: [String]
    /// Quant token (`Q4_K_M`, `IQ2_XXS`…), when the file name has one.
    public let label: String?
    public let totalBytes: Int64

    public init(paths: [String], label: String?, totalBytes: Int64) {
        self.paths = paths
        self.label = label
        self.totalBytes = totalBytes
    }

    public var id: String { firstPath }
    public var firstPath: String { paths.first ?? "" }
    /// AuraLocal's GGUF loader takes one file; split quants are skipped
    /// (``HuggingFaceRepo/ggufFiles(fromTreeJSON:owner:repo:revision:)``).
    public var isSplit: Bool { paths.count > 1 }

    /// Model quants of a repo (projectors excluded), smallest first.
    static func groups(in files: [RepoFile]) -> [GGUFQuantGroup] {
        let models = files.filter { $0.isGGUF && !$0.isProjector }
        let grouped = Dictionary(grouping: models) { shardBase($0.path) }
        return grouped.map { makeGroup(base: $0.key, parts: $0.value) }
            .sorted { ($0.totalBytes, $0.firstPath) < ($1.totalBytes, $1.firstPath) }
    }

    private static func makeGroup(base: String, parts: [RepoFile]) -> GGUFQuantGroup {
        let ordered = parts.sorted { $0.path < $1.path }
        let name = (base as NSString).lastPathComponent
        let label = HuggingFaceRepo.quantLabel(for: name + ".gguf")
            ?? (name.lowercased().contains("fp16") ? "F16" : nil)
        return GGUFQuantGroup(paths: ordered.map(\.path), label: label,
                              totalBytes: ordered.compactMap(\.sizeBytes).reduce(0, +))
    }

    /// `dir/model-Q4_K_M-00001-of-00002.gguf` → `dir/model-Q4_K_M`; a single file keeps its path.
    static func shardBase(_ path: String) -> String {
        let filename = (path as NSString).lastPathComponent
        guard HuggingFaceRepo.isShard(filename) else { return path }
        let base = (path as NSString).deletingPathExtension
        return base.split(separator: "-").dropLast(3).joined(separator: "-")
    }
}
