import Foundation
import TonamoxCore

/// One folder of compiled Core ML Stable Diffusion resources in a Hugging Face repo, found from file names alone.
public struct CoreMLDiffusionFolder: Sendable, Hashable {
    /// Repo-relative, without a trailing "/"; `""` is the repo root.
    public let path: String
    /// What a download fetches, relative to ``path``; directory entries end in "/".
    public let components: [String]
    /// Sum of the listed sizes of every component file; `nil` when the listing has no sizes (search results).
    public let downloadBytes: Int64?
    /// The Unet's (or first chunk's) `metadata.json`, repo-relative.
    public let unetMetadataPath: String
    /// The Unet, or both chunks.
    public let unetBytes: Int64?
    public let isChunked: Bool
    public let isRepoRoot: Bool
    /// `SafetyChecker.mlmodelc` ships in the folder; it is never downloaded.
    public let hasSafetyChecker: Bool
    /// Only `TextEncoder2.mlmodelc` (an SDXL-only folder).
    public let hasOnlyTextEncoder2: Bool
}

/// Finds the folders `ml-stable-diffusion` 1.1.1 can load, by file name.
public enum CoreMLDiffusionLayout {
    static let unetDirectory = "Unet.mlmodelc"
    static let firstChunkDirectory = "UnetChunk1.mlmodelc"
    static let secondChunkDirectory = "UnetChunk2.mlmodelc"
    static let textEncoderDirectory = "TextEncoder.mlmodelc"
    static let secondTextEncoderDirectory = "TextEncoder2.mlmodelc"
    static let decoderDirectory = "VAEDecoder.mlmodelc"
    static let safetyCheckerDirectory = "SafetyChecker.mlmodelc"
    static let tokenizerFiles = ["vocab.json", "merges.txt"]
    /// What an ML Program `.mlmodelc` holds at its top; a nested `Unet.mlmodelc/x.mlmodelc/` lacks them.
    static let compiledModelFiles = ["coremldata.bin", "model.mil", "weights/weight.bin"]

    static let genericReason = "No Core ML Stable Diffusion folder"

    /// Every folder that passes the layout rules (L1–L6), sorted by path.
    public static func folders(in files: [RepoFile]) -> [CoreMLDiffusionFolder] {
        let candidates = Set(files.compactMap { candidateFolder(of: $0.path) })
        return candidates.sorted().compactMap { makeFolder(at: $0, files: files) }
    }

    /// Why no folder qualified, most specific first; `nil` when one does.
    public static func unsupportedReason(in files: [RepoFile]) -> String? {
        guard folders(in: files).isEmpty else { return nil }
        let paths = files.map { $0.path.lowercased() }
        if paths.contains(where: { $0.hasSuffix(".zip") }) {
            return "Core ML archives (zip) are not supported yet"
        }
        if paths.contains(where: { $0.contains(".mlpackage/") }) {
            return "Uncompiled Core ML packages are not supported yet"
        }
        if paths.contains(where: { $0.hasPrefix("coreml_model.mlmodelc/") || $0.contains("/coreml_model.mlmodelc/") }) {
            return "Diffusers-style Core ML layout is not supported"
        }
        if paths.contains(where: { $0.contains(".aimodel/") }) {
            return "Core AI models need iOS 27; TonamoxImageGen uses Core ML"
        }
        return genericReason
    }

    // MARK: Rules

    /// The parent of the first `Unet.mlmodelc` or `UnetChunk1.mlmodelc` path segment.
    static func candidateFolder(of path: String) -> String? {
        let segments = path.split(separator: "/", omittingEmptySubsequences: false)
        guard let index = segments.dropLast().firstIndex(where: {
            $0 == unetDirectory || $0 == firstChunkDirectory
        }) else { return nil }
        return segments[..<index].joined(separator: "/")
    }

    static func makeFolder(at path: String, files: [RepoFile]) -> CoreMLDiffusionFolder? {
        let prefix = path.isEmpty ? "" : path + "/"
        let inside = files.filter { $0.path.hasPrefix(prefix) }
            .map { RepoFile(path: String($0.path.dropFirst(prefix.count)), sizeBytes: $0.sizeBytes) }
        let names = Set(inside.map(\.path))
        let hasDirectory: (String) -> Bool = { name in inside.contains { $0.path.hasPrefix(name + "/") } }

        // L1, L2
        guard tokenizerFiles.allSatisfy(names.contains), hasDirectory(decoderDirectory) else { return nil }
        // L3: the pipeline prefers chunks when both are present, so they are what gets downloaded.
        let isChunked = hasDirectory(firstChunkDirectory) && hasDirectory(secondChunkDirectory)
        guard isChunked || hasDirectory(unetDirectory) else { return nil }
        let unet = isChunked ? [firstChunkDirectory, secondChunkDirectory] : [unetDirectory]
        // L4
        let hasTextEncoder = hasDirectory(textEncoderDirectory)
        let hasSecondTextEncoder = hasDirectory(secondTextEncoderDirectory)
        guard hasTextEncoder || hasSecondTextEncoder else { return nil }
        let encoders = (hasTextEncoder ? [textEncoderDirectory] : [])
            + (hasSecondTextEncoder ? [secondTextEncoderDirectory] : [])
        // L5
        let directories = encoders + unet + [decoderDirectory]
        let complete = directories.allSatisfy { directory in
            compiledModelFiles.allSatisfy { names.contains("\(directory)/\($0)") }
        }
        guard complete else { return nil }

        let components = directories.map { $0 + "/" } + tokenizerFiles
        return CoreMLDiffusionFolder(
            path: path,
            components: components,
            downloadBytes: totalBytes(of: components, in: inside),
            unetMetadataPath: prefix + unet[0] + "/metadata.json",
            unetBytes: totalBytes(of: unet.map { $0 + "/" }, in: inside),
            isChunked: isChunked,
            isRepoRoot: path.isEmpty,
            hasSafetyChecker: hasDirectory(safetyCheckerDirectory),
            hasOnlyTextEncoder2: !hasTextEncoder)
    }

    /// `nil` as soon as one selected file has no listed size.
    static func totalBytes(of components: [String], in files: [RepoFile]) -> Int64? {
        let selected = files.filter { file in
            components.contains { $0.hasSuffix("/") ? file.path.hasPrefix($0) : file.path == $0 }
        }
        let sizes = selected.compactMap(\.sizeBytes)
        guard sizes.count == selected.count else { return nil }
        return sizes.reduce(0, +)
    }
}
