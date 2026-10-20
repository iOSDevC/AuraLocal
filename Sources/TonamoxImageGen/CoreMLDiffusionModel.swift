import Foundation
import TonamoxCore

/// A Core ML Stable Diffusion model hosted on Hugging Face, pinned to one revision. Only the files in
/// `components` are downloaded (no `VAEEncoder`, which only image-to-image needs). ``stableDiffusion21Base`` is
/// built in; ``init(variant:)`` adopts another folder that ``CoreMLDiffusionInspector`` judged runnable.
public struct CoreMLDiffusionModel: Sendable, Hashable, Identifiable, Codable {
    public let id: String
    public let displayName: String
    public let repoID: String
    /// The commit the files are fetched from, so a re-upload upstream cannot change what runs.
    public let revision: String
    /// The folder of compiled `.mlmodelc` resources inside the repo.
    public let folder: String
    /// Paths relative to `folder`; a trailing "/" selects a directory.
    public let components: [String]
    /// UI hint before ``plan()`` reports the exact size; exact for a model made by ``init(variant:)``.
    public let approximateDownloadBytes: Int64
    /// Output width and height in pixels, fixed by the Unet's input shape.
    public let imageSide: Int
    public let licenseName: String
    public let licenseURL: URL
    /// From the Unet's attention: split-einsum runs on the Neural Engine, ORIGINAL on the GPU.
    public let preferredComputeUnits: DiffusionComputeUnits
    /// Defaults for ``DiffusionRequest``; few-step distilled models use 1–8.
    public let recommendedSteps: Int
    public let recommendedGuidance: Float

    init(id: String, displayName: String, repoID: String, revision: String, folder: String, components: [String],
         approximateDownloadBytes: Int64, imageSide: Int, licenseName: String, licenseURL: URL,
         preferredComputeUnits: DiffusionComputeUnits = .neuralEngine, recommendedSteps: Int = 20,
         recommendedGuidance: Float = 7.5) {
        self.id = id
        self.displayName = displayName
        self.repoID = repoID
        self.revision = revision
        self.folder = folder
        self.components = components
        self.approximateDownloadBytes = approximateDownloadBytes
        self.imageSide = imageSide
        self.licenseName = licenseName
        self.licenseURL = licenseURL
        self.preferredComputeUnits = preferredComputeUnits
        self.recommendedSteps = recommendedSteps
        self.recommendedGuidance = recommendedGuidance
    }

    /// A model for a variant whose verdict is runs (with or without caveats), with its exact size and image side.
    /// The id leaves the revision out, so adopting a newer commit of the same folder replaces the record. The
    /// built-in model's own folder gives ``stableDiffusion21Base`` (at its pinned revision), since both would share
    /// one folder on disk. Throws ``CoreMLDiffusionModelError/notRunnable(repoID:folder:reasons:)`` otherwise.
    public init(variant: CoreMLDiffusionVariant) throws {
        let source = variant.folder
        guard variant.verdict.isRunnable, let bytes = source.downloadBytes, let side = variant.unet?.imageSide,
              !variant.revision.isEmpty else {
            throw CoreMLDiffusionModelError.notRunnable(repoID: variant.repoID, folder: source.path,
                                                        reasons: Self.reasons(variant))
        }
        let builtIn = Self.stableDiffusion21Base
        if variant.repoID.lowercased() == builtIn.repoID.lowercased(), source.path == builtIn.folder {
            self = builtIn
            return
        }
        let repoName = variant.repoID.split(separator: "/").last.map(String.init) ?? variant.repoID
        self.init(id: "hf:\(variant.repoID)/\(source.path)", displayName: repoName, repoID: variant.repoID,
                  revision: variant.revision, folder: source.path, components: source.components,
                  approximateDownloadBytes: bytes, imageSide: side, licenseName: variant.licenseName,
                  licenseURL: variant.licenseURL, preferredComputeUnits: variant.preferredComputeUnits,
                  recommendedSteps: variant.recommendedSteps, recommendedGuidance: variant.recommendedGuidance)
    }

    private static func reasons(_ variant: CoreMLDiffusionVariant) -> [String] {
        let blockers = variant.findings.filter { $0.level == .blocker }.map(\.title)
        if !blockers.isEmpty { return blockers }
        let unverified = variant.findings.filter { $0.rule == "R1" || ($0.rule == "FIT" && variant.fit == nil) }
        if !unverified.isEmpty { return unverified.map(\.title) }
        return ["\(variant.verdict.label): size or image side unknown"]
    }

    /// Apple's 6-bit palettized conversion of Stable Diffusion 2.1 base (`SPLIT_EINSUM_V2`): 512×512, ungated,
    /// no safety checker in the repo.
    public static let stableDiffusion21Base = CoreMLDiffusionModel(
        id: "sd-2.1-base-palettized",
        displayName: "Stable Diffusion 2.1 base (Core ML, 6-bit)",
        repoID: "apple/coreml-stable-diffusion-2-1-base-palettized",
        revision: "dca5f2fda9c5143c71aded72bbb4ce39f3c2cc9c",
        folder: "split_einsum_v2/compiled",
        components: ["TextEncoder.mlmodelc/", "Unet.mlmodelc/", "VAEDecoder.mlmodelc/", "vocab.json", "merges.txt"],
        approximateDownloadBytes: 1_073_000_000,
        imageSide: 512,
        licenseName: "CreativeML Open RAIL++-M",
        // The model card links the license text on stabilityai's repo, which now answers 401 anonymously.
        licenseURL: URL(string: "https://huggingface.co/apple/coreml-stable-diffusion-2-1-base-palettized")
            ?? URL(fileURLWithPath: "/"))

    /// The built-in ``stableDiffusion21Base``, which a model store never records.
    public var isBuiltIn: Bool {
        id == Self.stableDiffusion21Base.id
    }

    /// Where the resources live once downloaded: under ``ModelStorage/rootDirectory``, or the finished copy an
    /// earlier version left in Caches until it is migrated.
    public var resourcesDirectory: URL {
        ModelDownloads.completedFolder(repoID: repoID, folder: folder, revision: revision)
            ?? ModelDownloads.localFolder(repoID: repoID, folder: folder)
    }

    /// The download of `revision` finished and every component is still on disk.
    public var isDownloaded: Bool {
        isDownloaded(at: resourcesDirectory)
    }

    func isDownloaded(at directory: URL) -> Bool {
        ModelDownloads.isFolderComplete(at: directory, revision: revision) && components.allSatisfy {
            FileManager.default.fileExists(atPath: directory.appending(path: $0).path)
        }
    }

    /// Files and sizes to download, from the Hugging Face listing at `revision`.
    public func plan() async throws -> DownloadPlan {
        try await ModelDownloads.planFolder(repoID: repoID, revision: revision, folder: folder, include: components)
    }

    /// Downloads the components, resuming partial files and verifying each LFS file's SHA-256. Sends no Hugging Face
    /// token. Cancel the calling task to stop; partial files are kept.
    public func download(progress: @escaping @Sendable (Int64, Int64) -> Void) async throws {
        _ = try await ModelDownloads.downloadFolder(repoID: repoID, revision: revision, folder: folder,
                                                    include: components, progress: progress)
    }

    /// Removes this folder's downloaded and partial files, in the storage root and in Caches; other folders of the
    /// same repo stay. Unload any engine using it first. Core ML's compiled copy of the model lives in a
    /// system-managed cache that this leaves alone.
    public func delete() throws {
        try ModelDownloads.deleteFolder(repoID: repoID, folder: folder)
    }
}
