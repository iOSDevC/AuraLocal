import Foundation
import AuraCore

/// A Core ML Stable Diffusion model hosted on Hugging Face, pinned to one revision. Only the files in
/// `components` are downloaded (no `VAEEncoder`, which only image-to-image needs).
public struct CoreMLDiffusionModel: Sendable, Hashable, Identifiable {
    public let id: String
    public let displayName: String
    public let repoID: String
    /// The commit the files are fetched from, so a re-upload upstream cannot change what runs.
    public let revision: String
    /// The folder of compiled `.mlmodelc` resources inside the repo.
    public let folder: String
    /// Paths relative to `folder`; a trailing "/" selects a directory.
    public let components: [String]
    /// UI hint before ``plan()`` reports the exact size.
    public let approximateDownloadBytes: Int64
    /// Output width and height in pixels, fixed by the Unet's input shape.
    public let imageSide: Int
    public let licenseName: String
    public let licenseURL: URL

    init(id: String, displayName: String, repoID: String, revision: String, folder: String, components: [String],
         approximateDownloadBytes: Int64, imageSide: Int, licenseName: String, licenseURL: URL) {
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

    /// Where the resources live once downloaded.
    public var resourcesDirectory: URL {
        ModelDownloads.localFolder(repoID: repoID, folder: folder)
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

    /// Downloads the components, resuming partial files. Sends no Hugging Face token. Cancel the calling task to
    /// stop; partial files are kept.
    public func download(progress: @escaping @Sendable (Int64, Int64) -> Void) async throws {
        _ = try await ModelDownloads.downloadFolder(repoID: repoID, revision: revision, folder: folder,
                                                    include: components, progress: progress)
    }

    /// Removes the downloaded and partial files. Unload any engine using it first.
    public func delete() throws {
        try ModelDownloads.deleteRepo(repoID)
    }
}
