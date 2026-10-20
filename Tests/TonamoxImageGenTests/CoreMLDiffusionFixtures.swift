import Foundation
import Synchronization
import XCTest
import TonamoxCore
@testable import TonamoxImageGen

/// Recorded Hugging Face data (`Fixtures/CoreMLDiffusion`, copied verbatim from the 2026-10-15 survey) and
/// builders for folders and listings.
enum CoreMLDiffusionFixtures {
    static let sd21Split = "apple_coreml-stable-diffusion-2-1-base-palettized__split_einsum_v2~compiled__Unet"
    static let sd21Original = "apple_coreml-stable-diffusion-2-1-base-palettized__original~compiled__Unet"
    static let sd15Split = "apple_coreml-stable-diffusion-v1-5-palettized__split_einsum_v2~compiled__Unet"
    static let sd15Original = "apple_coreml-stable-diffusion-v1-5-palettized__original~compiled__Unet"
    static let sd21Fp16Chunk = "apple_coreml-stable-diffusion-2-1-base__split_einsum~compiled__UnetChunk1"
    static let turboBatchOne = "jivenchy_sd-turbo-coreml-384-6bit____Unet"
    static let inpainting = "jc-builds_sd-v1-5-inpainting-coreml____Unet"
    static let stateful = "neonforestmist_Clover-Image-Tiny-CoreML__stateful-v3__Unet"
    static let sdxlIOS = "apple_coreml-stable-diffusion-xl-base-ios__"
        + "coreml-stable-diffusion-xl-base-ios_split_einsum_compiled__UnetChunk1"
    static let sdxlWithoutTimeIDs = "Gatchamn_coreml-dreamshaper-xl-lightning-8bit____Unet"
    /// `?blobs=true` of `apple/coreml-stable-diffusion-v1-5-palettized`, trimmed to both compiled folders.
    static let sd15Listing = "apple_coreml-stable-diffusion-v1-5-palettized__listing"
    static let sd15Revision = "04a6a0bdd66fb8da470c14e56d762343ef579d88"

    static func data(_ name: String) throws -> Data {
        let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "json",
                                                  subdirectory: "Fixtures/CoreMLDiffusion"), name)
        return try Data(contentsOf: url)
    }

    static func unet(_ name: String) throws -> CoreMLUnetFacts {
        try XCTUnwrap(CoreMLUnetFacts.parse(try data(name)), name)
    }

    static func folder(_ path: String = "split_einsum_v2/compiled", bytes: Int64? = 1_073_000_000,
                       unetBytes: Int64? = 865_000_000, chunked: Bool = false,
                       safetyChecker: Bool = false, onlyTextEncoder2: Bool = false) -> CoreMLDiffusionFolder {
        let prefix = path.isEmpty ? "" : path + "/"
        let unet = chunked ? ["UnetChunk1.mlmodelc/", "UnetChunk2.mlmodelc/"] : ["Unet.mlmodelc/"]
        return CoreMLDiffusionFolder(
            path: path,
            components: ["TextEncoder.mlmodelc/"] + unet + ["VAEDecoder.mlmodelc/", "vocab.json", "merges.txt"],
            downloadBytes: bytes, unetMetadataPath: prefix + unet[0] + "metadata.json", unetBytes: unetBytes,
            isChunked: chunked, isRepoRoot: path.isEmpty, hasSafetyChecker: safetyChecker,
            hasOnlyTextEncoder2: onlyTextEncoder2)
    }

    static func listing(_ repoID: String = "apple/coreml-stable-diffusion-2-1-base-palettized",
                        tags: [String] = ["coreml"], gated: String? = nil, license: String? = "openrail++",
                        revision: String? = "dca5f2fda9c5143c71aded72bbb4ce39f3c2cc9c") -> HFRepoInfo {
        HFRepoInfo(repoID: repoID, files: [], tags: tags, pipelineTag: "text-to-image", gatedMode: gated,
                   licenseID: license, revision: revision)
    }

    /// The files of one compiled folder as Apple's converter writes them.
    static func compiledFiles(in folder: String, unet: [String] = ["Unet.mlmodelc"],
                              encoders: [String] = ["TextEncoder.mlmodelc"], extras: [String] = [],
                              size: Int64? = 10) -> [RepoFile] {
        let prefix = folder.isEmpty ? "" : folder + "/"
        let directories = encoders + unet + ["VAEDecoder.mlmodelc"] + extras
        let modelFiles = directories.flatMap { directory in
            ["coremldata.bin", "model.mil", "weights/weight.bin", "metadata.json"].map { "\(directory)/\($0)" }
        }
        return (modelFiles + ["vocab.json", "merges.txt"]).map { RepoFile(path: prefix + $0, sizeBytes: size) }
    }
}

// MARK: - HTTP stub

/// Serves canned replies by absolute URL and records every request. Unknown URLs answer 404.
final class InspectorStubProtocol: URLProtocol, @unchecked Sendable {
    struct Reply: Sendable {
        let status: Int
        let body: Data
    }

    private struct Ledger {
        var routes: [String: Reply] = [:]
        var requests: [URLRequest] = []
    }

    private static let table = Mutex(Ledger())

    static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [InspectorStubProtocol.self]
        return URLSession(configuration: configuration)
    }()

    static func reset() {
        table.withLock { $0 = Ledger() }
    }

    static func serve(_ url: URL, status: Int = 200, body: Data) {
        table.withLock { $0.routes[url.absoluteString] = Reply(status: status, body: body) }
    }

    static var requests: [URLRequest] {
        table.withLock { $0.requests }
    }

    static func requests(containing fragment: String) -> [URLRequest] {
        requests.filter { $0.url?.absoluteString.contains(fragment) ?? false }
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        guard let url = request.url else { return }
        let reply = Self.table.withLock { ledger -> Reply in
            ledger.requests.append(request)
            return ledger.routes[url.absoluteString] ?? Reply(status: 404, body: Data())
        }
        if let response = HTTPURLResponse(url: url, statusCode: reply.status, httpVersion: "HTTP/1.1",
                                          headerFields: ["Content-Type": "application/json"]) {
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        }
        client?.urlProtocol(self, didLoad: reply.body)
        client?.urlProtocolDidFinishLoading(self)
    }
}
