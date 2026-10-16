import XCTest
import AuraCore
@testable import AuraImageGen

/// The inspector against a stubbed Hugging Face: no network.
final class CoreMLDiffusionInspectorTests: XCTestCase {
    private typealias Fixtures = CoreMLDiffusionFixtures
    private typealias Stub = InspectorStubProtocol

    private let client = CoreMLDiffusionInspector(session: InspectorStubProtocol.session, requestTimeout: 5)

    override func setUp() {
        super.setUp()
        Stub.reset()
    }

    private static func listingURL(_ repoID: String) throws -> URL {
        try XCTUnwrap(URL(string: "https://huggingface.co/api/models/\(repoID)?blobs=true"))
    }

    private static func probeURL(_ repoID: String, revision: String, folder: String) throws -> URL {
        try XCTUnwrap(URL(string: "https://huggingface.co/\(repoID)/resolve/\(revision)/\(folder)"
                          + "/Unet.mlmodelc/metadata.json"))
    }

    private static func json(_ object: Any) throws -> Data {
        try JSONSerialization.data(withJSONObject: object)
    }

    private static func searchEntry(_ id: String, files: [String], downloads: Int = 1) -> [String: Any] {
        ["id": id, "sha": String(repeating: "a", count: 40), "downloads": downloads, "likes": 0, "gated": false,
         "siblings": files.map { ["rfilename": $0] }]
    }

    /// A listing with `count` compiled folders named `f0`…, each with the given size per file.
    private static func listing(_ repoID: String, folders count: Int, gated: Any = false,
                                sha: String? = Fixtures.sd15Revision) throws -> Data {
        let files = (0..<count).flatMap { Fixtures.compiledFiles(in: "f\($0)") }
        var object: [String: Any] = ["id": repoID, "gated": gated, "cardData": ["license": "creativeml-openrail-m"],
                                     "siblings": files.map { ["rfilename": $0.path, "size": 10] }]
        object["sha"] = sha
        return try json(object)
    }

    // MARK: Search

    func testSearchURLsQueryByBothTagsAndByNameWithCoreML() throws {
        let urls = CoreMLDiffusionInspector.searchURLs(query: " realistic vision ", limit: 50)
        XCTAssertEqual(urls.map(\.absoluteString), [
            "https://huggingface.co/api/models?filter=coreml&pipeline_tag=text-to-image&full=true&sort=downloads"
                + "&limit=50&search=realistic%20vision",
            "https://huggingface.co/api/models?filter=core-ml&pipeline_tag=text-to-image&full=true&sort=downloads"
                + "&limit=50&search=realistic%20vision",
            "https://huggingface.co/api/models?pipeline_tag=text-to-image&full=true&sort=downloads&limit=50"
                + "&search=realistic%20vision%20coreml",
        ])
    }

    func testAnEmptyQueryIsAllowed() {
        let urls = CoreMLDiffusionInspector.searchURLs(query: "", limit: 10)
        XCTAssertEqual(urls.count, 3)
        XCTAssertFalse(urls[0].absoluteString.contains("search="))
        XCTAssertFalse(urls[1].absoluteString.contains("search="))
        XCTAssertTrue(urls[2].absoluteString.hasSuffix("&search=coreml"))
    }

    func testAPastedRepoURLSearchesItsIDAndCoreMLIsNotAddedTwice() {
        let urls = CoreMLDiffusionInspector.searchURLs(query: "https://huggingface.co/CommuniDev/sd-turbo-coreml",
                                                       limit: 50)
        XCTAssertTrue(urls.allSatisfy { $0.absoluteString.hasSuffix("&search=CommuniDev/sd-turbo-coreml") },
                      "\(urls)")
    }

    func testSearchMergesDeduplicatesAndDropsNonCoreMLHits() async throws {
        let urls = CoreMLDiffusionInspector.searchURLs(query: "dreamshaper", limit: 50)
        let folder = Fixtures.compiledFiles(in: "Resources").map(\.path)
        Stub.serve(urls[0], body: try Self.json([
            Self.searchEntry("a/tagged", files: folder, downloads: 9),
            Self.searchEntry("b/zipped", files: ["dreamshaper_split-einsum.zip"]),
        ]))
        Stub.serve(urls[1], body: try Self.json([
            Self.searchEntry("a/tagged", files: folder),
            Self.searchEntry("e/other-tag", files: folder),
        ]))
        Stub.serve(urls[2], body: try Self.json([
            Self.searchEntry("a/tagged", files: folder),
            Self.searchEntry("c/diffusers", files: ["unet/diffusion_pytorch_model.safetensors"]),
            Self.searchEntry("d/untagged", files: folder),
        ]))

        let hits = try await client.search("dreamshaper")

        XCTAssertEqual(hits.map(\.id), ["a/tagged", "b/zipped", "e/other-tag", "d/untagged"])
        XCTAssertEqual(hits[0].candidateFolders, ["Resources"])
        XCTAssertEqual(hits[0].downloads, 9)
        XCTAssertEqual(hits[1].unsupportedReason, "Core ML archives (zip) are not supported yet")
        XCTAssertTrue(hits[1].candidateFolders.isEmpty)
        XCTAssertEqual(Stub.requests.count, 3)
    }

    func testSearchThrowsOnlyWhenEveryQueryFails() async throws {
        let urls = CoreMLDiffusionInspector.searchURLs(query: "x", limit: 50)
        Stub.serve(urls[0], status: 500, body: Data())
        Stub.serve(urls[1], status: 502, body: Data())
        Stub.serve(urls[2], body: try Self.json([Self.searchEntry("d/untagged",
                                                                  files: Fixtures.compiledFiles(in: "R").map(\.path))]))
        let partial = try await client.search("x")
        XCTAssertEqual(partial.map(\.id), ["d/untagged"])

        Stub.serve(urls[2], status: 503, body: Data())
        do {
            _ = try await client.search("x")
            XCTFail("expected a failure")
        } catch let error as CoreMLDiffusionModelError {
            XCTAssertEqual(error, .searchFailed(status: 500))
        }
    }

    // MARK: Inspect

    func testRecordedAppleListingGivesTwoPinnedVariants() async throws {
        let repo = "apple/coreml-stable-diffusion-v1-5-palettized"
        let revision = Fixtures.sd15Revision
        Stub.serve(try Self.listingURL(repo), body: try Fixtures.data(Fixtures.sd15Listing))
        Stub.serve(try Self.probeURL(repo, revision: revision, folder: "split_einsum_v2/compiled"),
                   body: try Fixtures.data(Fixtures.sd15Split))
        Stub.serve(try Self.probeURL(repo, revision: revision, folder: "original/compiled"),
                   body: try Fixtures.data(Fixtures.sd15Original))

        let report = try await client.inspect("https://huggingface.co/\(repo)", on: .iPhone17Pro)

        XCTAssertEqual(report.revision, revision)
        XCTAssertEqual(report.variants.map(\.folder.path), ["original/compiled", "split_einsum_v2/compiled"])
        XCTAssertEqual(report.variants.map(\.preferredComputeUnits), [.gpu, .neuralEngine])
        XCTAssertEqual(report.variants.map(\.verdict), [.runnableWithCaveats, .runnableWithCaveats],
                       "the card's license is `other` (R18); original also runs on the GPU (R10)")
        XCTAssertEqual(report.recommended?.folder.path, "split_einsum_v2/compiled")
        XCTAssertEqual(report.recommended?.folder.downloadBytes, 888_856_623)
        XCTAssertNil(report.unsupportedReason)

        let probes = Stub.requests(containing: "/resolve/")
        XCTAssertEqual(probes.count, 2)
        XCTAssertTrue(probes.allSatisfy { $0.url?.path.contains("/resolve/\(revision)/") ?? false })
        XCTAssertTrue(Stub.requests.allSatisfy { $0.value(forHTTPHeaderField: "Authorization") == nil })
    }

    func testAtMostFourFoldersAreProbed() async throws {
        let repo = "o/six-folders"
        Stub.serve(try Self.listingURL(repo), body: try Self.listing(repo, folders: 6))
        for index in 0..<6 {
            Stub.serve(try Self.probeURL(repo, revision: Fixtures.sd15Revision, folder: "f\(index)"),
                       body: try Fixtures.data(Fixtures.sd21Split))
        }

        let report = try await client.inspect(repo, on: .iPhone17Pro)

        XCTAssertEqual(Stub.requests(containing: "/resolve/").count, CoreMLDiffusionInspector.maxProbedFolders)
        XCTAssertEqual(report.variants.map(\.verdict), [.runnable, .runnable, .runnable, .runnable, .unknown, .unknown])
        XCTAssertTrue(report.variants[5].findings.contains { $0.rule == "R1" && $0.detail.hasPrefix("Not probed") })
    }

    func testAFailedProbeLeavesThatVariantUnknown() async throws {
        let repo = "o/two-folders"
        Stub.serve(try Self.listingURL(repo), body: try Self.listing(repo, folders: 2))
        Stub.serve(try Self.probeURL(repo, revision: Fixtures.sd15Revision, folder: "f0"),
                   body: try Fixtures.data(Fixtures.sd21Split))

        let report = try await client.inspect(repo, on: .iPhone17Pro)

        XCTAssertEqual(report.variants.map(\.verdict), [.runnable, .unknown])
        XCTAssertTrue(report.variants[1].findings.contains { $0.detail.contains("HTTP 404") })
    }

    func testAGatedRepoIsNeverProbed() async throws {
        let repo = "o/gated"
        Stub.serve(try Self.listingURL(repo), body: try Self.listing(repo, folders: 1, gated: "manual"))

        let report = try await client.inspect(repo, on: .iPhone17Pro)

        XCTAssertTrue(Stub.requests(containing: "/resolve/").isEmpty)
        XCTAssertEqual(report.variants.first?.verdict, .notRunnable)
        XCTAssertTrue(report.variants.first?.findings.contains { $0.rule == "R17" } ?? false)
    }

    func testListingFailuresThrow() async throws {
        await assertThrows(.repositoryUnavailable(repoID: "o/missing", status: 404)) {
            _ = try await self.client.inspect("o/missing", on: .iPhone17Pro)
        }

        let repo = "o/no-sha"
        Stub.serve(try Self.listingURL(repo), body: try Self.listing(repo, folders: 1, sha: nil))
        await assertThrows(.repositoryUnavailable(repoID: repo, status: nil)) {
            _ = try await self.client.inspect(repo, on: .iPhone17Pro)
        }

        await assertThrows(.invalidRepoID("not a repo")) {
            _ = try await self.client.inspect("not a repo", on: .iPhone17Pro)
        }
        XCTAssertTrue(Stub.requests(containing: "/resolve/").isEmpty)
    }

    func testAMissingOrPrivateRepoIsNotCalledGated() {
        let texts = [401, 403, 404].map {
            CoreMLDiffusionModelError.repositoryUnavailable(repoID: "o/r", status: $0).recoverySuggestion ?? ""
        }
        XCTAssertTrue(texts.allSatisfy { $0.hasPrefix("Not found, or private") && !$0.contains("gated") }, "\(texts)")
    }

    func testARepoWithoutFoldersSaysWhy() async throws {
        let repo = "o/zips"
        Stub.serve(try Self.listingURL(repo), body: try Self.json([
            "id": repo, "sha": Fixtures.sd15Revision, "siblings": [["rfilename": "model_split-einsum.zip", "size": 5]],
        ]))

        let report = try await client.inspect(repo, on: .iPhone17Pro)

        XCTAssertTrue(report.variants.isEmpty)
        XCTAssertEqual(report.unsupportedReason, "Core ML archives (zip) are not supported yet")
        XCTAssertNil(report.recommended)
    }

    private func assertThrows(_ expected: CoreMLDiffusionModelError, file: StaticString = #filePath,
                              line: UInt = #line, _ body: () async throws -> Void) async {
        do {
            try await body()
            XCTFail("expected \(expected)", file: file, line: line)
        } catch let error as CoreMLDiffusionModelError {
            XCTAssertEqual(error, expected, file: file, line: line)
        } catch {
            XCTFail("unexpected \(error)", file: file, line: line)
        }
    }
}
