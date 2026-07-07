import XCTest
@testable import AuraCore

/// `HuggingFaceRepo` — turn a repository URL (the page users actually paste) into its downloadable `.gguf`
/// files. Pure parsing + resolve-URL building are tested here against a canned tree-API payload; the network
/// fetch (`listGGUF`) is exercised live.
final class HuggingFaceRepoTests: XCTestCase {

    // MARK: - parse

    func testParsesBareRepoURL() throws {
        let ref = try XCTUnwrap(HuggingFaceRepo.parse("https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF"))
        XCTAssertEqual(ref.owner, "unsloth")
        XCTAssertEqual(ref.repo, "gemma-4-E2B-it-GGUF")
        XCTAssertEqual(ref.revision, "main")
    }

    func testParsesTreeBrowseURLWithRevision() throws {
        let ref = try XCTUnwrap(HuggingFaceRepo.parse("https://huggingface.co/owner/repo/tree/dev/subdir"))
        XCTAssertEqual(ref.owner, "owner")
        XCTAssertEqual(ref.repo, "repo")
        XCTAssertEqual(ref.revision, "dev")
    }

    func testParseToleratesTrailingSlashAndWww() throws {
        XCTAssertEqual(HuggingFaceRepo.parse("https://www.huggingface.co/o/r/")?.repo, "r")
        XCTAssertEqual(HuggingFaceRepo.parse("  https://huggingface.co/o/r  ")?.owner, "o")
    }

    func testParseRejectsNonRepoURLs() {
        let bad = [
            "https://huggingface.co/o/r/resolve/main/x.gguf",   // direct file → Model.fromHuggingFaceURL owns it
            "https://huggingface.co/o/r/blob/main/x.gguf",      // view-file link
            "https://huggingface.co/datasets/o/r",              // a dataset, not a model repo
            "https://huggingface.co/models",                    // a route, not a repo
            "https://example.com/o/r",                          // wrong host
            "http://huggingface.co/o/r",                        // not https
            "https://huggingface.co/onlyowner",                 // no repo segment
            "file:///tmp/x.gguf",
            "not a url"
        ]
        for url in bad { XCTAssertNil(HuggingFaceRepo.parse(url), "should reject: \(url)") }
    }

    // MARK: - URL construction

    func testTreeAPIURLIsRecursive() throws {
        let ref = HuggingFaceRepo.Ref(owner: "unsloth", repo: "gemma-4-E2B-it-GGUF", revision: "main")
        let url = try XCTUnwrap(HuggingFaceRepo.treeAPIURL(ref))
        XCTAssertEqual(url.absoluteString,
                       "https://huggingface.co/api/models/unsloth/gemma-4-E2B-it-GGUF/tree/main?recursive=true")
    }

    func testResolveURLPreservesSubfolderAndEncodes() throws {
        let root = try XCTUnwrap(HuggingFaceRepo.resolveURL(owner: "o", repo: "r", revision: "main",
                                                            path: "model.Q4_K_M.gguf"))
        XCTAssertEqual(root.absoluteString, "https://huggingface.co/o/r/resolve/main/model.Q4_K_M.gguf")

        let sub = try XCTUnwrap(HuggingFaceRepo.resolveURL(owner: "o", repo: "r", revision: "main",
                                                           path: "MTP/model Q8_0.gguf"))
        // Slashes preserved between segments; the space inside a segment is percent-encoded.
        XCTAssertEqual(sub.absoluteString, "https://huggingface.co/o/r/resolve/main/MTP/model%20Q8_0.gguf")
    }

    // MARK: - tree JSON → files

    private let treeJSON = Data("""
    [
      {"type":"directory","oid":"a","size":0,"path":"MTP"},
      {"type":"file","oid":"b","size":3891,"path":".gitattributes"},
      {"type":"file","oid":"c","size":97817664,"path":"MTP/gemma-4-E2B-it-Q8_0-MTP.gguf"},
      {"type":"file","oid":"d","size":28500,"path":"README.md"},
      {"type":"file","oid":"e","size":1600000000,"path":"gemma-4-E2B-it-Q4_K_M.gguf"},
      {"type":"file","oid":"f","size":900000000,"path":"gemma-4-E2B-it-Q2_K.gguf"}
    ]
    """.utf8)

    func testGGUFFilesFilterSortAndResolve() throws {
        let files = HuggingFaceRepo.ggufFiles(fromTreeJSON: treeJSON,
                                              owner: "unsloth", repo: "gemma-4-E2B-it-GGUF", revision: "main")
        // Only the three .gguf files; the directory, README, .gitattributes are dropped.
        XCTAssertEqual(files.count, 3)
        // Sorted smallest → largest: Q8_0-MTP (97 MB) < Q2_K (900 MB) < Q4_K_M (1.6 GB).
        XCTAssertEqual(files.map(\.filename),
                       ["gemma-4-E2B-it-Q8_0-MTP.gguf", "gemma-4-E2B-it-Q2_K.gguf", "gemma-4-E2B-it-Q4_K_M.gguf"])
        // Subfolder path is preserved in both `path` and the resolve URL.
        let sub = try XCTUnwrap(files.first)
        XCTAssertEqual(sub.path, "MTP/gemma-4-E2B-it-Q8_0-MTP.gguf")
        XCTAssertEqual(sub.resolveURL.absoluteString,
            "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/MTP/gemma-4-E2B-it-Q8_0-MTP.gguf")
        XCTAssertEqual(sub.sizeBytes, 97817664)
        // Each resolve URL round-trips through Model.fromURL (the load path).
        for file in files { XCTAssertNotNil(Model.fromURL(file.resolveURL.absoluteString)) }
    }

    func testMalformedJSONYieldsEmpty() {
        XCTAssertTrue(HuggingFaceRepo.ggufFiles(fromTreeJSON: Data("{not json".utf8),
                                                owner: "o", repo: "r", revision: "main").isEmpty)
    }

    // MARK: - quant labels

    func testQuantLabelExtraction() {
        XCTAssertEqual(HuggingFaceRepo.quantLabel(for: "gemma-4-E2B-it-Q4_K_M.gguf"), "Q4_K_M")
        XCTAssertEqual(HuggingFaceRepo.quantLabel(for: "google_gemma-3-1b-it-IQ3_XS.gguf"), "IQ3_XS")
        XCTAssertEqual(HuggingFaceRepo.quantLabel(for: "model-Q8_0.gguf"), "Q8_0")
        XCTAssertEqual(HuggingFaceRepo.quantLabel(for: "model-BF16.gguf"), "BF16")
        XCTAssertNil(HuggingFaceRepo.quantLabel(for: "model.gguf"))
    }

    // MARK: - robustness (review follow-ups)

    func testShortHfCoHostAccepted() throws {
        XCTAssertEqual(HuggingFaceRepo.parse("https://hf.co/unsloth/gemma-4-E2B-it-GGUF")?.repo,
                       "gemma-4-E2B-it-GGUF")
        // A direct hf.co file URL still loads via Model.fromURL.
        XCTAssertNotNil(Model.fromURL("https://hf.co/o/r/resolve/main/x.gguf"))
    }

    func testShardsAndTraversalPathsAreDropped() {
        let json = Data(#"""
        [
          {"type":"file","size":100,"path":"model-Q4_K_M-00001-of-00003.gguf"},
          {"type":"file","size":100,"path":"model-Q4_K_M-00002-of-00003.gguf"},
          {"type":"file","size":100,"path":"model-Q4_K_M-00003-of-00003.gguf"},
          {"type":"file","size":50,"path":"../../evil/repo/resolve/main/x.gguf"},
          {"type":"file","size":200,"path":"model-Q8_0.gguf"}
        ]
        """#.utf8)
        let files = HuggingFaceRepo.ggufFiles(fromTreeJSON: json, owner: "o", repo: "r", revision: "main")
        // Multi-part shards + the traversal path are dropped; only the single-file Q8_0 remains.
        XCTAssertEqual(files.map(\.filename), ["model-Q8_0.gguf"])
    }

    func testIsShard() {
        XCTAssertTrue(HuggingFaceRepo.isShard("model-Q4_K_M-00001-of-00003.gguf"))
        XCTAssertFalse(HuggingFaceRepo.isShard("model-Q4_K_M.gguf"))
        XCTAssertFalse(HuggingFaceRepo.isShard("model-1-of-3.gguf"))   // not zero-padded 5-digit
    }

    func testSizeText() {
        let sized = RemoteGGUFFile(path: "x.gguf", sizeBytes: 1_600_000_000,
                                   resolveURL: URL(string: "https://huggingface.co/o/r/resolve/main/x.gguf")!)
        XCTAssertNotNil(sized.sizeText)
        let unsized = RemoteGGUFFile(path: "x.gguf", sizeBytes: nil,
                                     resolveURL: URL(string: "https://huggingface.co/o/r/resolve/main/x.gguf")!)
        XCTAssertNil(unsized.sizeText)
    }

    func testListGGUFRejectsDirectFileURL() async {
        do {
            _ = try await HuggingFaceRepo.listGGUF(repoURL: "https://huggingface.co/o/r/resolve/main/x.gguf")
            XCTFail("a direct file URL is not a repository")
        } catch {
            XCTAssertEqual(error as? HuggingFaceRepo.RepoError, .notARepositoryURL)
        }
    }
}
