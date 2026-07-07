import XCTest
@testable import AuraCore

/// `Model.fromHuggingFaceURL` — the bring-your-own-model parser: the user pastes a Hugging Face `.gguf` URL
/// and AuraLocal downloads it. No bundled/curated weights, so the user owns the model-license choice.
final class UserModelTests: XCTestCase {

    func testResolveURL() throws {
        let url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        let m = try XCTUnwrap(Model.fromHuggingFaceURL(url))
        XCTAssertEqual(m.repoID, "TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
        XCTAssertEqual(m.ggufFilename, "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        XCTAssertEqual(m.format, .gguf)
        XCTAssertTrue(m.isUserSupplied)
        XCTAssertEqual(m.downloadURL?.absoluteString, url)      // resolve URL preserved verbatim
        XCTAssertEqual(m.displayName, "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    }

    func testBlobIsNormalizedToResolve() throws {
        let m = try XCTUnwrap(Model.fromHuggingFaceURL("https://huggingface.co/owner/repo/blob/main/model.Q4_K_M.gguf"))
        XCTAssertEqual(m.repoID, "owner/repo")
        XCTAssertEqual(m.downloadURL?.absoluteString, "https://huggingface.co/owner/repo/resolve/main/model.Q4_K_M.gguf")
    }

    func testRevisionAndSubfolderPreserved() throws {
        let m = try XCTUnwrap(Model.fromHuggingFaceURL("https://huggingface.co/org/name/resolve/abc123/quants/tiny.Q2_K.gguf"))
        XCTAssertEqual(m.repoID, "org/name")
        XCTAssertEqual(m.ggufFilename, "tiny.Q2_K.gguf")     // last path segment, subfolder stripped from filename
        XCTAssertEqual(m.downloadURL?.absoluteString, "https://huggingface.co/org/name/resolve/abc123/quants/tiny.Q2_K.gguf")
    }

    func testDisplayNameOverride() throws {
        let m = try XCTUnwrap(Model.fromHuggingFaceURL("https://huggingface.co/o/r/resolve/main/x.gguf", displayName: "My Model"))
        XCTAssertEqual(m.displayName, "My Model")
        XCTAssertEqual(m.id.hasPrefix("custom__"), true)
    }

    func testWwwHostAccepted() throws {
        XCTAssertNotNil(Model.fromHuggingFaceURL("https://www.huggingface.co/o/r/resolve/main/x.gguf"))
    }

    func testRejectsInvalid() {
        let bad = [
            "http://huggingface.co/o/r/resolve/main/x.gguf",          // not https
            "https://example.com/o/r/resolve/main/x.gguf",            // wrong host
            "https://huggingface.co/o/r/resolve/main/x.safetensors",  // not a .gguf
            "https://huggingface.co/o/r/resolve/main/.gguf",          // empty base name
            "https://huggingface.co/o/r/tree/main/x.gguf",            // not resolve/blob
            "https://huggingface.co/o/r",                             // bare repo, no file
            "https://huggingface.co/o/resolve/main/x.gguf",           // only one segment before resolve
            "not a url at all",
            "",
        ]
        for s in bad { XCTAssertNil(Model.fromHuggingFaceURL(s), "should reject: \(s)") }
    }

    // MARK: - Other sources (bridge)

    func testDirectHTTPSURL() throws {
        let url = "https://github.com/someorg/models/releases/download/v1/tinyllama.Q4_K_M.gguf"
        let m = try XCTUnwrap(Model.fromURL(url))
        XCTAssertEqual(m.ggufFilename, "tinyllama.Q4_K_M.gguf")
        XCTAssertEqual(m.format, .gguf)
        XCTAssertTrue(m.isUserSupplied)
        XCTAssertNil(m.localFileURL)
        XCTAssertEqual(m.downloadURL?.absoluteString, url)                 // downloaded verbatim from any host
        XCTAssertEqual(m.repoID, "url/github_com")                          // grouped by host for the cache
        XCTAssertEqual(m.resolvedFileURL?.lastPathComponent, "tinyllama.Q4_K_M.gguf")
    }

    func testLocalFileURL() throws {
        let m = try XCTUnwrap(Model.fromURL("file:///Users/me/models/phi-3.Q4_K_M.gguf"))
        XCTAssertEqual(m.ggufFilename, "phi-3.Q4_K_M.gguf")
        XCTAssertNil(m.downloadURL)                                         // nothing to download
        XCTAssertTrue(m.isUserSupplied)
        XCTAssertEqual(m.localFileURL?.path, "/Users/me/models/phi-3.Q4_K_M.gguf")
        XCTAssertEqual(m.resolvedFileURL, m.localFileURL)                   // loaded in place
    }

    func testFromURLAutoDetectRoutesBySource() throws {
        // file:// → local, huggingface.co → HF, other https .gguf → direct
        XCTAssertNotNil(Model.fromURL("file:///tmp/x.gguf")?.localFileURL)
        XCTAssertEqual(Model.fromURL("https://huggingface.co/o/r/resolve/main/x.gguf")?.repoID, "o/r")
        XCTAssertEqual(Model.fromURL("https://cdn.example.com/x.gguf")?.repoID, "url/cdn_example_com")
    }

    func testFromModelSourceExplicit() throws {
        let hf = try XCTUnwrap(Model.from(.huggingFace(url: "https://huggingface.co/o/r/resolve/main/x.gguf")))
        XCTAssertEqual(hf.repoID, "o/r")
        let direct = try XCTUnwrap(Model.from(.directURL(URL(string: "https://x.io/y.gguf")!)))
        XCTAssertEqual(direct.downloadURL?.absoluteString, "https://x.io/y.gguf")
        let local = try XCTUnwrap(Model.from(.localFile(URL(fileURLWithPath: "/m/z.gguf"))))
        XCTAssertEqual(local.localFileURL?.path, "/m/z.gguf")
    }

    func testOtherSourcesRejectInvalid() {
        XCTAssertNil(Model.fromURL("https://cdn.example.com/model.safetensors"))  // not gguf
        XCTAssertNil(Model.fromURL("ftp://host/x.gguf"))                          // unsupported scheme
        XCTAssertNil(Model.fromURL("file:///tmp/notgguf.bin"))                    // local but not gguf
        XCTAssertNil(Model.from(.directURL(URL(string: "http://x.io/y.gguf")!)))  // direct must be https
    }

    func testResolvedFileURLForRemoteModel() throws {
        let m = try XCTUnwrap(Model.fromURL("https://cdn.example.com/a.gguf"))
        // Remote model resolves to its download destination under the cache dir.
        XCTAssertEqual(m.resolvedFileURL, m.cacheDirectory.appendingPathComponent("a.gguf"))
    }

    func testCatalogModelHasNoDownloadURL() {
        // A JSON-decoded catalog model omits the key → nil → not user-supplied.
        let json = #"{"id":"x","repoID":"a/b","displayName":"X","category":"text","docTags":false,"format":"gguf","approximateSizeMB":100,"isUncensored":false,"ggufFilename":"x.gguf","defaultDocumentPrompt":null,"numLayers":1,"kvHeads":0,"headDim":0}"#
        let m = try? JSONDecoder().decode(Model.self, from: Data(json.utf8))
        XCTAssertNotNil(m)
        XCTAssertNil(m?.downloadURL)
        XCTAssertEqual(m?.isUserSupplied, false)
    }
}
