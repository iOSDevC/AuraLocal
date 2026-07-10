import XCTest
@testable import AuraCore

final class DownloadAuthTests: XCTestCase {

    func testNoAuthNeverAddsHeaders() {
        let h = NoDownloadAuth().headers(for: URL(string: "https://huggingface.co/x/resolve/main/f.gguf")!)
        XCTAssertTrue(h.isEmpty)
    }

    func testNoTokenMeansNoHeader() {
        KeychainStore.delete(for: KeychainDownloadAuth.huggingFaceAccount)
        let h = KeychainDownloadAuth().headers(for: URL(string: "https://huggingface.co/x/resolve/main/f.gguf")!)
        XCTAssertTrue(h.isEmpty, "public downloads must be unaffected")
    }

    func testHFTokenAddedToAPIHostButNotToSignedCDN() throws {
        do { try KeychainStore.save("hf_secret", for: KeychainDownloadAuth.huggingFaceAccount) }
        catch { throw XCTSkip("Keychain unavailable in this test host: \(error)") }
        defer { KeychainStore.delete(for: KeychainDownloadAuth.huggingFaceAccount) }

        let auth = KeychainDownloadAuth()
        // API host → token attached.
        XCTAssertEqual(
            auth.headers(for: URL(string: "https://huggingface.co/x/resolve/main/f.gguf")!)["Authorization"],
            "Bearer hf_secret")
        // Signed CDN (redirect target) → token must NOT leak.
        XCTAssertTrue(
            auth.headers(for: URL(string: "https://cdn-lfs.huggingface.co/repos/a/b/file?sig=xyz")!).isEmpty)
    }

    func testRequestHelperAppliesHeader() throws {
        do { try KeychainStore.save("hf_tok", for: KeychainDownloadAuth.huggingFaceAccount) }
        catch { throw XCTSkip("Keychain unavailable") }
        defer { KeychainStore.delete(for: KeychainDownloadAuth.huggingFaceAccount) }
        let req = KeychainDownloadAuth().request(for: URL(string: "https://huggingface.co/x")!)
        XCTAssertEqual(req.value(forHTTPHeaderField: "Authorization"), "Bearer hf_tok")
    }
}
