import XCTest
@testable import AuraCore

/// `ModelAdvisor` — which GGUF quant to download for a given Mac. Helps the user pick without knowing quant
/// naming: fit estimate per file + a recommended one (highest quality that still fits comfortably).
final class ModelAdvisorTests: XCTestCase {

    private func file(_ bytes: Int) -> RemoteGGUFFile {
        RemoteGGUFFile(path: "m-\(bytes).gguf", sizeBytes: bytes,
                       resolveURL: URL(string: "https://huggingface.co/o/r/resolve/main/m.gguf")!)
    }

    func testFitBuckets() {
        // 16 GB Mac (working set = 1.3× file; comfortable ≤ 0.6·RAM, tight ≤ 0.85·RAM).
        XCTAssertEqual(ModelAdvisor.fit(sizeBytes: 2_000_000_000, ramGB: 16), .comfortable)   // 2.6 ≤ 9.6
        XCTAssertEqual(ModelAdvisor.fit(sizeBytes: 9_000_000_000, ramGB: 16), .tight)         // 11.7 ≤ 13.6
        XCTAssertEqual(ModelAdvisor.fit(sizeBytes: 20_000_000_000, ramGB: 16), .tooLarge)     // 26 > 13.6
        XCTAssertEqual(ModelAdvisor.fit(sizeBytes: nil, ramGB: 16), .unknown)
        XCTAssertEqual(ModelAdvisor.fit(sizeBytes: 2_000_000_000, ramGB: 0), .unknown)
    }

    func testRecommendsLargestComfortable() {
        let files = [file(1_000_000_000), file(3_000_000_000), file(6_000_000_000), file(20_000_000_000)]
        // On 16 GB: 6 GB (×1.3 = 7.8 ≤ 9.6) is the largest comfortable; 20 GB is too large.
        XCTAssertEqual(ModelAdvisor.recommended(from: files, ramGB: 16)?.sizeBytes, 6_000_000_000)
    }

    func testRecommendsSmallestWhenNoneComfortable() {
        let files = [file(12_000_000_000), file(30_000_000_000)]   // on 16 GB none is comfortable
        let rec = ModelAdvisor.recommended(from: files, ramGB: 16)
        XCTAssertEqual(rec?.sizeBytes, 12_000_000_000)             // smallest tight (or smallest overall)
    }

    func testEmptyIsNil() {
        XCTAssertNil(ModelAdvisor.recommended(from: [], ramGB: 16))
    }
}
