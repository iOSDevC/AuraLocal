import XCTest
@testable import AuraCore

/// The compatibility filter's whole job is to NOT lie about what fits. The trap it must avoid is the
/// one this session kept finding: treating on-disk size as peak RAM. Diffusion models peak at ~2× their
/// weights (measured: FLUX schnell 4-bit 9.2 GB → ~20 GB), so the LLM fit math would wave a 6.8 GB FLUX
/// through as "excellent" on a machine where it actually thrashes.
final class HFSearchFitTests: XCTestCase {

    private let gb = 1_073_741_824.0
    private func bytes(_ g: Double) -> Int { Int(g * 1_073_741_824) }
    private func profile(availGB: Double) -> HardwareProfile {
        HardwareProfile(totalMemoryGB: availGB + 4, availableMemoryGB: availGB, deviceName: "test")
    }

    // MARK: - the headline: kind changes the verdict for the SAME size

    func testDiffusionModelIsNotJudgedByLLMMath() {
        let fluxBytes = bytes(6.8)             // a Q4 FLUX file, like the LM Studio screenshot's 6.81 GB
        let tight = profile(availGB: 13)       // this session's measured dev-Mac headroom

        let asDiffusion = HardwareAnalyzer.fitLevel(forWeightsBytes: fluxBytes, kind: .diffusion, profile: tight)
        let asLLM       = HardwareAnalyzer.fitLevel(forWeightsBytes: fluxBytes, kind: .llm, profile: tight)

        // 6.8 × 2.2 = 15 GB peak > 13 GB → must NOT pass; 6.8 × 1.15 = 7.8 GB → comfortable.
        XCTAssertEqual(asDiffusion, .tooLarge, "a 6.8 GB FLUX peaks ~15 GB — it can't be 'fine' on 13 GB")
        XCTAssertTrue([.good, .excellent].contains(asLLM), "the same bytes as an LLM genuinely fit")
        XCTAssertNotEqual(asDiffusion, asLLM, "kind must change the verdict")
    }

    func testDiffusionFitsOnABigMac() {
        let flux = HardwareAnalyzer.fitLevel(forWeightsBytes: bytes(6.8), kind: .diffusion, profile: profile(availGB: 32))
        XCTAssertEqual(flux, .excellent, "6.8 × 2.2 = 15 GB is comfortable in 32 GB")
    }

    func testLLMCanStreamWhenDiffusionCannot() {
        let big = bytes(20)                    // weights bigger than available
        let dev = profile(availGB: 10)
        XCTAssertEqual(HardwareAnalyzer.fitLevel(forWeightsBytes: big, kind: .llm, profile: dev), .streamingRequired)
        XCTAssertEqual(HardwareAnalyzer.fitLevel(forWeightsBytes: big, kind: .diffusion, profile: dev), .tooLarge)
    }

    func testZeroOrUnknownSizeIsTooLarge() {
        XCTAssertEqual(HardwareAnalyzer.fitLevel(forWeightsBytes: 0, kind: .llm, profile: profile(availGB: 32)), .tooLarge)
    }

    // MARK: - search JSON parsing (pure, no network)

    func testParsesSearchHitsIncludingPolymorphicGated() {
        let json = Data("""
        [
          {"id":"black-forest-labs/FLUX.1-schnell","likes":5407,"downloads":245990,
           "pipeline_tag":"text-to-image","tags":["diffusers","flux","text-to-image"],"gated":"auto"},
          {"id":"dhairyashil/FLUX.1-schnell-mflux-4bit","likes":2,"downloads":594,
           "pipeline_tag":"text-to-image","tags":["mlx","text-to-image"],"gated":false},
          {"id":"someone/Qwen2.5-7B-GGUF","downloads":10,"tags":["gguf"],"gated":false}
        ]
        """.utf8)
        let hits = HuggingFaceSearch.models(fromSearchJSON: json)
        XCTAssertEqual(hits.count, 3)

        let official = hits[0]
        XCTAssertEqual(official.id, "black-forest-labs/FLUX.1-schnell")
        XCTAssertEqual(official.likes, 5407)
        XCTAssertTrue(official.gated, "\"auto\" must count as gated (needs HF login)")
        XCTAssertTrue(official.isDiffusion)
        XCTAssertEqual(official.kind, .diffusion)

        let mirror = hits[1]
        XCTAssertFalse(mirror.gated, "false must be ungated")
        XCTAssertEqual(mirror.owner, "dhairyashil")

        let llm = hits[2]
        XCTAssertFalse(llm.isDiffusion, "a gguf text repo is not diffusion")
        XCTAssertEqual(llm.kind, .llm)
        XCTAssertEqual(llm.likes, 0, "missing likes defaults to 0")
    }

    func testSearchURLEncodesQueryAndIsEmptyForBlank() {
        XCTAssertNil(HuggingFaceSearch.searchURL(query: "   ", limit: 10, sort: .downloads))
        let url = HuggingFaceSearch.searchURL(query: "flux schnell", limit: 5, sort: .likes)
        XCTAssertEqual(url?.host, "huggingface.co")
        let q = url?.query ?? ""
        XCTAssertTrue(q.contains("search=flux%20schnell") || q.contains("search=flux+schnell"))
        XCTAssertTrue(q.contains("sort=likes"))
        XCTAssertTrue(q.contains("full=true"), "need full=true so tags/gated come back")
    }
}
