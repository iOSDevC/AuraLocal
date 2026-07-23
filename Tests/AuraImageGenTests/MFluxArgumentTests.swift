import XCTest
@testable import AuraImageGen

/// Locks the mflux CLI contract so a wrong flag can't ship silently. The flag SPELLINGS here were
/// verified against mflux's own CLI docs (`--lora-paths`, `--lora-scales`, `--model`, `--steps`,
/// `--seed`, `--width`, `--height`, `--quantize`, `--low-ram`, `--output`) — an adversarial reviewer
/// flagged that they must not be assumed. No model download or subprocess runs here.
final class MFluxArgumentTests: XCTestCase {

    private let out = URL(fileURLWithPath: "/tmp/out/image.png")

    func testMinimalRequestUsesModelPromptAndOutput() {
        let req = ImageGenRequest(prompt: "a red fox", quantize: nil)
        let a = MFluxEngine.buildArguments(for: req, output: out)
        XCTAssertEqual(pair(a, "--model"), "schnell")
        XCTAssertEqual(pair(a, "--prompt"), "a red fox")
        XCTAssertEqual(pair(a, "--output"), out.path)
        XCTAssertFalse(a.contains("--quantize"), "nil quantize must omit the flag")
        XCTAssertFalse(a.contains("--low-ram"))
    }

    func testLoRAFlagsMatchMfluxSpelling() {
        let req = ImageGenRequest(
            prompt: "portrait",
            loras: [
                LoRA(url: URL(fileURLWithPath: "/loras/a.safetensors"), scale: 1.0),
                LoRA(url: URL(fileURLWithPath: "/loras/b.safetensors"), scale: 0.8),
            ])
        let a = MFluxEngine.buildArguments(for: req, output: out)

        // --lora-paths <a> <b>, then --lora-scales <1.0> <0.8> — mflux's exact plural contract.
        let p = idx(a, "--lora-paths")
        XCTAssertEqual(Array(a[(p + 1)...(p + 2)]), ["/loras/a.safetensors", "/loras/b.safetensors"])
        let s = idx(a, "--lora-scales")
        XCTAssertEqual(Array(a[(s + 1)...(s + 2)]), ["1.0", "0.8"])
        XCTAssertLessThan(p, s, "paths must precede scales for positional pairing")
    }

    func testOptionalKnobsAppearOnlyWhenSet() {
        let req = ImageGenRequest(
            prompt: "x", model: "dev", steps: 20, seed: 42, width: 512, height: 768,
            quantize: 8, lowRAM: true)
        let a = MFluxEngine.buildArguments(for: req, output: out)
        XCTAssertEqual(pair(a, "--model"), "dev")
        XCTAssertEqual(pair(a, "--steps"), "20")
        XCTAssertEqual(pair(a, "--seed"), "42")
        XCTAssertEqual(pair(a, "--width"), "512")
        XCTAssertEqual(pair(a, "--height"), "768")
        XCTAssertEqual(pair(a, "--quantize"), "8")
        XCTAssertTrue(a.contains("--low-ram"))
    }

    func testNoLoRAOmitsLoraFlagsEntirely() {
        let a = MFluxEngine.buildArguments(for: ImageGenRequest(prompt: "x"), output: out)
        XCTAssertFalse(a.contains("--lora-paths"))
        XCTAssertFalse(a.contains("--lora-scales"))
    }

    #if os(macOS)
    /// A bogus override must fail with the install hint, not crash.
    func testMissingEngineThrowsInstallHint() {
        let engine = MFluxEngine(executableOverride: URL(fileURLWithPath: "/nope/mflux-generate"))
        XCTAssertFalse(engine.isAvailable)
        XCTAssertThrowsError(try engine.resolveExecutable()) { error in
            guard case ImageGenError.engineNotFound(let hint) = error else {
                return XCTFail("expected engineNotFound, got \(error)")
            }
            XCTAssertEqual(hint, "uv tool install mflux")
        }
    }
    #else
    /// Off macOS the engine must be unavailable and generation must refuse, not attempt a subprocess.
    func testUnsupportedPlatformRefuses() async {
        let engine = MFluxEngine()
        XCTAssertFalse(engine.isAvailable)
        do {
            _ = try await engine.generate(ImageGenRequest(prompt: "x"))
            XCTFail("should have thrown")
        } catch ImageGenError.unsupportedPlatform {
            // expected
        } catch {
            XCTFail("expected unsupportedPlatform, got \(error)")
        }
    }
    #endif

    // MARK: - helpers

    private func idx(_ a: [String], _ flag: String) -> Int {
        guard let i = a.firstIndex(of: flag) else { XCTFail("missing \(flag)"); return 0 }
        return i
    }
    private func pair(_ a: [String], _ flag: String) -> String? {
        guard let i = a.firstIndex(of: flag), i + 1 < a.count else { return nil }
        return a[i + 1]
    }
}
