import XCTest
@testable import AuraCore

/// "Does it fit?" was only half the question. A model can sit comfortably in RAM and
/// still take minutes per answer — reporting that as `.excellent` is worse than useless.
/// These lock the wall-clock half: decode is memory-bound, so tok/s ≈ bandwidth ÷ weights.
final class DecodeSpeedTests: XCTestCase {

    /// Chips we have no bandwidth figure for (iOS, future silicon) must say "unknown",
    /// never guess. A fabricated number here would silently poison every verdict.
    private let unknownChip = HardwareProfile(
        totalMemoryGB: 32, availableMemoryGB: 13, deviceName: "Mystery", memoryBandwidthGBs: nil
    )

    // MARK: - The anchor

    /// The efficiency constant is not a vibe: it is fitted to the one point we actually
    /// measured with llama-bench on this machine. If someone edits it, this fails.
    func testEstimateMatchesTheLlamaBenchMeasurement() {
        // M1 Pro, 200 GB/s spec, 11.78 GB of weights → 12.12 tok/s measured.
        let predicted = (200.0 * HardwareAnalyzer.bandwidthEfficiency) / 11.78
        XCTAssertEqual(predicted, 12.12, accuracy: 2.0,
            "efficiency constant must keep tracking the measured llama-bench point")
    }

    #if os(macOS)
    /// The table must actually resolve this Mac's chip — an undetected chip degrades
    /// every assessment to "speed unknown".
    func testThisMacsChipIsRecognised() throws {
        let bandwidth = try XCTUnwrap(HardwareProfile.current().memoryBandwidthGBs,
                                      "Apple Silicon bandwidth must resolve from the chip table")
        XCTAssertGreaterThan(bandwidth, 50)
        XCTAssertLessThanOrEqual(bandwidth, 800)
        print(String(format: "detected %.0f GB/s", bandwidth))
    }
    #endif

    // MARK: - Fit and speed are different questions

    /// The headline case: 64 GB of RAM makes a 40 GB model *fit*, but a 68 GB/s bus
    /// makes it unusable. Memory alone would have called this a win.
    func testAModelCanFitAndStillBeTooSlow() throws {
        let bigSlowMac = HardwareProfile(
            totalMemoryGB: 64, availableMemoryGB: 60, deviceName: "Apple M1", memoryBandwidthGBs: 68
        )
        let heavy = Model.llama3_1_70b_gguf
        let result = HardwareAnalyzer.assess(heavy, profile: bigSlowMac)

        XCTAssertTrue(result.fitLevel.isRunnable, "precondition: 60 GB of RAM fits this model")
        let tps = try XCTUnwrap(result.estimatedDecodeTokensPerSecond)
        XCTAssertLessThan(tps, 5, "a 70B on a 68 GB/s bus cannot be fast")
        XCTAssertEqual(result.speedLevel, .unusable)
        XCTAssertTrue(result.verdict.contains("too slow"), "verdict must admit it: \(result.verdict)")
    }

    /// Same model, wider bus → faster. Bandwidth is the lever, not RAM.
    func testSpeedScalesWithBandwidth() throws {
        func tps(bandwidth: Double) throws -> Double {
            let profile = HardwareProfile(totalMemoryGB: 64, availableMemoryGB: 60,
                                          deviceName: "test", memoryBandwidthGBs: bandwidth)
            return try XCTUnwrap(
                HardwareAnalyzer.estimatedDecodeTokensPerSecond(for: .llama3_1_8b_gguf, profile: profile))
        }
        XCTAssertGreaterThan(try tps(bandwidth: 400), try tps(bandwidth: 200))
    }

    /// Lighter weights stream faster through the same bus.
    func testSmallerModelsDecodeFaster() throws {
        let mac = HardwareProfile(totalMemoryGB: 32, availableMemoryGB: 13,
                                  deviceName: "Apple M1 Pro", memoryBandwidthGBs: 200)
        let small = try XCTUnwrap(
            HardwareAnalyzer.estimatedDecodeTokensPerSecond(for: .dolphin3_qwen25_1_5b_gguf, profile: mac))
        let large = try XCTUnwrap(
            HardwareAnalyzer.estimatedDecodeTokensPerSecond(for: .llama3_1_8b_gguf, profile: mac))
        XCTAssertGreaterThan(small, large)
    }

    // MARK: - Honesty when we don't know

    func testNoBandwidthMeansNoEstimateRatherThanAGuess() {
        let result = HardwareAnalyzer.assess(.llama3_1_8b_gguf, profile: unknownChip)
        XCTAssertNil(result.estimatedDecodeTokensPerSecond)
        XCTAssertEqual(result.speedLevel, .unknown)
        XCTAssertNil(result.estimatedResponseSeconds())
        XCTAssertEqual(result.verdict, result.fitLevel.label, "unknown speed must not fabricate a claim")
    }

    /// A 500-token answer at ~12 tok/s is ~40 s — the number a user can act on.
    func testResponseTimeIsReportedInSeconds() throws {
        let mac = HardwareProfile(totalMemoryGB: 32, availableMemoryGB: 13,
                                  deviceName: "Apple M1 Pro", memoryBandwidthGBs: 200)
        let result = HardwareAnalyzer.assess(.llama3_1_8b_gguf, profile: mac)
        let seconds = try XCTUnwrap(result.estimatedResponseSeconds(tokens: 500))
        let tps = try XCTUnwrap(result.estimatedDecodeTokensPerSecond)
        XCTAssertEqual(seconds, 500 / tps, accuracy: 0.001)
        XCTAssertGreaterThan(seconds, 0)
    }
}
