import XCTest
@testable import AuraCore

/// Gates `HardwareAnalyzer.estimatedDecodeTokensPerSecond` against decode speeds that were measured. When a gate
/// fails, fix the estimator, not `maxError`. Our own points are gated one by one; third-party points by the median
/// of each chip tier.
final class EstimatorAccuracyTests: XCTestCase {

    private static let maxError = 0.25

    private enum ChipTier: String, CaseIterable {
        case base, pro, max, ultra
    }

    private struct Point {
        let label: String
        let tier: ChipTier
        let bandwidthGBs: Double
        let weightsGiB: Double
        let measuredTokensPerSecond: Double
    }

    // MARK: - Our measurements (M1 Pro 32 GB, 200 GB/s spec)

    /// Sources: llama-bench output for the Gemma row; docs/guide/distributed.md "Measured results" for the rest.
    /// Weights are the shipped file sizes on Hugging Face.
    private static let own: [Point] = [
        // llama-bench build 9660, tg64, one repetition: gemma4-coding-Q8_0.gguf (Gemma 4, 11.91 B params).
        Point(label: "Gemma 4 12B Q8_0, llama.cpp", tier: .pro, bandwidthGBs: 200, weightsGiB: 11.78,
              measuredTokensPerSecond: 12.12),
        // llama-bench build 9660, tg64: Qwen3-14B-Q4_K_M.gguf.
        Point(label: "Qwen3-14B Q4_K_M, llama.cpp", tier: .pro, bandwidthGBs: 200, weightsGiB: 8.384,
              measuredTokensPerSecond: 13.9),
        // mlx-lm 0.31.3 greedy, (n − 1) ÷ decode time, 19.0–19.8 tok/s on a clean run: mlx-community/Qwen3-14B-4bit.
        Point(label: "Qwen3-14B 4-bit, MLX", tier: .pro, bandwidthGBs: 200, weightsGiB: 7.737,
              measuredTokensPerSecond: 19.4),
        // mlx-lm 0.31.3, alone on the GPU: mlx-community/Qwen3-0.6B-4bit.
        Point(label: "Qwen3-0.6B 4-bit, MLX", tier: .pro, bandwidthGBs: 200, weightsGiB: 0.3124,
              measuredTokensPerSecond: 218),
    ]

    /// Own points the estimator misses, and why. Each must keep failing the gate: a third miss fails the test, and
    /// so does one of these starting to pass (then remove it here and in docs/guide/benchmarks.md).
    private static let knownMisses: [String: String] = [
        "Qwen3-14B Q4_K_M, llama.cpp":
            "predicted 18.2, measured 13.9; the constants come from LLaMA 2 7B Q4_0, Q8_0 and F16 rows",
        "Qwen3-0.6B 4-bit, MLX":
            "the fixed 6.6 ms per token caps every estimate at 151 tok/s; MLX measured 4.6 ms here",
    ]

    func testEachOwnMeasurementStaysWithinTheGate() throws {
        for point in Self.own {
            try assertWithinGate(point)
        }
        let labels = Set(Self.own.map(\.label))
        XCTAssertTrue(Self.knownMisses.keys.allSatisfy(labels.contains), "a known miss names no point")
    }

    private func assertWithinGate(_ point: Point) throws {
        let error = try relativeError(point)
        let detail = try describe(point)
        guard let reason = Self.knownMisses[point.label] else {
            return XCTAssertLessThanOrEqual(error, Self.maxError, detail)
        }
        XCTExpectFailure("known miss: \(reason)") {
            XCTAssertLessThanOrEqual(error, Self.maxError, detail)
        }
    }

    // MARK: - Third party: llama.cpp discussion #4167

    /// LLaMA 2 7B tg128, build 8e672ef, from the discussion's summary table (fetched 2026-10-14). Not reproduced
    /// by us. The estimator constants were fitted to the M1 and M2 rows only.
    ///
    /// Excluded: the M5 and M6 rows (build c1d0e7a). `bandwidthTable` has no M5 entry, so the estimator returns
    /// nil on those Macs; on them the fitted overhead underestimates Max and Ultra chips by up to 44 %, so adding
    /// M5 to the table needs its own overhead.
    private struct Row {
        let chip: String
        let tier: ChipTier
        let bandwidth: Double
        let f16: Double?
        let q8: Double
        let q4: Double
    }

    private static func row(_ chip: String, _ tier: ChipTier, _ bandwidth: Double, _ f16: Double?, _ q8: Double,
                            _ q4: Double) -> Row {
        Row(chip: chip, tier: tier, bandwidth: bandwidth, f16: f16, q8: q8, q4: q4)
    }

    private static let discussion4167: [Row] = [
        row("M1 7c", .base, 68, nil, 7.92, 14.19), row("M1 8c", .base, 68, nil, 7.91, 14.15),
        row("M1 Pro 14c", .pro, 200, 12.75, 21.95, 35.52), row("M1 Pro 16c", .pro, 200, 12.75, 22.34, 36.41),
        row("M1 Max 24c", .max, 400, 22.55, 37.81, 54.61), row("M1 Max 32c", .max, 400, 23.03, 40.20, 61.19),
        row("M1 Ultra 48c", .ultra, 800, 33.92, 55.69, 74.93), row("M1 Ultra 64c", .ultra, 800, 37.01, 59.87, 83.73),
        row("M2 8c", .base, 100, nil, 12.18, 21.70), row("M2 10c", .base, 100, 6.72, 12.21, 21.91),
        row("M2 Pro 16c", .pro, 200, 12.47, 22.70, 37.87), row("M2 Pro 19c", .pro, 200, 13.06, 23.01, 38.86),
        row("M2 Max 30c", .max, 400, 24.16, 39.97, 60.99), row("M2 Max 38c", .max, 400, 24.65, 41.83, 65.95),
        row("M2 Ultra 60c", .ultra, 800, 39.86, 62.14, 88.64), row("M2 Ultra 76c", .ultra, 800, 41.02, 66.64, 94.27),
        row("M3 10c", .base, 100, nil, 12.27, 21.34),
        row("M3 Pro 14c", .pro, 150, nil, 17.44, 30.65), row("M3 Pro 18c", .pro, 150, 9.89, 17.53, 30.74),
        row("M3 Max 30c", .max, 300, 19.54, 34.30, 56.58), row("M3 Max 40c", .max, 400, 25.09, 42.75, 66.31),
        row("M3 Ultra 60c", .ultra, 800, 42.24, 63.55, 88.40), row("M3 Ultra 80c", .ultra, 800, 39.78, 63.93, 92.14),
        row("M4 10c", .base, 120, 7.43, 13.54, 24.11),
        row("M4 Pro 16c", .pro, 273, 17.19, 30.54, 49.64), row("M4 Pro 20c", .pro, 273, 17.18, 30.69, 50.74),
        row("M4 Max 32c", .max, 410, 24.29, 43.87, 69.95), row("M4 Max 40c", .max, 546, 31.64, 54.05, 83.06),
    ]

    /// llama-bench model sizes for LLaMA 2 7B, GiB.
    private static let f16GiB = 12.55, q8GiB = 6.67, q4GiB = 3.56

    private static var thirdPartyPoints: [Point] {
        discussion4167.flatMap { row in
            [row.f16.map { makePoint(row.chip, "F16", row.tier, row.bandwidth, f16GiB, $0) },
             makePoint(row.chip, "Q8_0", row.tier, row.bandwidth, q8GiB, row.q8),
             makePoint(row.chip, "Q4_0", row.tier, row.bandwidth, q4GiB, row.q4)].compactMap { $0 }
        }
    }

    private static func makePoint(_ chip: String, _ quant: String, _ tier: ChipTier, _ bandwidth: Double,
                              _ weightsGiB: Double, _ measured: Double) -> Point {
        Point(label: "\(chip) \(quant)", tier: tier, bandwidthGBs: bandwidth, weightsGiB: weightsGiB,
              measuredTokensPerSecond: measured)
    }

    /// Per tier, so a dozen base and Pro points cannot hide an Ultra that is off by 80 %.
    func testThirdPartyPointsStayWithinTheGatePerTier() throws {
        try assertEachTierWithinGate(Self.thirdPartyPoints, name: "#4167 all M1–M4 rows")
    }

    /// The M3 and M4 rows played no part in the fit.
    func testThirdPartyHeldOutPointsStayWithinTheGatePerTier() throws {
        let heldOut = Self.thirdPartyPoints.filter { $0.label.hasPrefix("M3") || $0.label.hasPrefix("M4") }
        XCTAssertFalse(heldOut.isEmpty)
        try assertEachTierWithinGate(heldOut, name: "#4167 held-out M3/M4 rows")
    }

    // MARK: - Helpers

    private func assertEachTierWithinGate(_ points: [Point], name: String) throws {
        for tier in ChipTier.allCases {
            let members = points.filter { $0.tier == tier }
            XCTAssertFalse(members.isEmpty, "\(name): no \(tier.rawValue) points")
            let median = Self.median(try members.map(relativeError))
            let detail = try members.map(describe).joined(separator: "; ")
            XCTAssertLessThanOrEqual(median, Self.maxError, "\(name), \(tier.rawValue): \(detail)")
        }
    }

    private func predicted(_ point: Point) throws -> Double {
        let profile = HardwareProfile(totalMemoryGB: 192, availableMemoryGB: 128, deviceName: point.label,
                                      memoryBandwidthGBs: point.bandwidthGBs)
        let bytes = Int64((point.weightsGiB * 1_073_741_824).rounded())
        let model = CompatibilityEvaluator.sizingModel(format: .gguf, bytes: bytes, layers: nil, kvHeads: nil, headDim: nil)
        return try XCTUnwrap(HardwareAnalyzer.estimatedDecodeTokensPerSecond(for: model, profile: profile))
    }

    private func relativeError(_ point: Point) throws -> Double {
        abs(try predicted(point) - point.measuredTokensPerSecond) / point.measuredTokensPerSecond
    }

    private func describe(_ point: Point) throws -> String {
        String(format: "%@ predicted %.1f measured %.2f", point.label, try predicted(point), point.measuredTokensPerSecond)
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return .infinity }
        let mid = sorted.count / 2
        return sorted.count.isMultiple(of: 2) ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
    }
}
