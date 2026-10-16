import CoreGraphics
import Darwin
import Foundation
import ImageIO
import XCTest
import AuraCore
@testable import AuraImageGen

/// Against huggingface.co on this Mac. End to end: inspect a second model, adopt it, download ~0.9 GB, generate one
/// image on the Neural Engine, then delete the folder. Skipped unless `AURA_LIVE_IMAGEGEN=1`.
/// `AURA_LIVE_IMAGEGEN_OUTPUT` (a directory) keeps a copy of the image.
final class LiveCoreMLDiffusionImportTests: XCTestCase {
    private static let repoID = "apple/coreml-stable-diffusion-v1-5-palettized"

    private struct LumaStats {
        let width: Int
        let height: Int
        let mean: Double
        let deviation: Double
    }

    func testInspectAdoptDownloadGenerateDelete() async throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(environment["AURA_LIVE_IMAGEGEN"] == "1", "set AURA_LIVE_IMAGEGEN=1 to run")
        #if !os(macOS)
        throw XCTSkip("macOS only")
        #else
        let suiteName = "LiveCoreMLDiffusionImportTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let builtIn = CoreMLDiffusionModel.stableDiffusion21Base
        let builtInWasDownloaded = builtIn.isDownloaded

        let report = try await CoreMLDiffusionInspector().inspect(Self.repoID, on: .thisDevice())
        print("[live] revision \(report.revision)")
        for variant in report.variants {
            print("[live] \(variant.folder.path): \(variant.verdict.label), \(variant.preferredComputeUnits), "
                  + "\(variant.folder.downloadBytes ?? -1) bytes; "
                  + variant.findings.map { "\($0.rule) \($0.level.label): \($0.title)" }.joined(separator: " | "))
        }
        XCTAssertEqual(report.revision, "04a6a0bdd66fb8da470c14e56d762343ef579d88")
        XCTAssertEqual(report.variants.map(\.folder.path), ["original/compiled", "split_einsum_v2/compiled"])
        let split = try XCTUnwrap(report.variants.last)
        let original = try XCTUnwrap(report.variants.first)
        // The card's license is `other`, so both carry the R18 caveat; original also runs on the GPU (R10).
        XCTAssertEqual(split.verdict, .runnableWithCaveats)
        XCTAssertEqual(split.preferredComputeUnits, .neuralEngine)
        XCTAssertEqual(original.verdict, .runnableWithCaveats)
        XCTAssertEqual(original.preferredComputeUnits, .gpu)
        let recommended = try XCTUnwrap(report.recommended)
        XCTAssertEqual(recommended.folder.path, "split_einsum_v2/compiled")

        let model = try CoreMLDiffusionModel(variant: recommended)
        let store = CoreMLDiffusionModelStore(defaults: defaults)
        store.remember(model)
        XCTAssertEqual(store.model(id: model.id), model)
        print("[live] downloading \(model.approximateDownloadBytes) bytes to \(model.resourcesDirectory.path)")

        let clock = ContinuousClock()
        let downloadStart = clock.now
        try await model.download { _, _ in }
        print("[live] download took \(clock.now - downloadStart)")
        XCTAssertTrue(model.isDownloaded)
        let siblingFolder = ModelDownloads.localFolder(repoID: Self.repoID, folder: "original/compiled")
        XCTAssertFalse(FileManager.default.fileExists(atPath: siblingFolder.path), "original/compiled not fetched")

        let engine = CoreMLDiffusionEngine(model: model, computeUnits: model.preferredComputeUnits,
                                           memoryMode: .automatic)
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("LiveCoreMLDiffusion-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: output) }
        let request = DiffusionRequest(prompt: "a lighthouse on a cliff at sunset", steps: 20, seed: 42)
        let result = try await engine.generate(request, outputDirectory: output)
        print("[live] generated in \(result.elapsed) on \(Self.machineModel()) (first run includes Core ML load)")
        let again = DiffusionRequest(prompt: "a lighthouse on a cliff at sunset", steps: 20, seed: 43)
        let second = try await engine.generate(again, outputDirectory: output.appendingPathComponent("second"))
        print("[live] second run in the same process: \(second.elapsed)")
        await engine.unload()

        let file = try XCTUnwrap(result.images.first?.fileURL)
        let stats = try XCTUnwrap(Self.lumaStats(of: file))
        print("[live] \(stats.width)×\(stats.height), mean luma \(stats.mean), stddev \(stats.deviation)")
        XCTAssertEqual(stats.width, 512)
        XCTAssertEqual(stats.height, 512)
        XCTAssertGreaterThan(stats.mean, 10, "not black")
        XCTAssertGreaterThan(stats.deviation, 5, "not flat")
        if let keep = environment["AURA_LIVE_IMAGEGEN_OUTPUT"] {
            let destination = URL(fileURLWithPath: keep).appendingPathComponent("sd15-split-einsum-v2-seed42.png")
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.copyItem(at: file, to: destination)
            print("[live] image saved to \(destination.path)")
        }

        try model.delete()
        store.forget(model)
        let repoFolder = model.resourcesDirectory.deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        XCTAssertFalse(FileManager.default.fileExists(atPath: repoFolder.path), repoFolder.path)
        XCTAssertEqual(builtIn.isDownloaded, builtInWasDownloaded, "the built-in folder is untouched")
        #endif
    }

    /// Network only (a few hundred KB): repos under either Core ML tag or none, a pasted URL, a mistyped id and
    /// the built-in model's own folder.
    func testSearchAndInspectAgainstHuggingFace() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["AURA_LIVE_IMAGEGEN"] == "1",
                          "set AURA_LIVE_IMAGEGEN=1 to run")
        let inspector = CoreMLDiffusionInspector()
        let turbo = try await inspector.search("turbo").map(\.id)
        let tiny = try await inspector.search("tiny").map(\.id)
        let dreamshaper = try await inspector.search("dreamshaper").map(\.id)
        let pasted = try await inspector.search("https://huggingface.co/CommuniDev/sd-turbo-coreml").map(\.id)
        print("[live] turbo \(turbo)\n[live] tiny \(tiny)\n[live] dreamshaper \(dreamshaper.count) hits")
        XCTAssertTrue(turbo.contains("CommuniDev/sd-turbo-coreml"), "tagged core-ml only")
        XCTAssertTrue(tiny.contains("kessenma/coreml-bk-sdm-tiny"), "tagged core-ml only")
        XCTAssertTrue(dreamshaper.contains("c2p-cmd/dreamshaper-7-coreml"), "untagged")
        XCTAssertTrue(pasted.contains("CommuniDev/sd-turbo-coreml"))

        let missing = "apple/does-not-exist-xyz-123"
        do {
            _ = try await inspector.inspect(missing)
            XCTFail("expected a failure")
        } catch let error as CoreMLDiffusionModelError {
            XCTAssertEqual(error, .repositoryUnavailable(repoID: missing, status: 401))
            print("[live] \(missing): \(error.recoverySuggestion ?? "")")
        }

        let builtIn = CoreMLDiffusionModel.stableDiffusion21Base
        let report = try await inspector.inspect(builtIn.repoID)
        let variant = try XCTUnwrap(report.variants.first { $0.folder.path == builtIn.folder })
        print("[live] \(builtIn.repoID) at \(report.revision): \(variant.verdict.label)")
        XCTAssertEqual(try CoreMLDiffusionModel(variant: variant), builtIn)
    }

    private static func machineModel() -> String {
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var bytes = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.model", &bytes, &size, nil, 0)
        return String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// Rec. 601 luma over every pixel of the PNG.
    private static func lumaStats(of file: URL) -> LumaStats? {
        guard let source = CGImageSourceCreateWithURL(file as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        let width = image.width
        let height = image.height
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { return nil }
        let luma = stride(from: 0, to: pixels.count, by: 4).map { index in
            0.299 * Double(pixels[index]) + 0.587 * Double(pixels[index + 1]) + 0.114 * Double(pixels[index + 2])
        }
        let mean = luma.reduce(0, +) / Double(luma.count)
        let variance = luma.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(luma.count)
        return LumaStats(width: width, height: height, mean: mean, deviation: variance.squareRoot())
    }
}
