import CoreGraphics
import Foundation
import ImageIO
import Synchronization
import UniformTypeIdentifiers
import AuraCore
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

/// The pipeline the engine drives; the seam that lets tests run without model weights.
protocol DiffusionPipelineRunning: AnyObject {
    func load() throws
    /// `shouldContinue(step, total, latentsAreFinite)` runs after each 1-based step; returning `false` stops
    /// with `[]`.
    func generate(prompt: String, negativePrompt: String, steps: Int, guidanceScale: Float, seed: UInt32,
                  imageCount: Int, shouldContinue: (Int, Int, Bool) -> Bool) throws -> [CGImage?]
    func unload()
}

/// Text-to-image on this device with Apple's Core ML Stable Diffusion (`ml-stable-diffusion`). Nothing leaves
/// the device: the model is downloaded once (``CoreMLDiffusionModel/download(progress:)``), then prompts and
/// images stay local.
///
/// One generation runs at a time; a second concurrent call throws ``OnDeviceImageGenError/busy``. Cancelling the
/// calling task stops between denoising steps; model preparation and the final decode cannot be interrupted.
/// The Simulator cannot run it (``OnDeviceImageGenError/simulatorUnsupported``).
public final class CoreMLDiffusionEngine: Sendable {
    typealias RunnerFactory = @Sendable (_ resources: URL, _ computeUnits: DiffusionComputeUnits,
                                         _ reduceMemory: Bool) throws -> any DiffusionPipelineRunning

    private let model: CoreMLDiffusionModel
    private let computeUnits: DiffusionComputeUnits
    private let memoryMode: DiffusionMemoryMode
    private let resourcesOverride: URL?
    private let makeRunner: RunnerFactory?
    private let availableMemory: @Sendable () -> Int?
    private let runsInSimulator: Bool
    private let unknownMemoryMeansExhausted: Bool
    // `generateImages` blocks for seconds to minutes; it must not occupy the cooperative pool.
    private let queue = DispatchQueue(label: "dev.auralocal.imagegen.coreml", qos: .userInitiated)
    private let slot = RunnerSlot()
    private let isBusy = Mutex(false)

    public convenience init(model: CoreMLDiffusionModel = .stableDiffusion21Base,
                            computeUnits: DiffusionComputeUnits = .neuralEngine,
                            memoryMode: DiffusionMemoryMode = .automatic) {
        self.init(model: model, computeUnits: computeUnits, memoryMode: memoryMode, resourcesDirectory: nil,
                  makeRunner: Self.platformRunnerFactory,
                  availableMemory: { HardwareProfile.availableMemoryBytes() })
    }

    init(model: CoreMLDiffusionModel, computeUnits: DiffusionComputeUnits, memoryMode: DiffusionMemoryMode,
         resourcesDirectory: URL?, makeRunner: RunnerFactory?, availableMemory: @escaping @Sendable () -> Int?,
         runsInSimulator: Bool = CoreMLDiffusionEngine.isSimulatorBuild,
         unknownMemoryMeansExhausted: Bool = CoreMLDiffusionEngine.unknownMemoryIsJetsamLimit) {
        self.model = model
        self.computeUnits = computeUnits
        self.memoryMode = memoryMode
        self.resourcesOverride = resourcesDirectory
        self.makeRunner = makeRunner
        self.availableMemory = availableMemory
        self.runsInSimulator = runsInSimulator
        self.unknownMemoryMeansExhausted = unknownMemoryMeansExhausted
    }

    /// Why this engine cannot run here whatever is downloaded: ``OnDeviceImageGenError/unsupportedPlatform``
    /// (visionOS) or ``OnDeviceImageGenError/simulatorUnsupported``; `nil` on an iPhone, iPad or Mac.
    public var unavailability: OnDeviceImageGenError? {
        if makeRunner == nil { return .unsupportedPlatform }
        if runsInSimulator { return .simulatorUnsupported }
        return nil
    }

    /// `false` where ``unavailability`` is set or when the model is not downloaded.
    public var isReady: Bool {
        unavailability == nil && model.isDownloaded(at: resources)
    }

    /// Generates `request.imageCount` images and writes each as `image-<index>.png` into `outputDirectory`
    /// (default: a new temporary folder). `onEvent` is called on a background queue.
    public func generate(_ request: DiffusionRequest, outputDirectory: URL? = nil,
                         onEvent: @escaping @Sendable (DiffusionEvent) -> Void = { _ in }) async throws
        -> DiffusionOutput {
        try request.validate()
        if let unavailability { throw unavailability }
        guard let makeRunner else { throw OnDeviceImageGenError.unsupportedPlatform }
        let resources = resources
        // Re-checked every call: iOS can purge Caches between runs.
        guard model.isDownloaded(at: resources) else {
            throw OnDeviceImageGenError.modelNotDownloaded(sizeBytes: model.approximateDownloadBytes)
        }
        let acquired = isBusy.withLock { busy -> Bool in
            guard !busy else { return false }
            busy = true
            return true
        }
        guard acquired else { throw OnDeviceImageGenError.busy }
        defer { isBusy.withLock { $0 = false } }

        let seed = request.seed ?? UInt32.random(in: 0...UInt32.max)
        let directory = outputDirectory ?? FileManager.default.temporaryDirectory
            .appendingPathComponent("auraimagegen-\(UUID().uuidString)", isDirectory: true)
        let job = Job(request: request, seed: seed, resources: resources, directory: directory,
                      required: Self.requiredMemoryBytes(for: model))
        let cancelled = CancellationFlag()
        let clock = ContinuousClock()
        let start = clock.now
        let images = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<[ImageGenResult], any Error>) in
                queue.async { [self] in
                    continuation.resume(with: Result {
                        try run(job, makeRunner: makeRunner, cancelled: cancelled, onEvent: onEvent)
                    })
                }
            }
        } onCancel: {
            cancelled.set()
        }
        return DiffusionOutput(images: images, seed: seed, steps: request.steps, elapsed: clock.now - start)
    }

    /// Frees the pipeline. Waits for a running generation; the next call reloads (seconds once compiled).
    public func unload() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            queue.async { [slot] in
                slot.runner?.unload()
                slot.runner = nil
                continuation.resume()
            }
        }
    }

    // MARK: Queue work

    private struct Job: Sendable {
        let request: DiffusionRequest
        let seed: UInt32
        let resources: URL
        let directory: URL
        let required: Int
    }

    /// Runs on `queue` only.
    private func run(_ job: Job, makeRunner: RunnerFactory, cancelled: CancellationFlag,
                     onEvent: @Sendable (DiffusionEvent) -> Void) throws -> [ImageGenResult] {
        if cancelled.isSet { throw CancellationError() }
        let runner = try loadedRunner(job, makeRunner: makeRunner, onEvent: onEvent)
        let request = job.request
        var nonFiniteStep: Int?
        let images: [CGImage?]
        do {
            images = try runner.generate(
                prompt: request.prompt, negativePrompt: request.negativePrompt, steps: request.steps,
                guidanceScale: request.guidanceScale, seed: job.seed, imageCount: request.imageCount
            ) { step, total, latentsAreFinite in
                guard !cancelled.isSet else { return false }
                // NaN latents only ever decode to a black image; stop instead of finishing the run.
                guard latentsAreFinite else {
                    nonFiniteStep = step
                    return false
                }
                onEvent(.step(step, of: total))
                if step == total { onEvent(.decoding) }
                return true
            }
        } catch let error as OnDeviceImageGenError {
            releaseStages(of: runner)
            throw error
        } catch {
            releaseStages(of: runner)
            throw OnDeviceImageGenError.generationFailed(error.localizedDescription)
        }
        if images.isEmpty {
            releaseStages(of: runner)
            if let nonFiniteStep {
                let hardware = computeUnits.hardwareName
                throw OnDeviceImageGenError.generationFailed(
                    "The model produced invalid values (NaN) at step \(nonFiniteStep) on \(hardware).")
            }
            if cancelled.isSet { throw CancellationError() }
            throw OnDeviceImageGenError.generationFailed("The pipeline returned no images.")
        }
        try FileManager.default.createDirectory(at: job.directory, withIntermediateDirectories: true)
        let results = try images.enumerated().compactMap { index, image in
            try image.map { try Self.makeResult(from: $0, index: index, in: job.directory) }
        }
        if results.isEmpty { throw OnDeviceImageGenError.filteredBySafetyChecker }
        return results
    }

    /// The loaded runner, creating and loading it under the memory gate on first use or after ``unload()``.
    private func loadedRunner(_ job: Job, makeRunner: RunnerFactory,
                              onEvent: @Sendable (DiffusionEvent) -> Void) throws -> any DiffusionPipelineRunning {
        if let runner = slot.runner { return runner }
        let available = availableMemory()
        try Self.checkMemory(availableBytes: available, requiredBytes: job.required,
                             unknownMeansExhausted: unknownMemoryMeansExhausted)
        let mode = Self.resolvedMemoryMode(memoryMode, availableBytes: available, requiredBytes: job.required)
        onEvent(.preparingModel)
        do {
            let runner = try makeRunner(job.resources, computeUnits, mode == .lowMemory)
            try runner.load()
            slot.runner = runner
            slot.reducesMemory = mode == .lowMemory
            return runner
        } catch let error as OnDeviceImageGenError {
            throw error
        } catch {
            throw OnDeviceImageGenError.modelLoadFailed(error.localizedDescription)
        }
    }

    /// A stopped or failed low-memory run can leave a stage (often the Unet) loaded; the pipeline only frees it
    /// after a completed run.
    private func releaseStages(of runner: any DiffusionPipelineRunning) {
        if slot.reducesMemory { runner.unload() }
    }

    // MARK: Policy (pure)

    private var resources: URL {
        resourcesOverride ?? model.resourcesDirectory
    }

    /// Estimated peak: weights × ``ModelKind/diffusion`` multiplier.
    static func requiredMemoryBytes(for model: CoreMLDiffusionModel) -> Int {
        let gigabytes = HardwareAnalyzer.peakMemoryGB(forWeightsBytes: Int(model.approximateDownloadBytes),
                                                      kind: .diffusion)
        return Int(gigabytes * 1_073_741_824)
    }

    /// Throws when the estimated peak does not fit. `availableBytes == nil` throws too when
    /// `unknownMeansExhausted`: iOS reports unknown when the app is already at its jetsam limit.
    static func checkMemory(availableBytes: Int?, requiredBytes: Int, unknownMeansExhausted: Bool) throws {
        switch availableBytes {
        case .some(let available) where available < requiredBytes:
            throw OnDeviceImageGenError.insufficientMemory(requiredBytes: Int64(requiredBytes),
                                                           availableBytes: Int64(available))
        case .none where unknownMeansExhausted:
            throw OnDeviceImageGenError.insufficientMemory(requiredBytes: Int64(requiredBytes), availableBytes: 0)
        default:
            return
        }
    }

    /// `.automatic` keeps the models resident only with twice the estimated peak free; unknown memory is treated
    /// as tight.
    static func resolvedMemoryMode(_ mode: DiffusionMemoryMode, availableBytes: Int?,
                                   requiredBytes: Int) -> DiffusionMemoryMode {
        switch mode {
        case .resident, .lowMemory:
            return mode
        case .automatic:
            guard let availableBytes else { return .lowMemory }
            return availableBytes >= 2 * requiredBytes ? .resident : .lowMemory
        }
    }

    /// The Simulator has no Neural Engine, and on the CPU this model's latents are NaN from the first step.
    static var isSimulatorBuild: Bool {
        #if targetEnvironment(simulator)
        true
        #else
        false
        #endif
    }

    /// On a Mac (Catalyst included) `nil` is not a jetsam signal; on iPhone and iPad it means the app is at its limit.
    static var unknownMemoryIsJetsamLimit: Bool {
        #if os(macOS) || targetEnvironment(macCatalyst)
        false
        #else
        true
        #endif
    }

    private static var platformRunnerFactory: RunnerFactory? {
        #if canImport(StableDiffusion)
        return { resources, computeUnits, reduceMemory in
            try StableDiffusionRunner(resources: resources, computeUnits: computeUnits, reduceMemory: reduceMemory)
        }
        #else
        return nil
        #endif
    }

    // MARK: Output

    static func makeResult(from image: CGImage, index: Int, in directory: URL) throws -> ImageGenResult {
        let fileURL = directory.appendingPathComponent("image-\(index).png")
        try writePNG(image, to: fileURL)
        return ImageGenResult(image: makePlatformImage(image), fileURL: fileURL)
    }

    static func writePNG(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString,
                                                                1, nil) else {
            throw OnDeviceImageGenError.generationFailed("Could not create \(url.lastPathComponent).")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw OnDeviceImageGenError.generationFailed("Could not write \(url.lastPathComponent).")
        }
    }

    static func makePlatformImage(_ image: CGImage) -> PlatformImage {
        #if canImport(UIKit)
        UIImage(cgImage: image)
        #else
        NSImage(cgImage: image, size: .zero)
        #endif
    }
}

// MARK: - Confinement helpers

/// The loaded pipeline. Unchecked because it is confined to the engine's serial `queue`.
private final class RunnerSlot: @unchecked Sendable {
    var runner: (any DiffusionPipelineRunning)?
    var reducesMemory = false
}

/// Set by the task's cancellation handler, read by the pipeline's step callback on the queue.
private final class CancellationFlag: Sendable {
    private let state = Mutex(false)

    func set() {
        state.withLock { $0 = true }
    }

    var isSet: Bool {
        state.withLock { $0 }
    }
}
