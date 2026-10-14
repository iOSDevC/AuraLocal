import Foundation

// MARK: - Request

/// A text-to-image request for ``CoreMLDiffusionEngine``. Out-of-range values throw
/// ``OnDeviceImageGenError/invalidRequest(field:reason:)``; nothing is clamped.
public struct DiffusionRequest: Sendable, Equatable {
    public var prompt: String
    public var negativePrompt: String
    /// Denoising steps, 1...50.
    public var steps: Int
    /// Classifier-free guidance, 1...20.
    public var guidanceScale: Float
    /// `nil` picks a random seed; ``DiffusionOutput/seed`` returns the one used.
    public var seed: UInt32?
    /// Images per run, 1...4. They share one seed.
    public var imageCount: Int

    public init(prompt: String, negativePrompt: String = "", steps: Int = 20, guidanceScale: Float = 7.5,
                seed: UInt32? = nil, imageCount: Int = 1) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.seed = seed
        self.imageCount = imageCount
    }

    static let stepRange = 1...50
    static let guidanceRange: ClosedRange<Float> = 1...20
    static let imageCountRange = 1...4

    func validate() throws {
        if prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw OnDeviceImageGenError.invalidRequest(field: "prompt", reason: "describe the image")
        }
        if !Self.stepRange.contains(steps) {
            throw Self.outOfRange("steps", steps, Self.stepRange)
        }
        if !Self.guidanceRange.contains(guidanceScale) {
            throw Self.outOfRange("guidanceScale", guidanceScale, Self.guidanceRange)
        }
        if !Self.imageCountRange.contains(imageCount) {
            throw Self.outOfRange("imageCount", imageCount, Self.imageCountRange)
        }
    }

    private static func outOfRange<Value: Comparable>(_ field: String, _ value: Value,
                                                      _ range: ClosedRange<Value>) -> OnDeviceImageGenError {
        .invalidRequest(field: field, reason: "must be \(range.lowerBound)–\(range.upperBound), got \(value)")
    }
}

// MARK: - Options

/// Core ML compute units for the pipeline. There is no CPU-only option: on the CPU this model's latents are NaN
/// from the first step, so every image comes out black.
public enum DiffusionComputeUnits: String, Sendable, CaseIterable {
    /// `.cpuAndNeuralEngine`
    case neuralEngine
    /// `.cpuAndGPU`
    case gpu

    var hardwareName: String {
        switch self {
        case .neuralEngine: "the Neural Engine"
        case .gpu: "the GPU"
        }
    }
}

/// Whether the pipeline stays in memory between runs.
public enum DiffusionMemoryMode: Sendable {
    /// `resident` when free memory is at least twice the estimated peak, otherwise `lowMemory`.
    case automatic
    /// All models stay loaded across calls: fastest repeat runs, highest footprint.
    case resident
    /// Models load per stage and unload after it: slower, smaller peak.
    case lowMemory
}

// MARK: - Progress and output

public enum DiffusionEvent: Sendable, Equatable {
    /// Loading the model. The first load on a device compiles it for the Neural Engine and can take minutes.
    case preparingModel
    /// Denoising step `n` of `total` finished (1-based).
    case step(Int, of: Int)
    /// Turning the latents into images.
    case decoding
}

public struct DiffusionOutput: Sendable {
    /// One per image that passed the safety checker, each with its PNG on disk.
    public let images: [ImageGenResult]
    public let seed: UInt32
    public let steps: Int
    public let elapsed: Duration
}
