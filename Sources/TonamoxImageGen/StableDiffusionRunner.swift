#if canImport(StableDiffusion)
import CoreGraphics
import CoreML
import Foundation
@preconcurrency import StableDiffusion

/// The real Core ML pipeline behind ``CoreMLDiffusionEngine``. Not thread-safe: the engine only touches it on
/// its serial queue.
final class StableDiffusionRunner: DiffusionPipelineRunning {
    private let pipeline: StableDiffusionPipeline

    init(resources: URL, computeUnits: DiffusionComputeUnits, reduceMemory: Bool) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = Self.mlComputeUnits(computeUnits)
        pipeline = try StableDiffusionPipeline(
            resourcesAt: resources, controlNet: [], configuration: configuration,
            disableSafety: false, reduceMemory: reduceMemory)
    }

    func load() throws {
        try pipeline.loadResources()
    }

    func generate(prompt: String, negativePrompt: String, steps: Int, guidanceScale: Float, seed: UInt32,
                  imageCount: Int, shouldContinue: (Int, Int, Bool) -> Bool) throws -> [CGImage?] {
        var configuration = StableDiffusionPipeline.Configuration(prompt: prompt)
        configuration.negativePrompt = negativePrompt
        configuration.stepCount = steps
        configuration.guidanceScale = guidanceScale
        configuration.seed = seed
        configuration.imageCount = imageCount
        configuration.schedulerType = .dpmSolverMultistepScheduler
        return try pipeline.generateImages(configuration: configuration) { progress in
            shouldContinue(progress.step + 1, progress.stepCount, Self.areFinite(progress.currentLatentSamples))
        }
    }

    func unload() {
        pipeline.unloadResources()
    }

    static func mlComputeUnits(_ units: DiffusionComputeUnits) -> MLComputeUnits {
        switch units {
        case .neuralEngine: .cpuAndNeuralEngine
        case .gpu: .cpuAndGPU
        }
    }

    static func areFinite(_ latents: [MLShapedArray<Float32>]) -> Bool {
        latents.allSatisfy { $0.scalars.allSatisfy(\.isFinite) }
    }
}
#endif
