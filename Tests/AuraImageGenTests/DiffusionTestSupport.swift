import CoreGraphics
import Foundation
import Synchronization
@testable import AuraImageGen

/// Stands in for the Core ML pipeline: returns solid-colour images and records what the engine asked for.
final class FakeRunner: DiffusionPipelineRunning, @unchecked Sendable {
    struct Call: Sendable, Equatable {
        let prompt: String
        let steps: Int
        let seed: UInt32
        let imageCount: Int
    }

    private struct State {
        var calls: [Call] = []
        var loads = 0
        var unloads = 0
        var stoppedAtStep: Int?
        var entered = false
    }

    /// `false` entries stand for images the safety checker removed; `nil` returns `imageCount` solid images.
    var outputs: [Bool]?
    var stepDelay: Duration = .zero
    var loadError: (any Error)?
    var generateError: (any Error)?
    /// When set, `generate` waits on it before the first step.
    var gate: DispatchSemaphore?

    /// Latents turn NaN from this 1-based step on.
    var nonFiniteFromStep: Int?

    private let record = Mutex(State())

    var calls: [Call] { record.withLock { $0.calls } }
    var loads: Int { record.withLock { $0.loads } }
    var unloads: Int { record.withLock { $0.unloads } }
    var stoppedAtStep: Int? { record.withLock { $0.stoppedAtStep } }
    var entered: Bool { record.withLock { $0.entered } }

    func load() throws {
        record.withLock { $0.loads += 1 }
        if let loadError { throw loadError }
    }

    func generate(prompt: String, negativePrompt: String, steps: Int, guidanceScale: Float, seed: UInt32,
                  imageCount: Int, shouldContinue: (Int, Int, Bool) -> Bool) throws -> [CGImage?] {
        record.withLock {
            $0.calls.append(Call(prompt: prompt, steps: steps, seed: seed, imageCount: imageCount))
            $0.entered = true
        }
        gate?.wait()
        if let generateError { throw generateError }
        for step in 1...steps {
            if stepDelay > .zero { Thread.sleep(forTimeInterval: Self.seconds(stepDelay)) }
            let finite = nonFiniteFromStep.map { step < $0 } ?? true
            if !shouldContinue(step, steps, finite) {
                record.withLock { $0.stoppedAtStep = step }
                return []
            }
        }
        let pattern = outputs ?? Array(repeating: true, count: imageCount)
        return pattern.map { $0 ? Self.solidImage() : nil }
    }

    func unload() {
        record.withLock { $0.unloads += 1 }
    }

    private static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    static func solidImage(side: Int = 8) -> CGImage? {
        guard let context = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        context.setFillColor(red: 0.9, green: 0.3, blue: 0.1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        return context.makeImage()
    }
}

/// Events in the order the engine emitted them.
final class EventLog: Sendable {
    private let events = Mutex<[DiffusionEvent]>([])

    func append(_ event: DiffusionEvent) {
        events.withLock { $0.append(event) }
    }

    var all: [DiffusionEvent] { events.withLock { $0 } }
}

/// What the engine passed to the runner factory.
final class FactoryLog: Sendable {
    private let reduceMemoryFlags = Mutex<[Bool]>([])

    func record(_ reduceMemory: Bool) {
        reduceMemoryFlags.withLock { $0.append(reduceMemory) }
    }

    var flags: [Bool] { reduceMemoryFlags.withLock { $0 } }
}

enum DiffusionFixtures {
    /// Mirrors the marker `ModelDownloads.downloadFolder` writes.
    static let completeMarker = ".complete"

    static func temporaryDirectory(_ label: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(label)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// A resources folder that passes `isDownloaded`, optionally without one component or the marker, or with a
    /// marker for `markerRevision` instead of the model's.
    static func downloadedResources(model: CoreMLDiffusionModel = .stableDiffusion21Base, omitting: String? = nil,
                                    marker: Bool = true, markerRevision: String? = nil) throws -> URL {
        let directory = try temporaryDirectory("CoreMLDiffusionResources")
        for component in model.components where component != omitting {
            try makeComponent(component, in: directory)
        }
        if marker {
            let revision = markerRevision ?? model.revision
            try Data(revision.utf8).write(to: directory.appendingPathComponent(completeMarker))
        }
        return directory
    }

    private static func makeComponent(_ component: String, in directory: URL) throws {
        let url = directory.appending(path: component)
        if component.hasSuffix("/") {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        } else {
            try Data("{}".utf8).write(to: url)
        }
    }

    /// `availableMemory` defaults to unknown with macOS semantics (unknown proceeds, in low-memory mode).
    static func engine(resources: URL, runner: FakeRunner?, memoryMode: DiffusionMemoryMode = .automatic,
                       availableMemory: @escaping @Sendable () -> Int? = { nil },
                       runsInSimulator: Bool = false, unknownMemoryMeansExhausted: Bool = false,
                       factoryLog: FactoryLog = FactoryLog()) -> CoreMLDiffusionEngine {
        var factory: CoreMLDiffusionEngine.RunnerFactory?
        if let runner {
            factory = { @Sendable _, _, reduceMemory in
                factoryLog.record(reduceMemory)
                return runner
            }
        }
        return CoreMLDiffusionEngine(model: .stableDiffusion21Base, computeUnits: .neuralEngine,
                                     memoryMode: memoryMode, resourcesDirectory: resources, makeRunner: factory,
                                     availableMemory: availableMemory, runsInSimulator: runsInSimulator,
                                     unknownMemoryMeansExhausted: unknownMemoryMeansExhausted)
    }
}
