import Foundation
import SoundAnalysis
import Synchronization

/// On-device sound classification via Apple's **SoundAnalysis** built-in classifier
/// (~300 everyday sounds: speech, music, dogs, sirens, …) over an audio file — no
/// model download. The SLM gets a ranked list of what can be heard in a recording.
/// Resilient: an unreadable file throws a typed error; silence is just low scores.
public struct SoundClassificationTool: SystemTool {
    public let id = "system.audio.sounds"
    public let displayName = "Sound classification (SoundAnalysis)"
    public let summary = "Identify everyday sounds (speech, music, animals, alarms, …) in an audio file with Apple's built-in SoundAnalysis classifier — no model download, works offline."
    public let category = SystemToolCategory.audio

    public init() {}

    public enum ToolError: LocalizedError {
        case unreadableAudio(String)
        case classifierUnavailable(String)
        case analysisFailed(String)

        public var errorDescription: String? {
            switch self {
            case .unreadableAudio(let reason): "The audio file could not be opened: \(reason)"
            case .classifierUnavailable(let reason): "The built-in sound classifier could not be loaded: \(reason)"
            case .analysisFailed(let reason): "Sound analysis failed: \(reason)"
            }
        }
    }

    /// How the classifier's per-window scores (≈3 s windows, 50 % overlap) become one
    /// score per sound for the whole file.
    public enum Aggregation: Sendable {
        /// Highest confidence in any window — "does this sound occur anywhere?"
        case peak
        /// Average confidence over all windows — "how much of the file is this sound?"
        case mean
    }

    public struct Classification: Sendable, Equatable {
        /// Classifier label, e.g. `speech`, `dog_bark`, `music`.
        public let identifier: String
        public let confidence: Double   // 0…1
    }

    public func availability() async -> SystemToolAvailability {
        do {
            _ = try Self.makeRequest()
            return .available
        } catch {
            return .unavailable(reason: error.localizedDescription)
        }
    }

    /// Every label the built-in classifier can produce, sorted.
    public func knownSounds() throws -> [String] {
        try Self.makeRequest().knownClassifications.sorted()
    }

    /// Classify the sounds in an audio file (any format AVFoundation reads), best first.
    /// Scores are aggregated across analysis windows per `aggregation`.
    public func classify(
        audioFileAt url: URL,
        maxResults: Int = 5,
        aggregation: Aggregation = .peak
    ) async throws -> [Classification] {
        guard maxResults > 0 else { return [] }
        let analyzer: SNAudioFileAnalyzer
        do {
            analyzer = try SNAudioFileAnalyzer(url: url)
        } catch {
            throw ToolError.unreadableAudio(error.localizedDescription)
        }
        let request = try Self.makeRequest()
        let collector = WindowCollector()
        try analyzer.add(request, withObserver: collector)

        let canceller = AnalysisCanceller(analyzer: analyzer)
        let reachedEnd: Bool = await withTaskCancellationHandler {
            await analyzer.analyze()
        } onCancel: {
            canceller.cancel()
        }
        try Task.checkCancellation()
        if let failure = collector.failure {
            throw ToolError.analysisFailed(failure)
        }
        guard reachedEnd else {
            throw ToolError.analysisFailed("The analysis stopped before the end of the file.")
        }
        return collector.ranked(by: aggregation, limit: maxResults)
    }

    private static func makeRequest() throws -> SNClassifySoundRequest {
        do {
            return try SNClassifySoundRequest(classifierIdentifier: .version1)
        } catch {
            throw ToolError.classifierUnavailable(error.localizedDescription)
        }
    }

    fileprivate static func makeClassification(_ entry: (key: String, value: Double)) -> Classification {
        Classification(identifier: entry.key, confidence: min(max(entry.value, 0), 1))
    }
}

/// Receives SoundAnalysis callbacks, which may arrive on any thread; all state sits
/// behind a `Mutex`, so the observer is genuinely `Sendable`.
private final class WindowCollector: NSObject, SNResultsObserving, Sendable {
    private struct Tally {
        var peaks: [String: Double] = [:]
        var sums: [String: Double] = [:]
        var windowCount = 0
        var failure: String?
    }

    private let scores = Mutex(Tally())

    var failure: String? { scores.withLock { $0.failure } }

    func request(_ request: any SNRequest, didProduce result: any SNResult) {
        guard let window = result as? SNClassificationResult else { return }
        let classifications = window.classifications
        scores.withLock { state in
            state.windowCount += 1
            for item in classifications {
                state.peaks[item.identifier] = max(state.peaks[item.identifier] ?? 0, item.confidence)
                state.sums[item.identifier, default: 0] += item.confidence
            }
        }
    }

    func request(_ request: any SNRequest, didFailWithError error: any Error) {
        let reason = error.localizedDescription
        scores.withLock { $0.failure = reason }
    }

    func ranked(
        by aggregation: SoundClassificationTool.Aggregation,
        limit: Int
    ) -> [SoundClassificationTool.Classification] {
        let snapshot = scores.withLock { $0 }
        guard snapshot.windowCount > 0 else { return [] }
        let windows = Double(snapshot.windowCount)
        let combined = switch aggregation {
        case .peak: snapshot.peaks
        case .mean: snapshot.sums.mapValues { $0 / windows }
        }
        // Ties broken by name so the order is deterministic.
        return combined
            .sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
            .prefix(limit)
            .map(SoundClassificationTool.makeClassification)
    }
}

/// `cancelAnalysis()` exists to be called from another thread while `analyze()` runs,
/// which is the only use made of the analyzer here.
private final class AnalysisCanceller: @unchecked Sendable {
    private let analyzer: SNAudioFileAnalyzer

    init(analyzer: SNAudioFileAnalyzer) {
        self.analyzer = analyzer
    }

    func cancel() {
        analyzer.cancelAnalysis()
    }
}
