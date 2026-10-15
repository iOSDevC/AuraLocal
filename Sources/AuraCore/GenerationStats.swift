import Foundation

// MARK: - GenerationStats

/// Token counts and timings of one generation, labelled with where they come from.
///
/// Read ``AuraLocal/lastGenerationStats`` after `chat`, `analyze` or a stream finishes. Check ``source``
/// before comparing runs: only ``Provenance/runtime`` counts are exact for both prompt and output.
public struct GenerationStats: Sendable, Hashable {

    /// Where the counts and durations come from.
    public enum Provenance: String, Sendable, Hashable, CaseIterable {
        /// The inference runtime's own counters (MLX `GenerateResult`): exact prompt and output token counts and
        /// the runtime's prefill duration. Decode is timed by AuraLocal from the first token callback to the last.
        case runtime
        /// AuraLocal's monotonic clock around the token stream. Output tokens are counted one per streamed piece,
        /// which llama.cpp emits once per sampled token; the prompt count is estimated (UTF-8 bytes ÷ 4 of the
        /// rendered prompt, chat template included), and ``GenerationStats/promptTime`` equals the time to first token.
        case measured
        /// Prompt and output counts estimated from text length (UTF-8 bytes ÷ 4). Used when streamed pieces are
        /// not tokens, e.g. llama.cpp with tools registered.
        case estimated

        /// Short label for display.
        public var label: String {
            switch self {
            case .runtime:   "runtime counters"
            case .measured:  "measured stream, prompt tokens estimated"
            case .estimated: "estimated from text length"
            }
        }
    }

    /// The backend that ran the generation.
    public let backend: BackendKind
    public let source: Provenance
    /// Tokens in the prompt. Exact only for ``Provenance/runtime``.
    public let promptTokens: Int
    /// Tokens generated, not counting the stop token.
    public let generatedTokens: Int
    /// Prefill: prompt processing up to the first generated token.
    public let promptTime: TimeInterval
    /// From when the backend starts the request (after waiting for an earlier one) to the first generated token.
    /// Includes chat-template rendering and tokenization.
    public let timeToFirstToken: TimeInterval
    /// From the first generated token to the last.
    public let decodeTime: TimeInterval

    public init(
        backend: BackendKind,
        source: Provenance,
        promptTokens: Int,
        generatedTokens: Int,
        promptTime: TimeInterval,
        timeToFirstToken: TimeInterval,
        decodeTime: TimeInterval
    ) {
        self.backend = backend
        self.source = source
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.promptTime = promptTime
        self.timeToFirstToken = timeToFirstToken
        self.decodeTime = decodeTime
    }

    /// Tokens after the first, divided by ``decodeTime``. The first token belongs to prefill, so counting it
    /// here would inflate short answers. `nil` below two tokens.
    public var decodeTokensPerSecond: Double? {
        guard generatedTokens > 1, decodeTime > 0 else { return nil }
        return Double(generatedTokens - 1) / decodeTime
    }

    /// ``promptTokens`` divided by ``promptTime``. Estimated unless ``source`` is ``Provenance/runtime``.
    public var prefillTokensPerSecond: Double? {
        guard promptTokens > 0, promptTime > 0 else { return nil }
        return Double(promptTokens) / promptTime
    }

    /// Rough token count for text the backend cannot tokenize for us.
    static func estimatedTokens(in text: String) -> Int {
        text.isEmpty ? 0 : max(1, (text.utf8.count + 3) / 4)
    }

    static func seconds(_ duration: Duration) -> TimeInterval {
        let parts = duration.components
        return TimeInterval(parts.seconds) + TimeInterval(parts.attoseconds) / 1e18
    }
}

extension GenerationStats.Provenance {
    /// Pinned LocalLLMClient 0.5.0 yields one piece per sampled token, except on the tool-calling path, which
    /// buffers pieces and drops tool-call text.
    static func llamaStream(toolsRegistered: Bool) -> Self {
        toolsRegistered ? .estimated : .measured
    }
}

// MARK: - StreamMeter

/// Times a token stream with a monotonic clock. Call ``recordPiece()`` once per streamed piece.
struct StreamMeter {
    private let start = ContinuousClock.now
    private var firstPiece: ContinuousClock.Instant?
    private var lastPiece: ContinuousClock.Instant?
    private(set) var pieces = 0

    mutating func recordPiece() {
        let now = ContinuousClock.now
        if firstPiece == nil { firstPiece = now }
        lastPiece = now
        pieces += 1
    }

    /// `.measured` counts the pieces; `.estimated` counts `output` by length. A stream with no pieces
    /// reports its whole duration as time to first token.
    func stats(backend: BackendKind, source: GenerationStats.Provenance, prompt: String, output: String) -> GenerationStats {
        let ttft = GenerationStats.seconds((firstPiece ?? ContinuousClock.now) - start)
        let decode = firstPiece.flatMap { first in lastPiece.map { GenerationStats.seconds($0 - first) } } ?? 0
        let generated = source == .estimated ? GenerationStats.estimatedTokens(in: output) : pieces
        return GenerationStats(
            backend: backend,
            source: source,
            promptTokens: GenerationStats.estimatedTokens(in: prompt),
            generatedTokens: generated,
            promptTime: ttft,
            timeToFirstToken: ttft,
            decodeTime: decode
        )
    }
}
