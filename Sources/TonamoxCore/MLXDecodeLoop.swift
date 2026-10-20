// MLX decode loop — compiled in only when the MLX products (mlx-swift-lm) are linked.
#if canImport(MLXLLM)
import Foundation
import MLX
import MLXLMCommon

// MARK: - Request and outcome

/// What one MLX reply may do.
struct MLXDecodeRequest: Sendable {
    /// Tokens the reply may generate, its stop token not counted.
    let budget: Int
    /// With tools, calls are parsed out of the reply in the model's format and only the text around them streams.
    let toolSpecs: [[String: any Sendable]]?
    /// Visible text of the turn's earlier rounds; `onToken` receives it followed by this reply's.
    let earlierText: String
    /// Whether the prompt may reuse the backend's prompt cache and leave its own behind.
    let reusesCache: Bool
}

/// One reply, with counts and timings for ``GenerationStats``.
struct MLXDecodeOutcome: Sendable {
    /// Everything decoded, or without tools' calls only the text around them.
    let text: String
    let calls: [ToolCall]
    let generatedTokens: Int
    /// The reply reached a stop token; `false` when the budget or a cancellation cut it.
    let finished: Bool
    /// The whole prompt, reused tokens included.
    let promptTokens: Int
    let reusedTokens: Int
    let promptTime: TimeInterval
    let timeToFirstToken: TimeInterval
    let decodeTime: TimeInterval
    let overflow: ContextWindowOverflow?

    /// One turn's stats from its replies: counts and prefill and decode times summed, the prompt of the last reply,
    /// time to first token of the first.
    static func stats(of replies: [MLXDecodeOutcome]) -> GenerationStats? {
        guard let first = replies.first, let last = replies.last else { return nil }
        return GenerationStats(
            backend: .mlx,
            source: .runtime,
            promptTokens: last.promptTokens,
            generatedTokens: replies.reduce(0) { $0 + $1.generatedTokens },
            promptTime: replies.reduce(0) { $0 + $1.promptTime },
            timeToFirstToken: first.timeToFirstToken,
            decodeTime: replies.reduce(0) { $0 + $1.decodeTime })
    }
}

// MARK: - MLXDecodeLoop

/// Samples one reply token by token on a cache it owns, so it knows exactly which tokens the cache holds afterwards.
enum MLXDecodeLoop {
    /// Runs inside `ModelContainer.perform`. `started` is when the request began, before the prompt was rendered.
    static func run(
        _ input: LMInput,
        request: MLXDecodeRequest,
        context: ModelContext,
        parameters: GenerateParameters,
        promptCache: MLXPromptCache,
        started: ContinuousClock.Instant,
        onToken: @escaping @MainActor (String) -> Void
    ) throws -> MLXDecodeOutcome {
        let prompt = input.text.tokens.asArray(Int.self)
        let reuses = request.reusesCache && reusable(input, parameters: parameters)
        let fresh = { context.model.newCache(parameters: parameters) }
        let lease = reuses
            ? promptCache.lease(for: prompt, window: parameters.maxKVSize, fresh: fresh)
            : MLXPromptCache.Lease(cache: fresh(), reused: 0)
        let feed = lease.reused > 0 ? suffix(of: prompt, from: lease.reused) : input

        let prefillStart = ContinuousClock.now
        var iterator = try TokenIterator(input: feed, model: context.model, cache: lease.cache, parameters: parameters)
        let stops = stopTokens(context)
        var reply = MLXReplyText(context: context, toolSpecs: request.toolSpecs)
        var fed: [Int] = []
        var clock = MLXTokenClock()
        var finished = false
        while let token = iterator.next() {
            clock.step()
            fed.append(token)
            if stops.contains(token) {
                finished = true
                break
            }
            clock.token()
            if reply.append(token) {
                deliver(request.earlierText + reply.visible, to: onToken)
            }
            if reply.generated >= request.budget || Task.isCancelled { break }
        }
        let calls = reply.finish()
        // TokenIterator keeps one step evaluating ahead; let it finish before the cache is stored or freed.
        Stream().synchronize()
        if reuses { promptCache.keep(lease.cache, holding: prompt + fed) }

        let overflow = parameters.maxKVSize.flatMap {
            ContextWindowOverflow.detect(promptTokens: prompt.count, generatedTokens: reply.generated, windowSize: $0)
        }.flatMap { candidate in
            cacheHonorsWindow(context.model, parameters) ? candidate : nil
        }
        return MLXDecodeOutcome(
            text: reply.visible,
            calls: calls,
            generatedTokens: reply.generated,
            finished: finished,
            promptTokens: prompt.count,
            reusedTokens: lease.reused,
            promptTime: clock.prefill(since: prefillStart),
            timeToFirstToken: clock.timeToFirstToken(since: started),
            decodeTime: clock.decode,
            overflow: overflow)
    }

    /// Tokens alone carry the prompt, and no logit processor needs to see the whole of it.
    private static func reusable(_ input: LMInput, parameters: GenerateParameters) -> Bool {
        input.image == nil && input.video == nil
            && parameters.repetitionPenalty == nil && parameters.presencePenalty == nil
            && parameters.frequencyPenalty == nil
    }

    private static func suffix(of prompt: [Int], from start: Int) -> LMInput {
        LMInput(tokens: MLXArray(Array(prompt[start...])))
    }

    /// The stop set mlx-swift-lm's own loops use.
    private static func stopTokens(_ context: ModelContext) -> Set<Int> {
        let extra = context.configuration.extraEOSTokens.compactMap { context.tokenizer.convertTokenToId($0) }
        let single = [context.tokenizer.eosTokenId, context.tokenizer.unknownTokenId].compactMap { $0 }
        return context.configuration.eosTokenIds.union(extra).union(single)
    }

    private static func deliver(_ text: String, to onToken: @escaping @MainActor (String) -> Void) {
        Task { @MainActor in onToken(text) }
    }

    /// Several architectures (Qwen3.5 text, Gemma 3/4, LFM2…) build their own caches and ignore `maxKVSize`, so
    /// nothing was dropped there. Comparing the caches built with and without it tells them apart.
    static func cacheHonorsWindow(_ model: any LanguageModel, _ parameters: GenerateParameters) -> Bool {
        var unbounded = parameters
        unbounded.maxKVSize = nil
        return model.newCache(parameters: parameters).map(\.maxSize)
            != model.newCache(parameters: unbounded).map(\.maxSize)
    }
}

// MARK: - MLXReplyText

/// The reply's text as it streams: every token decoded, or, with tools, the text around the calls the model's
/// format parser takes out.
struct MLXReplyText {
    private let codec: any MLXLMCommon.Tokenizer
    private var detokenizer: NaiveStreamingDetokenizer
    private let parser: ToolCallProcessor?
    private var tokens: [Int] = []
    private(set) var visible = ""

    var generated: Int { tokens.count }

    init(context: ModelContext, toolSpecs: [[String: any Sendable]]?) {
        codec = context.tokenizer
        detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)
        parser = toolSpecs.map { ToolCallProcessor(format: context.configuration.toolCallFormat ?? .json, tools: $0) }
    }

    /// Returns whether the visible text changed.
    mutating func append(_ token: Int) -> Bool {
        tokens.append(token)
        guard let parser else {
            visible = codec.decode(tokenIds: tokens)
            return true
        }
        detokenizer.append(token: token)
        guard let chunk = detokenizer.next(), let text = parser.processChunk(chunk), !text.isEmpty else { return false }
        visible += text
        return true
    }

    /// The calls parsed from the whole reply, in the order the model wrote them.
    mutating func finish() -> [ToolCall] {
        guard let parser else { return [] }
        parser.processEOS()
        return parser.toolCalls
    }
}

// MARK: - MLXTokenClock

/// Timings of one reply, matching the callback-based timing it replaces.
struct MLXTokenClock {
    private var firstStep: ContinuousClock.Instant?
    private var firstToken: ContinuousClock.Instant?
    private var lastToken: ContinuousClock.Instant?

    /// A token came out of the iterator, stop tokens included.
    mutating func step() {
        if firstStep == nil { firstStep = ContinuousClock.now }
    }

    /// A token joined the reply.
    mutating func token() {
        let now = ContinuousClock.now
        if firstToken == nil { firstToken = now }
        lastToken = now
    }

    /// Prefill: from the start of prompt processing to the first token out.
    func prefill(since start: ContinuousClock.Instant) -> TimeInterval {
        GenerationStats.seconds((firstStep ?? ContinuousClock.now) - start)
    }

    func timeToFirstToken(since start: ContinuousClock.Instant) -> TimeInterval {
        GenerationStats.seconds((firstToken ?? firstStep ?? ContinuousClock.now) - start)
    }

    var decode: TimeInterval {
        firstToken.flatMap { first in lastToken.map { GenerationStats.seconds($0 - first) } } ?? 0
    }
}
#endif
