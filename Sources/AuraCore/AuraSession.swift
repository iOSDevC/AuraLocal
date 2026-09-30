import Foundation
import LocalLLMClientCore

// MARK: - AuraProfileEngine

/// The engine seam an ``AuraSession`` drives — abstracted so the switch state machine is unit-testable with a
/// stub (no live GGUF model). `AuraLocal` conforms below.
///
/// `generate` is **async** (not a stream that hides a detached task): awaiting it awaits the actual decode, so a
/// caller can cancel + `await` an in-flight generation and be certain the (shared) llama.cpp context is idle
/// before `teardown()` frees it — the keystone's core safety invariant.
@MainActor public protocol AuraProfileEngine: AnyObject {
    func generate(prompt: String, systemPrompt: String?, maxTokens: Int,
                  onDelta: @escaping @MainActor (String) -> Void) async throws
    /// Release the model + its (shared) llama.cpp context. Idempotent.
    func teardown()
}

extension AuraLocal: AuraProfileEngine {
    public func generate(prompt: String, systemPrompt: String?, maxTokens: Int,
                         onDelta: @escaping @MainActor (String) -> Void) async throws {
        var last = 0
        _ = try await engine.generate(prompt: prompt, systemPrompt: systemPrompt, maxTokens: maxTokens) { @MainActor partial in
            let delta = String(partial.dropFirst(last))   // engine reports the cumulative text; emit the new tail
            last = partial.count
            if !delta.isEmpty { onDelta(delta) }
        }
    }

    /// Route teardown through `ModelManager` so a SHARED cached instance is evicted (not left a torn-down zombie
    /// in the cache); an owned tool-enabled instance is just unloaded. Identity-checked, so this is safe + idempotent.
    public func teardown() { ModelManager.shared.release(self) }
}

// MARK: - AuraSession

/// A conversation that can **switch profiles mid-session while preserving the transcript**. The transcript lives
/// outside the engine (in `ConversationStore`, keyed by `conversationID`), and both GGUF backends reset
/// `session.messages` every turn — so a switch is a heavyweight engine **rebuild**, not an in-place mutation, and
/// nothing conversational is lost (the new engine just pays one cold prefill on the next turn).
///
/// This is the mutable slot the immutable `AuraLocal`/`AuraEngine` can't provide. Engine-agnostic; does NOT depend
/// on Apple's beta `LanguageModelSession.DynamicProfile`.
@MainActor public final class AuraSession {
    /// Stable across profile switches — the key the transcript is stored under.
    public let conversationID: UUID
    public private(set) var profile: AuraProfile
    /// With an ``AuraProfile/outputSchema``, how many times ``stream(_:maxTokens:)`` re-prompts after a reply that
    /// breaks the schema. Each repair is a full extra generation. Negative values count as 0.
    public var maxRepairAttempts = 2

    private var engine: any AuraProfileEngine
    /// The active profile's compiled schema; `nil` streams plain deltas.
    private var validator: OutputSchemaValidator?
    /// The in-flight generation, tracked so a switch (or a new stream) can cancel + **await** it before the engine
    /// is torn down / a second decode starts (one decode at a time on the shared llama.cpp context).
    private var inFlight: Task<Void, Never>?
    /// True after a failed rebuild — the engine is torn down; allows a retry to ANY profile (even the current one)
    /// past the equality guard, so a load failure isn't a permanent dead end.
    private var switchFailed = false
    private let makeEngine: @MainActor (AuraProfile) async throws -> any AuraProfileEngine

    public init(conversationID: UUID = UUID(),
                profile: AuraProfile,
                makeEngine: @escaping @MainActor (AuraProfile) async throws -> any AuraProfileEngine = AuraSession.liveEngine) async throws {
        self.conversationID = conversationID
        self.profile = profile
        self.makeEngine = makeEngine
        self.validator = try profile.outputSchema.map(OutputSchemaValidator.init)   // before paying for a model load
        self.engine = try await makeEngine(profile)
    }

    /// The default factory: build a live GGUF engine via `ModelManager` for the profile's model + tools.
    /// (Threading `profile.sampling.temperature` into the load is a follow-up — today `ModelManager.load` doesn't
    /// take it, so the engine uses its default sampling.)
    @MainActor public static func liveEngine(_ profile: AuraProfile) async throws -> any AuraProfileEngine {
        try await ModelManager.shared.load(profile.model, tools: profile.tools)
    }

    /// Switch to a different profile, preserving the conversation. Order is deliberate: **cancel AND await** any
    /// in-flight generation first (awaiting the pump awaits the real decode — never tear down the shared context
    /// mid-decode), then free the old model's RAM, then build the new engine. `conversationID` is unchanged; the
    /// stored transcript is untouched. On a rebuild failure the session is left recoverable (a retry re-enters).
    /// A profile whose ``AuraProfile/outputSchema`` does not compile throws ``OutputSchemaError`` before anything
    /// is cancelled or torn down, so the current profile stays usable.
    public func switchProfile(to newProfile: AuraProfile) async throws {
        guard newProfile != profile || switchFailed else { return }
        let newValidator = try newProfile.outputSchema.map(OutputSchemaValidator.init)
        inFlight?.cancel()
        await inFlight?.value
        inFlight = nil
        engine.teardown()                              // free the old model before loading the new (on-device RAM)
        do {
            engine = try await makeEngine(newProfile)
        } catch {
            switchFailed = true                        // torn down; guard now lets a retry re-enter
            throw error
        }
        switchFailed = false
        profile = newProfile
        validator = newValidator
    }

    /// Stream a completion, injecting the ACTIVE profile's persona as the system prompt (the persona lives on the
    /// profile, never persisted as a conversation turn). Serializes behind any prior in-flight generation so two
    /// decodes never run on the shared context; tracked so `switchProfile` can cancel + await it before teardown.
    ///
    /// With an ``AuraProfile/outputSchema`` the reply is not streamed: the schema is appended to the system prompt,
    /// each generation is buffered, and its JSON is extracted and validated. A reply that breaks the schema is
    /// re-prompted with its violations up to ``maxRepairAttempts`` times. The stream then yields exactly one
    /// element, the conforming JSON text, or finishes throwing ``OutputSchemaError/violations(output:violations:)``
    /// for the last attempt.
    public func stream(_ prompt: String, maxTokens: Int? = nil) -> AsyncThrowingStream<String, Error> {
        let engine = self.engine
        let tokens = maxTokens ?? profile.sampling.maxTokens
        let prior = inFlight
        prior?.cancel()
        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
        let task: Task<Void, Never>
        if let validator {
            let system = Self.systemPrompt(profile.instructions, schema: validator.source)
            let repairs = max(0, maxRepairAttempts)
            task = Task { @MainActor in
                await prior?.value                     // one decode at a time on the shared context
                do {
                    let json = try await Self.conformingReply(to: prompt, system: system, maxTokens: tokens,
                                                              repairs: repairs, engine: engine, validator: validator)
                    continuation.yield(json)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        } else {
            let system = profile.instructions
            task = Task { @MainActor in
                await prior?.value                     // one decode at a time on the shared context
                do {
                    try Task.checkCancellation()
                    try await engine.generate(prompt: prompt, systemPrompt: system, maxTokens: tokens) { delta in
                        continuation.yield(delta)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
        inFlight = task
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }

    /// Stop any in-flight generation (e.g. on teardown). Safe to call anytime.
    public func cancel() {
        inFlight?.cancel()
        inFlight = nil
    }
}

// MARK: - Output schema

extension AuraSession {
    /// Generate, validate, and re-prompt with the violations until a reply conforms or the repairs run out.
    private static func conformingReply(to prompt: String, system: String, maxTokens: Int, repairs: Int,
                                        engine: any AuraProfileEngine,
                                        validator: OutputSchemaValidator) async throws -> String {
        var request = prompt
        var attempt = 0
        while true {
            try Task.checkCancellation()
            let output = try await bufferedReply(to: request, system: system, maxTokens: maxTokens, engine: engine)
            try Task.checkCancellation()              // a cancelled decode may return early instead of throwing
            let failure: OutputSchemaError
            switch validator.check(output) {
                case .success(let json): return json
                case .failure(let error): failure = error
            }
            guard attempt < repairs else { throw failure }
            attempt += 1
            request = repairPrompt(for: prompt, after: failure)
        }
    }

    private static func bufferedReply(to prompt: String, system: String, maxTokens: Int,
                                      engine: any AuraProfileEngine) async throws -> String {
        var reply = ""
        try await engine.generate(prompt: prompt, systemPrompt: system, maxTokens: maxTokens) { delta in
            reply += delta
        }
        return reply
    }

    private static func systemPrompt(_ instructions: String, schema: String) -> String {
        let contract = "Reply with a single JSON value that conforms to this JSON Schema, no prose:\n\(schema)"
        return instructions.isEmpty ? contract : "\(instructions)\n\n\(contract)"
    }

    private static func repairPrompt(for prompt: String, after failure: OutputSchemaError) -> String {
        guard case .violations(let output, let found) = failure else { return prompt }
        let problems = found.map { "- \($0.description)" }.joined(separator: "\n")
        return """
            \(prompt)

            Your previous reply did not conform to the required JSON Schema.
            Previous reply:
            \(output)

            Problems:
            \(problems)

            Reply again with only the corrected JSON value.
            """
    }
}
