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

/// A conversation that can **switch profiles mid-session**. A switch is a heavyweight engine **rebuild**, not an
/// in-place mutation; `conversationID` survives it. The session keeps no history: each ``stream(_:maxTokens:)``
/// call is a single turn, and storing turns (e.g. in `ConversationStore` under `conversationID`) is the caller's job.
///
/// This is the mutable slot the immutable `AuraLocal`/`AuraEngine` can't provide. Engine-agnostic; does NOT depend
/// on Apple's beta `LanguageModelSession.DynamicProfile`.
@MainActor public final class AuraSession {
    /// Stable across profile switches, for the caller to key stored turns by.
    public let conversationID: UUID
    public private(set) var profile: AuraProfile
    /// With an ``AuraProfile/outputSchema``, how many times ``stream(_:maxTokens:)`` re-prompts after a reply that
    /// breaks the schema. Each repair is a full extra generation. Negative values count as 0.
    public var maxRepairAttempts = 2

    private var engine: any AuraProfileEngine
    /// The active profile's compiled schema; `nil` streams plain deltas.
    private var validator: OutputSchemaValidator?
    /// The last generation or engine rebuild, kept even after ``cancel()``. Each new stream or switch awaits it
    /// before touching the engine, so work runs one at a time on the shared llama.cpp context.
    private var inFlight: Task<Void, Never>?
    /// Switches queued or running; while any is, a switch to the current profile is not a no-op.
    private var pendingSwitches = 0
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

    /// Switch to a different profile, keeping `conversationID`. Order is deliberate: **cancel AND await** the
    /// in-flight generation first (awaiting the pump awaits the real decode — never tear down the shared context
    /// mid-decode), then free the old model's RAM, then build the new engine. Streams and switches requested
    /// meanwhile queue behind the rebuild, so a stream started during a switch runs on the new profile.
    ///
    /// A profile whose ``AuraProfile/outputSchema`` does not compile throws ``OutputSchemaError`` before anything
    /// is cancelled or torn down, so the current profile stays usable. If building the new engine throws, the old
    /// one is already torn down: streams fail with ``AuraError/modelNotLoaded`` until a retry (to any profile,
    /// the current one included) succeeds.
    public func switchProfile(to newProfile: AuraProfile) async throws {
        guard newProfile != profile || switchFailed || pendingSwitches > 0 else { return }
        let newValidator = try newProfile.outputSchema.map(OutputSchemaValidator.init)
        let prior = inFlight
        prior?.cancel()
        pendingSwitches += 1
        let rebuild = Task { @MainActor in
            defer { self.pendingSwitches -= 1 }
            await prior?.value
            try await self.rebuild(to: newProfile, validator: newValidator)
        }
        inFlight = Task { _ = await rebuild.result }    // a later stream's cancel() must not abort the rebuild
        try await withTaskCancellationHandler {
            try await rebuild.value
        } onCancel: {
            rebuild.cancel()
        }
    }

    private func rebuild(to newProfile: AuraProfile, validator newValidator: OutputSchemaValidator?) async throws {
        guard newProfile != profile || switchFailed else { return }   // a queued switch already landed here
        try Task.checkCancellation()
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
    /// profile, never persisted as a conversation turn). Cancels the previous generation and waits for it, or for a
    /// switch in progress, before decoding; the profile in effect then is the one used.
    ///
    /// With an ``AuraProfile/outputSchema`` the reply is not streamed: the schema is appended to the system prompt,
    /// each generation is buffered, and its JSON is extracted and validated. A reply that breaks the schema is
    /// re-prompted with its violations up to ``maxRepairAttempts`` times. The stream then yields exactly one
    /// element, the conforming JSON text, or finishes throwing ``OutputSchemaError/violations(output:violations:)``
    /// for the last attempt.
    public func stream(_ prompt: String, maxTokens: Int? = nil) -> AsyncThrowingStream<String, Error> {
        let repairs = max(0, maxRepairAttempts)
        let prior = inFlight
        prior?.cancel()
        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
        let task = Task { @MainActor in
            await prior?.value                         // one decode at a time on the shared context
            do {
                try Task.checkCancellation()
                try await self.generate(prompt, maxTokens: maxTokens, repairs: repairs, into: continuation)
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        inFlight = task
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }

    /// Stop any in-flight generation (e.g. on teardown). Safe to call anytime; the next ``stream(_:maxTokens:)`` or
    /// ``switchProfile(to:)`` still waits for the cancelled decode to return.
    public func cancel() {
        inFlight?.cancel()
    }

    private func generate(_ prompt: String, maxTokens: Int?, repairs: Int,
                          into continuation: AsyncThrowingStream<String, Error>.Continuation) async throws {
        guard !switchFailed else { throw AuraError.modelNotLoaded }
        let engine = self.engine
        let tokens = maxTokens ?? profile.sampling.maxTokens
        if let validator {
            let system = Self.systemPrompt(profile.instructions, schema: validator.source)
            let json = try await Self.conformingReply(to: prompt, system: system, maxTokens: tokens,
                                                      repairs: repairs, engine: engine, validator: validator)
            continuation.yield(json)
        } else {
            try await engine.generate(prompt: prompt, systemPrompt: profile.instructions, maxTokens: tokens) { delta in
                continuation.yield(delta)
            }
        }
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
