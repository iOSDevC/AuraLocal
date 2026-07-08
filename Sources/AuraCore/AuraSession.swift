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

    private var engine: any AuraProfileEngine
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
    public func switchProfile(to newProfile: AuraProfile) async throws {
        guard newProfile != profile || switchFailed else { return }
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
    }

    /// Stream a completion, injecting the ACTIVE profile's persona as the system prompt (the persona lives on the
    /// profile, never persisted as a conversation turn). Serializes behind any prior in-flight generation so two
    /// decodes never run on the shared context; tracked so `switchProfile` can cancel + await it before teardown.
    public func stream(_ prompt: String, maxTokens: Int? = nil) -> AsyncThrowingStream<String, Error> {
        let engine = self.engine
        let system = profile.instructions
        let tokens = maxTokens ?? profile.sampling.maxTokens
        let prior = inFlight
        prior?.cancel()
        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
        let task = Task { @MainActor in
            await prior?.value                         // one decode at a time on the shared context
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
