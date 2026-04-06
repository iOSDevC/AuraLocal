import Foundation
import FoundationModels

// MARK: - AISessionError

@available(iOS 26, macOS 26, *)
public enum AISessionError: LocalizedError {
    case modelUnavailable(reason: String)
    case guardrailViolation(String)
    case contextWindowExceeded
    case rateLimited
    case concurrentRequest
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelUnavailable(let reason):
            return "Apple Intelligence unavailable: \(reason)"
        case .guardrailViolation(let detail):
            return "Content triggered safety filters: \(detail)"
        case .contextWindowExceeded:
            return "Context window exceeded. Try a shorter prompt or start a new session."
        case .rateLimited:
            return "Too many requests. Please wait a moment."
        case .concurrentRequest:
            return "Another request is already in progress."
        case .generationFailed(let detail):
            return "Generation failed: \(detail)"
        }
    }
}

// MARK: - AISession
//
// Wraps LanguageModelSession with a clean API aligned with AuraLocal conventions.
// Requires iOS 26+ / macOS 26+ with Apple Intelligence enabled.
//
// §Fix #20: Validates availability before creating session.
// §Fix #21: Maps LanguageModelSession.GenerationError to typed AISessionError.
// §Fix #22: Stream emits deltas instead of accumulated content.

@available(iOS 26, macOS 26, *)
public final class AISession: @unchecked Sendable {

    // MARK: - Public state

    /// True while the model is generating a response.
    public var isResponding: Bool { session.isResponding }

    /// Full conversation transcript — use to restore sessions.
    public var transcript: Transcript { session.transcript }

    // MARK: - Private

    private let session: LanguageModelSession

    // MARK: - Init

    /// Create a session with optional system instructions and tools.
    ///
    /// §Fix #20: Checks `SystemLanguageModel.default.availability` before creating
    /// the session. Throws `AISessionError.modelUnavailable` if not available.
    public init(
        instructions: String? = nil,
        tools: [any Tool] = [],
        useCase: SystemLanguageModel.UseCase? = nil
    ) throws {
        // §Fix #20: Always verify availability before session creation
        let availability = SystemLanguageModel.default.availability
        guard case .available = availability else {
            let reason: String
            if case .unavailable(let r) = availability {
                reason = String(describing: r)
            } else {
                reason = "Unknown"
            }
            throw AISessionError.modelUnavailable(reason: reason)
        }

        let model = useCase.map { SystemLanguageModel(useCase: $0) } ?? SystemLanguageModel.default
        if let instructions {
            self.session = LanguageModelSession(model: model, tools: tools) { instructions }
        } else {
            self.session = LanguageModelSession(model: model, tools: tools)
        }
    }

    /// Restore a session from a saved transcript.
    public init(transcript: Transcript, tools: [any Tool] = []) throws {
        let availability = SystemLanguageModel.default.availability
        guard case .available = availability else {
            let reason: String
            if case .unavailable(let r) = availability {
                reason = String(describing: r)
            } else {
                reason = "Unknown"
            }
            throw AISessionError.modelUnavailable(reason: reason)
        }

        self.session = LanguageModelSession(
            model:      SystemLanguageModel.default,
            tools:      tools,
            transcript: transcript
        )
    }

    // MARK: - Plain text

    /// Send a prompt and await the full response.
    /// §Fix #21: Wraps GenerationError into typed AISessionError.
    public func respond(
        to prompt: String,
        options: GenerationOptions = GenerationOptions()
    ) async throws -> String {
        do {
            let response = try await session.respond(to: prompt, options: options)
            return response.content
        } catch {
            throw mapGenerationError(error)
        }
    }

    /// Stream a plain-text response, yielding incremental **deltas** (not accumulated).
    /// §Fix #22: Computes deltas by tracking previous content length.
    public func stream(
        _ prompt: String,
        options: GenerationOptions = GenerationOptions()
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let responseStream = session.streamResponse(to: prompt, options: options)
                    var previousContent = ""
                    for try await partial in responseStream {
                        let currentContent = partial.content
                        if currentContent.count > previousContent.count {
                            let delta = String(currentContent.dropFirst(previousContent.count))
                            continuation.yield(delta)
                        }
                        previousContent = currentContent
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: self.mapGenerationError(error))
                }
            }
        }
    }

    // MARK: - Guided generation

    /// Generate a typed Swift value using @Generable structured output.
    public func respond<T: Generable>(
        to prompt: String,
        generating type: T.Type,
        options: GenerationOptions = GenerationOptions()
    ) async throws -> T {
        do {
            let response = try await session.respond(
                to: prompt,
                generating: type,
                options: options
            )
            return response.content
        } catch {
            throw mapGenerationError(error)
        }
    }

    /// Stream a structured response, yielding partial snapshots as fields fill in.
    public func stream<T: Generable>(
        _ prompt: String,
        generating type: T.Type,
        options: GenerationOptions = GenerationOptions()
    ) -> AsyncThrowingStream<T.PartiallyGenerated, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let stream = session.streamResponse(
                        to: prompt,
                        generating: type,
                        options: options
                    )
                    for try await partial in stream {
                        nonisolated(unsafe) let value = partial.content
                        continuation.yield(value)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: self.mapGenerationError(error))
                }
            }
        }
    }

    // MARK: - Prewarming

    /// Preload the model into memory before the first request.
    public func prewarm() {
        session.prewarm()
    }

    // MARK: - Error Mapping
    //
    // §Fix #21: Maps FoundationModels GenerationError into typed AISessionError
    // for downstream consumers to handle granularly.

    private func mapGenerationError(_ error: Error) -> Error {
        guard let genError = error as? LanguageModelSession.GenerationError else {
            return error
        }
        switch genError {
        case .guardrailViolation(let ctx):
            return AISessionError.guardrailViolation(String(describing: ctx))
        case .exceededContextWindowSize:
            return AISessionError.contextWindowExceeded
        case .rateLimited:
            return AISessionError.rateLimited
        case .concurrentRequests:
            return AISessionError.concurrentRequest
        default:
            return AISessionError.generationFailed(String(describing: genError))
        }
    }
}

// MARK: - AIAgent
//
// A single agent with a role, instructions, and a set of tools.
// Analogous to a CrewAI Agent — has a defined responsibility and
// can call tools to gather information before responding.

@available(iOS 26, macOS 26, *)
public struct AIAgent {

    public let role:         String
    public let instructions: String
    public let tools:        [any Tool]
    public let options:      GenerationOptions

    private let session: AISession

    /// §Fix #20: init is now throwing because AISession verifies availability.
    public init(
        role:         String,
        instructions: String,
        tools:        [any Tool] = [],
        options:      GenerationOptions = GenerationOptions()
    ) throws {
        self.role         = role
        self.instructions = instructions
        self.tools        = tools
        self.options      = options
        self.session      = try AISession(instructions: instructions, tools: tools)
        self.session.prewarm()
    }

    /// Run the agent and return the full response.
    public func run(_ input: String) async throws -> String {
        try await session.respond(to: input, options: options)
    }

    /// Run the agent and stream incremental response deltas.
    public func stream(_ input: String) -> AsyncThrowingStream<String, Error> {
        session.stream(input, options: options)
    }

    /// Run the agent and produce a typed structured output.
    public func run<T: Generable>(_ input: String, generating type: T.Type) async throws -> T {
        try await session.respond(to: input, generating: type, options: options)
    }
}

// MARK: - AIPipeline
//
// Orchestrates a sequential chain of AIAgents.
// Each agent's output becomes the next agent's input.
// Analogous to CrewAI's Crew with Process.sequential.

@available(iOS 26, macOS 26, *)
public final class AIPipeline: @unchecked Sendable {

    public let agents: [AIAgent]

    /// Step-by-step outputs from each agent in the last run.
    public private(set) var stepOutputs: [(role: String, output: String)] = []

    public init(agents: [AIAgent]) {
        precondition(!agents.isEmpty, "AIPipeline requires at least one agent")
        self.agents = agents
    }

    /// Execute all agents sequentially. Returns the final agent's output.
    public func run(
        input: String,
        onStep: ((String, String) -> Void)? = nil
    ) async throws -> String {
        stepOutputs = []
        var current = input
        for agent in agents {
            let output = try await agent.run(current)
            stepOutputs.append((role: agent.role, output: output))
            onStep?(agent.role, output)
            current = output
        }
        return current
    }

    /// Execute agents sequentially, streaming the final agent's output.
    /// All agents except the last run to completion first.
    public func stream(
        input: String,
        onStep: (@Sendable (String, String) -> Void)? = nil
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    self.stepOutputs = []
                    var current = input
                    for agent in self.agents.dropLast() {
                        let output = try await agent.run(current)
                        self.stepOutputs.append((role: agent.role, output: output))
                        onStep?(agent.role, output)
                        current = output
                    }
                    if let last = self.agents.last {
                        var full = ""
                        for try await token in last.stream(current) {
                            full += token
                            continuation.yield(token)
                        }
                        self.stepOutputs.append((role: last.role, output: full))
                        onStep?(last.role, full)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

// MARK: - AIAvailability

@available(iOS 26, macOS 26, *)
public enum AIAvailability {

    public enum Status: Equatable {
        case available
        case unavailable(reason: String)
    }

    public static var status: Status {
        switch SystemLanguageModel.default.availability {
            case .available:
                return .available
            case .unavailable(let reason):
                return .unavailable(reason: String(describing: reason))
            @unknown default:
                return .unavailable(reason: "Unknown availability state")
        }
    }

    public static var isAvailable: Bool { status == .available }
}

// MARK: - AIAvailabilityGate (SwiftUI)

#if canImport(SwiftUI)
import SwiftUI

/// Gate view — shows content only when Apple Intelligence is available.
@available(iOS 26, macOS 26, *)
public struct AIAvailabilityGate<Content: View>: View {
    let content: () -> Content

    public init(@ViewBuilder content: @escaping () -> Content) {
        self.content = content
    }

    public var body: some View {
        switch SystemLanguageModel.default.availability {
            case .available:
                content()
            case .unavailable(let reason):
                ContentUnavailableView {
                    Label("Apple Intelligence Unavailable", systemImage: "sparkles.slash")
                } description: {
                    Text(unavailableMessage(for: reason))
                }
            @unknown default:
                ContentUnavailableView("Unavailable", systemImage: "exclamationmark.triangle")
        }
    }

    private func unavailableMessage(
        for reason: SystemLanguageModel.Availability.UnavailableReason
    ) -> String {
        switch reason {
            case .deviceNotEligible:
                return "This feature requires an Apple Intelligence-compatible device."
            case .appleIntelligenceNotEnabled:
                return "Enable Apple Intelligence in Settings → Apple Intelligence & Siri."
            case .modelNotReady:
                return "The on-device model is still downloading. Try again shortly."
            @unknown default:
                return "Apple Intelligence is currently unavailable."
        }
    }
}
#endif
