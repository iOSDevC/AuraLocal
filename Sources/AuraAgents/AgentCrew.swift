import Foundation
import Combine
import FoundationModels
import AuraCore
import AuraDocs
import AuraAppleIntelligence

// Disambiguate: AuraCore re-exports LocalLLMClientCore, which (on recent main)
// also defines `GeneratedContent`, colliding with FoundationModels' type used by
// the @Generable macro below. Pin the name to FoundationModels here. Gated because
// FoundationModels.GeneratedContent is macOS 26 / iOS 26 only.
@available(iOS 26, macOS 26, *)
typealias GeneratedContent = FoundationModels.GeneratedContent

// MARK: - AgentMemory

@available(iOS 26, macOS 26, *)
actor AgentMemory {

    private let store: ConversationStore
    private(set) var conversationID: UUID?

    init(store: ConversationStore) {
        self.store = store
    }

    func beginRun(topic: String) async throws {
        let conv = try await store.createConversation(
            model: .qwen3_1_7b,
            title: "AgentCrew: \(topic)"
        )
        conversationID = conv.id
    }

    func write(role: String, content: String) async throws {
        guard let convID = conversationID else { return }
        _ = try await store.appendTurn(Turn(
            conversationID: convID,
            role: .assistant,
            content: "[\(role)]\n\(content)"
        ))
    }

    func readAll() async throws -> [(role: String, content: String)] {
        guard let convID = conversationID else { return [] }
        return try await store.turns(for: convID).compactMap { turn in
            guard turn.role == .assistant,
                  turn.content.hasPrefix("["),
                  let bracket = turn.content.firstIndex(of: "]")
            else { return nil }
            let role    = String(turn.content[turn.content.index(after: turn.content.startIndex)..<bracket])
            let content = String(turn.content[turn.content.index(after: bracket)...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return (role: role, content: content)
        }
    }

    func read(from agentRole: String) async throws -> String? {
        try await readAll().first(where: { $0.role == agentRole })?.content
    }
}

// MARK: - DocumentQueryTool

@available(iOS 26, macOS 26, *)
struct DocumentQueryTool: Tool {

    let name        = "queryDocuments"
    let description = "Search indexed documents for relevant information."

    @Generable
    struct Arguments {
        @Guide(description: "A specific, keyword-rich search query")
        var query: String
        @Guide(description: "Number of results to retrieve, between 1 and 8")
        var topK: Int
    }

    private let library: DocumentLibrary

    init(library: DocumentLibrary) { self.library = library }

    func call(arguments: Arguments) async throws -> String {
        let k      = min(max(arguments.topK, 1), 8)
        let answer = try await library.ask(arguments.query, topK: k)
        guard !answer.sources.isEmpty else {
            return "No relevant documents found for: \(arguments.query)"
        }
        let text = answer.sources.enumerated().map { i, src in
            "[\(i+1)] \(src.documentTitle) (p.\(src.pageNumber), score: \(String(format:"%.2f", src.score)))\n\(src.excerpt)"
        }.joined(separator: "\n\n")
        return "Found \(answer.sources.count) passages:\n\n\(text)"
    }
}

// MARK: - MemoryReadTool

@available(iOS 26, macOS 26, *)
struct MemoryReadTool: Tool {

    let name        = "readAgentMemory"
    let description = "Read the output produced by a previous agent in this pipeline run."

    @Generable
    struct Arguments {
        @Guide(description: "The role name of the agent to read from, e.g. 'Extractor', 'Reviewer', 'Architect'")
        var agentRole: String
    }

    private let memory: AgentMemory

    init(memory: AgentMemory) { self.memory = memory }

    func call(arguments: Arguments) async throws -> String {
        if let content = try await memory.read(from: arguments.agentRole) {
            return "Output from \(arguments.agentRole):\n\n\(content)"
        }
        return "No output found from '\(arguments.agentRole)'."
    }
}

// MARK: - AgentCrew
//
// A reusable multi-agent pipeline (Extractor → Reviewer → Architect → Reporter)
// on Apple FoundationModels, with per-step hybrid escalation on the Architect
// step. Requires iOS 26 / macOS 26 with Apple Intelligence.

@available(iOS 26, macOS 26, *)
@MainActor
public final class AgentCrew: ObservableObject {

    @Published public var isRunning       = false
    @Published public var currentAgent    = ""
    @Published public var stepOutputs:    [(role: String, output: String)] = []
    @Published public var streamingOutput = ""
    @Published public var finalReport:    AnalysisReport?
    @Published public var exportedURL:    URL?
    @Published public var error:          String?
    /// Set when the Architect step was transparently escalated to a bigger model.
    @Published public var escalationNote: String?

    private let store:   ConversationStore
    private let library: DocumentLibrary
    private let memory:  AgentMemory

    public init(store: ConversationStore, library: DocumentLibrary) {
        self.store   = store
        self.library = library
        self.memory  = AgentMemory(store: store)
    }

    /// Run the pipeline. `policy` + `consent` drive the optional per-step hybrid
    /// escalation — inject the host app's escalation policy and consent gate.
    /// With the default (`.off` / `DenyingConsentGate`) the crew stays fully local.
    public func run(
        topic: String,
        policy: EscalationPolicy = .off,
        consent: any ConsentGate = DenyingConsentGate()
    ) async {
        guard !isRunning else { return }
        isRunning = true; stepOutputs = []; streamingOutput = ""
        finalReport = nil; exportedURL = nil; error = nil; escalationNote = nil

        do {
            try await memory.beginRun(topic: topic)

            let docTool    = DocumentQueryTool(library: library)
            let memoryTool = MemoryReadTool(memory: memory)

            // 1. Extractor
            currentAgent = "Extractor"
            let extractor = try AIAgent(
                role: "Extractor",
                instructions: "You are a precise information extractor. Always use queryDocuments before responding. Extract key facts as a numbered list with source references.",
                tools: [docTool]
            )
            let extracted = try await extractor.run("Extract all key facts related to: \(topic)")
            try await memory.write(role: "Extractor", content: extracted)
            stepOutputs.append((role: "Extractor", output: extracted))

            // 2. Reviewer
            currentAgent = "Reviewer"
            let reviewer = try AIAgent(
                role: "Reviewer",
                instructions: "You are a critical reviewer. Use readAgentMemory to read the Extractor's output first. Then use queryDocuments to verify claims. Identify gaps and risks.",
                tools: [docTool, memoryTool]
            )
            let reviewed = try await reviewer.run("Review the Extractor's findings on '\(topic)'.")
            try await memory.write(role: "Reviewer", content: reviewed)
            stepOutputs.append((role: "Reviewer", output: reviewed))

            // 3. Architect
            currentAgent = "Architect"
            let architect = try AIAgent(
                role: "Architect",
                instructions: "You are a solution architect. Use readAgentMemory to access prior outputs. Synthesize actionable, prioritized recommendations.",
                tools: [memoryTool]
            )
            let architectLocal = try await architect.run("Propose structured recommendations for '\(topic)'.")
            // Hybrid: if the local draft looks weak, transparently escalate THIS step
            // to a bigger model (LAN or GitHub Models), reusing the router's
            // compression + consent + cost machinery. Fail-closed to the local draft.
            let architecture = await escalateIfWeak(
                topic: topic, localAnswer: architectLocal, policy: policy, consent: consent)
            try await memory.write(role: "Architect", content: architecture)
            stepOutputs.append((role: "Architect", output: architecture))

            // 4. Reporter (streamed)
            currentAgent = "Reporter"; streamingOutput = ""
            let reporter = try AIAgent(
                role: "Reporter",
                instructions: "You are a technical writer. Use readAgentMemory to read all prior outputs. Write a final report with: 1.Executive Summary 2.Key Findings 3.Issues & Risks 4.Recommendations 5.Next Steps.",
                tools: [memoryTool]
            )
            var fullReport = ""
            for try await token in reporter.stream("Write the final report on '\(topic)'.") {
                streamingOutput += token
                fullReport      += token
            }
            try await memory.write(role: "Reporter", content: fullReport)
            stepOutputs.append((role: "Reporter", output: fullReport))

            finalReport = AnalysisReport(topic: topic, stepOutputs: stepOutputs, fullText: fullReport, createdAt: Date())
            exportedURL = try await exportLatestDocument()

        } catch let sessionError as AISessionError {
            switch sessionError {
            case .rateLimited:
                self.error = "Rate limited by Apple Intelligence. Please wait a moment and try again."
            case .contextWindowExceeded:
                self.error = "Context window exceeded. Try a shorter topic or fewer documents."
            case .guardrailViolation:
                self.error = "Content was filtered by Apple Intelligence safety guardrails."
            case .modelUnavailable(let reason):
                self.error = "Apple Intelligence unavailable: \(reason)"
            default:
                self.error = sessionError.localizedDescription
            }
        } catch {
            self.error = error.localizedDescription
        }
        isRunning = false; currentAgent = ""
    }

    /// Escalate the Architect step to a bigger model when the local draft looks weak.
    /// Reuses HybridEscalator (router + compression + consent + cost). Returns the
    /// local answer unchanged when the policy is off, the router keeps it local, or
    /// anything fails.
    private func escalateIfWeak(
        topic: String, localAnswer: String,
        policy: EscalationPolicy, consent: any ConsentGate
    ) async -> String {
        guard policy.mode != .off else { return localAnswer }
        let reviewerNotes = (try? await memory.read(from: "Reviewer")) ?? ""
        do {
            let result = try await HybridEscalator().routeAndEscalate(
                policy: policy,
                systemPrompt: "You are a solution architect. Synthesize actionable, prioritized recommendations.",
                context: reviewerNotes,
                question: "Propose structured, prioritized recommendations for '\(topic)'.",
                localAnswer: localAnswer,
                consent: consent)
            if let result, !result.answer.isEmpty {
                var note = "Architect escalated → \(result.providerName)"
                if let usage = result.usage { note += " · \(usage.inputTokens)/\(usage.outputTokens) tok" }
                if result.fromCache { note += " · cached" }
                escalationNote = note
                return result.answer
            }
        } catch {
            // Fail-closed: keep the local draft on any escalation error.
        }
        return localAnswer
    }

    private func exportLatestDocument() async throws -> URL? {
        let docs = try await library.allDocuments()
        guard let first = docs.first else { return nil }
        let dest = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AgentCrewExports", isDirectory: true)
        return try await library.export(documentID: first.id, to: dest, format: .jsonlGz)
    }

    public func loadPastRuns() async throws -> [Conversation] {
        try await store.allConversations()
            .filter { $0.title.hasPrefix("AgentCrew:") }
    }
}

// MARK: - AnalysisReport

public struct AnalysisReport: Identifiable {
    public let id          = UUID()
    public let topic:      String
    public let stepOutputs: [(role: String, output: String)]
    public let fullText:   String
    public let createdAt:  Date

    public init(topic: String, stepOutputs: [(role: String, output: String)], fullText: String, createdAt: Date) {
        self.topic       = topic
        self.stepOutputs = stepOutputs
        self.fullText    = fullText
        self.createdAt   = createdAt
    }
}
