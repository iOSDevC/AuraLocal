import Foundation
import Combine
import SwiftUI
import AuraCore
import AuraDocs
import AuraAppleIntelligence
import AuraAgents

// MARK: - AppState
//
// Central dependency container shared across all tabs via @EnvironmentObject.
// Owns DocumentLibrary, ConversationStore, and the AgentCrew instance.
// All services are initialized once — no duplicate model loads.

@MainActor
final class AppState: ObservableObject {

    // MARK: - Published

    @Published var isReady       = false
    @Published var setupProgress = "Initializing…"
    @Published var setupError:     String?
    /// Set when setup stopped because a model is not on disk; the Agents tab offers to download these.
    @Published private(set) var modelsToDownload: [RequiredModel] = []

    // MARK: - Shared services

    let store   = ConversationStore.shared
    let library = DocumentLibrary.shared
    private var crewStorage: AnyObject?
    private var isSettingUp = false
    @available(iOS 26, macOS 26, *)
    var crew: AgentCrew? { crewStorage as? AgentCrew }

    // MARK: - Setup

    func setup() async {
        // The tab's `.task` and the placeholder's `onReady` can both call this once the models are on disk.
        guard !isReady, !isSettingUp else { return }
        let needed = Self.neededModels()
        guard needed.notDownloaded.isEmpty else {
            modelsToDownload = needed
            return
        }
        isSettingUp = true
        defer { isSettingUp = false }
        modelsToDownload = []

        do {
            // 1. Open SQLite store — throws (not async)
            setupProgress = "Opening database…"
            try await store.open()

            // 2. Load LLM via ModelManager
            setupProgress = "Loading language model…"
            let llm = try await ModelManager.shared.load(.qwen3_1_7b) { [weak self] p in
                self?.setupProgress = p
            }

            // 3. Load vision model (only if fits in memory alongside LLM)
            setupProgress = "Loading vision model…"
            let vlm: AuraLocal
            if needed.contains(where: { $0.entry == .fastVLM_0_5b_fp16 }) {
                vlm = try await ModelManager.shared.load(.fastVLM_0_5b_fp16) { [weak self] p in
                    self?.setupProgress = p
                }
            } else {
                let available = HardwareProfile.current().availableMemoryGB
                setupProgress = "Skipping vision model (low memory: \(String(format: "%.1f", available)) GB available)…"
                // Fall back to loading on-demand when Vision tab is used
                vlm = llm  // placeholder — VisionTab will lazy-load if needed
            }

            // 4. Configure DocumentLibrary — synchronous
            setupProgress = "Initializing document index…"
            // The Library tab may have chosen e5 already; replacing it with TF-IDF would re-embed its index.
            await library.configureIfNeeded(embeddingProvider: AutoEmbeddingProvider(), llm: llm, visionLLM: vlm)
            try await library.open()
            await library.refreshCorpus()

            // 5. Build AgentCrew
            setupProgress = "Initializing agents…"
            if #available(iOS 26, macOS 26, *) {
                crewStorage = AgentCrew(store: store, library: library)
            }

            setupProgress = "Ready"
            isReady = true

        } catch {
            setupError    = error.localizedDescription
            setupProgress = "Setup failed"
        }
    }

    /// The language model, plus the vision model when both fit in memory together. §Fix #19: leave 15 % headroom.
    private static func neededModels() -> [RequiredModel] {
        let answers = RequiredModel(.qwen3_1_7b, purpose: "Answers the agents' questions about your documents")
        let joint = Model.qwen3_1_7b.estimatedRuntimeMemoryGB + Model.fastVLM_0_5b_fp16.estimatedRuntimeMemoryGB
        guard joint <= HardwareProfile.current().availableMemoryGB * 0.85 else { return [answers] }
        return [answers, RequiredModel(.fastVLM_0_5b_fp16, purpose: "Reads the text in images you add")]
    }
}
