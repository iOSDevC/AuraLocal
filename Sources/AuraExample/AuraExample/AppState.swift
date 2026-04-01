import Foundation
import Combine
import SwiftUI
import AuraCore
import AuraDocs
import AuraAppleIntelligence

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

    // MARK: - Shared services

    let store   = ConversationStore.shared
    let library = DocumentLibrary.shared
    private(set) var crew: AgentCrew?

    // MARK: - Setup

    func setup() async {
        guard !isReady else { return }

        do {
            // 1. Open SQLite store — throws (not async)
            setupProgress = "Opening database…"
            try await store.open()

            // 2. Load LLM via ModelManager
            setupProgress = "Loading language model…"
            let llm = try await ModelManager.shared.load(.qwen3_1_7b) { [weak self] p in
                self?.setupProgress = p
            }

            // §Fix #19: Verify joint RAM budget before loading the second model.
            // On iOS, never exceed ~60% of total RAM for both models combined.
            let llmMemory = Model.qwen3_1_7b.estimatedRuntimeMemoryGB
            let vlmMemory = Model.fastVLM_0_5b_fp16.estimatedRuntimeMemoryGB
            let profile   = HardwareProfile.current()
            let jointUsage = llmMemory + vlmMemory
            let safeLimit  = profile.availableMemoryGB * 0.85  // leave 15% headroom

            // 3. Load vision model (only if fits in memory alongside LLM)
            setupProgress = "Loading vision model…"
            let vlm: AuraLocal
            if jointUsage <= safeLimit {
                vlm = try await ModelManager.shared.load(.fastVLM_0_5b_fp16) { [weak self] p in
                    self?.setupProgress = p
                }
            } else {
                setupProgress = "Skipping vision model (low memory: \(String(format: "%.1f", profile.availableMemoryGB)) GB available)…"
                // Fall back to loading on-demand when Vision tab is used
                vlm = llm  // placeholder — VisionTab will lazy-load if needed
            }

            // 4. Configure DocumentLibrary — synchronous
            setupProgress = "Initializing document index…"
            await library.configure(embeddingProvider: AutoEmbeddingProvider(), llm: llm, visionLLM: vlm)
            try await library.open()
            await library.refreshCorpus()

            // 5. Build AgentCrew
            setupProgress = "Initializing agents…"
            if #available(iOS 26, macOS 26, *) {
                crew = AgentCrew(store: store, library: library)
            }

            setupProgress = "Ready"
            isReady = true

        } catch {
            setupError    = error.localizedDescription
            setupProgress = "Setup failed"
        }
    }
}
