import SwiftUI
import UniformTypeIdentifiers
import AuraCore
import AuraDocs

// MARK: - PipelineTab

struct PipelineTab: View {
    @EnvironmentObject var appState: AppState
    @State private var topic        = ""
    @State private var showReport   = false
    @FocusState private var focused: Bool

    private var crew: AgentCrew? { appState.crew }

    var body: some View {
        NavigationStack {
            Group {
                if !appState.isReady {
                    setupView
                } else {
                    ScrollView {
                        VStack(spacing: 20) {
                            topicInput
                            if let crew {
                                agentGrid(crew: crew)
                                if !crew.stepOutputs.isEmpty {
                                    stepList(crew: crew)
                                }
                                if crew.isRunning && !crew.streamingOutput.isEmpty {
                                    streamingCard(crew: crew)
                                }
                                if crew.finalReport != nil {
                                    reportBanner
                                }
                                if let err = crew.error {
                                    errorCard(err)
                                }
                            }
                        }
                        .padding()
                    }
                }
            }
            .navigationTitle("Agent Crew")
            .sheet(isPresented: $showReport) {
                if let report = crew?.finalReport {
                    ReportSheet(report: report, exportedURL: crew?.exportedURL)
                }
            }
        }
    }

    // MARK: - Setup

    private var setupView: some View {
        VStack(spacing: 16) {
            ProgressView().scaleEffect(1.3)
            Text(appState.setupProgress)
                .font(.subheadline).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            if let err = appState.setupError {
                Text(err).font(.caption).foregroundStyle(.red)
            }
        }
        .padding(32)
    }

    // MARK: - Topic input

    private var topicInput: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Analysis Topic").font(.headline)
            HStack(spacing: 10) {
                TextField(
                    "e.g. Security vulnerabilities in authentication flow",
                    text: $topic,
                    axis: .vertical
                )
                .lineLimit(2...4)
                .padding(12)
                .background(Color.secondarySystemBackground, in: RoundedRectangle(cornerRadius: 12))
                .focused($focused)

                // §Fix #15: Minimum 60pt target for visionOS eye-tracking
                Button {
                    focused = false
                    Task { await crew?.run(topic: topic) }
                } label: {
                    Image(systemName: "play.fill")
                        .font(.system(size: 20))
                        .foregroundStyle(.white)
                        .frame(width: 60, height: 60)
                        .background(canRun ? Color.blue : Color.gray,
                                    in: RoundedRectangle(cornerRadius: 14))
                }
                .hoverEffect()
                .disabled(!canRun)
                .accessibilityLabel("Run pipeline")
                .accessibilityHint(canRun ? "Starts the agent pipeline" : "Enter a topic first")
            }
        }
    }

    private var canRun: Bool {
        guard let crew else { return false }
        return !crew.isRunning && !topic.trimmingCharacters(in: .whitespaces).isEmpty
    }

    // MARK: - Agent grid

    private let agentNames = ["Extractor", "Reviewer", "Architect", "Reporter"]

    // §Fix #8: Adaptive grid — 2 columns on compact, up to 4 on wide layouts
    private func agentGrid(crew: AgentCrew) -> some View {
        LazyVGrid(
            columns: [GridItem(.adaptive(minimum: 160, maximum: 300))],
            spacing: 12
        ) {
            ForEach(agentNames, id: \.self) { name in
                AgentCard(
                    name:     name,
                    isActive: crew.isRunning && crew.currentAgent == name,
                    isDone:   crew.stepOutputs.contains(where: { $0.role == name })
                )
            }
        }
    }

    // MARK: - Step outputs

    private func stepList(crew: AgentCrew) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Agent Outputs").font(.headline)
            ForEach(crew.stepOutputs, id: \.role) { step in
                StepOutputCard(role: step.role, output: step.output)
            }
        }
    }

    // MARK: - Streaming

    private func streamingCard(crew: AgentCrew) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Reporter writing…", systemImage: "pencil.line")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.blue)
            ScrollView {
                Text(crew.streamingOutput)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
            }
            .frame(maxHeight: 200)
            .background(Color.secondarySystemBackground, in: RoundedRectangle(cornerRadius: 10))
        }
    }

    // MARK: - Report banner

    private var reportBanner: some View {
        Button { showReport = true } label: {
            HStack {
                Image(systemName: "doc.text.fill").foregroundStyle(.white)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Report Ready")
                        .font(.subheadline.weight(.semibold)).foregroundStyle(.white)
                    Text("Tap to view full analysis")
                        .font(.caption).foregroundStyle(.white.opacity(0.8))
                }
                Spacer()
                Image(systemName: "chevron.right").foregroundStyle(.white.opacity(0.7))
            }
            .padding(16)
            .frame(minHeight: 60)
            .background(Color.green, in: RoundedRectangle(cornerRadius: 14))
        }
        .hoverEffect()
        .accessibilityLabel("View report")
    }

    // MARK: - Error

    private func errorCard(_ message: String) -> some View {
        Label(message, systemImage: "exclamationmark.triangle.fill")
            .font(.subheadline).foregroundStyle(.red)
            .padding(12)
            .background(Color.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - AgentCard

private struct AgentCard: View {
    let name:     String
    let isActive: Bool
    let isDone:   Bool

    var body: some View {
        HStack(spacing: 10) {
            ZStack {
                Circle()
                    .fill(isActive ? Color.blue.opacity(0.15) :
                          isDone   ? Color.green.opacity(0.15) :
                                     Color.secondary.opacity(0.1))
                    .frame(width: 36, height: 36)
                if isActive {
                    ProgressView().scaleEffect(0.7)
                } else {
                    Image(systemName: isDone ? "checkmark" : "hourglass")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(isDone ? .green : .secondary)
                }
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(name).font(.subheadline.weight(.medium))
                Text(isActive ? "Running…" : isDone ? "Complete" : "Waiting")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(12)
        .frame(minHeight: 60)
        .background(
            isActive ? Color.blue.opacity(0.05) :
            isDone   ? Color.green.opacity(0.05) :
                       Color.secondarySystemBackground,
            in: RoundedRectangle(cornerRadius: 12)
        )
        .hoverEffect()
        .accessibilityLabel("\(name): \(isActive ? "Running" : isDone ? "Complete" : "Waiting")")
    }
}

// MARK: - StepOutputCard

private struct StepOutputCard: View {
    let role:   String
    let output: String
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(.spring(duration: 0.25)) { expanded.toggle() }
            } label: {
                HStack {
                    Label(role, systemImage: icon(for: role))
                        .font(.subheadline.weight(.medium))
                    Spacer()
                    Image(systemName: expanded ? "chevron.up" : "chevron.down")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.plain)
            .hoverEffect()

            Text(output)
                .font(.caption)
                .foregroundStyle(expanded ? .primary : .secondary)
                .lineLimit(expanded ? nil : 2)
                .animation(.spring(duration: 0.25), value: expanded)
        }
        .padding(12)
        .background(Color.secondarySystemBackground, in: RoundedRectangle(cornerRadius: 12))
    }

    private func icon(for role: String) -> String {
        switch role {
        case "Extractor":  return "magnifyingglass"
        case "Reviewer":   return "checkmark.shield"
        case "Architect":  return "square.3.layers.3d"
        case "Reporter":   return "doc.text"
        default:           return "person"
        }
    }
}

// MARK: - ReportSheet

struct ReportSheet: View {
    let report:      AnalysisReport
    let exportedURL: URL?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(report.topic).font(.title2.weight(.bold))
                        Text(report.createdAt.formatted(date: .long, time: .shortened))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Divider()
                    Text(report.fullText).font(.body)
                    Divider()
                    Text("Pipeline Steps").font(.headline)
                    ForEach(report.stepOutputs, id: \.role) { step in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(step.role)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                                .textCase(.uppercase)
                            Text(step.output)
                                .font(.caption)
                                .lineLimit(4)
                        }
                        .padding(10)
                        .background(Color.secondarySystemBackground,
                                    in: RoundedRectangle(cornerRadius: 8))
                    }
                    if let url = exportedURL {
                        Divider()
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Export").font(.headline)
                            ShareLink(item: url) {
                                Label(url.lastPathComponent,
                                      systemImage: "square.and.arrow.up")
                                    .font(.subheadline)
                            }
                            Text("JSONL.GZ — compatible with LangChain / HuggingFace")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Analysis Report")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

// MARK: - DocumentsTab

struct DocumentsTab: View {
    @EnvironmentObject var appState: AppState
    @State private var showPicker        = false
    @State private var isIndexing        = false
    @State private var indexingProgress  = ""
    @State private var documents:        [IndexedDocument] = []

    var body: some View {
        NavigationStack {
            Group {
                if documents.isEmpty && !isIndexing {
                    emptyState
                } else {
                    documentList
                }
            }
            .navigationTitle("Document Index")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button { showPicker = true } label: { Image(systemName: "plus") }
                        .disabled(isIndexing || !appState.isReady)
                }
            }
            .fileImporter(
                isPresented: $showPicker,
                allowedContentTypes: [
                    .pdf, .plainText,
                    UTType(filenameExtension: "docx") ?? .data,
                    UTType(filenameExtension: "md")   ?? .text
                ],
                allowsMultipleSelection: true
            ) { result in
                if case .success(let urls) = result {
                    Task { await index(urls: urls) }
                }
            }
            .task { await reload() }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "doc.badge.plus")
                .font(.system(size: 52)).foregroundStyle(.secondary)
            Text("No documents indexed").font(.headline)
            Text("Add documents so agents can search them during pipeline runs.")
                .font(.subheadline).foregroundStyle(.secondary)
                .multilineTextAlignment(.center).padding(.horizontal, 32)
            Button("Add Documents") { showPicker = true }
                .buttonStyle(.borderedProminent).disabled(!appState.isReady)
        }
    }

    private var documentList: some View {
        List {
            if isIndexing {
                Section {
                    HStack(spacing: 12) {
                        ProgressView()
                        Text(indexingProgress).font(.subheadline).foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                }
            }
            Section {
                ForEach(documents) { doc in
                    HStack(spacing: 12) {
                        Image(systemName: "doc.text.fill").foregroundStyle(.blue)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(doc.title).font(.subheadline.weight(.medium))
                            Text("\(doc.chunkCount) chunks · \(doc.indexedAt, style: .relative)")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                }
                .onDelete { offsets in
                    Task {
                        for i in offsets {
                            try? await appState.library.removeDocument(id: documents[i].id)
                        }
                        await reload()
                    }
                }
            } header: {
                Text("\(documents.count) document\(documents.count == 1 ? "" : "s")")
            }
        }
    }

    private func index(urls: [URL]) async {
        isIndexing = true
        for url in urls {
            do {
                _ = try await appState.library.add(url: url) { p in
                    indexingProgress = p
                }
            } catch {
                indexingProgress = "Error: \(error.localizedDescription)"
            }
        }
        await appState.library.refreshCorpus()
        await reload()
        isIndexing       = false
        indexingProgress = ""
    }

    private func reload() async {
        documents = (try? await appState.library.allDocuments()) ?? []
    }
}

// MARK: - HistoryTab

struct HistoryTab: View {
    @EnvironmentObject var appState: AppState
    @State private var runs:        [Conversation] = []
    @State private var selectedRun: Conversation?
    @State private var turns:       [Turn] = []

    var body: some View {
        NavigationStack {
            List {
                if runs.isEmpty {
                    ContentUnavailableView(
                        "No past runs",
                        systemImage: "clock.arrow.circlepath",
                        description: Text("Run a pipeline to see results here.")
                    )
                } else {
                    ForEach(runs) { run in
                        Button {
                            Task { await select(run) }
                        } label: {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(run.title ?? run.id.uuidString)
                                    .font(.subheadline.weight(.medium))
                                    .foregroundStyle(.primary)
                                Text(run.createdAt, style: .date)
                                    .font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                    }
                    .onDelete { offsets in
                        Task {
                            for i in offsets {
                                try? await appState.store.deleteConversation(id: runs[i].id)
                            }
                            await reload()
                        }
                    }
                }
            }
            .navigationTitle("Run History")
            .sheet(item: $selectedRun) { run in
                RunDetailSheet(run: run, turns: turns)
            }
            .task { await reload() }
        }
    }

    private func select(_ run: Conversation) async {
        turns       = (try? await appState.store.turns(for: run.id)) ?? []
        selectedRun = run
    }

    private func reload() async {
        runs = ((try? await appState.store.allConversations()) ?? [])
            .filter { $0.title.hasPrefix("AgentCrew:") == true }
            .sorted { $0.createdAt > $1.createdAt }
    }
}

// MARK: - RunDetailSheet

private struct RunDetailSheet: View {
    let run:   Conversation
    let turns: [Turn]
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                ForEach(turns) { turn in
                    VStack(alignment: .leading, spacing: 6) {
                        Text(parseRole(turn))
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .textCase(.uppercase)
                        Text(parseContent(turn))
                            .font(.subheadline)
                            .lineLimit(6)
                    }
                    .padding(.vertical, 4)
                }
            }
            .navigationTitle(run.title ?? "Run Detail")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func parseRole(_ turn: Turn) -> String {
        guard turn.content.hasPrefix("["),
              let end = turn.content.firstIndex(of: "]")
        else { return turn.role.rawValue }
        return String(turn.content[turn.content.index(after: turn.content.startIndex)..<end])
    }

    private func parseContent(_ turn: Turn) -> String {
        guard turn.content.hasPrefix("["),
              let end = turn.content.firstIndex(of: "]")
        else { return turn.content }
        return String(turn.content[turn.content.index(after: end)...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Color helpers

private extension Color {
    static var secondarySystemBackground: Color {
        #if os(iOS)
        Color(.secondarySystemBackground)
        #else
        Color(nsColor: .controlBackgroundColor)
        #endif
    }
}
